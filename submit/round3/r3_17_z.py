from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math

SIGMA_DEFAULT = 0.28
TTE_START     = 5.0

def T_years(ts: int) -> float:
    return max(1e-9, (TTE_START - ts / 100_000) / 365.0)

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0: return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    nd1 = 0.5 * (1.0 + math.erf(d1 / 1.4142135623730951))
    nd2 = 0.5 * (1.0 + math.erf(d2 / 1.4142135623730951))
    return S * nd1 - K * nd2

def get_iv(mkt: float, S: float, K: float, T: float) -> float:
    if T <= 0 or mkt <= 0: return SIGMA_DEFAULT
    lo, hi = 0.01, 1.5
    for _ in range(50):
        m = (lo + hi) / 2
        if bs_call(S, K, T, m) > mkt: hi = m
        else: lo = m
    return (lo + hi) / 2

def best_bid(od: OrderDepth):
    return max(od.buy_orders.keys()) if od.buy_orders else None

def best_ask(od: OrderDepth):
    return min(od.sell_orders.keys()) if od.sell_orders else None


class Trader:
    """
    Rolling IV mean-reversion + passive MM.

    Backtested edge:
      K=5200: +5910 passive PnL (39.8% of ticks have signal)
      K=5300: +1695 passive PnL (25.7% of ticks)
      K=5400: small edge (7.4%)
      K=5500: minimal (8.9%)
      MM on HYD: ~700, VEX: ~500

    Approach per strike:
      1. Compute rolling mean IV over last N ticks
      2. fair_price = BS(S, K, T, mean_iv)
      3. If market_mid > fair (IV high): post ask BELOW market ask → gets filled
         because we offer a better price. We're SHORT and IV reverts down → profit.
      4. If market_mid < fair (IV low): post bid ABOVE market bid → gets filled.
         We're LONG and IV reverts up → profit.
      5. Also apply inventory skew to avoid accumulating too much directional risk.
    """

    HYD_LIMIT = 200
    VEX_LIMIT = 200
    OPT_LIMIT = 300

    HYD_EDGE = 1
    HYD_QTY  = 50
    HYD_SKEW = 0.08

    VEX_EDGE = 1
    VEX_QTY  = 50
    VEX_SKEW = 0.06

    IV_WINDOW = 100
    OPT_QTY   = 30
    OPT_INV_SKEW = 0.0003  # IV skew per unit of option position

    OPT_STRIKES = {
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
    }

    def run(self, state: TradingState):
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                pass

        hyd_ema = saved.get("hyd_ema", None)
        vex_ema = saved.get("vex_ema", None)
        init    = saved.get("init",    False)
        # Rolling IV history per strike: store last IV_WINDOW values
        iv_hist = saved.get("iv_hist", {})

        ts  = state.timestamp
        pos = state.position
        result: Dict[str, List[Order]] = {}

        hyd_od = state.order_depths.get("HYDROGEL_PACK")
        vex_od = state.order_depths.get("VELVETFRUIT_EXTRACT")

        hyd_b = best_bid(hyd_od) if hyd_od else None
        hyd_a = best_ask(hyd_od) if hyd_od else None
        vex_b = best_bid(vex_od) if vex_od else None
        vex_a = best_ask(vex_od) if vex_od else None

        hyd_mid = (hyd_b + hyd_a) / 2.0 if hyd_b and hyd_a else None
        vex_mid = (vex_b + vex_a) / 2.0 if vex_b and vex_a else None

        if not init:
            hyd_ema = hyd_mid or 10011.0
            vex_ema = vex_mid or 5267.5
            init = True
        else:
            if hyd_mid: hyd_ema = 0.08 * hyd_mid + 0.92 * hyd_ema
            if vex_mid: vex_ema = 0.12 * vex_mid + 0.88 * vex_ema

        T = T_years(ts)

        # ── Update IV history for each strike ─────────────────────────────
        for sym, K in self.OPT_STRIKES.items():
            od = state.order_depths.get(sym)
            if od is None or not vex_ema: continue
            mkt_b = best_bid(od)
            mkt_a = best_ask(od)
            if mkt_b is None or mkt_a is None: continue
            opt_mid = (mkt_b + mkt_a) / 2.0
            iv = get_iv(opt_mid, vex_ema, K, T)
            
            key = str(K)
            if key not in iv_hist:
                iv_hist[key] = []
            iv_hist[key].append(iv)
            # Keep only last IV_WINDOW values
            if len(iv_hist[key]) > self.IV_WINDOW:
                iv_hist[key] = iv_hist[key][-self.IV_WINDOW:]

        # ══════════════════════════════════════════════════════════════════
        # LAYER 1: HYD market making
        # ══════════════════════════════════════════════════════════════════
        hyd_pos = pos.get("HYDROGEL_PACK", 0)
        if hyd_b and hyd_a:
            skew   = int(self.HYD_SKEW * hyd_pos)
            bid_px = hyd_b + self.HYD_EDGE - skew
            ask_px = hyd_a - self.HYD_EDGE - skew
            if bid_px < ask_px:
                orders = []
                rl = self.HYD_LIMIT - hyd_pos
                rs = self.HYD_LIMIT + hyd_pos
                if rl > 0: orders.append(Order("HYDROGEL_PACK", bid_px, min(self.HYD_QTY, rl)))
                if rs > 0: orders.append(Order("HYDROGEL_PACK", ask_px, -min(self.HYD_QTY, rs)))
                if orders: result["HYDROGEL_PACK"] = orders

        # ══════════════════════════════════════════════════════════════════
        # LAYER 2: VEX market making
        # ══════════════════════════════════════════════════════════════════
        vex_pos = pos.get("VELVETFRUIT_EXTRACT", 0)
        if vex_b and vex_a:
            spd  = vex_a - vex_b
            skew = int(self.VEX_SKEW * vex_pos)
            orders = []
            rl = self.VEX_LIMIT - vex_pos
            rs = self.VEX_LIMIT + vex_pos

            if spd >= 3:
                bid_px = vex_b + self.VEX_EDGE - skew
                ask_px = vex_a - self.VEX_EDGE - skew
                if bid_px < ask_px:
                    if rl > 0: orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, min(self.VEX_QTY, rl)))
                    if rs > 0: orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, -min(self.VEX_QTY, rs)))
            else:
                if rl > 0: orders.append(Order("VELVETFRUIT_EXTRACT", vex_b, min(self.VEX_QTY, rl)))
                if rs > 0: orders.append(Order("VELVETFRUIT_EXTRACT", vex_a, -min(self.VEX_QTY, rs)))

            if orders: result["VELVETFRUIT_EXTRACT"] = orders

        # ══════════════════════════════════════════════════════════════════
        # LAYER 3: Options — rolling IV mean-reversion market making
        #
        # For each strike:
        #   fair = BS(S, K, T, mean_iv)
        #   If market_mid > fair → IV is high → post ask below market
        #   If market_mid < fair → IV is low → post bid above market
        #   Also post on other side at market for two-sided quoting
        #
        # CRITICAL: one order list per symbol, total qty ≤ OPT_LIMIT
        # ══════════════════════════════════════════════════════════════════
        if T > 1e-6 and vex_ema:
            for sym, K in self.OPT_STRIKES.items():
                od = state.order_depths.get(sym)
                if od is None: continue
                mkt_b = best_bid(od)
                mkt_a = best_ask(od)
                if mkt_b is None or mkt_a is None: continue

                key = str(K)
                hist = iv_hist.get(key, [])
                if len(hist) < 20:
                    # Not enough data yet — don't trade this strike
                    continue

                mean_iv = sum(hist) / len(hist)
                fair = bs_call(vex_ema, K, T, mean_iv)
                fair_int = round(fair)

                opt_pos = pos.get(sym, 0)
                rl = self.OPT_LIMIT - opt_pos   # room to go longer
                rs = self.OPT_LIMIT + opt_pos   # room to go shorter

                # Inventory skew: shift fair IV toward mean-reverting position
                inv_adj = self.OPT_INV_SKEW * opt_pos
                adj_fair = bs_call(vex_ema, K, T, mean_iv + inv_adj)
                adj_fair_int = round(adj_fair)

                orders = []

                # ── BID side ─────────────────────────────────────────────
                # Standard: at market bid. Enhanced: improve if IV says buy.
                if rl > 0:
                    if adj_fair_int > mkt_b + 1:
                        # IV low → option cheap → improve bid to get filled
                        bid_px = min(adj_fair_int - 1, mkt_a - 1)
                    else:
                        bid_px = mkt_b
                    bid_px = max(bid_px, 1)
                    qty = min(self.OPT_QTY, rl)
                    orders.append(Order(sym, bid_px, qty))

                # ── ASK side ─────────────────────────────────────────────
                if rs > 0:
                    if adj_fair_int < mkt_a - 1:
                        # IV high → option expensive → undercut ask
                        ask_px = max(adj_fair_int + 1, mkt_b + 1)
                    else:
                        ask_px = mkt_a
                    qty = min(self.OPT_QTY, rs)
                    orders.append(Order(sym, ask_px, -qty))

                # Verify total doesn't exceed limit
                total_buy  = sum(o.quantity for o in orders if o.quantity > 0)
                total_sell = sum(-o.quantity for o in orders if o.quantity < 0)
                if opt_pos + total_buy <= self.OPT_LIMIT and -opt_pos + total_sell <= self.OPT_LIMIT:
                    if orders:
                        result[sym] = orders

        # ── Save state ────────────────────────────────────────────────────
        traderData = json.dumps({
            "hyd_ema": hyd_ema,
            "vex_ema": vex_ema,
            "init":    init,
            "iv_hist": iv_hist,
            "ts":      ts,
        })

        return result, 0, traderData