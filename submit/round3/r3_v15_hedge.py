from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math


# ── Black-Scholes with CORRECT parameters ────────────────────────────────────
# sigma = 0.28 (annualized, calibrated from market: K=5300 mid=53, T=5/365)
# T = TTE_days / 365  (T in years)
# TTE at start of round 3 = 5 days. At ts=t: TTE = 5 - t/100000 days
# Liquidation at end of round: TTE_remaining = 4 days → T_liq = 4/365

SIGMA = 0.28
TTE_START = 5.0          # days at ts=0
ROUND_DAYS = 1.0         # 1 day passes during the round (100000 timestamps)
TTE_LIQ = TTE_START - ROUND_DAYS   # = 4.0 days at liquidation

def tte_days(ts: int) -> float:
    """TTE in days at timestamp ts."""
    return TTE_START - ts / 100_000

def T_years(ts: int) -> float:
    """T in years at timestamp ts."""
    return max(0.0, tte_days(ts)) / 365.0

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    return S * nd1 - K * nd2

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Delta of European call."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))

def best_bid(od: OrderDepth):
    return max(od.buy_orders.keys()) if od.buy_orders else None

def best_ask(od: OrderDepth):
    return min(od.sell_orders.keys()) if od.sell_orders else None


class Trader:
    """
    ── CONFIRMED MARKET MODEL ────────────────────────────────────────────────
    Options are European calls on VEX (VELVETFRUIT_EXTRACT), liquidated at
    end of round against hidden fair value = BS(S_final, K, 4/365, 0.28).

    sigma_annual = 0.28 (calibrated: BS(5267.5, 5300, 5/365, 0.28) = 53.00 ✓)
    T at ts=t = (5 - t/100000) / 365 years (TTE decays from 5d to 4d)
    
    Market mid ≈ BS(S, K, T_current, 0.28) within 0–6 ticks (bid-ask half-spread)
    → Market fairly prices options, but at T_current (not T_liq)
    
    EDGE FROM SHORTING (sell at market_ask, liq at BS_4d):
      K=5200: +7.57/contract × 300 = +2271
      K=5300: +8.57/contract × 300 = +2572
      K=5400: +1.43/contract × 300 = +429
      K=5500: +2.34/contract × 300 = +701
    
    DELTA RISK: short all 4 = net delta -425, VEX limit 200
    Strategy: short K=5300+5400+5500, go LONG VEX +200 as delta hedge
      → residual delta ≈ -27 (manageable)
      → theta income: +3702 net

    ── HYD/VEX MM ────────────────────────────────────────────────────────────
    HYD: 16-tick spread always. 20 taker events per run (sizes 2-6).
         Already capturing 90%+ of available flow. Near-optimal.
    VEX: 5-tick spread (mostly). 52 taker events (sizes 3-14).
         Tight-spread fix: join queue at mkt_bid/mkt_ask when spread ≤ 2.
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

    # Options to SHORT: collect theta. K=5300 has highest edge per contract.
    # Include 5200 despite ITM delta risk — it has the most total theta.
    # Delta hedge with long VEX position.
    OPT_SHORTS = {
        "VEV_5200": 5200,   # edge=7.57/contract × 300 = +2271
        "VEV_5300": 5300,   # edge=8.57/contract × 300 = +2572
        "VEV_5400": 5400,   # edge=1.43/contract × 300 = +429
        "VEV_5500": 5500,   # edge=2.34/contract × 300 = +701
    }

    EXPIRY_TS = 100_000

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

        # Current option fair value uses live VEX and current T
        T_now = T_years(ts)

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 1: HYD market making
        # ══════════════════════════════════════════════════════════════════════
        hyd_pos = pos.get("HYDROGEL_PACK", 0)
        if hyd_b and hyd_a:
            skew   = int(self.HYD_SKEW * hyd_pos)
            bid_px = hyd_b + self.HYD_EDGE - skew
            ask_px = hyd_a - self.HYD_EDGE - skew
            if bid_px < ask_px:
                orders = []
                room_l = self.HYD_LIMIT - hyd_pos
                room_s = self.HYD_LIMIT + hyd_pos
                if room_l > 0:
                    orders.append(Order("HYDROGEL_PACK", bid_px, min(self.HYD_QTY, room_l)))
                if room_s > 0:
                    orders.append(Order("HYDROGEL_PACK", ask_px, -min(self.HYD_QTY, room_s)))
                if orders:
                    result["HYDROGEL_PACK"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 2: VEX market making + delta hedge for option book
        # 
        # VEX does double duty:
        # a) Standard MM: quote inside spread, capture flow
        # b) Delta hedge: target a LONG VEX position to offset short option delta
        #
        # Net option delta (short all 4 × 300 at start):
        #   ≈ -425. VEX limit=200. Go long 200 VEX → residual ≈ -225.
        # With only K=5300+5400+5500 shorted:
        #   delta ≈ -228. Go long 200 VEX → residual ≈ -28 (near neutral).
        #
        # Implementation: bias our VEX quotes toward BUYING (skew negative)
        # when we need more long VEX for hedging.
        # ══════════════════════════════════════════════════════════════════════
        vex_pos = pos.get("VELVETFRUIT_EXTRACT", 0)

        # Compute current net option delta to determine hedge need
        net_opt_delta = 0.0
        if T_now > 0 and vex_ema:
            for sym, K in self.OPT_SHORTS.items():
                opt_pos = pos.get(sym, 0)
                if opt_pos != 0:
                    d = bs_delta(vex_ema, K, T_now, SIGMA)
                    net_opt_delta += opt_pos * d   # negative when short

        # Target VEX position: offset as much option delta as possible
        target_vex = int(min(self.VEX_LIMIT, max(-self.VEX_LIMIT, -net_opt_delta)))
        hedge_gap  = target_vex - vex_pos  # positive → need to buy VEX

        if vex_b and vex_a:
            spd    = vex_a - vex_b
            # Skew: if hedge_gap > 0 (need long), shift bid up / ask up to attract sellers
            # Use VEX_SKEW for MM mean-reversion, plus hedge_skew for delta mgmt
            mm_skew    = int(self.VEX_SKEW * vex_pos)
            orders     = []
            room_l = self.VEX_LIMIT - vex_pos
            room_s = self.VEX_LIMIT + vex_pos

            if spd >= 3:
                bid_px = vex_b + self.VEX_EDGE - mm_skew
                ask_px = vex_a - self.VEX_EDGE - mm_skew
                if bid_px < ask_px:
                    if room_l > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, min(self.VEX_QTY, room_l)))
                    if room_s > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, -min(self.VEX_QTY, room_s)))
            else:
                # Tight spread: join queue at market prices
                if room_l > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", vex_b, min(self.VEX_QTY, room_l)))
                if room_s > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", vex_a, -min(self.VEX_QTY, room_s)))

            # Additional aggressive buy if significantly under-hedged
            if hedge_gap > 20 and room_l > 0 and vex_a:
                # Hit the ask to hedge immediately
                hedge_qty = min(hedge_gap, room_l, self.VEX_QTY)
                orders.append(Order("VELVETFRUIT_EXTRACT", vex_a, hedge_qty))

            if orders:
                result["VELVETFRUIT_EXTRACT"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 3: Options — SHORT all strikes to collect theta
        #
        # Fair value = BS(S_now, K, T_now, sigma=0.28)
        # Liquidation = BS(S_end, K, 4/365, 0.28) ← still has time value!
        # Edge = market_ask - liq_fair = theta decay over 1 day
        #
        # Execution: post asks at mkt_ask - 1 (undercut by 1 tick for priority)
        # When spread = 1: just post at mkt_ask (can't undercut)
        #
        # Also POST BIDS at mkt_bid so we are the passive buyer when taker-sellers
        # arrive — capturing the spread while reducing our short exposure
        # (or adding to it via the ask side)
        # ══════════════════════════════════════════════════════════════════════
        if T_now > 0.0001 and vex_ema:
            for sym, K in self.OPT_SHORTS.items():
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                mkt_b = best_bid(od)
                mkt_a = best_ask(od)
                if mkt_b is None or mkt_a is None:
                    continue

                opt_pos  = pos.get(sym, 0)
                room_s   = self.OPT_LIMIT + opt_pos   # how much more we can short
                room_l   = self.OPT_LIMIT - opt_pos   # how much long we can add

                orders = []

                # ── SELL side (primary): undercut market ask by 1 for priority ──
                if room_s > 0:
                    ask_px = mkt_a - 1 if (mkt_a - mkt_b) > 1 else mkt_a
                    if ask_px > mkt_b:
                        orders.append(Order(sym, ask_px, -min(self.OPT_LIMIT, room_s)))

                # ── BUY side: post at mkt_bid only for K=5400 (tiny positive edge) ──
                # For other strikes, buying at bid is a losing trade after theta.
                # K=5400 bid=16, liq_fair=16.57 → +0.57 edge if filled.
                if K == 5400 and room_l > 0 and mkt_b:
                    orders.append(Order(sym, mkt_b, min(self.OPT_LIMIT, room_l)))

                if orders:
                    result[sym] = orders

        traderData = json.dumps({
            "hyd_ema": hyd_ema,
            "vex_ema": vex_ema,
            "init":    init,
            "ts":      ts,
        })

        return result, 0, traderData