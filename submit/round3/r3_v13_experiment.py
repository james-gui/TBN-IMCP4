from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    return S * nd1 - K * nd2


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Delta of a European call (dC/dS)."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))


def implied_vol(market_price: float, S: float, K: float, T: float,
                lo: float = 0.001, hi: float = 0.20) -> float:
    """
    Binary search for implied vol.
    Bounds lo/hi set to [0.001, 0.20] because actual market sigma ~ 0.03.
    (Old code used [0.01, 3.0] which was wildly off.)
    """
    if T <= 0 or market_price <= 0:
        return 0.032
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if bs_call(S, K, T, mid) > market_price:
            hi = mid
        else:
            lo = mid
    iv = (lo + hi) / 2.0
    return iv if lo < iv < hi else 0.032


def get_mid(order_depth: OrderDepth):
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0


class Trader:

    # ── Position limits ────────────────────────────────────────────────────
    VEX_LIMIT = 50
    HYD_LIMIT = 20
    OPT_LIMIT = 300   # per option strike

    # ── VEX market making ──────────────────────────────────────────────────
    VEX_ALPHA      = 0.15   # EMA speed
    VEX_HALF_SPRD  = 4      # ticks half-spread
    VEX_SKEW_SCALE = 0.12   # ticks shift per unit position
    VEX_QTY        = 20

    # ── HYD market making ──────────────────────────────────────────────────
    HYD_ALPHA      = 0.10
    HYD_HALF_SPRD  = 8
    HYD_SKEW_SCALE = 0.30
    HYD_QTY        = 10

    # ── Options ───────────────────────────────────────────────────────────
    # CRITICAL FIX: sigma ~ 0.032 (not 0.28!) — back-calculated from
    # market prices: K=5300 mid=53, K=5400 mid=17, K=5500 mid=6.5
    # at S=5267, T=1.0 → all imply sigma ≈ 0.030–0.033
    SIGMA_INIT     = 0.032
    SIGMA_ALPHA    = 0.05   # slow EMA on sigma — market vol is stable
    OPT_HALF_SPRD  = 1      # 1 tick half-spread around BS fair value
    OPT_QTY        = 20
    OPT_SKEW_SCALE = 0.003  # very small: options are discrete-priced

    STRIKES = {
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
    }
    # Skip deep ITM strikes (VEV_5000/5100): huge delta, no edge, pure gamma risk
    # Their fair value is almost entirely intrinsic → spread too tight to profit

    EXPIRY_TS = 100000

    # ── Delta hedging ─────────────────────────────────────────────────────
    # We track net option delta and hedge with VEX
    HEDGE_THRESHOLD = 5.0   # hedge if |net_delta| > this

    def run(self, state: TradingState):
        # ── Load persisted state ─────────────────────────────────────────
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                pass

        vex_ema = saved.get("vex_ema", None)
        hyd_ema = saved.get("hyd_ema", None)
        sigma   = saved.get("sigma",   self.SIGMA_INIT)
        init    = saved.get("init",    False)

        ts = state.timestamp
        result: Dict[str, List[Order]] = {}

        # ── Mid prices ───────────────────────────────────────────────────
        vex_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        hyd_depth = state.order_depths.get("HYDROGEL_PACK")

        vex_mid = get_mid(vex_depth) if vex_depth else None
        hyd_mid = get_mid(hyd_depth) if hyd_depth else None

        if vex_mid is None or hyd_mid is None:
            traderData = json.dumps({
                "vex_ema": vex_ema, "hyd_ema": hyd_ema,
                "sigma": sigma, "init": init, "ts": ts
            })
            return result, 0, traderData

        # ── EMA update ──────────────────────────────────────────────────
        if not init:
            vex_ema = vex_mid
            hyd_ema = hyd_mid
            init = True
        else:
            vex_ema = self.VEX_ALPHA * vex_mid + (1 - self.VEX_ALPHA) * vex_ema
            hyd_ema = self.HYD_ALPHA * hyd_mid + (1 - self.HYD_ALPHA) * hyd_ema

        # ── Implied vol update ──────────────────────────────────────────
        # Use OTM strikes only (K=5400, 5500) for cleaner IV signal
        T_remain = max(0.0, (self.EXPIRY_TS - ts)) / self.EXPIRY_TS
        if T_remain > 0.001:
            vols = []
            for sym, K in [("VEV_5400", 5400), ("VEV_5500", 5500)]:
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                mid = get_mid(od)
                if mid and mid > 0.3:
                    iv = implied_vol(mid, vex_ema, K, T_remain,
                                     lo=0.001, hi=0.15)
                    if 0.001 < iv < 0.15:
                        vols.append(iv)
            if vols:
                new_sigma = sum(vols) / len(vols)
                sigma = self.SIGMA_ALPHA * new_sigma + (1 - self.SIGMA_ALPHA) * sigma

        # ── Positions ────────────────────────────────────────────────────
        positions = state.position
        vex_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
        hyd_pos = positions.get("HYDROGEL_PACK", 0)

        # ── Compute net option delta for hedge ───────────────────────────
        net_opt_delta = 0.0
        if T_remain > 0.001:
            for sym, K in self.STRIKES.items():
                opt_pos = positions.get(sym, 0)
                if opt_pos != 0:
                    delta = bs_delta(vex_ema, K, T_remain, sigma)
                    net_opt_delta += opt_pos * delta

        # ── 1. VEX orders (MM + delta hedge) ────────────────────────────
        vex_orders: List[Order] = []
        vex_skew = self.VEX_SKEW_SCALE * vex_pos

        for lvl in range(2):
            half = self.VEX_HALF_SPRD + lvl * 2
            qty  = self.VEX_QTY - lvl * 5

            bid_px = int(vex_ema - half - vex_skew)
            ask_px = int(vex_ema + half - vex_skew)

            room_long  = self.VEX_LIMIT - vex_pos
            room_short = self.VEX_LIMIT + vex_pos

            if room_long > 0 and qty > 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT", bid_px,
                                        min(qty, room_long)))
            if room_short > 0 and qty > 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT", ask_px,
                                        -min(qty, room_short)))

        # Delta hedge: if net option delta is large, add an offsetting VEX order
        hedge_qty = int(-net_opt_delta - vex_pos)
        if abs(hedge_qty) >= self.HEDGE_THRESHOLD:
            hedge_qty = max(-self.VEX_LIMIT - vex_pos,
                            min(self.VEX_LIMIT - vex_pos, hedge_qty))
            if hedge_qty > 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT",
                                        int(vex_ema - 1), hedge_qty))
            elif hedge_qty < 0:
                vex_orders.append(Order("VELVETFRUIT_EXTRACT",
                                        int(vex_ema + 1), hedge_qty))

        if vex_orders:
            result["VELVETFRUIT_EXTRACT"] = vex_orders

        # ── 2. HYD orders ────────────────────────────────────────────────
        hyd_orders: List[Order] = []
        hyd_skew = self.HYD_SKEW_SCALE * hyd_pos

        for lvl in range(2):
            half = self.HYD_HALF_SPRD + lvl * 4
            qty  = self.HYD_QTY - lvl * 3

            bid_px = int(hyd_ema - half - hyd_skew)
            ask_px = int(hyd_ema + half - hyd_skew)

            room_long  = self.HYD_LIMIT - hyd_pos
            room_short = self.HYD_LIMIT + hyd_pos

            if room_long > 0 and qty > 0:
                hyd_orders.append(Order("HYDROGEL_PACK", bid_px,
                                        min(qty, room_long)))
            if room_short > 0 and qty > 0:
                hyd_orders.append(Order("HYDROGEL_PACK", ask_px,
                                        -min(qty, room_short)))

        if hyd_orders:
            result["HYDROGEL_PACK"] = hyd_orders

        # ── 3. Options market making ─────────────────────────────────────
        if T_remain > 0.001:
            for sym, K in self.STRIKES.items():
                od = state.order_depths.get(sym)
                if od is None:
                    continue

                fair = bs_call(vex_ema, K, T_remain, sigma)

                # Skip if essentially worthless (OTM and nearly expired)
                if fair < 0.5:
                    continue

                opt_pos  = positions.get(sym, 0)
                opt_skew = self.OPT_SKEW_SCALE * opt_pos

                bid_px = max(0, int(fair - self.OPT_HALF_SPRD - opt_skew))
                ask_px = int(fair + self.OPT_HALF_SPRD - opt_skew)

                if bid_px <= 0 or ask_px <= bid_px:
                    continue

                opt_orders: List[Order] = []
                room_long  = self.OPT_LIMIT - opt_pos
                room_short = self.OPT_LIMIT + opt_pos

                if room_long > 0:
                    opt_orders.append(Order(sym, bid_px,
                                            min(self.OPT_QTY, room_long)))
                if room_short > 0:
                    opt_orders.append(Order(sym, ask_px,
                                            -min(self.OPT_QTY, room_short)))

                if opt_orders:
                    result[sym] = opt_orders

        # ── Save state ───────────────────────────────────────────────────
        traderData = json.dumps({
            "vex_ema": vex_ema,
            "hyd_ema": hyd_ema,
            "sigma":   sigma,
            "init":    init,
            "ts":      ts,
        })

        return result, 0, traderData