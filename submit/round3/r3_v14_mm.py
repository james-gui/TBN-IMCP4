from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    return S * nd1 - K * nd2


def implied_vol(mkt: float, S: float, K: float, T: float) -> float:
    if T <= 0 or mkt <= 0:
        return 0.031
    lo, hi = 0.001, 0.12
    for _ in range(50):
        m = (lo + hi) / 2.0
        if bs_call(S, K, T, m) > mkt:
            hi = m
        else:
            lo = m
    iv = (lo + hi) / 2.0
    return iv if 0.001 < iv < 0.12 else 0.031


def best_bid(od: OrderDepth):
    return max(od.buy_orders.keys()) if od.buy_orders else None

def best_ask(od: OrderDepth):
    return min(od.sell_orders.keys()) if od.sell_orders else None

def mid(od: OrderDepth):
    b, a = best_bid(od), best_ask(od)
    return (b + a) / 2.0 if b is not None and a is not None else None


class Trader:

    # ── Position limits ───────────────────────────────────────────────────────
    HYD_LIMIT = 200
    VEX_LIMIT = 200
    OPT_LIMIT = 300

    # ── HYD: spread=16 ticks, quote 1 inside each side ───────────────────────
    HYD_ALPHA = 0.08
    HYD_EDGE  = 1      # ticks inside best bid/ask
    HYD_QTY   = 50     # large lots — want 200 filled in ~4 trades
    HYD_SKEW  = 0.08   # per unit of position

    # ── VEX: spread=5 ticks, quote 1 inside each side ────────────────────────
    VEX_ALPHA = 0.12
    VEX_EDGE  = 1
    VEX_QTY   = 50
    VEX_SKEW  = 0.06

    # ── Options: sigma≈0.031, post AT market ask to guarantee fills ───────────
    # Key insight from log analysis: market bots trade at specific timestamps.
    # We need to be sitting at the market ask price to get filled when they come.
    # Post full OPT_LIMIT qty immediately to maximise theta collected.
    SIGMA_INIT  = 0.031
    SIGMA_ALPHA = 0.03

    # All option strikes and their market context at ts=0:
    # VEV_5200: bid=102 ask=106 spd=4 | BS_fair=104 → sell at 106 (mkt ask)
    # VEV_5300: bid=52  ask=54  spd=2 | BS_fair=50  → sell at 54 (mkt ask)
    # VEV_5400: bid=16  ask=18  spd=2 | BS_fair=20  → sell at 18 (mkt ask)
    # VEV_5500: bid=6   ask=7   spd=1 | BS_fair=6   → sell at 7  (mkt ask)
    OPT_STRIKES = {
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
    }

    EXPIRY_TS = 100000

    def run(self, state: TradingState):
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except Exception:
                pass

        hyd_ema = saved.get("hyd_ema", None)
        vex_ema = saved.get("vex_ema", None)
        sigma   = saved.get("sigma",   self.SIGMA_INIT)
        init    = saved.get("init",    False)

        ts  = state.timestamp
        pos = state.position
        result: Dict[str, List[Order]] = {}

        hyd_od = state.order_depths.get("HYDROGEL_PACK")
        vex_od = state.order_depths.get("VELVETFRUIT_EXTRACT")

        hyd_mid = mid(hyd_od) if hyd_od else None
        vex_mid = mid(vex_od) if vex_od else None

        # ── EMA init ──────────────────────────────────────────────────────────
        if not init:
            hyd_ema = hyd_mid or 10011.0
            vex_ema = vex_mid or 5267.5
            init = True
        else:
            if hyd_mid:
                hyd_ema = self.HYD_ALPHA * hyd_mid + (1 - self.HYD_ALPHA) * hyd_ema
            if vex_mid:
                vex_ema = self.VEX_ALPHA * vex_mid + (1 - self.VEX_ALPHA) * vex_ema

        # ── Sigma update ──────────────────────────────────────────────────────
        T_rem = max(0.0, (self.EXPIRY_TS - ts)) / self.EXPIRY_TS
        if T_rem > 0.001 and vex_ema:
            vols = []
            for sym, K in [("VEV_5400", 5400), ("VEV_5500", 5500)]:
                od = state.order_depths.get(sym)
                if od:
                    m = mid(od)
                    if m and m > 0.3:
                        iv = implied_vol(m, vex_ema, K, T_rem)
                        if 0.001 < iv < 0.12:
                            vols.append(iv)
            if vols:
                sigma = self.SIGMA_ALPHA * (sum(vols)/len(vols)) + (1-self.SIGMA_ALPHA) * sigma

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 1: HYD market making — quote inside 16-tick spread
        # ══════════════════════════════════════════════════════════════════════
        hyd_pos = pos.get("HYDROGEL_PACK", 0)
        if hyd_od and hyd_ema:
            hyd_b = best_bid(hyd_od)
            hyd_a = best_ask(hyd_od)
            if hyd_b is not None and hyd_a is not None:
                skew    = int(self.HYD_SKEW * hyd_pos)
                bid_px  = hyd_b + self.HYD_EDGE - skew
                ask_px  = hyd_a - self.HYD_EDGE - skew

                if bid_px < ask_px:
                    orders = []
                    room_l = self.HYD_LIMIT - hyd_pos
                    room_s = self.HYD_LIMIT + hyd_pos
                    if room_l > 0:
                        orders.append(Order("HYDROGEL_PACK", bid_px,
                                            min(self.HYD_QTY, room_l)))
                    if room_s > 0:
                        orders.append(Order("HYDROGEL_PACK", ask_px,
                                            -min(self.HYD_QTY, room_s)))
                    if orders:
                        result["HYDROGEL_PACK"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 2: VEX market making — quote inside 5-tick spread
        # Use HYD as cross-asset signal (ratio ~1.9)
        # ══════════════════════════════════════════════════════════════════════
        vex_pos = pos.get("VELVETFRUIT_EXTRACT", 0)
        if vex_od and vex_ema and hyd_ema:
            vex_b = best_bid(vex_od)
            vex_a = best_ask(vex_od)
            if vex_b is not None and vex_a is not None:
                skew   = int(self.VEX_SKEW * vex_pos)
                bid_px = vex_b + self.VEX_EDGE - skew
                ask_px = vex_a - self.VEX_EDGE - skew

                if bid_px < ask_px:
                    orders = []
                    room_l = self.VEX_LIMIT - vex_pos
                    room_s = self.VEX_LIMIT + vex_pos
                    if room_l > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", bid_px,
                                            min(self.VEX_QTY, room_l)))
                    if room_s > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", ask_px,
                                            -min(self.VEX_QTY, room_s)))
                    if orders:
                        result["VELVETFRUIT_EXTRACT"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 3: Options — post AT market ask/bid, max quantity immediately
        #
        # Critical fixes vs v16:
        # 1. ask_px = min(our_fair_ask, mkt_ask)  ← join market ask, don't overshoot
        # 2. bid_px = max(our_fair_bid, mkt_bid)  ← join market bid queue
        # 3. OPT_QTY = OPT_LIMIT to fill as fast as possible
        # 4. Only skip if ask would cross below mkt_bid (nonsensical)
        # ══════════════════════════════════════════════════════════════════════
        if T_rem > 0.001 and vex_ema:
            for sym, K in self.OPT_STRIKES.items():
                od = state.order_depths.get(sym)
                if od is None:
                    continue

                mkt_b = best_bid(od)
                mkt_a = best_ask(od)
                if mkt_b is None or mkt_a is None:
                    continue

                fair    = bs_call(vex_ema, K, T_rem, sigma)
                if fair < 0.5:
                    continue

                opt_pos  = pos.get(sym, 0)
                opt_skew = int(self.OPT_SKEW * opt_pos) if hasattr(self, 'OPT_SKEW') else 0

                fair_int = round(fair)

                # BID: join at market bid (or our fair bid if higher)
                # We want to BUY options when they're cheap vs fair
                our_bid = max(fair_int - 1 - opt_skew, mkt_b)
                our_bid = min(our_bid, mkt_a - 1)   # never cross spread

                # ASK: join at market ask (or our fair ask if lower)
                # KEY FIX: clamp DOWN to mkt_ask so we actually get filled
                our_ask = min(fair_int + 1 - opt_skew, mkt_a)
                our_ask = max(our_ask, mkt_b + 1)   # never cross spread

                if our_bid <= 0 or our_ask <= our_bid:
                    continue

                orders = []
                room_l = self.OPT_LIMIT - opt_pos
                room_s = self.OPT_LIMIT + opt_pos

                # Post full limit qty to fill as fast as possible
                if room_l > 0:
                    orders.append(Order(sym, our_bid,  min(self.OPT_LIMIT, room_l)))
                if room_s > 0:
                    orders.append(Order(sym, our_ask, -min(self.OPT_LIMIT, room_s)))

                if orders:
                    result[sym] = orders

        traderData = json.dumps({
            "hyd_ema": hyd_ema,
            "vex_ema": vex_ema,
            "sigma":   sigma,
            "init":    init,
            "ts":      ts,
        })

        return result, 0, traderData

    OPT_SKEW = 0.01