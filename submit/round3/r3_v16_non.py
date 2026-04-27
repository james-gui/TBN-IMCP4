from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
import math


SIGMA   = 0.28
TTE_LIQ = 4.0 / 365   # years remaining at liquidation (4 days)

def bs_call(S: float, K: float, T: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1  = (math.log(S / K) + 0.5 * SIGMA**2 * T) / (SIGMA * sqT)
    d2  = d1 - SIGMA * sqT
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    return S * nd1 - K * nd2

def T_now(ts: int) -> float:
    """Time remaining in years at timestamp ts."""
    return max(0.0, (5.0 - ts / 100_000) / 365.0)

def best_bid(od: OrderDepth):
    return max(od.buy_orders.keys()) if od.buy_orders else None

def best_ask(od: OrderDepth):
    return min(od.sell_orders.keys()) if od.sell_orders else None


class Trader:
    """
    CONFIRMED MODEL:
      sigma_annual=0.28, T=(5-ts/100000)/365 years
      Liquidation fair = BS(S_final, K, 4/365, 0.28)

    OPTION EDGE (sell at mkt_bid aggressively, guaranteed fills):
      K=5300: bid≈52  → liq≈45.4 → +6.6/contract × 300 = +1972
      K=5200: bid≈102 → liq≈98.4 → +3.6/contract × 300 = +1071
      K=5500: bid≈6   → liq≈4.7  → +1.3/contract × 300 = +402
      K=5400: SKIP (negative edge at bid)
      Total guaranteed: ~3445

    FIX: Previously both aggressive (-300) and passive (-300) orders were sent
    per symbol → total -600 → "exceeded limit of 300" error every tick.
    Now: ONE order per symbol per tick. Aggressive (hit bid) only.
    Once position is filled at -300, stop ordering that symbol.
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

    # Aggressive short: hit market bid immediately for guaranteed fills.
    # Edge = bid_price - BS(S_final, K, 4/365, 0.28) > 0 for these strikes.
    OPT_SHORTS = {
        "VEV_5200": 5200,   # edge ≈ +3.6/contract
        "VEV_5300": 5300,   # edge ≈ +6.6/contract  ← best
        "VEV_5500": 5500,   # edge ≈ +1.3/contract
        # VEV_5400: SKIP — negative edge at bid (-0.57)
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
                    orders.append(Order("HYDROGEL_PACK", bid_px,
                                        min(self.HYD_QTY, room_l)))
                if room_s > 0:
                    orders.append(Order("HYDROGEL_PACK", ask_px,
                                        -min(self.HYD_QTY, room_s)))
                if orders:
                    result["HYDROGEL_PACK"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 2: VEX market making
        # ══════════════════════════════════════════════════════════════════════
        vex_pos = pos.get("VELVETFRUIT_EXTRACT", 0)
        if vex_b and vex_a:
            spd  = vex_a - vex_b
            skew = int(self.VEX_SKEW * vex_pos)
            orders = []
            room_l = self.VEX_LIMIT - vex_pos
            room_s = self.VEX_LIMIT + vex_pos

            if spd >= 3:
                bid_px = vex_b + self.VEX_EDGE - skew
                ask_px = vex_a - self.VEX_EDGE - skew
                if bid_px < ask_px:
                    if room_l > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", bid_px,
                                            min(self.VEX_QTY, room_l)))
                    if room_s > 0:
                        orders.append(Order("VELVETFRUIT_EXTRACT", ask_px,
                                            -min(self.VEX_QTY, room_s)))
            else:
                # Tight spread (1-2 ticks): join queue
                if room_l > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", vex_b,
                                        min(self.VEX_QTY, room_l)))
                if room_s > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", vex_a,
                                        -min(self.VEX_QTY, room_s)))

            if orders:
                result["VELVETFRUIT_EXTRACT"] = orders

        # ══════════════════════════════════════════════════════════════════════
        # LAYER 3: Options — ONE order per symbol per tick
        #
        # AGGRESSIVE SHORT: sell at market bid (taker order, immediate fill).
        # Edge confirmed positive for K=5200, 5300, 5500.
        #
        # ONE order per symbol = qty capped at remaining room to -OPT_LIMIT.
        # Once fully short (-300), no more orders for that symbol.
        # No passive ask order to avoid double-counting the position limit.
        # ══════════════════════════════════════════════════════════════════════
        if vex_ema:
            for sym, K in self.OPT_SHORTS.items():
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                mkt_b = best_bid(od)
                if mkt_b is None:
                    continue

                opt_pos = pos.get(sym, 0)
                room_s  = self.OPT_LIMIT + opt_pos  # remaining short capacity

                if room_s <= 0:
                    continue   # already at full short limit

                # Verify edge is still positive with current VEX
                fair_liq = bs_call(vex_ema, K, TTE_LIQ)
                if mkt_b <= fair_liq:
                    continue   # edge gone (S moved up too much), skip

                # Single sell order at market bid — immediate fill
                qty = min(self.OPT_LIMIT, room_s)
                result[sym] = [Order(sym, mkt_b, -qty)]

        traderData = json.dumps({
            "hyd_ema": hyd_ema,
            "vex_ema": vex_ema,
            "init":    init,
            "ts":      ts,
        })

        return result, 0, traderData