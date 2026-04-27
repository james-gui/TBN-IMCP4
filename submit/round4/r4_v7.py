from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json


# ============================================================
# IMC Prosperity 4 — Round 4  v8
# ============================================================
#
# LEARNINGS FROM v7:
#   HP wall-mid approach: +1101  (was +415 in v4 — DOUBLED)
#   VF aggressive M67 take: -538  (was +258 in v4 — BROKEN)
#
# ROOT CAUSE OF VF FAILURE:
#   The M67 "TAKE at ask_wall to +100" strategy caused us to
#   buy 100 units at unfavorable prices every time M67 fired.
#   VF then kept falling, giving us massive MTM losses (-1091 at worst).
#   The problem: M67 trades at the ASK WALL (highest ask = worst buy price).
#   We paid max possible price, then VF drifted down.
#
# V8 STRATEGY:
#   HYDROGEL: Keep exact v7 wall-mid approach (it works!)
#   VELVETFRUIT: Wall-mid passive MM only. NO M67 aggressive taking.
#   The M67 signal bias (gentle long skew) from v3/v4 was better —
#   but even that wasn't great. Pure MM on VF is the safer choice.


LIMITS = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}


class Trader:

    def get_book(self, od: OrderDepth):
        bids = {p: abs(v) for p, v in od.buy_orders.items()}
        asks = {p: abs(v) for p, v in od.sell_orders.items()}
        return bids, asks

    def get_wall_mid(self, bids, asks):
        if not bids or not asks:
            return None, None, None
        bid_wall = min(bids.keys())   # deepest/lowest bid level
        ask_wall = max(asks.keys())   # deepest/highest ask level
        return bid_wall, ask_wall, (bid_wall + ask_wall) / 2

    def mm(self, product: str, od: OrderDepth, pos: int) -> List[Order]:
        """
        Frankfurt Hedgehogs StaticTrader wall-mid market making.
        Proven to work: HP went from +415 (v4) to +1101 (v7).

        Steps:
        1. TAKE: buy any ask < wall_mid, sell any bid > wall_mid
        2. CLEAR: unwind inventory at wall_mid
        3. MAKE: overbid best sub-wall-mid bid; underbid best supra-wall-mid ask
        """
        bids, asks = self.get_book(od)
        bid_wall, ask_wall, wall_mid = self.get_wall_mid(bids, asks)
        if wall_mid is None:
            return []

        limit = LIMITS[product]
        orders = []
        room_buy  = limit - pos
        room_sell = limit + pos

        # 1. TAKE
        for ap in sorted(asks):
            if ap >= wall_mid:
                break
            vol = min(asks[ap], room_buy)
            if vol > 0:
                orders.append(Order(product, ap, vol))
                room_buy -= vol

        for bp in sorted(bids, reverse=True):
            if bp <= wall_mid:
                break
            vol = min(bids[bp], room_sell)
            if vol > 0:
                orders.append(Order(product, bp, -vol))
                room_sell -= vol

        # 2. CLEAR inventory at wall_mid
        if pos > 0 and room_sell > 0:
            clear = min(pos, room_sell)
            orders.append(Order(product, int(wall_mid), -clear))
            room_sell -= clear
        elif pos < 0 and room_buy > 0:
            clear = min(-pos, room_buy)
            orders.append(Order(product, int(wall_mid + 0.5), clear))
            room_buy -= clear

        # 3. MAKE: undercut/overbid existing bot quotes
        bid_price = int(bid_wall + 1)
        for bp in sorted(bids, reverse=True):
            if bp < wall_mid:
                bid_price = max(bid_price, (bp + 1) if bids[bp] > 1 else bp)
                break

        ask_price = int(ask_wall - 1)
        for ap in sorted(asks):
            if ap > wall_mid:
                ask_price = min(ask_price, (ap - 1) if asks[ap] > 1 else ap)
                break

        if room_buy > 0:
            orders.append(Order(product, bid_price, room_buy))
        if room_sell > 0:
            orders.append(Order(product, ask_price, -room_sell))

        return orders

    def run(self, state: TradingState):
        result = {}

        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self.mm(
                "HYDROGEL_PACK", state.order_depths["HYDROGEL_PACK"], pos
            )

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self.mm(
                "VELVETFRUIT_EXTRACT", state.order_depths["VELVETFRUIT_EXTRACT"], pos
            )

        return result, 0, ""