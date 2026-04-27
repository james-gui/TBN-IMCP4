from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

# ============================================================
# IMC Prosperity 4 — Round 4  v10
# ============================================================
#
# V9 RESULTS (IMC official): +902 total (HP +559, VF +342)
# Best result so far. Chart: clean steady climb.
#
# REMAINING PROBLEM: -676 drawdown from peak (42%)
# ROOT CAUSE: inventory MTM at end of run
#   HP accumulated +20 long, price fell 32.5 pts = ~650 MTM loss
#   VF went from +358 to -68 near end (similar: long position + price fall)
#
# V10 CHANGES (inventory management only, nothing else):
#   HARD_GATE: 0.40 → 0.30 (stop bidding at ±60 units instead of ±80)
#   SOFT_SKEW: 0.30 → 0.50 (stronger fair value tilt against inventory)
#   This keeps positions smaller throughout and unwinds faster.

LIMITS    = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
HARD_GATE = 0.30   # stop new quotes on heavy side beyond 30% of limit (±60)
SOFT_SKEW = 0.50   # fair value shifts 0.5 spreads per unit normalised position


class Trader:

    def get_book(self, od: OrderDepth):
        bids = {p: abs(v) for p, v in od.buy_orders.items()}
        asks = {p: abs(v) for p, v in od.sell_orders.items()}
        return bids, asks

    def mm(self, product: str, od: OrderDepth, pos: int) -> List[Order]:
        bids, asks = self.get_book(od)
        if not bids or not asks:
            return []

        best_bid = max(bids)
        best_ask = min(asks)
        spread   = best_ask - best_bid
        best_mid = (best_bid + best_ask) / 2

        limit = LIMITS[product]
        orders = []
        room_buy  = limit - pos
        room_sell = limit + pos

        # Soft skew: shift fair value against inventory
        norm_pos = pos / limit
        fair_mid = best_mid - norm_pos * SOFT_SKEW * spread

        # Hard gate: stop adding to large positions
        gate     = HARD_GATE * limit
        can_buy  = pos <  gate
        can_sell = pos > -gate

        # TAKE: hit prices on wrong side of fair_mid
        for ap in sorted(asks):
            if ap >= fair_mid or not can_buy:
                break
            vol = min(asks[ap], room_buy)
            if vol > 0:
                orders.append(Order(product, ap, vol))
                room_buy -= vol

        for bp in sorted(bids, reverse=True):
            if bp <= fair_mid or not can_sell:
                break
            vol = min(bids[bp], room_sell)
            if vol > 0:
                orders.append(Order(product, bp, -vol))
                room_sell -= vol

        # MAKE: overbid/underbid for price priority
        if can_buy and room_buy > 0:
            bid_price = int(best_bid)
            for bp in sorted(bids, reverse=True):
                if bp < fair_mid:
                    bid_price = bp + 1 if bids[bp] > 1 else bp
                    break
            bid_price = min(bid_price, int(fair_mid))
            orders.append(Order(product, bid_price, room_buy))

        if can_sell and room_sell > 0:
            ask_price = int(best_ask)
            for ap in sorted(asks):
                if ap > fair_mid:
                    ask_price = ap - 1 if asks[ap] > 1 else ap
                    break
            ask_price = max(ask_price, int(fair_mid) + 1)
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