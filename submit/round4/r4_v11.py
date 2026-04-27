from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

# ============================================================
# IMC Prosperity 4 — Round 4  v11
# ============================================================
#
# GAP ANALYSIS (v9/v10 = ~902/1000 ticks, top teams = 2667-6667):
#
# GAP 1 — FILL RATE (biggest gap, ~2411/1000 ticks missing):
#   Hard gate at 30% stops bidding at pos=60, but Mark 38 keeps
#   coming and we miss those fills. At 17.47 edge/fill, every
#   missed fill costs us real money. We captured 19% of theoretical.
#   FIX: Remove hard gate entirely. Use ONLY soft skew to manage
#   inventory. The skew makes our bid price less competitive as
#   we get longer, naturally reducing fill rate — but we never
#   stop quoting completely. This keeps us in the market always.
#
# GAP 2 — VEV_4000 (est ~100-200/1000 ticks):
#   VEV_4000 has Mark14/38 dynamics like HP, spread=20.9 (wider than HP!)
#   9 bot trades in 1000 ticks we're not intercepting.
#   FIX: Add VEV_4000 with same MM logic as HP.
#
# GAP 3 — VF FILL RATE:
#   38 units/level in VF book, we fill 6. Mark 55 noise trader
#   does ~400 trades/day (40/1000 ticks). We capture ~10%.
#   FIX: Same as HP — remove gate, let skew do the work.
#
# SOFT SKEW MECHANICS (replaces hard gate):
#   fair_mid = best_mid - (pos/limit) * SKEW * spread
#   When pos=0: fair_mid = best_mid (symmetric)
#   When pos=+limit: fair_mid = best_mid - SKEW*spread (lean to sell)
#   When pos=-limit: fair_mid = best_mid + SKEW*spread (lean to buy)
#   We always quote both sides, but prices shift naturally.
#   At high SKEW (1.0+), our bid falls below best_bid when very long
#   → effectively stops buying without a hard gate.

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
}

# Skew strength per product
# Higher = more aggressive inventory reversion
# HP and VEV_4000: wider spread so we can afford stronger skew
# VF: tighter spread, moderate skew
SKEW = {
    "HYDROGEL_PACK": 1.5,
    "VELVETFRUIT_EXTRACT": 1.0,
    "VEV_4000": 2.0,   # extra aggressive for delta-1 product
}


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
        skew  = SKEW[product]
        orders = []
        room_buy  = limit - pos
        room_sell = limit + pos

        # Pure soft skew — no hard gate
        # fair_mid shifts continuously against inventory
        norm_pos = pos / limit   # -1 to +1
        fair_mid = best_mid - norm_pos * skew * spread

        # TAKE: aggressively hit prices on the wrong side of fair_mid
        for ap in sorted(asks):
            if ap >= fair_mid or room_buy <= 0:
                break
            vol = min(asks[ap], room_buy)
            if vol > 0:
                orders.append(Order(product, ap, vol))
                room_buy -= vol

        for bp in sorted(bids, reverse=True):
            if bp <= fair_mid or room_sell <= 0:
                break
            vol = min(bids[bp], room_sell)
            if vol > 0:
                orders.append(Order(product, bp, -vol))
                room_sell -= vol

        # MAKE: overbid/underbid existing bot quotes for price priority
        # Bid: find highest bot-bid below fair_mid, place +1 above it
        if room_buy > 0:
            bid_price = int(best_bid)
            for bp in sorted(bids, reverse=True):
                if bp < fair_mid:
                    bid_price = bp + 1 if bids[bp] > 1 else bp
                    break
            # Never cross fair_mid
            bid_price = min(bid_price, int(fair_mid))
            if bid_price >= best_bid - 1:  # only post if competitive
                orders.append(Order(product, bid_price, room_buy))

        # Ask: find lowest bot-ask above fair_mid, place -1 below it
        if room_sell > 0:
            ask_price = int(best_ask)
            for ap in sorted(asks):
                if ap > fair_mid:
                    ask_price = ap - 1 if asks[ap] > 1 else ap
                    break
            # Never cross fair_mid
            ask_price = max(ask_price, int(fair_mid) + 1)
            if ask_price <= best_ask + 1:  # only post if competitive
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

        # VEV_4000: same Mark14/38 dynamics as HP, spread 20.9
        if "VEV_4000" in state.order_depths:
            pos = state.position.get("VEV_4000", 0)
            result["VEV_4000"] = self.mm(
                "VEV_4000", state.order_depths["VEV_4000"], pos
            )

        return result, 0, ""