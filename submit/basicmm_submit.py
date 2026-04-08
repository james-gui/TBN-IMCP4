from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ── Strategy: Basic Market Making ─────────────────────────────────────────────
#
# Core principle: undercut the competition to attract market takers.
#
# If the best public bid is 9992 and best ask is 10008, other market makers are
# quoting somewhere inside that spread. We beat them by quoting one tick better:
#   our_bid = best_bid + 1  →  e.g. 9993  (best deal for sellers / takers)
#   our_ask = best_ask - 1  →  e.g. 10007 (best deal for buyers / takers)
#
# We earn the spread between our buy and sell fills. The tighter we quote, the
# more fills we attract — but we need the spread > 1 tick to make money.
#
# Applied to BOTH products (EMERALDS and TOMATOES) identically.
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid

            # Only quote if there's room to undercut (spread must be > 1 tick)
            if spread > 1:
                our_bid = best_bid + 1   # one tick better than best public bid
                our_ask = best_ask - 1   # one tick better than best public ask

                max_buy = POSITION_LIMIT - pos
                max_sell = POSITION_LIMIT + pos

                if max_buy > 0:
                    orders.append(Order(product, our_bid, min(20, max_buy)))
                if max_sell > 0:
                    orders.append(Order(product, our_ask, -min(20, max_sell)))

            result[product] = orders

        return result, 0, ""
