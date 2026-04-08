import json
from typing import Any, List, Dict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice, observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

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
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
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

                logger.print(
                    f"[{product}] spread={spread} best={best_bid}/{best_ask} "
                    f"quoting={our_bid}/{our_ask} pos={pos}"
                )
            else:
                # Spread is 1 tick — no room to undercut without crossing
                logger.print(f"[{product}] spread={spread} too tight, skipping")

            result[product] = orders

        logger.flush(state, result, 0, "")
        return result, 0, ""
