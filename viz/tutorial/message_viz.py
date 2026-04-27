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

POSITION_LIMIT = 80


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                trader_data = {}

        if "tomato_prices" not in trader_data:
            trader_data["tomato_prices"] = []

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "EMERALDS":
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())

                    our_bid = best_bid + 1
                    our_ask = best_ask - 1

                    max_buy = POSITION_LIMIT - pos
                    max_sell = POSITION_LIMIT + pos

                    if max_buy > 0:
                        orders.append(Order(product, our_bid, min(15, max_buy)))
                    if max_sell > 0:
                        orders.append(Order(product, our_ask, -min(15, max_sell)))

                    logger.print(f"[EMERALDS] bid={best_bid} ask={best_ask} our_bid={our_bid} our_ask={our_ask} pos={pos}")

            elif product == "TOMATOES":
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2

                    trader_data["tomato_prices"].append(mid_price)
                    if len(trader_data["tomato_prices"]) > 100:
                        trader_data["tomato_prices"] = trader_data["tomato_prices"][-100:]

                    if len(trader_data["tomato_prices"]) >= 20:
                        mean_price = sum(trader_data["tomato_prices"][-20:]) / 20
                    else:
                        mean_price = 5000

                    max_buy = POSITION_LIMIT - pos
                    max_sell = POSITION_LIMIT + pos

                    if best_ask < mean_price - 10 and max_buy > 0:
                        orders.append(Order(product, best_ask, min(10, max_buy)))
                    if best_bid > mean_price + 10 and max_sell > 0:
                        orders.append(Order(product, best_bid, -min(10, max_sell)))

                    if not orders and len(trader_data["tomato_prices"]) >= 5:
                        our_bid = int(mean_price - 5)
                        our_ask = int(mean_price + 5)
                        if max_buy > 0 and our_bid < best_ask:
                            orders.append(Order(product, our_bid, min(5, max_buy)))
                        if max_sell > 0 and our_ask > best_bid:
                            orders.append(Order(product, our_ask, -min(5, max_sell)))

                    logger.print(f"[TOMATOES] mean={mean_price:.1f} bid={best_bid} ask={best_ask} pos={pos}")

            result[product] = orders

        traderData = json.dumps(trader_data)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
