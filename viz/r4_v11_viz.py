import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List


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

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
}

SKEW = {
    "HYDROGEL_PACK": 1.5,
    "VELVETFRUIT_EXTRACT": 1.0,
    "VEV_4000": 2.0,
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

        norm_pos = pos / limit
        fair_mid = best_mid - norm_pos * skew * spread

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

        if room_buy > 0:
            bid_price = int(best_bid)
            for bp in sorted(bids, reverse=True):
                if bp < fair_mid:
                    bid_price = bp + 1 if bids[bp] > 1 else bp
                    break
            bid_price = min(bid_price, int(fair_mid))
            if bid_price >= best_bid - 1:
                orders.append(Order(product, bid_price, room_buy))

        if room_sell > 0:
            ask_price = int(best_ask)
            for ap in sorted(asks):
                if ap > fair_mid:
                    ask_price = ap - 1 if asks[ap] > 1 else ap
                    break
            ask_price = max(ask_price, int(fair_mid) + 1)
            if ask_price <= best_ask + 1:
                orders.append(Order(product, ask_price, -room_sell))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
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

        if "VEV_4000" in state.order_depths:
            pos = state.position.get("VEV_4000", 0)
            result["VEV_4000"] = self.mm(
                "VEV_4000", state.order_depths["VEV_4000"], pos
            )

        logger.flush(state, result, 0, "")
        return result, 0, ""
