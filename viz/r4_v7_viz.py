import json
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import Any, List, Dict, Optional


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

        logger.flush(state, result, 0, "")
        return result, 0, ""
