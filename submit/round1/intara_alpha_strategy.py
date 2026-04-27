from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if state.traderData:
            data = json.loads(state.traderData)
        else:
            data = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            orders: List[Order] = []

            if product == "INTARIAN_PEPPER_ROOT":
                orders = self.trade_pepper_root(state, order_depth, position)

            elif product == "ASH_COATED_OSMIUM":
                prev_mid = data.get("osmium_prev_mid")
                orders, new_mid = self.trade_osmium(state, order_depth, position, prev_mid)
                if new_mid is not None:
                    data["osmium_prev_mid"] = new_mid

            result[product] = orders

        trader_data = json.dumps(data)
        return result, conversions, trader_data

    def best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return best_bid or best_ask

    def estimate_pepper_fair(self, state: TradingState, order_depth: OrderDepth):
        mid = self.mid_price(order_depth)
        if mid is None:
            return None
        day_base = round(mid / 1000) * 1000
        return day_base + state.timestamp / 1000.0

    def trade_pepper_root(self, state: TradingState, order_depth: OrderDepth, position: int):
        orders = []
        limit = POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        fair = self.estimate_pepper_fair(state, order_depth)
        if fair is None:
            return orders

        best_bid, best_ask = self.best_bid_ask(order_depth)
        buy_capacity = limit - position
        sell_capacity = limit + position

        acceptable_buy = math.floor(fair + 6)

        if order_depth.sell_orders and buy_capacity > 0:
            for ask in sorted(order_depth.sell_orders):
                vol = -order_depth.sell_orders[ask]
                if ask <= acceptable_buy:
                    qty = min(vol, buy_capacity)
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                    buy_capacity -= qty

        if buy_capacity > 0:
            bid_price = math.floor(fair - 2)
            if best_bid:
                bid_price = min(bid_price, best_bid + 1)
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, buy_capacity))

        acceptable_sell = math.ceil(fair + 10)

        if order_depth.buy_orders and sell_capacity > 0:
            for bid in sorted(order_depth.buy_orders, reverse=True):
                vol = order_depth.buy_orders[bid]
                if bid >= acceptable_sell:
                    qty = min(vol, sell_capacity)
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                    sell_capacity -= qty

        return orders

    def trade_osmium(self, state: TradingState, order_depth: OrderDepth, position: int, prev_mid):
        orders = []
        limit = POSITION_LIMITS["ASH_COATED_OSMIUM"]

        mid = self.mid_price(order_depth)
        if mid is None:
            return orders, prev_mid

        if prev_mid is None:
            fair = mid
        else:
            fair = mid - 0.5 * (mid - prev_mid)

        buy_capacity = limit - position
        sell_capacity = limit + position

        skew = 0.05 * position

        take_buy = math.floor(fair - 2 - skew)
        take_sell = math.ceil(fair + 2 - skew)

        if order_depth.sell_orders and buy_capacity > 0:
            for ask in sorted(order_depth.sell_orders):
                vol = -order_depth.sell_orders[ask]
                if ask <= take_buy:
                    qty = min(vol, buy_capacity)
                    orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                    buy_capacity -= qty

        if order_depth.buy_orders and sell_capacity > 0:
            for bid in sorted(order_depth.buy_orders, reverse=True):
                vol = order_depth.buy_orders[bid]
                if bid >= take_sell:
                    qty = min(vol, sell_capacity)
                    orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                    sell_capacity -= qty

        best_bid, best_ask = self.best_bid_ask(order_depth)

        bid_price = math.floor(fair - 3 - skew)
        ask_price = math.ceil(fair + 3 - skew)

        if best_bid:
            bid_price = min(bid_price, best_bid + 1)
        if best_ask:
            ask_price = max(ask_price, best_ask - 1)

        if bid_price < ask_price:
            size = 12
            if buy_capacity > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid_price, min(size, buy_capacity)))
            if sell_capacity > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask_price, -min(size, sell_capacity)))

        return orders, mid
