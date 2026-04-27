from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

POSITION_LIMIT = 80


class Trader:
    def run(self, state: TradingState):
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

                    max_buy = POSITION_LIMIT - pos
                    max_sell = POSITION_LIMIT + pos

                    if max_buy > 0:
                        orders.append(Order(product, best_bid + 1, min(15, max_buy)))
                    if max_sell > 0:
                        orders.append(Order(product, best_ask - 1, -min(15, max_sell)))

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

            result[product] = orders

        traderData = json.dumps(trader_data)
        return result, 0, traderData
