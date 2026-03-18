from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

class Trader:
    def run(self, state: TradingState):
        """
        Algorithmic trading bot implementing:
        1. Market Making on EMERALDS (stable asset)
        2. Mean Reversion on TOMATOES (volatile asset)
        """
        result: Dict[str, List[Order]] = {}
        POSITION_LIMIT = 80
        
        # Load previous state if exists
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except:
                trader_data = {}
        
        # Initialize price history for tomatoes
        if "tomato_prices" not in trader_data:
            trader_data["tomato_prices"] = []
        
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            
            if product == "EMERALDS":
                # STRATEGY 1: Market Making on Emeralds
                # Emeralds are stable around 10,000 with tight spread
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid
                    
                    # Place orders just inside the spread to capture edge
                    # Buy slightly above best bid, sell slightly below best ask
                    our_bid_price = best_bid + 1
                    our_ask_price = best_ask - 1
                    
                    # Calculate order sizes respecting position limits
                    max_buy = POSITION_LIMIT - current_position
                    max_sell = POSITION_LIMIT + current_position
                    
                    # Place buy order if we have room
                    if max_buy > 0:
                        buy_qty = min(15, max_buy)  # Trade in reasonable chunks
                        orders.append(Order(product, our_bid_price, buy_qty))
                    
                    # Place sell order if we have room
                    if max_sell > 0:
                        sell_qty = min(15, max_sell)
                        orders.append(Order(product, our_ask_price, -sell_qty))
            
            elif product == "TOMATOES":
                # STRATEGY 2: Mean Reversion on Tomatoes
                # Tomatoes fluctuate around 5000 mean
                if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Track price history
                    trader_data["tomato_prices"].append(mid_price)
                    # Keep only last 100 prices for efficiency
                    if len(trader_data["tomato_prices"]) > 100:
                        trader_data["tomato_prices"] = trader_data["tomato_prices"][-100:]
                    
                    # Calculate mean and threshold
                    if len(trader_data["tomato_prices"]) >= 20:
                        mean_price = sum(trader_data["tomato_prices"][-20:]) / 20
                    else:
                        mean_price = 5000  # Default mean
                    
                    # Mean reversion thresholds
                    buy_threshold = mean_price - 10  # Buy when below mean
                    sell_threshold = mean_price + 10  # Sell when above mean
                    
                    # Calculate order sizes respecting position limits
                    max_buy = POSITION_LIMIT - current_position
                    max_sell = POSITION_LIMIT + current_position
                    
                    # BUY logic: If ask price is below threshold and we have room
                    if best_ask < buy_threshold and max_buy > 0:
                        buy_qty = min(10, max_buy)
                        orders.append(Order(product, best_ask, buy_qty))
                    
                    # SELL logic: If bid price is above threshold and we have room
                    if best_bid > sell_threshold and max_sell > 0:
                        sell_qty = min(10, max_sell)
                        orders.append(Order(product, best_bid, -sell_qty))
                    
                    # Market making around fair value when not trading
                    if not orders and len(trader_data["tomato_prices"]) >= 5:
                        # Place orders near fair value
                        our_bid = int(mean_price - 5)
                        our_ask = int(mean_price + 5)
                        
                        if max_buy > 0 and our_bid < best_ask:
                            orders.append(Order(product, our_bid, min(5, max_buy)))
                        
                        if max_sell > 0 and our_ask > best_bid:
                            orders.append(Order(product, our_ask, -min(5, max_sell)))
            
            result[product] = orders
        
        # Serialize state for next iteration
        traderData = json.dumps(trader_data)
        
        # No conversions in tutorial round
        conversions = 0
        
        return result, conversions, traderData
