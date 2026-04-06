from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import List, Dict, Any
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
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
