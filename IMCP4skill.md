---
name: imc-prosperity-trading
description: Use when implementing trading algorithms for IMC Prosperity challenge, working with order books, or writing Python trading bots
---

# IMC Prosperity Trading

## Minimal Working Example

```python
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        POSITION_LIMIT = 10  # varies per product — check rules

        trader_data = json.loads(state.traderData) if state.traderData else {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)
            max_buy = POSITION_LIMIT - position
            max_sell = POSITION_LIMIT + position

            if order_depth.sell_orders and max_buy > 0:
                best_ask = min(order_depth.sell_orders.keys())
                orders.append(Order(product, best_ask, min(5, max_buy)))  # BUY

            if order_depth.buy_orders and max_sell > 0:
                best_bid = max(order_depth.buy_orders.keys())
                orders.append(Order(product, best_bid, -min(5, max_sell)))  # SELL

            result[product] = orders

        conversions = 0
        return result, conversions, json.dumps(trader_data)
```

## File Structure

```python
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """Returns: (orders_dict, conversions, traderData)"""
        pass
```

## Data Structures

### TradingState
```python
@dataclass
class TradingState:
    timestamp: int                              # Increments by 100 each iteration
    traderData: str                             # Empty string "" on first run, not None
    position: Dict[str, int]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]          # YOUR fills from the PREVIOUS timestamp
    market_trades: Dict[str, List[Trade]]       # All other market fills from PREVIOUS timestamp
    observations: Observation
```

### OrderDepth
```python
@dataclass
class OrderDepth:
    buy_orders: Dict[int, int]   # price -> quantity (POSITIVE)
    sell_orders: Dict[int, int]  # price -> quantity (NEGATIVE)
```

### Order
```python
@dataclass
class Order:
    symbol: str
    price: int        # MUST be int — float prices are rejected silently
    quantity: int     # Positive=BUY, Negative=SELL

Order("PRODUCT", 100, 5)   # Buy
Order("PRODUCT", 100, -5)  # Sell
```

### Trade
```python
@dataclass
class Trade:
    symbol: str
    price: int
    quantity: int
    buyer: str    # "SUBMISSION" if you bought
    seller: str   # "SUBMISSION" if you sold
    timestamp: int
```

### ConversionObservation (Round 2+ only — unavailable in tutorial)
```python
@dataclass
class ConversionObservation:
    bidPrice: float
    askPrice: float
    transportFees: float
    exportTariff: float
    importTariff: float
    sugarPrice: float      # Round 3+
    sunlightIndex: float   # Round 3+

# Access pattern:
obs = state.observations.conversionObservations.get(product)
if obs:
    print(obs.bidPrice, obs.askPrice, obs.transportFees)
```

## Common Operations

```python
# Best prices — always guard against empty order book
best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None

# Prices for orders MUST be int — cast floats explicitly
order_price = int(mid_price)          # or int(round(mid_price))

# Quantities (positive)
bid_qty = order_depth.buy_orders[price]           # Already positive
ask_qty = -order_depth.sell_orders[price]         # Convert negative to positive

# Position
current_position = state.position.get(product, 0)
max_buy = position_limit - current_position
max_sell = position_limit + current_position

# Multiple orders at different price levels are allowed for the same product
orders.append(Order(product, best_ask, buy_qty))          # aggressive fill
orders.append(Order(product, best_ask - 2, passive_qty))  # passive resting quote

# State persistence
import json
trader_data = json.loads(state.traderData) if state.traderData else {}
traderData = json.dumps(trader_data)

# Check your own fills from the previous timestamp
for trade in state.own_trades.get(product, []):
    if trade.buyer == "SUBMISSION":
        print(f"Filled BUY {trade.quantity} @ {trade.price}")
    elif trade.seller == "SUBMISSION":
        print(f"Filled SELL {trade.quantity} @ {trade.price}")
```

## Return Value

```python
result = {
    "PRODUCT1": [Order("PRODUCT1", 100, 5), Order("PRODUCT1", 110, -5)],
    "PRODUCT2": [Order("PRODUCT2", 200, 10)]
}
conversions = 0  # Round 2+ only; use 0 in tutorial round
traderData = json.dumps({"key": "value"})

return result, conversions, traderData
```

## Constraints

- **Timeout:** 900ms per `run()` call
- **State size:** 50,000 characters max for `traderData`
- **Position limits:** Enforced per-product (check rules for each product)
- **Libraries:** Standard Python 3.12 + pandas, numpy, statistics, math, typing, jsonpickle
- **Iterations:** 1,000 (testing), 10,000 (final)
- **Timestamp:** Increments by 100 each iteration

## Important Notes

- Sell order quantities in `OrderDepth.sell_orders` are NEGATIVE
- Order quantities: Positive = BUY, Negative = SELL
- **Order prices must be `int`** — passing a float silently rejects the order
- Orders exceeding position limits are silently rejected (no error thrown)
- Orders execute immediately if price crosses the spread
- Resting orders auto-cancel at end of iteration
- `traderData` persists across iterations (max 50,000 chars); it's `""` (empty string) on the very first run
- `own_trades` contains YOUR fills from the **previous** timestamp, not the current one
- You can submit multiple `Order` objects at different price levels for the same product
- Guard against empty `buy_orders` / `sell_orders` dicts before calling `max()` / `min()`
- `conversions` is only used in Round 2+; always return `0` in the tutorial round
