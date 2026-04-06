# CLAUDE.md

## Project Overview
IMC Prosperity 4 trading algorithm submission. The algorithm runs inside the `prosperity4btx` backtester simulator and is evaluated against historical market data across multiple rounds/days.

## File Structure
- `message.py` — the trading algorithm (the `Trader` class). This is the only file submitted to the competition.
- `datamodel.py` — provided by IMC; defines all types used by the backtester. Do not modify.
- `run_backtest.py` — local automation script: runs the backtest and opens results in the visualizer.
- `backtests/` — auto-generated logs when running without `--out`.

## Setup (first time after cloning)
```bash
git clone --recurse-submodules <repo-url>
python setup_visualizer.py   # requires Node 18+ and pnpm; builds visualizer/dist/
```

## Running the Algorithm

### With auto-open visualizer (preferred)
```bash
python run_backtest.py 0
python run_backtest.py 0 --merge-pnl
```
Serves everything locally on port 8765 — no internet needed. Opens:
`http://localhost:8765/imc-prosperity-3-visualizer/?open=http://localhost:8765/output.log`

### Manually
```bash
prosperity4btx message.py 0 --out output.log
```

Day argument: `0` is round 0 (tutorial). Higher numbers correspond to later competition rounds.

## Algorithm Structure (`message.py`)

### Logger (do not change)
The `Logger` class and `logger = Logger()` global at the top of `message.py` are required boilerplate for the visualizer. Output must use `logger.print()` not `print()`. The `logger.flush(state, result, conversions, traderData)` call at the end of `run()` is mandatory.

The Logger class is NOT provided by the IMC platform at runtime — it must be present in the submitted file. The platform's starter template includes it as boilerplate; never strip it out.

### IMC template vs. this implementation
The IMC platform stub uses `trader_data = ""` (plain string, unused). This repo's implementation uses `trader_data` as a `dict` that is JSON-serialized into `traderData: str` before being passed to `logger.flush()` and returned. When adding new strategies, persist state by adding keys to the `trader_data` dict — not by using a bare string.

### Trader.run() contract
```python
def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
    ...
    return result, conversions, traderData
```
- `result`: dict mapping product symbol → list of `Order` objects to place this tick
- `conversions`: int, number of conversions requested (0 if unused)
- `traderData`: arbitrary string persisted across ticks (use JSON)

### Position limits
`POSITION_LIMIT = 80` per product. Orders that would exceed this are rejected by the exchange. Always check `state.position.get(product, 0)` before placing orders.

### Persisting state across ticks
Load at the top of `run()`:
```python
trader_data = json.loads(state.traderData) if state.traderData else {}
```
Serialize at the bottom before flush:
```python
traderData = json.dumps(trader_data)
```

## Current Strategies

### EMERALDS — Market Making
- Stable asset, tight spread around ~10,000
- Places buy at `best_bid + 1`, sell at `best_ask - 1`
- Order size: 15 units per side, capped by position limit

### TOMATOES — Mean Reversion
- Volatile asset, mean ~5,000
- Tracks last 100 mid-prices; uses 20-period rolling mean
- Buys when `best_ask < mean - 10`, sells when `best_bid > mean + 10`
- Falls back to market making at `mean ± 5` when no signal
- Order size: 10 units for signals, 5 units for market making

## Key Types (from `datamodel.py`)
- `TradingState`: snapshot of the market for one tick. Key fields: `timestamp`, `order_depths`, `position`, `own_trades`, `market_trades`, `observations`, `traderData`
- `OrderDepth`: `buy_orders: Dict[int, int]` and `sell_orders: Dict[int, int]` (price → volume)
- `Order(symbol, price, quantity)`: positive quantity = buy, negative = sell
- `Observation`: `plainValueObservations` and `conversionObservations` (for conversion arbitrage)

## Adding a New Product
1. Add an `elif product == "NEW_PRODUCT":` branch inside the `for product in state.order_depths` loop in `Trader.run()`
2. Follow the same pattern: get `order_depth`, compute signals, append to `orders`, respect `POSITION_LIMIT`
3. If you need per-product state, add a key to `trader_data` dict

## Visualizer Notes
- `visualizer/` is a git submodule (`jmerle/imc-prosperity-3-visualizer`). Built output lives at `visualizer/dist/`. Never commit `visualizer/dist/` — it is gitignored inside the submodule.
- `setup_visualizer.py` is the one-time build script. Re-run it if the submodule is updated (`git submodule update --remote visualizer`).
- `run_backtest.py` runs a single Python HTTP server on port 8765 routing:
  - `/imc-prosperity-3-visualizer/` → `visualizer/dist/`
  - `/output.log` → project root `output.log`
- The server must stay running while the visualizer is open in the browser.
- Use `logger.print("msg")` anywhere in `run()` to emit debug logs visible in the visualizer's Lambda Logs panel.
