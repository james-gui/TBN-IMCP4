# CLAUDE.md

## Project Overview
IMC Prosperity 4 trading algorithm submission. The algorithm runs inside the `prosperity4btx` backtester simulator and is evaluated against historical market data across multiple rounds/days.

## File Naming Convention

Every strategy has two paired files:

| Suffix | Purpose | Has Logger | Use with |
|---|---|---|---|
| `_viz.py` | Backtesting + visualizer | YES | `run_backtest.py --algo` |
| `_submit.py` | IMC platform submission | NO | Upload to IMC directly |

**Rule**: whenever a `_viz.py` file is created or modified, the corresponding `_submit.py` must also be created/updated to match — same logic, same Logger class included (IMC requires it), same `logger.flush()` call.

Current strategy pairs:
- `asymmm` — asymmetric MM with inventory skew and EMA fair value
- `attempt2` — momentum MM using fast/slow EMA crossover to skew quotes
- `basicmm` — basic MM undercutting best bid/ask by one tick
- `flatmm` — symmetric MM with basic inventory skew
- `impmm` — improved MM with tighter taker thresholds and per-product limits
- `jamestutorialr` — tutorial version, MM on EMERALDS + mean reversion on TOMATOES
- `message` — momentum MM with adaptive EMA-based trend skew
- `mmfinetuned` — fine-tuned MM with cubic inventory skew and wider quotes
- `mmspreadgate` — fine-tuned MM variant with optimized parameters
- `momomm` — momentum-enhanced MM extending mmfinetuned with EMA signals
- `spreadmm` — spread-filtered MM that only quotes above min profit threshold
- `targetmm` — target-inventory MM using mean reversion to set position targets

## File Structure
- `viz/` — strategy files with Logger (for backtesting/visualizer). Also contains a copy of `datamodel.py` so prosperity4btx can find it when running files from this folder.
- `submit/` — strategy files **with Logger** (upload these to IMC). Logger is required by IMC. No datamodel.py needed — IMC provides it at runtime.
- `datamodel.py` — root copy, source of truth. If IMC updates it, copy it to `viz/datamodel.py` too.
- `run_backtest.py` — runs backtest + opens local visualizer automatically
- `setup_visualizer.py` — one-time build script for the local visualizer
- `visualizer/` — git submodule: jmerle/imc-prosperity-3-visualizer
- `backtests/` — auto-generated logs when running without `--out` (gitignored)

## Setup (first time after cloning)
```bash
git clone --recurse-submodules <repo-url>
python setup_visualizer.py   # requires Node 18+ and pnpm; builds visualizer/dist/
```

## Running the Algorithm

### With auto-open visualizer (preferred)
```bash
python run_backtest.py 0                               # runs viz/message_viz.py (default)
python run_backtest.py 0 --algo viz/attempt2_viz.py   # runs a specific strategy
python run_backtest.py 0 --merge-pnl
python run_backtest.py 0 --data data/                 # explicit custom data directory
```
Serves everything locally on port 8765 — no internet needed.

**Custom data:** If a `data/` directory exists at the project root, it is used automatically (no flag needed). Place CSV files in `data/round<N>/` following the naming pattern `prices_round_<N>_day_<D>.csv` and `trades_round_<N>_day_<D>.csv`.

### Manually
```bash
prosperity4btx viz/message_viz.py 0 --out output.log
```

Day argument: `0` is round 0 (tutorial). Higher numbers correspond to later competition rounds.

## Logger Rules (viz files only)

- **Do not modify** the `Logger` class — required verbatim for the visualizer
- **Always call** `logger.flush(state, result, conversions, traderData)` as the last line before `return`
- **Use `logger.print()`** instead of `print()` for debug output — bare `print()` breaks the log format
- The Logger class must be present in every `_viz.py` file

### IMC template vs. this implementation
The IMC platform stub uses `trader_data = ""` (plain string, unused). This repo uses `trader_data` as a `dict` serialized to JSON. Persist state by adding keys to the dict, not with a bare string.

## Trader.run() contract
```python
def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
    ...
    return result, conversions, traderData
```
- `result`: dict mapping product symbol → list of `Order` objects to place this tick
- `conversions`: int, number of conversions requested (0 if unused)
- `traderData`: arbitrary string persisted across ticks (use JSON)

## Position Limits
`POSITION_LIMIT = 80` per product. Always check `state.position.get(product, 0)` before placing orders.

## Key Types (from `datamodel.py`)
- `TradingState`: `timestamp`, `order_depths`, `position`, `own_trades`, `market_trades`, `observations`, `traderData`
- `OrderDepth`: `buy_orders: Dict[int, int]` and `sell_orders: Dict[int, int]` (price → volume)
- `Order(symbol, price, quantity)`: positive quantity = buy, negative = sell

## Adding a New Strategy
1. Create `viz/{name}_viz.py` with Logger boilerplate + trading logic
2. Create matching `submit/{name}_submit.py` — identical logic, Logger class removed, `logger.flush()` removed, `logger.print()` → `print()`
3. Test with `python run_backtest.py 0 --algo viz/{name}_viz.py`
4. Submit `submit/{name}_submit.py` to IMC

## Visualizer Notes
- `visualizer/` is a git submodule (`jmerle/imc-prosperity-3-visualizer`). Never commit `visualizer/dist/`.
- `run_backtest.py` runs a single Python HTTP server on port 8765:
  - `/imc-prosperity-3-visualizer/` → `visualizer/dist/`
  - `/output.log` → project root `output.log`
- Server must stay running while the visualizer is open in the browser
- Use `logger.print("msg")` in `_viz.py` files for debug logs visible in the Lambda Logs panel
