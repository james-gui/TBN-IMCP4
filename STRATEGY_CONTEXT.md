# Strategy Context

Reference this file when writing new trading strategies for this repo.

---

## Competition Overview (IMC Prosperity 3 — Frankfurt Hedgehogs, 2nd place globally)

Source: Frankfurt Hedgehogs writeup (Timo Diehm, Arne Witt, Marvin Schuster)

### Core Philosophy
- Deep structural understanding beats clever code. Always ask: *how was this data generated?*
- Simple, robust strategies with few parameters beat overfitted complex ones
- Never deploy a strategy you can't explain from first principles
- Prioritize flat parameter landscapes over peak backtested performance
- The backtester is for **relative comparison** between strategies, not absolute P&L prediction

---

## Order Flow Mechanics (Critical)

At each timestep the simulation processes orders in this sequence:
1. Clear all previous orders
2. Deep-liquidity maker bots post orders
3. Occasional taker bots fill
4. **Our bot acts** (take or make)
5. Other bots (usually more takers)

**Implication**: Speed and cancellation are irrelevant. You get a full snapshot of the book and can submit any combination of passive/aggressive orders. No race conditions.

---

## Products (Round 0 / Tutorial — Prosperity 4)

| Product | Nature | Typical Price | Notes |
|---|---|---|---|
| EMERALDS | Stable | ~10,000 | Tight spread, low volatility, good for pure market making |
| TOMATOES | Volatile | ~5,000 | Wider spread, may trend or mean-revert |

---

## Market Making Principles (from Rainforest Resin / Kelp analysis)

- **True price = WallMid** = best approximation of fair value at each tick
- **WallMid** = mid-price of the deepest/most stable bids and asks in the book
- Goal: buy below true price, sell above true price. The gap is your "edge"
- Quote 1 tick better than the best existing orders to attract takers
- If spread ≤ 1 tick: no room to profitably undercut both sides — skip or use a different approach
- Immediately take any fills available below (for buys) or above (for sells) fair value before placing maker quotes
- Flatten inventory at zero edge (true price) when position becomes too skewed

### What Works
- Inside-spread quoting (`best_bid + 1` / `best_ask - 1`) reliably gets fills
- Larger order sizes capture more fill when spread is available
- Simple strategies outperform complex ones — avoid overfitting
- Always quoting both sides beats signal-gated strategies

### What Doesn't Work
- Quotes outside the spread — never get filled
- Tight position limits — cap P&L even with correct logic
- Momentum/EMA strategies — underperform on stable/range-bound data
- Mean reversion signals that fire rarely — most profit comes from the fallback MM anyway

---

## Strategy Performance (Round 0, backtester days -2 and -1)

| Strategy | File | Total P&L | Notes |
|---|---|---|---|
| Improved Market Making | `impmm_viz.py` | 30,321 | take_edge=0 for EMERALDS + TOMATOES soft_limit=35 |
| Basic Market Making | `basicmm_viz.py` | 29,496 | Undercuts best bid/ask by 1 tick on both products |
| Message (MM + mean rev) | `message_viz.py` | 27,346 | MM on EMERALDS, mean reversion on TOMATOES |
| James Tutorial | `jamestutorialr_viz.py` | ~27,000 | Same as message, slightly different implementation |
| Attempt2 (EMA momentum MM) | `attempt2_viz.py` | 5,374 | Momentum skew + inventory management, underperforms on this data |

---

## ETF / Basket Arbitrage (Round 2 insight)

- Baskets mean-revert toward their synthetic (constituent) value
- Trade baskets when spread vs synthetic crosses a fixed threshold
- Do NOT use z-scores unless volatility is known to vary — adds unnecessary complexity
- Do NOT blindly hedge with constituents — reduces expected value slightly while reducing variance
- Subtract a running estimated premium if the basket trades at a persistent offset from synthetic value
- Use flat parameter regions in grid search — not the peak

---

## Informed Trader Detection (Olivia pattern — Squid Ink / Croissants)

- One bot (Olivia) consistently buys 15 lots at the daily low, sells 15 lots at the daily high
- Detection: track running daily min/max; flag trades at daily extremes in the expected direction
- Use trader ID directly when available (Round 5); use pattern inference in earlier rounds
- False positives managed by monitoring for contradicting new extrema
- Cross-product signal: Olivia's position on an underlying (e.g. Croissants) can bias basket spread thresholds

---

## Options (Round 3 insight)

- Build a **volatility smile**: plot implied volatility (IV) vs moneyness across strikes, fit a parabola
- Detrend to isolate IV deviations from the smile → convert to price deviations via Black-Scholes
- Scalp options when price deviates from smile-implied theoretical value
- **Gamma scalping**: buy options, rehedge deltas from gamma exposure → small but consistent positive EV
- **Mean reversion on underlying**: track fast EMA, trade deviations at fixed thresholds (no vol scaling)
- Test 1-lag negative autocorrelation in returns to validate mean reversion before trading it

---

## Location Arbitrage / Hidden Taker Bots (Round 4 insight — Macarons / Orchids)

- A hidden taker bot fills offers priced near `int(externalBid + 0.5)` ~60% of the time
- Standard arbitrage only: buy locally when local ask < external bid (after fees); sell when local bid > external ask
- Hidden edge: place limit sells at `int(externalBid + 0.5)` to capture ~3 SeaShells above naive best bid
- Quote larger sizes (20–30 units) to profit from conversions even on non-fills

---

## Backtesting Approach

- Use local backtester (`prosperity4btx`) for rapid iteration and relative strategy comparison
- Use IMC platform backtester for final validation — different data, closer to real bot behavior
- Never optimize purely for peak historical P&L — choose parameter sets from flat, stable landscape regions
- Train-test splits are necessary but not sufficient; demand theoretical justification for every signal
- Avoid any feature that requires storing large historical windows (slows serialization)

---

## Competition Rules

- **Position limit**: 80 units per product (long or short). Orders exceeding this are rejected.
- **No fees** in the backtester.
- **Conversion limit**: varies by product (e.g. 10 units/tick for Macarons)
- `traderData` string persists across ticks — use JSON or jsonpickle to store state

---

## File Convention (this repo)

| Suffix | Purpose | Has Logger |
|---|---|---|
| `viz/{name}_viz.py` | Backtesting + visualizer | YES |
| `submit/{name}_submit.py` | Upload to IMC | NO |

Run backtest: `python run_backtest.py 0 --algo viz/{name}_viz.py`

---

## Key Code Patterns

### Reading the order book
```python
best_bid = max(order_depth.buy_orders.keys())
best_ask = min(order_depth.sell_orders.keys())
spread = best_ask - best_bid
mid_price = (best_bid + best_ask) / 2
```

### Position-safe order sizing
```python
pos = state.position.get(product, 0)
max_buy = POSITION_LIMIT - pos
max_sell = POSITION_LIMIT + pos
```

### Immediately taking favorable prices (before placing maker quotes)
```python
for ask_px in sorted(order_depth.sell_orders.keys()):
    if ask_px < fair_value and remaining_buy > 0:
        qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
        orders.append(Order(product, ask_px, qty))
        remaining_buy -= qty
```

### Persisting state across ticks
```python
# Load
trader_data = json.loads(state.traderData) if state.traderData else {}
# Save before flush
traderData = json.dumps(trader_data)
```

### EMA update
```python
def update_ema(prev: float, new_val: float, span: int) -> float:
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev
```
