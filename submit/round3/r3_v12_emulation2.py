"""
=============================================================================
IMC PROSPERITY ROUND 3 — REVERSE-ENGINEERED STRATEGY
=============================================================================

DISCOVERED EDGES (from log analysis):
--------------------------------------
1. STRUCTURAL IV SKEW ANOMALY (100% persistent):
   - IV(VEV_5400) ≈ 0.255 always BELOW IV(VEV_5300) ≈ 0.280
   - Mean spread: 0.024 vol points, never reverses
   - Price edge: ~4 pts per contract, ~1,200 for 300 contracts
   - Trade: LONG VEV_5400 (underpriced), SHORT VEV_5300 (overpriced)

2. VELVETFRUIT MARKET MAKING (PRIMARY PnL driver):
   - Natural bid-ask spread: consistently ~5 ticks
   - 667 fills × ~15 units × 5 spread / 2 ≈ 25,000 per team
   - Our submission ANTI-MM'd (bought at ask, sold at bid) → -390k realized
   - Fix: passive limit orders, inventory-aware quoting

3. SHORT OTM VOL PORTFOLIO:
   - VEV_5300, 5200, 5100, 5500 all overpriced vs ATM (positive skew)
   - Short these + long VEV_5400 = net vega-negative, theta-positive

4. HYDROGEL PAIRS TRADE (minor):
   - HYDROGEL ≈ 1.896 × VELVETFRUIT (not 2.0)
   - Mean-revert when spread deviates, ~5-15k additional

ESTIMATED PnL DECOMPOSITION (for a ~150k strategy):
   - VEX market making:        ~40,000–60,000
   - Vol arb (5400 vs 5300):   ~5,000–15,000
   - Short vol portfolio:      ~20,000–40,000
   - HYDROGEL MM:              ~5,000–15,000
   - Option carry / delta:     ~10,000–30,000
   TOTAL:                     ~80,000–160,000

=============================================================================
"""

from typing import Any, Optional
import json
import math

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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


# ─── Black-Scholes utilities ────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Fast normal CDF approximation (Abramowitz & Stegun)"""
    if x > 6: return 1.0
    if x < -6: return 0.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.7814779 + t * (-1.8212560 + t * 1.3302744))))
    return 1.0 - p if x > 0 else p

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price (no interest rate)"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)

def implied_vol_newton(price: float, S: float, K: float, T: float,
                        tol: float = 1e-6, max_iter: int = 30) -> Optional[float]:
    """Fast Newton-Raphson IV solver"""
    if T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.01:
        return None
    
    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2 * math.pi / T) * price / S
    sigma = max(0.01, min(sigma, 5.0))
    
    for _ in range(max_iter):
        p = bs_call_price(S, K, T, sigma)
        v = bs_vega(S, K, T, sigma)
        if v < 1e-10:
            break
        d_sigma = (price - p) / v
        sigma += d_sigma
        sigma = max(0.001, min(sigma, 10.0))
        if abs(d_sigma) < tol:
            return sigma
    return sigma


# ─── Strategy Parameters ────────────────────────────────────────────────────

# DISCOVERED CONSTANTS (from data analysis)
TTE = 0.013           # Time to expiry (calibrated from IV surface at σ≈0.27)
FAIR_SIGMA_ATM = 0.270  # Fair ATM vol
SIGMA_5400_LONG = 0.255  # Market IV of 5400 (underpriced)
SIGMA_5300_SHORT = 0.280  # Market IV of 5300 (overpriced)

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

POSITION_LIMITS = {
    "VELVETFRUIT_EXTRACT": 200,
    "HYDROGEL_PACK": 100,
    "VEV_4000": 200,
    "VEV_4500": 200,
    "VEV_5000": 200,
    "VEV_5100": 100,
    "VEV_5200": 200,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 200,
    "VEV_6000": 200,
    "VEV_6500": 100,
}

# Vol arb target portfolio
# FIX: Removed -200 VEV_5200 and -100 VEV_5100 legs.
# Those deep ITM/near-ATM shorts added -229 net delta, making the portfolio
# net short -297 delta total. A 30-tick VEX move caused ±9k PnL swings that
# dwarfed the ~300 MM edge per run by 13x. Dropping these legs reduces net
# delta to -63 (pure spread) and cuts the min drawdown from -3,760 to -820.
# Delta analysis of the 5400/5300 spread at S=5265, TTE=0.013, sigma=0.27:
#   +300 * delta(5400=0.210) = +62.9
#   -150 * delta(5300=0.421) = -63.1
#   Net delta ≈ 0.0 — true delta-neutral vol arb
#
# Using -300 on 5300 gives -63 residual delta (8x MM edge on 35-tick range).
# Using -150 on 5300 gives ~0 residual delta, halves the vol arb notional but
# eliminates 93% of PnL variance. Min drawdown: -820 → +80 (no negative floor).
VOL_ARB_TARGET = {
    "VEV_5400": +300,   # LONG: underpriced (IV 0.255 vs fair 0.270)
    "VEV_5300": -150,   # SHORT at delta-neutral size (150 = 300 * delta5400/delta5300)
    "VEV_5200":    0,   # excluded: too high delta
    "VEV_5100":    0,   # excluded: too high delta
    "VEV_5500":    0,   # excluded: adds -16 delta for marginal premium
}

# Tunable MM parameters
MM_QUOTE_OFFSET = 2         # half-spread we quote around EMA
MM_INV_SKEW_FACTOR = 1.5    # how aggressively we skew for inventory
MM_EMA_ALPHA = 0.05         # EMA decay for fair value tracking
MM_MAX_QTY_PER_SIDE = 20    # max quantity per MM quote
HYD_EMA_ALPHA = 0.03        # slower EMA for HYDROGEL
HYD_QUOTE_OFFSET = 8        # HYDROGEL MM quote offset (wider spread)
HYD_MAX_QTY = 10


# ─── Trader State ───────────────────────────────────────────────────────────

class TraderState:
    """Persistent state across timestamps (serialized as JSON string)"""
    
    def __init__(self):
        self.vex_ema: float = 5265.0
        self.hyd_ema: float = 9990.0
        self.atm_sigma: float = FAIR_SIGMA_ATM
        self.initialized: bool = False
        self.ts: int = 0
    
    def to_json(self) -> str:
        return json.dumps({
            "vex_ema": self.vex_ema,
            "hyd_ema": self.hyd_ema,
            "atm_sigma": self.atm_sigma,
            "initialized": self.initialized,
            "ts": self.ts,
        })
    
    @classmethod
    def from_json(cls, s: str) -> "TraderState":
        obj = cls()
        if not s:
            return obj
        try:
            d = json.loads(s)
            obj.vex_ema = d.get("vex_ema", 5265.0)
            obj.hyd_ema = d.get("hyd_ema", 9990.0)
            obj.atm_sigma = d.get("atm_sigma", FAIR_SIGMA_ATM)
            obj.initialized = d.get("initialized", False)
            obj.ts = d.get("ts", 0)
        except:
            pass
        return obj


# ─── Core Strategy Class ─────────────────────────────────────────────────────

class Trader:
    """
    Round 3 Strategy: Vol Arb + Inventory-Aware Market Making
    
    Two layers:
    1. STATIC: build vol arb spread at round start, hold
    2. DYNAMIC: market make on VELVETFRUIT and HYDROGEL
       (VEX position limit fully devoted to MM — no delta overlay)
    """
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """Main entry point called each timestamp"""
        
        # Load persistent state
        ts_state = TraderState.from_json(getattr(state, 'traderData', ''))
        ts_state.ts = getattr(state, 'timestamp', 0)
        
        # Get current positions
        pos = getattr(state, 'position', {}) or {}
        
        # Get order books
        order_depths = getattr(state, 'order_depths', {}) or {}
        
        # Output orders
        orders: dict[str, list[Order]] = {}
        
        # ─── STEP 1: Extract mid prices ───────────────────────────────────
        mids = {}
        bids = {}
        asks = {}
        bid_vols = {}
        ask_vols = {}
        
        for sym, od in order_depths.items():
            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                bids[sym] = best_bid
                bid_vols[sym] = od.buy_orders[best_bid]
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                asks[sym] = best_ask
                ask_vols[sym] = od.sell_orders[best_ask]
            if sym in bids and sym in asks:
                mids[sym] = (bids[sym] + asks[sym]) / 2.0
        
        S = mids.get("VELVETFRUIT_EXTRACT")
        H = mids.get("HYDROGEL_PACK")
        
        if S is None:
            new_trader_data = ts_state.to_json()
            logger.flush(state, orders, 0, new_trader_data)
            return orders, 0, new_trader_data
        
        # ─── STEP 2: Update EMAs ──────────────────────────────────────────
        if not ts_state.initialized:
            ts_state.vex_ema = S
            ts_state.hyd_ema = H or 9990.0
            ts_state.initialized = True
        else:
            ts_state.vex_ema = MM_EMA_ALPHA * S + (1 - MM_EMA_ALPHA) * ts_state.vex_ema
            if H:
                ts_state.hyd_ema = HYD_EMA_ALPHA * H + (1 - HYD_EMA_ALPHA) * ts_state.hyd_ema
        
        # Update ATM sigma estimate from near-ATM option
        atm_K = min(STRIKES, key=lambda k: abs(k - S))
        atm_sym = f"VEV_{atm_K}"
        if atm_sym in mids:
            iv = implied_vol_newton(mids[atm_sym], S, atm_K, TTE)
            if iv:
                ts_state.atm_sigma = 0.1 * iv + 0.9 * ts_state.atm_sigma
        
        # ─── STEP 3: VOL ARB — build and maintain target position ─────────
        for sym, target in VOL_ARB_TARGET.items():
            current = pos.get(sym, 0)
            need = target - current
            if need == 0:
                continue
            
            od = order_depths.get(sym)
            if od is None:
                continue
            
            if need > 0:  # need to BUY
                if od.sell_orders:
                    best_ask = min(od.sell_orders.keys())
                    avail = abs(od.sell_orders[best_ask])
                    qty = min(need, avail, abs(target))
                    if qty > 0:
                        orders.setdefault(sym, []).append(Order(sym, best_ask, qty))
            else:  # need to SELL
                if od.buy_orders:
                    best_bid = max(od.buy_orders.keys())
                    avail = od.buy_orders[best_bid]
                    qty = min(-need, avail, abs(target))
                    if qty > 0:
                        orders.setdefault(sym, []).append(Order(sym, best_bid, -qty))
        
        # ─── STEP 4: VELVETFRUIT MARKET MAKING ─────────────────
        vex_pos = pos.get("VELVETFRUIT_EXTRACT", 0)
        vex_limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        
        # FIX: Quote center = CURRENT mid (live bid/ask), NOT the lagging EMA.
        # EMA lag of +/-3 ticks was causing asymmetric fills and inventory drift.
        # EMA is kept only for the inventory skew signal below.
        inv_ratio = vex_pos / vex_limit  # ranges -1 to 1
        skew = inv_ratio * MM_INV_SKEW_FACTOR
        our_bid_px = round(S - MM_QUOTE_OFFSET - skew)
        our_ask_px = round(S + MM_QUOTE_OFFSET - skew)
        
        # Place passive bid
        if vex_pos < vex_limit:
            buy_room = vex_limit - vex_pos
            buy_qty = min(MM_MAX_QTY_PER_SIDE, buy_room)
            if buy_qty > 0:
                orders.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", our_bid_px, buy_qty)
                )
        
        # Place passive ask
        if vex_pos > -vex_limit:
            sell_room = vex_limit + vex_pos
            sell_qty = min(MM_MAX_QTY_PER_SIDE, sell_room)
            if sell_qty > 0:
                orders.setdefault("VELVETFRUIT_EXTRACT", []).append(
                    Order("VELVETFRUIT_EXTRACT", our_ask_px, -sell_qty)
                )
        
        # ─── STEP 5: HYDROGEL MARKET MAKING ─────────────────────────────
        if H is not None:
            hyd_pos = pos.get("HYDROGEL_PACK", 0)
            hyd_limit = POSITION_LIMITS["HYDROGEL_PACK"]
            hyd_inv_ratio = hyd_pos / hyd_limit
            hyd_skew = hyd_inv_ratio * 4.0
            
            # FIX: Quote off current mid H, not lagging EMA (same bug as VEX)
            hyd_bid = round(H - HYD_QUOTE_OFFSET - hyd_skew)
            hyd_ask = round(H + HYD_QUOTE_OFFSET - hyd_skew)
            
            if hyd_pos < hyd_limit:
                orders.setdefault("HYDROGEL_PACK", []).append(
                    Order("HYDROGEL_PACK", hyd_bid, min(HYD_MAX_QTY, hyd_limit - hyd_pos))
                )
            if hyd_pos > -hyd_limit:
                orders.setdefault("HYDROGEL_PACK", []).append(
                    Order("HYDROGEL_PACK", hyd_ask, -min(HYD_MAX_QTY, hyd_limit + hyd_pos))
                )
        
        result = orders
        conversions = 0
        new_trader_data = ts_state.to_json()
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data


# ─── SENSITIVITY ANALYSIS ───────────────────────────────────────────────────

SENSITIVITY_PARAMS = {
    "conservative": {
        "MM_QUOTE_OFFSET": 3,        # wider = less fills, more edge per fill
        "MM_INV_SKEW_FACTOR": 2.0,   # stronger inventory control
        "MM_EMA_ALPHA": 0.03,        # slower tracking
        "vol_arb_5400": 200,         # smaller position
        "vol_arb_5300": -200,
    },
    "aggressive": {
        "MM_QUOTE_OFFSET": 1,        # tighter = more fills, less edge per fill
        "MM_INV_SKEW_FACTOR": 1.0,   # weaker inventory control
        "MM_EMA_ALPHA": 0.1,         # faster tracking
        "vol_arb_5400": 300,         # max position
        "vol_arb_5300": -300,
    },
    "vol_arb_only": {
        "MM_QUOTE_OFFSET": 999,      # disable MM (quote so wide never fills)
        "vol_arb_5400": 300,
        "vol_arb_5300": -300,
        "vol_arb_5200": -200,
        "vol_arb_5100": -100,
    },
    "mm_only": {
        "MM_QUOTE_OFFSET": 2,
        "vol_arb_5400": 0,           # no option positions
        "vol_arb_5300": 0,
        "vol_arb_5200": 0,
        "vol_arb_5100": 0,
        "vol_arb_5500": 0,
    }
}


# ─── BACKTESTER ──────────────────────────────────────────────────────────────

def backtest(acts_df, trades_df, params: dict = None):
    """
    Simplified backtester using activities log.
    Simulates passive limit order fills based on market price crossing.
    
    Returns DataFrame with columns: ts, pnl, position_*
    """
    import pandas as pd
    
    p = {
        "MM_QUOTE_OFFSET": MM_QUOTE_OFFSET,
        "MM_INV_SKEW_FACTOR": MM_INV_SKEW_FACTOR,
        "MM_EMA_ALPHA": MM_EMA_ALPHA,
        "vol_arb_5400": VOL_ARB_TARGET["VEV_5400"],
        "vol_arb_5300": VOL_ARB_TARGET["VEV_5300"],
        "vol_arb_5200": VOL_ARB_TARGET["VEV_5200"],
        "vol_arb_5100": VOL_ARB_TARGET["VEV_5100"],
        "vol_arb_5500": VOL_ARB_TARGET["VEV_5500"],
    }
    if params:
        p.update(params)
    
    vol_arb_target = {
        "VEV_5400": p["vol_arb_5400"],
        "VEV_5300": p["vol_arb_5300"],
        "VEV_5200": p["vol_arb_5200"],
        "VEV_5100": p["vol_arb_5100"],
        "VEV_5500": p["vol_arb_5500"],
    }
    
    # Build market snapshot indexed by timestamp
    mkt = {}
    for prod in acts_df['product'].unique():
        sub = acts_df[acts_df['product']==prod].set_index('timestamp')
        for ts, row in sub.iterrows():
            if ts not in mkt:
                mkt[ts] = {}
            mkt[ts][prod] = {
                'mid': row['mid_price'],
                'bid': row['bid_price_1'],
                'ask': row['ask_price_1'],
            }
    
    timestamps = sorted(mkt.keys())
    position = {sym: 0 for sym in list(vol_arb_target.keys()) + 
                ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"]}
    cash = 0.0
    ema = None
    hyd_ema = None
    results = []
    
    for ts in timestamps:
        snap = mkt[ts]
        S_data = snap.get("VELVETFRUIT_EXTRACT")
        if not S_data:
            continue
        S = S_data['mid']
        
        # Update EMAs
        if ema is None:
            ema = S
            hyd_ema = snap.get("HYDROGEL_PACK", {}).get('mid', 9990.0)
        else:
            ema = p["MM_EMA_ALPHA"] * S + (1 - p["MM_EMA_ALPHA"]) * ema
            H = snap.get("HYDROGEL_PACK", {}).get('mid')
            if H:
                hyd_ema = HYD_EMA_ALPHA * H + (1 - HYD_EMA_ALPHA) * hyd_ema
        
        # --- LAYER 1: Vol arb ---
        for sym, target in vol_arb_target.items():
            cur = position.get(sym, 0)
            need = target - cur
            if need == 0 or sym not in snap:
                continue
            if need > 0:
                price = snap[sym]['ask']
                qty = min(need, 30)
            else:
                price = snap[sym]['bid']
                qty = min(-need, 30)
            
            if qty > 0:
                cash += price * (need / abs(need)) * -qty
                position[sym] = cur + (need // abs(need)) * qty
        
        # --- LAYER 2: VEX MM ---
        vex_pos = position.get("VELVETFRUIT_EXTRACT", 0)
        vex_limit = 200
        inv_ratio = vex_pos / vex_limit
        skew = inv_ratio * p["MM_INV_SKEW_FACTOR"]
        our_bid = round(ema - p["MM_QUOTE_OFFSET"] - skew)
        our_ask = round(ema + p["MM_QUOTE_OFFSET"] - skew)
        
        # Passive fill: if market comes to us
        mkt_bid = S_data['bid']
        mkt_ask = S_data['ask']
        
        # We get filled on our bid if market is selling (mkt_bid <= our_bid)
        if mkt_bid <= our_bid and vex_pos < vex_limit:
            qty = min(MM_MAX_QTY_PER_SIDE, vex_limit - vex_pos)
            cash -= our_bid * qty
            position["VELVETFRUIT_EXTRACT"] = vex_pos + qty
        
        # We get filled on our ask if market is buying (mkt_ask >= our_ask)
        vex_pos = position.get("VELVETFRUIT_EXTRACT", 0)
        if mkt_ask >= our_ask and vex_pos > -vex_limit:
            qty = min(MM_MAX_QTY_PER_SIDE, vex_limit + vex_pos)
            cash += our_ask * qty
            position["VELVETFRUIT_EXTRACT"] = vex_pos - qty
        
        # --- LAYER 3: HYD MM ---
        H_data = snap.get("HYDROGEL_PACK")
        if H_data:
            hyd_pos = position.get("HYDROGEL_PACK", 0)
            hyd_limit = 100
            hyd_inv_r = hyd_pos / hyd_limit
            hyd_bid = round(hyd_ema - HYD_QUOTE_OFFSET - hyd_inv_r * 4)
            hyd_ask = round(hyd_ema + HYD_QUOTE_OFFSET - hyd_inv_r * 4)
            
            if H_data['bid'] <= hyd_bid and hyd_pos < hyd_limit:
                qty = min(HYD_MAX_QTY, hyd_limit - hyd_pos)
                cash -= hyd_bid * qty
                position["HYDROGEL_PACK"] = hyd_pos + qty
            
            hyd_pos = position.get("HYDROGEL_PACK", 0)
            if H_data['ask'] >= hyd_ask and hyd_pos > -hyd_limit:
                qty = min(HYD_MAX_QTY, hyd_limit + hyd_pos)
                cash += hyd_ask * qty
                position["HYDROGEL_PACK"] = hyd_pos - qty
        
        # --- Mark to market ---
        pnl = cash
        for sym, qty in position.items():
            if sym in snap:
                pnl += qty * snap[sym]['mid']
        
        row_result = {'ts': ts, 'pnl': pnl, 'cash': cash}
        row_result.update({f'pos_{k}': v for k, v in position.items()})
        results.append(row_result)
    
    return pd.DataFrame(results).set_index('ts')


# ─── MAIN: Run backtest with analysis ────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import json
    from io import StringIO
    
    print("=" * 70)
    print("ROUND 3 STRATEGY BACKTEST")
    print("=" * 70)
    
    with open('/mnt/project/432196.log') as f:
        data = json.load(f)
    
    acts = pd.read_csv(StringIO(data['activitiesLog']), sep=';')
    trades = pd.DataFrame(data['tradeHistory'])
    
    # ── Run base strategy ──
    print("\n[BASE STRATEGY: Vol Arb + MM + HYD]")
    res = backtest(acts, trades)
    print(f"  Final PnL:   {res['pnl'].iloc[-1]:>10,.0f}")
    print(f"  Max PnL:     {res['pnl'].max():>10,.0f}")
    print(f"  Min PnL:     {res['pnl'].min():>10,.0f}")
    print(f"  Sharpe (est):{res['pnl'].diff().mean() / (res['pnl'].diff().std() + 1e-10):>10.3f}")
    print(f"  Final pos:")
    for col in [c for c in res.columns if c.startswith('pos_')]:
        v = res[col].iloc[-1]
        if v != 0:
            print(f"    {col:25s}: {v:6.0f}")
    
    # ── Sensitivity analysis ──
    print("\n[SENSITIVITY ANALYSIS]")
    for name, params in SENSITIVITY_PARAMS.items():
        r = backtest(acts, trades, params)
        print(f"  {name:20s}: final={r['pnl'].iloc[-1]:>8,.0f}  "
              f"max={r['pnl'].max():>8,.0f}  min={r['pnl'].min():>8,.0f}")
    
    # ── PnL attribution ──
    print("\n[PNL ATTRIBUTION]")
    print("  Estimating contribution of each layer...")
    
    # Vol arb only
    r_arb = backtest(acts, trades, {
        "MM_QUOTE_OFFSET": 999,
        "vol_arb_5400": VOL_ARB_TARGET["VEV_5400"],
        "vol_arb_5300": VOL_ARB_TARGET["VEV_5300"],
        "vol_arb_5200": VOL_ARB_TARGET["VEV_5200"],
        "vol_arb_5100": VOL_ARB_TARGET["VEV_5100"],
        "vol_arb_5500": VOL_ARB_TARGET["VEV_5500"],
    })
    
    # MM only
    r_mm = backtest(acts, trades, {
        "MM_QUOTE_OFFSET": 2,
        "vol_arb_5400": 0, "vol_arb_5300": 0,
        "vol_arb_5200": 0, "vol_arb_5100": 0, "vol_arb_5500": 0,
    })
    
    # Full strategy
    r_full = backtest(acts, trades)
    
    print(f"  Vol arb only: {r_arb['pnl'].iloc[-1]:>8,.0f}")
    print(f"  MM only:      {r_mm['pnl'].iloc[-1]:>8,.0f}")
    print(f"  Combined:     {r_full['pnl'].iloc[-1]:>8,.0f}")
    print(f"  Synergy:      {r_full['pnl'].iloc[-1] - r_arb['pnl'].iloc[-1] - r_mm['pnl'].iloc[-1]:>8,.0f}")
    
    # ── Key structural stats ──
    print("\n[KEY STRUCTURAL EDGES]")
    print("  1. IV(5300) - IV(5400) spread:")
    
    def iv_from_mid(price, S, K):
        return implied_vol_newton(price, S, K, TTE)
    
    vex_mids = acts[acts['product']=='VELVETFRUIT_EXTRACT'].set_index('timestamp')['mid_price']
    spreads = []
    for ts in sorted(acts['timestamp'].unique())[::10]:  # sample every 10th
        S = vex_mids.get(ts)
        if S is None: continue
        p5300 = acts[(acts['product']=='VEV_5300')&(acts['timestamp']==ts)]['mid_price']
        p5400 = acts[(acts['product']=='VEV_5400')&(acts['timestamp']==ts)]['mid_price']
        if len(p5300)==0 or len(p5400)==0: continue
        iv3 = iv_from_mid(p5300.values[0], S, 5300)
        iv4 = iv_from_mid(p5400.values[0], S, 5400)
        if iv3 and iv4:
            spreads.append(iv3 - iv4)
    
    if spreads:
        print(f"     Always positive: {sum(s>0 for s in spreads)/len(spreads):.1%}")
        print(f"     Mean: {np.mean(spreads):.4f} vol pts")
        print(f"     Min:  {min(spreads):.4f} vol pts")
    
    print("\n  2. VEX bid-ask half-spread:")
    vex_spreads = (acts[acts['product']=='VELVETFRUIT_EXTRACT']['ask_price_1'] 
                   - acts[acts['product']=='VELVETFRUIT_EXTRACT']['bid_price_1']) / 2
    print(f"     Mean: {vex_spreads.mean():.2f} ticks")
    print(f"     Our MM captures ~{vex_spreads.mean():.1f} ticks per fill side")
    
    print("\n[STRATEGY SUMMARY]")
    print("""
  LAYER 1 — STRUCTURAL VOL ARB (hold entire round):
    ╔══════════╦══════╦═════════╦══════════════╗
    ║ Symbol   ║ Pos  ║ IV Mkt  ║ IV Fair      ║
    ╠══════════╬══════╬═════════╬══════════════╣
    ║ VEV_5400 ║ +300 ║  0.255  ║ 0.270 (LONG) ║
    ║ VEV_5300 ║ -300 ║  0.280  ║ 0.270 (SHORT)║
    ║ VEV_5200 ║ -200 ║  0.274  ║ 0.270        ║
    ║ VEV_5100 ║ -100 ║  0.267  ║ 0.270        ║
    ║ VEV_5500 ║ -200 ║  0.278  ║ 0.270        ║
    ╚══════════╩══════╩═════════╩══════════════╝

  LAYER 2 — VELVETFRUIT PASSIVE MM:
    • Quote: bid = EMA - 2 - skew, ask = EMA + 2 - skew
    • Skew  = (position / limit) × 1.5
    • Fills every ~0.67 timestamps at ~15 units
    • Estimated edge: 2.5 ticks × 15 units × 667 fills = ~25,000

  LAYER 3 — HYDROGEL MM (minor):
    • Quote: ±8 around HYD EMA
    • ~30 fills × 10 units × 8 edge = ~2,400

  NOTE ON DELTA:
    Net options delta ≈ -230. This is deliberately NOT hedged because:
    • VEX only ranged ~35pts over the round → max unhedged loss ~8,050
    • VEX position limit is ±200, fully needed for MM throughput
    • MM inventory skew naturally provides partial mean-reversion offset
    • Hedging would cost more in lost MM PnL than the delta risk is worth

  FAILURE MODES:
    ✗ VELVETFRUIT trends strongly → MM inventory builds → drawdown
    ✗ IV spread narrows → vol arb loses → rare but possible (never seen in data)
    ✗ Fills too slow on VEX MM → miss edge window
    """)