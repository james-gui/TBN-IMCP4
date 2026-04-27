"""
IMC Prosperity 4 – Round 3 | r3_v11 | "Vol Selling + Delta Hedge"
═══════════════════════════════════════════════════════════════════

BUGS FIXED FROM v8 (log 432196):
  1. VEV MM removed — was crossing spread every tick, losing -23k
  2. VEV_5400 direction fixed — was LONG (buying premium), now SHORT
  3. All option positions maxed to 300 limit immediately

STRATEGY:
  - SHORT 300 of every OTM voucher (5000-5500) at market bid
  - Hold until liquidation at "hidden fair value" (intrinsic or low-vol BS)
  - Delta hedge net short delta with VEV underlying (limit 200)
  - Light HYDROGEL MM for extra ~700 XIRECS
  - No VEV market-making (spread = 5, always lose crossing it)

EXPECTED PnL: +30k-35k per round (intrinsic liq), ~100k across 3 rounds
"""

import json
import math
from typing import Any, Dict, List
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

VEV = "VELVETFRUIT_EXTRACT"
HYDROGEL = "HYDROGEL_PACK"

VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_NAMES = {k: f"VEV_{k}" for k in VOUCHER_STRIKES}

# Position limits
LIMITS = {
    VEV: 200,
    HYDROGEL: 200,
}
for k in VOUCHER_STRIKES:
    LIMITS[VOUCHER_NAMES[k]] = 300

# Strikes to SHORT (all OTM + near-ATM)
SHORT_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# TTE config: Round 3 starts at TTE=5 days
# Each round is 100,000 timestamps = 1 day
TIMESTAMPS_PER_DAY = 100_000
DAYS_PER_YEAR = 365.0

# Vol parameters
MARKET_IV = 0.27       # What bots price at
REALIZED_VOL = 0.05    # Actual VEV vol
HEDGE_VOL = 0.10       # Conservative vol for delta calc

# Delta hedge parameters
DELTA_BAND = 15        # Re-hedge when net delta exceeds ±15
EMA_ALPHA = 0.02       # Slow EMA for VEV fair value

# HYDROGEL MM parameters  
HYDROGEL_MAKE_WIDTH = 5
HYDROGEL_ORDER_SIZE = 10
HYDROGEL_INV_LIMIT = 50


# ═══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / 1.4142135623730951)) / 2.0


def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call delta."""
    if T <= 1e-9:
        return 1.0 if S > K else (0.5 if abs(S - K) < 0.01 else 0.0)
    if S <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return _norm_cdf(d1)


def _bs_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 1e-9 or S <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


# ═══════════════════════════════════════════════════════════════════════
# ORDER HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _fill_to_target(symbol: str, od: OrderDepth, pos: int,
                    target: int, limit: int) -> List[Order]:
    """
    Aggressively fill orders to reach target position.
    Takes liquidity from the book.
    """
    orders: List[Order] = []
    need = target - pos

    if need > 0 and od.sell_orders:
        # Need to BUY — hit asks
        for px in sorted(od.sell_orders):
            if need <= 0:
                break
            avail = min(-od.sell_orders[px], need, limit - pos)
            if avail <= 0:
                break
            orders.append(Order(symbol, px, avail))
            pos += avail
            need -= avail

    elif need < 0 and od.buy_orders:
        # Need to SELL — hit bids
        for px in sorted(od.buy_orders, reverse=True):
            if need >= 0:
                break
            avail = min(od.buy_orders[px], -need, limit + pos)
            if avail <= 0:
                break
            orders.append(Order(symbol, px, -avail))
            pos -= avail
            need += avail

    return orders


def _get_mid(od: OrderDepth) -> float | None:
    """Get mid price from order depth."""
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


# ═══════════════════════════════════════════════════════════════════════
# LOGGER (minimal, keeps traderData small)
# ═══════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *args: Any) -> None:
        self.logs += " ".join(map(str, args)) + "\n"

    def flush(self, state, result, conversions, trader_data):
        print(json.dumps({
            "state": "",
            "orders": {s: [{"symbol": o.symbol, "price": o.price, "quantity": o.quantity}
                           for o in orders] for s, orders in result.items()},
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":")))
        self.logs = ""


logger = Logger()


# ═══════════════════════════════════════════════════════════════════════
# TRADER
# ═══════════════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # ── Deserialize ──────────────────────────────────────────────
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        vev_ema = ts.get("vev_ema", None)
        hydrogel_ema = ts.get("hydrogel_ema", None)
        round_start_tte = ts.get("round_start_tte", 5.0)  # days

        t = state.timestamp

        # ── TTE calculation ──────────────────────────────────────────
        tte_days = round_start_tte - (t / TIMESTAMPS_PER_DAY)
        T = max(tte_days / DAYS_PER_YEAR, 1e-6)

        # ── Get current prices ───────────────────────────────────────
        vev_mid = None
        hydrogel_mid = None

        if VEV in state.order_depths:
            vev_mid = _get_mid(state.order_depths[VEV])
        if HYDROGEL in state.order_depths:
            hydrogel_mid = _get_mid(state.order_depths[HYDROGEL])

        # Update EMAs
        if vev_mid is not None:
            if vev_ema is None:
                vev_ema = vev_mid
            else:
                vev_ema = EMA_ALPHA * vev_mid + (1 - EMA_ALPHA) * vev_ema

        if hydrogel_mid is not None:
            if hydrogel_ema is None:
                hydrogel_ema = hydrogel_mid
            else:
                hydrogel_ema = EMA_ALPHA * hydrogel_mid + (1 - EMA_ALPHA) * hydrogel_ema

        S = vev_mid if vev_mid is not None else (vev_ema if vev_ema is not None else 5260.0)

        # ═════════════════════════════════════════════════════════════
        # LAYER 1: OPTION VOL SELLING (THE MAIN ENGINE)
        # ═════════════════════════════════════════════════════════════
        #
        # SHORT every OTM voucher to position limit (-300).
        # Market prices at 27% IV, realized is ~5%.
        # Liquidation at intrinsic = pure profit on all time value.
        #
        # Do this IMMEDIATELY at t=0 and every tick until filled.

        for K in SHORT_STRIKES:
            name = VOUCHER_NAMES[K]
            if name not in state.order_depths:
                continue

            od = state.order_depths[name]
            pos = state.position.get(name, 0)
            limit = LIMITS[name]

            # Target: SHORT to -limit (-300)
            target = -limit

            if pos > target:
                orders = _fill_to_target(name, od, pos, target, limit)
                if orders:
                    result[name] = orders
                    logger.print(f"VOL_SELL {name}: pos={pos} -> target={target}, {len(orders)} orders")

        # ═════════════════════════════════════════════════════════════
        # LAYER 2: DELTA HEDGE WITH VEV
        # ═════════════════════════════════════════════════════════════
        #
        # Compute net delta from all option positions.
        # Offset with VEV underlying to stay delta-neutral.
        # Only trade VEV for hedging, NOT for market-making.

        if VEV in state.order_depths and vev_mid is not None:
            net_option_delta = 0.0

            for K in VOUCHER_STRIKES:
                name = VOUCHER_NAMES[K]
                pos_v = state.position.get(name, 0)
                if pos_v == 0:
                    continue
                d = _bs_delta(S, K, T, HEDGE_VOL)
                net_option_delta += d * pos_v  # short pos = negative delta contrib

            # Current VEV position
            vev_pos = state.position.get(VEV, 0)

            # Total portfolio delta = option delta + VEV position
            total_delta = net_option_delta + vev_pos

            # Target VEV position to neutralize delta
            vev_target = -round(net_option_delta)
            vev_target = max(-LIMITS[VEV], min(LIMITS[VEV], vev_target))

            # Only hedge if delta exceeds band
            if abs(total_delta) > DELTA_BAND:
                od_vev = state.order_depths[VEV]
                hedge_orders = _fill_to_target(VEV, od_vev, vev_pos, vev_target, LIMITS[VEV])
                if hedge_orders:
                    result[VEV] = hedge_orders
                    logger.print(f"DELTA_HEDGE: opt_delta={net_option_delta:.1f} vev_pos={vev_pos} "
                                 f"total={total_delta:.1f} -> target={vev_target}")

        # ═════════════════════════════════════════════════════════════
        # LAYER 3: HYDROGEL MARKET MAKING (BONUS ~700 XIRECS)
        # ═════════════════════════════════════════════════════════════

        if HYDROGEL in state.order_depths and hydrogel_ema is not None:
            od_h = state.order_depths[HYDROGEL]
            pos_h = state.position.get(HYDROGEL, 0)
            limit_h = LIMITS[HYDROGEL]

            if od_h.buy_orders and od_h.sell_orders:
                bb = max(od_h.buy_orders)
                ba = min(od_h.sell_orders)
                fv = hydrogel_ema

                h_orders: List[Order] = []

                # Take mispriced orders
                for px in sorted(od_h.sell_orders):
                    if px < fv - 1 and pos_h < limit_h:
                        q = min(-od_h.sell_orders[px], limit_h - pos_h, HYDROGEL_ORDER_SIZE)
                        if q > 0:
                            h_orders.append(Order(HYDROGEL, px, q))
                            pos_h += q

                for px in sorted(od_h.buy_orders, reverse=True):
                    if px > fv + 1 and pos_h > -limit_h:
                        q = min(od_h.buy_orders[px], limit_h + pos_h, HYDROGEL_ORDER_SIZE)
                        if q > 0:
                            h_orders.append(Order(HYDROGEL, px, -q))
                            pos_h -= q

                # Post passive quotes
                skew = -round((pos_h / max(limit_h, 1)) * 3)
                our_bid = round(fv) + skew - HYDROGEL_MAKE_WIDTH
                our_ask = round(fv) + skew + HYDROGEL_MAKE_WIDTH

                # Stay inside the book
                our_bid = min(our_bid, ba - 1)
                our_ask = max(our_ask, bb + 1)

                remaining_buy = limit_h - pos_h
                remaining_sell = limit_h + pos_h

                if our_bid < our_ask:
                    bid_size = HYDROGEL_ORDER_SIZE
                    ask_size = HYDROGEL_ORDER_SIZE

                    # Reduce size when inventory is high
                    if pos_h > HYDROGEL_INV_LIMIT:
                        bid_size = max(1, bid_size // 3)
                    elif pos_h < -HYDROGEL_INV_LIMIT:
                        ask_size = max(1, ask_size // 3)

                    if remaining_buy > 0 and our_bid < fv:
                        h_orders.append(Order(HYDROGEL, our_bid, min(bid_size, remaining_buy)))
                    if remaining_sell > 0 and our_ask > fv:
                        h_orders.append(Order(HYDROGEL, our_ask, -min(ask_size, remaining_sell)))

                if h_orders:
                    result[HYDROGEL] = h_orders

        # ── Serialize state ──────────────────────────────────────────
        new_ts = json.dumps({
            "vev_ema": vev_ema,
            "hydrogel_ema": hydrogel_ema,
            "round_start_tte": round_start_tte,
        })

        logger.flush(state, result, conversions, new_ts)
        return result, conversions, new_ts