"""r4_v12_nocap — data-driven fixes from v9 per-product breakdown.

v9 analysis:
  - VEX: -812 PnL with pos=+101. Long VEX in bear market → disable VEX trading.
  - VEV 5100-5500: ALL hitting soft_cap=150. Leaving money on table → raise to 250.
  - VEV 4000-5000: only -11 pos (deep ITM, ask_ok rarely fires) → cap stays at 50.
"""

import json
import math
from typing import Any, Dict, List

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
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
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

HYDROGEL = "HYDROGEL_PACK"
VEX = "VELVETFRUIT_EXTRACT"
VEV_ALL = ["VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
           "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"]

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}

POSITION_LIMITS = {HYDROGEL: 200, VEX: 200}
for sym in VEV_ALL:
    POSITION_LIMITS[sym] = 300

# HYDROGEL: unchanged from v14
HYDROGEL_CFG = {"ema_alpha": 0.05, "make_size": 25, "soft_inv_cap": 150, "skew_div": 50.0,
                "take_edge": 7.0, "take_size": 15, "imb_take_threshold": 0.2, "imb_passive_threshold": 0.3}
# VEX: no-take passive market-making, tight cap=30, fast EMA to track price
VEX_CFG_TIGHT = {"ema_alpha": 0.15, "make_size": 5, "soft_inv_cap": 30, "skew_div": 15.0}
VEX_CFG       = {"ema_alpha": 0.05, "make_size": 20, "soft_inv_cap": 150, "skew_div": 60.0,
                 "take_edge": 7.0, "take_size": 25, "imb_take_threshold": 0.2, "imb_passive_threshold": 0.3}

# BS parameters — near-zero sigma → fair ≈ intrinsic → always lean SHORT options
BS_SIGMA      = 0.001   # near-zero: fair = intrinsic(S-K,0). Market has time value → sell it.
BS_T_TOTAL    = 9 / 252
TICKS_PER_DAY = 10000

VEV_MAKE_SIZE  = 20    # raised from 10 — more passive size per tick
VEV_TAKE_SIZE  = 20    # raised from 15
VEV_TAKE_EDGE  = 3.0
VEV_SOFT_CAP   = 300    # raised from 150 — 5100-5500 all hit the cap in v9
VEV_SKEW_DIV   = 50.0

# Deep ITM — raised caps (market has some buyers there)
VEV_ITM_CAP = {"VEV_4000": 150, "VEV_4500": 150, "VEV_5000": 150}

VEV_SKIP = {"VEV_6000", "VEV_6500"}

MAX_TICK_DELTA = 50

VEX_EMA_FAST_ALPHA = 0.10
VEX_EMA_SLOW_ALPHA = 0.02



# ── Black-Scholes + Monte Carlo ───────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Abramowitz & Stegun inverse normal CDF approximation."""
    if p <= 0.0: return -8.0
    if p >= 1.0: return  8.0
    q = p if p < 0.5 else 1.0 - p
    t = math.sqrt(-2.0 * math.log(q))
    z = t - (2.515517 + 0.802853*t + 0.010328*t*t) / \
            (1.0 + 1.432788*t + 0.189269*t*t + 0.001308*t*t*t)
    return -z if p < 0.5 else z


def mc_call_fair(S: float, K: float, T: float, sigma: float,
                 drift: float = 0.0, n: int = 200) -> float:
    """MC European call under GBM with real-world drift + antithetic variates.
    drift < 0 (bear) → OTM calls worth less → sell even more aggressively.
    """
    if T <= 1e-9:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    adj = (drift - 0.5 * sigma * sigma) * T
    total = 0.0
    for i in range(1, n + 1):
        # Halton sequence base 2 → uniform [0,1]
        f, r, j = 0.5, 0.0, i
        while j > 0:
            r += f * (j & 1); j >>= 1; f *= 0.5
        u = max(1e-9, min(1 - 1e-9, r))
        Z = _norm_ppf(u)
        S1 = S * math.exp(adj + sigma * sqrt_T * Z)
        S2 = S * math.exp(adj - sigma * sqrt_T * Z)  # antithetic
        total += max(S1 - K, 0.0) + max(S2 - K, 0.0)
    return total / (2 * n)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    intrinsic = max(S - K, 0.0)
    if T <= 1e-9 or sigma <= 0:
        return intrinsic
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def implied_vol(S: float, K: float, T: float, market_price: float,
                lo: float = 0.001, hi: float = 5.0) -> float | None:
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + 0.5:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = bs_call(S, K, T, mid)
        if p < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def volume_weighted_mid(order_depth: OrderDepth) -> float:
    best_bid = max(order_depth.buy_orders)
    best_ask = min(order_depth.sell_orders)
    bid_vol = order_depth.buy_orders[best_bid]
    ask_vol = abs(order_depth.sell_orders[best_ask])
    return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)


def get_regime(vex_order_depth: OrderDepth, ema_state: dict,
               vex_market_trades: list) -> str:
    vwm = volume_weighted_mid(vex_order_depth)
    fast = ema_state.get("vex_fast", vwm)
    slow = ema_state.get("vex_slow", vwm)
    fast = VEX_EMA_FAST_ALPHA * vwm + (1 - VEX_EMA_FAST_ALPHA) * fast
    slow = VEX_EMA_SLOW_ALPHA * vwm + (1 - VEX_EMA_SLOW_ALPHA) * slow
    ema_state["vex_fast"] = fast
    ema_state["vex_slow"] = slow
    return "bull" if fast > slow else "bear"


def passive_inside_wall(sym, order_depth, pos, ema_state, cfg, pos_limit, allow_take=False, suppress_bid=False):
    orders: List[Order] = []
    best_bid = max(order_depth.buy_orders)
    best_ask = min(order_depth.sell_orders)

    vwm = volume_weighted_mid(order_depth)
    alpha = cfg["ema_alpha"]
    if sym not in ema_state:
        ema_state[sym] = vwm
    ema_state[sym] = alpha * vwm + (1 - alpha) * ema_state[sym]
    fair_unskewed = ema_state[sym]
    fair = fair_unskewed - pos / cfg["skew_div"]

    max_buy = pos_limit - pos
    max_sell = pos_limit + pos
    cur_pos = pos
    soft_cap = cfg["soft_inv_cap"]

    if allow_take:
        take_edge = cfg.get("take_edge", float("inf"))
        take_size = int(cfg.get("take_size", 0))
        imb_threshold = cfg.get("imb_take_threshold", None)
        if imb_threshold is not None:
            bv1 = order_depth.buy_orders[best_bid]
            av1 = abs(order_depth.sell_orders[best_ask])
            imb = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0
        else:
            imb = 0.0

        if (imb_threshold is None or imb >= -imb_threshold) and \
           best_ask <= fair_unskewed - take_edge and max_buy > 0 and cur_pos < soft_cap:
            vol = min(take_size, abs(order_depth.sell_orders[best_ask]), max_buy, soft_cap - cur_pos)
            if vol > 0:
                orders.append(Order(sym, int(best_ask), vol))
                max_buy -= vol; cur_pos += vol

        if (imb_threshold is None or imb <= imb_threshold) and \
           best_bid >= fair_unskewed + take_edge and max_sell > 0 and cur_pos > -soft_cap:
            vol = min(take_size, abs(order_depth.buy_orders[best_bid]), max_sell, soft_cap + cur_pos)
            if vol > 0:
                orders.append(Order(sym, int(best_bid), -vol))
                max_sell -= vol; cur_pos -= vol

    if best_ask - best_bid < 2:
        return orders

    our_bid = best_bid + 1
    our_ask = best_ask - 1
    if our_bid >= our_ask:
        return orders

    bid_ok = (not suppress_bid) and our_bid <= fair - 0.5
    ask_ok = our_ask >= fair + 0.5

    imb_passive_threshold = cfg.get("imb_passive_threshold", None)
    if imb_passive_threshold is not None:
        bv1_p = order_depth.buy_orders[best_bid]
        av1_p = abs(order_depth.sell_orders[best_ask])
        imb_p = (bv1_p - av1_p) / (bv1_p + av1_p) if (bv1_p + av1_p) > 0 else 0.0
        if imb_p < -imb_passive_threshold:
            bid_ok = False
        if imb_p > imb_passive_threshold:
            ask_ok = False

    make_size = int(cfg["make_size"])
    if bid_ok and max_buy > 0 and cur_pos < soft_cap:
        size = min(make_size, max_buy)
        if size > 0:
            orders.append(Order(sym, int(our_bid), size))
    if ask_ok and max_sell > 0 and cur_pos > -soft_cap:
        size = min(make_size, max_sell)
        if size > 0:
            orders.append(Order(sym, int(our_ask), -size))

    return orders


def bs_market_make(sym: str, order_depth: OrderDepth, pos: int,
                   fair_bs: float, pos_limit: int,
                   soft_cap: int = VEV_SOFT_CAP) -> List[Order]:
    """SELL-ONLY option writer. fair_bs ≈ intrinsic (σ≈0) → market > fair → always ask."""
    orders: List[Order] = []
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders

    best_bid = max(order_depth.buy_orders)
    best_ask = min(order_depth.sell_orders)

    if best_ask - best_bid < 2:
        return orders

    our_ask = best_ask - 1
    max_sell = pos_limit + pos

    # Skew: as we go short, fair_bs rises (skew pushes fair up), limiting further selling
    fair = fair_bs - pos / VEV_SKEW_DIV  # pos<0 → fair rises → ask_ok harder to satisfy

    # Sell take: market bid well above our intrinsic fair → sell into it
    if best_bid >= fair_bs + VEV_TAKE_EDGE and max_sell > 0 and pos > -soft_cap:
        vol = min(VEV_TAKE_SIZE, abs(order_depth.buy_orders[best_bid]),
                  max_sell, soft_cap + pos)
        if vol > 0:
            orders.append(Order(sym, int(best_bid), -vol))
            max_sell -= vol

    # Passive ask only — NO bids ever
    ask_ok = our_ask >= fair + 0.5 and pos > -soft_cap and max_sell > 0

    if ask_ok:
        orders.append(Order(sym, int(our_ask), -min(VEV_MAKE_SIZE, max_sell)))

    return orders


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}
        ema_state: Dict[str, float] = trader_state.get("ema", {})

        # ── Track day number for TTE ─────────────────────────────────────────
        prev_ts = trader_state.get("prev_ts", state.timestamp)
        day = trader_state.get("day", 1)
        if state.timestamp < prev_ts:
            day += 1  # new day detected
        trader_state["prev_ts"] = state.timestamp
        trader_state["day"] = day

        # Time to expiry (fraction of BS_T_TOTAL remaining)
        elapsed = (day - 1) * TICKS_PER_DAY + state.timestamp // 100
        total_ticks = 3 * TICKS_PER_DAY
        tte = max(BS_T_TOTAL * (1.0 - elapsed / total_ticks), 1e-6)

        # ── VEX mid (underlying for BS) ──────────────────────────────────────
        vex_mid = None
        if VEX in state.order_depths:
            vex_od = state.order_depths[VEX]
            if vex_od.buy_orders and vex_od.sell_orders:
                vex_mid = volume_weighted_mid(vex_od)

        # ── Regime (for legacy products) ────────────────────────────────────
        regime = "bull"
        if vex_mid is not None:
            vex_mkt_trades = state.market_trades.get(VEX, [])
            regime = get_regime(state.order_depths[VEX], ema_state, vex_mkt_trades)

        # ── Compute BS fair for all VEV strikes ──────────────────────────────
        bs_fairs: Dict[str, float] = {}
        if vex_mid is not None:
            for sym, K in VEV_STRIKES.items():
                bs_fairs[sym] = bs_call(vex_mid, K, tte, BS_SIGMA)

        # ── Cross-strike IV calibration (optional signal) ───────────────────
        ivols = []
        for sym, K in VEV_STRIKES.items():
            if sym in VEV_SKIP or sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders:
                continue
            mkt_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
            if vex_mid and tte > 1e-6:
                iv = implied_vol(vex_mid, K, tte, mkt_mid)
                if iv and 0.05 < iv < 2.0:
                    ivols.append(iv)

        # MC fair: use realized vol (σ≈0.05) + regime drift for each strike
        # Bear drift → OTM calls worth less → always sell. Bull drift → worth slightly more.
        sigma = BS_SIGMA
        MC_SIGMA = 0.05   # realized vol estimate
        # Drift: bear = -5% annualized, bull = +5%, neutral = 0
        mc_drift = -0.05 if regime == "bear" else (0.05 if regime == "bull" else 0.0)
        mc_fairs: dict = {}
        if vex_mid is not None and tte > 1e-6:
            for sym, K in VEV_STRIKES.items():
                if sym not in VEV_SKIP:
                    mc_fairs[sym] = mc_call_fair(vex_mid, K, tte, MC_SIGMA, mc_drift, n=100)

        # ── Main trading loop ─────────────────────────────────────────────────
        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue
            pos = state.position.get(product, 0)

            if product == HYDROGEL:
                result[product] = passive_inside_wall(
                    product, order_depth, pos, ema_state, HYDROGEL_CFG,
                    POSITION_LIMITS[HYDROGEL], allow_take=True)

            elif product == VEX:
                # Re-enabled: no takes, tight cap=30, bear → suppress bids to avoid long accumulation
                suppress_bid = (regime == "bear")
                result[product] = passive_inside_wall(
                    product, order_depth, pos, ema_state,
                    VEX_CFG_TIGHT, POSITION_LIMITS[VEX],
                    allow_take=False, suppress_bid=suppress_bid)

            elif product in VEV_ALL:
                if product in VEV_SKIP:
                    result[product] = []
                    continue
                # Use MC fair (with drift) instead of BS(σ≈0) fair
                fair_val = mc_fairs.get(product, bs_fairs.get(product, None))
                if fair_val is not None:
                    soft_cap = VEV_ITM_CAP.get(product, VEV_SOFT_CAP)
                    result[product] = bs_market_make(
                        product, order_depth, pos,
                        fair_val, POSITION_LIMITS[product],
                        soft_cap=soft_cap)
                else:
                    result[product] = []

            else:
                result[product] = []

        # ── Delta hedge: buy VEX to offset short call delta ──────────────────
        # Aggregate delta of all short VEV positions (using MC_SIGMA for realism)
        if vex_mid is not None and tte > 1e-6 and VEX in state.order_depths:
            vex_od = state.order_depths[VEX]
            if vex_od.buy_orders and vex_od.sell_orders:
                total_delta = 0.0
                for sym, K in VEV_STRIKES.items():
                    if sym in VEV_SKIP:
                        continue
                    vev_pos = state.position.get(sym, 0)
                    if vev_pos < 0:  # only hedge short positions
                        d = bs_delta(vex_mid, K, tte, MC_SIGMA)
                        total_delta += d * vev_pos  # negative * positive delta = net negative

                # target_vex_pos: go long VEX to offset negative delta
                # scale by 0.5 — partial hedge to avoid overloading VEX position limit
                target_vex_pos = int(-total_delta * 0.5)
                target_vex_pos = max(-POSITION_LIMITS[VEX], min(POSITION_LIMITS[VEX], target_vex_pos))

                vex_pos = state.position.get(VEX, 0)
                hedge_qty = target_vex_pos - vex_pos

                if hedge_qty != 0:
                    vex_best_ask = min(vex_od.sell_orders)
                    vex_best_bid = max(vex_od.buy_orders)
                    hedge_orders = result.get(VEX, [])
                    if hedge_qty > 0:  # need to buy VEX
                        hedge_qty = min(hedge_qty, POSITION_LIMITS[VEX] - vex_pos)
                        if hedge_qty > 0:
                            hedge_orders.append(Order(VEX, vex_best_ask, hedge_qty))
                    else:  # need to sell VEX
                        hedge_qty = max(hedge_qty, -(POSITION_LIMITS[VEX] + vex_pos))
                        if hedge_qty < 0:
                            hedge_orders.append(Order(VEX, vex_best_bid, hedge_qty))
                    result[VEX] = hedge_orders

        # ── Per-tick exposure cap ─────────────────────────────────────────────
        for product, orders in result.items():
            if not orders:
                continue
            buy_qty  = sum(o.quantity for o in orders if o.quantity > 0)
            sell_qty = -sum(o.quantity for o in orders if o.quantity < 0)
            if buy_qty > MAX_TICK_DELTA:
                scale = MAX_TICK_DELTA / buy_qty
                for o in orders:
                    if o.quantity > 0:
                        o.quantity = max(1, int(o.quantity * scale))
            if sell_qty > MAX_TICK_DELTA:
                scale = MAX_TICK_DELTA / sell_qty
                for o in orders:
                    if o.quantity < 0:
                        o.quantity = -max(1, int(-o.quantity * scale))

        new_trader_data = json.dumps({"ema": ema_state, "day": day,
                                      "prev_ts": state.timestamp})
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data