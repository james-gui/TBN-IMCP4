"""r3_v39_atm5300 — expand ATM_STRIKES to include VEV_5300.

Based on r3_v37_regime_noskew ($24,192). Single change:
  ATM_STRIKES add VEV_5300 (was only 5000/5100/5200).
VEX ~5250 makes 5300 near-ATM; it's the 4th-highest vol strike (0.0109).
Scaling it up in bull regime captures more EMA-lag buy opportunities.
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

# ── Products ────────────────────────────────────────────────────────────────
HYDROGEL = "HYDROGEL_PACK"
VEX = "VELVETFRUIT_EXTRACT"
VEV_ALL = ["VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
           "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"]

POSITION_LIMITS = {HYDROGEL: 200, VEX: 200}
for sym in VEV_ALL:
    POSITION_LIMITS[sym] = 300

# ── Base configs (unchanged from v140) ───────────────────────────────────────
HYDROGEL_CFG = {"ema_alpha": 0.05, "make_size": 25, "soft_inv_cap": 150, "skew_div": 50.0,
                "take_edge": 7.0, "take_size": 15, "imb_take_threshold": 0.2, "imb_passive_threshold": 0.3}
VEX_CFG       = {"ema_alpha": 0.05, "make_size": 20, "soft_inv_cap": 150, "skew_div": 60.0,
                 "take_edge": 2.0, "take_size": 25, "imb_take_threshold": 0.2, "imb_passive_threshold": 0.3}
VEV_CFG       = {"ema_alpha": 0.02, "make_size": 10, "soft_inv_cap": 200, "skew_div": 10000.0,
                 "take_size": 15, "take_edge": 0.5, "imb_take_threshold": 0.15}
VEV_ALPHA_OVERRIDE     = {"VEV_5300": 0.05, "VEV_5000": 0.01, "VEV_5200": 0.03, "VEV_5400": 0.025}
VEV_TAKE_EDGE_OVERRIDE = {"VEV_4000": 0.2, "VEV_5500": 0.2}

MAX_TICK_DELTA = 50

# ── Regime config (v141 additions) ───────────────────────────────────────────
VEX_EMA_FAST_ALPHA = 0.10   # ~10-tick momentum
VEX_EMA_SLOW_ALPHA = 0.02   # ~50-tick trend

# Strikes that get scaled up in bull regime (highest spread EV near ATM)
ATM_STRIKES = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"}

# Far-OTM strikes suppressed always (zero vol confirmed in notebook analysis)
FAR_OTM_STRIKES = {"VEV_6000", "VEV_6500"}
VEV_SKIP = {"VEV_6000", "VEV_6500"}

BULL_SCALE       = 1.75   # multiplier on make_size + take_size for ATM strikes in bull regime
BULL_SKEW_FACTOR = 0.80   # tighten ask skew divisor in bull (lean short, capture mean-reversion)


def volume_weighted_mid(order_depth: OrderDepth) -> float:
    best_bid = max(order_depth.buy_orders)
    best_ask = min(order_depth.sell_orders)
    bid_vol = order_depth.buy_orders[best_bid]
    ask_vol = abs(order_depth.sell_orders[best_ask])
    return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)


def get_regime(vex_order_depth: OrderDepth, ema_state: dict) -> str:
    """Bull if VEX fast EMA > slow EMA, bear otherwise.

    Updates ema_state['vex_fast'] and ema_state['vex_slow'] in-place.
    Returns 'bull' or 'bear'.
    """
    vwm = volume_weighted_mid(vex_order_depth)
    fast = ema_state.get("vex_fast", vwm)
    slow = ema_state.get("vex_slow", vwm)
    fast = VEX_EMA_FAST_ALPHA * vwm + (1 - VEX_EMA_FAST_ALPHA) * fast
    slow = VEX_EMA_SLOW_ALPHA * vwm + (1 - VEX_EMA_SLOW_ALPHA) * slow
    ema_state["vex_fast"] = fast
    ema_state["vex_slow"] = slow
    return "bull" if fast > slow else "bear"


def passive_inside_wall(sym, order_depth, pos, ema_state, cfg, pos_limit, allow_take=False, regime="bull"):
    """Quote 1 tick inside the wall on each side, with EMA-fair safety.

    If allow_take=True and cfg has take_edge / take_size, also fire a small
    take order when the visible best is far enough from our EMA fair to
    overcome the half-spread cost.
    """
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

    # --- selective take ---
    if allow_take:
        if cfg.get("adaptive_take"):
            spread = best_ask - best_bid
            take_edge = max(2.0, spread / 2.0 + 1.0)
        else:
            take_edge = cfg.get("take_edge", float("inf"))
        take_size = int(cfg.get("take_size", 0))

        imb_threshold = cfg.get("imb_take_threshold", None)
        if imb_threshold is not None:
            bv1 = order_depth.buy_orders[best_bid]
            av1 = abs(order_depth.sell_orders[best_ask])
            imb = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0
        else:
            imb = 0.0

        buy_ok = (imb_threshold is None) or (imb >= -imb_threshold)
        # In bear regime, block option buy-takes to stop accumulating long delta
        if sym.startswith("VEV_") and regime == "bear":
            buy_ok = False
        if buy_ok and best_ask <= fair_unskewed - take_edge and max_buy > 0:
            vol = min(take_size, abs(order_depth.sell_orders[best_ask]), max_buy)
            if vol > 0:
                orders.append(Order(sym, int(best_ask), vol))
                max_buy -= vol
                cur_pos += vol

        sell_ok = (imb_threshold is None) or (imb <= imb_threshold)
        if sell_ok and best_bid >= fair_unskewed + take_edge and max_sell > 0:
            vol = min(take_size, abs(order_depth.buy_orders[best_bid]), max_sell)
            if vol > 0:
                orders.append(Order(sym, int(best_bid), -vol))
                max_sell -= vol
                cur_pos -= vol

    if best_ask - best_bid < 2:
        return orders

    our_bid = best_bid + 1
    our_ask = best_ask - 1
    if our_bid >= our_ask:
        return orders

    bid_ok = our_bid <= fair - 0.5
    ask_ok = our_ask >= fair + 0.5

    # In bear regime, suppress option passive bids (stop accumulating long calls)
    if sym.startswith("VEV_") and regime == "bear":
        bid_ok = False

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
    soft_cap = cfg["soft_inv_cap"]

    if bid_ok and max_buy > 0 and cur_pos < soft_cap:
        size = min(make_size, max_buy)
        if size > 0:
            orders.append(Order(sym, int(our_bid), size))
    if ask_ok and max_sell > 0 and cur_pos > -soft_cap:
        size = min(make_size, max_sell)
        if size > 0:
            orders.append(Order(sym, int(our_ask), -size))

    return orders


def apply_regime_to_cfg(cfg: dict, sym: str, regime: str) -> dict:
    """Return a (possibly modified) copy of cfg based on current regime.

    Bull:  scale up make_size + take_size on ATM strikes; tighten ask skew.
    Bear:  zero out far-OTM strikes (suppress quoting entirely).
    """
    if sym in ATM_STRIKES and regime == "bull":
        cfg = dict(cfg)
        cfg["make_size"] = int(cfg["make_size"] * BULL_SCALE)
        cfg["take_size"] = int(cfg.get("take_size", 0) * BULL_SCALE)
        # Tighten skew divisor on ask side → lean short, capture mean-rev
        cfg["skew_div"] = cfg["skew_div"] * BULL_SKEW_FACTOR
    elif sym in FAR_OTM_STRIKES and regime == "bear":
        cfg = dict(cfg)
        cfg["make_size"] = 0   # suppress passive quoting
        cfg["take_size"] = 0   # suppress takes
    return cfg


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}
        ema_state: Dict[str, float] = trader_state.get("ema", {})

        # ── Regime detection (v141) ──────────────────────────────────────────
        regime = "bull"  # default if VEX not visible
        if VEX in state.order_depths:
            vex_od = state.order_depths[VEX]
            if vex_od.buy_orders and vex_od.sell_orders:
                regime = get_regime(vex_od, ema_state)

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
                result[product] = passive_inside_wall(
                    product, order_depth, pos, ema_state, VEX_CFG,
                    POSITION_LIMITS[VEX], allow_take=True)

            elif product in VEV_ALL:
                if product in VEV_SKIP:
                    result[product] = []
                    continue
                # Build per-strike cfg (v140 overrides)
                cfg = dict(VEV_CFG)
                if product in VEV_ALPHA_OVERRIDE:
                    cfg["ema_alpha"] = VEV_ALPHA_OVERRIDE[product]
                if product in VEV_TAKE_EDGE_OVERRIDE:
                    cfg["take_edge"] = VEV_TAKE_EDGE_OVERRIDE[product]
                # Apply regime adjustments (v141)
                cfg = apply_regime_to_cfg(cfg, product, regime)
                result[product] = passive_inside_wall(
                    product, order_depth, pos, ema_state, cfg,
                    POSITION_LIMITS[product], allow_take=True, regime=regime)

            else:
                result[product] = []

        # ── Per-tick exposure cap (v140, unchanged) ──────────────────────────
        for product, orders in result.items():
            if not orders:
                continue
            buy_qty = sum(o.quantity for o in orders if o.quantity > 0)
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

        new_trader_data = json.dumps({"ema": ema_state})
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data