import json
from typing import Any, List, Dict

import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
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
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice, observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ─────────────────────────────────────────────
#  CONFIGURATION — tune these per round
# ─────────────────────────────────────────────
POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

# Per-product strategy parameters
PARAMS = {
    # EMERALDS: momentum-based market making
    # EMA tracks the trend; fast-slow crossover skews quotes in trend direction.
    # Stable baseline (~10000) but we let the EMAs adapt rather than pinning.
    "EMERALDS": {
        "strategy": "momentum_mm",
        "fair_value_fixed": None,
        "use_fixed_fv": False,
        "ema_fast": 5,               # Fast EMA for momentum signal
        "ema_slow": 20,              # Slow EMA as trend baseline
        "momentum_weight": 0.4,      # Skew quotes in direction of momentum
        "spread_half": 4,            # Half-spread for maker quotes
        "inventory_skew_factor": 1.0,
        "take_edge": 4,              # Min edge to take a bot order
    },
    # TOMATOES: momentum-based market making
    # More volatile and trending product; momentum skew is more aggressive.
    "TOMATOES": {
        "strategy": "momentum_mm",
        "fair_value_fixed": None,
        "use_fixed_fv": False,
        "ema_fast": 5,               # Fast EMA for momentum signal
        "ema_slow": 20,              # Slow EMA as trend baseline
        "momentum_weight": 0.4,      # Same skew weight as EMERALDS
        "spread_half": 5,            # Slightly wider half-spread (more volatile)
        "inventory_skew_factor": 1.5,# Lean harder against position (volatile product)
        "take_edge": 3,              # Lower edge threshold (more active taking)
    },
}

# ─────────────────────────────────────────────
#  EMA helper
# ─────────────────────────────────────────────
def update_ema(prev: float, new_val: float, span: int) -> float:
    """Exponential moving average update step."""
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev


class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        # ── Load persistent state ──────────────────────────────────────────
        s = {}
        if state.traderData:
            try:
                s = jsonpickle.decode(state.traderData)
            except Exception:
                s = {}

        # Initialise per-product state buckets
        for product in PARAMS:
            if product not in s:
                s[product] = {"ema_fast": None, "ema_slow": None}

        result: Dict[str, List[Order]] = {}

        for product, cfg in PARAMS.items():
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            pos_limit = POSITION_LIMITS.get(product, 20)
            current_pos = state.position.get(product, 0)

            # ── 1. Compute mid price from order book ──────────────────────
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            # ── 2. Update EMAs ─────────────────────────────────────────────
            ps = s[product]
            if ps["ema_fast"] is None:
                ps["ema_fast"] = mid_price
                ps["ema_slow"] = mid_price
            else:
                ps["ema_fast"] = update_ema(ps["ema_fast"], mid_price, cfg["ema_fast"])
                ps["ema_slow"] = update_ema(ps["ema_slow"], mid_price, cfg["ema_slow"])

            # ── 3. Fair value with momentum skew ──────────────────────────
            momentum = ps["ema_fast"] - ps["ema_slow"]   # positive = uptrend
            fair_value = ps["ema_slow"] + cfg["momentum_weight"] * momentum

            # ── 4. Inventory skew ──────────────────────────────────────────
            inv_skew = -current_pos * cfg["inventory_skew_factor"]
            skewed_fv = fair_value + inv_skew

            # ── 5. Quote levels ────────────────────────────────────────────
            half = cfg["spread_half"]
            our_bid_price = round(skewed_fv - half)
            our_ask_price = round(skewed_fv + half)

            # ── 6. Taker logic: hit bots when they're mispriced ───────────
            take_edge = cfg["take_edge"]
            remaining_buy  = pos_limit - current_pos
            remaining_sell = pos_limit + current_pos

            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px <= fair_value - take_edge:
                    vol = -order_depth.sell_orders[ask_px]
                    qty = min(vol, remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px >= fair_value + take_edge:
                    vol = order_depth.buy_orders[bid_px]
                    qty = min(vol, remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                else:
                    break

            # ── 7. Maker logic: post resting quotes ───────────────────────
            maker_bid = min(our_bid_price, best_ask - 1)   # allow inside spread, don't cross ask
            maker_ask = max(our_ask_price, best_bid + 1)   # allow inside spread, don't cross bid

            if remaining_buy > 0:
                orders.append(Order(product, maker_bid, remaining_buy))
            if remaining_sell > 0:
                orders.append(Order(product, maker_ask, -remaining_sell))

            result[product] = orders

            logger.print(
                f"[{product}] ts={state.timestamp} "
                f"mid={mid_price:.1f} fv={fair_value:.1f} skewed={skewed_fv:.1f} "
                f"mom={momentum:.2f} pos={current_pos} "
                f"q={maker_bid}/{maker_ask}"
            )

        # ── Persist state ──────────────────────────────────────────────────
        traderData = jsonpickle.encode(s)
        conversions = 0

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
