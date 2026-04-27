import json
from typing import Any, List, Dict

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

# ── Strategy: Asymmetric Market Making ────────────────────────────────────────
#
# EMERALDS: fixed fv=10000, size skew only, inventory pause at ±70.
#
# TOMATOES: EMA of volume-weighted mid as fv, plus:
#   - price_skew: shift both quotes toward the unwinding direction when inventory
#     builds up (long → shift down → ask becomes more competitive, faster unwind)
#   - size_skew: reduce size on the accumulating side
#   - inventory_pause: go one-sided when position is extreme (±55)
#   - fv anchor: never quote on wrong side of fair value
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
EOD_THRESHOLD = 990_000

PARAMS = {
    "EMERALDS": {
        "fair_value":      10000,
        "order_size":      20,
        "skew_factor":     0.5,
        "price_skew":      0.0,
        "take_edge":       1,
        "min_spread":      2,
        "ema_alpha":       None,
        "inventory_pause": 70,
    },
    "TOMATOES": {
        "fair_value":      None,
        "order_size":      20,
        "skew_factor":     1.0,
        "price_skew":      2.0,
        "take_edge":       1,
        "min_spread":      4,
        "ema_alpha":       0.15,
        "inventory_pause": 55,
    },
}


def volume_weighted_mid(order_depth: OrderDepth) -> float:
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    bid_vol  = order_depth.buy_orders[best_bid]
    ask_vol  = abs(order_depth.sell_orders[best_ask])
    return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        tick_in_day = state.timestamp % 1_000_000

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}
        ema_state: Dict[str, float] = trader_state.get("ema", {})

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            # ── EOD flatten ───────────────────────────────────────────────────
            if tick_in_day >= EOD_THRESHOLD:
                if pos > 0:
                    orders.append(Order(product, best_bid, -pos))
                elif pos < 0:
                    orders.append(Order(product, best_ask, -pos))
                logger.print(f"[{product}] EOD flatten pos={pos}")
                result[product] = orders
                continue

            spread = best_ask - best_bid
            cfg = PARAMS.get(product, {
                "fair_value": None, "order_size": 20, "skew_factor": 0.0,
                "price_skew": 0.0, "take_edge": 1, "min_spread": 2,
                "ema_alpha": None, "inventory_pause": 70,
            })

            # ── Fair value ────────────────────────────────────────────────────
            if cfg["fair_value"] is not None:
                fv = cfg["fair_value"]
            else:
                vwm = volume_weighted_mid(order_depth)
                alpha = cfg["ema_alpha"]
                if product not in ema_state:
                    ema_state[product] = vwm
                ema_state[product] = alpha * vwm + (1 - alpha) * ema_state[product]
                fv = ema_state[product]

            take_edge      = cfg["take_edge"]
            remaining_buy  = POSITION_LIMIT - pos
            remaining_sell = POSITION_LIMIT + pos

            # ── Taker leg ─────────────────────────────────────────────────────
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px <= fv - take_edge:
                    qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                    logger.print(f"[{product}] TAKE buy {qty}@{ask_px} fv={fv:.2f}")
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px >= fv + take_edge:
                    qty = min(order_depth.buy_orders[bid_px], remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                    logger.print(f"[{product}] TAKE sell {qty}@{bid_px} fv={fv:.2f}")
                else:
                    break

            # ── Maker leg (asymmetric) ─────────────────────────────────────────
            min_spread  = cfg["min_spread"]
            inv_pause   = cfg["inventory_pause"]
            skew_ratio  = pos / POSITION_LIMIT
            size_skew   = cfg["skew_factor"]
            price_skew  = cfg["price_skew"]

            if spread > min_spread:
                our_bid = best_bid + 1
                our_ask = best_ask - 1

                tick_shift = round(price_skew * skew_ratio)
                our_bid -= tick_shift
                our_ask -= tick_shift

                base_size = cfg["order_size"]
                bid_size  = max(1, round(base_size * (1 - size_skew * skew_ratio)))
                ask_size  = max(1, round(base_size * (1 + size_skew * skew_ratio)))

                post_bid = pos < inv_pause
                post_ask = pos > -inv_pause

                if post_bid and our_bid < fv and remaining_buy > 0:
                    orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
                    logger.print(f"[{product}] BID {our_bid} sz={bid_size} pos={pos} shift={tick_shift}")
                if post_ask and our_ask > fv and remaining_sell > 0:
                    orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))
                    logger.print(f"[{product}] ASK {our_ask} sz={ask_size} pos={pos} shift={tick_shift}")

            result[product] = orders

        new_trader_data = json.dumps({"ema": ema_state})
        logger.flush(state, result, 0, new_trader_data)
        return result, 0, new_trader_data
