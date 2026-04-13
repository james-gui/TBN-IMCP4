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

# ── Strategy: best80530 ────────────────────────────────────────────────────────
#
# Best platform result: 80,530 seashells.
#
# EMERALDS: fixed fv=10000, larger size (32), quote at fv±2, take at fv±2.
#           Cubic skew + soft/hard inventory limits (40/60).
#
# TOMATOES: VWM EMA (alpha=0.12) as fv, quote at fv±3 / take at fv±2.
#           Tighter inventory limits (30/50) — volatile product.
#           Cubic skew for smooth size reduction near limits.
#
# Maker quotes: min(best_bid+1, fv-make_width) bid / max(best_ask-1, fv+make_width) ask
#   → undercuts competitors but never tighter than fv±make_width.
#   → fv anchor: never quote on wrong side of fv.
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
EOD_THRESHOLD = 990_000

PARAMS = {
    "EMERALDS": {
        "fair_value":      10000,
        "take_width":      2,
        "make_width":      2,
        "order_size":      32,
        "inventory_limit": 40,
        "inventory_hard":  60,
        "ema_alpha":       None,
    },
    "TOMATOES": {
        "fair_value":      None,
        "take_width":      2,
        "make_width":      3,
        "order_size":      24,
        "inventory_limit": 30,
        "inventory_hard":  50,
        "ema_alpha":       0.12,
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

            cfg = PARAMS.get(product, {
                "fair_value": None, "take_width": 2, "make_width": 3,
                "order_size": 20, "inventory_limit": 35, "inventory_hard": 55,
                "ema_alpha": 0.12,
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
                logger.print(f"[{product}] vwm={vwm:.2f} fv={fv:.2f} pos={pos}")

            take_width = cfg["take_width"]
            make_width = cfg["make_width"]
            base_size  = cfg["order_size"]
            inv_limit  = cfg["inventory_limit"]
            inv_hard   = cfg["inventory_hard"]

            remaining_buy  = POSITION_LIMIT - pos
            remaining_sell = POSITION_LIMIT + pos

            # ── Layer 1: taker ────────────────────────────────────────────────
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px <= fv - take_width:
                    qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                    logger.print(f"[{product}] TAKE buy {qty}@{ask_px} fv={fv:.2f}")
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px >= fv + take_width:
                    qty = min(order_depth.buy_orders[bid_px], remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                    logger.print(f"[{product}] TAKE sell {qty}@{bid_px} fv={fv:.2f}")
                else:
                    break

            # ── Layer 2: maker ────────────────────────────────────────────────
            our_bid = min(best_bid + 1, round(fv) - make_width)
            our_ask = max(best_ask - 1, round(fv) + make_width)

            our_ask = max(our_ask, best_bid + 1)
            our_bid = min(our_bid, best_ask - 1)

            if our_bid >= our_ask:
                result[product] = orders
                continue

            # Cubic skew — flat near zero, steep near limits
            skew_ratio = (pos / POSITION_LIMIT) ** 3

            bid_size = max(1, round(base_size * (1 - abs(skew_ratio) if pos > 0 else 1)))
            ask_size = max(1, round(base_size * (1 - abs(skew_ratio) if pos < 0 else 1)))

            want_bid = pos < inv_hard
            want_ask = pos > -inv_hard

            if pos > inv_limit:
                scale    = 1 - (pos - inv_limit) / (inv_hard - inv_limit)
                bid_size = max(1, round(bid_size * scale))
            elif pos < -inv_limit:
                scale    = 1 - (-pos - inv_limit) / (inv_hard - inv_limit)
                ask_size = max(1, round(ask_size * scale))

            if want_bid and our_bid < fv and remaining_buy > 0:
                orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
                logger.print(f"[{product}] BID {our_bid} sz={bid_size} pos={pos}")
            if want_ask and our_ask > fv and remaining_sell > 0:
                orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))
                logger.print(f"[{product}] ASK {our_ask} sz={ask_size} pos={pos}")

            result[product] = orders

        new_trader_data = json.dumps({"ema": ema_state})
        logger.flush(state, result, 0, new_trader_data)
        return result, 0, new_trader_data
