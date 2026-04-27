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

# ── Strategy: Improved Market Making ─────────────────────────────────────────
#
# Improvements over basicmm:
#
# 1. EMERALDS take_edge=0: take asks/bids at exactly fair value (10000).
#    Backtester data shows ~333 ticks where bots post asks at 10000.
#    With take_edge=1 we skip them. With take_edge=0 we buy at FV and sell
#    at 10007 via maker leg → +7 ticks per unit captured.
#
# 2. Per-product soft_limit: when |pos| > soft_limit, stop quoting the
#    side that would increase inventory. Only quote the flattening side.
#    Prevents position from maxing out at ±80 during trending markets.
#    TOMATOES trends significantly intraday — this caps the inventory loss.
#
# 3. Higher skew for TOMATOES (1.5 vs 0.5): more aggressive size reduction
#    as position builds up on the trending side.
#
# Per-product params:
#   fair_value:  fixed true price (None = use mid-price dynamically each tick)
#   order_size:  base units to quote per side per tick (maker leg)
#   skew_factor: inventory skew (0.0 = off, 2.0 = bid_size→0 at half pos limit)
#   take_edge:   min ticks of edge vs FV required to take a bot order
#                (0 = take anything at or better than FV)
#   soft_limit:  stop quoting the accumulating side above this position
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80

PARAMS = {
    "EMERALDS": {
        "fair_value": 10000,
        "order_size": 20,
        "skew_factor": 0.5,
        "take_edge": 0,      # take asks/bids at exactly FV (10000) — free edge vs basicmm's take_edge=1
        "soft_limit": 80,    # no soft cap — EMERALDS mean-reverts, always quote both sides
    },
    "TOMATOES": {
        "fair_value": None,
        "order_size": 20,
        "skew_factor": 1.0,  # moderate reduction as inventory builds
        "take_edge": 2,
        "soft_limit": 35,    # stop bidding above 35 — TOMATOES trends intraday, cap long exposure
    },
}


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            spread = best_ask - best_bid

            cfg = PARAMS.get(product, {"fair_value": None, "order_size": 20, "skew_factor": 0.0, "take_edge": 1, "soft_limit": POSITION_LIMIT})
            fv = cfg["fair_value"] if cfg["fair_value"] is not None else (best_bid + best_ask) / 2
            take_edge = cfg["take_edge"]
            soft_limit = cfg.get("soft_limit", POSITION_LIMIT)

            remaining_buy  = POSITION_LIMIT - pos
            remaining_sell = POSITION_LIMIT + pos

            # ── 1. TAKER LEG ──────────────────────────────────────────────────
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px <= fv - take_edge:
                    qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                    logger.print(f"[{product}] TAKE buy {qty}@{ask_px} fv={fv:.1f}")
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px >= fv + take_edge:
                    qty = min(order_depth.buy_orders[bid_px], remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                    logger.print(f"[{product}] TAKE sell {qty}@{bid_px} fv={fv:.1f}")
                else:
                    break

            # ── 2. MAKER LEG ──────────────────────────────────────────────────
            if spread > 1:
                # For fixed-FV products: use the deepest level *below* FV as bid
                # reference and *above* FV as ask reference. This prevents quoting
                # at best_bid+1 when best_bid==FV (which would place a buy ABOVE FV).
                if cfg.get("fair_value") is not None:
                    deeper_bids = [p for p in order_depth.buy_orders if p < fv]
                    deeper_asks = [p for p in order_depth.sell_orders if p > fv]
                    make_bid_ref = max(deeper_bids) if deeper_bids else best_bid
                    make_ask_ref = min(deeper_asks) if deeper_asks else best_ask
                else:
                    make_bid_ref = best_bid
                    make_ask_ref = best_ask
                our_bid = make_bid_ref + 1
                our_ask = make_ask_ref - 1

                base_size = cfg["order_size"]
                skew = cfg["skew_factor"]
                skew_ratio = pos / POSITION_LIMIT  # -1.0 to +1.0
                bid_size = max(1, round(base_size * (1 - skew * skew_ratio)))
                ask_size = max(1, round(base_size * (1 + skew * skew_ratio)))

                # Soft limit: stop quoting the side that would worsen inventory
                quote_bid = remaining_buy > 0 and pos < soft_limit
                quote_ask = remaining_sell > 0 and pos > -soft_limit

                if quote_bid:
                    orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
                if quote_ask:
                    orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))

                logger.print(
                    f"[{product}] MAKE {our_bid if quote_bid else '-'}/{our_ask if quote_ask else '-'}"
                    f" sz={bid_size if quote_bid else 0}/{ask_size if quote_ask else 0} pos={pos}"
                )
            else:
                logger.print(f"[{product}] spread={spread} too tight for maker quotes")

            result[product] = orders

        logger.flush(state, result, 0, "")
        return result, 0, ""
