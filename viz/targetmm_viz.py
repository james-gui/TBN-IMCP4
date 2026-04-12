import json
from typing import Any, List, Dict, Optional

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

# ── Strategy: Target-Inventory Market Making ──────────────────────────────────
#
# Per tick:
#   1. EWMA: update exponential moving average of mid-price.
#   2. TARGET: signal = mid - EWMA (mean-reversion: +signal → overbought → short).
#              target_pos = clip(-round(signal * signal_scalar), -LIMIT, +LIMIT)
#              During warmup: target_pos = None (no directional view, taker free).
#   3. TAKER: sweep book toward target. Only buy if target > pos (capped at gap),
#             only sell if target < pos (capped at gap). Still requires price edge.
#   4. MAKER: quote inside spread. Skew sizes based on (pos - target_pos) so
#             quotes lean toward closing the gap to target.
#
# EMERALDS: no EWMA / target (stable product). Pure taker + maker.
# EOD: flatten all inventory in the final 100 ticks of each day.
# ──────────────────────────────────────────────────────────────────────────────

POSITION_LIMIT = 80
EOD_THRESHOLD = 990_000

PARAMS = {
    "EMERALDS": {
        "fair_value": 10000,
        "order_size": 20,
        "skew_factor": 0.5,
        "take_edge": 1,
        "min_spread": 2,
    },
    "TOMATOES": {
        "fair_value": None,
        "order_size": 20,
        "skew_factor": 0.5,
        "take_edge": 1,
        "min_spread": 4,
        "ema_span": 100,          # EWMA lookback in ticks
        "warmup_ticks": 100,      # ticks before EWMA is trusted
        "signal_scalar": 4.0,     # target_pos units per tick of EMA deviation
    },
}


def update_ema(prev: float, new_val: float, span: int) -> float:
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        tick_in_day = state.timestamp % 1_000_000
        trader_data: dict = json.loads(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2

            # ── EOD LIQUIDATION PHASE ──────────────────────────────────────────
            if tick_in_day >= EOD_THRESHOLD:
                if pos > 0:
                    orders.append(Order(product, best_bid, -pos))
                elif pos < 0:
                    orders.append(Order(product, best_ask, -pos))
                logger.print(f"[{product}] EOD flatten pos={pos}")
                result[product] = orders
                continue

            # ── NORMAL TRADING PHASE ───────────────────────────────────────────
            spread = best_ask - best_bid
            cfg = PARAMS.get(product, {"fair_value": None, "order_size": 20, "skew_factor": 0.0, "take_edge": 1, "min_spread": 2})
            fv = cfg["fair_value"] if cfg["fair_value"] is not None else mid
            take_edge = cfg["take_edge"]

            remaining_buy  = POSITION_LIMIT - pos
            remaining_sell = POSITION_LIMIT + pos

            # ── STEP 1: EWMA + TARGET INVENTORY ───────────────────────────────
            target_pos: Optional[int] = None  # None = no view, taker fires freely

            if "ema_span" in cfg:
                pstate = trader_data.setdefault(product, {"ema": mid, "ticks": 0})
                ema = update_ema(pstate["ema"], mid, cfg["ema_span"])
                ticks = pstate["ticks"] + 1
                pstate["ema"] = ema
                pstate["ticks"] = ticks

                fv = ema  # EWMA is our fair value estimate

                if ticks >= cfg["warmup_ticks"]:
                    signal = mid - ema  # positive = price above avg = mean-revert short
                    raw_target = -signal * cfg["signal_scalar"]
                    target_pos = int(max(-POSITION_LIMIT, min(POSITION_LIMIT, round(raw_target))))
                    logger.print(
                        f"[{product}] EMA={ema:.1f} mid={mid:.1f} signal={signal:.2f} "
                        f"target={target_pos} pos={pos}"
                    )
                else:
                    logger.print(f"[{product}] WARMUP tick={ticks}/{cfg['warmup_ticks']}")

            # ── STEP 2: TAKER LEG (gated by target) ───────────────────────────
            # If we have a target: only take in the direction that closes the gap,
            # capped at the gap size. If no target: taker fires freely.
            if target_pos is None:
                taker_buy_cap  = remaining_buy
                taker_sell_cap = remaining_sell
            else:
                taker_buy_cap  = max(0, target_pos - pos)
                taker_sell_cap = max(0, pos - target_pos)

            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0 or taker_buy_cap <= 0:
                    break
                if ask_px <= fv - take_edge:
                    qty = min(-order_depth.sell_orders[ask_px], remaining_buy, taker_buy_cap)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy  -= qty
                    taker_buy_cap  -= qty
                    logger.print(f"[{product}] TAKE buy {qty}@{ask_px} fv={fv:.1f} target={target_pos}")
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0 or taker_sell_cap <= 0:
                    break
                if bid_px >= fv + take_edge:
                    qty = min(order_depth.buy_orders[bid_px], remaining_sell, taker_sell_cap)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell  -= qty
                    taker_sell_cap  -= qty
                    logger.print(f"[{product}] TAKE sell {qty}@{bid_px} fv={fv:.1f} target={target_pos}")
                else:
                    break

            # ── STEP 3: MAKER LEG (skew toward target) ────────────────────────
            min_spread = cfg.get("min_spread", 2)
            if spread > min_spread:
                our_bid = best_bid + 1
                our_ask = best_ask - 1

                base_size = cfg["order_size"]
                skew = cfg["skew_factor"]

                # Skew relative to target: positive ratio = too long vs target → lean ask
                if target_pos is not None:
                    raw_ratio = (pos - target_pos) / POSITION_LIMIT
                else:
                    raw_ratio = pos / POSITION_LIMIT
                skew_ratio = max(-1.0, min(1.0, raw_ratio))

                bid_size = max(1, round(base_size * (1 - skew * skew_ratio)))
                ask_size = max(1, round(base_size * (1 + skew * skew_ratio)))

                if remaining_buy > 0:
                    orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
                if remaining_sell > 0:
                    orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))

                logger.print(
                    f"[{product}] MAKE {our_bid}/{our_ask} sz={bid_size}/{ask_size} "
                    f"pos={pos} target={target_pos} skew={skew_ratio:.2f}"
                )
            else:
                logger.print(f"[{product}] spread={spread} <= min_spread={min_spread}, skipping maker")

            result[product] = orders

        traderData = json.dumps(trader_data)
        logger.flush(state, result, 0, traderData)
        return result, 0, traderData
