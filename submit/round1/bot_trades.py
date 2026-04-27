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

PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
MAX_SIZE_HISTORY = 50   # max trades stored per counterparty
MAX_EXTREMA      = 20   # max extrema records kept


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        try:
            stats = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            stats = {}

        for product in PRODUCTS:
            if product not in stats:
                stats[product] = {
                    "buyers":         {},
                    "sellers":        {},
                    "by_size":        {},
                    "price_hi":       None,
                    "price_lo":       None,
                    "extrema_trades": [],
                    "total_trades":   0,
                }

            s = stats[product]
            trades: List[Trade] = state.market_trades.get(product, [])

            # ── Update rolling price hi/lo from order book mid ──
            od = state.order_depths.get(product)
            if od and od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
                s["price_hi"] = mid if s["price_hi"] is None else max(s["price_hi"], mid)
                s["price_lo"] = mid if s["price_lo"] is None else min(s["price_lo"], mid)

            for t in trades:
                buyer  = t.buyer  or "ANON"
                seller = t.seller or "ANON"
                qty    = abs(t.quantity)
                px     = t.price
                ts     = state.timestamp

                s["total_trades"] += 1

                # ── 1. Buyer frequency + size profile (capped) ──
                if buyer not in s["buyers"]:
                    s["buyers"][buyer] = []
                s["buyers"][buyer].append(qty)
                s["buyers"][buyer] = s["buyers"][buyer][-MAX_SIZE_HISTORY:]

                # ── 2. Seller frequency + size profile (capped) ──
                if seller not in s["sellers"]:
                    s["sellers"][seller] = []
                s["sellers"][seller].append(qty)
                s["sellers"][seller] = s["sellers"][seller][-MAX_SIZE_HISTORY:]

                # ── 3. Size → counterparty mapping ──
                size_key = str(qty)
                if size_key not in s["by_size"]:
                    s["by_size"][size_key] = {"buyers": [], "sellers": []}
                s["by_size"][size_key]["buyers"].append(buyer)
                s["by_size"][size_key]["sellers"].append(seller)
                # Cap to last 50 per size bucket
                s["by_size"][size_key]["buyers"]  = s["by_size"][size_key]["buyers"][-MAX_SIZE_HISTORY:]
                s["by_size"][size_key]["sellers"] = s["by_size"][size_key]["sellers"][-MAX_SIZE_HISTORY:]

                # ── 4. Extrema proximity: within 3 ticks of rolling hi/lo ──
                near_hi = s["price_hi"] is not None and px >= s["price_hi"] - 3
                near_lo = s["price_lo"] is not None and px <= s["price_lo"] + 3
                if near_hi or near_lo:
                    tag = "HI" if near_hi else "LO"
                    s["extrema_trades"].append([ts, buyer, seller, qty, px, tag])
                    s["extrema_trades"] = s["extrema_trades"][-MAX_EXTREMA:]

            # ── Summary log every 10k timestamps ──
            if state.timestamp % 10_000 == 0:
                logger.print(f"\n=== {product} @ t={state.timestamp} | total_trades={s['total_trades']} ===")
                logger.print(f"price_hi={s['price_hi']} price_lo={s['price_lo']}")

                # Top 5 buyers by trade count
                top_buyers = sorted(s["buyers"].items(), key=lambda x: len(x[1]), reverse=True)[:5]
                logger.print("TOP BUYERS:")
                for name, sizes in top_buyers:
                    unique_sizes = sorted(set(sizes))
                    logger.print(f"  {name}: {len(sizes)} trades, sizes={unique_sizes}")

                # Top 5 sellers by trade count
                top_sellers = sorted(s["sellers"].items(), key=lambda x: len(x[1]), reverse=True)[:5]
                logger.print("TOP SELLERS:")
                for name, sizes in top_sellers:
                    unique_sizes = sorted(set(sizes))
                    logger.print(f"  {name}: {len(sizes)} trades, sizes={unique_sizes}")

                # Size fingerprint
                logger.print("SIZE FINGERPRINT:")
                for size_key, data in sorted(s["by_size"].items(), key=lambda x: int(x[0])):
                    ub = len(set(data["buyers"]))
                    us = len(set(data["sellers"]))
                    flag = " <<< INSIDER CANDIDATE" if (ub == 1 or us == 1) else ""
                    logger.print(f"  size {size_key}: {ub} unique buyers, {us} unique sellers{flag}")

                # Last 10 extrema trades
                logger.print("RECENT EXTREMA TRADES:")
                for rec in s["extrema_trades"][-10:]:
                    ts_, buyer_, seller_, qty_, px_, tag_ = rec
                    logger.print(f"  [{tag_}] t={ts_} buyer={buyer_} seller={seller_} qty={qty_} px={px_}")

        logger.flush(state, {}, 0, json.dumps(stats))
        return {}, 0, json.dumps(stats)