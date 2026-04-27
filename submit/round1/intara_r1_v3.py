
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
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
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
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff,
                observation.sugarPrice, observation.sunlightIndex,
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

POSITION_LIMITS = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

ROOT_CFG = {
    "target_position": 80,
    "aggressive_buy_offset": 8,   # lift asks up to fair + 8
    "passive_bid_offset": 0,      # keep bid very near fair
    "sell_offset": 20,            # almost never sell
}

OSMIUM_CFG = {
    "take_edge": 1.0,             # small edge threshold
    "make_width": 1,              # tighter quote width
    "order_size": 14,
    "inventory_limit": 35,
    "inventory_hard": 65,
    "skew_per_unit": 0.05,
    "eod_flatten_time": 995_000,
}


class Trader:
    def best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def infer_root_base(self, timestamp: int, root_mid: float) -> float:
        # Pepper root appears to follow a near-linear upward drift:
        # fair ~= base + timestamp / 1000
        return root_mid - (timestamp / 1000.0)

    def root_fair(self, state: TradingState, root_base: float) -> float:
        return root_base + state.timestamp / 1000.0

    def trade_root(self, state: TradingState, order_depth: OrderDepth, position: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        limit = POSITION_LIMITS["INTARIAN_PEPPER_ROOT"]
        fair = self.root_fair(state, root_base)
        best_bid, best_ask = self.best_bid_ask(order_depth)

        buy_capacity = limit - position
        sell_capacity = limit + position

        # 1) Aggressively accumulate long inventory
        acceptable_buy = math.floor(fair + ROOT_CFG["aggressive_buy_offset"])
        for ask_px in sorted(order_depth.sell_orders.keys()):
            if buy_capacity <= 0:
                break
            if ask_px <= acceptable_buy:
                qty = min(-order_depth.sell_orders[ask_px], buy_capacity)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, qty))
                    buy_capacity -= qty
                    position += qty
            else:
                break

        # 2) If not full yet, keep a supporting passive bid very near fair
        if buy_capacity > 0 and position < ROOT_CFG["target_position"]:
            our_bid = math.floor(fair - ROOT_CFG["passive_bid_offset"])
            if best_bid is not None:
                our_bid = min(our_bid, best_bid + 1)
            if best_ask is not None:
                our_bid = min(our_bid, best_ask - 1)
            qty = min(ROOT_CFG["target_position"] - position, buy_capacity)
            if qty > 0 and best_ask is not None and our_bid < best_ask:
                orders.append(Order("INTARIAN_PEPPER_ROOT", our_bid, qty))

        # 3) Only sell into obviously rich bids
        expensive_sell = math.ceil(fair + ROOT_CFG["sell_offset"])
        for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_capacity <= 0:
                break
            if bid_px >= expensive_sell:
                qty = min(order_depth.buy_orders[bid_px], sell_capacity)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bid_px, -qty))
                    sell_capacity -= qty
            else:
                break

        return orders

    def predict_osmium_fair(self, mids: List[float], fallback_mid: float) -> float:
        if len(mids) == 0:
            return fallback_mid
        if len(mids) == 1:
            return mids[-1]
        if len(mids) == 2:
            return sum(mids) / 2.0

        # Use a mild mean-reversion model, but keep it conservative.
        delta1 = mids[-1] - mids[-2]
        delta2 = mids[-2] - mids[-3]
        return mids[-1] - 0.35 * delta1 - 0.15 * delta2

    def trade_osmium(self, order_depth: OrderDepth, position: int, mids: List[float]) -> tuple[List[Order], List[float]]:
        orders: List[Order] = []
        limit = POSITION_LIMITS["ASH_COATED_OSMIUM"]

        mid = self.mid_price(order_depth)
        if mid is None:
            return orders, mids

        mids = (mids + [mid])[-3:]
        fair = self.predict_osmium_fair(mids, mid)

        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders, mids

        buy_capacity = limit - position
        sell_capacity = limit + position

        # 1) Take small stale edges
        for ask_px in sorted(order_depth.sell_orders.keys()):
            if buy_capacity <= 0:
                break
            if fair - ask_px >= OSMIUM_CFG["take_edge"]:
                qty = min(-order_depth.sell_orders[ask_px], buy_capacity)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_px, qty))
                    buy_capacity -= qty
            else:
                break

        for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_capacity <= 0:
                break
            if bid_px - fair >= OSMIUM_CFG["take_edge"]:
                qty = min(order_depth.buy_orders[bid_px], sell_capacity)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_px, -qty))
                    sell_capacity -= qty
            else:
                break

        # 2) Always make a two-sided market
        skew = OSMIUM_CFG["skew_per_unit"] * position
        our_bid = math.floor(fair - OSMIUM_CFG["make_width"] - skew)
        our_ask = math.ceil(fair + OSMIUM_CFG["make_width"] - skew)

        # Stay competitive
        our_bid = min(our_bid, best_bid + 1)
        our_ask = max(our_ask, best_ask - 1)

        # Do not cross the spread
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if our_bid < our_ask:
            base_size = OSMIUM_CFG["order_size"]
            inv_limit = OSMIUM_CFG["inventory_limit"]
            inv_hard = OSMIUM_CFG["inventory_hard"]

            bid_size = base_size
            ask_size = base_size

            # Reduce size on the inventory-worsening side
            if position > 0:
                ratio = min(1.0, position / limit)
                bid_size = max(1, round(base_size * (1 - ratio ** 2)))
            elif position < 0:
                ratio = min(1.0, (-position) / limit)
                ask_size = max(1, round(base_size * (1 - ratio ** 2)))

            if position > inv_limit:
                scale = 1 - (position - inv_limit) / max(1, (inv_hard - inv_limit))
                bid_size = max(1, round(bid_size * max(0.0, scale)))
            elif position < -inv_limit:
                scale = 1 - ((-position) - inv_limit) / max(1, (inv_hard - inv_limit))
                ask_size = max(1, round(ask_size * max(0.0, scale)))

            if position < inv_hard and buy_capacity > 0:
                orders.append(Order("ASH_COATED_OSMIUM", our_bid, min(bid_size, buy_capacity)))
            if position > -inv_hard and sell_capacity > 0:
                orders.append(Order("ASH_COATED_OSMIUM", our_ask, -min(ask_size, sell_capacity)))

        return orders, mids

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        root_base = trader_state.get("root_base")
        osmium_mids = trader_state.get("osmium_mids", [])

        # Initialize root base from first observed root mid
        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            root_mid = self.mid_price(state.order_depths["INTARIAN_PEPPER_ROOT"])
            if root_mid is not None:
                root_base = self.infer_root_base(state.timestamp, root_mid)

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            position = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            if product == "INTARIAN_PEPPER_ROOT":
                if root_base is not None:
                    orders = self.trade_root(state, order_depth, position, root_base)

            elif product == "ASH_COATED_OSMIUM":
                best_bid, best_ask = self.best_bid_ask(order_depth)
                if state.timestamp % 1_000_000 >= OSMIUM_CFG["eod_flatten_time"]:
                    if position > 0 and best_bid is not None:
                        orders.append(Order(product, best_bid, -position))
                    elif position < 0 and best_ask is not None:
                        orders.append(Order(product, best_ask, -position))
                else:
                    orders, osmium_mids = self.trade_osmium(order_depth, position, osmium_mids)

            result[product] = orders

        trader_data = json.dumps({
            "root_base": root_base,
            "osmium_mids": osmium_mids,
        })
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
