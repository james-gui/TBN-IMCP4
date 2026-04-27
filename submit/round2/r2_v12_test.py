import json
import math
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

POSITION_LIMIT = 80
OSMIUM_EOD     = 995_000

ROOT_CFG = {
    "aggressive_buy_offset": 10,
    "passive_bid_offset":    0,
}

# v26: take_edge=0 + skew_div=30 + max_make_vol=25
# We know take_edge=0 and skew_div=30 are best, but max_make_vol=25 was
# never tested with this combo (original v18 had take_edge=1, skew_div=20).
OSMIUM_CFG = {
    "take_edge":    0,
    "skew_div":     20,
    "max_make_vol": 25,
}


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        root_base: float | None = trader_state.get("root_base")

        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                root_base = mid - state.timestamp / 1000.0

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            if product == "INTARIAN_PEPPER_ROOT":
                if root_base is not None:
                    orders = self._trade_root(state, order_depth, pos, root_base)

            elif product == "ASH_COATED_OSMIUM":
                if tick >= OSMIUM_EOD:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)
                    if pos > 0:
                        orders.append(Order(product, best_bid, -pos))
                    elif pos < 0:
                        orders.append(Order(product, best_ask, -pos))
                else:
                    orders = self._trade_osmium(order_depth, pos)

            result[product] = orders

        new_trader_data = json.dumps({"root_base": root_base})
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data

    def _trade_root(self, state: TradingState, order_depth: OrderDepth, pos: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        product  = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        buy_cap  = POSITION_LIMIT - pos

        agg_buy = math.floor(fair + ROOT_CFG["aggressive_buy_offset"])
        for ask_px in sorted(order_depth.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask_px <= agg_buy:
                qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                orders.append(Order(product, ask_px, qty))
                buy_cap -= qty
                pos     += qty
            else:
                break

        if buy_cap > 0:
            our_bid = math.floor(fair - ROOT_CFG["passive_bid_offset"])
            our_bid = min(our_bid, best_bid + 1)
            our_bid = min(our_bid, best_ask - 1)
            if our_bid < best_ask and our_bid > 0:
                orders.append(Order(product, our_bid, buy_cap))

        return orders

    def _trade_osmium(self, order_depth: OrderDepth, pos: int) -> List[Order]:
        orders  = []
        product = "ASH_COATED_OSMIUM"

        buy_orders  = sorted(order_depth.buy_orders.items(),  key=lambda x: -x[0])
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x:  x[0])

        wall_bid = buy_orders[-1][0]
        wall_ask = sell_orders[-1][0]
        wall_mid = (wall_bid + wall_ask) / 2.0

        fv = wall_mid - pos / OSMIUM_CFG["skew_div"]

        take_edge    = OSMIUM_CFG["take_edge"]
        max_make_vol = OSMIUM_CFG["max_make_vol"]
        max_buy      = POSITION_LIMIT - pos
        max_sell     = POSITION_LIMIT + pos
        cur_pos      = pos

        for ask_px, ask_vol in sell_orders:
            ask_vol = abs(ask_vol)
            if ask_px <= fv - take_edge and max_buy > 0:
                vol = min(ask_vol, max_buy)
                orders.append(Order(product, ask_px, vol))
                max_buy -= vol; cur_pos += vol
            elif ask_px <= wall_mid and cur_pos < 0 and max_buy > 0:
                vol = min(ask_vol, min(-cur_pos, max_buy))
                orders.append(Order(product, ask_px, vol))
                max_buy -= vol; cur_pos += vol

        for bid_px, bid_vol in buy_orders:
            bid_vol = abs(bid_vol)
            if bid_px >= fv + take_edge and max_sell > 0:
                vol = min(bid_vol, max_sell)
                orders.append(Order(product, bid_px, -vol))
                max_sell -= vol; cur_pos -= vol
            elif bid_px >= wall_mid and cur_pos > 0 and max_sell > 0:
                vol = min(bid_vol, min(cur_pos, max_sell))
                orders.append(Order(product, bid_px, -vol))
                max_sell -= vol; cur_pos -= vol

        bid_px = wall_bid + 1
        for bp, bv in buy_orders:
            if bv > 1 and bp + 1 < fv:
                bid_px = max(bid_px, bp + 1); break
            elif bp < fv:
                bid_px = max(bid_px, bp); break

        ask_px = wall_ask - 1
        for ap, av in sell_orders:
            av = abs(av)
            if av > 1 and ap - 1 > fv:
                ask_px = min(ask_px, ap - 1); break
            elif ap > fv:
                ask_px = min(ask_px, ap); break

        bid_px = int(bid_px); ask_px = int(ask_px)
        if bid_px >= ask_px:
            bid_px = math.floor(fv) - 1
            ask_px = math.ceil(fv)  + 1

        if max_buy > 0:
            orders.append(Order(product, bid_px,  min(max_make_vol, max_buy)))
        if max_sell > 0:
            orders.append(Order(product, ask_px, -min(max_make_vol, max_sell)))

        return orders