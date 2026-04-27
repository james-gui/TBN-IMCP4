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
    "aggressive_buy_offset":  10,
    "aggressive_sell_offset": 8,
    "passive_bid_offset":     0,
    "sell_offset":            20,
    "buy_offset":             20,
    "target_position":        80,
    # ── Deviation stop-loss ──
    "dump_threshold":         15,
    "recover_threshold":      8,
}

OSMIUM_CFG = {
    "make_width":      1,
    "take_width":      0,
    "order_size":      20,
    "inv_limit":       30,
    "ema_alpha":       0.10,
    "passive_reserve": 10,
    "tq_skew":         2,
    "tq_l1_thresh":    8,
    "tq_l2_thresh":    15,
}


def _wall_mid(order_depth: OrderDepth) -> float | None:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


def _detect_tq(order_depth: OrderDepth) -> int:
    asks = sorted(order_depth.sell_orders.items())
    bids = sorted(order_depth.buy_orders.items(), reverse=True)
    l1_ask = abs(asks[0][1]) if asks else 0
    l2_ask = sum(abs(v) for _, v in asks[1:])
    l1_bid = bids[0][1] if bids else 0
    l2_bid = sum(v for _, v in bids[1:])
    thr1 = OSMIUM_CFG["tq_l1_thresh"]
    thr2 = OSMIUM_CFG["tq_l2_thresh"]
    if l1_ask <= thr1 and l2_ask >= thr2:
        return +1
    if l1_bid <= thr1 and l2_bid >= thr2:
        return -1
    return 0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        root_base: float | None  = trader_state.get("root_base")
        root_mode: str           = trader_state.get("root_mode", "calibrating")
        osm_ema:   float | None  = trader_state.get("osm_ema", None)

        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                root_base = mid - state.timestamp / 1000.0
                root_mode = "long"

        if "ASH_COATED_OSMIUM" in state.order_depths:
            wm = _wall_mid(state.order_depths["ASH_COATED_OSMIUM"])
            if wm is not None:
                alpha = OSMIUM_CFG["ema_alpha"]
                osm_ema = wm if osm_ema is None else alpha * wm + (1 - alpha) * osm_ema

        if root_base is not None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                fair_now    = root_base + state.timestamp / 1000.0
                current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                deviation   = current_mid - fair_now
                if root_mode == "long":
                    if deviation < -ROOT_CFG["dump_threshold"]:
                        root_mode = "short"
                elif root_mode == "short":
                    if deviation > ROOT_CFG["recover_threshold"]:
                        root_mode = "long"

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            if product == "INTARIAN_PEPPER_ROOT":
                if root_base is not None:
                    if root_mode == "long":
                        orders = self._trade_root_long(state, order_depth, pos, root_base)
                    elif root_mode == "short":
                        orders = self._trade_root_short(state, order_depth, pos, root_base)

            elif product == "ASH_COATED_OSMIUM":
                if tick >= OSMIUM_EOD:
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)
                    if pos > 0:
                        orders.append(Order(product, best_bid, -pos))
                    elif pos < 0:
                        orders.append(Order(product, best_ask, -pos))
                else:
                    orders = self._trade_osmium(order_depth, pos, osm_ema)

            result[product] = orders

        new_trader_data = json.dumps({"root_base": root_base, "root_mode": root_mode, "osm_ema": osm_ema})
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data

    def _trade_root_long(self, state: TradingState, order_depth: OrderDepth, pos: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        product  = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = POSITION_LIMIT + pos

        acceptable_buy = math.floor(fair + ROOT_CFG["aggressive_buy_offset"])
        for ask_px in sorted(order_depth.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask_px <= acceptable_buy:
                qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                orders.append(Order(product, ask_px, qty))
                buy_cap -= qty
                pos     += qty
            else:
                break

        if buy_cap > 0 and pos < ROOT_CFG["target_position"]:
            our_bid = math.floor(fair - ROOT_CFG["passive_bid_offset"])
            our_bid = min(our_bid, best_bid + 1)
            our_bid = min(our_bid, best_ask - 1)
            if our_bid < best_ask:
                qty = min(ROOT_CFG["target_position"] - pos, buy_cap)
                if qty > 0:
                    orders.append(Order(product, our_bid, qty))

        expensive_sell = math.ceil(fair + ROOT_CFG["sell_offset"])
        for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_cap <= 0:
                break
            if bid_px >= expensive_sell:
                qty = min(order_depth.buy_orders[bid_px], sell_cap)
                orders.append(Order(product, bid_px, -qty))
                sell_cap -= qty
            else:
                break

        return orders

    def _trade_root_short(self, state: TradingState, order_depth: OrderDepth, pos: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        product  = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        target   = -ROOT_CFG["target_position"]
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = POSITION_LIMIT + pos

        acceptable_sell = math.ceil(fair - ROOT_CFG["aggressive_sell_offset"])
        for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_cap <= 0:
                break
            if bid_px >= acceptable_sell:
                qty = min(order_depth.buy_orders[bid_px], sell_cap)
                orders.append(Order(product, bid_px, -qty))
                sell_cap -= qty
                pos      -= qty
            else:
                break

        if sell_cap > 0 and pos > target:
            our_ask = math.ceil(fair + ROOT_CFG["passive_bid_offset"])
            our_ask = max(our_ask, best_ask - 1)
            our_ask = max(our_ask, best_bid + 1)
            if our_ask > best_bid:
                qty = min(pos - target, sell_cap)
                if qty > 0:
                    orders.append(Order(product, our_ask, -qty))

        cheap_buy = math.floor(fair - ROOT_CFG["buy_offset"])
        for ask_px in sorted(order_depth.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask_px <= cheap_buy:
                qty = min(-order_depth.sell_orders[ask_px], buy_cap)
                orders.append(Order(product, ask_px, qty))
                buy_cap -= qty
            else:
                break

        return orders

    def _trade_osmium(self, order_depth: OrderDepth, pos: int, osm_ema: float | None) -> List[Order]:
        orders  = []
        product = "ASH_COATED_OSMIUM"

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)

        fv = osm_ema if osm_ema is not None else _wall_mid(order_depth)
        if fv is None:
            return orders

        tq        = _detect_tq(order_depth)
        inv_ratio = pos / POSITION_LIMIT
        skew      = -round(inv_ratio * 2)
        fv_adj    = fv + (tq * OSMIUM_CFG["tq_skew"] if tq == 1 else 0)

        rem_buy  = POSITION_LIMIT - pos
        rem_sell = POSITION_LIMIT + pos

        inv_limit = OSMIUM_CFG["inv_limit"]
        base_size = OSMIUM_CFG["order_size"]

        bid_size = base_size
        if pos > inv_limit:
            scale    = 1.0 - (pos - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            bid_size = max(1, round(base_size * scale))

        sell_size = base_size
        if pos < -inv_limit:
            scale     = 1.0 - ((-pos) - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            sell_size = max(1, round(base_size * scale))

        make_width = OSMIUM_CFG["make_width"]

        # Passive bid
        our_bid = round(fv_adj) + skew - make_width
        our_bid = min(our_bid, best_ask - 1)
        if our_bid > best_bid and rem_buy > 0 and our_bid < fv_adj:
            qty = min(bid_size, rem_buy)
            orders.append(Order(product, our_bid, qty))
            rem_buy -= qty

        # Passive ask — fills when spread tightens or price spikes up
        our_ask = round(fv_adj) - skew + make_width
        our_ask = max(our_ask, best_bid + 1)
        if our_ask < best_ask and rem_sell > 0 and our_ask > fv_adj:
            qty = min(sell_size, rem_sell)
            orders.append(Order(product, our_ask, -qty))
            rem_sell -= qty

        # TQ=+1: take aggressively when thin ask wall predicts upward move
        take_buy = max(0, rem_buy - OSMIUM_CFG["passive_reserve"])
        if tq == +1 and take_buy > 0:
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if take_buy <= 0:
                    break
                if ask_px <= fv_adj - OSMIUM_CFG["take_width"]:
                    qty = min(-order_depth.sell_orders[ask_px], take_buy)
                    orders.append(Order(product, ask_px, qty))
                    take_buy -= qty
                else:
                    break

        return orders