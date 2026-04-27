import json
import math
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

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
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

    def compress_orders(self, orders):
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
    "aggressive_buy_offset":  8,
    "aggressive_sell_offset": 8,
    "passive_bid_offset":     0,
    "sell_offset":            20,
    "buy_offset":             20,
    "target_position":        80,
    "dump_threshold":         15,
    "recover_threshold":      8,
    "burst_min_len":          3,
    "burst_take_offset":      1,
}

OSM_CFG = {
    "make_width":       2,
    "take_width":       1,
    "order_size":       20,
    "inv_limit":        30,
    "ema_alpha":        0.10,
    # How many units to RESERVE for passive maker fills regardless of take activity.
    # Ensures passive orders always have capacity → earns maker fee volume.
    "passive_reserve":  10,
    # L3 insider signal: shift quotes by this many ticks when detected.
    "l3_skew":          2,
}


def wall_mid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


def detect_l3(order_depth: OrderDepth) -> int:
    """
    L3 insider signal — 96-100% directional accuracy, ~47 appearances per round.
    A small-volume probe quote appears as the innermost level of a 3-level book.
    Returns +1 (price going up), -1 (price going down), or 0 (no signal).
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0

    all_asks = sorted(order_depth.sell_orders.items())   # ascending price
    all_bids = sorted(order_depth.buy_orders.items(), reverse=True)  # descending price

    if len(all_asks) >= 3:
        inner_vol = abs(all_asks[0][1])
        l2_vol    = abs(all_asks[1][1])
        if inner_vol < l2_vol:
            return +1   # small ask probe → price going UP

    if len(all_bids) >= 3:
        inner_vol = all_bids[0][1]
        l2_vol    = all_bids[1][1]
        if inner_vol < l2_vol:
            return -1   # small bid probe → price going DOWN

    return 0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        root_base:        float | None = ts.get("root_base")
        root_mode:        str          = ts.get("root_mode", "calibrating")
        root_burst_px:    int | None   = ts.get("root_burst_px",    None)
        root_burst_qty:   int | None   = ts.get("root_burst_qty",   None)
        root_burst_count: int          = ts.get("root_burst_count", 0)
        osm_ema:          float | None = ts.get("osm_ema", None)

        # ── Root state updates ─────────────────────────────────────────────────
        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                root_base = mid - state.timestamp / 1000.0
                root_mode = "long"

        if root_base is not None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                fair_now    = root_base + state.timestamp / 1000.0
                current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                deviation   = current_mid - fair_now
                if root_mode == "long"  and deviation < -ROOT_CFG["dump_threshold"]:
                    root_mode = "short"
                elif root_mode == "short" and deviation > ROOT_CFG["recover_threshold"]:
                    root_mode = "long"

        for t in state.market_trades.get("INTARIAN_PEPPER_ROOT", []):
            qty = abs(t.quantity)
            px  = round(t.price)
            if qty == root_burst_qty and px == root_burst_px:
                root_burst_count += 1
            else:
                root_burst_qty   = qty
                root_burst_px    = px
                root_burst_count = 1

        burst_confirmed = root_burst_count >= ROOT_CFG["burst_min_len"]
        floor_price     = root_burst_px if burst_confirmed else None

        # ── Osmium EMA update ──────────────────────────────────────────────────
        if "ASH_COATED_OSMIUM" in state.order_depths:
            od = state.order_depths["ASH_COATED_OSMIUM"]
            wm = wall_mid(od)
            if wm is not None:
                alpha = OSM_CFG["ema_alpha"]
                osm_ema = wm if osm_ema is None else alpha * wm + (1 - alpha) * osm_ema

        # ── Root trading ───────────────────────────────────────────────────────
        if "INTARIAN_PEPPER_ROOT" in state.order_depths and root_base is not None:
            od  = state.order_depths["INTARIAN_PEPPER_ROOT"]
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            if od.buy_orders and od.sell_orders:
                if root_mode == "long":
                    result["INTARIAN_PEPPER_ROOT"] = self._root_long(state, od, pos, root_base, floor_price)
                elif root_mode == "short":
                    result["INTARIAN_PEPPER_ROOT"] = self._root_short(state, od, pos, root_base)
                else:
                    result["INTARIAN_PEPPER_ROOT"] = []
            else:
                result["INTARIAN_PEPPER_ROOT"] = []

        # ── Osmium trading ─────────────────────────────────────────────────────
        if "ASH_COATED_OSMIUM" in state.order_depths:
            od  = state.order_depths["ASH_COATED_OSMIUM"]
            pos = state.position.get("ASH_COATED_OSMIUM", 0)
            if tick >= OSMIUM_EOD:
                orders = []
                if od.buy_orders and od.sell_orders:
                    best_bid = max(od.buy_orders)
                    best_ask = min(od.sell_orders)
                    if pos > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", best_bid, -pos))
                    elif pos < 0:
                        orders.append(Order("ASH_COATED_OSMIUM", best_ask, -pos))
                result["ASH_COATED_OSMIUM"] = orders
            else:
                result["ASH_COATED_OSMIUM"] = self._trade_osmium(od, pos, osm_ema)

        new_trader_data = json.dumps({
            "root_base":        root_base,
            "root_mode":        root_mode,
            "root_burst_px":    root_burst_px,
            "root_burst_qty":   root_burst_qty,
            "root_burst_count": root_burst_count,
            "osm_ema":          osm_ema,
        })
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data

    # ── Root long ──────────────────────────────────────────────────────────────
    def _root_long(self, state, od, pos, root_base, floor_price):
        orders  = []
        product = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = POSITION_LIMIT + pos

        # PRIORITY ORDER (changed from v5):
        # 1. Opportunistic SELL first — if price has spiked to fair+20, sell NOW
        #    before consuming any buy capacity. In v5 this came last, meaning if
        #    buy_cap was exhausted by takes, sell never fired even at a spike.
        expensive_sell = math.ceil(fair + ROOT_CFG["sell_offset"])
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if sell_cap <= 0: break
            if bid_px >= expensive_sell:
                qty = min(od.buy_orders[bid_px], sell_cap)
                orders.append(Order(product, bid_px, -qty))
                sell_cap -= qty
            else:
                break

        # 2. Aggressive take — consume cheap asks up to take_limit
        if floor_price is not None:
            take_limit = floor_price + ROOT_CFG["burst_take_offset"]
        else:
            take_limit = math.floor(fair + ROOT_CFG["aggressive_buy_offset"])

        for ask_px in sorted(od.sell_orders.keys()):
            if buy_cap <= 0: break
            if ask_px <= take_limit:
                qty = min(-od.sell_orders[ask_px], buy_cap)
                orders.append(Order(product, ask_px, qty))
                buy_cap -= qty; pos += qty
            else:
                break

        # 3. Passive bid with remaining capacity
        if buy_cap > 0 and pos < ROOT_CFG["target_position"]:
            our_bid = math.floor(fair - ROOT_CFG["passive_bid_offset"])
            our_bid = min(our_bid, best_bid + 1)
            our_bid = min(our_bid, best_ask - 1)
            if our_bid < best_ask:
                qty = min(ROOT_CFG["target_position"] - pos, buy_cap)
                if qty > 0:
                    orders.append(Order(product, our_bid, qty))

        return orders

    # ── Root short (unchanged) ─────────────────────────────────────────────────
    def _root_short(self, state, od, pos, root_base):
        orders  = []
        product = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        target   = -ROOT_CFG["target_position"]
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = POSITION_LIMIT + pos

        acceptable_sell = math.ceil(fair - ROOT_CFG["aggressive_sell_offset"])
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if sell_cap <= 0: break
            if bid_px >= acceptable_sell:
                qty = min(od.buy_orders[bid_px], sell_cap)
                orders.append(Order(product, bid_px, -qty))
                sell_cap -= qty; pos -= qty
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
        for ask_px in sorted(od.sell_orders.keys()):
            if buy_cap <= 0: break
            if ask_px <= cheap_buy:
                qty = min(-od.sell_orders[ask_px], buy_cap)
                orders.append(Order(product, ask_px, qty))
                buy_cap -= qty
            else:
                break

        return orders

    # ── Osmium ─────────────────────────────────────────────────────────────────
    def _trade_osmium(self, od, pos, osm_ema):
        orders  = []
        product = "ASH_COATED_OSMIUM"

        fv = osm_ema if osm_ema is not None else wall_mid(od)
        if fv is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        # Detect L3 insider signal before computing quotes
        l3 = detect_l3(od)

        # Shift fair value and quotes in direction of L3 signal
        fv_adjusted = fv + l3 * OSM_CFG["l3_skew"]

        inv_ratio = pos / POSITION_LIMIT
        skew      = -round(inv_ratio * 2)

        our_bid = round(fv_adjusted) + skew - OSM_CFG["make_width"]
        our_ask = round(fv_adjusted) + skew + OSM_CFG["make_width"]
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if our_bid >= our_ask:
            return orders

        base      = OSM_CFG["order_size"]
        inv_limit = OSM_CFG["inv_limit"]
        passive_reserve = OSM_CFG["passive_reserve"]

        bid_size = base
        ask_size = base
        if pos > inv_limit:
            scale    = 1.0 - (pos - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            bid_size = max(1, round(base * scale))
        elif pos < -inv_limit:
            scale    = 1.0 - ((-pos) - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            ask_size = max(1, round(base * scale))

        # PRIORITY ORDER (key change from v5):
        #
        # NEW: Passive maker orders are submitted FIRST.
        # Reason: the exchange fills orders in submission order at a given price.
        # By posting our passive bid/ask first, we get queue priority over other
        # bots that may also be quoting at the same level. This directly increases
        # maker fill rate → more maker fee volume → higher chance of hitting the
        # top-50% threshold for the +25% position limit bonus.
        #
        # We reserve `passive_reserve` units of capacity for passive fills regardless
        # of what the take loop does. This guarantees passive orders are never
        # crowded out by our own take activity.

        remaining_buy  = POSITION_LIMIT - pos
        remaining_sell = POSITION_LIMIT + pos

        # Reserve capacity for passive before take loop runs
        passive_buy_cap  = min(bid_size, max(0, remaining_buy))
        passive_sell_cap = min(ask_size, max(0, remaining_sell))

        # 1. PASSIVE ORDERS FIRST — queue priority for maker fees
        if remaining_buy > 0 and our_bid < fv_adjusted:
            orders.append(Order(product, our_bid,  min(bid_size, remaining_buy)))
            remaining_buy -= min(bid_size, remaining_buy)

        if remaining_sell > 0 and our_ask > fv_adjusted:
            orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))
            remaining_sell -= min(ask_size, remaining_sell)

        # 2. TAKE ORDERS SECOND — only use capacity not reserved for passive
        # Cap take capacity so passive reserve is guaranteed
        take_buy_cap  = max(0, (POSITION_LIMIT - pos) - passive_reserve)
        take_sell_cap = max(0, (POSITION_LIMIT + pos) - passive_reserve)

        # L3 gates: only take in the direction the signal confirms
        # (if l3=0, no taking — wait for signal)
        if l3 == +1 and take_buy_cap > 0:
            for ask_px in sorted(od.sell_orders.keys()):
                if take_buy_cap <= 0: break
                if ask_px <= fv_adjusted - OSM_CFG["take_width"]:
                    qty = min(-od.sell_orders[ask_px], take_buy_cap)
                    orders.append(Order(product, ask_px, qty))
                    take_buy_cap -= qty
                else:
                    break

        elif l3 == -1 and take_sell_cap > 0:
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if take_sell_cap <= 0: break
                if bid_px >= fv_adjusted + OSM_CFG["take_width"]:
                    qty = min(od.buy_orders[bid_px], take_sell_cap)
                    orders.append(Order(product, bid_px, -qty))
                    take_sell_cap -= qty
                else:
                    break

        return orders