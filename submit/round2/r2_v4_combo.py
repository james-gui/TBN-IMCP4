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
    "make_width":  2,
    "take_width":  1,
    "order_size":  20,
    "inv_limit":   30,
    # EMA alpha reduced to 0.10 — the wild swings were caused by α=0.30
    # responding fully to single-trade price spikes (e.g. 9991→10016 in one tick).
    # At α=0.10 the EMA needs ~22 ticks to move halfway to a new level,
    # smoothing out the outlier trades while still tracking the mean.
    "ema_alpha":   0.10,
}


def wall_mid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        # ── Root state ─────────────────────────────────────────────────────────
        root_base:        float | None = ts.get("root_base")
        root_mode:        str          = ts.get("root_mode", "calibrating")
        root_burst_px:    int | None   = ts.get("root_burst_px",    None)
        root_burst_qty:   int | None   = ts.get("root_burst_qty",   None)
        root_burst_count: int          = ts.get("root_burst_count", 0)

        # ── Osmium state ───────────────────────────────────────────────────────
        osm_ema: float | None = ts.get("osm_ema", None)

        # ══════════════════════════════════════════════════════════════════════
        # ROOT — state updates
        # ══════════════════════════════════════════════════════════════════════

        # Calibrate linear model
        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                root_base = mid - state.timestamp / 1000.0
                root_mode = "long"

        # Dump/recover mode switching on linear deviation
        if root_base is not None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                fair_now    = root_base + state.timestamp / 1000.0
                current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                deviation   = current_mid - fair_now
                if root_mode == "long" and deviation < -ROOT_CFG["dump_threshold"]:
                    root_mode = "short"
                elif root_mode == "short" and deviation > ROOT_CFG["recover_threshold"]:
                    root_mode = "long"

        # Burst detector
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

        # ══════════════════════════════════════════════════════════════════════
        # OSMIUM — state update
        # FIX: EMA now uses wall_mid from the order book, NOT market trade prices.
        #
        # The α=0.30 trade-price EMA was swinging 9991↔10016 each tick because
        # market_trades shows only 1 trade per tick (the staircase bot's fill),
        # and that single price jumps wildly as the bot cycles through its levels.
        # Each spike fully moved the EMA by 30%, flipping our fair value and
        # causing the take loop to cross the spread in the wrong direction.
        #
        # Wall_mid is anchored to the largest resting orders on each side —
        # stable, not subject to single-trade spikes, and confirmed by our
        # earlier analysis to track the true fair value (99.8% of ticks L2>L1).
        # Using α=0.10 on wall_mid gives a smooth, stable fair value estimate.
        # ══════════════════════════════════════════════════════════════════════
        if "ASH_COATED_OSMIUM" in state.order_depths:
            od = state.order_depths["ASH_COATED_OSMIUM"]
            wm = wall_mid(od)
            if wm is not None:
                alpha = OSM_CFG["ema_alpha"]
                osm_ema = wm if osm_ema is None else alpha * wm + (1 - alpha) * osm_ema

        # ══════════════════════════════════════════════════════════════════════
        # ROOT — trading
        # ══════════════════════════════════════════════════════════════════════
        if "INTARIAN_PEPPER_ROOT" in state.order_depths and root_base is not None:
            od  = state.order_depths["INTARIAN_PEPPER_ROOT"]
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            if od.buy_orders and od.sell_orders:
                if root_mode == "long":
                    result["INTARIAN_PEPPER_ROOT"] = self._root_long(
                        state, od, pos, root_base, floor_price)
                elif root_mode == "short":
                    result["INTARIAN_PEPPER_ROOT"] = self._root_short(
                        state, od, pos, root_base)
                else:
                    result["INTARIAN_PEPPER_ROOT"] = []
            else:
                result["INTARIAN_PEPPER_ROOT"] = []

        # ══════════════════════════════════════════════════════════════════════
        # OSMIUM — trading
        # ══════════════════════════════════════════════════════════════════════
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

        # ── Serialize ──────────────────────────────────────────────────────────
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

    def _root_long(self, state, od, pos, root_base, floor_price):
        orders  = []
        product = "INTARIAN_PEPPER_ROOT"
        fair     = root_base + state.timestamp / 1000.0
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        buy_cap  = POSITION_LIMIT - pos
        sell_cap = POSITION_LIMIT + pos

        # Take: use tighter of burst floor or linear fair+8
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

        # Passive bid
        if buy_cap > 0 and pos < ROOT_CFG["target_position"]:
            our_bid = math.floor(fair - ROOT_CFG["passive_bid_offset"])
            our_bid = min(our_bid, best_bid + 1)
            our_bid = min(our_bid, best_ask - 1)
            if our_bid < best_ask:
                qty = min(ROOT_CFG["target_position"] - pos, buy_cap)
                if qty > 0:
                    orders.append(Order(product, our_bid, qty))

        # Opportunistic sell at extreme premium
        expensive_sell = math.ceil(fair + ROOT_CFG["sell_offset"])
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if sell_cap <= 0: break
            if bid_px >= expensive_sell:
                qty = min(od.buy_orders[bid_px], sell_cap)
                orders.append(Order(product, bid_px, -qty))
                sell_cap -= qty
            else:
                break

        return orders

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

    def _trade_osmium(self, od, pos, osm_ema):
        orders  = []
        product = "ASH_COATED_OSMIUM"

        fv = osm_ema if osm_ema is not None else wall_mid(od)
        if fv is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        remaining_buy  = POSITION_LIMIT - pos
        remaining_sell = POSITION_LIMIT + pos

        # Take
        for ask_px in sorted(od.sell_orders.keys()):
            if remaining_buy <= 0: break
            if ask_px <= fv - OSM_CFG["take_width"]:
                qty = min(-od.sell_orders[ask_px], remaining_buy)
                orders.append(Order(product, ask_px, qty))
                remaining_buy -= qty
            else:
                break

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if remaining_sell <= 0: break
            if bid_px >= fv + OSM_CFG["take_width"]:
                qty = min(od.buy_orders[bid_px], remaining_sell)
                orders.append(Order(product, bid_px, -qty))
                remaining_sell -= qty
            else:
                break

        # Make
        inv_ratio = pos / POSITION_LIMIT
        skew      = -round(inv_ratio * 2)

        our_bid = round(fv) + skew - OSM_CFG["make_width"]
        our_ask = round(fv) + skew + OSM_CFG["make_width"]
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if our_bid >= our_ask:
            return orders

        base      = OSM_CFG["order_size"]
        inv_limit = OSM_CFG["inv_limit"]
        bid_size  = base
        ask_size  = base

        if pos > inv_limit:
            scale    = 1.0 - (pos - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            bid_size = max(1, round(base * scale))
        elif pos < -inv_limit:
            scale    = 1.0 - ((-pos) - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            ask_size = max(1, round(base * scale))

        if remaining_buy > 0 and our_bid < fv:
            orders.append(Order(product, our_bid,  min(bid_size, remaining_buy)))
        if remaining_sell > 0 and our_ask > fv:
            orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))

        return orders