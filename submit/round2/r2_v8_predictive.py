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
    # make_width=1: grid search shows 15 fills at mw=1 vs 6 at mw=2 per round
    # Total edge: 623 vs 294 — mw=1 is 2x better despite lower per-fill edge
    "make_width":       1,
    # take_width=0: captures marginal spikes at exactly fv (6 extra fills/round)
    "take_width":       0,
    "order_size":       20,
    "inv_limit":        30,
    "ema_alpha":        0.10,
    "passive_reserve":  10,
    # Trapped quote signal: shift quotes by this many ticks when detected.
    # 96% accuracy, ~57 appearances per round, avg move +4.3 ticks.
    "tq_skew":          2,
    # Thresholds calibrated from data: L1≤8, L2≥15 → 96% accuracy
    "tq_l1_thresh":     8,
    "tq_l2_thresh":     15,
}


def wall_mid(order_depth: OrderDepth) -> Optional[float]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


def detect_trapped_quote(order_depth: OrderDepth) -> int:
    """
    Trapped quote signal — 96% directional accuracy, ~70 appearances per round.

    Fires when L1 is thin (≤8 units) while L2+ is thick (≥15 units).
    Structural meaning: the innermost quote is nearly depleted while a large
    wall sits behind it. Once L1 is lifted, price jumps to the wall (~4 ticks).

    This supersedes the earlier L3 probe signal — it catches all L3 events
    plus ~20 additional pure-volume cases at the same accuracy level.

    Calibrated from data: L1≤8 & L2≥15 → 96% accuracy, avg move +4.31 ticks.

    Returns +1 (price going up), -1 (price going down), or 0 (no signal).
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0

    l1_thresh = OSM_CFG["tq_l1_thresh"]
    l2_thresh = OSM_CFG["tq_l2_thresh"]

    asks = sorted(order_depth.sell_orders.items())          # ascending price
    bids = sorted(order_depth.buy_orders.items(), reverse=True)  # descending price

    l1_ask_vol = abs(asks[0][1]) if asks else 0
    l2_ask_vol = sum(abs(v) for _, v in asks[1:])
    l1_bid_vol = bids[0][1] if bids else 0
    l2_bid_vol = sum(v for _, v in bids[1:])

    # Thin ask wall with heavy support behind → L1 ask about to be lifted → UP
    if l1_ask_vol <= l1_thresh and l2_ask_vol >= l2_thresh:
        return +1

    # Thin bid wall with heavy support behind → L1 bid about to be lifted → DOWN
    if l1_bid_vol <= l1_thresh and l2_bid_vol >= l2_thresh:
        return -1

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

        # Detect trapped quote signal
        tq = detect_trapped_quote(od)

        # Shift fair value in direction of signal
        fv_adjusted = fv + tq * OSM_CFG["tq_skew"]

        inv_ratio = pos / POSITION_LIMIT
        skew      = -round(inv_ratio * 2)

        remaining_buy  = POSITION_LIMIT - pos
        remaining_sell = POSITION_LIMIT + pos

        base      = OSM_CFG["order_size"]
        inv_limit = OSM_CFG["inv_limit"]

        bid_size = base
        if pos > inv_limit:
            scale    = 1.0 - (pos - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            bid_size = max(1, round(base * scale))

        sell_size = base
        if pos < -inv_limit:
            scale     = 1.0 - ((-pos) - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
            sell_size = max(1, round(base * scale))

        # ── 1. PASSIVE BID FIRST (queue priority for maker fees) ──────────────
        # Passive buys work: avg fill is 0.4 ticks below fv — good edge.
        # Post at fv_adjusted - make_width, clamped inside spread.
        our_bid = round(fv_adjusted) + skew - OSM_CFG["make_width"]
        our_bid = min(our_bid, best_ask - 1)

        if our_bid > best_bid and remaining_buy > 0 and our_bid < fv_adjusted:
            orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
            remaining_buy -= min(bid_size, remaining_buy)

        # ── 2. TQ SIGNAL TAKES ───────────────────────────────────────────────
        # MARKET STRUCTURE: spread = 16 ticks → passive asks NEVER fill (0 fills).
        # To sell we must HIT the bid. Only do so when TQ=-1 confirms direction.
        #
        # TQ=+1: lift cheap ask (buy aggressively) — as before
        # TQ=-1: hit bid (sell aggressively) — NEW. Fixes the long accumulation bias.
        #         avg bb-fv = -1.56 ticks, costs edge but reduces inventory risk.

        take_buy_cap  = max(0, remaining_buy  - OSM_CFG["passive_reserve"])
        take_sell_cap = max(0, remaining_sell - OSM_CFG["passive_reserve"])

        if tq == +1 and take_buy_cap > 0:
            for ask_px in sorted(od.sell_orders.keys()):
                if take_buy_cap <= 0: break
                if ask_px <= fv_adjusted - OSM_CFG["take_width"]:
                    qty = min(-od.sell_orders[ask_px], take_buy_cap)
                    orders.append(Order(product, ask_px, qty))
                    take_buy_cap -= qty
                else:
                    break

        elif tq == -1 and take_sell_cap > 0:
            # Hit the bid — only sensible sell mechanism in a 16-tick spread
            qty = min(sell_size, take_sell_cap)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        return orders