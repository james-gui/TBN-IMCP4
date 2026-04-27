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

# ── Root configuration ──────────────────────────────────────────────
ROOT_CFG = {
    "aggressive_buy_offset": 8,
    "aggressive_sell_offset": 8,
    "passive_bid_offset":    0,
    "sell_offset":           20,
    "buy_offset":            20,
    "target_position":       80,
    # ── Deviation stop-loss ──
    "dump_threshold":        15,
    "recover_threshold":     8,
}

# ── Osmium configuration (v13 — spread-aware FV-taking) ──────────────
# Key finding: tight spreads (5-7) have 70-100% fill rate but FV-taking
# during them is ADVERSE because EMA lags the book move that caused
# the spread to tighten. Disabling FV-takes during tight spreads
# eliminates ~300 PnL of losses from v11's 688 → projected ~1000.
OSMIUM_CFG = {
    "take_width":       1,     # moderate taking
    "take_min_spread":  12,    # NEW: only FV-take when spread >= 12

    "make_width":       2,     # penny-jump primary
    "order_size":       30,

    "inventory_limit":  30,
    "inventory_hard":   50,
    "ema_alpha":        0.20,

    "reduce_threshold": 18,
    "reduce_size":      12,

    "inventory_skew":   3.0,

    "eod_start":        85_000,
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
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        # ── Root state ──
        root_base: float | None     = trader_state.get("root_base")
        root_mode: str               = trader_state.get("root_mode", "calibrating")

        # ── Osmium state ──
        ema_state: Dict[str, float]  = trader_state.get("ema", {})

        # ── Root calibration (instant, single mid) ──
        if root_base is None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                root_base = mid - state.timestamp / 1000.0
                root_mode = "long"

        # ── Drift detection & mode switching ──
        if root_base is not None and "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od = state.order_depths["INTARIAN_PEPPER_ROOT"]
            if od.buy_orders and od.sell_orders:
                fair_now = root_base + state.timestamp / 1000.0
                current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
                deviation = current_mid - fair_now

                if root_mode == "long":
                    if deviation < -ROOT_CFG["dump_threshold"]:
                        root_mode = "short"
                elif root_mode == "short":
                    if deviation > ROOT_CFG["recover_threshold"]:
                        root_mode = "long"

        # ── Generate orders ──
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
                    orders = self._trade_osmium(order_depth, pos, ema_state, tick)

            result[product] = orders

        new_trader_data = json.dumps({
            "root_base": root_base,
            "root_mode": root_mode,
            "ema": ema_state,
        })
        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data

    # ── Root: Long mode ──────────────────────────────────────────────
    def _trade_root_long(self, state: TradingState, order_depth: OrderDepth, pos: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        product = "INTARIAN_PEPPER_ROOT"
        fair = root_base + state.timestamp / 1000.0
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

    # ── Root: Short mode ─────────────────────────────────────────────
    def _trade_root_short(self, state: TradingState, order_depth: OrderDepth, pos: int, root_base: float) -> List[Order]:
        orders: List[Order] = []
        product = "INTARIAN_PEPPER_ROOT"
        fair = root_base + state.timestamp / 1000.0
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        target = -ROOT_CFG["target_position"]
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

    # ── Osmium: Market-making v13 (spread-aware FV-taking) ─────────────
    def _trade_osmium(self, order_depth: OrderDepth, pos: int, ema_state: Dict[str, float],
                      tick: int = 0) -> List[Order]:
        orders: List[Order] = []
        product = "ASH_COATED_OSMIUM"

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        spread   = best_ask - best_bid

        # ── Fair value: fast EMA + inventory skew ────────────────────
        vwm   = volume_weighted_mid(order_depth)
        alpha = OSMIUM_CFG["ema_alpha"]
        if product not in ema_state:
            ema_state[product] = vwm
        ema_state[product] = alpha * vwm + (1 - alpha) * ema_state[product]
        raw_fv = ema_state[product]

        inv_skew = OSMIUM_CFG["inventory_skew"]
        fv = raw_fv - inv_skew * (pos / POSITION_LIMIT)
        fv_round = round(fv)

        take_width    = OSMIUM_CFG["take_width"]
        make_width    = OSMIUM_CFG["make_width"]
        base_size     = OSMIUM_CFG["order_size"]
        inv_limit     = OSMIUM_CFG["inventory_limit"]
        inv_hard      = OSMIUM_CFG["inventory_hard"]
        reduce_thresh = OSMIUM_CFG["reduce_threshold"]
        reduce_size   = OSMIUM_CFG["reduce_size"]

        remaining_buy  = POSITION_LIMIT - pos
        remaining_sell = POSITION_LIMIT + pos

        # ── EOD gradual wind-down ────────────────────────────────────
        eod_start = OSMIUM_CFG["eod_start"]
        if tick >= eod_start:
            eod_progress = min(1.0, (tick - eod_start) / (OSMIUM_EOD - eod_start))
            eod_target = max(0, int(inv_hard * (1 - eod_progress)))
            inv_limit = min(inv_limit, eod_target)
            inv_hard  = min(inv_hard, eod_target + 5)
            reduce_thresh = min(reduce_thresh, max(0, eod_target - 3))

        # ── FV taking (ONLY during wide spreads) ─────────────────────
        # During tight spreads (<=10), EMA lags the book move that caused
        # the spread to tighten. FV-taking then = buying at ask / selling
        # at bid with negative edge. Measured loss: ~300 PnL in v11.
        take_min_spread = OSMIUM_CFG["take_min_spread"]
        if spread >= take_min_spread:
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px < fv - take_width:
                    qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px > fv + take_width:
                    qty = min(order_depth.buy_orders[bid_px], remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                else:
                    break

        # ── Position reduction at fair value ─────────────────────────
        if pos > reduce_thresh and remaining_sell > 0:
            reduce_px = max(fv_round, best_bid + 1)
            reduce_px = min(reduce_px, best_ask)
            if reduce_px > best_bid:
                qty = min(reduce_size, pos - reduce_thresh, remaining_sell)
                if qty > 0:
                    orders.append(Order(product, reduce_px, -qty))
                    remaining_sell -= qty

        elif pos < -reduce_thresh and remaining_buy > 0:
            reduce_px = min(fv_round, best_ask - 1)
            reduce_px = max(reduce_px, best_bid)
            if reduce_px < best_ask:
                qty = min(reduce_size, -pos - reduce_thresh, remaining_buy)
                if qty > 0:
                    orders.append(Order(product, reduce_px, qty))
                    remaining_buy -= qty

        # ── Penny-jump passive quotes ────────────────────────────────
        our_bid = best_bid + 1
        our_ask = best_ask - 1

        our_bid = min(our_bid, fv_round - make_width)
        our_ask = max(our_ask, fv_round + make_width)

        our_ask = max(our_ask, best_bid + 1)
        our_bid = min(our_bid, best_ask - 1)

        if our_bid >= our_ask:
            return orders

        # Inventory-skewed sizing
        pos_ratio = pos / POSITION_LIMIT
        skew = pos_ratio ** 3
        bid_size = max(1, round(base_size * (1 - abs(skew) if pos > 0 else 1)))
        ask_size = max(1, round(base_size * (1 - abs(skew) if pos < 0 else 1)))

        if pos > inv_limit:
            scale    = max(0, 1 - (pos - inv_limit) / max(1, inv_hard - inv_limit))
            bid_size = max(1, round(bid_size * scale))
        elif pos < -inv_limit:
            scale    = max(0, 1 - ((-pos) - inv_limit) / max(1, inv_hard - inv_limit))
            ask_size = max(1, round(ask_size * scale))

        want_bid = pos < inv_hard
        want_ask = pos > -inv_hard

        if want_bid and our_bid < fv and remaining_buy > 0:
            orders.append(Order(product, our_bid, min(bid_size, remaining_buy)))
        if want_ask and our_ask > fv and remaining_sell > 0:
            orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))

        return orders