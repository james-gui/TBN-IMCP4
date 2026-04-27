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

# ── Osmium config ──────────────────────────────────────────────────────────────
OSM_CFG = {
    "make_width":  2,    # post at fv ± 2
    "take_width":  1,    # lift/hit if price is within 1 tick of fv
    "order_size":  20,
    "inv_limit":   30,   # start skewing quotes above this abs(pos)
    # BUG FIX 1: removed inv_hard cap that was blocking all asks when pos > 60
    # BUG FIX 2: use trade-price EMA instead of wall_mid EMA to avoid initial lag
    "ema_alpha":   0.30, # faster alpha on trade price — converges in ~10 trades
}

# ── Root config ────────────────────────────────────────────────────────────────
ROOT_CFG = {
    "burst_min_len": 3,  # ticks of same qty+px before floor confirmed
    # BUG FIX 3: removed the passive bid at burst_px+1 which was getting
    # overtaken by best_bid and blocked. Now we take aggressively at burst_px
    # since burst_px IS the fair value — we should pay up to get filled fast.
    "take_limit_offset": 1,  # take any ask <= burst_px + 1
}


def wall_mid(order_depth: OrderDepth) -> Optional[float]:
    """Midpoint of the largest-volume bid and ask levels."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        # ── Root state ─────────────────────────────────────────────────────────
        root_burst_px    = ts.get("root_burst_px",    None)
        root_burst_qty   = ts.get("root_burst_qty",   None)
        root_burst_count = ts.get("root_burst_count", 0)

        # ── Osmium state ───────────────────────────────────────────────────────
        # BUG FIX 2: EMA now tracks actual market trade prices, not wall_mid
        # wall_mid at round start = (9998+10019)/2 = 10008.5, but market trades
        # at ~10004. This caused our ask to sit at ~10010 for thousands of ticks,
        # generating zero sell fills and a stuck long position.
        osm_trade_ema = ts.get("osm_trade_ema", None)

        # ── Update Root burst detector ─────────────────────────────────────────
        root_trades: List[Trade] = state.market_trades.get("INTARIAN_PEPPER_ROOT", [])
        for t in root_trades:
            qty = abs(t.quantity)
            px  = round(t.price)
            if qty == root_burst_qty and px == root_burst_px:
                root_burst_count += 1
            else:
                root_burst_qty   = qty
                root_burst_px    = px
                root_burst_count = 1

        floor_confirmed = root_burst_count >= ROOT_CFG["burst_min_len"]
        floor_price     = root_burst_px if floor_confirmed else None

        # ── Update Osmium trade EMA ────────────────────────────────────────────
        osm_trades: List[Trade] = state.market_trades.get("ASH_COATED_OSMIUM", [])
        alpha = OSM_CFG["ema_alpha"]
        for t in osm_trades:
            osm_trade_ema = t.price if osm_trade_ema is None else (
                alpha * t.price + (1 - alpha) * osm_trade_ema
            )

        # ── Trade ROOT ─────────────────────────────────────────────────────────
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od  = state.order_depths["INTARIAN_PEPPER_ROOT"]
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            orders: List[Order] = []

            if od.buy_orders and od.sell_orders:
                best_bid = max(od.buy_orders)
                best_ask = min(od.sell_orders)
                buy_cap  = POSITION_LIMIT - pos

                if buy_cap > 0:
                    if floor_price is not None:
                        # BUG FIX 3: The old passive bid at floor_px+1 kept getting
                        # overtaken by best_bid as the book drifted up, leaving our
                        # bid below best_bid with no fills for 20k+ ticks.
                        #
                        # The correct approach: the staircase bot trades at burst_px
                        # which sits ~3 ticks ABOVE best_bid and ~12 ticks BELOW best_ask.
                        # These are aggressive takes happening inside the spread.
                        # We should do the same — take any ask at or below floor_px+1.
                        # This gets us filled immediately at the floor price.
                        take_limit = floor_price + ROOT_CFG["take_limit_offset"]
                        for ask_px in sorted(od.sell_orders.keys()):
                            if buy_cap <= 0:
                                break
                            if ask_px <= take_limit:
                                qty = min(-od.sell_orders[ask_px], buy_cap)
                                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, qty))
                                buy_cap -= qty
                            else:
                                break

                        # Secondary: passive bid at best_bid+1 to catch any residual
                        # capacity after the take sweep (earns maker fee)
                        if buy_cap > 0:
                            our_bid = best_bid + 1
                            if our_bid < best_ask:
                                orders.append(Order("INTARIAN_PEPPER_ROOT", our_bid, buy_cap))

                    else:
                        # No burst confirmed yet — sit at best_bid+1 passively
                        our_bid = best_bid + 1
                        if our_bid < best_ask:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", our_bid, buy_cap))

            result["INTARIAN_PEPPER_ROOT"] = orders

        # ── Trade OSMIUM ───────────────────────────────────────────────────────
        if "ASH_COATED_OSMIUM" in state.order_depths:
            od  = state.order_depths["ASH_COATED_OSMIUM"]
            pos = state.position.get("ASH_COATED_OSMIUM", 0)
            orders = []

            # Fair value: use trade EMA if available, else fall back to wall_mid
            fv = osm_trade_ema
            if fv is None:
                fv = wall_mid(od)
            if fv is None:
                result["ASH_COATED_OSMIUM"] = orders
            else:
                if od.buy_orders and od.sell_orders:
                    best_bid = max(od.buy_orders)
                    best_ask = min(od.sell_orders)

                    remaining_buy  = POSITION_LIMIT - pos
                    remaining_sell = POSITION_LIMIT + pos

                    # ── 1. Take: cross spread only when clearly mispriced ──
                    for ask_px in sorted(od.sell_orders.keys()):
                        if remaining_buy <= 0:
                            break
                        if ask_px <= fv - OSM_CFG["take_width"]:
                            qty = min(-od.sell_orders[ask_px], remaining_buy)
                            orders.append(Order("ASH_COATED_OSMIUM", ask_px, qty))
                            remaining_buy -= qty
                        else:
                            break

                    for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                        if remaining_sell <= 0:
                            break
                        if bid_px >= fv + OSM_CFG["take_width"]:
                            qty = min(od.buy_orders[bid_px], remaining_sell)
                            orders.append(Order("ASH_COATED_OSMIUM", bid_px, -qty))
                            remaining_sell -= qty
                        else:
                            break

                    # ── 2. Make: inventory-skewed quotes around fv ──
                    # Skew shifts both bid and ask against the inventory direction.
                    # BUG FIX 1: removed inv_hard guard which was suppressing all
                    # asks when pos > 60. Now we always post both sides unless
                    # we've hit the hard position limit (±80).
                    inv_ratio = pos / POSITION_LIMIT   # [-1, 1]
                    skew      = -round(inv_ratio * 2)  # [-2, +2] ticks

                    our_bid = round(fv) + skew - OSM_CFG["make_width"]
                    our_ask = round(fv) + skew + OSM_CFG["make_width"]

                    # Clamp inside spread
                    our_bid = min(our_bid, best_ask - 1)
                    our_ask = max(our_ask, best_bid + 1)

                    if our_bid >= our_ask:
                        result["ASH_COATED_OSMIUM"] = orders
                    else:
                        base = OSM_CFG["order_size"]
                        inv_limit = OSM_CFG["inv_limit"]

                        # Scale size down linearly from inv_limit to POSITION_LIMIT
                        bid_size = base
                        ask_size = base
                        if pos > inv_limit:
                            scale    = 1.0 - (pos - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
                            bid_size = max(1, round(base * scale))
                        elif pos < -inv_limit:
                            scale    = 1.0 - ((-pos) - inv_limit) / max(POSITION_LIMIT - inv_limit, 1)
                            ask_size = max(1, round(base * scale))

                        # Post bid only if below fv (don't buy above fair value)
                        if remaining_buy > 0 and our_bid < fv:
                            orders.append(Order("ASH_COATED_OSMIUM", our_bid,
                                                min(bid_size, remaining_buy)))

                        # Post ask only if above fv (don't sell below fair value)
                        # BUG FIX 1: no inv_hard guard here — always try to post ask
                        # when we have long inventory to reduce, as long as ask > fv
                        if remaining_sell > 0 and our_ask > fv:
                            orders.append(Order("ASH_COATED_OSMIUM", our_ask,
                                                -min(ask_size, remaining_sell)))

            result["ASH_COATED_OSMIUM"] = orders

        # ── Serialize ──────────────────────────────────────────────────────────
        new_trader_data = json.dumps({
            "root_burst_px":    root_burst_px,
            "root_burst_qty":   root_burst_qty,
            "root_burst_count": root_burst_count,
            "osm_trade_ema":    osm_trade_ema,
        })

        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data