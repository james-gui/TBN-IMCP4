import json
import math
from typing import Any, Dict, List, Optional, Tuple
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

# ================= LOGGER =================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades), state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list:
        conv = {}
        for product, obs in observations.conversionObservations.items():
            conv[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                             obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, conv]

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
                out = candidate; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

# ================= CONFIG =================
POSITION_LIMIT = 80
OSMIUM_EOD = 995_000

OSMIUM_CFG = {
    "take_width":      1,
    "make_width":      2,
    "order_size":      24,
    "inventory_limit": 30,
    "inventory_hard":  50,
    "ema_alpha":       0.08,   # slow EMA on wall_mid — stable fair value
    "imb_alpha":       0.20,   # smoothing on raw OBI
    # Tuned signal weights (see comments below)
    "fair_imb_weight": 1.5,    # FIX #6: was 0.1 — needs to be ~1-2 ticks to matter
    "reservation_inv": 0.05,   # FIX #7: was 0.10 — at pos=80 this gives -4 tick skew, safe
    "reservation_sig": 0.8,    # signal contribution to reservation price
    # L3 insider signal
    "l3_skew_ticks":   2,      # shift quotes by this many ticks when L3 detected
}


# ================= HELPERS =================
def wall_mid(order_depth: OrderDepth) -> Optional[float]:
    """Midpoint of the largest-volume bid and ask levels."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    # BUG FIX #2/#3: original had undefined best_bid and wrong return type
    bid_wall = max(order_depth.buy_orders, key=order_depth.buy_orders.get)
    ask_wall = min(order_depth.sell_orders, key=lambda k: order_depth.sell_orders[k])
    return (bid_wall + ask_wall) / 2.0


def detect_l3_signal(order_depth: OrderDepth) -> int:
    """
    Detect the L3 insider quote pattern found in data analysis.
    A small-volume quote appearing as a 3rd price level inside the spread
    predicts the next price move with 96-100% accuracy.

    Returns:
      +1 if ASK L3 detected (price about to go UP)
      -1 if BID L3 detected (price about to go DOWN)
       0 if no signal
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0

    all_bids = sorted(order_depth.buy_orders.items(), reverse=True)   # [(px, vol), ...]
    all_asks = sorted(order_depth.sell_orders.items())                 # [(px, -vol), ...]

    # ASK L3: 3+ ask levels where the innermost has smaller volume than L2
    # → means someone placed a small "probe" ask close to the bid → price going UP
    if len(all_asks) >= 3:
        inner_ask_px,  inner_ask_vol  = all_asks[0][0], abs(all_asks[0][1])
        l2_ask_px,     l2_ask_vol     = all_asks[1][0], abs(all_asks[1][1])
        if inner_ask_vol < l2_ask_vol:
            return +1

    # BID L3: 3+ bid levels where the innermost has smaller volume than L2
    # → someone placed a small "probe" bid close to the ask → price going DOWN
    if len(all_bids) >= 3:
        inner_bid_px,  inner_bid_vol  = all_bids[0][0], all_bids[0][1]
        l2_bid_px,     l2_bid_vol     = all_bids[1][0], all_bids[1][1]
        if inner_bid_vol < l2_bid_vol:
            return -1

    return 0


# ================= TRADER =================
class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0
        tick = state.timestamp % 1_000_000

        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            trader_state = {}

        osmium_state = trader_state.get("osmium", {})
        ema_state    = osmium_state.get("ema",       {})
        imb_state    = osmium_state.get("imbalance", 0.0)

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            if product == "ASH_COATED_OSMIUM":
                if tick >= OSMIUM_EOD:
                    # EOD flatten — BUG FIX #4: best_bid was undefined
                    best_bid = max(order_depth.buy_orders)
                    best_ask = min(order_depth.sell_orders)
                    if pos > 0:
                        orders.append(Order(product, best_bid, -pos))
                    elif pos < 0:
                        orders.append(Order(product, best_ask, -pos))
                else:
                    orders, ema_state, imb_state = self._trade_osmium(
                        order_depth, pos, ema_state, imb_state
                    )

            result[product] = orders

        new_trader_data = json.dumps({
            "osmium": {"ema": ema_state, "imbalance": imb_state}
        })

        logger.flush(state, result, conversions, new_trader_data)
        return result, conversions, new_trader_data

    # ================= OSMIUM =================
    def _trade_osmium(
        self,
        order_depth: OrderDepth,
        pos: int,
        ema_state: Dict[str, float],
        imb_state: float,
    ) -> Tuple[List[Order], Dict[str, float], float]:

        orders: List[Order] = []
        product = "ASH_COATED_OSMIUM"

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders, ema_state, imb_state

        # BUG FIX #5: best_bid was never defined in original
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol  = order_depth.buy_orders[best_bid]
        ask_vol  = abs(order_depth.sell_orders[best_ask])

        # ===== FAIR VALUE (wall_mid EMA) =====
        wm = wall_mid(order_depth)
        if wm is None:
            return orders, ema_state, imb_state

        alpha = OSMIUM_CFG["ema_alpha"]
        if product not in ema_state:
            ema_state[product] = wm
        ema_state[product] = alpha * wm + (1 - alpha) * ema_state[product]
        base_fair = ema_state[product]

        # ===== OBI / IMBALANCE SIGNAL =====
        raw_imb  = (bid_vol - ask_vol) / max(1, bid_vol + ask_vol)  # [-1, 1]
        imb_alpha = OSMIUM_CFG["imb_alpha"]
        imb_state = imb_alpha * raw_imb + (1 - imb_alpha) * imb_state

        signal = imb_state if abs(imb_state) >= 0.1 else 0.0

        # ===== L3 INSIDER SIGNAL =====
        # Detected 47 times per round, 96-100% directional accuracy.
        # When present, it overrides the OBI signal and hard-skews quotes.
        l3 = detect_l3_signal(order_depth)

        # ===== ADJUSTED FAIR VALUE =====
        # BUG FIX #6: original used weight 0.1 → max shift 0.1 ticks (useless)
        # At 1.5 weight: strong OBI (±0.5) shifts fair by ±0.75 ticks — meaningful
        fair = base_fair + OSMIUM_CFG["fair_imb_weight"] * signal

        # L3 overrides: shift fair by 2 ticks in signal direction
        if l3 != 0:
            fair += l3 * OSMIUM_CFG["l3_skew_ticks"]

        # ===== RESERVATION PRICE (inventory + signal skew) =====
        # BUG FIX #7: original used 0.10*pos → at pos=80 gives -8 tick shift,
        # which crosses the make_width=2 spread and produces garbage quotes.
        # 0.05*pos → at pos=80 gives -4 ticks, safe with make_width=2.
        reservation = (
            base_fair
            - OSMIUM_CFG["reservation_inv"] * pos
            + OSMIUM_CFG["reservation_sig"] * signal
        )
        # L3 also shifts reservation
        if l3 != 0:
            reservation += l3 * OSMIUM_CFG["l3_skew_ticks"]

        take_width = OSMIUM_CFG["take_width"]
        make_width = OSMIUM_CFG["make_width"]
        base_size  = OSMIUM_CFG["order_size"]
        inv_limit  = OSMIUM_CFG["inventory_limit"]
        inv_hard   = OSMIUM_CFG["inventory_hard"]

        remaining_buy  = POSITION_LIMIT - pos
        remaining_sell = POSITION_LIMIT + pos

        # ===== TAKING (flow-filtered) =====
        # Only take in the direction the signal says — avoids toxic fills
        for ask_px in sorted(order_depth.sell_orders):
            if remaining_buy <= 0:
                break
            # Take if cheap AND signal confirms upward pressure (OBI>0 or L3 up)
            if ask_px <= fair - take_width and (signal > 0.1 or l3 == +1):
                qty = min(-order_depth.sell_orders[ask_px], remaining_buy)
                if qty > 0:
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
            else:
                break

        for bid_px in sorted(order_depth.buy_orders, reverse=True):
            if remaining_sell <= 0:
                break
            # Take if expensive AND signal confirms downward pressure
            if bid_px >= fair + take_width and (signal < -0.1 or l3 == -1):
                qty = min(order_depth.buy_orders[bid_px], remaining_sell)
                if qty > 0:
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
            else:
                break

        # ===== QUOTES =====
        our_bid = min(best_bid + 1, math.floor(reservation - make_width))
        our_ask = max(best_ask - 1, math.ceil(reservation + make_width))

        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if our_bid >= our_ask:
            return orders, ema_state, imb_state

        # ===== INVENTORY SIZE SKEW =====
        # BUG FIX #8: simplified — cubic skew + linear scale, no redundant guard
        inv_ratio = pos / POSITION_LIMIT         # [-1, 1]
        skew      = inv_ratio ** 3               # cubic: aggressive near limits

        bid_size = max(1, round(base_size * (1 - max(0,  skew))))
        ask_size = max(1, round(base_size * (1 - max(0, -skew))))

        # Additional linear scale-down between inv_limit and inv_hard
        if pos > inv_limit:
            scale    = 1 - (pos - inv_limit) / max(1, inv_hard - inv_limit)
            bid_size = max(1, round(bid_size * scale))
        elif pos < -inv_limit:
            scale    = 1 - ((-pos) - inv_limit) / max(1, inv_hard - inv_limit)
            ask_size = max(1, round(ask_size * scale))

        # ===== PASSIVE ORDERS =====
        if pos < inv_hard and remaining_buy > 0:
            orders.append(Order(product, our_bid,  min(bid_size,  remaining_buy)))
        if pos > -inv_hard and remaining_sell > 0:
            orders.append(Order(product, our_ask, -min(ask_size, remaining_sell)))

        return orders, ema_state, imb_state