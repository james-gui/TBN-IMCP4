from datamodel import OrderDepth, TradingState, Order, UserId
from typing import List, Dict
import jsonpickle

# ─────────────────────────────────────────────
#  CONFIGURATION — tune these per round
# ─────────────────────────────────────────────
POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

PARAMS = {
    "EMERALDS": {
        "ema_fast": 5,
        "ema_slow": 20,
        "momentum_weight": 0.4,
        "spread_half": 4,
        "inventory_skew_factor": 1.0,
        "take_edge": 4,
    },
    "TOMATOES": {
        "ema_fast": 5,
        "ema_slow": 20,
        "momentum_weight": 0.4,
        "spread_half": 5,
        "inventory_skew_factor": 1.5,
        "take_edge": 3,
    },
}

def update_ema(prev: float, new_val: float, span: int) -> float:
    alpha = 2.0 / (span + 1)
    return alpha * new_val + (1 - alpha) * prev


class Trader:

    def run(self, state: TradingState):
        s = {}
        if state.traderData:
            try:
                s = jsonpickle.decode(state.traderData)
            except Exception:
                s = {}

        for product in PARAMS:
            if product not in s:
                s[product] = {"ema_fast": None, "ema_slow": None}

        result: Dict[str, List[Order]] = {}

        for product, cfg in PARAMS.items():
            if product not in state.order_depths:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            pos_limit = POSITION_LIMITS.get(product, 20)
            current_pos = state.position.get(product, 0)

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            ps = s[product]
            if ps["ema_fast"] is None:
                ps["ema_fast"] = mid_price
                ps["ema_slow"] = mid_price
            else:
                ps["ema_fast"] = update_ema(ps["ema_fast"], mid_price, cfg["ema_fast"])
                ps["ema_slow"] = update_ema(ps["ema_slow"], mid_price, cfg["ema_slow"])

            momentum = ps["ema_fast"] - ps["ema_slow"]
            fair_value = ps["ema_slow"] + cfg["momentum_weight"] * momentum

            inv_skew = -current_pos * cfg["inventory_skew_factor"]
            skewed_fv = fair_value + inv_skew

            half = cfg["spread_half"]
            our_bid_price = round(skewed_fv - half)
            our_ask_price = round(skewed_fv + half)

            take_edge = cfg["take_edge"]
            remaining_buy  = pos_limit - current_pos
            remaining_sell = pos_limit + current_pos

            for ask_px in sorted(order_depth.sell_orders.keys()):
                if remaining_buy <= 0:
                    break
                if ask_px <= fair_value - take_edge:
                    vol = -order_depth.sell_orders[ask_px]
                    qty = min(vol, remaining_buy)
                    orders.append(Order(product, ask_px, qty))
                    remaining_buy -= qty
                else:
                    break

            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining_sell <= 0:
                    break
                if bid_px >= fair_value + take_edge:
                    vol = order_depth.buy_orders[bid_px]
                    qty = min(vol, remaining_sell)
                    orders.append(Order(product, bid_px, -qty))
                    remaining_sell -= qty
                else:
                    break

            maker_bid = min(our_bid_price, best_ask - 1)
            maker_ask = max(our_ask_price, best_bid + 1)

            if remaining_buy > 0:
                orders.append(Order(product, maker_bid, remaining_buy))
            if remaining_sell > 0:
                orders.append(Order(product, maker_ask, -remaining_sell))

            result[product] = orders

        traderData = jsonpickle.encode(s)
        conversions = 0
        return result, conversions, traderData
