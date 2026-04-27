"""
Round 4 Trading Strategy - IMC Prosperity Format
=================================================
1. HYDROGEL_PACK market making (capture Mark 38 taker flow)
2. VELVETFRUIT_EXTRACT market making (capture Mark 55 + Mark 67 flow)
3. VEV option buying (IV ~3.5% < RV ~4.8%) + delta hedge

No external libraries - pure Python stdlib only.
"""

import json
import math
from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# datamodel types (IMC provides these on their servers)
# Included here so the file compiles standalone for local testing.
# On submission, remove or guard with try/except if IMC pre-loads them.
# ---------------------------------------------------------------------------

try:
    from datamodel import TradingState, Order, OrderDepth, Trade, Symbol, Product
except ImportError:
    Symbol = str
    Product = str

    class Order:
        def __init__(self, symbol: str, price: int, quantity: int) -> None:
            self.symbol = symbol
            self.price = price
            self.quantity = quantity
        def __str__(self):
            return f"({self.symbol}, {self.price}, {self.quantity})"
        def __repr__(self):
            return self.__str__()

    class OrderDepth:
        def __init__(self):
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}

    class Trade:
        def __init__(self, symbol="", price=0, quantity=0,
                     buyer="", seller="", timestamp=0):
            self.symbol = symbol
            self.price = price
            self.quantity = quantity
            self.buyer = buyer
            self.seller = seller
            self.timestamp = timestamp

    class TradingState:
        def __init__(self, traderData="", timestamp=0, listings=None,
                     order_depths=None, own_trades=None, market_trades=None,
                     position=None, observations=None):
            self.traderData = traderData
            self.timestamp = timestamp
            self.listings = listings or {}
            self.order_depths = order_depths or {}
            self.own_trades = own_trades or {}
            self.market_trades = market_trades or {}
            self.position = position or {}
            self.observations = observations


# ---------------------------------------------------------------------------
# ── Black-Scholes helpers (pure python) ──────────────────────────────────
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Abramowitz-Stegun approximation, max error 1.5e-7."""
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def bs_call_price(S: float, K: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0 or S <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def bs_delta(S: float, K: float, sigma: float, T: float) -> float:
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HP_POS_LIMIT = 25
HP_SKEW_THRESH = 15

VF_POS_LIMIT = 25
VF_SKEW_THRESH = 15

MARKET_IV = 0.035
REALIZED_VOL = 0.0484
T_START = 0.8       # 4 DTE at round start
T_END = 0.6         # 3 DTE at round end
MAX_TS = 999900

OPT_POS_LIMIT = 30
OPT_STRIKES = [5200, 5300, 5400]
DELTA_LIMIT = 40


# ---------------------------------------------------------------------------
# Trader (the class IMC looks for)
# ---------------------------------------------------------------------------

class Trader:

    def run(self, state: TradingState):
        """
        Called each tick. Returns (result, conversions, traderData).
        """
        # ---- restore persisted state ----
        mem: dict = {}
        if state.traderData:
            try:
                mem = json.loads(state.traderData)
            except Exception:
                mem = {}

        last_hedge_ts: int = mem.get("lh", -999999)

        result: Dict[str, List[Order]] = {}
        ts = state.timestamp

        # ---- helpers ----
        def pos(sym: str) -> int:
            return state.position.get(sym, 0)

        def bbid(d: OrderDepth) -> int:
            return max(d.buy_orders.keys()) if d.buy_orders else 0

        def bask(d: OrderDepth) -> int:
            return min(d.sell_orders.keys()) if d.sell_orders else 999999

        def get_T() -> float:
            frac = ts / MAX_TS if MAX_TS > 0 else 0
            return T_START - (T_START - T_END) * frac

        def net_delta(vf_mid: float) -> float:
            d = float(pos('VELVETFRUIT_EXTRACT'))
            T = get_T()
            for k in OPT_STRIKES:
                p = pos(f'VEV_{k}')
                if p != 0:
                    d += p * bs_delta(vf_mid, k, MARKET_IV, T)
            return d

        # ---- get VF market data (used by multiple sections) ----
        vf_mid = 5250.0
        vf_b, vf_a = 0, 999999
        if 'VELVETFRUIT_EXTRACT' in state.order_depths:
            d = state.order_depths['VELVETFRUIT_EXTRACT']
            vf_b, vf_a = bbid(d), bask(d)
            if vf_b > 0 and vf_a < 999999:
                vf_mid = (vf_b + vf_a) / 2.0

        # ================================================================
        # 1. HYDROGEL_PACK MARKET MAKING
        # ================================================================
        if 'HYDROGEL_PACK' in state.order_depths:
            depth = state.order_depths['HYDROGEL_PACK']
            b, a = bbid(depth), bask(depth)
            hp_pos = pos('HYDROGEL_PACK')

            if b > 0 and a < 999999 and (a - b) >= 2:
                my_bid = b + 1
                my_ask = a - 1
                if my_bid >= my_ask:
                    m = (b + a) // 2
                    my_bid, my_ask = m - 1, m + 1

                bid_qty, ask_qty = 4, 4
                if hp_pos > HP_SKEW_THRESH:
                    my_bid -= 1; bid_qty, ask_qty = 2, 6
                elif hp_pos < -HP_SKEW_THRESH:
                    my_ask += 1; bid_qty, ask_qty = 6, 2

                hp_orders: List[Order] = []
                if hp_pos < HP_POS_LIMIT:
                    q = min(bid_qty, HP_POS_LIMIT - hp_pos)
                    if q > 0:
                        hp_orders.append(Order('HYDROGEL_PACK', my_bid, q))
                if hp_pos > -HP_POS_LIMIT:
                    q = min(ask_qty, HP_POS_LIMIT + hp_pos)
                    if q > 0:
                        hp_orders.append(Order('HYDROGEL_PACK', my_ask, -q))
                if hp_orders:
                    result['HYDROGEL_PACK'] = hp_orders

        # ================================================================
        # 2-5. VELVETFRUIT: MM + OPTIONS + HEDGE + EMERGENCY
        # ================================================================
        # Track VF order budget within this tick to avoid exceeding limits
        vf_pos = pos('VELVETFRUIT_EXTRACT')
        vf_buy_budget = VF_POS_LIMIT - vf_pos      # max we can buy
        vf_sell_budget = VF_POS_LIMIT + vf_pos      # max we can sell (as positive number)
        vf_orders: List[Order] = []

        # ---- 2. VF MARKET MAKING ----
        if vf_b > 0 and vf_a < 999999 and (vf_a - vf_b) >= 2:
            eff_pos = int(round(net_delta(vf_mid)))

            my_bid = vf_b + 1
            my_ask = vf_a - 1
            if my_bid >= my_ask:
                m = (vf_b + vf_a) // 2
                my_bid, my_ask = m - 1, m + 1

            bid_qty, ask_qty = 4, 4
            if eff_pos > VF_SKEW_THRESH:
                my_bid -= 1; bid_qty, ask_qty = 2, 6
            elif eff_pos < -VF_SKEW_THRESH:
                my_ask += 1; bid_qty, ask_qty = 6, 2

            q = min(bid_qty, vf_buy_budget)
            if q > 0:
                vf_orders.append(Order('VELVETFRUIT_EXTRACT', my_bid, q))
                vf_buy_budget -= q

            q = min(ask_qty, vf_sell_budget)
            if q > 0:
                vf_orders.append(Order('VELVETFRUIT_EXTRACT', my_ask, -q))
                vf_sell_budget -= q

        # ---- 3. OPTION BUYING ----
        cur_delta = net_delta(vf_mid)
        T = get_T()

        for strike in OPT_STRIKES:
            sym = f'VEV_{strike}'
            if sym not in state.order_depths:
                continue

            depth = state.order_depths[sym]
            opt_pos = pos(sym)

            if opt_pos >= OPT_POS_LIMIT:
                continue
            if abs(cur_delta) > DELTA_LIMIT * 0.7:
                continue

            a = bask(depth)
            if a >= 999999:
                continue

            fair = bs_call_price(vf_mid, strike, REALIZED_VOL, T)
            if fair - a < 2.0:
                continue

            qty = min(4, OPT_POS_LIMIT - opt_pos)
            if qty > 0:
                if sym not in result:
                    result[sym] = []
                result[sym].append(Order(sym, a, qty))

        # ---- 4. DELTA HEDGING (periodic) ----
        nd = net_delta(vf_mid)

        if abs(nd) > 3 and (ts - last_hedge_ts >= 1000):
            last_hedge_ts = ts
            trade_qty = -int(round(nd))
            trade_qty = max(-30, min(30, trade_qty))

            if trade_qty > 0 and vf_buy_budget > 0 and vf_a < 999999:
                q = min(trade_qty, vf_buy_budget)
                if q > 0:
                    vf_orders.append(Order('VELVETFRUIT_EXTRACT', vf_a, q))
                    vf_buy_budget -= q

            elif trade_qty < 0 and vf_sell_budget > 0 and vf_b > 0:
                q = min(-trade_qty, vf_sell_budget)
                if q > 0:
                    vf_orders.append(Order('VELVETFRUIT_EXTRACT', vf_b, -q))
                    vf_sell_budget -= q

        # ---- 5. EMERGENCY DELTA FLATTEN ----
        nd = net_delta(vf_mid)
        if abs(nd) > DELTA_LIMIT:
            excess = int(abs(nd) - DELTA_LIMIT) + 2
            excess = min(excess, 30)

            if nd > 0 and vf_sell_budget > 0 and vf_b > 0:
                q = min(excess, vf_sell_budget)
                if q > 0:
                    vf_orders.append(Order('VELVETFRUIT_EXTRACT', vf_b, -q))
                    vf_sell_budget -= q

            elif nd < 0 and vf_buy_budget > 0 and vf_a < 999999:
                q = min(excess, vf_buy_budget)
                if q > 0:
                    vf_orders.append(Order('VELVETFRUIT_EXTRACT', vf_a, q))
                    vf_buy_budget -= q

        if vf_orders:
            result['VELVETFRUIT_EXTRACT'] = vf_orders

        # ---- save state ----
        mem["lh"] = last_hedge_ts
        trader_data = json.dumps(mem)

        conversions = 0
        return result, conversions, trader_data