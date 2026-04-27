"""
IMC Prosperity Round 3 — DP-Derived Mean Reversion Strategy (v3)
=================================================================

FINDING: Both VEX and HYD are Ornstein-Uhlenbeck mean-reverting processes.
DP analysis on 3 days of historical data confirmed:
  - VEX buys at -5.5 below mean, sells at +5.5 above mean
  - HYD buys at -10.5 below mean, sells at +10.5 above mean

KEY FIXES vs prior versions:
  1. PASSIVE orders only — posting at bid/ask to COLLECT the 5-16 tick spread
     (taker orders were paying 37k/run in spread costs, destroying the signal)
  2. Init mu = opening price, update with EWMA alpha=0.005
     (fixed prior of 5250 was 12 pts below true mean → permanent short bias)

BACKTEST (passive, 1000-tick emulator data):  ~50-70k per run
BACKTEST (historical 10k-tick data, 3 days):  ~250k total
"""

from typing import Any
import json

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders),
            conversions, "", ""
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {}
        for product, obs in observations.conversionObservations.items():
            conv[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                             obs.exportTariff, obs.importTariff,
                             obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, conv]

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
                out = candidate; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ── Strategy parameters (DP-derived) ─────────────────────────────────────────

# OU thresholds from DP trade analysis across 3 historical days
VEX_THRESHOLD = 5.5    # full position when |dev| = 5.5 ticks
VEX_LIMIT     = 200
VEX_STEP      = 20     # max units per tick

HYD_THRESHOLD = 10.5   # full position when |dev| = 10.5 ticks
HYD_LIMIT     = 100
HYD_STEP      = 10

# EWMA for mean estimation: init to opening price, alpha=0.005 (half-life 138 ticks)
# Fast enough to track within a 1000-tick run without being dragged by short-term noise
MU_ALPHA = 0.005

# Vol arb: structural IV mispricing, delta-neutral sized
VOL_ARB = {
    "VEV_5400": +300,   # underpriced IV
    "VEV_5300": -150,   # delta-neutral: 300 * delta(5400)/delta(5300)
}


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        vex_mu       = ts.get("vex_mu")        # None on first tick
        hyd_mu       = ts.get("hyd_mu")
        vol_arb_done = ts.get("vol_arb_done", False)

        pos = state.position or {}
        ods = state.order_depths

        def best_bid(sym):
            od = ods.get(sym)
            return max(od.buy_orders) if od and od.buy_orders else None

        def best_ask(sym):
            od = ods.get(sym)
            return min(od.sell_orders) if od and od.sell_orders else None

        def mid(sym):
            b = best_bid(sym); a = best_ask(sym)
            return (b + a) / 2.0 if b and a else None

        # ── VELVETFRUIT: OU mean reversion, passive orders ────────────────────
        S = mid("VELVETFRUIT_EXTRACT")
        if S is not None:
            vex_mu = S if vex_mu is None else MU_ALPHA * S + (1 - MU_ALPHA) * vex_mu

            cur = pos.get("VELVETFRUIT_EXTRACT", 0)
            dev = S - vex_mu
            target = int(max(-VEX_LIMIT, min(VEX_LIMIT, -dev / VEX_THRESHOLD * VEX_LIMIT)))
            need = target - cur

            orders: list[Order] = []
            if need > 0:
                bid = best_bid("VELVETFRUIT_EXTRACT")
                if bid is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", bid, min(need, VEX_STEP)))
            elif need < 0:
                ask = best_ask("VELVETFRUIT_EXTRACT")
                if ask is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", ask, -min(-need, VEX_STEP)))
            result["VELVETFRUIT_EXTRACT"] = orders

        # ── HYDROGEL: OU mean reversion, passive orders ───────────────────────
        H = mid("HYDROGEL_PACK")
        if H is not None:
            hyd_mu = H if hyd_mu is None else MU_ALPHA * H + (1 - MU_ALPHA) * hyd_mu

            cur = pos.get("HYDROGEL_PACK", 0)
            dev = H - hyd_mu
            target = int(max(-HYD_LIMIT, min(HYD_LIMIT, -dev / HYD_THRESHOLD * HYD_LIMIT)))
            need = target - cur

            orders: list[Order] = []
            if need > 0:
                bid = best_bid("HYDROGEL_PACK")
                if bid is not None:
                    orders.append(Order("HYDROGEL_PACK", bid, min(need, HYD_STEP)))
            elif need < 0:
                ask = best_ask("HYDROGEL_PACK")
                if ask is not None:
                    orders.append(Order("HYDROGEL_PACK", ask, -min(-need, HYD_STEP)))
            result["HYDROGEL_PACK"] = orders

        # ── VOL ARB: structural options mispricing, hold all round ────────────
        if not vol_arb_done:
            all_done = True
            for sym, target in VOL_ARB.items():
                cur = pos.get(sym, 0)
                need = target - cur
                if need == 0:
                    continue
                all_done = False
                od = ods.get(sym)
                if od is None:
                    continue
                orders: list[Order] = []
                if need > 0 and od.sell_orders:
                    ask = min(od.sell_orders)
                    orders.append(Order(sym, ask, min(need, abs(target))))
                elif need < 0 and od.buy_orders:
                    bid = max(od.buy_orders)
                    orders.append(Order(sym, bid, -min(-need, abs(target))))
                result[sym] = orders
            if all_done:
                vol_arb_done = True

        # ── Save state ────────────────────────────────────────────────────────
        new_td = json.dumps({"vex_mu": vex_mu, "hyd_mu": hyd_mu,
                             "vol_arb_done": vol_arb_done})
        logger.flush(state, result, conversions, new_td)
        return result, conversions, new_td