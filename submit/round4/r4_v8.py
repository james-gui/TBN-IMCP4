from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import json

# ============================================================
# IMC Prosperity 4 — Round 4  v9
# ============================================================
#
# ROOT CAUSE ANALYSIS from external 3-day backtester (Final PnL: 97):
#
# Problem 1 — POSITION DRIFT:
#   HP kept accumulating a large long position (0→+60% of limit).
#   Over 3 days the price eventually reverted, wiping all gains.
#   Cause: our MAKE quotes were symmetric but fills weren't,
#   and the CLEAR step posted at a price that never got hit.
#   Fix: HARD INVENTORY GATES. When |pos| > 40% of limit,
#   only quote the unwind side. No exceptions.
#
# Problem 2 — MARK 67 UNRELIABLE:
#   Historical data: 165 M67 trades over 3 days.
#   Actual backtester: only 2 M67 trades! The price paths differ.
#   M67 barely fires in production → any M67 strategy is noise.
#   Fix: Remove M67 signal entirely.
#
# Problem 3 — CLEAR STEP NEVER FIRED:
#   We posted sell orders at wall_mid, but wall_mid sits between
#   best_bid and best_ask — nobody bids there, so it never filled.
#   Fix: Use SOFT SKEW (shift fair value) not separate CLEAR orders.
#
# V9 CORE APPROACH:
#   Best mid as fair value (wall_mid ≈ best_mid for our 2-level book).
#   Overbid/underbid to get price priority over bots.
#   HARD gate: if long >40% limit, quote ask only. If short >40%, bid only.
#   SOFT skew: fair value shifts against inventory to gradually unwind.

LIMITS = {"HYDROGEL_PACK": 200, "VELVETFRUIT_EXTRACT": 200}
HARD_GATE = 0.4   # fraction of limit — stop quoting one side beyond this
SOFT_SKEW = 0.3   # how much to shift fair value per unit of normalised pos


class Trader:

    def get_book(self, od: OrderDepth):
        bids = {p: abs(v) for p, v in od.buy_orders.items()}
        asks = {p: abs(v) for p, v in od.sell_orders.items()}
        return bids, asks

    def mm(self, product: str, od: OrderDepth, pos: int) -> List[Order]:
        bids, asks = self.get_book(od)
        if not bids or not asks:
            return []

        best_bid = max(bids)
        best_ask = min(asks)
        best_mid = (best_bid + best_ask) / 2

        limit = LIMITS[product]
        orders = []
        room_buy  = limit - pos
        room_sell = limit + pos

        # ── Soft skew: shift fair value against position ──────────────
        # When long, fair_mid drifts down → our ask drops, bid drops
        # This makes us more likely to sell and less likely to buy
        norm_pos = pos / limit  # range -1 to +1
        fair_mid = best_mid - norm_pos * SOFT_SKEW * (best_ask - best_bid)

        # ── Hard gates: stop adding to a large position ───────────────
        gate = HARD_GATE * limit
        can_buy  = pos < gate    # stop buying if already very long
        can_sell = pos > -gate   # stop selling if already very short

        # ── TAKE: hit prices on the wrong side of fair_mid ────────────
        for ap in sorted(asks):
            if ap >= fair_mid or not can_buy:
                break
            vol = min(asks[ap], room_buy)
            if vol > 0:
                orders.append(Order(product, ap, vol))
                room_buy -= vol

        for bp in sorted(bids, reverse=True):
            if bp <= fair_mid or not can_sell:
                break
            vol = min(bids[bp], room_sell)
            if vol > 0:
                orders.append(Order(product, bp, -vol))
                room_sell -= vol

        # ── MAKE: overbid/underbid to get price priority ──────────────
        # Frankfurt pattern: find best bot-bid below fair_mid, bid bp+1
        #                    find best bot-ask above fair_mid, ask ap-1
        if can_buy and room_buy > 0:
            bid_price = int(best_bid)  # fallback
            for bp in sorted(bids, reverse=True):
                if bp < fair_mid:
                    bid_price = bp + 1 if bids[bp] > 1 else bp
                    break
            bid_price = min(bid_price, int(fair_mid))  # never cross fair
            orders.append(Order(product, bid_price, room_buy))

        if can_sell and room_sell > 0:
            ask_price = int(best_ask)  # fallback
            for ap in sorted(asks):
                if ap > fair_mid:
                    ask_price = ap - 1 if asks[ap] > 1 else ap
                    break
            ask_price = max(ask_price, int(fair_mid) + 1)  # never cross fair
            orders.append(Order(product, ask_price, -room_sell))

        return orders

    def run(self, state: TradingState):
        result = {}

        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self.mm(
                "HYDROGEL_PACK", state.order_depths["HYDROGEL_PACK"], pos
            )

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self.mm(
                "VELVETFRUIT_EXTRACT", state.order_depths["VELVETFRUIT_EXTRACT"], pos
            )

        return result, 0, ""