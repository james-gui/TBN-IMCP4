import json
import math
from typing import Any, List, Dict, Optional
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ══════════════════════════════════════════════════════════════════════════════
#  IMC Prosperity 4 – Round 3  │  r3_v10  │  "Theta Farming + Vol Selling"
# ══════════════════════════════════════════════════════════════════════════════
#
#  STRATEGY THESIS
#  ───────────────
#  The IMC bots price ALL options at ~27% implied vol regardless of TTE.
#  Realized vol in this round is only ~5% annualized.
#  More importantly: liquidation is at INTRINSIC VALUE (or very low vol),
#  which means every unit of time value the market charges is pure profit
#  for the short side.
#
#  Evidence:
#  • VEV range = 35pts over full day  →  realized vol ≈ 5% annualized
#  • VEV_5300 market price ≈ 50  →  intrinsic = 0  →  all time value
#  • SHORT 300x VEV_5300 @ 49 bid, liq @ 0 = +14,700 that strike alone
#  • 5 OTM strikes × avg ~7k = ~35k from options alone
#  • Compounded over 3 rounds → potential 100k+
#
#  LAYERS
#  ──────
#  Layer 1: SHORT OTM calls (5100–5500) at max 300 each immediately.
#           Re-fill if position is closed for any reason.
#           These expire worthless or nearly worthless.
#
#  Layer 2: SHORT near-ATM calls (5000) at 300 — small edge but free money.
#
#  Layer 3: LONG deep-ITM calls (4000, 4500) at 300 — market prices below
#           intrinsic adjusted for liquidation (+1.6 per unit).
#
#  Layer 4: DELTA HEDGE with VEV to be net delta-neutral.
#           Net short delta from layers 1+2 offset by ITM longs.
#           Residual hedged with VEV spot (±60 units typically).
#
#  Layer 5: VEV market-making for residual flow.
#
#  Layer 6: HYDROGEL market-making (spread=16, easy edge).
#
#  Layer 7: Insider signal (HYDROGEL anon → OTM cluster, LONG VEV).
#           Kept from v9 as a bonus layer, not the primary edge.
#
# ══════════════════════════════════════════════════════════════════════════════


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

# ─── Products ────────────────────────────────────────────────────────────────
HYDROGEL = "HYDROGEL_PACK"
VEV      = "VELVETFRUIT_EXTRACT"
VOUCHERS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}
LIMITS = {HYDROGEL: 200, VEV: 200}
for _v in VOUCHERS:
    LIMITS[_v] = 300

# ─── Round parameters ────────────────────────────────────────────────────────
TTE_START_DAYS = 5        # TTE at start of round 3
TICKS_PER_DAY  = 100_000

# ─── Configuration ───────────────────────────────────────────────────────────
CFG = {
    # Implied vol used for BS fair value and delta calculation
    # Set to ATM market IV (~27%). This is what we assume IMC uses for liq.
    # Even if wrong, our short positions profit from ANY liq vol below market IV.
    "pricing_sigma":       0.264,

    # Vol selling: target SHORT positions for OTM/near-ATM options
    # and LONG positions for deep ITM (buy below intrinsic).
    # Positions filled by posting at best bid/ask, not taking aggressively.
    # "vol_short_strikes": those we SELL (collect premium, expect liq < mkt)
    # "vol_long_strikes":  those we BUY (market below intrinsic liq value)
    "vol_short_strikes":   ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"],
    "vol_long_strikes":    ["VEV_4000", "VEV_4500"],
    "vol_skip_strikes":    ["VEV_6000", "VEV_6500"],  # bid=0, no fills possible

    # Target position: fill to this as fast as possible, hold until liquidation
    "vol_short_target":    -300,   # max short per OTM strike
    "vol_long_target":     300,    # max long per ITM strike

    # Delta hedging: keep net book delta within this band before acting
    "delta_band":          10,     # rehedge if |net_delta| > this
    "delta_hedge_size":    20,     # units per hedge order

    # VEV market making (runs when NOT in signal mode)
    "ema_alpha":           0.15,
    "make_width":          2,
    "take_width":          1,
    "vev_order_size":      15,
    "inv_limit":           50,
    "inv_hard":            150,

    # HYDROGEL market making
    "hydrogel_order_size": 8,
    "hydrogel_inv_limit":  30,

    # Insider signal (v9 carry-over, low weight)
    "min_otm_strikes":     3,
    "hydrogel_window":     6000,
    "signal_exit_ticks":   4000,
    "signal_size":         0.30,   # reduced: vol selling is the main edge
    "profit_exit":         8,
    "loss_exit":           8,
    "cooldown_ticks":      10000,
}


# ─── BS Helpers ──────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or S <= 0:
        return max(S - K, 0.0)
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Delta of a European call."""
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sqT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqT)
    return _norm_cdf(d1)


# ─── Order-book helpers ──────────────────────────────────────────────────────

def _best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders) if od.buy_orders else None

def _best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders) if od.sell_orders else None

def _wall_mid(od: OrderDepth) -> Optional[float]:
    b, a = _best_bid(od), _best_ask(od)
    if b is None or a is None:
        return None
    return (b + a) / 2.0

def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def _mm_orders(
    symbol: str, od: OrderDepth, pos: int, fv: float, limit: int,
    order_size: int, inv_limit: int, inv_hard: int,
    make_w: int = 2, take_w: int = 1,
) -> List[Order]:
    orders: List[Order] = []
    if not od.buy_orders or not od.sell_orders:
        return orders

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    rem_buy  = limit - pos
    rem_sell = limit + pos

    for ask_px in sorted(od.sell_orders):
        if rem_buy <= 0: break
        if ask_px <= fv - take_w:
            qty = min(-od.sell_orders[ask_px], rem_buy)
            orders.append(Order(symbol, ask_px, qty))
            rem_buy -= qty
        else: break

    for bid_px in sorted(od.buy_orders, reverse=True):
        if rem_sell <= 0: break
        if bid_px >= fv + take_w:
            qty = min(od.buy_orders[bid_px], rem_sell)
            orders.append(Order(symbol, bid_px, -qty))
            rem_sell -= qty
        else: break

    skew = -round((pos / limit) * 2)
    our_bid = round(fv) + skew - make_w
    our_ask = round(fv) + skew + make_w
    our_bid = min(our_bid, best_ask - 1)
    our_ask = max(our_ask, best_bid + 1)

    if our_bid < our_ask:
        bs = ask_s = order_size
        if pos > inv_limit:
            sc = 1.0 - (pos - inv_limit) / max(inv_hard - inv_limit, 1)
            bs = max(1, round(order_size * sc))
        elif pos < -inv_limit:
            sc = 1.0 - ((-pos) - inv_limit) / max(inv_hard - inv_limit, 1)
            ask_s = max(1, round(order_size * sc))

        if pos < inv_hard and rem_buy > 0 and our_bid < fv:
            orders.append(Order(symbol, our_bid, min(bs, rem_buy)))
        if pos > -inv_hard and rem_sell > 0 and our_ask > fv:
            orders.append(Order(symbol, our_ask, -min(ask_s, rem_sell)))

    return orders


# ─── Trader ──────────────────────────────────────────────────────────────────

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        vev_ema         = ts.get("vev_ema", None)
        hydrogel_ema    = ts.get("hydrogel_ema", None)
        otm_signal_ts   = ts.get("otm_signal_ts", -(10**9))
        hydrogel_sig_ts = ts.get("hydrogel_sig_ts", -(10**9))
        vev_entry_px    = ts.get("vev_entry_px", None)
        cooldown_until  = ts.get("cooldown_until", -(10**9))
        signal_dir      = ts.get("signal_dir", None)

        t = state.timestamp
        sigma = CFG["pricing_sigma"]
        tte_days  = max(TTE_START_DAYS - t / TICKS_PER_DAY, 0.001)
        tte_years = tte_days / 365.0

        # ── Update EMAs ───────────────────────────────────────────────────────
        alpha = CFG["ema_alpha"]
        if VEV in state.order_depths:
            wm = _wall_mid(state.order_depths[VEV])
            if wm is not None:
                vev_ema = wm if vev_ema is None else alpha * wm + (1-alpha) * vev_ema

        if HYDROGEL in state.order_depths:
            wm = _wall_mid(state.order_depths[HYDROGEL])
            if wm is not None:
                hydrogel_ema = wm if hydrogel_ema is None else alpha * wm + (1-alpha) * hydrogel_ema

        S = vev_ema if vev_ema is not None else 5262.0

        # ════════════════════════════════════════════════════════════════════
        #  LAYER 1-3: VOL SELLING – the primary edge
        #
        #  For SHORT strikes (5000–5500):
        #    Immediately post sell orders at best bid to fill short positions.
        #    We want -300 as fast as possible and hold. No take-profit, no stop.
        #    Liquidation at intrinsic makes these worth ~0 at expiry.
        #
        #  For LONG strikes (4000, 4500):
        #    Market price ≈ intrinsic - 1.6. BUY at ask to get 300 long.
        #    At liquidation we earn the small edge from being underpriced.
        #
        #  Sizing: fill to target immediately, re-fill if knocked out.
        # ════════════════════════════════════════════════════════════════════

        vol_short_set = set(CFG["vol_short_strikes"])
        vol_long_set  = set(CFG["vol_long_strikes"])
        vol_skip_set  = set(CFG["vol_skip_strikes"])

        # Track net option delta for hedge calculation
        net_option_delta = 0.0

        for voucher in VOUCHERS:
            od = state.order_depths.get(voucher)
            pos = state.position.get(voucher, 0)
            orders: List[Order] = []

            if voucher in vol_skip_set or od is None:
                result[voucher] = orders
                continue

            K = STRIKES[voucher]
            fair = bs_call(S, K, tte_years, sigma)
            delta = bs_delta(S, K, tte_years, sigma)

            # Accumulate net option delta for hedging
            net_option_delta += delta * pos

            bid = _best_bid(od)
            ask = _best_ask(od)

            if voucher in vol_short_set:
                # ── Want to be SHORT 300 ─────────────────────────────────
                target = CFG["vol_short_target"]  # -300
                if pos > target and bid is not None and bid > 0:
                    # Fill shortfall by hitting the bids
                    need = pos - target  # units to sell
                    for bid_px in sorted(od.buy_orders, reverse=True):
                        if need <= 0: break
                        qty = min(od.buy_orders[bid_px], need)
                        orders.append(Order(voucher, bid_px, -qty))
                        need -= qty

            elif voucher in vol_long_set:
                # ── Want to be LONG 300 ──────────────────────────────────
                target = CFG["vol_long_target"]  # +300
                if pos < target and ask is not None:
                    need = target - pos
                    for ask_px in sorted(od.sell_orders):
                        if need <= 0: break
                        qty = min(-od.sell_orders[ask_px], need)
                        orders.append(Order(voucher, ask_px, qty))
                        need -= qty

            result[voucher] = orders

        # ════════════════════════════════════════════════════════════════════
        #  LAYER 4: DELTA HEDGE with VEV
        #
        #  net_option_delta = sum(delta_i * pos_i) across all vouchers
        #  To be delta-neutral, we need VEV position = -net_option_delta
        #  If net_option_delta = -60, we need +60 VEV.
        #  We hedge in chunks to avoid spiking impact.
        # ════════════════════════════════════════════════════════════════════

        vev_pos = state.position.get(VEV, 0)
        target_vev_hedge = -round(net_option_delta)
        target_vev_hedge = max(-LIMITS[VEV], min(LIMITS[VEV], target_vev_hedge))
        hedge_gap = target_vev_hedge - vev_pos
        delta_band = CFG["delta_band"]

        hedge_orders: List[Order] = []
        od_vev = state.order_depths.get(VEV)

        if od_vev is not None and vev_ema is not None:
            if abs(hedge_gap) > delta_band:
                hedge_size = CFG["delta_hedge_size"]
                hedge_qty = min(abs(hedge_gap), hedge_size)

                if hedge_gap > 0:
                    # Need to BUY VEV to hedge
                    ask = _best_ask(od_vev)
                    if ask is not None:
                        rem = LIMITS[VEV] - vev_pos
                        qty = min(hedge_qty, rem)
                        if qty > 0:
                            hedge_orders.append(Order(VEV, ask, qty))
                else:
                    # Need to SELL VEV to hedge
                    bid = _best_bid(od_vev)
                    if bid is not None:
                        rem = LIMITS[VEV] + vev_pos
                        qty = min(hedge_qty, rem)
                        if qty > 0:
                            hedge_orders.append(Order(VEV, bid, -qty))

        # ════════════════════════════════════════════════════════════════════
        #  LAYER 7: INSIDER SIGNAL (bonus layer, from v9)
        #
        #  If HYDROGEL anon trade → OTM cluster within 6000 ticks,
        #  take a directional VEV position (long if 5300/5400, short if 5500+).
        #  Kept small (30% size) since vol selling is the main edge.
        # ════════════════════════════════════════════════════════════════════

        hydrogel_age  = t - hydrogel_sig_ts
        hydrogel_valid = hydrogel_age <= CFG["hydrogel_window"]

        if otm_signal_ts < 0 or (t - otm_signal_ts) > CFG["signal_exit_ticks"]:
            for trd in state.market_trades.get(HYDROGEL, []):
                if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                    hydrogel_sig_ts = t
                    hydrogel_age = 0
                    hydrogel_valid = True
                    break

        otm_bought: set = set()
        OTM_SIGNAL_VOUCHERS = ["VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"]
        for v in OTM_SIGNAL_VOUCHERS:
            for trd in state.market_trades.get(v, []):
                if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                    otm_bought.add(v)
                    break

        new_signal = (
            len(otm_bought) >= CFG["min_otm_strikes"]
            and hydrogel_valid
            and t >= cooldown_until
        )
        if new_signal:
            otm_signal_ts = t
            has_low = any(v in otm_bought for v in ["VEV_5300", "VEV_5400"])
            signal_dir = "long" if has_low else "short"
            vev_entry_px = None

        signal_age    = t - otm_signal_ts
        in_cooldown   = t < cooldown_until
        signal_active = (signal_age <= CFG["signal_exit_ticks"]) and not in_cooldown

        signal_orders: List[Order] = []
        if od_vev is not None and vev_ema is not None and signal_active:
            if vev_entry_px is None:
                vev_entry_px = vev_ema
            price_gain = ((vev_ema - vev_entry_px) if signal_dir == "long"
                          else (vev_entry_px - vev_ema))
            hit_exit = (price_gain >= CFG["profit_exit"] or price_gain <= -CFG["loss_exit"])
            if hit_exit:
                signal_active  = False
                vev_entry_px   = None
                otm_signal_ts  = -(10**9)
                signal_dir     = None
                if price_gain <= -CFG["loss_exit"]:
                    cooldown_until = t + CFG["cooldown_ticks"]

        if signal_active and od_vev is not None:
            sig_target = round(LIMITS[VEV] * CFG["signal_size"])
            if signal_dir == "short":
                sig_target = -sig_target
            sig_gap = sig_target - vev_pos

            if sig_gap > 0:
                ask = _best_ask(od_vev)
                if ask is not None:
                    qty = min(sig_gap, LIMITS[VEV] - vev_pos)
                    if qty > 0:
                        signal_orders.append(Order(VEV, ask, qty))
            elif sig_gap < 0:
                bid = _best_bid(od_vev)
                if bid is not None:
                    qty = min(-sig_gap, LIMITS[VEV] + vev_pos)
                    if qty > 0:
                        signal_orders.append(Order(VEV, bid, -qty))

        # ════════════════════════════════════════════════════════════════════
        #  LAYER 5: VEV MARKET MAKING (residual, when no signal)
        # ════════════════════════════════════════════════════════════════════

        if od_vev is not None and vev_ema is not None:
            if signal_active:
                result[VEV] = signal_orders or hedge_orders
            elif hedge_orders:
                # Delta hedge takes priority over pure MM
                result[VEV] = hedge_orders
            else:
                # Normal MM
                result[VEV] = _mm_orders(
                    VEV, od_vev, vev_pos, vev_ema, LIMITS[VEV],
                    CFG["vev_order_size"], CFG["inv_limit"], CFG["inv_hard"],
                )

        # ════════════════════════════════════════════════════════════════════
        #  LAYER 6: HYDROGEL MARKET MAKING
        # ════════════════════════════════════════════════════════════════════

        if HYDROGEL in state.order_depths and hydrogel_ema is not None:
            od_h = state.order_depths[HYDROGEL]
            pos_h = state.position.get(HYDROGEL, 0)
            result[HYDROGEL] = _mm_orders(
                HYDROGEL, od_h, pos_h, hydrogel_ema, LIMITS[HYDROGEL],
                CFG["hydrogel_order_size"], CFG["hydrogel_inv_limit"],
                CFG["hydrogel_inv_limit"] * 2, make_w=4, take_w=2,
            )

        # ── Serialize ─────────────────────────────────────────────────────────
        new_ts = json.dumps({
            "vev_ema":         vev_ema,
            "hydrogel_ema":    hydrogel_ema,
            "otm_signal_ts":   otm_signal_ts,
            "hydrogel_sig_ts": hydrogel_sig_ts,
            "vev_entry_px":    vev_entry_px,
            "cooldown_until":  cooldown_until,
            "signal_dir":      signal_dir,
        })
        logger.flush(state, result, conversions, new_ts)
        return result, conversions, new_ts