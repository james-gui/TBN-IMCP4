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

# ─── Products ────────────────────────────────────────────────────────────────
HYDROGEL     = "HYDROGEL_PACK"
VEV          = "VELVETFRUIT_EXTRACT"
VOUCHERS     = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
OTM_VOUCHERS = ["VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"]
STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000, "VEV_6500": 6500,
}
LIMITS = {HYDROGEL: 200, VEV: 200}
for _v in VOUCHERS:
    LIMITS[_v] = 300

# ─── Round 3 TTE ─────────────────────────────────────────────────────────────
TTE_DAYS_START = 5
TICKS_PER_DAY  = 100_000

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "ema_alpha":           0.15,
    "make_width":          2,
    "take_width":          1,
    "order_size":          15,
    "inv_limit":           50,
    "inv_hard":            100,
    # Options
    "default_vol":         0.5,
    "vol_alpha":           0.05,
    "voucher_edge":        2,
    # ── Signal (FIXED) ───────────────────────────────────────────────────────
    # REQUIREMENT: HYDROGEL anon trade must precede OTM cluster within this window
    # to confirm the trade is a TRUE insider signal (not MM rebalancing noise).
    # Removing size_weak entirely — false positives without HYDROGEL precursor
    # are noise and the strategy was LOSING on them in v8.
    "min_otm_strikes":     3,     # simultaneous OTM buys needed to confirm
    "hydrogel_window":     6000,  # ticks HYDROGEL signal remains valid (was 5000, widened slightly)
    "signal_exit_ticks":   4000,  # ticks to stay in signal position after OTM cluster
    # ── Direction rule (FIXED from v8) ───────────────────────────────────────
    # Insider BUYS OTM CALLS = bullish on VEV. We follow them LONG.
    # Exception: if ONLY VEV_5500+ in the basket (higher strike = farther OTM
    # with no 5300/5400) -> empirically VEV falls after -> go SHORT.
    # v8 was going SHORT on ALL signals which is wrong 8/10 of the time.
    "bearish_min_strike":  "VEV_5500",  # if min_strike >= this and no 5300/5400 -> SHORT
    "signal_size":         0.75,  # fraction of VEV limit to use (150 units)
    "profit_exit":         8,
    "loss_exit":           8,
    "cooldown_ticks":      10000,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or S <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def implied_vol(market_price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic or T <= 1e-9:
        return None
    lo, hi = 0.01, 10.0
    for _ in range(40):
        mid_v = (lo + hi) / 2.0
        if bs_call(S, K, T, mid_v) > market_price:
            hi = mid_v
        else:
            lo = mid_v
    return (lo + hi) / 2.0


def _wall_mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bid = max(od.buy_orders, key=od.buy_orders.get)
    ask = min(od.sell_orders, key=lambda k: od.sell_orders[k])
    return (bid + ask) / 2.0


def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def _mm_orders(
    symbol: str,
    od: OrderDepth,
    pos: int,
    fv: float,
    limit: int,
) -> List[Order]:
    """Standard EMA market-making: take mispriced levels, post inside spread."""
    orders: List[Order] = []
    if not od.buy_orders or not od.sell_orders:
        return orders

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    remaining_buy  = limit - pos
    remaining_sell = limit + pos
    take_w  = CFG["take_width"]
    make_w  = CFG["make_width"]
    base    = CFG["order_size"]
    inv_lim = CFG["inv_limit"]
    inv_hard = CFG["inv_hard"]

    for ask_px in sorted(od.sell_orders.keys()):
        if remaining_buy <= 0:
            break
        if ask_px <= fv - take_w:
            qty = min(-od.sell_orders[ask_px], remaining_buy)
            orders.append(Order(symbol, ask_px, qty))
            remaining_buy -= qty
        else:
            break

    for bid_px in sorted(od.buy_orders.keys(), reverse=True):
        if remaining_sell <= 0:
            break
        if bid_px >= fv + take_w:
            qty = min(od.buy_orders[bid_px], remaining_sell)
            orders.append(Order(symbol, bid_px, -qty))
            remaining_sell -= qty
        else:
            break

    inv_ratio = pos / limit
    skew = -round(inv_ratio * 2)
    our_bid = round(fv) + skew - make_w
    our_ask = round(fv) + skew + make_w
    our_bid = min(our_bid, best_ask - 1)
    our_ask = max(our_ask, best_bid + 1)

    if our_bid < our_ask:
        bid_size = ask_size = base
        if pos > inv_lim:
            scale    = 1.0 - (pos - inv_lim) / max(inv_hard - inv_lim, 1)
            bid_size = max(1, round(base * scale))
        elif pos < -inv_lim:
            scale    = 1.0 - ((-pos) - inv_lim) / max(inv_hard - inv_lim, 1)
            ask_size = max(1, round(base * scale))

        if pos < inv_hard and remaining_buy > 0 and our_bid < fv:
            orders.append(Order(symbol, our_bid, min(bid_size, remaining_buy)))
        if pos > -inv_hard and remaining_sell > 0 and our_ask > fv:
            orders.append(Order(symbol, our_ask, -min(ask_size, remaining_sell)))

    return orders


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # ── Deserialize state ─────────────────────────────────────────────────
        try:
            ts = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            ts = {}

        vev_ema          = ts.get("vev_ema", None)
        hydrogel_ema     = ts.get("hydrogel_ema", None)
        implied_sigma    = ts.get("implied_sigma", CFG["default_vol"])
        otm_signal_ts    = ts.get("otm_signal_ts", -(10 ** 9))
        hydrogel_sig_ts  = ts.get("hydrogel_sig_ts", -(10 ** 9))
        vev_entry_price  = ts.get("vev_entry_price", None)
        cooldown_until   = ts.get("cooldown_until", -(10 ** 9))
        # NEW: remember which direction we entered the signal trade
        signal_direction = ts.get("signal_direction", None)  # "long" or "short"
        # NEW: remember the min OTM strike from when signal fired
        signal_min_strike = ts.get("signal_min_strike", None)

        t = state.timestamp

        # ── TTE ───────────────────────────────────────────────────────────────
        tte_days  = max(TTE_DAYS_START - t / TICKS_PER_DAY, 0.0)
        tte_years = tte_days / 365.0

        # ── Update EMAs ───────────────────────────────────────────────────────
        alpha = CFG["ema_alpha"]
        if VEV in state.order_depths:
            wm = _wall_mid(state.order_depths[VEV])
            if wm is not None:
                vev_ema = wm if vev_ema is None else alpha * wm + (1 - alpha) * vev_ema

        if HYDROGEL in state.order_depths:
            wm = _wall_mid(state.order_depths[HYDROGEL])
            if wm is not None:
                hydrogel_ema = wm if hydrogel_ema is None else alpha * wm + (1 - alpha) * hydrogel_ema

        # ── Update implied vol from near-ATM voucher ──────────────────────────
        if vev_ema is not None and tte_years > 0:
            atm_v = min(VOUCHERS, key=lambda v: abs(STRIKES[v] - vev_ema))
            od_atm = state.order_depths.get(atm_v)
            if od_atm:
                mkt_mid = _mid(od_atm)
                if mkt_mid is not None and mkt_mid > 0:
                    iv = implied_vol(mkt_mid, vev_ema, STRIKES[atm_v], tte_years)
                    if iv is not None:
                        va = CFG["vol_alpha"]
                        implied_sigma = va * iv + (1 - va) * implied_sigma

        # ── Detect HYDROGEL insider signal ────────────────────────────────────
        # Anonymous HYDROGEL trade = Step 1 of insider sequence.
        # Only update if we're not already in a signal (avoid overwriting).
        if otm_signal_ts < 0 or (t - otm_signal_ts) > CFG["signal_exit_ticks"]:
            for trd in state.market_trades.get(HYDROGEL, []):
                if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                    hydrogel_sig_ts = t
                    break

        # ── Detect OTM cross-strike cluster ───────────────────────────────────
        # FIX: Only fire if HYDROGEL precursor seen within hydrogel_window.
        # This eliminates the false positive MM-rebalancing clusters entirely.
        hydrogel_age = t - hydrogel_sig_ts
        hydrogel_valid = hydrogel_age <= CFG["hydrogel_window"]

        otm_bought: set = set()
        for v in OTM_VOUCHERS:
            for trd in state.market_trades.get(v, []):
                if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                    otm_bought.add(v)
                    break

        # Only accept the OTM cluster as a REAL signal when HYDROGEL preceded it
        new_signal = (
            len(otm_bought) >= CFG["min_otm_strikes"]
            and hydrogel_valid
            and t >= cooldown_until
        )

        if new_signal:
            otm_signal_ts = t
            # Determine direction: BULLISH unless basket is VEV_5500+ only
            min_otm_bought = min(otm_bought)
            signal_min_strike = min_otm_bought
            # FIX: Follow the insider (buying calls = bullish = LONG VEV)
            # Exception: min_strike >= VEV_5500 with no 5300/5400 -> bearish
            has_low_strike = any(v in otm_bought for v in ["VEV_5300", "VEV_5400"])
            if has_low_strike:
                signal_direction = "long"
            else:
                signal_direction = "short"
            vev_entry_price = None  # will be set on first fill

        # ── Signal state ──────────────────────────────────────────────────────
        signal_age    = t - otm_signal_ts
        in_cooldown   = t < cooldown_until
        signal_active = (signal_age <= CFG["signal_exit_ticks"]) and not in_cooldown

        # ── VEV: MM + CORRECTED signal direction ─────────────────────────────
        if VEV in state.order_depths and vev_ema is not None:
            od  = state.order_depths[VEV]
            pos = state.position.get(VEV, 0)
            fv  = vev_ema

            # Check exit conditions for open signal position
            if signal_active and vev_entry_price is not None:
                price_gain = (
                    (fv - vev_entry_price) if signal_direction == "long"
                    else (vev_entry_price - fv)
                )
                hit_profit = price_gain >= CFG["profit_exit"]
                hit_stop   = price_gain <= -CFG["loss_exit"]
                if hit_profit or hit_stop:
                    signal_active    = False
                    vev_entry_price  = None
                    otm_signal_ts    = -(10 ** 9)
                    signal_direction = None
                    if hit_stop:
                        cooldown_until = t + CFG["cooldown_ticks"]

            if signal_active:
                if vev_entry_price is None:
                    vev_entry_price = fv
                target_pos = round(LIMITS[VEV] * CFG["signal_size"])
                if signal_direction == "short":
                    target_pos = -target_pos

                orders: List[Order] = []
                if signal_direction == "long" and od.sell_orders:
                    # BUY aggressively into ask side
                    for ask_px in sorted(od.sell_orders.keys()):
                        if pos >= target_pos:
                            break
                        if ask_px <= fv + 5:
                            qty = min(-od.sell_orders[ask_px], target_pos - pos)
                            orders.append(Order(VEV, ask_px, qty))
                            pos += qty
                        else:
                            break

                elif signal_direction == "short" and od.buy_orders:
                    # SELL aggressively into bid side
                    for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                        if pos <= target_pos:
                            break
                        if bid_px >= fv - 5:
                            qty = min(od.buy_orders[bid_px], pos - target_pos)
                            orders.append(Order(VEV, bid_px, -qty))
                            pos -= qty
                        else:
                            break

                result[VEV] = orders
            else:
                if not signal_active:
                    vev_entry_price  = None
                    signal_direction = None
                result[VEV] = _mm_orders(VEV, od, pos, fv, LIMITS[VEV])

        # ── HYDROGEL: market-make (re-enabled) ───────────────────────────────
        # v8 disabled this; HYDROGEL has a declining trend but MM with proper
        # inventory skew still earns on the spread. The key is to NOT hold large
        # directional inventory — let skew flatten the book.
        if HYDROGEL in state.order_depths and hydrogel_ema is not None:
            od  = state.order_depths[HYDROGEL]
            pos = state.position.get(HYDROGEL, 0)
            result[HYDROGEL] = _mm_orders(HYDROGEL, od, pos, hydrogel_ema, LIMITS[HYDROGEL])

        # ── Vouchers: BS-edge MM on ITM/ATM, ignore deep OTM ─────────────────
        # FIX: Do NOT short OTM vouchers on signal. They're all worthless at
        # expiry (VEV < 5300 all day), so shorting them gives 0 edge and
        # creates unnecessary inventory risk. Instead:
        # - ITM vouchers (4000, 4500, 5000, 5100, 5200): market-make using BS fair value
        # - Deep OTM (5300+): skip entirely (nearly zero value, wide spreads)
        if vev_ema is not None and tte_years > 0:
            for voucher in VOUCHERS:
                od = state.order_depths.get(voucher)
                if od is None:
                    continue
                pos   = state.position.get(voucher, 0)
                K     = STRIKES[voucher]
                orders = []

                # Skip deep OTM (5300+): no edge, all nearly worthless this round
                if K >= 5300:
                    result[voucher] = orders
                    continue

                if not od.buy_orders or not od.sell_orders:
                    result[voucher] = orders
                    continue

                fair_v = bs_call(vev_ema, K, tte_years, implied_sigma)
                edge   = CFG["voucher_edge"]
                limit  = LIMITS[voucher]
                rem_buy  = limit - pos
                rem_sell = limit + pos

                # Take mispriced asks
                for ask_px in sorted(od.sell_orders.keys()):
                    if rem_buy <= 0:
                        break
                    if ask_px <= fair_v - edge:
                        qty = min(-od.sell_orders[ask_px], rem_buy)
                        orders.append(Order(voucher, ask_px, qty))
                        rem_buy -= qty
                    else:
                        break

                # Take mispriced bids
                for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                    if rem_sell <= 0:
                        break
                    if bid_px >= fair_v + edge:
                        qty = min(od.buy_orders[bid_px], rem_sell)
                        orders.append(Order(voucher, bid_px, -qty))
                        rem_sell -= qty
                    else:
                        break

                result[voucher] = orders

        # ── Serialize state ───────────────────────────────────────────────────
        new_ts = json.dumps({
            "vev_ema":           vev_ema,
            "hydrogel_ema":      hydrogel_ema,
            "implied_sigma":     implied_sigma,
            "otm_signal_ts":     otm_signal_ts,
            "hydrogel_sig_ts":   hydrogel_sig_ts,
            "vev_entry_price":   vev_entry_price,
            "cooldown_until":    cooldown_until,
            "signal_direction":  signal_direction,
            "signal_min_strike": signal_min_strike,
        })
        logger.flush(state, result, conversions, new_ts)
        return result, conversions, new_ts