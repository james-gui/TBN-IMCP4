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
TTE_DAYS_START = 5        # days remaining at start of round 3
TICKS_PER_DAY  = 100_000

# ─── Config ──────────────────────────────────────────────────────────────────
CFG = {
    "ema_alpha":           0.15,
    "make_width":          2,
    "take_width":          1,
    "order_size":          20,
    "inv_limit":           80,
    "inv_hard":            160,
    # Options
    "default_vol":         0.5,   # annual σ fallback
    "vol_alpha":           0.05,  # slow EMA for ATM implied vol
    "voucher_edge":        2,     # min BS edge to take a voucher
    # Signal
    "min_otm_strikes":     3,     # simultaneous OTM buys needed to confirm
    "hydrogel_window":     5000,  # ticks HYDROGEL signal remains valid
    "signal_exit_ticks":   5000,  # abandon signal after this long
    "profit_exit":         5,     # exit VEV long after this gain
    # Position sizing by signal strength (fraction of limit)
    "size_weak":           0.50,  # OTM signal only
    "size_strong":         1.00,  # OTM + HYDROGEL signal
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price (r=0)."""
    if T <= 1e-9 or S <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def implied_vol(market_price: float, S: float, K: float, T: float) -> Optional[float]:
    """Binary-search implied vol from an observed call price."""
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

        t = state.timestamp

        # ── TTE (decreases as timestamp increases within the round) ───────────
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
        # Use single vol calibrated from the option closest to ATM; no vol surface.
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

        # ── Detect HYDROGEL insider signal (no-ID trades only) ───────────────
        for trd in state.market_trades.get(HYDROGEL, []):
            if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                hydrogel_sig_ts = t
                break

        # ── Detect OTM cross-strike signal (no-ID trades only) ───────────────
        # Fingerprint: ≥3 OTM vouchers bought by entity with no buyer/seller ID
        otm_bought: set = set()
        for v in OTM_VOUCHERS:
            for trd in state.market_trades.get(v, []):
                if trd.quantity > 0 and trd.buyer == "" and trd.seller == "":
                    otm_bought.add(v)
                    break
        if len(otm_bought) >= CFG["min_otm_strikes"]:
            otm_signal_ts = t

        # ── Signal state ──────────────────────────────────────────────────────
        signal_age    = t - otm_signal_ts
        hydrogel_age  = t - hydrogel_sig_ts
        signal_active = signal_age <= CFG["signal_exit_ticks"]
        signal_strong = signal_active and hydrogel_age <= CFG["hydrogel_window"]
        size_frac     = CFG["size_strong"] if signal_strong else (CFG["size_weak"] if signal_active else 0.0)

        # ── HYDROGEL: watch-only (no MM — trends too hard, bleeds position) ──
        # hydrogel_ema still updated above for signal context

        # ── Trade VEV (MM + signal long) ──────────────────────────────────────
        if VEV in state.order_depths and vev_ema is not None:
            od  = state.order_depths[VEV]
            pos = state.position.get(VEV, 0)
            fv  = vev_ema

            if signal_active:
                # Exit on profit target
                price_gain = (fv - vev_entry_price) if vev_entry_price is not None else 0.0
                if price_gain >= CFG["profit_exit"]:
                    signal_active = False
                    vev_entry_price = None

            if signal_active and od.sell_orders:
                if vev_entry_price is None:
                    vev_entry_price = fv
                target_pos = round(LIMITS[VEV] * size_frac)
                orders: List[Order] = []
                for ask_px in sorted(od.sell_orders.keys()):
                    if pos >= target_pos:
                        break
                    if ask_px <= fv + 5:
                        qty = min(-od.sell_orders[ask_px], target_pos - pos)
                        orders.append(Order(VEV, ask_px, qty))
                        pos += qty
                    else:
                        break
                result[VEV] = orders
            else:
                if not signal_active:
                    vev_entry_price = None
                result[VEV] = _mm_orders(VEV, od, pos, fv, LIMITS[VEV])

        # ── Trade Vouchers (BS arb + signal shadow) ───────────────────────────
        if vev_ema is not None:
            for voucher in VOUCHERS:
                od = state.order_depths.get(voucher)
                if od is None:
                    continue
                pos  = state.position.get(voucher, 0)
                K    = STRIKES[voucher]
                fair = bs_call(vev_ema, K, tte_years, implied_sigma)
                orders = []
                buy_cap  = LIMITS[voucher] - pos
                sell_cap = LIMITS[voucher] + pos

                if not od.buy_orders or not od.sell_orders:
                    result[voucher] = orders
                    continue

                # Signal shadow: buy exact OTM strikes the insider bought
                if signal_active and voucher in otm_bought:
                    target_pos = round(LIMITS[voucher] * size_frac)
                    for ask_px in sorted(od.sell_orders.keys()):
                        if pos >= target_pos:
                            break
                        qty = min(-od.sell_orders[ask_px], target_pos - pos)
                        orders.append(Order(voucher, ask_px, qty))
                        pos += qty
                    result[voucher] = orders
                    continue

                # No BS arb — vouchers only traded on insider signal
                result[voucher] = []

        # ── Serialize state ───────────────────────────────────────────────────
        new_ts = json.dumps({
            "vev_ema":         vev_ema,
            "hydrogel_ema":    hydrogel_ema,
            "implied_sigma":   implied_sigma,
            "otm_signal_ts":   otm_signal_ts,
            "hydrogel_sig_ts": hydrogel_sig_ts,
            "vev_entry_price": vev_entry_price,
        })
        logger.flush(state, result, conversions, new_ts)
        return result, conversions, new_ts
