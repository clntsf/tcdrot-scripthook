"""
Long-Vol-First Trading Module for RITC 2026

Strategy:
- Primary: buy underpriced ATM straddles (long vol / long gamma)
- Secondary: optionally add small covered far-OTM shorts only under strict caps
- Hedge RTM delta with adaptive moneyness-based cadence
"""

import math
import re
import time
from typing import Optional

from api import TradingAPI
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# -- Config ---------------------------------------------------------------------

STRIKES = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
CALL_TICKERS = [f"RTM1C{s}" for s in STRIKES]
PUT_TICKERS = [f"RTM1P{s}" for s in STRIKES]
ALL_OPTION_TICKERS = CALL_TICKERS + PUT_TICKERS

DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
API_PORT = getenv("ROT_API_PORT")

RISK_FREE_RATE = 0.0
TRADING_YEAR_DAYS = 240
MONTH_DAYS = 20
OPTION_MULTIPLIER = 100  # shares per contract (100:1)

MAX_BATCH_SIZE = 100      # max contracts per single option order (API limit)
RTM_MAX_ORDER = 10000     # max shares per single RTM order (API limit)
RTM_SHARE_LIMIT = 50000   # RTM net position limit
OPTIONS_NET_LIMIT = 1000  # options net limit (|long - short| contracts)

DEFAULT_DELTA_LIMIT = 1000
SUBHEAT_TICKS = 75
UNWIND_AT_TICK = 59

BASE_QTY_CALL = 400
BASE_QTY_PUT = 400
BASE_QTY_SINGLE_LEG = 200
LONG_STRADDLE_MIN_EDGE = 0.05
SINGLE_LEG_MIN_EDGE = 0.04
SINGLE_LEG_MIN_ABS_DELTA = 0.35
SINGLE_LEG_MAX_ABS_DELTA = 0.65
COVERED_SHORT_MIN_EDGE = 0.10
TAIL_SHORT_MAX_ABS_DELTA = 0.10
SHORT_CONTRACT_CAP_FRAC = 0.25
SHORT_VEGA_CAP_FRAC = 0.20
TAIL_SHORT_TARGET_QTY = 50

HEDGE_BAND_ATM_FRAC = 0.50
HEDGE_BAND_MID_FRAC = 0.75
HEDGE_BAND_DEEP_FRAC = 0.95
ATM_HEAVY_OTM = 1.0
DEEP_OTM = 3.0

DELTA_STRESS_EXIT_FRAC = 0.95
DELTA_STRESS_EXIT_TICKS = 2
MAX_CONSEC_EXEC_FAILURES = 3

# -- Shared state ----------------------------------------------------------------

_state: dict = {
    "rtm_price": 50.0,
    "tte": MONTH_DAYS / TRADING_YEAR_DAYS,
    "rtm_position": 0,
    "positions": {},
    "realized_vol": 0.35,
    "delta_limit": DEFAULT_DELTA_LIMIT,
    "penalty_pct": 0.0,
    "last_news_id": 0,
    "case_status": "STOPPED",
    "ticks_left": 9999,
    "case_tick": 0,
    "subheat_id": -1,
    "current_subheat_id": -1,
    "built_this_subheat": False,
    "stress_ticks_over_delta_band": 0,
    "last_entry_strike": None,
    "execution_failures": 0,
    "last_straddle_edge": 0.0,
    "tail_short_contract_util": 0.0,
    "tail_short_vega_util": 0.0,
    "hedge_band_frac": HEDGE_BAND_MID_FRAC,
    "mode": "WAIT",
}

# -- Black-Scholes ----------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun polynomial approximation."""
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    sigma = max(sigma, 1e-6)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if is_call:
        return max(0.0, S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))
    return max(0.0, K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1))


def _bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 0:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sigma = max(sigma, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    nd1 = _norm_cdf(d1)
    return nd1 if is_call else nd1 - 1.0


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega per 1.00 volatility unit."""
    if T <= 0:
        return 0.0
    sigma = max(sigma, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma)
    return S * math.sqrt(T) * _norm_pdf(d1)


def _implied_vol(
    mkt: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    tol: float = 0.0001,
    max_iter: int = 100,
) -> Optional[float]:
    if T <= 0:
        return None
    intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
    if mkt < intrinsic - tol:
        return None
    lo, hi = 0.001, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = _bs_price(S, K, T, r, mid, is_call)
        if abs(p - mkt) < tol:
            return mid
        if p < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# -- Helpers ---------------------------------------------------------------------


def _parse_ticker(ticker: str) -> tuple:
    """Returns (is_call: bool, strike: float)."""
    is_call = "C" in ticker
    strike = float(ticker.split("C" if is_call else "P")[1])
    return is_call, strike


def _option_model_inputs(ticker: str, mkt: Optional[float]) -> tuple:
    """Returns (sigma, abs_delta, vega_per_contract, edge, strike, is_call)."""
    S = _state["rtm_price"]
    T = _state["tte"]
    rv = _state["realized_vol"]
    is_call, K = _parse_ticker(ticker)

    sigma = rv
    if mkt is not None:
        iv = _implied_vol(mkt, S, K, T, RISK_FREE_RATE, is_call)
        if iv is not None:
            sigma = iv

    fair = _bs_price(S, K, T, RISK_FREE_RATE, rv, is_call)
    edge = fair - (mkt if mkt is not None else fair)
    abs_delta = abs(_bs_delta(S, K, T, RISK_FREE_RATE, sigma, is_call))
    vega_per_contract = _bs_vega(S, K, T, RISK_FREE_RATE, sigma) * OPTION_MULTIPLIER
    return sigma, abs_delta, vega_per_contract, edge, K, is_call


# -- News ------------------------------------------------------------------------


def _poll_news(tapi: TradingAPI) -> None:
    """Fetch new news items and update state (delta limit, realized vol)."""
    try:
        since = _state["last_news_id"] if _state["last_news_id"] > 0 else None
        items = tapi.get_news(since=since)
    except Exception as e:
        print(f"[news error] {e}")
        return

    if not items:
        return

    for item in items:
        nid = item.get("news_id", 0)
        headline = (item.get("headline") or "").lower()
        body = item.get("body") or ""

        _state["last_news_id"] = max(_state["last_news_id"], nid)

        if "delta limit" in headline:
            m_limit = re.search(r"\s([0-9]+)\s", body)
            m_penalty = re.search(r"([\d.]+)\s*%", body)
            if m_limit:
                _state["delta_limit"] = int(m_limit.group(1))
            if m_penalty:
                _state["penalty_pct"] = float(m_penalty.group(1)) / 100
            print(
                f"[news] delta_limit={_state['delta_limit']}"
                f"  penalty={_state['penalty_pct']:.1%}"
            )
            continue

        pcts = re.findall(r"([\d.]+)\s*%", body)
        if len(pcts) == 1:
            vol = float(pcts[0]) / 100
            _state["realized_vol"] = vol
            print(f"[news] realized_vol -> {vol:.1%}")


# -- State sync ------------------------------------------------------------------


def _sync_state(tapi: TradingAPI) -> dict:
    """Fetch market data, update _state, return {ticker: mid_price} for options."""
    case = tapi.get_case()
    _state["case_status"] = case.get("status", "STOPPED")
    tick = case["tick"]
    ticks_left = case["ticks_per_period"] - tick
    _state["ticks_left"] = ticks_left
    _state["case_tick"] = tick
    _state["subheat_id"] = (tick - 1) // SUBHEAT_TICKS

    ticks_left_in_subheat = SUBHEAT_TICKS - ((tick - 1) % SUBHEAT_TICKS)
    _state["tte"] = max(0.0, ticks_left_in_subheat / SUBHEAT_TICKS) * (MONTH_DAYS / TRADING_YEAR_DAYS)

    securities = tapi.get_securities()
    prices: dict = {}
    new_positions: dict = {}

    for sec in securities:
        ticker = sec["ticker"]
        bid, ask, last = sec.get("bid"), sec.get("ask"), sec.get("last")
        mid = (bid + ask) / 2 if bid and ask else (last or 0.0)

        if ticker == "RTM":
            _state["rtm_price"] = mid
            _state["rtm_position"] = int(sec.get("position", 0))
        elif ticker in ALL_OPTION_TICKERS:
            prices[ticker] = mid
            qty = int(sec.get("position", 0))
            if qty != 0:
                new_positions[ticker] = qty

    _state["positions"] = new_positions
    _state["prices"] = prices
    return prices


# -- Portfolio analytics ---------------------------------------------------------


def _portfolio_delta() -> float:
    """Total signed delta in underlying shares (options + RTM hedge)."""
    S = _state["rtm_price"]
    T = _state["tte"]
    rv = _state["realized_vol"]
    prices = _state.get("prices", {})

    opt_delta = 0.0
    for ticker, qty in _state["positions"].items():
        is_call, K = _parse_ticker(ticker)
        sigma = rv
        mkt = prices.get(ticker)
        if mkt is not None:
            iv = _implied_vol(mkt, S, K, T, RISK_FREE_RATE, is_call)
            if iv is not None:
                sigma = iv
        d = _bs_delta(S, K, T, RISK_FREE_RATE, sigma, is_call)
        opt_delta += qty * d * OPTION_MULTIPLIER

    return opt_delta + _state["rtm_position"]


def _aggregate_option_exposure() -> dict:
    """Aggregate long/short contract and vega exposure, and average |K-S|."""
    S = _state["rtm_price"]
    prices = _state.get("prices", {})

    long_contracts = 0
    short_contracts = 0
    short_calls = 0
    short_puts = 0
    long_vega = 0.0
    short_vega = 0.0
    weighted_otm_numer = 0.0
    weighted_otm_denom = 0.0

    for ticker, qty in _state["positions"].items():
        mkt = prices.get(ticker)
        _, _, vega_pc, _, K, is_call = _option_model_inputs(ticker, mkt)
        size = abs(qty)
        weighted_otm_numer += abs(K - S) * size
        weighted_otm_denom += size

        if qty > 0:
            long_contracts += qty
            long_vega += qty * vega_pc
        elif qty < 0:
            sq = -qty
            short_contracts += sq
            short_vega += sq * vega_pc
            if is_call:
                short_calls += sq
            else:
                short_puts += sq

    avg_otm = weighted_otm_numer / weighted_otm_denom if weighted_otm_denom > 0 else 0.0
    return {
        "long_contracts": long_contracts,
        "short_contracts": short_contracts,
        "short_calls": short_calls,
        "short_puts": short_puts,
        "long_vega": long_vega,
        "short_vega": short_vega,
        "avg_otm": avg_otm,
    }


def _tail_short_utilization(exposure: dict) -> tuple:
    long_contracts = exposure["long_contracts"]
    long_vega = exposure["long_vega"]
    if long_contracts <= 0 or long_vega <= 0:
        return 0.0, 0.0, 0, 0.0, 0, 0.0

    cap_contracts = int(math.floor(SHORT_CONTRACT_CAP_FRAC * long_contracts))
    cap_vega = SHORT_VEGA_CAP_FRAC * long_vega

    short_contracts = exposure["short_contracts"]
    short_vega = exposure["short_vega"]

    c_util = short_contracts / cap_contracts if cap_contracts > 0 else 0.0
    v_util = short_vega / cap_vega if cap_vega > 0 else 0.0

    return c_util, v_util, short_contracts, short_vega, cap_contracts, cap_vega


def _hedge_band_fraction() -> float:
    exposure = _aggregate_option_exposure()
    avg_otm = exposure["avg_otm"]
    if avg_otm <= ATM_HEAVY_OTM:
        return HEDGE_BAND_ATM_FRAC
    if avg_otm >= DEEP_OTM:
        return HEDGE_BAND_DEEP_FRAC
    return HEDGE_BAND_MID_FRAC


# -- Signals ---------------------------------------------------------------------


def _straddle_candidates(prices: dict) -> list:
    """Return candidate dicts with straddle edge, ranked ATM-first then edge."""
    S = _state["rtm_price"]
    out = []

    for strike in STRIKES:
        c_ticker = f"RTM1C{strike}"
        p_ticker = f"RTM1P{strike}"
        c_mkt = prices.get(c_ticker)
        p_mkt = prices.get(p_ticker)
        if c_mkt is None or p_mkt is None:
            continue

        _, _, _, c_edge, _, _ = _option_model_inputs(c_ticker, c_mkt)
        _, _, _, p_edge, _, _ = _option_model_inputs(p_ticker, p_mkt)
        straddle_edge = c_edge + p_edge
        out.append(
            {
                "strike": float(strike),
                "distance": abs(float(strike) - S),
                "call_ticker": c_ticker,
                "put_ticker": p_ticker,
                "call_edge": c_edge,
                "put_edge": p_edge,
                "straddle_edge": straddle_edge,
            }
        )

    out.sort(key=lambda x: (x["distance"], -x["straddle_edge"]))
    return out


def _best_underpriced_straddle(prices: dict) -> Optional[dict]:
    for c in _straddle_candidates(prices):
        if c["straddle_edge"] >= LONG_STRADDLE_MIN_EDGE:
            return c
    return None


def _best_underpriced_single_leg(prices: dict) -> Optional[dict]:
    """Fallback when no straddle qualifies: pick near-ATM underpriced single leg."""
    S = _state["rtm_price"]
    candidates = []

    for ticker, mkt in prices.items():
        _, abs_delta, _, edge, K, is_call = _option_model_inputs(ticker, mkt)
        if edge < SINGLE_LEG_MIN_EDGE:
            continue
        if not (SINGLE_LEG_MIN_ABS_DELTA <= abs_delta <= SINGLE_LEG_MAX_ABS_DELTA):
            continue
        candidates.append(
            {
                "ticker": ticker,
                "edge": edge,
                "abs_delta": abs_delta,
                "distance": abs(K - S),
                "strike": K,
                "side": "CALL" if is_call else "PUT",
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["distance"], abs(x["abs_delta"] - 0.50), -x["edge"]))
    return candidates[0]


def _covered_tail_short_candidates(prices: dict, side: Optional[str] = None) -> list:
    """
    Return overpriced far-OTM short candidates with low absolute delta.
    side: None|'CALL'|'PUT'
    """
    candidates = []

    for ticker, mkt in prices.items():
        _, abs_delta, vega_pc, edge, K, is_call = _option_model_inputs(ticker, mkt)
        if edge > -COVERED_SHORT_MIN_EDGE:
            continue
        if abs_delta > TAIL_SHORT_MAX_ABS_DELTA:
            continue

        opt_side = "CALL" if is_call else "PUT"
        if side is not None and opt_side != side:
            continue

        candidates.append(
            {
                "ticker": ticker,
                "edge": edge,
                "abs_delta": abs_delta,
                "vega_pc": vega_pc,
                "distance": abs(K - _state["rtm_price"]),
                "side": opt_side,
            }
        )

    candidates.sort(key=lambda x: (x["edge"], -x["distance"]))
    return candidates


# -- Execution -------------------------------------------------------------------


def _execute_trade(tapi: TradingAPI, ticker: str, action: str, qty: int) -> bool:
    try:
        tapi.post_order(ticker, "MARKET", qty, action)
        sign = 1 if action == "BUY" else -1
        _state["positions"][ticker] = _state["positions"].get(ticker, 0) + sign * qty
        if _state["positions"][ticker] == 0:
            _state["positions"].pop(ticker, None)
        print(f"  {action:4s} {qty:3d}x {ticker}")
        return True
    except Exception as e:
        print(f"  [order error] {action} {qty}x {ticker}: {e}")
        return False


def _execute_batched_option_order(tapi: TradingAPI, ticker: str, action: str, total_qty: int) -> bool:
    if total_qty <= 0:
        return True

    sent = 0
    ok = True
    while sent < total_qty:
        batch = min(MAX_BATCH_SIZE, total_qty - sent)
        if not _execute_trade(tapi, ticker, action, batch):
            ok = False
            break
        sent += batch
        if sent < total_qty:
            time.sleep(0.1)
    return ok and sent == total_qty


def _build_long_straddle(tapi: TradingAPI, prices: dict) -> bool:
    if _state["built_this_subheat"]:
        return True

    candidate = _best_underpriced_straddle(prices)
    if not candidate:
        single = _best_underpriced_single_leg(prices)
        if not single:
            _state["last_straddle_edge"] = 0.0
            print("  [build] no underpriced straddle or near-ATM single-leg above threshold")
            return True

        qty = min(BASE_QTY_SINGLE_LEG, OPTIONS_NET_LIMIT)
        print(f"  [build] RV={_state['realized_vol']:.1%} RTM={_state['rtm_price']:.2f}")
        print(
            f"    SINGLE   BUY {qty}x {single['ticker']}"
            f" edge={single['edge']:+.3f} |delta|={single['abs_delta']:.3f}"
        )
        ok = _execute_batched_option_order(tapi, single["ticker"], "BUY", qty)
        _state["built_this_subheat"] = True
        _state["last_entry_strike"] = single["strike"]
        return ok

    _state["last_straddle_edge"] = candidate["straddle_edge"]

    # Keep one-cycle behavior and stay safely under the options net cap.
    total_long = BASE_QTY_CALL + BASE_QTY_PUT
    if total_long > OPTIONS_NET_LIMIT:
        scale = OPTIONS_NET_LIMIT / total_long
        qty_call = max(1, int(BASE_QTY_CALL * scale))
        qty_put = max(1, int(BASE_QTY_PUT * scale))
    else:
        qty_call = BASE_QTY_CALL
        qty_put = BASE_QTY_PUT

    print(f"  [build] RV={_state['realized_vol']:.1%} RTM={_state['rtm_price']:.2f}")
    print(
        f"    STRADDLE BUY {qty_call}x {candidate['call_ticker']}"
        f" + {qty_put}x {candidate['put_ticker']}"
        f" edge={candidate['straddle_edge']:+.3f}"
    )

    ok1 = _execute_batched_option_order(tapi, candidate["call_ticker"], "BUY", qty_call)
    time.sleep(0.1)
    ok2 = _execute_batched_option_order(tapi, candidate["put_ticker"], "BUY", qty_put)

    _state["built_this_subheat"] = True
    _state["last_entry_strike"] = candidate["strike"]
    return ok1 and ok2


def _covered_short_capacity_for_candidate(candidate: dict, exposure: dict) -> int:
    long_contracts = exposure["long_contracts"]
    long_vega = exposure["long_vega"]

    if long_contracts <= 0 or long_vega <= 0:
        return 0

    short_contracts = exposure["short_contracts"]
    short_vega = exposure["short_vega"]

    cap_contracts = int(math.floor(SHORT_CONTRACT_CAP_FRAC * long_contracts))
    cap_vega = SHORT_VEGA_CAP_FRAC * long_vega

    contract_room = cap_contracts - short_contracts
    vega_room = cap_vega - short_vega
    if contract_room <= 0 or vega_room <= 0:
        return 0

    vega_pc = max(candidate["vega_pc"], 1e-9)
    qty_by_vega = int(vega_room / vega_pc)

    return max(0, min(TAIL_SHORT_TARGET_QTY, contract_room, qty_by_vega))


def _maybe_add_tail_shorts(tapi: TradingAPI, prices: dict) -> bool:
    """Add small covered shorts only while already net long vol exposure exists."""
    exposure = _aggregate_option_exposure()
    long_contracts = exposure["long_contracts"]
    if long_contracts <= 0:
        return True

    short_calls = exposure["short_calls"]
    short_puts = exposure["short_puts"]

    if short_calls <= short_puts:
        sides = ("CALL", "PUT")
    else:
        sides = ("PUT", "CALL")

    all_ok = True
    placed_any = False

    for side in sides:
        exposure = _aggregate_option_exposure()
        side_candidates = _covered_tail_short_candidates(prices, side=side)
        if not side_candidates:
            continue

        best = side_candidates[0]
        qty = _covered_short_capacity_for_candidate(best, exposure)
        if qty <= 0:
            continue

        print(
            f"  [tail short] SELL {qty}x {best['ticker']}"
            f" edge={best['edge']:+.3f} |delta|={best['abs_delta']:.3f}"
        )
        ok = _execute_batched_option_order(tapi, best["ticker"], "SELL", qty)
        all_ok = all_ok and ok
        placed_any = True
        time.sleep(0.1)

    if not placed_any:
        return True
    return all_ok


def _close_all(tapi: TradingAPI) -> bool:
    """Interleaved close of all option positions to stay within net limits."""
    to_close = sorted(
        [(t, abs(q), "SELL" if q > 0 else "BUY") for t, q in _state["positions"].items() if q != 0],
        key=lambda x: x[1],
        reverse=True,
    )

    if not to_close:
        return True

    print(f"  [close all] {len(to_close)} position(s)")

    queues = []
    for ticker, qty, action in to_close:
        q = []
        rem = qty
        while rem > 0:
            b = min(MAX_BATCH_SIZE, rem)
            q.append((ticker, b, action))
            rem -= b
        queues.append(q)

    all_ok = True
    while any(queues):
        for q in queues:
            if q:
                ticker, batch, action = q.pop(0)
                ok = _execute_trade(tapi, ticker, action, batch)
                all_ok = all_ok and ok
                if any(queues):
                    time.sleep(0.1)

    _state["positions"].clear()
    return all_ok


def _execute_hedge(tapi: TradingAPI, force: bool = False) -> bool:
    """
    Trade RTM to bring portfolio delta toward zero, capped by RTM net headroom.
    In hold phase, apply adaptive moneyness-based hedge trigger band.
    """
    delta = _portfolio_delta()
    shares = -round(delta)
    if shares == 0:
        return True

    band_frac = _hedge_band_fraction()
    _state["hedge_band_frac"] = band_frac

    if not force:
        band_shares = band_frac * _state["delta_limit"]
        if abs(shares) < band_shares:
            return True

    net_limit = RTM_SHARE_LIMIT
    try:
        for lim in tapi.get_limits():
            if lim.get("ticker") == "RTM":
                net_limit = int(lim.get("net_limit", RTM_SHARE_LIMIT))
                break
    except Exception:
        pass

    current_net = abs(_state["rtm_position"])
    room = max(0, int(net_limit * 0.95) - current_net)
    shares = max(-room, min(room, shares))

    if shares == 0:
        return True

    action = "BUY" if shares > 0 else "SELL"
    total_qty = abs(shares)
    sent = 0

    print(f"  HEDGE {action:4s} {total_qty} RTM  (Delta was {delta:+.0f})")
    while sent < total_qty:
        batch = min(RTM_MAX_ORDER, total_qty - sent)
        try:
            tapi.post_order("RTM", "MARKET", batch, action)
            _state["rtm_position"] += batch if action == "BUY" else -batch
            sent += batch
        except Exception as e:
            print(f"  [hedge error] {action} {batch} RTM: {e}")
            return False
        if sent < total_qty:
            time.sleep(0.1)

    return True


def _risk_exit(tapi: TradingAPI, reason: str) -> bool:
    print(f"  [risk-exit] {reason}")
    close_ok = _close_all(tapi)
    hedge_ok = _execute_hedge(tapi, force=True)
    _state["stress_ticks_over_delta_band"] = 0
    _state["built_this_subheat"] = True
    return close_ok and hedge_ok


# -- Main loop -------------------------------------------------------------------


def run_trading_loop() -> None:
    import sys

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    _log = open("logs.txt", "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log)

    tapi = TradingAPI(API_KEY, port=API_PORT)

    print("=" * 60)
    print("VOL LONGBIAS -- live loop started")
    print("=" * 60)

    while True:
        try:
            _poll_news(tapi)
            prices = _sync_state(tapi)

            if _state["case_status"] != "ACTIVE":
                print(f"[waiting] case status: {_state['case_status']}")
                time.sleep(1)
                continue

            if _state["subheat_id"] != _state["current_subheat_id"]:
                _state["current_subheat_id"] = _state["subheat_id"]
                _state["built_this_subheat"] = False
                _state["stress_ticks_over_delta_band"] = 0
                _state["last_entry_strike"] = None
                _state["execution_failures"] = 0
                print(f"\n[new subheat] id={_state['current_subheat_id']}")

            tick_in_heat = (_state["case_tick"] - 1) % SUBHEAT_TICKS
            nearest_sig = _straddle_candidates(prices)
            _state["last_straddle_edge"] = nearest_sig[0]["straddle_edge"] if nearest_sig else 0.0

            if tick_in_heat >= UNWIND_AT_TICK:
                _state["mode"] = "UNWIND"
            elif not _state["positions"] and not _state["built_this_subheat"]:
                _state["mode"] = "BUILD"
            elif not _state["positions"]:
                _state["mode"] = "WAIT"
            else:
                _state["mode"] = "HOLD"

            exposure = _aggregate_option_exposure()
            c_util, v_util, _, _, _, _ = _tail_short_utilization(exposure)
            _state["tail_short_contract_util"] = c_util
            _state["tail_short_vega_util"] = v_util

            print(
                f"\n[tick {_state['case_tick']} +{tick_in_heat}]"
                f" subheat={_state['current_subheat_id']}"
                f" RTM=${_state['rtm_price']:.2f}"
                f" RV={_state['realized_vol']:.1%}"
                f" Delta={_portfolio_delta():.0f}/{_state['delta_limit']}"
                f" mode={_state['mode']}"
                f" straddle_edge={_state['last_straddle_edge']:+.3f}"
                f" tail_util(c={c_util:.0%},v={v_util:.0%})"
                f" hedge_band={_state['hedge_band_frac']:.0%}"
            )

            step_ok = True

            if tick_in_heat >= UNWIND_AT_TICK:
                _state["mode"] = "UNWIND"
                if _state["positions"]:
                    print(f"  [unwind] sub-heat tick {tick_in_heat}")
                    ok1 = _close_all(tapi)
                    ok2 = _execute_hedge(tapi, force=True)
                    step_ok = ok1 and ok2
                    _state["built_this_subheat"] = True

            elif not _state["positions"]:
                if not _state["built_this_subheat"]:
                    _state["mode"] = "BUILD"
                    ok1 = _build_long_straddle(tapi, prices)
                    ok2 = _execute_hedge(tapi, force=True)
                    step_ok = ok1 and ok2
                else:
                    _state["mode"] = "WAIT"

            else:
                _state["mode"] = "HOLD"
                ok1 = _maybe_add_tail_shorts(tapi, prices)
                ok2 = _execute_hedge(tapi)
                step_ok = ok1 and ok2

                post_hedge_delta = abs(_portfolio_delta())
                stress_band = DELTA_STRESS_EXIT_FRAC * _state["delta_limit"]
                if post_hedge_delta >= stress_band:
                    _state["stress_ticks_over_delta_band"] += 1
                else:
                    _state["stress_ticks_over_delta_band"] = 0

                if _state["stress_ticks_over_delta_band"] >= DELTA_STRESS_EXIT_TICKS:
                    _state["mode"] = "RISK_EXIT"
                    step_ok = _risk_exit(
                        tapi,
                        (
                            "delta stress persisted "
                            f"({_state['stress_ticks_over_delta_band']} ticks >= {DELTA_STRESS_EXIT_FRAC:.0%} limit)"
                        ),
                    )

            if step_ok:
                _state["execution_failures"] = 0
            else:
                _state["execution_failures"] += 1

            if _state["positions"] and _state["execution_failures"] >= MAX_CONSEC_EXEC_FAILURES:
                _state["mode"] = "RISK_EXIT"
                _risk_exit(
                    tapi,
                    f"execution failures reached {_state['execution_failures']} consecutive ticks",
                )
                _state["execution_failures"] = 0

        except Exception as e:
            print(f"[loop error] {e}")

        time.sleep(1)


if __name__ == "__main__":
    run_trading_loop()
