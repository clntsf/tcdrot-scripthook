"""
Volatility Arbitrage Trading Module for RITC 2026

Strategy: compare implied vol (from market prices) to realized vol (from news)
and trade the difference, delta-hedging with RTM.
"""

import math
import re
import time
from typing import Optional

from api import TradingAPI
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

STRIKES            = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
CALL_TICKERS       = [f"RTM1C{s}" for s in STRIKES]
PUT_TICKERS        = [f"RTM1P{s}" for s in STRIKES]
ALL_OPTION_TICKERS = CALL_TICKERS + PUT_TICKERS

DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
API_PORT = getenv("ROT_API_PORT")

RISK_FREE_RATE     = 0.0
TRADING_YEAR_DAYS  = 240
MONTH_DAYS         = 20
OPTION_MULTIPLIER  = 100  # shares per contract (100:1)

MAX_BATCH_SIZE      = 100    # max contracts per single option order (API limit)
RTM_MAX_ORDER       = 10000  # max shares per single RTM order (API limit)
RTM_SHARE_LIMIT     = 50000  # RTM net position limit
OPTIONS_NET_LIMIT   = 1000   # options net limit — straddle delta ≈ 0, single-leg capped by RTM budget
MIN_EDGE_BUY        = 0.05   # minimum edge to buy an option (relaxed — ATM options have lower edge)
DEFAULT_DELTA_LIMIT = 1000   # fallback delta limit until news arrives
RTM_HEDGE_FRAC      = 0.60   # max fraction of RTM_SHARE_LIMIT used for initial hedge
HEDGE_BAND_FRAC     = 0.75   # re-hedge in hold phase when |delta| > this * delta_limit
SUBHEAT_TICKS       = 75     # ticks per sub-heat (options expiry cycle)
UNWIND_AT_TICK      = 59     # close at this offset within sub-heat (abs ticks 60,135,210,285)

# ── Shared state ───────────────────────────────────────────────────────────────

_state: dict = {
    "rtm_price":       50.0,
    "tte":             MONTH_DAYS / TRADING_YEAR_DAYS,   # years to expiry
    "rtm_position":    0,
    "positions":       {},   # ticker -> signed qty (+ long, - short)
    "realized_vol":    0.35,
    "delta_limit":     DEFAULT_DELTA_LIMIT,
    "penalty_pct":     0.0,
    "last_news_id":    0,
    "case_status":     "STOPPED",
    "ticks_left":      9999,
    "case_tick":       0,
}

# ── Black-Scholes (pure functions) ─────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun polynomial approximation."""
    a1, a2, a3, a4, a5, p = (
        0.254829592, -0.284496736, 1.421413741,
        -1.453152027, 1.061405429, 0.3275911,
    )
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


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


def _implied_vol(
    mkt: float, S: float, K: float, T: float, r: float, is_call: bool,
    tol: float = 0.0001, max_iter: int = 100,
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


# ── Ticker helpers ─────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> tuple:
    """Returns (is_call: bool, strike: float)."""
    is_call = "C" in ticker
    strike = float(ticker.split("C" if is_call else "P")[1])
    return is_call, strike


# ── News ───────────────────────────────────────────────────────────────────────

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
        nid      = item.get("news_id", 0)
        headline = (item.get("headline") or "").lower()
        body     = item.get("body") or ""

        _state["last_news_id"] = max(_state["last_news_id"], nid)

        # Parse delta limit and penalty announcement (tick 1)
        if "delta limit" in headline:
            m_limit   = re.search(r"\s([0-9]+)\s", body)
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

        # Parse realized vol: exactly one % → point estimate; multiple → range, skip
        pcts = re.findall(r"([\d.]+)\s*%", body)
        if len(pcts) == 1:
            vol = float(pcts[0]) / 100
            _state["realized_vol"] = vol
            print(f"[news] realized_vol → {vol:.1%}")


# ── State sync ─────────────────────────────────────────────────────────────────

def _sync_state(tapi: TradingAPI) -> dict:
    """Fetch market data, update _state, return {ticker: mid_price} for options."""
    case = tapi.get_case()
    _state["case_status"] = case.get("status", "STOPPED")
    tick        = case["tick"]
    ticks_left  = case["ticks_per_period"] - tick
    _state["ticks_left"] = ticks_left
    _state["case_tick"]  = tick
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
            _state["rtm_price"]    = mid
            _state["rtm_position"] = int(sec.get("position", 0))
        elif ticker in ALL_OPTION_TICKERS:
            prices[ticker] = mid
            qty = int(sec.get("position", 0))
            if qty != 0:
                new_positions[ticker] = qty

    _state["positions"] = new_positions
    _state["prices"]    = prices   # keep latest prices for IV-based delta calc
    return prices


# ── Portfolio delta ────────────────────────────────────────────────────────────

def _portfolio_delta() -> float:
    """Total signed delta in underlying shares (options + RTM hedge).

    Uses market-implied vol per option for accurate OTM delta; falls back to
    realized vol if IV cannot be solved (intrinsic-only, T=0, etc.).
    """
    S      = _state["rtm_price"]
    T      = _state["tte"]
    rv     = _state["realized_vol"]
    prices = _state.get("prices", {})
    opt_delta = 0.0
    for ticker, qty in _state["positions"].items():
        is_call, K = _parse_ticker(ticker)
        sigma = rv  # fallback
        mkt = prices.get(ticker)
        if mkt is not None:
            iv = _implied_vol(mkt, S, K, T, RISK_FREE_RATE, is_call)
            if iv is not None:
                sigma = iv
        d = _bs_delta(S, K, T, RISK_FREE_RATE, sigma, is_call)
        opt_delta += qty * d * OPTION_MULTIPLIER
    return opt_delta + _state["rtm_position"]


def _build_position(tapi: TradingAPI, prices: dict) -> None:
    """
    Pure long-gamma strategy: buy the ATM call AND ATM put (straddle).

    For each of calls and puts, selects the strike closest to RTM spot with
    edge ≥ MIN_EDGE_BUY. Buys OPTIONS_NET_LIMIT contracts of each leg.
    ATM straddle delta is approximately zero, so the RTM hedge handles residual.
    If only one side has sufficient edge, that single leg is capped by RTM budget.
    """
    S  = _state["rtm_price"]
    T  = _state["tte"]
    rv = _state["realized_vol"]

    calls, puts = [], []
    for ticker, mkt in prices.items():
        is_call, K = _parse_ticker(ticker)
        fair  = _bs_price(S, K, T, RISK_FREE_RATE, rv, is_call)
        edge  = fair - mkt
        if edge < MIN_EDGE_BUY:
            continue
        delta = _bs_delta(S, K, T, RISK_FREE_RATE, rv, is_call)
        entry = (ticker, edge, abs(delta), abs(K - S))
        (calls if is_call else puts).append(entry)

    # ATM-closest first
    calls.sort(key=lambda x: x[3])
    puts.sort(key=lambda x: x[3])

    legs = []
    if calls:
        legs.append(calls[0])
    if puts:
        legs.append(puts[0])

    if not legs:
        print("  [build] no ATM options with sufficient edge")
        return

    rtm_budget = RTM_SHARE_LIMIT * RTM_HEDGE_FRAC
    is_straddle = len(legs) == 2

    print(f"  [build] RV={rv:.1%}  RTM={S:.2f}  ({'straddle' if is_straddle else 'single leg'})")
    for ticker, edge, abs_delta, otm_dist in legs:
        if is_straddle:
            # Call and put deltas cancel; use full net limit
            qty = OPTIONS_NET_LIMIT
        else:
            # Single leg: cap by RTM hedge budget
            delta_per_contract = abs_delta * OPTION_MULTIPLIER
            max_by_rtm = int(rtm_budget / max(delta_per_contract, 1))
            qty = min(OPTIONS_NET_LIMIT, max_by_rtm)

        print(f"    BUY {qty}x {ticker}  edge={edge:+.3f}  |K-S|={otm_dist:.1f}")
        sent = 0
        while sent < qty:
            batch = min(MAX_BATCH_SIZE, qty - sent)
            _execute_trade(tapi, ticker, "BUY", batch)
            sent += batch
            if sent < qty:
                time.sleep(0.1)



def _close_all(tapi: TradingAPI) -> None:
    """Interleaved close of all option positions to stay within net limits."""
    to_close = sorted(
        [(t, abs(q), "SELL" if q > 0 else "BUY") for t, q in _state["positions"].items() if q != 0],
        key=lambda x: x[1], reverse=True,   # largest position first
    )
    if not to_close:
        return
    print(f"  [close all] {len(to_close)} position(s)")

    # Build a batch queue per position, then round-robin through them
    queues = []
    for ticker, qty, action in to_close:
        q = []
        rem = qty
        while rem > 0:
            b = min(MAX_BATCH_SIZE, rem)
            q.append((ticker, b, action))
            rem -= b
        queues.append(q)

    while any(queues):
        for q in queues:
            if q:
                ticker, batch, action = q.pop(0)
                _execute_trade(tapi, ticker, action, batch)
                if any(queues):
                    time.sleep(0.1)

    _state["positions"].clear()


# ── Execution ──────────────────────────────────────────────────────────────────

def _execute_trade(tapi: TradingAPI, ticker: str, action: str, qty: int) -> None:
    try:
        tapi.post_order(ticker, "MARKET", qty, action)
        sign = 1 if action == "BUY" else -1
        _state["positions"][ticker] = _state["positions"].get(ticker, 0) + sign * qty
        print(f"  {action:4s} {qty:3d}x {ticker}")
    except Exception as e:
        print(f"  [order error] {action} {qty}x {ticker}: {e}")


def _execute_hedge(tapi: TradingAPI, force: bool = False) -> None:
    """
    Trade RTM to bring portfolio delta to zero, capped by RTM net headroom.
    In the hold phase (force=False) skip if delta is within HEDGE_BAND to
    avoid paying spreads on tiny re-hedges every tick.
    """
    delta  = _portfolio_delta()
    shares = -round(delta)   # buy if delta < 0, sell if delta > 0
    if shares == 0:
        return
    if not force and abs(shares) < HEDGE_BAND_FRAC * _state["delta_limit"]:
        return

    # Cap to available RTM net headroom.
    # Use locally-tracked rtm_position as fallback so a failed API call never
    # leaves shares uncapped (which previously caused 160k+ RTM orders).
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
        return

    action    = "BUY" if shares > 0 else "SELL"
    total_qty = abs(shares)
    sent      = 0
    print(f"  HEDGE {action:4s} {total_qty} RTM  (Δ was {delta:+.0f})")
    while sent < total_qty:
        batch = min(RTM_MAX_ORDER, total_qty - sent)
        try:
            tapi.post_order("RTM", "MARKET", batch, action)
            _state["rtm_position"] += batch if action == "BUY" else -batch
            sent += batch
        except Exception as e:
            print(f"  [hedge error] {action} {batch} RTM: {e}")
            break
        if sent < total_qty:
            time.sleep(0.1)


# ── Main loop ──────────────────────────────────────────────────────────────────

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
    print("VOL ARBIT — live loop started")
    print("=" * 60)

    while True:
        try:
            _poll_news(tapi)
            prices = _sync_state(tapi)

            if _state["case_status"] != "ACTIVE":
                print(f"[waiting] case status: {_state['case_status']}")
                time.sleep(1)
                continue

            tick_in_heat = (_state["case_tick"] - 1) % SUBHEAT_TICKS
            print(
                f"\n[tick {_state['case_tick']} +{tick_in_heat}]"
                f"  RTM=${_state['rtm_price']:.2f}"
                f"  RV={_state['realized_vol']:.1%}"
                f"  Δ={_portfolio_delta():.0f}/{_state['delta_limit']}"
                f"  pos={len(_state['positions'])}"
            )

            # Safety: if no options but RTM position lingers (e.g. hedge failed during
            # unwind), close it every tick until flat — never build on an orphaned hedge
            if not _state["positions"] and _state["rtm_position"] != 0:
                print(f"  [orphan RTM] no options, RTM={_state['rtm_position']:+d} — force closing")
                _execute_hedge(tapi, force=True)
                time.sleep(1)
                continue

            if tick_in_heat >= UNWIND_AT_TICK:
                # ── Unwind phase (ticks 60-74): close and stay flat ──────────
                if _state["positions"]:
                    print(f"  [unwind] sub-heat tick {tick_in_heat}")
                    _close_all(tapi)
                    _execute_hedge(tapi, force=True)
                    time.sleep(1)
                    continue

            elif not _state["positions"]:
                # ── Build phase (tick 0 of each sub-heat, positions empty) ───
                _build_position(tapi, prices)
                _execute_hedge(tapi, force=True)
                time.sleep(1)
                continue

            else:
                # ── Hold phase: banded re-hedge only ─────────────────────────
                _execute_hedge(tapi)

        except Exception as e:
            print(f"[loop error] {e}")

        time.sleep(1)


if __name__ == "__main__":
    run_trading_loop()
