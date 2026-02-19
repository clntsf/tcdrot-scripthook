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
OPTIONS_NET_LIMIT   = 1000   # options net limit (|long - short| contracts)
PRIMARY_QTY         = 1750   # contracts for the best-edge position (with secondary)
SECONDARY_QTY       = 750    # contracts for the best opposite-direction position
MIN_EDGE            = 0.10   # minimum edge to open a new position
DEFAULT_DELTA_LIMIT = 1000   # fallback delta limit until news arrives
HEDGE_BAND          = 25     # only re-hedge in hold phase when |delta| exceeds this
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


# ── Limits ─────────────────────────────────────────────────────────────────────


def _position_signed_delta(ticker: str, action: str, abs_delta: float) -> float:
    """Signed delta contribution of a position: +ve means long-delta, -ve means short-delta.

    BUY call → +delta,  SELL call → -delta
    BUY put  → -delta,  SELL put  → +delta
    """
    is_call, _ = _parse_ticker(ticker)
    sign = 1 if action == "BUY" else -1
    return sign * abs_delta if is_call else -sign * abs_delta


def _build_position(tapi: TradingAPI, prices: dict) -> None:
    """
    Enter PRIMARY_QTY of the best-edge option and SECONDARY_QTY of the best
    truly-opposite-delta option. Batches are interleaved to keep net position low.
    If no opposite-delta secondary exists, primary is capped to 1000 (net limit).
    """
    signals = _identify_mispricings(prices)
    if not signals:
        print("  [build] no mispricings above threshold")
        return

    p_ticker, p_action, p_edge, p_abs_delta = signals[0]
    p_sign = _position_signed_delta(p_ticker, p_action, p_abs_delta)

    def is_opposite(s) -> bool:
        # Must be opposite action (keeps net position within |long-short| limit)
        # AND opposite signed delta (partially offsets primary's delta exposure)
        return (s[1] != p_action
                and _position_signed_delta(s[0], s[1], s[3]) * p_sign < 0)

    # Find secondary: prefer opposite-action + opposite-delta (reduces net delta);
    # fall back to any opposite-action option — there's always some edge at week start.
    all_sigs_any = _identify_mispricings(prices, min_edge=0.0)
    sec = next((s for s in signals[1:] if is_opposite(s)), None)
    if sec is None:
        sec = next((s for s in all_sigs_any if s[0] != p_ticker and is_opposite(s)), None)
    if sec is None:
        # No delta-opposite found; take best opposite-action regardless of delta sign
        sec = next((s for s in all_sigs_any if s[0] != p_ticker and s[1] != p_action), None)

    s_ticker = sec[0] if sec else None
    s_action = sec[1] if sec else None
    s_edge   = sec[2] if sec else 0.0
    s_total  = SECONDARY_QTY if sec else 0

    if sec:
        p_total = PRIMARY_QTY
    else:
        # No secondary: cap by options net limit AND by how much RTM can hedge
        delta_per_contract = p_abs_delta * OPTION_MULTIPLIER
        max_by_rtm = int(RTM_SHARE_LIMIT / max(delta_per_contract, 1))
        p_total = min(OPTIONS_NET_LIMIT, max_by_rtm)

    print(f"  [build] RV={_state['realized_vol']:.1%}")
    print(f"    PRIMARY   {p_action:4s} {p_total}x {p_ticker}  edge={p_edge:+.3f}")
    if sec:
        print(f"    SECONDARY {s_action:4s} {s_total}x {s_ticker}  edge={s_edge:+.3f}")

    # Interleave primary and secondary batches to stay within net limits
    p_sent = 0
    s_sent = 0
    while p_sent < p_total or s_sent < s_total:
        if p_sent < p_total:
            batch = min(MAX_BATCH_SIZE, p_total - p_sent)
            _execute_trade(tapi, p_ticker, p_action, batch)
            p_sent += batch
            if p_sent < p_total or s_sent < s_total:
                time.sleep(0.1)
        if sec and s_sent < s_total:
            batch = min(MAX_BATCH_SIZE, s_total - s_sent)
            _execute_trade(tapi, s_ticker, s_action, batch)
            s_sent += batch
            if p_sent < p_total or s_sent < s_total:
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


# ── Mispricings ────────────────────────────────────────────────────────────────

def _identify_mispricings(prices: dict, min_edge: float = MIN_EDGE) -> list:
    """
    Return list of (ticker, action, edge, abs_delta) sorted by |edge| desc.
    action is 'BUY' (underpriced) or 'SELL' (overpriced).
    """
    S  = _state["rtm_price"]
    T  = _state["tte"]
    rv = _state["realized_vol"]
    signals = []

    for ticker, mkt in prices.items():
        is_call, K = _parse_ticker(ticker)
        iv = _implied_vol(mkt, S, K, T, RISK_FREE_RATE, is_call)
        if iv is None:
            continue
        fair  = _bs_price(S, K, T, RISK_FREE_RATE, rv, is_call)
        edge  = fair - mkt
        if abs(edge) < min_edge:
            continue
        delta = _bs_delta(S, K, T, RISK_FREE_RATE, rv, is_call)
        signals.append((ticker, "BUY" if edge > 0 else "SELL", edge, abs(delta)))

    signals.sort(key=lambda x: abs(x[2]), reverse=True)
    return signals


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
    if not force and abs(shares) < HEDGE_BAND:
        return

    # Cap to available RTM net headroom
    try:
        for lim in tapi.get_limits():
            if lim.get("ticker") == "RTM":
                room = int(lim.get("net_limit", 50000) * 0.95 - abs(lim.get("net", 0)))
                shares = max(-room, min(room, shares))
                break
    except Exception:
        pass

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

    _log = open("logs.txt", "a")
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
