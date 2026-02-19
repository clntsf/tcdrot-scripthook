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

RISK_FREE_RATE     = 0.0
TRADING_YEAR_DAYS  = 240
MONTH_DAYS         = 20
OPTION_MULTIPLIER  = 1   # shares per contract

MAX_BATCH_SIZE      = 100   # max contracts per single order (API limit)
PRIMARY_QTY         = 1750  # contracts for the best-edge position
SECONDARY_QTY       = 750   # contracts for the best opposite-direction position
MIN_EDGE            = 0.10  # minimum edge to open a new position
CLOSE_THRESHOLD     = 0.05  # edge threshold to close — pragmatic, not hair-trigger
DEFAULT_DELTA_LIMIT = 1000  # fallback delta limit until news arrives
TIME_UNWIND_TICKS   = 5     # force-close this many ticks before end of sub-heat
HEDGE_BAND          = 200   # only re-hedge in hold phase when |delta| exceeds this

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
    "rv_at_build":     None,  # realized_vol when we last built the position
    "case_status":     "STOPPED",
    "ticks_left":      9999,
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
    ticks_left  = case["ticks_per_period"] - case["tick"]
    ticks_total = case["ticks_per_period"]
    _state["ticks_left"] = ticks_left
    _state["tte"] = max(0.0, ticks_left / ticks_total) * (MONTH_DAYS / TRADING_YEAR_DAYS)

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
    return prices


# ── Portfolio delta ────────────────────────────────────────────────────────────

def _portfolio_delta() -> float:
    """Total signed delta in underlying shares (options + RTM hedge)."""
    S  = _state["rtm_price"]
    T  = _state["tte"]
    rv = _state["realized_vol"]
    opt_delta = 0.0
    for ticker, qty in _state["positions"].items():
        is_call, K = _parse_ticker(ticker)
        d = _bs_delta(S, K, T, RISK_FREE_RATE, rv, is_call)
        opt_delta += qty * d * OPTION_MULTIPLIER
    return opt_delta + _state["rtm_position"]


# ── Limits ─────────────────────────────────────────────────────────────────────


def _build_position(tapi: TradingAPI, prices: dict) -> None:
    """
    Enter PRIMARY_QTY of the best-edge option and SECONDARY_QTY of the best
    option in the opposite direction. Orders sent in MAX_BATCH_SIZE chunks.
    """
    signals = _identify_mispricings(prices)
    if not signals:
        print("  [build] no mispricings above threshold")
        _state["rv_at_build"] = _state["realized_vol"]
        return

    p_ticker, p_action, p_edge, _ = signals[0]
    opposite = "BUY" if p_action == "SELL" else "SELL"
    sec = next((s for s in signals[1:] if s[1] == opposite), None)

    print(f"  [build] RV={_state['realized_vol']:.1%}")
    print(f"    PRIMARY   {p_action:4s} {PRIMARY_QTY}x {p_ticker}  edge={p_edge:+.3f}")
    _send_batches(tapi, p_ticker, p_action, PRIMARY_QTY)

    if sec:
        s_ticker, s_action, s_edge, _ = sec
        print(f"    SECONDARY {s_action:4s} {SECONDARY_QTY}x {s_ticker}  edge={s_edge:+.3f}")
        _send_batches(tapi, s_ticker, s_action, SECONDARY_QTY)

    _state["rv_at_build"] = _state["realized_vol"]


def _close_all(tapi: TradingAPI) -> None:
    """Force-close every open option position (used on vol regime change)."""
    for ticker, qty in list(_state["positions"].items()):
        if qty == 0:
            continue
        action = "SELL" if qty > 0 else "BUY"
        print(f"  FORCE CLOSE {ticker} ({qty:+d})")
        _send_batches(tapi, ticker, action, abs(qty))
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

def _send_batches(tapi: TradingAPI, ticker: str, action: str, total: int) -> None:
    """Send `total` contracts in MAX_BATCH_SIZE chunks."""
    sent = 0
    while sent < total:
        batch = min(MAX_BATCH_SIZE, total - sent)
        _execute_trade(tapi, ticker, action, batch)
        sent += batch


def _execute_trade(tapi: TradingAPI, ticker: str, action: str, qty: int) -> None:
    try:
        tapi.post_order(ticker, "MARKET", qty, action)
        sign = 1 if action == "BUY" else -1
        _state["positions"][ticker] = _state["positions"].get(ticker, 0) + sign * qty
        print(f"  {action:4s} {qty:3d}x {ticker}")
    except Exception as e:
        print(f"  [order error] {action} {qty}x {ticker}: {e}")


def _close_positions(tapi: TradingAPI, prices: dict) -> None:
    S  = _state["rtm_price"]
    T  = _state["tte"]
    rv = _state["realized_vol"]

    for ticker, qty in list(_state["positions"].items()):
        mkt = prices.get(ticker)
        if mkt is None or qty == 0:
            continue
        is_call, K = _parse_ticker(ticker)
        fair = _bs_price(S, K, T, RISK_FREE_RATE, rv, is_call)
        edge = fair - mkt

        if qty > 0 and edge <= CLOSE_THRESHOLD:
            try:
                tapi.post_order(ticker, "MARKET", abs(qty), "SELL")
                del _state["positions"][ticker]
                print(f"  CLOSE LONG  {ticker}  edge={edge:+.2f}")
            except Exception as e:
                print(f"  [close error] {ticker}: {e}")
        elif qty < 0 and edge >= -CLOSE_THRESHOLD:
            try:
                tapi.post_order(ticker, "MARKET", abs(qty), "BUY")
                del _state["positions"][ticker]
                print(f"  CLOSE SHORT {ticker}  edge={edge:+.2f}")
            except Exception as e:
                print(f"  [close error] {ticker}: {e}")


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

    action = "BUY" if shares > 0 else "SELL"
    qty    = abs(shares)
    try:
        tapi.post_order("RTM", "MARKET", qty, action)
        _state["rtm_position"] += shares
        print(f"  HEDGE {action:4s} {qty} RTM  (Δ was {delta:+.0f})")
    except Exception as e:
        print(f"  [hedge error] {e}")


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_trading_loop() -> None:
    tapi = TradingAPI(API_KEY)

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

            print(
                f"\n[tick] RTM=${_state['rtm_price']:.2f}"
                f"  RV={_state['realized_vol']:.1%}"
                f"  tl={_state['ticks_left']}"
                f"  Δ={_portfolio_delta():.0f}/{_state['delta_limit']}"
                f"  pos={len(_state['positions'])}"
            )

            rv    = _state["realized_vol"]
            tl    = _state["ticks_left"]

            # Time unwind: force-close within TIME_UNWIND_TICKS of sub-heat end
            if _state["positions"] and tl <= TIME_UNWIND_TICKS:
                print(f"  [time unwind] {tl} ticks left")
                _close_all(tapi)
                _execute_hedge(tapi, force=True)
                _state["rv_at_build"] = None  # arm for next vol announcement
                time.sleep(1)
                continue

            # New vol reading → close old position and build fresh maximum position
            if rv != _state["rv_at_build"]:
                if _state["positions"]:
                    print(f"  [vol update] {_state['rv_at_build']:.1%} → {rv:.1%}, rebuilding")
                    _close_all(tapi)
                    _execute_hedge(tapi, force=True)
                _build_position(tapi, prices)
                _execute_hedge(tapi, force=True)
                time.sleep(1)
                continue

            # Holding: only re-hedge if delta has drifted beyond HEDGE_BAND
            _close_positions(tapi, prices)
            _execute_hedge(tapi)  # banded — skips small drifts

        except Exception as e:
            print(f"[loop error] {e}")

        time.sleep(1)


if __name__ == "__main__":
    run_trading_loop()
