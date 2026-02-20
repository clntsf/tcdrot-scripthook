"""
RITC 2026 — Rebate Farmer (risk-averse)

Philosophy:
  - In a competitive market, spread revenue compresses to ~zero.
    Rebates are fixed by the case and don't compress.
  - Quote aggressively for queue position (penny-improve), collect
    rebates on every fill, stay nearly flat.
  - Never let per-ticker position exceed TICKER_CAP; above SKEW_THRESHOLD
    only quote the side that reduces inventory.
  - Weighted by rebate: WNTR ($0.025) gets the most size, SPNG ($0.01) the least.
"""

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# ── Environment ──
load_dotenv(Path(__file__).parent / ".env")
API_KEY = getenv("ROT_API_KEY")
BASE = f"http://localhost:{getenv('ROT_API_PORT')}/v1"

s = requests.Session()
s.headers.update({"X-API-key": API_KEY})

# ── Case constants ──
TICKERS   = ["SPNG", "SMMR", "ATMN", "WNTR"]
REBATES   = {"SPNG": 0.01, "SMMR": 0.02, "ATMN": 0.015, "WNTR": 0.025}
MARKET_FEE = 0.02

# ── Risk parameters ──
# Hard position cap per ticker. Above this we only quote the reducing side.
TICKER_CAP   = 2000
# Above this we start shrinking the adding-side quote.
SKEW_THRESHOLD = 800

# ── Sizing — weighted by rebate value ──
# Higher rebate = more aggressive quoting = larger position risk worth taking.
BASE_SIZES = {"SPNG": 300, "SMMR": 600, "ATMN": 500, "WNTR": 900}
MAX_ORDER_SIZE = 5000

# ── Timing ──
STARTUP_THRESHOLD  = 7    # freeze: sit out news spike
NEWS_WIDEN_TICKS   = 5    # still widen a little inside news window
NEWS_SIZE_MULT     = 0.3  # much smaller during news (but still collect rebates)
UNWIND_LIMIT_START = 52
UNWIND_AGGRESSIVE  = 55
UNWIND_MARKET      = 58

OVERNIGHT_GROSS_LIMIT = 9000

# ── URLs ──
URL_CANCEL     = f"{BASE}/commands/cancel"
URL_CASE       = f"{BASE}/case"
URL_SECURITIES = f"{BASE}/securities"
URL_ORDERS     = f"{BASE}/orders"

# ── Per-thread sessions ──
_thread_sessions: dict = {}

def _sess():
    import threading
    tid = threading.current_thread().ident
    if tid not in _thread_sessions:
        sess = requests.Session()
        sess.headers.update({"X-API-key": API_KEY})
        _thread_sessions[tid] = sess
    return _thread_sessions[tid]

def _cancel():
    _sess().post(URL_CANCEL, params={"all": 1})

def _fetch_case():
    r = _sess().get(URL_CASE)
    if r.ok:
        c = r.json()
        return c["tick"], c["status"]
    return 0, "STOPPED"

def _fetch_securities():
    r = _sess().get(URL_SECURITIES)
    if not r.ok:
        return None
    return {
        sec["ticker"]: (sec.get("bid"), sec.get("ask"), sec.get("position", 0))
        for sec in r.json()
    }

def _post_order(params):
    _sess().post(URL_ORDERS, params=params)


def _quote_size(ticker, pos, t_min, news_active):
    """
    Return (buy_qty, sell_qty) for this ticker.
    - Above TICKER_CAP: only quote the reducing side (0 on the adding side)
    - Above SKEW_THRESHOLD: taper the adding side down toward zero
    - During news: multiply everything by NEWS_SIZE_MULT
    """
    base = BASE_SIZES[ticker]
    if news_active:
        base = max(100, int(base * NEWS_SIZE_MULT))

    if pos >= TICKER_CAP:
        return 0, base          # only sell
    if pos <= -TICKER_CAP:
        return base, 0          # only buy

    # Taper the inventory-adding side between threshold and cap
    abs_pos = abs(pos)
    if abs_pos > SKEW_THRESHOLD:
        taper = 1.0 - (abs_pos - SKEW_THRESHOLD) / (TICKER_CAP - SKEW_THRESHOLD)
        taper = max(0.05, taper)
        if pos > 0:
            buy_qty  = max(100, int(base * taper))
            sell_qty = base
        else:
            buy_qty  = base
            sell_qty = max(100, int(base * taper))
    else:
        buy_qty  = base
        sell_qty = base

    return buy_qty, sell_qty


def _quote_prices(bid, ask, ticker, pos):
    """
    Penny-improve the market to be first in queue.
    Minimum half-spread: 0.005 (never cross the market).
    Apply a small inventory skew to nudge the mid.
    """
    mid = (bid + ask) * 0.5
    half = (ask - bid) * 0.5

    # Inventory skew: nudge mid against position
    SKEW = 0.0002
    skewed_mid = mid - SKEW * pos

    # Penny-improve: quote one tick inside the market, but respect min half-spread
    our_half = max(0.005, half - 0.01)
    our_bid = round(skewed_mid - our_half, 2)
    our_ask = round(skewed_mid + our_half, 2)

    # Hard safety: never cross the market
    our_bid = min(our_bid, round(ask - 0.01, 2))
    our_ask = max(our_ask, round(bid + 0.01, 2))

    return our_bid, our_ask


def main():
    pool = ThreadPoolExecutor(max_workers=16)

    # Wait for case
    tick, status = _fetch_case()
    while status != "ACTIVE":
        time.sleep(1)
        tick, status = _fetch_case()

    # Limits (fixed per trial)
    r = s.get(f"{BASE}/limits")
    max_gross = 15000
    if r.ok:
        lims = r.json()
        if lims:
            max_gross = lims[0].get("gross_limit", 15000)

    # Log
    log_dir = Path(__file__).parent.parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f"rebate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_fh = open(log_path, "w")
    log_fh.write("tick|t_min|agg|" + "|".join(f"{t}:pos,bid,ask,our_bid,our_ask,bqty,sqty" for t in TICKERS) + "|actions\n")

    cycle_times = []

    try:
        while True:
            t0 = time.perf_counter()

            # ── Batch 1: cancel + state in parallel ──
            f_cancel = pool.submit(_cancel)
            f_case   = pool.submit(_fetch_case)
            f_secs   = pool.submit(_fetch_securities)

            tick, status = f_case.result()
            if status != "ACTIVE":
                break
            secs = f_secs.result()
            f_cancel.result()

            if not secs:
                cycle_times.append(time.perf_counter() - t0)
                continue

            t_min = tick % 60
            news_active = t_min < NEWS_WIDEN_TICKS

            agg = sum(abs(secs.get(t, (0, 0, 0))[2]) for t in TICKERS)

            log_parts = [str(tick), str(t_min), str(int(agg))]
            actions = []
            order_futures = []

            # ── Freeze ──
            if t_min < STARTUP_THRESHOLD:
                for t in TICKERS:
                    d = secs.get(t, (0, 0, 0))
                    log_parts.append(f"{d[2]},{d[1]},{d[0]},0,0,0,0")
                log_fh.write("|".join(log_parts) + "|freeze\n")
                cycle_times.append(time.perf_counter() - t0)
                continue

            # ── Unwind zone ──
            if t_min >= UNWIND_LIMIT_START:
                for ticker in TICKERS:
                    d = secs.get(ticker)
                    if not d:
                        continue
                    bid, ask, pos = d
                    log_parts.append(f"{pos},{bid},{ask},0,0,0,0")
                    if pos == 0 or bid is None or ask is None:
                        continue

                    inverted = bid >= ask
                    if agg <= OVERNIGHT_GROSS_LIMIT and inverted:
                        actions.append(f"hold:{ticker}")
                        continue

                    action  = "SELL" if pos > 0 else "BUY"
                    abs_pos = int(abs(pos))

                    if t_min >= UNWIND_MARKET:
                        remaining = abs_pos
                        while remaining > 0:
                            chunk = min(remaining, MAX_ORDER_SIZE)
                            order_futures.append(pool.submit(_post_order, {
                                "ticker": ticker, "type": "MARKET",
                                "quantity": chunk, "action": action,
                            }))
                            remaining -= chunk
                        actions.append(f"unwind:{ticker}:{action}:{abs_pos}:MKT")
                    elif t_min >= UNWIND_AGGRESSIVE:
                        price = round((bid + 0.01) if action == "SELL" else (ask - 0.01), 2)
                        remaining = abs_pos
                        while remaining > 0:
                            chunk = min(remaining, MAX_ORDER_SIZE)
                            order_futures.append(pool.submit(_post_order, {
                                "ticker": ticker, "type": "LIMIT",
                                "quantity": chunk, "action": action, "price": price,
                            }))
                            remaining -= chunk
                        actions.append(f"unwind:{ticker}:{action}:{abs_pos}:LMT@{price}")
                    else:
                        qty   = min(max(100, abs_pos // 3), MAX_ORDER_SIZE)
                        price = round(bid if action == "SELL" else ask, 2)
                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": qty, "action": action, "price": price,
                        }))
                        actions.append(f"unwind:{ticker}:{action}:{qty}:LMT@{price}")

            # ── Quoting zone ──
            else:
                for ticker in TICKERS:
                    d = secs.get(ticker)
                    if not d:
                        log_parts.append("0,0,0,0,0,0,0")
                        continue
                    bid, ask, pos = d

                    if bid is None or ask is None or bid >= ask:
                        log_parts.append(f"{pos},{bid},{ask},0,0,0,0")
                        actions.append(f"skip:{ticker}:inverted")
                        continue

                    our_bid, our_ask = _quote_prices(bid, ask, ticker, pos)
                    buy_qty, sell_qty = _quote_size(ticker, pos, t_min, news_active)

                    # Gross limit guard
                    if agg + max(buy_qty, sell_qty) >= max_gross:
                        log_parts.append(f"{pos},{bid},{ask},{our_bid},{our_ask},0,0")
                        actions.append(f"skip:{ticker}:gross_limit")
                        continue

                    if buy_qty > 0:
                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": buy_qty, "action": "BUY", "price": our_bid,
                        }))
                    if sell_qty > 0:
                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": sell_qty, "action": "SELL", "price": our_ask,
                        }))

                    log_parts.append(f"{pos},{bid},{ask},{our_bid},{our_ask},{buy_qty},{sell_qty}")
                    actions.append(f"quote:{ticker}:{buy_qty}/{sell_qty}:{our_bid}/{our_ask}")

            # Wait for orders then log
            for f in order_futures:
                f.result()

            log_fh.write("|".join(log_parts) + "|" + ";".join(actions) + "\n")
            cycle_times.append(time.perf_counter() - t0)

    except KeyboardInterrupt:
        pass
    finally:
        log_fh.close()
        pool.shutdown(wait=False)

    if cycle_times:
        n   = len(cycle_times)
        avg = sum(cycle_times) / n
        mn  = min(cycle_times)
        mx  = max(cycle_times)
        cycle_times.sort()
        p50 = cycle_times[n // 2]
        p95 = cycle_times[int(n * 0.95)]
        print(f"\n  Cycles: {n}")
        print(f"  Avg: {avg*1000:.1f}ms  Min: {mn*1000:.1f}ms  Max: {mx*1000:.1f}ms")
        print(f"  P50: {p50*1000:.1f}ms  P95: {p95*1000:.1f}ms")
        print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
