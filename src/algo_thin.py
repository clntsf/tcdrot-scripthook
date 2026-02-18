"""RITC 2026 Algorithmic Market Making — THIN (parallel HTTP, max speed)"""

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

# ── Constants ──
TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]
MARKET_FEE = 0.02
REBATES = {"SPNG": 0.01, "SMMR": 0.02, "ATMN": 0.015, "WNTR": 0.025}
MIN_HALF_SPREAD = {t: max(0.0, MARKET_FEE - REBATES[t]) + 0.02 for t in TICKERS}
REBATE_MULT = {t: 1.0 + (REBATES[t] - 0.01) / 0.01 for t in TICKERS}

SKEW_FACTOR = 0.0003
BASE_ORDER_SIZE = 3500
MAX_ORDER_SIZE = 10000
EWMA_ALPHA = 0.3
EWMA_COMP = 1 - EWMA_ALPHA
STARTUP_THRESHOLD = 7
UNWIND_LIMIT_START = 53
UNWIND_AGGRESSIVE = 55
UNWIND_MARKET = 58
NEWS_WIDEN_TICKS = 5
NEWS_SPREAD_MULT = 2.0
OVERNIGHT_GROSS_LIMIT = 9000
N_TICKERS = 4

ewma_mids = [None, None, None, None]  # indexed by ticker order

# Pre-build URLs to avoid f-string in hot loop
URL_CANCEL = f"{BASE}/commands/cancel"
URL_CASE = f"{BASE}/case"
URL_SECURITIES = f"{BASE}/securities"
URL_ORDERS = f"{BASE}/orders"

# Thread pool for parallel HTTP — one session per thread
thread_sessions = {}

def get_session():
    """Get or create a requests.Session for the current thread."""
    import threading
    tid = threading.current_thread().ident
    if tid not in thread_sessions:
        sess = requests.Session()
        sess.headers.update({"X-API-key": API_KEY})
        thread_sessions[tid] = sess
    return thread_sessions[tid]


def _cancel():
    get_session().post(URL_CANCEL, params={"all": 1})

def _fetch_case():
    r = get_session().get(URL_CASE)
    if r.ok:
        c = r.json()
        return c["tick"], c["status"]
    return 0, "STOPPED"

def _fetch_securities():
    r = get_session().get(URL_SECURITIES)
    if not r.ok:
        return None
    return {
        sec["ticker"]: (sec.get("bid"), sec.get("ask"), sec.get("position", 0))
        for sec in r.json()
    }

def _post_order(params):
    get_session().post(URL_ORDERS, params=params)


def main():
    pool = ThreadPoolExecutor(max_workers=12)

    # Wait for case
    tick, status = _fetch_case()
    while status != "ACTIVE":
        time.sleep(1)
        tick, status = _fetch_case()

    # Fetch limits once
    r = s.get(f"{BASE}/limits")
    max_exp = 15000
    if r.ok:
        lims = r.json()
        if lims:
            max_exp = lims[0].get("gross_limit", 15000)
    inv_max_exp = 1.0 / max_exp if max_exp else 1.0
    per_ticker_max = max_exp / N_TICKERS

    # Open log file
    log_dir = Path(__file__).parent.parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_fh = open(log_path, "w")
    log_fh.write("tick|t_min|agg_exp|" + "|".join(f"{t}:pos,bid,ask" for t in TICKERS) + "|actions\n")

    cycle_times = []

    try:
        while True:
            t0 = time.perf_counter()

            # ── Batch 1: cancel + case + securities in parallel ──
            f_cancel = pool.submit(_cancel)
            f_case = pool.submit(_fetch_case)
            f_secs = pool.submit(_fetch_securities)

            tick, status = f_case.result()
            if status != "ACTIVE":
                break
            secs = f_secs.result()
            f_cancel.result()  # ensure cancel completed

            if not secs:
                cycle_times.append(time.perf_counter() - t0)
                continue

            t_min = tick % 60

            # Aggregate exposure
            agg = 0
            for i in range(N_TICKERS):
                d = secs.get(TICKERS[i])
                if d:
                    agg += abs(d[2])

            # Freeze zone — nothing to do
            if t_min < STARTUP_THRESHOLD:
                log_fh.write(f"{tick}|{t_min}|{agg}|" + "|".join(
                    f"{secs.get(t, (0,0,0))[2]},{secs.get(t, (0,0,0))[0]},{secs.get(t, (0,0,0))[1]}"
                    for t in TICKERS) + "|freeze\n")
                cycle_times.append(time.perf_counter() - t0)
                continue

            # ── Batch 2: compute + fire all orders in parallel ──
            order_futures = []
            actions = []

            if t_min >= UNWIND_LIMIT_START:
                # Unwind zone
                for i in range(N_TICKERS):
                    d = secs.get(TICKERS[i])
                    if not d:
                        continue
                    bid, ask, pos = d
                    if pos == 0 or bid is None or ask is None:
                        continue
                    if bid >= ask and agg <= OVERNIGHT_GROSS_LIMIT:
                        continue

                    action = "SELL" if pos > 0 else "BUY"
                    abs_pos = int(abs(pos))

                    ticker = TICKERS[i]
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
                        qty = min(max(100, abs_pos // 3), MAX_ORDER_SIZE)
                        price = round(bid if action == "SELL" else ask, 2)
                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": qty, "action": action, "price": price,
                        }))
                        actions.append(f"unwind:{ticker}:{action}:{qty}:LMT@{price}")
            else:
                # Quoting zone
                for i in range(N_TICKERS):
                    d = secs.get(TICKERS[i])
                    if not d:
                        continue
                    bid, ask, pos = d
                    if bid is None or ask is None or bid >= ask:
                        continue

                    mid = (bid + ask) * 0.5
                    market_half = (ask - bid) * 0.5
                    ticker = TICKERS[i]

                    # EWMA inline
                    prev = ewma_mids[i]
                    if prev is None:
                        ewma_mids[i] = mid
                        vol_dev = 0.0
                    else:
                        ewma_mids[i] = EWMA_ALPHA * mid + EWMA_COMP * prev
                        vol_dev = abs(mid - ewma_mids[i])

                    # Skewed quotes
                    skewed_mid = mid - SKEW_FACTOR * pos
                    half_spread = max(MIN_HALF_SPREAD[ticker], market_half - 0.01)
                    if t_min < NEWS_WIDEN_TICKS:
                        half_spread *= NEWS_SPREAD_MULT

                    our_bid = round(skewed_mid - half_spread, 2)
                    our_ask = round(skewed_mid + half_spread, 2)
                    our_bid = min(our_bid, round(ask - 0.01, 2))
                    our_ask = max(our_ask, round(bid + 0.01, 2))

                    if vol_dev > 0.10:
                        adj = round(vol_dev * 0.5, 2)
                        our_bid = round(our_bid - adj, 2)
                        our_ask = round(our_ask + adj, 2)

                    # Order size
                    headroom = max(0, max_exp - agg)
                    size = min(BASE_ORDER_SIZE, headroom * 0.25)  # /N_TICKERS
                    if per_ticker_max > 0:
                        size *= max(0.1, 1.0 - abs(pos) / per_ticker_max)
                    size *= REBATE_MULT[ticker]
                    qty = max(100, min(int(size), MAX_ORDER_SIZE))

                    if agg + qty * 2 < max_exp:
                        # Asymmetric sizing: bigger on the side that flattens inventory
                        if pos > 0:
                            # Long → want to sell more, buy less
                            buy_qty = max(100, int(qty * 0.5))
                            sell_qty = qty
                        elif pos < 0:
                            # Short → want to buy more, sell less
                            buy_qty = qty
                            sell_qty = max(100, int(qty * 0.5))
                        else:
                            buy_qty = qty
                            sell_qty = qty

                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": buy_qty, "action": "BUY", "price": our_bid,
                        }))
                        order_futures.append(pool.submit(_post_order, {
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": sell_qty, "action": "SELL", "price": our_ask,
                        }))
                        actions.append(f"quote:{ticker}:{buy_qty}/{sell_qty}:{our_bid}/{our_ask}")
                    else:
                        actions.append(f"skip:{ticker}")

            # Wait for all orders to land before next cycle
            for f in order_futures:
                f.result()

            cycle_times.append(time.perf_counter() - t0)

    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False)

    # Report
    if cycle_times:
        n = len(cycle_times)
        avg = sum(cycle_times) / n
        mn = min(cycle_times)
        mx = max(cycle_times)
        cycle_times.sort()
        p50 = cycle_times[n // 2]
        p95 = cycle_times[int(n * 0.95)]
        print(f"\n  Cycles: {n}")
        print(f"  Avg: {avg*1000:.1f}ms  Min: {mn*1000:.1f}ms  Max: {mx*1000:.1f}ms")
        print(f"  P50: {p50*1000:.1f}ms  P95: {p95*1000:.1f}ms")


if __name__ == "__main__":
    main()
