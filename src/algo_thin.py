"""RITC 2026 Algorithmic Market Making — THIN (no logging, no prints, max speed)"""

import time
import requests
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
MIN_HALF_SPREAD = {t: max(0.0, MARKET_FEE - REBATES[t]) + 0.01 for t in TICKERS}

SKEW_FACTOR = 0.0001
BASE_ORDER_SIZE = 1800
MAX_ORDER_SIZE = 10000
EWMA_ALPHA = 0.3
STARTUP_THRESHOLD = 7
UNWIND_LIMIT_START = 53
UNWIND_AGGRESSIVE = 55
UNWIND_MARKET = 58
NEWS_WIDEN_TICKS = 5
NEWS_SPREAD_MULT = 2.0
OVERNIGHT_GROSS_LIMIT = 9000
N_TICKERS = len(TICKERS)

ewma_mids = {t: None for t in TICKERS}


# ── API (inlined for speed) ──

def fetch_tick():
    r = s.get(f"{BASE}/case")
    if r.ok:
        c = r.json()
        return c["tick"], c["status"]
    return 0, "STOPPED"


def fetch_securities():
    r = s.get(f"{BASE}/securities")
    if not r.ok:
        return {}
    return {
        sec["ticker"]: (sec.get("bid"), sec.get("ask"), sec.get("position", 0))
        for sec in r.json()
    }


def fetch_gross_limit():
    r = s.get(f"{BASE}/limits")
    if r.ok:
        lims = r.json()
        if lims:
            return lims[0].get("gross_limit", 15000)
    return 15000


# ── Core logic (no allocations, no string formatting) ──

def main():
    # Wait for case
    tick, status = fetch_tick()
    while status != "ACTIVE":
        time.sleep(1)
        tick, status = fetch_tick()

    max_exp = fetch_gross_limit()
    cycle_times = []

    try:
        while True:
            t0 = time.perf_counter()

            # Cancel all
            s.post(f"{BASE}/commands/cancel", params={"all": 1})

            # Fetch state
            tick, status = fetch_tick()
            if status != "ACTIVE":
                break
            secs = fetch_securities()

            t_min = tick % 60

            # Aggregate exposure
            agg = 0
            for t in TICKERS:
                d = secs.get(t)
                if d:
                    agg += abs(d[2])

            # Freeze zone
            if t_min < STARTUP_THRESHOLD:
                cycle_times.append(time.perf_counter() - t0)
                continue

            # Unwind zone
            if t_min >= UNWIND_LIMIT_START:
                for ticker in TICKERS:
                    d = secs.get(ticker)
                    if not d:
                        continue
                    bid, ask, pos = d
                    if pos == 0 or bid is None or ask is None:
                        continue

                    inverted = bid >= ask
                    if agg <= OVERNIGHT_GROSS_LIMIT and inverted:
                        continue

                    action = "SELL" if pos > 0 else "BUY"
                    abs_pos = int(abs(pos))

                    if t_min >= UNWIND_MARKET:
                        remaining = abs_pos
                        while remaining > 0:
                            chunk = min(remaining, MAX_ORDER_SIZE)
                            s.post(f"{BASE}/orders", params={
                                "ticker": ticker, "type": "MARKET",
                                "quantity": chunk, "action": action,
                            })
                            remaining -= chunk
                    elif t_min >= UNWIND_AGGRESSIVE:
                        price = round((bid + 0.01) if action == "SELL" else (ask - 0.01), 2)
                        remaining = abs_pos
                        while remaining > 0:
                            chunk = min(remaining, MAX_ORDER_SIZE)
                            s.post(f"{BASE}/orders", params={
                                "ticker": ticker, "type": "LIMIT",
                                "quantity": chunk, "action": action,
                                "price": price,
                            })
                            remaining -= chunk
                    else:
                        qty = max(100, abs_pos // 3)
                        price = round(bid if action == "SELL" else ask, 2)
                        s.post(f"{BASE}/orders", params={
                            "ticker": ticker, "type": "LIMIT",
                            "quantity": min(qty, MAX_ORDER_SIZE), "action": action,
                            "price": price,
                        })

                cycle_times.append(time.perf_counter() - t0)
                continue

            # Quoting zone
            for ticker in TICKERS:
                d = secs.get(ticker)
                if not d:
                    continue
                bid, ask, pos = d
                if bid is None or ask is None or bid >= ask:
                    continue

                mid = (bid + ask) * 0.5
                market_half = (ask - bid) * 0.5

                # EWMA
                prev = ewma_mids[ticker]
                if prev is None:
                    ewma_mids[ticker] = mid
                    vol_dev = 0.0
                else:
                    ewma_mids[ticker] = EWMA_ALPHA * mid + (1 - EWMA_ALPHA) * prev
                    vol_dev = abs(mid - ewma_mids[ticker])

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
                size = min(BASE_ORDER_SIZE, headroom / N_TICKERS)
                per_ticker_max = max_exp / N_TICKERS
                if per_ticker_max > 0:
                    size *= max(0.1, 1.0 - abs(pos) / per_ticker_max)
                size *= 1.0 + (REBATES[ticker] - 0.01) / 0.01
                qty = max(100, min(int(size), MAX_ORDER_SIZE))

                if agg + qty * 2 < max_exp:
                    s.post(f"{BASE}/orders", params={
                        "ticker": ticker, "type": "LIMIT",
                        "quantity": qty, "action": "BUY", "price": our_bid,
                    })
                    s.post(f"{BASE}/orders", params={
                        "ticker": ticker, "type": "LIMIT",
                        "quantity": qty, "action": "SELL", "price": our_ask,
                    })

            cycle_times.append(time.perf_counter() - t0)

    except KeyboardInterrupt:
        pass

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
