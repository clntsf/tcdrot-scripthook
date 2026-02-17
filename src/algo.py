"""RITC 2026 Algorithmic Market Making Case — v2 (Inventory-Skewed Quoting)"""

import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# ── Environment ──
DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
API_PORT = getenv("ROT_API_PORT")
BASE = f"http://localhost:{API_PORT}/v1"

s = requests.Session()
s.headers.update({"X-API-key": API_KEY})

# ── ANSI colours ──
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Case constants ──
TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]
MARKET_FEE = 0.02

REBATES = {"SPNG": 0.01, "SMMR": 0.02, "ATMN": 0.015, "WNTR": 0.025}
MIN_HALF_SPREAD = {t: max(0.0, MARKET_FEE - REBATES[t]) + 0.01 for t in TICKERS}
# SPNG: 0.02, SMMR: 0.01, ATMN: 0.015, WNTR: 0.01

# ── Tuning parameters ──
SKEW_FACTOR = 0.0002       # mid shift per unit of inventory (doubled for faster mean-reversion)
BASE_ORDER_SIZE = 3000     # up from 1000 — we were only using 6% of gross limit
MAX_ORDER_SIZE = 10000
MAX_PER_TICKER = 5000      # hard cap per-ticker position to avoid blowups
EWMA_ALPHA = 0.3           # mid-price tracker decay

# Close-out thresholds (tick within each 60-tick minute)
UNWIND_LIMIT_START = 50    # was 45 — data shows we flatten in 5-7 ticks, no need to start so early
UNWIND_AGGRESSIVE = 54     # was 50
UNWIND_MARKET = 57

# Post-news behaviour
NEWS_WIDEN_TICKS = 3       # reduced from 5 — only widen for first 3 ticks
NEWS_SPREAD_MULT = 2.5     # slightly wider during news to avoid adverse selection
NEWS_SIZE_MULT = 0.3       # reduce order size during news ticks (new)

# EWMA state
ewma_mids = {t: None for t in TICKERS}


# ════════════════════════════════════════════════════════════
# DATA FETCHING — minimal API calls per loop
# ════════════════════════════════════════════════════════════

def fetch_tick():
    """GET /case → (tick, status, ticks_per_period)"""
    resp = s.get(f"{BASE}/case")
    if resp.ok:
        c = resp.json()
        return c["tick"], c["status"], c.get("ticks_per_period", 300)
    return 0, "STOPPED", 300


def fetch_all_securities():
    """Single GET /securities → dict keyed by ticker."""
    resp = s.get(f"{BASE}/securities")
    if not resp.ok:
        return {}
    return {
        sec["ticker"]: {
            "bid": sec.get("bid"),
            "ask": sec.get("ask"),
            "last": sec.get("last"),
            "position": sec.get("position", 0),
            "bid_size": sec.get("bid_size", 0),
            "ask_size": sec.get("ask_size", 0),
        }
        for sec in resp.json()
    }


def fetch_limits():
    """GET /limits → list of limit dicts."""
    resp = s.get(f"{BASE}/limits")
    return resp.json() if resp.ok else []


def cancel_all_orders():
    """Cancel every open order in one call."""
    s.post(f"{BASE}/commands/cancel", params={"all": 1})


def place_order(ticker, action, quantity, price=None):
    """Place a limit (if price given) or market order. Returns response dict or None."""
    quantity = min(int(quantity), MAX_ORDER_SIZE)
    if quantity <= 0:
        return None
    params = {
        "ticker": ticker,
        "type": "LIMIT" if price is not None else "MARKET",
        "quantity": quantity,
        "action": action,
    }
    if price is not None:
        params["price"] = round(price, 2)
    resp = s.post(f"{BASE}/orders", params=params)
    return resp.json() if resp.ok else None


# ════════════════════════════════════════════════════════════
# CORE ALGORITHM
# ════════════════════════════════════════════════════════════

def compute_aggregate_exposure(securities_data):
    """Sum of |position| across our four tickers."""
    return sum(abs(securities_data.get(t, {}).get("position", 0)) for t in TICKERS)


def update_ewma(ticker, current_mid):
    """Update EWMA and return |deviation| from the smoothed mid."""
    if ewma_mids[ticker] is None:
        ewma_mids[ticker] = current_mid
        return 0.0
    ewma_mids[ticker] = EWMA_ALPHA * current_mid + (1 - EWMA_ALPHA) * ewma_mids[ticker]
    return abs(current_mid - ewma_mids[ticker])


def compute_skewed_quotes(ticker, best_bid, best_ask, position, tick_in_minute):
    """
    Inventory-skewed quoting with penny improvement,
    news widening, and minimum profitable spread.
    """
    mid = (best_bid + best_ask) / 2
    market_half = (best_ask - best_bid) / 2

    # Inventory skew: if long, lower mid → more aggressive sell
    skewed_mid = mid - SKEW_FACTOR * position

    # Half-spread: penny-improve the market, but enforce minimum
    half_spread = max(MIN_HALF_SPREAD[ticker], market_half - 0.01)

    # Widen after news (first ticks of each minute)
    if tick_in_minute < NEWS_WIDEN_TICKS:
        half_spread *= NEWS_SPREAD_MULT

    our_bid = round(skewed_mid - half_spread, 2)
    our_ask = round(skewed_mid + half_spread, 2)

    # Safety: never quote through the market
    our_bid = min(our_bid, best_ask - 0.01)
    our_ask = max(our_ask, best_bid + 0.01)

    return our_bid, our_ask


def compute_order_size(ticker, position, aggregate_exposure, max_exposure, tick_in_minute):
    """
    Dynamic sizing: scales with headroom, decays with inventory,
    boosts for higher-rebate tickers, reduces during news.
    """
    headroom = max(0, max_exposure - aggregate_exposure)
    per_ticker = headroom / len(TICKERS)
    size = min(BASE_ORDER_SIZE, per_ticker)

    # Reduce as per-ticker inventory grows
    per_ticker_max = max_exposure / len(TICKERS)
    if per_ticker_max > 0:
        size *= max(0.1, 1.0 - abs(position) / per_ticker_max)

    # Hard cap: don't add to a position that's already at per-ticker limit
    if abs(position) >= MAX_PER_TICKER:
        size *= 0.1  # minimal quoting only

    # Rebate boost (normalised to SPNG baseline)
    size *= 1.0 + (REBATES[ticker] - 0.01) / 0.01

    # Post-news caution: smaller size for first few ticks after minute boundary
    if tick_in_minute < NEWS_WIDEN_TICKS:
        size *= NEWS_SIZE_MULT

    return max(100, min(int(size), MAX_ORDER_SIZE))


# ════════════════════════════════════════════════════════════
# TIERED CLOSE-OUT
# ════════════════════════════════════════════════════════════

def handle_unwind(tick_in_minute, securities_data):
    """
    Tiered unwind before minute close:
      45-49  passive limits (1/3 of position at best bid/ask)
      50-56  aggressive limits (penny inside, full position)
      57-59  market orders (guarantee flat)
    Returns list of (ticker, action, qty, price_or_None).
    """
    orders = []
    for ticker in TICKERS:
        sec = securities_data.get(ticker, {})
        position = sec.get("position", 0)
        if position == 0:
            continue

        best_bid = sec.get("bid")
        best_ask = sec.get("ask")
        if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
            continue

        action = "SELL" if position > 0 else "BUY"
        abs_pos = abs(position)

        if tick_in_minute >= UNWIND_MARKET:
            orders.append((ticker, action, abs_pos, None))
        elif tick_in_minute >= UNWIND_AGGRESSIVE:
            price = (best_bid + 0.01) if action == "SELL" else (best_ask - 0.01)
            orders.append((ticker, action, abs_pos, round(price, 2)))
        elif tick_in_minute >= UNWIND_LIMIT_START:
            unwind_qty = max(100, abs_pos // 3)
            price = best_bid if action == "SELL" else best_ask
            orders.append((ticker, action, unwind_qty, round(price, 2)))

    return orders


# ════════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════════

def open_log():
    """Create logs/ dir and open a timestamped log file."""
    log_dir = Path(__file__).parent.parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    fh = open(log_path, "w")
    # Header line
    fh.write("tick|t_min|agg_exp|gross|net|gross_lim|net_lim|"
             + "|".join(f"{t}:pos,bid,ask,our_bid,our_ask,qty" for t in TICKERS)
             + "|actions\n")
    return fh, log_path


def log_tick(fh, tick, tick_in_minute, securities_data, quotes, aggregate_exposure,
             limits, actions):
    """Write one structured line per tick."""
    gross = net = gross_lim = net_lim = 0
    if limits:
        lim = limits[0]
        gross = lim.get("gross", 0)
        net = lim.get("net", 0)
        gross_lim = lim.get("gross_limit", 0)
        net_lim = lim.get("net_limit", 0)

    parts = [str(tick), str(tick_in_minute), str(aggregate_exposure),
             f"{gross}", f"{net}", f"{gross_lim}", f"{net_lim}"]

    for t in TICKERS:
        sec = securities_data.get(t, {})
        q = quotes.get(t, {})
        parts.append(
            f"{sec.get('position', 0)},"
            f"{sec.get('bid', 0)},{sec.get('ask', 0)},"
            f"{q.get('our_bid', 0)},{q.get('our_ask', 0)},{q.get('qty', 0)}"
        )

    parts.append(";".join(actions) if actions else "")
    fh.write("|".join(parts) + "\n")
    fh.flush()


# ════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════

def main():
    print()
    print(f"  {BOLD}RITC Algorithmic Market Maker v2{RESET}")
    print(f"  {DIM}Ctrl+C to stop{RESET}")
    print()

    # Wait for case
    tick, status, _tpp = fetch_tick()
    while status != "ACTIVE":
        print(f"  {YELLOW}Waiting for case... (status={status}){RESET}")
        time.sleep(1)
        tick, status, _tpp = fetch_tick()

    # Read limits from API
    limits = fetch_limits()
    max_exposure = limits[0].get("gross_limit", 15000) if limits else 15000

    # Setup logging
    log_fh, log_path = open_log()

    print(f"  max_exposure={max_exposure} (from API)")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Rebates: { {t: f'${v:.3f}' for t, v in REBATES.items()} }")
    print(f"  Log: {log_path}")
    print(f"  {GREEN}ACTIVE — starting at tick {tick}{RESET}")
    print()

    try:
        while status == "ACTIVE":
            loop_start = time.time()

            # 1. Cancel all outstanding orders
            cancel_all_orders()

            # 2. Fetch state (3 API calls total)
            tick, status, _tpp = fetch_tick()
            if status != "ACTIVE":
                break
            securities_data = fetch_all_securities()
            limits = fetch_limits()
            if limits:
                max_exposure = limits[0].get("gross_limit", max_exposure)

            aggregate_exposure = compute_aggregate_exposure(securities_data)
            tick_in_minute = tick % 60

            # Exposure display
            exposure_pct = (aggregate_exposure / max_exposure * 100) if max_exposure else 0
            if exposure_pct >= 90:
                exp_c = RED
            elif exposure_pct >= 70:
                exp_c = YELLOW
            else:
                exp_c = GREEN

            actions = []
            quotes = {}  # for logging

            # 3. UNWIND or QUOTE
            if tick_in_minute >= UNWIND_LIMIT_START:
                phase = ("MKT" if tick_in_minute >= UNWIND_MARKET
                         else "AGG" if tick_in_minute >= UNWIND_AGGRESSIVE
                         else "LMT")
                print(f"  {RED}{BOLD}[tick {tick} | t={tick_in_minute}] UNWIND ({phase}){RESET}"
                      f"  exp: {exp_c}{aggregate_exposure}/{max_exposure}{RESET}")

                for ticker, action, qty, price in handle_unwind(tick_in_minute, securities_data):
                    # Split large orders into MAX_ORDER_SIZE chunks
                    remaining = qty
                    while remaining > 0:
                        chunk = min(remaining, MAX_ORDER_SIZE)
                        place_order(ticker, action, chunk, price)
                        remaining -= chunk
                    tag = "MKT" if price is None else f"LMT@{price:.2f}"
                    print(f"    {action} {qty:>6} {ticker} {tag}")
                    actions.append(f"unwind:{ticker}:{action}:{qty}:{tag}")

            else:
                print(f"  {DIM}[tick {tick} | t={tick_in_minute}]{RESET}"
                      f"  exp: {exp_c}{aggregate_exposure}/{max_exposure} ({exposure_pct:.0f}%){RESET}")

                for ticker in TICKERS:
                    sec = securities_data.get(ticker, {})
                    best_bid = sec.get("bid")
                    best_ask = sec.get("ask")
                    position = sec.get("position", 0)

                    if not best_bid or not best_ask or best_bid <= 0 or best_ask <= 0:
                        print(f"    {ticker}: {YELLOW}no book (bid={best_bid} ask={best_ask}){RESET}")
                        continue

                    mid = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid

                    # EWMA volatility check
                    vol_dev = update_ewma(ticker, mid)

                    # Compute skewed quotes
                    our_bid, our_ask = compute_skewed_quotes(
                        ticker, best_bid, best_ask, position, tick_in_minute
                    )

                    # Extra widening if mid is moving fast
                    if vol_dev > 0.10:
                        our_bid = round(our_bid - vol_dev * 0.5, 2)
                        our_ask = round(our_ask + vol_dev * 0.5, 2)

                    # Dynamic order size
                    order_qty = compute_order_size(
                        ticker, position, aggregate_exposure, max_exposure, tick_in_minute
                    )

                    # Quote if headroom allows
                    if aggregate_exposure + order_qty * 2 < max_exposure:
                        place_order(ticker, "BUY", order_qty, our_bid)
                        place_order(ticker, "SELL", order_qty, our_ask)
                        our_spd = our_ask - our_bid
                        act = f"{GREEN}Q {order_qty:>5} @ {our_bid:.2f}/{our_ask:.2f} spd={our_spd:.2f} reb=${REBATES[ticker]:.3f}{RESET}"
                        actions.append(f"quote:{ticker}:{order_qty}:{our_bid:.2f}/{our_ask:.2f}")
                    else:
                        act = f"{RED}at limit — skipped{RESET}"
                        actions.append(f"skip:{ticker}")

                    pos_c = RED if abs(position) > 3000 else YELLOW if abs(position) > 1500 else GREEN
                    print(f"    {ticker:<6} bid={best_bid:.2f} ask={best_ask:.2f}"
                          f" spd={spread:.2f} pos={pos_c}{position:>+6.0f}{RESET}  {act}")

                    quotes[ticker] = {"our_bid": our_bid, "our_ask": our_ask, "qty": order_qty}

            # 4. Log
            log_tick(log_fh, tick, tick_in_minute, securities_data, quotes,
                     aggregate_exposure, limits, actions)

            # 5. Pace the loop (~0.25s target cycle)
            elapsed = time.time() - loop_start
            time.sleep(max(0.05, 0.25 - elapsed))

    except KeyboardInterrupt:
        print(f"\n  {BOLD}Shutting down.{RESET}")
    finally:
        log_fh.close()

    print(f"  {DIM}Final: status={status}, tick={tick}{RESET}")
    print(f"  Log saved to {log_path}")


if __name__ == "__main__":
    main()
