"""
RITC 2026 - GUI-First Liquidity Monitor (v4)

WORKFLOW:
  1. Run this script alongside the RIT Client
  2. Script shows tender offers with edge calculations
  3. YOU accept/decline tenders in the RIT GUI (faster, no typing!)
  4. Script DETECTS when you accept (position changes)
  5. Script AUTO-FIRES limit orders for unwinding
  6. SAFETY: Auto-cancels orders when position > 15,000 (configurable)

This lets you focus on trading while the script handles automation.

Usage: python liquidity_monitor.py
"""

import requests
import time
import threading
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
API_PORT = getenv("ROT_API_PORT")
# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "http://localhost:9999/v1"

# Safety cutoff: Cancel all limit orders once residual position is small
POSITION_SAFETY_THRESHOLD = 15000  # Absolute shares (either direction)
POSITION_LIMIT = 25000  # Actual case limit for reference

MAX_ORDER_SIZE = 10000  # Max shares per single order
AUCTION_MARGIN = 0.04   # Cents buffer on auction bids
PRICE_TICK = 0.01  # Tick size for anti-cross clamp
AUTO_MARKET_FLATTEN_THRESHOLD = 10000  # Auto-flatten if abs(position) is below this
TENDER_DETECTION_MIN_DELTA = 25000  # Min position jump to classify as tender fill

# Polling intervals (seconds)
POSITION_POLL_INTERVAL = 0.10  # Faster polling reduces batched manual-fill deltas
TENDER_POLL_INTERVAL = 1.0    # How often to check for new tenders

# Fixed unwind/reprice policy
UNWIND_HEDGE_RATIO = 0.80
POSITION_FREEZE_THRESHOLD = 20000
TREND_WARMUP_SECONDS = 3.5
TREND_REPRICE_MIN_INTERVAL = 0.5
TREND_MIN_FAVORABLE_MOVE = 0.01
ORDER_STAGGER_SECONDS = 0.02

# Tender fixed-price projected P&L model
TENDER_EDGE_VWAP_COVERAGE_RATIO = 0.65
TENDER_PROFIT_RECOMMEND = 5000
TENDER_PROFIT_STRONG = 10000
TENDER_PROFIT_RECOMMEND_HIGH_RISK = 8000

# Unwind pricing matrix
UNWIND_LADDER_INCREMENTS_CENTS = [0, 1, 2, 3, 4, 5]
UNWIND_OFFSET_MATRIX_CENTS = {
    ("High", "Low"): 4,
    ("High", "Medium"): 6,
    ("High", "High"): 8,
    ("Medium", "Low"): 6,
    ("Medium", "Medium"): 8,
    ("Medium", "High"): 12,
    ("Low", "Low"): 9,
    ("Low", "Medium"): 12,
    ("Low", "High"): 15,
}

# ============================================================
# ANSI COLOUR CODES
# ============================================================
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DARK_GREEN = "\033[32m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ============================================================
# TICKER CHARACTERISTICS - from case package
# ============================================================
TICKER_INFO = {
    # Sub-Heat 1
    "RITC": {"volatility": "Low",    "liquidity": "Medium", "commission": 0.02, "start_price": 50},
    "COMP": {"volatility": "Medium", "liquidity": "High",   "commission": 0.02, "start_price": 40},
    # Sub-Heat 2
    "TRNT": {"volatility": "High",   "liquidity": "Medium", "commission": 0.01, "start_price": 15},
    "MTRL": {"volatility": "Low",    "liquidity": "Low",    "commission": 0.01, "start_price": 30},
    # Sub-Heat 3
    "BLU":  {"volatility": "High",   "liquidity": "High",   "commission": 0.04, "start_price": 10},
    "RED":  {"volatility": "Low",    "liquidity": "Medium", "commission": 0.03, "start_price": 25},
    "GRN":  {"volatility": "Medium", "liquidity": "Medium", "commission": 0.02, "start_price": 30},
    # Sub-Heat 4
    "WDY":  {"volatility": "Medium", "liquidity": "High",   "commission": 0.02, "start_price": 12},
    "BZZ":  {"volatility": "High",   "liquidity": "Medium", "commission": 0.02, "start_price": 18},
    "BNN":  {"volatility": "Medium", "liquidity": "Medium", "commission": 0.03, "start_price": 24},
    # Sub-Heat 5
    "VNS":  {"volatility": "High",   "liquidity": "Medium", "commission": 0.02, "start_price": 20},
    "MRS":  {"volatility": "Medium", "liquidity": "High",   "commission": 0.02, "start_price": 75},
    "JPTR": {"volatility": "Low",    "liquidity": "Medium", "commission": 0.02, "start_price": 35},
    "STRN": {"volatility": "High",   "liquidity": "Medium", "commission": 0.02, "start_price": 50},
}

# ============================================================
# GLOBAL STATE
# ============================================================
s = requests.Session()
s.headers.update({"X-API-Key": API_KEY})

shutdown_flag = False

# Track positions to detect tender fills
# { "TICKER": {"position": 0, "vwap": 0, "last_change_time": None} }
position_tracker = defaultdict(lambda: {"position": 0, "vwap": 0, "last_change_time": None, "entry_price": None})

# Track active managed unwind state
managed_unwind_tracker = defaultdict(
    lambda: {
        "active": False,
        "unwind_action": None,
        "tender_detected_at": None,
        "trend_ref_last": None,
        "last_reprice_time": 0.0,
        "last_managed_position": 0,
    }
)

# Track which tenders we've seen (to avoid re-announcing)
seen_tender_ids = set()

# Track tickers with active limit orders (for safety monitoring)
active_order_tickers = set()
# Track tickers already auto-flattened while below threshold (avoid re-firing each poll)
auto_market_flatten_triggered = set()

# Lock for thread safety
state_lock = threading.Lock()

# ============================================================
# API HELPERS
# ============================================================
def api_get(endpoint, params=None):
    """Send a GET request to the RIT API."""
    try:
        resp = s.get(f"{BASE_URL}{endpoint}", params=params or {})
        if resp.ok:
            return resp.json()
        return None
    except requests.ConnectionError:
        return None

def api_post(endpoint, params=None):
    """Send a POST request to the RIT API."""
    try:
        resp = s.post(f"{BASE_URL}{endpoint}", params=params or {})
        if resp.ok:
            return resp.json()
        return None
    except requests.ConnectionError:
        return None

def to_float(value, default=None):
    """Best-effort float conversion."""
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return default
        try:
            return float(text)
        except ValueError:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# TENDER HELPERS
# ============================================================
def get_ticker_warnings(ticker):
    """Return coloured warning string if ticker is risky."""
    info = TICKER_INFO.get(ticker.upper())
    if not info:
        return ""
    
    warnings = []
    if info["volatility"] == "High":
        warnings.append(f"{RED}{BOLD}[WARN] HIGH VOL{RESET}")
    if info["liquidity"] == "Low":
        warnings.append(f"{RED}{BOLD}[WARN] LOW LIQ{RESET}")
    
    if not warnings:
        return f"{GREEN}SAFE{RESET}"
    
    return "  ".join(warnings)


def get_bucket_base_offset_cents(ticker):
    """Return (base_offset_cents, liquidity, volatility) for unwind pricing."""
    info = TICKER_INFO.get(ticker.upper(), {})
    liquidity = str(info.get("liquidity", "Medium")).title()
    volatility = str(info.get("volatility", "Medium")).title()
    
    if liquidity not in ("Low", "Medium", "High"):
        liquidity = "Medium"
    if volatility not in ("Low", "Medium", "High"):
        volatility = "Medium"
    
    base = UNWIND_OFFSET_MATRIX_CENTS.get((liquidity, volatility), 8)
    return base, liquidity, volatility


def is_exposure_increasing(old_position, new_position):
    """True when absolute exposure increases after a position change."""
    return abs(new_position) > abs(old_position)


def get_tender_recommend_threshold(ticker):
    """Recommendation threshold by ticker risk bucket."""
    info = TICKER_INFO.get(str(ticker).upper(), {})
    if info.get("volatility") == "High" and info.get("liquidity") == "Low":
        return TENDER_PROFIT_RECOMMEND_HIGH_RISK
    return TENDER_PROFIT_RECOMMEND


def calc_unwind_vwap(ticker, qty, unwind_side):
    """Calculate VWAP to unwind a position by walking the order book."""
    book = api_get("/securities/book", params={"ticker": ticker, "limit": 50})
    if not book:
        return None, 0, 0
    
    if unwind_side == "SELL":
        levels = book.get("bids", [])
    else:
        levels = book.get("asks", [])
    
    if not levels:
        return None, 0, 0
    
    total_cost = 0.0
    total_filled = 0
    levels_used = 0
    remaining = qty
    
    for level in levels:
        price = level.get("price", 0)
        volume = level.get("quantity", 0)
        
        if price <= 0 or volume <= 0:
            continue
        
        fill_here = min(remaining, volume)
        total_cost += price * fill_here
        total_filled += fill_here
        remaining -= fill_here
        levels_used += 1
        
        if remaining <= 0:
            break
    
    if total_filled == 0:
        return None, 0, 0
    
    vwap = total_cost / total_filled
    return vwap, total_filled, levels_used


def calc_tender_edge(ticker, action, qty, price):
    """Projected fixed-price tender P&L (MTM minus unwind slippage cost)."""
    if price is None:
        return None
    if qty is None or qty <= 0:
        return None
    
    # Get current market price
    sec_data = api_get("/securities", params={"ticker": ticker})
    if not sec_data:
        return None
    
    sec = sec_data[0] if isinstance(sec_data, list) else sec_data
    bid = to_float(sec.get("bid"), default=0) or 0
    ask = to_float(sec.get("ask"), default=0) or 0
    last = to_float(sec.get("last"), default=0) or 0

    if action == "BUY":
        # We buy from tender, then unwind by selling.
        market_ref = ask if ask > 0 else last
        unwind_side = "SELL"
    else:
        # We sell to tender, then unwind by buying.
        market_ref = bid if bid > 0 else last
        unwind_side = "BUY"

    if market_ref <= 0:
        return None

    mtm_per_share = (market_ref - price) if action == "BUY" else (price - market_ref)
    mtm_component = mtm_per_share * qty

    unwind_qty = max(1, int(round(qty * TENDER_EDGE_VWAP_COVERAGE_RATIO)))
    unwind_vwap, filled, levels = calc_unwind_vwap(ticker, unwind_qty, unwind_side)

    if unwind_vwap is None or filled <= 0:
        return {
            "projected_profit": None,
            "mtm_component": mtm_component,
            "unwind_slippage_cost": None,
            "market_ref": market_ref,
            "tender_price": price,
            "unwind_vwap": None,
            "unwind_qty": unwind_qty,
            "unwind_filled": filled,
            "coverage_pct": 0,
            "unwind_side": unwind_side,
            "levels_used": 0,
        }

    if unwind_side == "SELL":
        unwind_ref = bid if bid > 0 else last
        slippage_per_share = max(0.0, (unwind_ref - unwind_vwap))
    else:
        unwind_ref = ask if ask > 0 else last
        slippage_per_share = max(0.0, (unwind_vwap - unwind_ref))

    unwind_slippage_cost = slippage_per_share * filled
    projected_profit = mtm_component - unwind_slippage_cost
    coverage_pct = (filled / unwind_qty) * 100 if unwind_qty > 0 else 0

    return {
        "projected_profit": projected_profit,
        "mtm_component": mtm_component,
        "unwind_slippage_cost": unwind_slippage_cost,
        "market_ref": market_ref,
        "tender_price": price,
        "unwind_vwap": unwind_vwap,
        "unwind_qty": unwind_qty,
        "unwind_filled": filled,
        "coverage_pct": coverage_pct,
        "unwind_side": unwind_side,
        "levels_used": levels,
    }


def calc_auction_price(ticker, action, qty):
    """Calculate recommended auction bid price."""
    info = TICKER_INFO.get(ticker.upper(), {})
    commission = info.get("commission", 0.02)
    margin = AUCTION_MARGIN
    
    if action == "BUY":
        vwap, filled, levels = calc_unwind_vwap(ticker, qty, "SELL")
        if vwap is None:
            return None, None, 0, margin, commission
        coverage_pct = (filled / qty) * 100 if qty > 0 else 0
        recommended = round(vwap - margin - (2 * commission), 2)
        return recommended, vwap, coverage_pct, margin, commission
    else:
        vwap, filled, levels = calc_unwind_vwap(ticker, qty, "BUY")
        if vwap is None:
            return None, None, 0, margin, commission
        coverage_pct = (filled / qty) * 100 if qty > 0 else 0
        recommended = round(vwap + margin + (2 * commission), 2)
        return recommended, vwap, coverage_pct, margin, commission


# ============================================================
# ORDER PLACEMENT
# ============================================================
def clear_managed_unwind_state(ticker):
    """Clear ticker state after unwind completes or is disarmed."""
    with state_lock:
        state = managed_unwind_tracker[ticker]
        state["active"] = False
        state["unwind_action"] = None
        state["tender_detected_at"] = None
        state["trend_ref_last"] = None
        state["last_reprice_time"] = 0.0
        state["last_managed_position"] = 0
        active_order_tickers.discard(ticker)
        auto_market_flatten_triggered.discard(ticker)


def get_unwind_anchor_price(ticker, unwind_action):
    """
    Build a single quote snapshot and return:
    (anchor_price, best_bid, best_ask).

    Anchor policy:
    - SELL ladder anchor = best bid (fallback last, then best ask)
    - BUY ladder anchor  = best ask (fallback last, then best bid)
    """
    book = api_get("/securities/book", params={"ticker": ticker, "limit": 1}) or {}
    sec_data = api_get("/securities", params={"ticker": ticker})
    sec = sec_data[0] if isinstance(sec_data, list) and sec_data else (sec_data or {})

    best_bid = None
    best_ask = None

    if book.get("bids"):
        best_bid = to_float(book["bids"][0].get("price"), default=None)
    if book.get("asks"):
        best_ask = to_float(book["asks"][0].get("price"), default=None)

    last = to_float(sec.get("last"), default=None)
    anchor_price = None

    if unwind_action == "SELL":
        if best_bid is not None and best_bid > 0:
            anchor_price = best_bid
        elif last is not None and last > 0:
            anchor_price = last
        elif best_ask is not None and best_ask > 0:
            anchor_price = best_ask
    else:
        if best_ask is not None and best_ask > 0:
            anchor_price = best_ask
        elif last is not None and last > 0:
            anchor_price = last
        elif best_bid is not None and best_bid > 0:
            anchor_price = best_bid

    return anchor_price, best_bid, best_ask


def place_unwind_limits(ticker, position_qty, reason="reconcile", cancel_existing=True):
    """
    Reconcile limit ladder to the current position with fixed hedge ratio.
    Returns True if at least one order is placed.
    """
    if position_qty == 0:
        return False

    abs_qty = abs(position_qty)
    if abs_qty < POSITION_FREEZE_THRESHOLD:
        return False

    unwind_action = "SELL" if position_qty > 0 else "BUY"
    base_offset_cents, _, _ = get_bucket_base_offset_cents(ticker)
    ladder_cents = [base_offset_cents + inc for inc in UNWIND_LADDER_INCREMENTS_CENTS]

    target_qty = int(round(abs_qty * UNWIND_HEDGE_RATIO))
    if target_qty <= 0:
        return False

    if cancel_existing:
        cancel_all_orders_for_ticker(ticker)

    anchor_price, best_bid, best_ask = get_unwind_anchor_price(ticker, unwind_action)
    if anchor_price is None or anchor_price <= 0:
        print(f"  {YELLOW}Could not anchor unwind prices for {ticker}; skipping limit refresh.{RESET}")
        return False
    if unwind_action == "SELL" and (best_bid is None or best_bid <= 0):
        print(f"  {YELLOW}Missing best bid for {ticker}; skipping SELL ladder refresh.{RESET}")
        return False
    if unwind_action == "BUY" and (best_ask is None or best_ask <= 0):
        print(f"  {YELLOW}Missing best ask for {ticker}; skipping BUY ladder refresh.{RESET}")
        return False

    num_orders = min(6, len(ladder_cents))
    per_order_qty = min(target_qty // num_orders, MAX_ORDER_SIZE)
    if per_order_qty == 0:
        per_order_qty = min(target_qty, MAX_ORDER_SIZE)
        num_orders = 1

    remainder = target_qty - (per_order_qty * num_orders)
    if unwind_action == "SELL":
        ladder_label = ", ".join(f"+{c}c" for c in ladder_cents[:num_orders])
    else:
        ladder_label = ", ".join(f"-{c}c" for c in ladder_cents[:num_orders])

    print(f"\n  {CYAN}{BOLD}>> LIMIT RECONCILE ({reason}): {ticker} {unwind_action}{RESET}")
    print(f"  {CYAN}   Target hedge: {UNWIND_HEDGE_RATIO*100:.0f}% of {abs_qty:,} = {target_qty:,} shares{RESET}")
    print(f"  {CYAN}   Anchor: {anchor_price:.2f}  |  Ladder: {ladder_label}{RESET}")

    placed_count = 0
    total_placed_qty = 0

    for i in range(num_orders):
        if total_placed_qty >= target_qty:
            break

        this_qty = per_order_qty
        if i == 0 and remainder > 0:
            this_qty = min(per_order_qty + remainder, MAX_ORDER_SIZE)
        this_qty = min(this_qty, target_qty - total_placed_qty, MAX_ORDER_SIZE)
        if this_qty <= 0:
            break

        offset_dollars = ladder_cents[i] / 100.0
        if unwind_action == "SELL":
            # Long unwind: place sell limits above best bid by margin ladder.
            raw_price = anchor_price + offset_dollars
            min_passive = best_bid + PRICE_TICK
            price = round(max(raw_price, min_passive), 2)
        else:
            # Short unwind: place buy limits below best ask by margin ladder.
            raw_price = anchor_price - offset_dollars
            max_passive = best_ask - PRICE_TICK
            price = round(min(raw_price, max_passive), 2)

        if price <= 0:
            print(f"  {YELLOW}   Skipping {unwind_action} {this_qty} at invalid price {price:.2f}{RESET}")
            continue

        params = {
            "ticker": ticker,
            "type": "LIMIT",
            "quantity": this_qty,
            "action": unwind_action,
            "price": price,
        }

        result = api_post("/orders", params=params)
        if result:
            oid = result.get("order_id", "?")
            filled = result.get("quantity_filled", 0)
            total_placed_qty += this_qty
            placed_count += 1
            if filled > 0:
                print(f"  {GREEN}   #{oid}: {unwind_action} {this_qty} @ {price:.2f} - {filled} FILLED!{RESET}")
            else:
                print(f"     #{oid}: {unwind_action} {this_qty} @ {price:.2f}")

        if ORDER_STAGGER_SECONDS > 0:
            time.sleep(ORDER_STAGGER_SECONDS)

    if placed_count > 0:
        with state_lock:
            active_order_tickers.add(ticker)

    print(f"  {CYAN}   Placed {placed_count} orders ({total_placed_qty} shares){RESET}")
    return placed_count > 0


def cancel_all_orders_for_ticker(ticker):
    """Cancel all open orders for a specific ticker."""
    result = api_post("/commands/cancel", params={"query": f"Ticker='{ticker}'"})
    return result is not None


def cancel_all_orders():
    """Cancel ALL open orders."""
    result = api_post("/commands/cancel", params={"query": ""})
    return result is not None


def market_flatten_position(ticker, position_qty):
    """Flatten a position with MARKET orders in MAX_ORDER_SIZE chunks."""
    if position_qty == 0:
        return True

    if position_qty > 0:
        action = "SELL"
        remaining = position_qty
    else:
        action = "BUY"
        remaining = abs(position_qty)

    all_ok = True
    while remaining > 0:
        chunk = min(remaining, MAX_ORDER_SIZE)
        params = {
            "ticker": ticker,
            "type": "MARKET",
            "quantity": chunk,
            "action": action,
        }
        result = api_post("/orders", params=params)
        if result:
            filled = result.get("quantity_filled", 0)
            vwap = result.get("vwap", 0)
            print(f"  {GREEN}   MKT {action} {chunk} {ticker} -> filled {filled} @ {vwap:.2f}{RESET}")
        else:
            print(f"  {RED}   MKT {action} {chunk} {ticker} failed{RESET}")
            all_ok = False
            break
        remaining -= chunk
        if remaining > 0:
            time.sleep(0.1)

    return all_ok


# ============================================================
# POSITION MONITORING THREAD
# ============================================================
def position_monitor_loop():
    """
    Main monitoring loop:
    1. Detects tender fills from large exposure-increasing position jumps
    2. Reconciles/cancel-replaces limit orders on every unwind position change
    3. Freezes updates below 20k and cancels limits below 15k
    4. Reprices on favorable trend only after warm-up
    5. Auto-flattens any non-zero position below threshold
    """
    global position_tracker, active_order_tickers, auto_market_flatten_triggered
    print(f"\n  {CYAN}Position monitor started. Watching for tender fills...{RESET}\n")
    while not shutdown_flag:
        try:
            data = api_get("/securities")
            if not data:
                time.sleep(POSITION_POLL_INTERVAL)
                continue

            current_time = time.time()
            for sec in data:
                ticker = sec.get("ticker", "")
                if not ticker:
                    continue

                new_position = int(sec.get("position", 0))
                new_vwap = to_float(sec.get("vwap"), default=0) or 0
                last_price = to_float(sec.get("last"), default=0) or 0

                with state_lock:
                    old_data = position_tracker[ticker]
                    old_position = int(old_data.get("position", 0))
                    position_delta = new_position - old_position
                    position_tracker[ticker]["position"] = new_position
                    position_tracker[ticker]["vwap"] = new_vwap
                    managed_state = dict(managed_unwind_tracker[ticker])

                tender_detected = (
                    abs(position_delta) >= TENDER_DETECTION_MIN_DELTA
                    and is_exposure_increasing(old_position, new_position)
                )

                if tender_detected:
                    print(f"\n  {MAGENTA}{BOLD}!! TENDER DETECTED: {ticker}{RESET}")
                    print(f"  {MAGENTA}   Position: {old_position:+,} -> {new_position:+,} (delta {position_delta:+,}){RESET}")
                    with state_lock:
                        position_tracker[ticker]["entry_price"] = new_vwap
                        position_tracker[ticker]["last_change_time"] = current_time
                        managed_unwind_tracker[ticker]["active"] = True
                        managed_unwind_tracker[ticker]["unwind_action"] = "SELL" if new_position > 0 else "BUY"
                        managed_unwind_tracker[ticker]["tender_detected_at"] = current_time
                        managed_unwind_tracker[ticker]["trend_ref_last"] = last_price if last_price > 0 else None
                        managed_unwind_tracker[ticker]["last_reprice_time"] = 0.0
                        managed_unwind_tracker[ticker]["last_managed_position"] = new_position
                        managed_state = dict(managed_unwind_tracker[ticker])

                    if abs(new_position) >= POSITION_FREEZE_THRESHOLD:
                        place_unwind_limits(ticker, new_position, reason="tender_detected", cancel_existing=True)

                abs_position = abs(new_position)
                flat_processed = False

                if managed_state.get("active"):
                    if new_position == 0:
                        print(f"\n  {GREEN}{BOLD}>> FLAT: {ticker} - Cancelling remaining orders{RESET}")
                        cancel_all_orders_for_ticker(ticker)
                        clear_managed_unwind_state(ticker)
                        with state_lock:
                            position_tracker[ticker]["entry_price"] = None
                        flat_processed = True
                    elif abs_position < POSITION_SAFETY_THRESHOLD:
                        print(f"\n  {RED}{BOLD}!! SAFETY TRIGGER: {ticker} at {new_position:+,} shares{RESET}")
                        print(f"  {RED}   Below {POSITION_SAFETY_THRESHOLD:,} shares - CANCELLING ALL LIMIT ORDERS{RESET}")
                        cancel_all_orders_for_ticker(ticker)
                        clear_managed_unwind_state(ticker)
                    elif abs_position < POSITION_FREEZE_THRESHOLD:
                        with state_lock:
                            managed_unwind_tracker[ticker]["last_managed_position"] = new_position
                    else:
                        refresh_reason = None
                        should_refresh = False

                        if position_delta != 0 and not tender_detected:
                            should_refresh = True
                            refresh_reason = "position_delta"
                        else:
                            trend_ref = managed_state.get("trend_ref_last")
                            tender_detected_at = managed_state.get("tender_detected_at")
                            last_reprice_time = float(managed_state.get("last_reprice_time") or 0.0)
                            unwind_action = managed_state.get("unwind_action") or ("SELL" if new_position > 0 else "BUY")

                            warmup_passed = (
                                tender_detected_at is not None
                                and (current_time - tender_detected_at) >= TREND_WARMUP_SECONDS
                            )
                            interval_passed = (current_time - last_reprice_time) >= TREND_REPRICE_MIN_INTERVAL

                            if warmup_passed and interval_passed and trend_ref is not None and last_price > 0:
                                move = last_price - trend_ref
                                if unwind_action == "SELL":
                                    favorable = move >= TREND_MIN_FAVORABLE_MOVE
                                else:
                                    favorable = move <= -TREND_MIN_FAVORABLE_MOVE
                                if favorable:
                                    should_refresh = True
                                    refresh_reason = "favorable_trend"

                        if should_refresh:
                            place_unwind_limits(
                                ticker,
                                new_position,
                                reason=refresh_reason,
                                cancel_existing=True,
                            )
                            with state_lock:
                                managed_unwind_tracker[ticker]["last_managed_position"] = new_position
                                if refresh_reason == "favorable_trend":
                                    managed_unwind_tracker[ticker]["last_reprice_time"] = current_time
                                    if last_price > 0:
                                        managed_unwind_tracker[ticker]["trend_ref_last"] = last_price
                                elif last_price > 0 and managed_unwind_tracker[ticker].get("trend_ref_last") is None:
                                    managed_unwind_tracker[ticker]["trend_ref_last"] = last_price
                        elif last_price > 0 and managed_state.get("trend_ref_last") is None:
                            with state_lock:
                                managed_unwind_tracker[ticker]["trend_ref_last"] = last_price

                if not flat_processed and old_position != 0 and new_position == 0:
                    print(f"\n  {GREEN}{BOLD}>> FLAT: {ticker} - Cancelling remaining orders{RESET}")
                    cancel_all_orders_for_ticker(ticker)
                    clear_managed_unwind_state(ticker)
                    with state_lock:
                        position_tracker[ticker]["entry_price"] = None

                # Auto market flatten rule requested by user
                should_auto_flatten = False
                with state_lock:
                    if 0 < abs_position < AUTO_MARKET_FLATTEN_THRESHOLD and ticker not in auto_market_flatten_triggered:
                        auto_market_flatten_triggered.add(ticker)
                        should_auto_flatten = True
                    elif abs_position >= AUTO_MARKET_FLATTEN_THRESHOLD or abs_position == 0:
                        auto_market_flatten_triggered.discard(ticker)
                if should_auto_flatten:
                    print(f"\n  {YELLOW}{BOLD}>> AUTO MKT FLATTEN: {ticker} {new_position:+,} shares (< {AUTO_MARKET_FLATTEN_THRESHOLD:,}){RESET}")
                    print(f"  {YELLOW}   Sending opposite MARKET order for {abs_position:,} shares{RESET}")
                    cancel_all_orders_for_ticker(ticker)
                    flatten_ok = market_flatten_position(ticker, new_position)
                    if flatten_ok:
                        with state_lock:
                            active_order_tickers.discard(ticker)
                    else:
                        with state_lock:
                            auto_market_flatten_triggered.discard(ticker)
        except Exception:
            pass
        time.sleep(POSITION_POLL_INTERVAL)
# ============================================================
# TENDER DISPLAY THREAD
# ============================================================
def tender_display_loop():
    """Display new tenders with edge calculations."""
    global seen_tender_ids
    
    while not shutdown_flag:
        try:
            tenders = api_get("/tenders")
            if tenders:
                for t in tenders:
                    tid = t.get("tender_id")
                    if tid is None or str(tid) in seen_tender_ids:
                        continue
                    
                    seen_tender_ids.add(str(tid))
                    
                    ticker = t.get("ticker", "?")
                    action = t.get("action", "?")
                    qty = t.get("quantity", 0)
                    price_raw = t.get("price")
                    price_num = to_float(price_raw, default=None)
                    expires = t.get("expires", "?")
                    is_fixed = t.get("is_fixed_bid", True)
                    
                    warnings = get_ticker_warnings(ticker)
                    
                    # Check if it's competitive (auction) or fixed price
                    if not is_fixed or price_num is None:
                        # COMPETITIVE AUCTION
                        rec_price, unwind_vwap, coverage, margin, commission = calc_auction_price(
                            ticker, action, qty
                        )
                        
                        print(f"\n  {RED}{BOLD}==== AUCTION #{tid}: {action} {qty:,} {ticker} ===={RESET}")
                        print(f"  {warnings}")
                        
                        if rec_price is not None:
                            print(f"  Unwind VWAP: {unwind_vwap:.4f}  (book covers {coverage:.0f}%)")
                            print(f"  {CYAN}{BOLD}>> RECOMMENDED BID: {rec_price:.2f}{RESET}")
                            if coverage < 100:
                                print(f"  {RED}[WARN] Book only {coverage:.0f}% deep - real cost worse!{RESET}")
                        else:
                            print(f"  {RED}Could not calc VWAP - thin book{RESET}")
                        
                        print(f"  {DIM}Accept in GUI, or type: bid {tid}{RESET}")
                    
                    else:
                        # FIXED PRICE TENDER
                        edge = calc_tender_edge(ticker, action, qty, price_num)
                        rec_threshold = get_tender_recommend_threshold(ticker)

                        if edge and edge.get("projected_profit") is not None:
                            projected = edge["projected_profit"]
                            if projected >= TENDER_PROFIT_STRONG:
                                edge_color = DARK_GREEN
                            elif projected >= rec_threshold:
                                edge_color = GREEN
                            else:
                                edge_color = RED

                            edge_str = f"proj ${projected:,.0f}"
                            unwind_vwap = edge.get("unwind_vwap")
                            coverage_pct = edge.get("coverage_pct", 0)
                            mtm_component = edge.get("mtm_component", 0)
                            slippage = edge.get("unwind_slippage_cost", 0)
                            print(f"\n  {BOLD}==== TENDER #{tid}: {action} {qty:,} {ticker} @ {price_num:.2f} ===={RESET}")
                            print(f"  {warnings}  |  {edge_color}{edge_str}{RESET}  |  rec >= ${rec_threshold:,.0f}  |  expires tick {expires}")
                            print(f"  MTM: {mtm_component:+,.0f}  |  Unwind cost: {slippage:,.0f}  |  VWAP(65%): {unwind_vwap:.4f} ({coverage_pct:.0f}% fill)")
                        else:
                            edge_color = YELLOW
                            edge_str = "proj unknown"
                            mtm_component = edge.get("mtm_component", 0) if edge else 0
                            print(f"\n  {BOLD}==== TENDER #{tid}: {action} {qty:,} {ticker} @ {price_num:.2f} ===={RESET}")
                            print(f"  {warnings}  |  {edge_color}{edge_str}{RESET}  |  expires tick {expires}")
                            print(f"  MTM component: {mtm_component:+,.0f}  |  VWAP depth unavailable")

                        print(f"  {DIM}Accept/decline in RIT GUI - script will auto-fire limits{RESET}")
                    
                    print("> ", end="", flush=True)
        
        except Exception:
            pass
        
        time.sleep(TENDER_POLL_INTERVAL)


# ============================================================
# POSITION DISPLAY
# ============================================================
def display_positions():
    """Show current positions with entry prices and P&L."""
    data = api_get("/securities")
    if not data:
        print("  Cannot connect to RIT.")
        return
    
    held = [sec for sec in data if sec.get("position", 0) != 0]
    
    if not held:
        print("  No open positions.")
        return
    
    print(f"\n  {'Ticker':<8} {'Pos':>10} {'Entry':>8} {'Last':>8} {'Unrl P&L':>10} {'% to Limit':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    
    total_unrl = 0
    for sec in held:
        ticker = sec.get("ticker", "?")
        position = sec.get("position", 0)
        vwap = sec.get("vwap", 0)
        last = sec.get("last", 0)
        unrealised = sec.get("unrealized", 0)
        total_unrl += unrealised
        
        # Get entry price from tracker if available
        with state_lock:
            entry = position_tracker[ticker].get("entry_price", vwap) or vwap
        
        # Calculate % towards limit
        pct_of_limit = (abs(position) / POSITION_LIMIT) * 100
        if pct_of_limit > 60:
            pct_color = RED
        elif pct_of_limit > 40:
            pct_color = YELLOW
        else:
            pct_color = GREEN
        
        print(f"  {ticker:<8} {position:>+10,} {entry:>8.2f} {last:>8.2f} {unrealised:>+10.2f} {pct_color}{pct_of_limit:>9.1f}%{RESET}")
    
    trader = api_get("/trader")
    realised = 0
    if trader:
        realised = trader.get("realized", 0) if isinstance(trader, dict) else 0
    
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<8} {'':>10} {'':>8} {'':>8} {total_unrl:>+10.2f}")
    print(f"\n  Realised: {realised:+.2f}  |  Combined: {realised + total_unrl:+.2f}")


def display_open_orders():
    """Show open orders."""
    orders = api_get("/orders", params={"status": "OPEN"})
    if not orders or len(orders) == 0:
        print("  No open orders.")
        return
    
    print(f"\n  {'ID':<8} {'Side':<6} {'Ticker':<8} {'Qty':>8} {'Filled':>8} {'Price':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    
    for o in orders:
        oid = o.get("order_id", "?")
        side = o.get("action", "?")
        ticker = o.get("ticker", "?")
        qty = o.get("quantity", 0)
        filled = o.get("quantity_filled", 0)
        price = o.get("price", 0)
        print(f"  {oid:<8} {side:<6} {ticker:<8} {qty:>8} {filled:>8} {price:>10.2f}")


# ============================================================
# MANUAL COMMANDS (minimal - most trading via GUI)
# ============================================================
def cmd_bid(args):
    """Submit auction bid (for when you want script to place it)."""
    if not args:
        print("  Usage: bid TENDER_ID [PRICE]")
        return
    
    tender_id = args[0]
    
    # Look up tender
    tenders = api_get("/tenders")
    tender_info = None
    if tenders:
        for t in tenders:
            if str(t.get("tender_id")) == str(tender_id):
                tender_info = t
                break
    
    if not tender_info:
        print(f"  Tender #{tender_id} not found.")
        return
    
    ticker = tender_info.get("ticker", "")
    action = tender_info.get("action", "")
    qty = int(tender_info.get("quantity", 0))
    
    # Get price
    if len(args) >= 2:
        try:
            price = round(float(args[1]), 2)
        except ValueError:
            print(f"  Bad price: {args[1]}")
            return
    else:
        rec_price, _, _, _, _ = calc_auction_price(ticker, action, qty)
        if rec_price is None:
            print("  Could not calculate auction price. Specify manually: bid ID PRICE")
            return
        price = rec_price
        print(f"  Using calculated price: {price:.2f}")
    
    # Submit
    result = api_post(f"/tenders/{tender_id}", params={"price": price})
    if result:
        print(f"  {GREEN}Bid submitted: #{tender_id} @ {price:.2f}{RESET}")
        # Position monitor will detect the fill and place limits


def cmd_flat(args):
    """Flatten a position with market orders."""
    if not args:
        print("  Usage: flat TICKER")
        return
    
    ticker = args[0].upper()
    
    data = api_get("/securities", params={"ticker": ticker})
    if not data:
        return
    
    sec = data[0] if isinstance(data, list) else data
    position = sec.get("position", 0)
    
    if position == 0:
        print(f"  Already flat in {ticker}.")
        return
    
    if position > 0:
        action = "SELL"
        qty = position
    else:
        action = "BUY"
        qty = abs(position)
    
    print(f"  Flattening: {action} {qty} {ticker} @ MKT...")
    
    remaining = qty
    while remaining > 0:
        chunk = min(remaining, MAX_ORDER_SIZE)
        params = {
            "ticker": ticker,
            "type": "MARKET",
            "quantity": chunk,
            "action": action,
        }
        result = api_post("/orders", params=params)
        if result:
            filled = result.get("quantity_filled", 0)
            vwap = result.get("vwap", 0)
            print(f"  {GREEN}Filled {filled} @ {vwap:.2f}{RESET}")
        else:
            print(f"  {RED}Order failed{RESET}")
            break
        remaining -= chunk
        if remaining > 0:
            time.sleep(0.1)


def cmd_cancel(args):
    """Cancel orders."""
    if not args:
        display_open_orders()
        return
    
    ticker = args[0].upper()
    if cancel_all_orders_for_ticker(ticker):
        print(f"  Cancelled all orders for {ticker}.")
        clear_managed_unwind_state(ticker)


def cmd_cancel_all():
    """Cancel all orders."""
    if cancel_all_orders():
        print("  Cancelled ALL orders.")
        with state_lock:
            active_order_tickers.clear()
            auto_market_flatten_triggered.clear()
            for ticker in list(managed_unwind_tracker.keys()):
                state = managed_unwind_tracker[ticker]
                state["active"] = False
                state["unwind_action"] = None
                state["tender_detected_at"] = None
                state["trend_ref_last"] = None
                state["last_reprice_time"] = 0.0
                state["last_managed_position"] = 0


def cmd_help():
    """Show help."""
    print()
    print(f"  {BOLD}GUI-FIRST LIQUIDITY MONITOR v4{RESET}")
    print(f"  " + "=" * 45)
    print()
    print(f"  {CYAN}WORKFLOW:{RESET}")
    print(f"  1. Tenders appear here with edge calculations")
    print(f"  2. Accept/decline in RIT GUI (no typing needed!)")
    print(f"  3. Script auto-detects fills (>= {TENDER_DETECTION_MIN_DELTA:,}), places/rebalances limit orders")
    print(f"  4. Limit policy: 80% hedge, freeze < {POSITION_FREEZE_THRESHOLD:,}, cancel < {POSITION_SAFETY_THRESHOLD:,}")
    print(f"  5. Auto-flatten: if abs(position) < {AUTO_MARKET_FLATTEN_THRESHOLD:,}, send MARKET opposite")
    print(f"  6. Trend reprice: favorable-only after {TREND_WARMUP_SECONDS:.1f}s (>= {TREND_MIN_FAVORABLE_MOVE:.2f}, every {TREND_REPRICE_MIN_INTERVAL:.1f}s)")
    print()
    print(f"  {BOLD}COMMANDS:{RESET}")
    print(f"  pos          Show positions + entry prices + P&L")
    print(f"  c            Show open orders")
    print(f"  c TICKER     Cancel orders for ticker")
    print(f"  ca           Cancel ALL orders")
    print(f"  flat TICKER  Market flatten position")
    print(f"  bid ID       Submit auction bid (auto-price)")
    print(f"  bid ID 14.50 Submit auction at your price")
    print(f"  status       Connection info")
    print(f"  h            This help")
    print(f"  q            Quit")
    print()


def cmd_status():
    """Show status."""
    data = api_get("/case")
    if data:
        status = data.get("status", "UNKNOWN")
        tick = data.get("tick", "?")
        total = data.get("ticks_per_period", "?")
        name = data.get("name", "Unknown")
        print(f"  Connected: {name}")
        print(f"  Status: {status}  |  Tick: {tick}/{total}")
        print(f"  Poll interval: {POSITION_POLL_INTERVAL:.2f}s")
        print(f"  Tender detection delta: {TENDER_DETECTION_MIN_DELTA:,} shares")
        print(f"  Hedge ratio: {UNWIND_HEDGE_RATIO*100:.0f}%")
        print(f"  Freeze threshold: {POSITION_FREEZE_THRESHOLD:,} shares")
        print(f"  Safety threshold: cancel limits when abs(position) < {POSITION_SAFETY_THRESHOLD:,}")
        print(f"  Auto MKT flatten threshold: {AUTO_MARKET_FLATTEN_THRESHOLD:,} shares")
        print(f"  Trend reprice: favorable-only after {TREND_WARMUP_SECONDS:.1f}s, >= {TREND_MIN_FAVORABLE_MOVE:.2f}, every {TREND_REPRICE_MIN_INTERVAL:.1f}s")
    else:
        print("  Cannot connect to RIT.")


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    global shutdown_flag
    
    print()
    print(f"  {BOLD}==================================================={RESET}")
    print(f"  {BOLD}  RITC 2026 - GUI-First Liquidity Monitor (v4){RESET}")
    print(f"  {BOLD}==================================================={RESET}")
    print()
    print(f"  {CYAN}Accept/decline tenders in RIT GUI - script handles the rest{RESET}")
    print(f"  {YELLOW}Hedge {UNWIND_HEDGE_RATIO*100:.0f}% | freeze < {POSITION_FREEZE_THRESHOLD:,} | cancel < {POSITION_SAFETY_THRESHOLD:,}{RESET}")
    print()
    
    cmd_status()
    print()
    
    # Start background threads
    position_thread = threading.Thread(target=position_monitor_loop, daemon=True)
    position_thread.start()
    
    tender_thread = threading.Thread(target=tender_display_loop, daemon=True)
    tender_thread.start()
    
    print(f"  Type {BOLD}h{RESET} for commands, {BOLD}q{RESET} to quit.")
    print()
    
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Shutting down.")
            shutdown_flag = True
            break
        
        if not raw:
            continue
        
        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd in ("q", "quit"):
            print("  Shutting down.")
            shutdown_flag = True
            break
        elif cmd in ("h", "help"):
            cmd_help()
        elif cmd == "pos":
            display_positions()
        elif cmd == "c":
            cmd_cancel(args)
        elif cmd == "ca":
            cmd_cancel_all()
        elif cmd == "flat":
            cmd_flat(args)
        elif cmd == "bid":
            cmd_bid(args)
        elif cmd == "status":
            cmd_status()
        else:
            print(f"  Unknown: {cmd}. Type h for help.")


if __name__ == "__main__":
    main()
