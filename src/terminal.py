"""
RITC 2026 - Trading Terminal (v2 - Liquidity Case Enhanced)
A command-line trading interface for the RIT Market Simulator.

Changes in v2:
  - Auto limit orders placed on tender accept (2-5¢ from top of book)
  - Red flags on tenders for HIGH VOLATILITY or LOW LIQUIDITY tickers
  - Auto-kill all orders when position unwinds to zero
  - Max 10,000 shares per order enforced

Usage: Run this script while the RIT Client is connected to a server.
       Type h for a list of commands.
"""

import requests
import time
import threading
import math
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# ============================================================
# CONFIGURATION - Change these to match your RIT setup
# ============================================================
DOTENV_PATH = Path(__file__).parent / ".env"
load_dotenv(DOTENV_PATH)
API_KEY = getenv("ROT_API_KEY")
BASE_URL = "http://localhost:9999/v1"

MAX_ORDER_SIZE = 10000  # Max shares per single order

# ============================================================
# ANSI COLOUR CODES - for terminal highlighting
# ============================================================
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ============================================================
# TICKER CHARACTERISTICS - from case package screenshots
# ============================================================
# Used to flag risky tenders. Update if case changes.
# Format: "TICKER": {"volatility": "High"/"Medium"/"Low",
#                     "liquidity": "High"/"Medium"/"Low",
#                     "commission": 0.02, "start_price": 50}
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

def get_ticker_warnings(ticker):
    """Return coloured warning string if ticker is risky."""
    info = TICKER_INFO.get(ticker.upper())
    if not info:
        return ""
    
    warnings = []
    if info["volatility"] == "High":
        warnings.append(f"{RED}{BOLD}⚠ HIGH VOLATILITY{RESET}")
    if info["liquidity"] == "Low":
        warnings.append(f"{RED}{BOLD}⚠ LOW LIQUIDITY{RESET}")
    if info["liquidity"] == "Medium" and info["volatility"] == "High":
        # Extra caution: high vol + only medium liquidity = hard to unwind
        if "HIGH VOLATILITY" not in str(warnings):
            warnings.append(f"{YELLOW}⚠ CAUTION: Med Liq + High Vol{RESET}")
    
    if not warnings:
        # Safe ticker — show green
        return f"{GREEN}✓ Vol:{info['volatility']} Liq:{info['liquidity']}{RESET}"
    
    return "  ".join(warnings) + f"  (Vol:{info['volatility']} Liq:{info['liquidity']})"


# ============================================================
# SESSION SETUP
# ============================================================
s = requests.Session()
s.headers.update({"X-API-Key": API_KEY})

# Global shutdown flag for background threads
shutdown_flag = False

# Global state for repeat command
last_order = None

# Global state for auto-kill monitoring
# When we accept a tender, we track which ticker to monitor
auto_kill_ticker = None
auto_kill_active = False

# ============================================================
# API HELPERS
# ============================================================
def api_get(endpoint, params=None):
    """Send a GET request to the RIT API. Returns parsed JSON or None."""
    try:
        resp = s.get(f"{BASE_URL}{endpoint}", params=params or {})
        if resp.ok:
            return resp.json()
        else:
            print(f"  API error: {resp.status_code} - {resp.text[:100]}")
            return None
    except requests.ConnectionError:
        print("  Cannot connect to RIT. Is the client running?")
        return None

def api_post(endpoint, params=None):
    """Send a POST request to the RIT API. Returns parsed JSON or None."""
    try:
        resp = s.post(f"{BASE_URL}{endpoint}", params=params or {})
        if resp.ok:
            return resp.json()
        else:
            print(f"  API error: {resp.status_code} - {resp.text[:100]}")
            return None
    except requests.ConnectionError:
        print("  Cannot connect to RIT. Is the client running?")
        return None

def api_delete(endpoint, params=None):
    """Send a DELETE request to the RIT API. Returns True if successful."""
    try:
        resp = s.delete(f"{BASE_URL}{endpoint}", params=params or {})
        if resp.ok:
            return True
        else:
            print(f"  API error: {resp.status_code} - {resp.text[:100]}")
            return False
    except requests.ConnectionError:
        print("  Cannot connect to RIT. Is the client running?")
        return False


# ============================================================
# AUTO LIMIT ORDER PLACEMENT
# ============================================================
# After accepting a tender, this places 6 limit orders 2-5¢
# from top of book to start unwinding the position.
# ============================================================
def place_unwind_limits(ticker, position_qty):
    """
    Place limit orders to start unwinding a tender position.
    
    Logic:
      - If we BOUGHT (positive position), place SELL limits near best bid
      - If we SOLD (negative position), place BUY limits near best ask
      - Orders placed 2-5¢ from the top of the book (spread out)
      - 6 orders, each up to 10,000 shares
      - Total limit order volume = min(position, 6 * 10,000)
    """
    if position_qty == 0:
        return
    
    # Determine unwind direction and reference price
    if position_qty > 0:
        # We're long — need to SELL — reference the best bid
        unwind_action = "SELL"
        book = api_get("/securities/book", params={"ticker": ticker, "limit": 1})
        if not book or not book.get("bids"):
            print(f"  {YELLOW}Warning: No bids in book for {ticker}. Skipping auto limits.{RESET}")
            return
        ref_price = book["bids"][0]["price"]
        # Place sell limits ABOVE the best bid (2-5¢ higher, toward the ask)
        # This means we're offering to sell at competitive prices near top of book
        offsets = [0.02, 0.02, 0.03, 0.03, 0.04, 0.05]
    else:
        # We're short — need to BUY — reference the best ask
        unwind_action = "BUY"
        book = api_get("/securities/book", params={"ticker": ticker, "limit": 1})
        if not book or not book.get("asks"):
            print(f"  {YELLOW}Warning: No asks in book for {ticker}. Skipping auto limits.{RESET}")
            return
        ref_price = book["asks"][0]["price"]
        # Place buy limits BELOW the best ask (2-5¢ lower, toward the bid)
        offsets = [-0.02, -0.02, -0.03, -0.03, -0.04, -0.05]
    
    abs_qty = abs(position_qty)
    num_orders = 6
    
    # Spread quantity across orders, each capped at MAX_ORDER_SIZE
    per_order_qty = min(abs_qty // num_orders, MAX_ORDER_SIZE)
    if per_order_qty == 0:
        per_order_qty = min(abs_qty, MAX_ORDER_SIZE)
        num_orders = 1
    
    # Handle remainder: add to first orders
    remainder = abs_qty - (per_order_qty * num_orders)
    
    print(f"\n  {CYAN}>> AUTO LIMIT ORDERS: Placing {num_orders} x {unwind_action} limits for {ticker}{RESET}")
    print(f"  {CYAN}   Reference price (top of book): {ref_price:.2f}{RESET}")
    
    placed_count = 0
    total_placed_qty = 0
    
    for i in range(num_orders):
        if total_placed_qty >= abs_qty:
            break
            
        # Calculate this order's quantity
        this_qty = per_order_qty
        if i == 0 and remainder > 0:
            this_qty = min(per_order_qty + remainder, MAX_ORDER_SIZE)
        this_qty = min(this_qty, abs_qty - total_placed_qty, MAX_ORDER_SIZE)
        
        if this_qty <= 0:
            break
        
        # Calculate price with offset
        price = round(ref_price + offsets[i], 2)
        
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
                print(f"  {GREEN}   #{oid}: {unwind_action} {this_qty} @ {price:.2f} — {filled} already filled!{RESET}")
            else:
                print(f"     #{oid}: {unwind_action} {this_qty} @ {price:.2f}")
        else:
            print(f"  {RED}   Failed to place order at {price:.2f}{RESET}")
        
        time.sleep(0.1)  # Small delay between orders to avoid rate limit
    
    print(f"  {CYAN}   Placed {placed_count} limit orders, total qty: {total_placed_qty}/{abs_qty}{RESET}")
    if total_placed_qty < abs_qty:
        remaining = abs_qty - total_placed_qty
        print(f"  {YELLOW}   Remaining {remaining} shares — use manual orders (b/s) or market orders to unwind{RESET}")
    print("> ", end="", flush=True)


# ============================================================
# AUTO-KILL MONITOR
# ============================================================
# Background thread that watches position after tender accept.
# When position returns to 0, kills all open orders for safety.
# ============================================================
def auto_kill_monitor():
    """Monitor position and kill all orders when flat."""
    global auto_kill_active, auto_kill_ticker
    
    ticker = auto_kill_ticker
    if not ticker:
        return
    
    # Give a moment for the tender to process
    time.sleep(1)
    
    # First, confirm we actually have a position
    data = api_get("/securities", params={"ticker": ticker})
    if not data:
        auto_kill_active = False
        return
    
    sec = data[0] if isinstance(data, list) else data
    initial_pos = sec.get("position", 0)
    
    if initial_pos == 0:
        auto_kill_active = False
        return
    
    print(f"\n  {CYAN}>> AUTO-KILL ARMED: Watching {ticker} (pos: {initial_pos}). Will kill orders when flat.{RESET}")
    print("> ", end="", flush=True)
    
    # Poll every 0.5 seconds
    while auto_kill_active and not shutdown_flag:
        try:
            data = api_get("/securities", params={"ticker": ticker})
            if data:
                sec = data[0] if isinstance(data, list) else data
                pos = sec.get("position", 0)
                
                if pos == 0:
                    # FLAT! Kill all orders immediately
                    api_post("/commands/cancel", params={"query": f"Ticker='{ticker}'"})
                    print(f"\n  {GREEN}{BOLD}>> FLAT in {ticker}! All orders KILLED automatically.{RESET}")
                    print("> ", end="", flush=True)
                    auto_kill_active = False
                    return
        except Exception:
            pass
        
        time.sleep(0.5)


# ============================================================
# COMMAND: buy / sell orders
# ============================================================
def cmd_order(action, args):
    """Place a buy or sell order."""
    global last_order
    if len(args) < 2:
        print(f"  Usage: {'b' if action == 'BUY' else 's'} TICKER QTY [PRICE]")
        print(f"  Example: {'b' if action == 'BUY' else 's'} RTM 100 25.50")
        return

    ticker = args[0].upper()
    try:
        qty = int(args[1])
    except ValueError:
        print(f"  Bad quantity: {args[1]}. Must be a whole number.")
        return

    if len(args) >= 3:
        try:
            price = round(float(args[2]), 2)
        except ValueError:
            print(f"  Bad price: {args[2]}.")
            return
        order_type = "LIMIT"
    else:
        price = None
        order_type = "MARKET"

    last_order = (action, args)

    # Split into chunks of MAX_ORDER_SIZE if needed
    remaining = qty
    while remaining > 0:
        chunk = min(remaining, MAX_ORDER_SIZE)
        params = {
            "ticker": ticker,
            "type": order_type,
            "quantity": chunk,
            "action": action,
        }
        if price is not None:
            params["price"] = price

        result = api_post("/orders", params=params)
        if result:
            oid = result.get("order_id", "?")
            filled = result.get("quantity_filled", 0)
            side = "BUY" if action == "BUY" else "SELL"
            price_str = f"@ {price}" if price else "@ MKT"

            if filled == chunk:
                vwap = result.get("vwap", price or 0)
                print(f"  FILLED: {side} {chunk} {ticker} {price_str}  (VWAP: {vwap:.2f})")
            elif filled > 0:
                vwap = result.get("vwap", price or 0)
                print(f"  PARTIAL: {side} {filled}/{chunk} {ticker} {price_str}  (VWAP: {vwap:.2f}, order #{oid} open)")
            else:
                print(f"  OPEN: {side} {chunk} {ticker} {price_str}  (order #{oid})")
        else:
            break  # Stop if an order fails

        remaining -= chunk
        if remaining > 0:
            time.sleep(0.1)  # Brief pause between chunks

# ============================================================
# COMMAND: positions
# ============================================================
def cmd_positions():
    """Show all current positions with VWAP and unrealised P&L."""
    data = api_get("/securities")
    if not data:
        return

    held = [sec for sec in data if sec.get("position", 0) != 0]

    if not held:
        print("  No open positions.")
        return

    print(f"  {'Ticker':<12} {'Pos':>8} {'VWAP':>10} {'Last':>10} {'Unrl P&L':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")

    total_unrl = 0
    for sec in held:
        ticker = sec.get("ticker", "?")
        position = sec.get("position", 0)
        vwap = sec.get("vwap", 0)
        last = sec.get("last", 0)
        unrealised = sec.get("unrealized", 0)
        total_unrl += unrealised

        print(f"  {ticker:<12} {position:>8} {vwap:>10.2f} {last:>10.2f} {unrealised:>12.2f}")

    trader = api_get("/trader")
    realised = 0
    if trader:
        realised = trader.get("realized", 0) if isinstance(trader, dict) else 0

    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL':<12} {'':>8} {'':>10} {'':>10} {total_unrl:>12.2f}")
    print(f"  Realised P&L: {realised:.2f}   |   Combined: {realised + total_unrl:.2f}")

# ============================================================
# COMMAND: cancel orders
# ============================================================
def cmd_cancel(args):
    """Cancel open orders."""
    if not args:
        orders = api_get("/orders", params={"status": "OPEN"})
        if not orders:
            print("  No open orders.")
            return
        if len(orders) == 0:
            print("  No open orders.")
            return

        print(f"  {'ID':<8} {'Side':<6} {'Ticker':<12} {'Qty':>6} {'Filled':>6} {'Price':>10}")
        print(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*6} {'-'*6} {'-'*10}")
        for o in orders:
            oid = o.get("order_id", "?")
            side = o.get("action", "?")
            ticker = o.get("ticker", "?")
            qty = o.get("quantity", 0)
            filled = o.get("quantity_filled", 0)
            price = o.get("price", 0)
            print(f"  {oid:<8} {side:<6} {ticker:<12} {qty:>6} {filled:>6} {price:>10.2f}")
        return

    ticker = args[0].upper()
    result = api_post("/commands/cancel", params={"query": f"Ticker='{ticker}'"})
    if result is not None:
        print(f"  Cancelled all open orders for {ticker}.")

def cmd_cancel_all():
    """Cancel every open order across all tickers."""
    result = api_post("/commands/cancel", params={"query": ""})
    if result is not None:
        print("  Cancelled ALL open orders.")

def cmd_repeat():
    """Repeat the last buy or sell order."""
    global last_order
    if last_order is None:
        print("  No previous order to repeat.")
        return
    action, args = last_order
    cmd_order(action, args)

# ============================================================
# COMMAND: flatten
# ============================================================
def cmd_flatten(args):
    """Flatten position in a ticker by sending market orders to get to zero."""
    if not args:
        print("  Usage: flat TICKER")
        return

    ticker = args[0].upper()

    data = api_get("/securities", params={"ticker": ticker})
    if not data:
        return

    sec = data[0] if isinstance(data, list) and len(data) > 0 else data
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

    # Split into chunks of MAX_ORDER_SIZE
    remaining = qty
    total_filled = 0
    print(f"  Flattening: {action} {qty} {ticker} @ MKT...")

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
            total_filled += filled
            vwap = result.get("vwap", 0)
            if filled < chunk:
                print(f"  PARTIAL: {filled}/{chunk} filled @ VWAP {vwap:.2f}")
                break
        else:
            break
        remaining -= chunk
        if remaining > 0:
            time.sleep(0.1)

    if total_filled == qty:
        print(f"  {GREEN}FLAT: {action} {qty} {ticker} complete.{RESET}")
    else:
        print(f"  Filled {total_filled}/{qty}. Remaining position still open.")


# ============================================================
# VOLATILITY CASE CONFIGURATION
# ============================================================
VOL_UNDERLYING = "RTM"
VOL_CALL_PREFIX = "RTM1C"
VOL_PUT_PREFIX = "RTM1P"
VOL_RATE = 0.0
VOL_DAYS_PER_PERIOD = 30
VOL_IMPLIED_VOL = 0.20

# ============================================================
# BLACK-SCHOLES HELPERS
# ============================================================
def norm_cdf(x):
    """Standard normal cumulative distribution (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_greeks(S, K, T, r, sigma, option_type):
    """Calculate Black-Scholes Greeks for a single option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    nd1_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)

    gamma = nd1_pdf / (S * sigma * math.sqrt(T))
    vega = S * nd1_pdf * math.sqrt(T) / 100

    if option_type == "C":
        delta = norm_cdf(d1)
        theta = (-(S * nd1_pdf * sigma) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm_cdf(d2)) / 365
    else:
        delta = norm_cdf(d1) - 1
        theta = (-(S * nd1_pdf * sigma) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm_cdf(-d2)) / 365

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

def parse_option_ticker(ticker):
    """Parse an option ticker to extract strike and type."""
    if ticker.startswith(VOL_CALL_PREFIX):
        try:
            strike = float(ticker[len(VOL_CALL_PREFIX):])
            return strike, "C"
        except ValueError:
            return None, None
    elif ticker.startswith(VOL_PUT_PREFIX):
        try:
            strike = float(ticker[len(VOL_PUT_PREFIX):])
            return strike, "P"
        except ValueError:
            return None, None
    return None, None

# ============================================================
# COMMAND: risk — portfolio Greeks for the vol case
# ============================================================
def cmd_risk(args):
    """Show portfolio Greeks across all option positions."""
    global VOL_IMPLIED_VOL

    if args:
        try:
            VOL_IMPLIED_VOL = float(args[0])
            print(f"  IV updated to {VOL_IMPLIED_VOL:.2%}")
        except ValueError:
            print(f"  Bad IV value: {args[0]}. Use decimal like 0.25")
            return

    data = api_get("/securities")
    if not data:
        return

    case_data = api_get("/case")
    if not case_data:
        return

    tick = case_data.get("tick", 0)
    total_ticks = case_data.get("ticks_per_period", 300)
    ticks_remaining = max(total_ticks - tick, 1)
    T = (ticks_remaining / total_ticks) * (VOL_DAYS_PER_PERIOD / 365.0)

    underlying_price = 0
    for sec in data:
        if sec.get("ticker") == VOL_UNDERLYING:
            underlying_price = sec.get("last", 0)
            break

    if underlying_price <= 0:
        print(f"  Cannot find underlying {VOL_UNDERLYING} or price is 0.")
        return

    total_delta = 0
    total_gamma = 0
    total_vega = 0
    total_theta = 0

    print(f"\n  S={underlying_price:.2f}  IV={VOL_IMPLIED_VOL:.2%}  T={T:.4f}yr  Tick={tick}/{total_ticks}")
    print(f"  {'Ticker':<12} {'Pos':>6} {'Strike':>8} {'Type':>5} {'Delta':>8} {'Gamma':>8} {'Vega':>8} {'Theta':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for sec in data:
        ticker = sec.get("ticker", "")
        position = sec.get("position", 0)
        if position == 0:
            continue
        if ticker == VOL_UNDERLYING:
            total_delta += position
            print(f"  {ticker:<12} {position:>6} {'':>8} {'STK':>5} {position:>8.1f} {'':>8} {'':>8} {'':>8}")
            continue
        strike, opt_type = parse_option_ticker(ticker)
        if strike is None:
            continue
        greeks = bs_greeks(underlying_price, strike, T, VOL_RATE, VOL_IMPLIED_VOL, opt_type)
        pos_delta = greeks["delta"] * position
        pos_gamma = greeks["gamma"] * position
        pos_vega = greeks["vega"] * position
        pos_theta = greeks["theta"] * position
        total_delta += pos_delta
        total_gamma += pos_gamma
        total_vega += pos_vega
        total_theta += pos_theta
        print(f"  {ticker:<12} {position:>6} {strike:>8.1f} {'C' if opt_type == 'C' else 'P':>5} {pos_delta:>8.1f} {pos_gamma:>8.2f} {pos_vega:>8.2f} {pos_theta:>8.2f}")

    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<12} {'':>6} {'':>8} {'':>5} {total_delta:>8.1f} {total_gamma:>8.2f} {total_vega:>8.2f} {total_theta:>8.2f}")
    print()

# ============================================================
# COMMAND: status
# ============================================================
def cmd_status():
    """Show current case info to verify we are connected."""
    data = api_get("/case")
    if data:
        status = data.get("status", "UNKNOWN")
        tick = data.get("tick", "?")
        total = data.get("ticks_per_period", "?")
        period = data.get("period", "?")
        name = data.get("name", "Unknown Case")
        print(f"  Connected to: {name}")
        print(f"    Status: {status}  |  Tick: {tick}/{total}  |  Period: {period}")

# ============================================================
# COMMAND: help
# ============================================================
def cmd_help():
    """Print all available commands."""
    print()
    print("  RITC TRADING TERMINAL v2 - COMMANDS")
    print("  " + "=" * 50)
    print("  ORDERS:")
    print("  b TICKER QTY PRICE  Limit buy  (auto-splits >10k)")
    print("  b TICKER QTY        Market buy")
    print("  s TICKER QTY PRICE  Limit sell (auto-splits >10k)")
    print("  s TICKER QTY        Market sell")
    print("  r                   Repeat last order")
    print()
    print("  POSITIONS & RISK:")
    print("  pos                  All positions + VWAP + P&L")
    print("  flat TICKER          Market order to flatten (splits >10k)")
    print("  risk                 Portfolio Greeks (vol case)")
    print("  risk 0.25            Recalc with new IV")
    print()
    print("  ORDER MANAGEMENT:")
    print("  c                    Show open orders")
    print("  c TICKER             Cancel orders for ticker")
    print("  ca                   Cancel ALL open orders")
    print()
    print("  TENDERS (Liquidity Case):")
    print("  tenders              Show active tenders")
    print(f"  accept ID            Accept + {CYAN}AUTO limit orders + auto-kill{RESET}")
    print("  decline ID           Decline tender offer")
    print()
    print(f"  {CYAN}NEW: On accept, 6 limit orders placed 2-5¢ from book.{RESET}")
    print(f"  {CYAN}     Orders auto-killed when position returns to 0.{RESET}")
    print(f"  {RED}     Tenders flagged for HIGH VOL / LOW LIQ tickers.{RESET}")
    print()
    print("  INFO:")
    print("  status               Connection & case info")
    print()
    print("  OTHER:")
    print("  h                    Show this help")
    print("  q                    Quit")
    print()


# ============================================================
# COMMAND: tender accept / decline
# ============================================================
def cmd_tender_accept(args):
    """Accept a tender offer, then auto-place limit orders and arm auto-kill."""
    global auto_kill_ticker, auto_kill_active

    if not args:
        print("  Usage: accept TENDER_ID")
        return

    tender_id = args[0]

    # First, get tender details so we know what we're accepting
    tenders = api_get("/tenders")
    tender_info = None
    if tenders:
        for t in tenders:
            if str(t.get("tender_id")) == str(tender_id):
                tender_info = t
                break

    # Accept the tender
    result = api_post(f"/tenders/{tender_id}", params={"price": 0})
    if result is not None:
        print(f"  {GREEN}{BOLD}Tender #{tender_id} ACCEPTED.{RESET}")

        if tender_info:
            ticker = tender_info.get("ticker", "")
            action = tender_info.get("action", "")
            qty = tender_info.get("quantity", 0)

            # Determine position direction from the tender
            # If tender action is BUY, we bought → position is positive
            # If tender action is SELL, we sold → position is negative
            if action == "BUY":
                position_qty = qty
            else:
                position_qty = -qty

            # Place auto limit orders to start unwinding
            time.sleep(0.3)  # Brief pause for tender to settle
            place_unwind_limits(ticker, position_qty)

            # Arm auto-kill monitor
            auto_kill_ticker = ticker
            auto_kill_active = True
            kill_thread = threading.Thread(target=auto_kill_monitor, daemon=True)
            kill_thread.start()
        else:
            print(f"  {YELLOW}Could not get tender details — no auto limits placed.{RESET}")
            print(f"  Trade manually or use: b/s TICKER QTY PRICE")

def cmd_tender_decline(args):
    """Decline a tender offer by its ID."""
    if not args:
        print("  Usage: decline TENDER_ID")
        return
    tender_id = args[0]
    result = api_delete(f"/tenders/{tender_id}")
    if result:
        print(f"  Tender #{tender_id} declined.")

def cmd_tenders():
    """Show all active tender offers with risk flags."""
    data = api_get("/tenders")
    if not data or len(data) == 0:
        print("  No active tenders.")
        return
    for t in data:
        tid = t.get("tender_id", "?")
        ticker = t.get("ticker", "?")
        action = t.get("action", "?")
        qty = t.get("quantity", 0)
        price = t.get("price", 0)
        expires = t.get("expires", "?")
        warnings = get_ticker_warnings(ticker)
        print(f"  TENDER #{tid}: {action} {qty} {ticker} @ {price:.2f}  (expires tick {expires})")
        if warnings:
            print(f"     {warnings}")


# ============================================================
# BACKGROUND: News feed
# ============================================================
def news_feed_loop():
    """Background thread: poll for news and print new items."""
    seen_ids = set()

    while not shutdown_flag:
        try:
            resp = s.get(f"{BASE_URL}/news", params={"limit": 10})
            if resp.ok:
                items = resp.json()
                for item in reversed(items):
                    nid = item.get("news_id")
                    if nid and nid not in seen_ids:
                        seen_ids.add(nid)
                        tick = item.get("tick", "?")
                        headline = item.get("headline", "")
                        body = item.get("body", "")
                        text = headline if headline else body[:120]
                        print(f"\n  ** NEWS [Tick {tick}]: {text}")
                        print("> ", end="", flush=True)
        except Exception:
            pass

        time.sleep(2)


# ============================================================
# BACKGROUND: Tender monitor (enhanced with risk flags)
# ============================================================
def tender_monitor_loop():
    """Background thread: poll for tenders and alert with risk flags."""
    seen_ids = set()

    while not shutdown_flag:
        try:
            resp = s.get(f"{BASE_URL}/tenders")
            if resp.ok:
                tenders = resp.json()
                for t in tenders:
                    tid = t.get("tender_id")
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        ticker = t.get("ticker", "?")
                        action = t.get("action", "?")
                        qty = t.get("quantity", 0)
                        price = t.get("price", 0)
                        expires = t.get("expires", "?")

                        # Get current market price for edge calculation
                        sec_data = api_get("/securities", params={"ticker": ticker})
                        mkt_price = 0
                        if sec_data:
                            sec = sec_data[0] if isinstance(sec_data, list) else sec_data
                            if action == "BUY":
                                mkt_price = sec.get("ask", sec.get("last", 0))
                            else:
                                mkt_price = sec.get("bid", sec.get("last", 0))

                        if action == "BUY" and mkt_price > 0:
                            edge = mkt_price - price
                            edge_str = f"+{edge:.2f} edge" if edge > 0 else f"{edge:.2f} edge"
                        elif action == "SELL" and mkt_price > 0:
                            edge = price - mkt_price
                            edge_str = f"+{edge:.2f} edge" if edge > 0 else f"{edge:.2f} edge"
                        else:
                            edge_str = "mkt unknown"

                        # Get ticker risk warnings
                        warnings = get_ticker_warnings(ticker)

                        print(f"\n  !! TENDER #{tid}: {action} {qty} {ticker} @ {price:.2f}  ({edge_str}, expires tick {expires})")
                        if warnings:
                            print(f"     {warnings}")
                        print(f"     -> Type: accept {tid}  or  decline {tid}")
                        print("> ", end="", flush=True)
        except Exception:
            pass

        time.sleep(1)


# ============================================================
# MAIN COMMAND LOOP
# ============================================================
def main():
    global shutdown_flag

    print()
    print(f"  {BOLD}RITC Trading Terminal v2 — Liquidity Case Enhanced{RESET}")
    print(f"  Type h for help, q to quit.")
    print()

    cmd_status()
    print()

    # Start background threads
    news_thread = threading.Thread(target=news_feed_loop, daemon=True)
    news_thread.start()

    tender_thread = threading.Thread(target=tender_monitor_loop, daemon=True)
    tender_thread.start()

    print(f"  Background monitors: news + tenders active.")
    print(f"  {CYAN}Auto limit orders + auto-kill on tender accept: ENABLED{RESET}")
    print()

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("  Shutting down.")
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
        elif cmd == "status":
            cmd_status()
        elif cmd == "b":
            cmd_order("BUY", args)
        elif cmd == "s":
            cmd_order("SELL", args)
        elif cmd == "r":
            cmd_repeat()
        elif cmd == "pos":
            cmd_positions()
        elif cmd == "c":
            cmd_cancel(args)
        elif cmd == "ca":
            cmd_cancel_all()
        elif cmd == "flat":
            cmd_flatten(args)
        elif cmd == "risk":
            cmd_risk(args)
        elif cmd == "accept":
            cmd_tender_accept(args)
        elif cmd == "decline":
            cmd_tender_decline(args)
        elif cmd == "tenders":
            cmd_tenders()
        else:
            print(f"  Unknown command: {cmd}. Type h for help.")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
