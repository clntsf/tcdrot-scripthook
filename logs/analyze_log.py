#!/usr/bin/env python3
"""Deep forensic analysis of trading log."""
import csv
import sys
from collections import defaultdict

LOG_FILE = "/Users/colin/Desktop/TCD/tcdrot-scripthook/logs/20260217_231457.txt"

TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]
TICKER_IDX = {t: i for i, t in enumerate(TICKERS)}

def parse_ticker(field):
    """Parse 'pos,bid,ask,our_bid,our_ask,qty' into dict."""
    parts = field.split(",")
    return {
        "pos": float(parts[0]),
        "bid": float(parts[1]),
        "ask": float(parts[2]),
        "our_bid": float(parts[3]),
        "our_ask": float(parts[4]),
        "qty": float(parts[5]),
        "mid": (float(parts[1]) + float(parts[2])) / 2
    }

def parse_line(line):
    """Parse a single log line."""
    parts = line.strip().split("|")
    if len(parts) < 13:
        return None
    row = {
        "tick": int(parts[0]),
        "t_min": int(parts[1]),
        "agg_exp": float(parts[2]),
        "gross": float(parts[3]),
        "net": float(parts[4]),
        "gross_lim": float(parts[5]),
        "net_lim": float(parts[6]),
        "nlv": float(parts[7]),
        "actions": parts[12] if len(parts) > 12 else ""
    }
    for i, t in enumerate(TICKERS):
        # strip "TICKER:" prefix if present
        field = parts[8 + i]
        if ":" in field and not field[0].isdigit() and not field[0] == "-":
            field = field.split(":", 1)[1]
        row[t] = parse_ticker(field)
    return row

def load_data():
    rows = []
    with open(LOG_FILE) as f:
        for i, line in enumerate(f):
            if i == 0:  # header
                continue
            line = line.strip()
            if not line:
                continue
            r = parse_line(line)
            if r:
                rows.append(r)
    return rows

def dedupe_ticks(rows):
    """Keep only first entry per unique tick (some ticks have multiple log entries)."""
    seen = set()
    deduped = []
    for r in rows:
        if r["tick"] not in seen:
            seen.add(r["tick"])
            deduped.append(r)
    return deduped

def analysis_1_nlv_trajectory(rows):
    print("=" * 80)
    print("1. NLV TRAJECTORY ANALYSIS")
    print("=" * 80)

    # Get unique ticks with first NLV
    deduped = dedupe_ticks(rows)

    print(f"\nTotal unique ticks: {len(deduped)}")
    print(f"First NLV: {deduped[0]['nlv']:.2f} (tick {deduped[0]['tick']})")
    print(f"Final NLV: {deduped[-1]['nlv']:.2f} (tick {deduped[-1]['tick']})")
    print(f"Max NLV: {max(r['nlv'] for r in deduped):.2f}")
    print(f"Min NLV: {min(r['nlv'] for r in deduped):.2f}")

    # NLV changes between consecutive unique ticks
    drops = []
    for i in range(1, len(deduped)):
        delta = deduped[i]["nlv"] - deduped[i-1]["nlv"]
        drops.append((delta, deduped[i]["tick"], deduped[i]["t_min"], deduped[i]))

    # Correlate with t_min
    t_min_nlv_changes = defaultdict(list)
    for delta, tick, t_min, r in drops:
        t_min_nlv_changes[t_min].append(delta)

    print("\n--- Average NLV change by t_min bucket ---")
    buckets = {
        "News (0-5)": range(0, 6),
        "Early (6-15)": range(6, 16),
        "Mid (16-35)": range(16, 36),
        "Late (36-49)": range(36, 50),
        "Unwind (50-59)": range(50, 60)
    }
    for name, rng in buckets.items():
        changes = []
        for t in rng:
            changes.extend(t_min_nlv_changes.get(t, []))
        if changes:
            avg = sum(changes) / len(changes)
            total = sum(changes)
            print(f"  {name}: avg={avg:+.2f}, total={total:+.2f}, n={len(changes)}")

    # Top 10 drops
    drops.sort(key=lambda x: x[0])
    print("\n--- 10 largest single-tick NLV drops ---")
    for delta, tick, t_min, r in drops[:10]:
        print(f"  tick={tick}, t_min={t_min}, NLV change={delta:+.2f}, NLV={r['nlv']:.2f}")

    print("\n--- 10 largest single-tick NLV gains ---")
    drops.sort(key=lambda x: -x[0])
    for delta, tick, t_min, r in drops[:10]:
        print(f"  tick={tick}, t_min={t_min}, NLV change={delta:+.2f}, NLV={r['nlv']:.2f}")

    return deduped

def analysis_2_adverse_selection(rows):
    print("\n" + "=" * 80)
    print("2. ADVERSE SELECTION ANALYSIS")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    for ticker in TICKERS:
        fills = []
        for i in range(1, len(deduped)):
            prev_pos = deduped[i-1][ticker]["pos"]
            curr_pos = deduped[i][ticker]["pos"]
            delta_pos = curr_pos - prev_pos
            if abs(delta_pos) > 0.1:
                fills.append({
                    "idx": i,
                    "tick": deduped[i]["tick"],
                    "t_min": deduped[i]["t_min"],
                    "delta_pos": delta_pos,
                    "fill_mid": deduped[i][ticker]["mid"],
                    "prev_mid": deduped[i-1][ticker]["mid"],
                    "side": "BUY" if delta_pos > 0 else "SELL",
                    "size": abs(delta_pos)
                })

        if not fills:
            print(f"\n  {ticker}: No fills detected")
            continue

        # Calculate subsequent price moves (3-5 ticks after fill)
        buy_moves = []
        sell_moves = []
        for f in fills:
            idx = f["idx"]
            # Look 3-5 unique ticks ahead
            for lookahead in [3, 5]:
                future_idx = min(idx + lookahead, len(deduped) - 1)
                future_mid = deduped[future_idx][ticker]["mid"]
                price_move = future_mid - f["fill_mid"]

                if f["side"] == "BUY":
                    # If we bought, adverse = price goes DOWN
                    if lookahead == 5:
                        buy_moves.append((price_move, f["size"], f["tick"]))
                elif f["side"] == "SELL":
                    # If we sold, adverse = price goes UP
                    if lookahead == 5:
                        sell_moves.append((price_move, f["size"], f["tick"]))

        print(f"\n  {ticker}:")
        print(f"    Total fills: {len(fills)}")
        print(f"    BUY fills: {len([f for f in fills if f['side']=='BUY'])}, total size: {sum(f['size'] for f in fills if f['side']=='BUY'):.0f}")
        print(f"    SELL fills: {len([f for f in fills if f['side']=='SELL'])}, total size: {sum(f['size'] for f in fills if f['side']=='SELL'):.0f}")

        if buy_moves:
            avg_move = sum(m[0] for m in buy_moves) / len(buy_moves)
            adverse_pct = sum(1 for m in buy_moves if m[0] < 0) / len(buy_moves) * 100
            size_weighted = sum(m[0]*m[1] for m in buy_moves) / sum(m[1] for m in buy_moves)
            print(f"    After BUY: avg 5-tick move={avg_move:+.4f}, size-weighted={size_weighted:+.4f}, adverse%={adverse_pct:.1f}%")

        if sell_moves:
            avg_move = sum(m[0] for m in sell_moves) / len(sell_moves)
            adverse_pct = sum(1 for m in sell_moves if m[0] > 0) / len(sell_moves) * 100
            size_weighted = sum(m[0]*m[1] for m in sell_moves) / sum(m[1] for m in sell_moves)
            print(f"    After SELL: avg 5-tick move={avg_move:+.4f}, size-weighted={size_weighted:+.4f}, adverse%={adverse_pct:.1f}%")

def analysis_3_unwind_cost(rows):
    print("\n" + "=" * 80)
    print("3. UNWIND COST ANALYSIS")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    # Group by minute (tick // 60 essentially, but use actual tick values)
    # Identify minute boundaries by looking at when t_min resets to 0
    minutes = []
    current_minute_start = 0
    for i in range(1, len(deduped)):
        if deduped[i]["t_min"] < deduped[i-1]["t_min"]:
            minutes.append((current_minute_start, i-1))
            current_minute_start = i
    minutes.append((current_minute_start, len(deduped)-1))

    print(f"\nTotal trading minutes identified: {len(minutes)}")

    for mi, (start_idx, end_idx) in enumerate(minutes):
        # Find NLV at t_min=49 and t_min=50
        nlv_at_49 = None
        nlv_at_50 = None
        nlv_at_start = deduped[start_idx]["nlv"]
        nlv_at_end = deduped[end_idx]["nlv"]

        for i in range(start_idx, end_idx + 1):
            if deduped[i]["t_min"] == 49:
                nlv_at_49 = deduped[i]["nlv"]
            if deduped[i]["t_min"] == 50:
                if nlv_at_50 is None:
                    nlv_at_50 = deduped[i]["nlv"]

        unwind_actions = []
        for i in range(start_idx, end_idx + 1):
            if deduped[i]["t_min"] >= 50 and "unwind" in deduped[i].get("actions", ""):
                unwind_actions.append(deduped[i])

        print(f"\n  Minute {mi+1} (ticks {deduped[start_idx]['tick']}-{deduped[end_idx]['tick']}):")
        print(f"    NLV start (t_min=0 area): {nlv_at_start:.2f}")
        if nlv_at_49 is not None:
            print(f"    NLV at t_min=49: {nlv_at_49:.2f}")
        if nlv_at_50 is not None:
            print(f"    NLV at t_min=50: {nlv_at_50:.2f}")
        print(f"    NLV end: {nlv_at_end:.2f}")
        if nlv_at_49 is not None and nlv_at_50 is not None:
            print(f"    Unwind cost (NLV@49 - NLV@end): {nlv_at_49 - nlv_at_end:+.2f}")
        if nlv_at_49 is not None:
            print(f"    Trading PnL (NLV@49 - NLV@start): {nlv_at_49 - nlv_at_start:+.2f}")
        print(f"    Unwind actions: {len(unwind_actions)}")

def analysis_4_per_minute_pnl(rows):
    print("\n" + "=" * 80)
    print("4. PER-MINUTE PNL")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    minutes = []
    current_minute_start = 0
    for i in range(1, len(deduped)):
        if deduped[i]["t_min"] < deduped[i-1]["t_min"]:
            minutes.append((current_minute_start, i-1))
            current_minute_start = i
    minutes.append((current_minute_start, len(deduped)-1))

    total_profit = 0
    total_loss = 0
    for mi, (start_idx, end_idx) in enumerate(minutes):
        s = deduped[start_idx]
        e = deduped[end_idx]
        pnl = e["nlv"] - s["nlv"]
        status = "PROFIT" if pnl >= 0 else "LOSS"
        if pnl >= 0:
            total_profit += pnl
        else:
            total_loss += pnl
        print(f"  Minute {mi+1}: ticks {s['tick']}-{e['tick']}, NLV {s['nlv']:.2f} -> {e['nlv']:.2f}, PnL={pnl:+.2f} [{status}]")

    print(f"\n  Total from profitable minutes: {total_profit:+.2f}")
    print(f"  Total from losing minutes: {total_loss:+.2f}")
    print(f"  Net: {total_profit + total_loss:+.2f}")

def analysis_5_order_size_vs_adverse(rows):
    print("\n" + "=" * 80)
    print("5. ORDER SIZE vs ADVERSE SELECTION")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    large_size_fills = []  # qty > 3000
    small_size_fills = []  # qty < 1000
    mid_size_fills = []    # 1000-3000

    for i in range(1, len(deduped)):
        for ticker in TICKERS:
            prev_pos = deduped[i-1][ticker]["pos"]
            curr_pos = deduped[i][ticker]["pos"]
            delta = curr_pos - prev_pos
            if abs(delta) < 0.1:
                continue

            qty = deduped[i-1][ticker]["qty"]  # quoted qty before fill
            fill_mid = deduped[i][ticker]["mid"]

            # 5-tick lookahead
            future_idx = min(i + 5, len(deduped) - 1)
            future_mid = deduped[future_idx][ticker]["mid"]
            price_move = future_mid - fill_mid

            # Adverse selection: bought and price went down, or sold and price went up
            if delta > 0:
                adverse = price_move < 0
                adverse_amt = -price_move * abs(delta)
            else:
                adverse = price_move > 0
                adverse_amt = price_move * abs(delta)

            entry = {
                "ticker": ticker,
                "qty": qty,
                "delta": delta,
                "adverse": adverse,
                "adverse_amt": adverse_amt,
                "price_move": price_move
            }

            if qty > 3000:
                large_size_fills.append(entry)
            elif qty < 1000:
                small_size_fills.append(entry)
            else:
                mid_size_fills.append(entry)

    for label, fills in [("Large (>3000)", large_size_fills), ("Mid (1000-3000)", mid_size_fills), ("Small (<1000)", small_size_fills)]:
        if not fills:
            print(f"\n  {label}: No fills")
            continue
        adverse_count = sum(1 for f in fills if f["adverse"])
        adverse_pct = adverse_count / len(fills) * 100
        total_adverse_cost = sum(f["adverse_amt"] for f in fills if f["adverse"])
        avg_adverse_cost = total_adverse_cost / adverse_count if adverse_count > 0 else 0
        print(f"\n  {label}:")
        print(f"    Fills: {len(fills)}")
        print(f"    Adversely selected: {adverse_count}/{len(fills)} ({adverse_pct:.1f}%)")
        print(f"    Total adverse cost: {total_adverse_cost:.2f}")
        print(f"    Avg adverse cost per fill: {avg_adverse_cost:.2f}")
        # Break down by ticker
        for ticker in TICKERS:
            tf = [f for f in fills if f["ticker"] == ticker]
            if tf:
                adv = sum(1 for f in tf if f["adverse"])
                print(f"      {ticker}: {len(tf)} fills, {adv} adverse ({adv/len(tf)*100:.0f}%)")

def analysis_6_spread_capture(rows):
    print("\n" + "=" * 80)
    print("6. SPREAD CAPTURE ANALYSIS")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    for ticker in TICKERS:
        buy_fills = []
        sell_fills = []

        for i in range(1, len(deduped)):
            prev_pos = deduped[i-1][ticker]["pos"]
            curr_pos = deduped[i][ticker]["pos"]
            delta = curr_pos - prev_pos
            if abs(delta) < 0.1:
                continue

            our_bid = deduped[i-1][ticker]["our_bid"]
            our_ask = deduped[i-1][ticker]["our_ask"]
            mkt_bid = deduped[i][ticker]["bid"]
            mkt_ask = deduped[i][ticker]["ask"]
            mid_at_fill = (mkt_bid + mkt_ask) / 2

            if our_bid > 0 and our_ask > 0:
                quoted_spread = our_ask - our_bid
            else:
                quoted_spread = None

            if delta > 0:
                # We got filled on our BUY (at our_bid)
                fill_price = our_bid if our_bid > 0 else mkt_ask
                edge = mid_at_fill - fill_price  # positive = good (bought below mid)
                buy_fills.append({
                    "tick": deduped[i]["tick"],
                    "delta": delta,
                    "fill_price": fill_price,
                    "mid": mid_at_fill,
                    "edge": edge,
                    "spread": quoted_spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask
                })
            else:
                # We got filled on our SELL (at our_ask)
                fill_price = our_ask if our_ask > 0 else mkt_bid
                edge = fill_price - mid_at_fill  # positive = good (sold above mid)
                sell_fills.append({
                    "tick": deduped[i]["tick"],
                    "delta": delta,
                    "fill_price": fill_price,
                    "mid": mid_at_fill,
                    "edge": edge,
                    "spread": quoted_spread,
                    "our_bid": our_bid,
                    "our_ask": our_ask
                })

        print(f"\n  {ticker}:")
        if buy_fills:
            avg_edge = sum(f["edge"] for f in buy_fills) / len(buy_fills)
            spreads = [f["spread"] for f in buy_fills if f["spread"] is not None]
            avg_spread = sum(spreads) / len(spreads) if spreads else 0
            positive_edge = sum(1 for f in buy_fills if f["edge"] > 0)
            total_edge_dollars = sum(f["edge"] * f["delta"] for f in buy_fills)
            print(f"    BUY fills: {len(buy_fills)}, avg edge={avg_edge:+.4f}, positive edge={positive_edge}/{len(buy_fills)}")
            print(f"    Avg quoted spread: {avg_spread:.4f}")
            print(f"    Total edge ($ terms): {total_edge_dollars:+.2f}")
        if sell_fills:
            avg_edge = sum(f["edge"] for f in sell_fills) / len(sell_fills)
            spreads = [f["spread"] for f in sell_fills if f["spread"] is not None]
            avg_spread = sum(spreads) / len(spreads) if spreads else 0
            positive_edge = sum(1 for f in sell_fills if f["edge"] > 0)
            total_edge_dollars = sum(f["edge"] * abs(f["delta"]) for f in sell_fills)
            print(f"    SELL fills: {len(sell_fills)}, avg edge={avg_edge:+.4f}, positive edge={positive_edge}/{len(sell_fills)}")
            print(f"    Avg quoted spread: {avg_spread:.4f}")
            print(f"    Total edge ($ terms): {total_edge_dollars:+.2f}")
        if not buy_fills and not sell_fills:
            print(f"    No fills")

def analysis_7_wntr_investigation(rows):
    print("\n" + "=" * 80)
    print("7. WNTR REBATE BOOST INVESTIGATION (2.5x ORDER SIZE)")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    # Compare WNTR vs other tickers in terms of fill frequency, adverse selection, PnL contribution
    for ticker in TICKERS:
        fills = []
        for i in range(1, len(deduped)):
            prev_pos = deduped[i-1][ticker]["pos"]
            curr_pos = deduped[i][ticker]["pos"]
            delta = curr_pos - prev_pos
            if abs(delta) < 0.1:
                continue

            fill_mid = deduped[i][ticker]["mid"]
            future_idx = min(i + 5, len(deduped) - 1)
            future_mid = deduped[future_idx][ticker]["mid"]
            price_move = future_mid - fill_mid

            if delta > 0:
                adverse = price_move < 0
                adverse_cost = -price_move * delta if adverse else 0
            else:
                adverse = price_move > 0
                adverse_cost = price_move * abs(delta) if adverse else 0

            fills.append({
                "tick": deduped[i]["tick"],
                "delta": delta,
                "size": abs(delta),
                "adverse": adverse,
                "adverse_cost": adverse_cost,
                "price_move": price_move,
                "qty_quoted": deduped[i-1][ticker]["qty"]
            })

        if not fills:
            print(f"\n  {ticker}: No fills")
            continue

        total_size = sum(f["size"] for f in fills)
        adverse_count = sum(1 for f in fills if f["adverse"])
        total_adverse_cost = sum(f["adverse_cost"] for f in fills)
        avg_fill_size = total_size / len(fills)
        avg_qty_quoted = sum(f["qty_quoted"] for f in fills) / len(fills)

        print(f"\n  {ticker}:")
        print(f"    Fills: {len(fills)}")
        print(f"    Total volume: {total_size:.0f}")
        print(f"    Avg fill size: {avg_fill_size:.0f}")
        print(f"    Avg qty quoted: {avg_qty_quoted:.0f}")
        print(f"    Adverse fills: {adverse_count}/{len(fills)} ({adverse_count/len(fills)*100:.1f}%)")
        print(f"    Total adverse cost: {total_adverse_cost:.2f}")
        print(f"    Adverse cost per unit volume: {total_adverse_cost/total_size:.4f}" if total_size > 0 else "")

    # WNTR specific: track positions
    print("\n  --- WNTR Position Tracking ---")
    max_wntr_pos = 0
    min_wntr_pos = 0
    for r in deduped:
        pos = r["WNTR"]["pos"]
        if pos > max_wntr_pos:
            max_wntr_pos = pos
        if pos < min_wntr_pos:
            min_wntr_pos = pos
    print(f"    Max WNTR position: {max_wntr_pos:.0f}")
    print(f"    Min WNTR position: {min_wntr_pos:.0f}")

def analysis_8_largest_drops(rows):
    print("\n" + "=" * 80)
    print("8. FIVE LARGEST SINGLE-TICK NLV DROPS (DETAILED)")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    drops = []
    for i in range(1, len(deduped)):
        delta = deduped[i]["nlv"] - deduped[i-1]["nlv"]
        drops.append((delta, i))

    drops.sort(key=lambda x: x[0])

    for rank, (delta, i) in enumerate(drops[:5], 1):
        curr = deduped[i]
        prev = deduped[i-1]
        print(f"\n  #{rank}: NLV drop = {delta:+.2f}")
        print(f"    Tick {prev['tick']} -> {curr['tick']}, t_min {prev['t_min']} -> {curr['t_min']}")
        print(f"    NLV: {prev['nlv']:.2f} -> {curr['nlv']:.2f}")
        print(f"    Actions: {curr['actions']}")
        for t in TICKERS:
            prev_pos = prev[t]["pos"]
            curr_pos = curr[t]["pos"]
            pos_delta = curr_pos - prev_pos
            prev_mid = prev[t]["mid"]
            curr_mid = curr[t]["mid"]
            mid_move = curr_mid - prev_mid
            if abs(pos_delta) > 0.1 or abs(mid_move) > 0.02:
                print(f"    {t}: pos {prev_pos:.0f} -> {curr_pos:.0f} (delta={pos_delta:+.0f}), "
                      f"mid {prev_mid:.3f} -> {curr_mid:.3f} (move={mid_move:+.3f}), "
                      f"bid={curr[t]['bid']:.2f}, ask={curr[t]['ask']:.2f}")

def analysis_extra_position_at_unwind(rows):
    """Look at what positions we're stuck with at unwind time."""
    print("\n" + "=" * 80)
    print("EXTRA: POSITION SIZE AT UNWIND ENTRY (t_min=49->50)")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    for i in range(len(deduped)):
        if deduped[i]["t_min"] == 49:
            # Check if next tick starts unwind
            if i + 1 < len(deduped) and deduped[i+1]["t_min"] == 50:
                r = deduped[i]
                print(f"\n  Tick {r['tick']}, NLV={r['nlv']:.2f}")
                for t in TICKERS:
                    pos = r[t]["pos"]
                    mid = r[t]["mid"]
                    if abs(pos) > 0:
                        print(f"    {t}: pos={pos:.0f}, mid={mid:.3f}, exposure=${abs(pos)*mid:.0f}")

def analysis_extra_stuck_positions(rows):
    """Analyze periods where positions are stuck (no fills for many ticks) while prices move against us."""
    print("\n" + "=" * 80)
    print("EXTRA: STUCK POSITION ANALYSIS (long holds with adverse price moves)")
    print("=" * 80)

    deduped = dedupe_ticks(rows)

    for ticker in TICKERS:
        # Find stretches where position doesn't change
        stretches = []
        i = 0
        while i < len(deduped):
            pos = deduped[i][ticker]["pos"]
            if abs(pos) > 100:
                j = i
                while j < len(deduped) and abs(deduped[j][ticker]["pos"] - pos) < 0.1:
                    j += 1
                if j - i >= 5:  # at least 5 ticks stuck
                    start_mid = deduped[i][ticker]["mid"]
                    end_mid = deduped[j-1][ticker]["mid"]
                    mid_change = end_mid - start_mid
                    pnl_impact = mid_change * pos  # positive pos: price up = good; negative pos: price down = good
                    stretches.append({
                        "start_tick": deduped[i]["tick"],
                        "end_tick": deduped[j-1]["tick"],
                        "duration": j - i,
                        "pos": pos,
                        "start_mid": start_mid,
                        "end_mid": end_mid,
                        "mid_change": mid_change,
                        "pnl_impact": pnl_impact
                    })
                i = j
            else:
                i += 1

        if stretches:
            worst = sorted(stretches, key=lambda s: s["pnl_impact"])[:3]
            print(f"\n  {ticker} - worst stuck periods:")
            for s in worst:
                print(f"    ticks {s['start_tick']}-{s['end_tick']} ({s['duration']} ticks), pos={s['pos']:.0f}, "
                      f"mid {s['start_mid']:.3f}->{s['end_mid']:.3f} ({s['mid_change']:+.3f}), "
                      f"PnL impact={s['pnl_impact']:+.2f}")

if __name__ == "__main__":
    print("Loading data...")
    rows = load_data()
    print(f"Loaded {len(rows)} log entries")

    deduped = analysis_1_nlv_trajectory(rows)
    analysis_2_adverse_selection(rows)
    analysis_3_unwind_cost(rows)
    analysis_4_per_minute_pnl(rows)
    analysis_5_order_size_vs_adverse(rows)
    analysis_6_spread_capture(rows)
    analysis_7_wntr_investigation(rows)
    analysis_8_largest_drops(rows)
    analysis_extra_position_at_unwind(rows)
    analysis_extra_stuck_positions(rows)
