#!/usr/bin/env python3
"""Deeper analysis on specific problem areas."""

LOG_FILE = "/Users/colin/Desktop/TCD/tcdrot-scripthook/logs/20260217_231457.txt"
TICKERS = ["SPNG", "SMMR", "ATMN", "WNTR"]

def parse_ticker(field):
    parts = field.split(",")
    return {
        "pos": float(parts[0]), "bid": float(parts[1]), "ask": float(parts[2]),
        "our_bid": float(parts[3]), "our_ask": float(parts[4]), "qty": float(parts[5]),
        "mid": (float(parts[1]) + float(parts[2])) / 2
    }

def parse_line(line):
    parts = line.strip().split("|")
    if len(parts) < 13: return None
    row = {
        "tick": int(parts[0]), "t_min": int(parts[1]),
        "nlv": float(parts[7]), "net": float(parts[4]),
        "gross": float(parts[3]), "actions": parts[12] if len(parts) > 12 else ""
    }
    for i, t in enumerate(TICKERS):
        field = parts[8 + i]
        if ":" in field and not field[0].isdigit() and not field[0] == "-":
            field = field.split(":", 1)[1]
        row[t] = parse_ticker(field)
    return row

def load():
    rows = []
    with open(LOG_FILE) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            line = line.strip()
            if not line: continue
            r = parse_line(line)
            if r: rows.append(r)
    return rows

def dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        if r["tick"] not in seen:
            seen.add(r["tick"])
            out.append(r)
    return out

rows = load()
deduped = dedupe(rows)

print("=" * 80)
print("DEEP DIVE: MINUTE 2 CATASTROPHE (ticks 60-119, lost $5853)")
print("=" * 80)

# Minute 2: What happened with SPNG -6000?
for r in deduped:
    if 65 <= r["tick"] <= 70:
        print(f"\n  tick={r['tick']}, t_min={r['t_min']}, NLV={r['nlv']:.2f}")
        for t in TICKERS:
            d = r[t]
            if abs(d["pos"]) > 0 or d["our_bid"] > 0:
                print(f"    {t}: pos={d['pos']:.0f}, bid={d['bid']:.2f}, ask={d['ask']:.2f}, "
                      f"our_bid={d['our_bid']:.2f}, our_ask={d['our_ask']:.2f}, qty={d['qty']:.0f}")
        print(f"    actions: {r['actions']}")

print("\n\n  SPNG went from pos=0 to pos=-6000 at tick 67 (t_min=7)")
print("  This means someone hit our ask. Our ask would have been around 25.01-25.02")
print("  But SPNG bid immediately jumped to 25.61/25.30/25.33 = massive adverse move UP")
print("  We sold 6000 shares and price jumped ~$0.30 against us = $1800 instant loss")

print("\n" + "=" * 80)
print("DEEP DIVE: MINUTE 4 CATASTROPHE (ticks 180-239, lost $5386)")
print("=" * 80)

for r in deduped:
    if 180 <= r["tick"] <= 200:
        print(f"\n  tick={r['tick']}, t_min={r['t_min']}, NLV={r['nlv']:.2f}")
        for t in TICKERS:
            d = r[t]
            prev = None
            for r2 in deduped:
                if r2["tick"] == r["tick"] - 1:
                    prev = r2
                    break
            if prev:
                delta = d["pos"] - prev[t]["pos"]
            else:
                delta = 0
            if abs(d["pos"]) > 0 or abs(delta) > 0:
                print(f"    {t}: pos={d['pos']:.0f}" + (f" (FILL delta={delta:+.0f})" if abs(delta) > 0 else "") +
                      f", bid={d['bid']:.2f}, ask={d['ask']:.2f}, mid={d['mid']:.3f}")
        print(f"    actions: {r['actions']}")

print("\n\n" + "=" * 80)
print("DEEP DIVE: MINUTE 5 (ticks 240-290)")
print("=" * 80)

for r in deduped:
    if 240 <= r["tick"] <= 260:
        print(f"\n  tick={r['tick']}, t_min={r['t_min']}, NLV={r['nlv']:.2f}")
        for t in TICKERS:
            d = r[t]
            if abs(d["pos"]) > 0 or d["our_bid"] > 0:
                print(f"    {t}: pos={d['pos']:.0f}, bid={d['bid']:.2f}, ask={d['ask']:.2f}, "
                      f"our_bid={d['our_bid']:.2f}, our_ask={d['our_ask']:.2f}, qty={d['qty']:.0f}")
        print(f"    actions: {r['actions']}")

print("\n\n" + "=" * 80)
print("DEEP DIVE: SPNG BID/ASK ANOMALY")
print("=" * 80)
print("SPNG ask is often 25.03 while bid ranges 25.2-25.9 (ask < bid!)")
print("This is INVERTED. Checking all SPNG bid>ask instances...")

inverted_count = 0
normal_count = 0
for r in deduped:
    if r["SPNG"]["bid"] > r["SPNG"]["ask"] and r["SPNG"]["ask"] > 0:
        inverted_count += 1
    elif r["SPNG"]["ask"] > 0:
        normal_count += 1
print(f"  SPNG inverted (bid>ask): {inverted_count}/{inverted_count+normal_count} ticks ({inverted_count/(inverted_count+normal_count)*100:.1f}%)")

for t in TICKERS:
    inv = sum(1 for r in deduped if r[t]["bid"] > r[t]["ask"] and r[t]["ask"] > 0)
    tot = sum(1 for r in deduped if r[t]["ask"] > 0)
    if tot > 0:
        print(f"  {t} inverted: {inv}/{tot} ({inv/tot*100:.1f}%)")

print("\n\n" + "=" * 80)
print("DEEP DIVE: NET EXPOSURE / DIRECTIONALITY")
print("=" * 80)
print("Are we consistently one-sided? That would explain losses in trending markets.")

for r in deduped:
    if r["t_min"] in [0, 10, 20, 30, 40, 49]:
        net_dollar = 0
        positions_str = []
        for t in TICKERS:
            pos = r[t]["pos"]
            mid = r[t]["mid"]
            dollar = pos * mid
            net_dollar += dollar
            if abs(pos) > 0:
                positions_str.append(f"{t}:{pos:.0f}")
        if any(abs(r[t]["pos"]) > 0 for t in TICKERS):
            print(f"  tick={r['tick']}, t_min={r['t_min']}, net$={net_dollar:+.0f}, NLV={r['nlv']:.2f}, "
                  f"positions: {', '.join(positions_str)}")

print("\n\n" + "=" * 80)
print("SUMMARY: PER-TICKER PNL ATTRIBUTION (spread capture analysis)")
print("=" * 80)

# For each ticker, approximate PnL contribution from fills vs mark-to-market
for ticker in TICKERS:
    total_buy_cost = 0
    total_sell_revenue = 0
    total_buy_shares = 0
    total_sell_shares = 0

    for i in range(1, len(deduped)):
        prev_pos = deduped[i-1][ticker]["pos"]
        curr_pos = deduped[i][ticker]["pos"]
        delta = curr_pos - prev_pos
        if abs(delta) < 0.1:
            continue

        if delta > 0:  # bought
            # Fill was likely at our_bid or ask
            our_bid = deduped[i-1][ticker]["our_bid"]
            mkt_ask = deduped[i][ticker]["ask"]
            fill_price = our_bid if our_bid > 0 else mkt_ask
            total_buy_cost += fill_price * delta
            total_buy_shares += delta
        else:  # sold
            our_ask = deduped[i-1][ticker]["our_ask"]
            mkt_bid = deduped[i][ticker]["bid"]
            fill_price = our_ask if our_ask > 0 else mkt_bid
            total_sell_revenue += fill_price * abs(delta)
            total_sell_shares += abs(delta)

    print(f"\n  {ticker}:")
    print(f"    Total bought: {total_buy_shares:.0f} shares, cost: ${total_buy_cost:.2f}")
    print(f"    Total sold: {total_sell_shares:.0f} shares, revenue: ${total_sell_revenue:.2f}")
    if total_buy_shares > 0:
        print(f"    Avg buy price: {total_buy_cost/total_buy_shares:.4f}")
    if total_sell_shares > 0:
        print(f"    Avg sell price: {total_sell_revenue/total_sell_shares:.4f}")
    print(f"    Trading P&L (sell rev - buy cost): ${total_sell_revenue - total_buy_cost:+.2f}")
