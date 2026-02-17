from sys import argv

lines = open(argv[1]).readlines()
data = []
for line in lines[1:]:
    line = line.strip()
    if not line: continue
    parts = line.split('|')
    if len(parts) < 8: continue
    tick = int(parts[0])
    t_min = int(parts[1])
    agg_exp = float(parts[2])
    gross = float(parts[3])
    gross_lim = int(parts[5])
    tickers = {}
    for i, name in enumerate(['SPNG','SMMR','ATMN','WNTR']):
        vals = parts[7+i].split(',')
        tickers[name] = {'pos': float(vals[0])}
    actions = parts[11] if len(parts) > 11 else ''
    data.append({'tick': tick, 't_min': t_min, 'agg_exp': agg_exp, 'gross': gross, 'gross_lim': gross_lim, 'tickers': tickers, 'actions': actions})

# Position patterns
print('=== POSITION PATTERNS ===')
for name in ['SPNG','SMMR','ATMN','WNTR']:
    positions = [d['tickers'][name]['pos'] for d in data if 'quote' in d['actions']]
    if positions:
        print(f'{name}: min={min(positions):.0f}, max={max(positions):.0f}, mean_abs={sum(abs(p) for p in positions)/len(positions):.0f}')
        # how often positive vs negative
        pos_count = sum(1 for p in positions if p > 0)
        neg_count = sum(1 for p in positions if p < 0)
        zero_count = sum(1 for p in positions if p == 0)
        print(f'  Long: {pos_count}, Short: {neg_count}, Flat: {zero_count}')

print()
print('=== AGGREGATE EXPOSURE UTILIZATION ===')
quoting_data = [d for d in data if 'quote' in d['actions']]
agg_exps = [d['agg_exp'] for d in quoting_data]
print(f'During quoting phases:')
print(f'  Mean agg_exp: {sum(agg_exps)/len(agg_exps):.0f} ({sum(agg_exps)/len(agg_exps)/50000*100:.1f}% of 50k)')
print(f'  Max agg_exp: {max(agg_exps):.0f} ({max(agg_exps)/50000*100:.1f}% of 50k)')
print(f'  Min agg_exp: {min(agg_exps):.0f} ({min(agg_exps)/50000*100:.1f}% of 50k)')

# Distribution
buckets = [0, 2000, 4000, 6000, 8000, 10000]
for i in range(len(buckets)-1):
    count = sum(1 for a in agg_exps if buckets[i] <= a < buckets[i+1])
    print(f'  {buckets[i]}-{buckets[i+1]}: {count} entries ({count/len(agg_exps)*100:.1f}%)')
over = sum(1 for a in agg_exps if a >= 10000)
print(f'  10000+: {over} entries ({over/len(agg_exps)*100:.1f}%)')

print()
print('=== TICK GAPS ===')
quoting_ticks = [d['tick'] for d in quoting_data]
gaps = [quoting_ticks[i+1] - quoting_ticks[i] for i in range(len(quoting_ticks)-1)]
# Filter out cross-minute gaps
same_minute_gaps = []
for i in range(len(quoting_data)-1):
    if quoting_data[i+1]['t_min'] > quoting_data[i]['t_min']:  # same minute
        same_minute_gaps.append(quoting_ticks[i+1] - quoting_ticks[i])
if same_minute_gaps:
    print(f'Tick gaps (same minute, quoting entries only):')
    print(f'  Mean: {sum(same_minute_gaps)/len(same_minute_gaps):.1f}')
    print(f'  Min: {min(same_minute_gaps)}, Max: {max(same_minute_gaps)}')
    from collections import Counter
    gc = Counter(same_minute_gaps)
    print(f'  Distribution: {dict(sorted(gc.items()))}')

# All consecutive log entries (not just quoting)
all_ticks = [d['tick'] for d in data]
all_gaps = [all_ticks[i+1] - all_ticks[i] for i in range(len(all_ticks)-1) if all_ticks[i+1] > all_ticks[i]]
if all_gaps:
    print(f'All log entry gaps (when tick advances):')
    print(f'  Mean: {sum(all_gaps)/len(all_gaps):.2f}')
    gc2 = Counter(all_gaps)
    print(f'  Distribution: {dict(sorted(gc2.items()))}')