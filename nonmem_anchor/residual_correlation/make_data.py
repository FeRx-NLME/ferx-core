"""Derive the #1081 anchor dataset from ferx-core data/mbma_naproxen.csv.
Natural-scale DV (= DV*WPSE), L2 = arm id (1..36), VIS = visit index in arm,
T1..T5 = the arm's visit times (for the continuous-time AR Cholesky)."""
import csv, sys, collections
src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src)))
arms = collections.OrderedDict()
for r in rows:
    arms.setdefault((r['ID'], r['ARM']), []).append(r)
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ID','ARM','TIME','DV','TRT','FLARE','WPSE','MDV','L2','VIS','T1','T2','T3','T4','T5'])
    for l2, (key, rs) in enumerate(arms.items(), start=1):
        ts = [float(r['TIME']) for r in rs]
        assert ts == sorted(ts) and len(set(ts)) == len(ts)
        tcols = ts + [0.0]*(5-len(ts))
        for vis, r in enumerate(rs, start=1):
            dv = float(r['DV'])*float(r['WPSE'])
            w.writerow([r['ID'], r['ARM'], r['TIME'], repr(dv), r['TRT'], r['FLARE'], r['WPSE'], r['MDV'], l2, vis] + [repr(t) for t in tcols])
