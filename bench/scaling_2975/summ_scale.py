import json, sys, math, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.startswith("{")]
schemes = list(dict.fromkeys(r["scheme"] for r in rows))
by = collections.defaultdict(dict)
for r in rows: by[r["model"]][r["scheme"]] = r
print(f"{'model':28s} " + " ".join(f"{s:>16s}" for s in schemes))
tot = {s: [0, 0, 0, 0] for s in schemes}  # iters, ebe, worse-OFV count, errors
for m, d in by.items():
    ofvs = [r["ofv"] for r in d.values() if r.get("ofv") is not None and math.isfinite(r["ofv"])]
    best = min(ofvs) if ofvs else float("nan")
    cells = []
    for s in schemes:
        r = d.get(s)
        if not r or r.get("ofv") is None:
            cells.append(f"{'ERR':>16s}"); tot[s][3] += 1; continue
        bad = r["ofv"] - best > 1e-3
        flag = ("!" if bad else "") + ("" if r.get("conv") else "~")
        cells.append(f"{r['iters']:>4d}it {r['ebe']:>6d}e{flag:2s}")
        tot[s][0] += r["iters"]; tot[s][1] += r["ebe"]; tot[s][2] += bad
    print(f"{m:28s} " + " ".join(cells))
print("\nTOTAL iterations / EBE solves / fits ending >1e-3 above best / errors")
for s in schemes: print(f"  {s:8s} {tot[s][0]:6d} {tot[s][1]:8d} {tot[s][2]:3d} {tot[s][3]:3d}")
