# Writes ss_infusion_overlap_ref_{1cpt,2cpt}.csv:
#   python3 ss_infusion_overlap_ref.py ss_infusion_overlap.csv 1cpt > ss_infusion_overlap_ref_1cpt.csv
"""Exact reference for overlapping SS infusions (T_inf > II) on linear 1-/2-cpt IV models.

Independent of ferx: modal matrix exponential (closed-form 2x2 eigensystem), pulse-train
superposition for the steady-state state at the SS record (every virtual pulse summed
explicitly until its contribution underflows), then exact piecewise propagation of every
infusion window still running after the record and of the later doses.
Usage: python3 ref.py <dataset.csv> <model: 1cpt|2cpt> > out.csv
"""
import csv, math, sys

def modes(cl, v, q=None, v2=None):
    if q is None:
        k = cl / v
        return [(-k, [[1.0]])]
    k10, k12, k21 = cl / v, q / v, q / v2
    a = [[-(k10 + k12), k21], [k12, -k21]]
    tr = a[0][0] + a[1][1]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    disc = math.sqrt(tr * tr / 4 - det)
    l1, l2 = tr / 2 + disc, tr / 2 - disc
    def proj(li, lj):
        return [[(a[r][c] - (lj if r == c else 0.0)) / (li - lj) for c in range(2)] for r in range(2)]
    return [(l1, proj(l1, l2)), (l2, proj(l2, l1))]

def matvec(p, x):
    return [math.fsum(p[r][c] * x[c] for c in range(len(x))) for r in range(len(p))]

def propagate(x, h, md, rate):
    """x(h) for constant input `rate` into compartment 0."""
    n = len(x)
    b = [rate] + [0.0] * (n - 1)
    out = [[] for _ in range(n)]
    for lam, p in md:
        px, pb = matvec(p, x), matvec(p, b)
        e = math.exp(lam * h)
        g = math.expm1(lam * h) / lam
        for r in range(n):
            out[r] += [px[r] * e, pb[r] * g]
    return [math.fsum(o) for o in out]

def pulse_state(t, a, dur, rate, md, n):
    """State at t from one infusion arriving at a (t >= a)."""
    e = min(t, a + dur)
    x = propagate([0.0] * n, e - a, md, rate)
    return propagate(x, t - e, md, 0.0)

def subject_preds(rows, model):
    def pars(r):
        cl = 2.0 * (float(r["WT"]) / 70.0) ** 0.75
        return dict(cl=cl, v=20.0) if model == "1cpt" else dict(cl=cl, v=20.0, q=4.0, v2=40.0)
    n = 1 if model == "1cpt" else 2
    windows = []  # (start, end, rate) live after the last reset
    x = [0.0] * n
    t = None
    out = []
    # records in file order; a dose row's parameters are its own, an interval runs under the
    # record that ends it (NONMEM)
    for r in rows:
        tr = float(r["TIME"])
        md = modes(**pars(r))
        if t is not None and tr > t:
            # exact piecewise propagation under the governing record's parameters
            cuts = sorted({tr} | {w for s, e, _ in windows for w in (s, e) if t < w < tr})
            cur = t
            for c in cuts:
                mid = 0.5 * (cur + c)
                rate = math.fsum(rt for s, e, rt in windows if s <= mid <= e)
                x = propagate(x, c - cur, md, rate)
                cur = c
        t = tr
        if r["EVID"] == "1":
            amt, rate = float(r["AMT"]), float(r["RATE"])
            lag, ii = float(r.get("LG", 0) or 0), float(r["II"] or 0)
            dur = amt / rate
            if r["SS"] == "1":
                x = [0.0] * n
                windows = []
                j, acc = 1, [[] for _ in range(n)]
                while True:
                    a = tr + lag - j * ii
                    if a >= tr:  # (not reached for lag < II)
                        raise SystemExit("lag >= II not covered")
                    s = pulse_state(tr, a, dur, rate, md, n)
                    for i in range(n):
                        acc[i].append(s[i])
                    if a + dur > tr:
                        windows.append((tr, a + dur, rate))
                    if j > 50 and max(abs(v) for v in s) < 1e-300:
                        break
                    j += 1
                    if j > 200000:
                        break
                x = [math.fsum(a_) for a_ in acc]
            windows.append((tr + lag, tr + lag + dur, rate))
        elif r["EVID"] == "0" or r["MDV"] == "0":
            out.append((r["ID"], tr, x[0] / 20.0))
    return out

def main():
    path, model = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(path)))
    print("ID,TIME,PRED")
    ids = []
    for r in rows:
        if r["ID"] not in ids:
            ids.append(r["ID"])
    for i in ids:
        for (sid, tm, p) in subject_preds([r for r in rows if r["ID"] == i], model):
            print(f"{sid},{tm!r},{p!r}")

main()
