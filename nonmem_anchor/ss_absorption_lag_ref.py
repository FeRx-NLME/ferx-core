# Writes ss_absorption_lag_ref_{zo,fo}.csv:
#   python3 ss_absorption_lag_ref.py ss_absorption_lag.csv zo > ss_absorption_lag_ref_zo.csv
#   python3 ss_absorption_lag_ref.py ss_absorption_lag.csv fo > ss_absorption_lag_ref_fo.csv
"""Exact reference for SS=1 doses into an absorption input with a lag (#2786).

Independent of ferx.  1-cpt disposition (zo: CMT 1 = central, zero-order input of duration DR;
fo: CMT 1 = depot with ka, CMT 2 = central; bolus or an infusion into the depot), the modal
matrix exponential of the linear system, and a pulse-train superposition for the steady state
at the SS record: every virtual pulse arriving at  t0 - psi - j*II  (j = 0, 1, ...) is summed
explicitly until it underflows, with psi = max(II - lag, 0) (a lag >= II clamps to the pulse
landing on the record; ss_seed_phase).  The dose's own pulse arrives at t0 + lag.  Pulses of the
train still running after the record (a window longer than the time since its arrival) and every
later dose are propagated exactly, piecewise at every window edge and bolus arrival, under the
parameters of the record that ends each interval (NONMEM).  CL = 2 (WT/70)^0.75, V = 20, ka = 0.7.
Usage: python3 ref.py <dataset.csv> <zo|fo> [wrap] > out.csv
The optional `wrap` writes the unclamped periodic train instead (ss_absorption_lag_ref_*_wrap.csv):
identical for every lag < II, different only for lag > II (IDs 7 and 15), where NONMEM's analytic
ADVAN1/ADVAN2 follow it from t0 + lag - II on and ferx follows the documented clamp.
"""
import csv, math, sys

KA = 0.7


def system(cl, v, model):
    k = cl / v
    if model == "zo":
        return 1, [(-k, [[1.0]])]
    a = [[-KA, 0.0], [KA, -k]]
    l1, l2 = -KA, -k

    def proj(li, lj):
        return [[(a[r][c] - (lj if r == c else 0.0)) / (li - lj) for c in range(2)] for r in range(2)]

    return 2, [(l1, proj(l1, l2)), (l2, proj(l2, l1))]


def matvec(p, x):
    return [math.fsum(p[r][c] * x[c] for c in range(len(x))) for r in range(len(p))]


def propagate(x, h, md, rate):
    """x(h) for a constant input `rate` into compartment 0."""
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


def pulse_state(t, a, dur, rate, bolus, md, n):
    """State at t >= a from one pulse arriving at a: a window (rate, dur) or a bolus into cmt 0."""
    if bolus:
        x = [0.0] * n
        x[0] = bolus
        return propagate(x, t - a, md, 0.0)
    e = min(t, a + dur)
    x = propagate([0.0] * n, e - a, md, rate)
    return propagate(x, t - e, md, 0.0)


def subject_preds(rows, model):
    def pars(r):
        return system(2.0 * (float(r["WT"]) / 70.0) ** 0.75, 20.0, model)

    obs_col = 0 if model == "zo" else 1
    windows = []  # (start, end, rate) into compartment 0
    boluses = []  # (arrival, amount) into compartment 0
    x, t, out = None, None, []
    for r in rows:
        tr = float(r["TIME"])
        n, md = pars(r)
        if x is None:
            x = [0.0] * n
        for _, amt in [b for b in boluses if b[0] <= (t if t is not None else tr)]:
            x[0] += amt  # a lag-0 bolus lands on the record itself
        boluses = [b for b in boluses if b[0] > (t if t is not None else tr)]
        if t is not None and tr > t:
            edges = {tr}
            for s, e, _ in windows:
                edges |= {w for w in (s, e) if t < w < tr}
            edges |= {a for a, _ in boluses if t < a < tr}
            cur = t
            for c in sorted(edges):
                mid = 0.5 * (cur + c)
                rate = math.fsum(rt for s, e, rt in windows if s <= mid <= e)
                x = propagate(x, c - cur, md, rate)
                cur = c
                for _, amt in [b for b in boluses if b[0] == c]:
                    x[0] += amt
                boluses = [b for b in boluses if b[0] != c]
        t = tr
        if r["EVID"] == "1":
            amt, rate = float(r["AMT"]), float(r["RATE"])
            lag, ii = float(r["LG"]), float(r["II"] or 0)
            if model == "zo":
                dur = float(r["DR"])
                wrate, bolus = amt / dur, 0.0
            elif rate > 0:
                dur = amt / rate
                wrate, bolus = rate, 0.0
            else:
                dur, wrate, bolus = 0.0, 0.0, amt
            if r["SS"] == "1":
                windows, boluses = [], []
                # clamp: a lag >= II leaves the train on the record (ferx, ss_seed_phase);
                # wrap: the unclamped periodic train, arrivals t0 + lag - j*II for every j
                # (what NONMEM's analytic ADVAN1/ADVAN2 deliver from t0 + lag - II on)
                psi = (-lag) % ii if WRAP else max(ii - lag, 0.0)
                acc = [[] for _ in range(n)]
                j = 0
                while True:
                    a = tr - psi - j * ii
                    s = pulse_state(tr, a, dur, wrate, bolus, md, n)
                    for i in range(n):
                        acc[i].append(s[i])
                    if not bolus and a + dur > tr:
                        windows.append((tr, a + dur, wrate))
                    if j > 50 and max(abs(v) for v in s) < 1e-300:
                        break
                    j += 1
                    if j > 200000:
                        break
                x = [math.fsum(a_) for a_ in acc]
            arrivals = [tr + lag]
            if WRAP and r["SS"] == "1":
                arrivals = []
                k = 0
                while tr + lag - k * ii > tr:  # the train's pulses still to arrive after the record
                    arrivals.append(tr + lag - k * ii)
                    k += 1
            for a in arrivals:
                if bolus:
                    boluses.append((a, bolus))
                else:
                    windows.append((a, a + dur, wrate))
        elif r["EVID"] == "0" or r["MDV"] == "0":
            out.append((r["ID"], tr, x[obs_col] / 20.0))
    return out


WRAP = len(sys.argv) > 3 and sys.argv[3] == "wrap"


def main():
    path, model = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(open(path)))
    lo, hi = (1, 9) if model == "zo" else (11, 17)
    print("ID,TIME,PRED")
    for i in sorted({int(r["ID"]) for r in rows}):
        if lo <= i <= hi:
            for sid, tm, p in subject_preds([r for r in rows if int(r["ID"]) == i], model):
                print(f"{sid},{tm!r},{p!r}")


main()
