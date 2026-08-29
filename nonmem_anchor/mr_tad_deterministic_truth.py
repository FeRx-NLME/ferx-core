#!/usr/bin/env python3
"""Independent RK4 reference for the #1124 anchor's `$DES` construction.

NONMEM has no `TAD` built-in in `$DES`, so `mr_tad.ctl` derives it from `T`:

    TDOS = 0.0
    IF (T.GE.12.0) TDOS = 12.0
    TADX = T - TDOS

That is an *authored* construction, and authoring `TAD` in `$DES` is where the
`tad_lag_{A,B}` anchors went wrong twice before being settled the same way (an
`MPAST` version was 40 % off; an `MTIME`-inside-`IF` version did not execute).
This script re-derives the trajectory from the model equations with a plain
stdlib RK4 — no NONMEM, no ferx — so the control stream is checked rather than
trusted.

Compare against the deterministic NONMEM variant (`$OMEGA 0 FIX`, so `IPRED` is a
pure function of `THETA` and every subject is identical):

    python3 nonmem_anchor/mr_tad_deterministic_truth.py
    # vs the IPRED column of mr_tad_det.tab

Usage: pass the NONMEM table to compare directly, e.g.

    python3 nonmem_anchor/mr_tad_deterministic_truth.py mr_tad_det.tab
"""
import importlib.util
import os
import sys

_SPEC = importlib.util.spec_from_file_location(
    "sim", os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulate_mr_tad_data.py")
)
sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sim)


def truth():
    """{obs time: IPRED} at the typical-value parameter point (eta = 0)."""
    amounts = sim.simulate(sim.TVCL, sim.TVV, sim.TVKA)
    return {t: a / sim.TVV for t, a in amounts.items()}


def read_table(path):
    """{time: IPRED} from a NONMEM `$TABLE` file (first subject; all identical)."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("TABLE")]
    header = lines[0].split()
    i_t, i_p = header.index("TIME"), header.index("IPRED")
    out = {}
    for ln in lines[1:]:
        f = ln.split()
        t, p = float(f[i_t]), float(f[i_p])
        out.setdefault(t, p)
    return out


def main():
    ref = truth()
    if len(sys.argv) < 2:
        for t in sorted(ref):
            print(f"{t:8.2f} {ref[t]:.6f}")
        return
    tab = read_table(sys.argv[1])
    worst = 0.0
    print(f"{'TIME':>8} {'NONMEM':>12} {'RK4 truth':>12} {'rel':>10}")
    for t in sorted(ref):
        if t not in tab:
            continue
        a, b = tab[t], ref[t]
        rel = abs(a - b) / (1.0 + abs(b))
        worst = max(worst, rel)
        print(f"{t:8.2f} {a:12.6f} {b:12.6f} {rel:10.2e}")
    print(f"\nworst relative deviation: {worst:.3e}")
    # The table prints 6 significant figures, so agreement is bounded below by
    # table precision, not by either integrator.
    sys.exit(0 if worst < 1e-4 else 1)


if __name__ == "__main__":
    main()
