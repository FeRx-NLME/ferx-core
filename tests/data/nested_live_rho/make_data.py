"""Build nested_live_rho.csv for #2393: nested_levels_3's design with AR(1) residuals.

The rows, IDs and design columns (ID, STUDY, ARM, ARMID, STUDYID, SLOT, TIME, NARM, DOSE)
are ../../../nonmem_anchor/nested_levels_3.csv's, in its order (each arm's five visits
contiguous and in TIME order). DV is simulated afresh from the model
`tests/profiled_levels_correlation_fits.rs` fits (`nested_text()`):

    EMAX = TVEMAX * exp(ETA_SP + K_ST + K_ARM)
    y    = TVE0 + PLACEBO[STUDYID] + EMAX * DOSE / (ED50 + DOSE) * TIME / (TIME + ET50)
    DV   = y + eps,   eps AR(1) within each arm (lag = visit index), var ADD, corr RHO

with the population values nested_levels_3.ferx starts at, a per-study placebo offset,
and RHO = 0.9: nested_levels_3.csv has almost no within-arm serial correlation, so a fit
of that data puts ρ at 0.02 and no fit to convergence exercises a live ρ on a nested
model. The fit lands at ρ = 0.648, not 0.9: the per-study PLACEBO and the per-arm K_ARM
absorb part of a five-visit arm's serial correlation (at 0.55 the fit gave 0.15, at 0.8
0.43; `rho_check.py` measures the split, quoted in RHO_LIVE's comment in the test).
Stdlib `random` (Mersenne Twister, `gauss`) under a fixed seed, so the file regenerates
bit-for-bit.
"""
import csv
import math
import os
import random

TVE0, TVEMAX, ED50, ET50 = 10.1818, 22.0140, 33.4048, 3.20150
OM_SP, K_ST, K_ARM = 0.0699388, 0.0904362, 3.75478  # K_ARM ~ N(0, K_ARM / NARM)
ADD, RHO = 0.255164, 0.9
SD_PLACEBO = 1.0

here = os.path.dirname(os.path.abspath(__file__))


def simulate():
    """The design rows with `DV` simulated, each also carrying its residual as `EPS`."""
    src = os.path.join(here, "..", "..", "..", "nonmem_anchor", "nested_levels_3.csv")
    with open(src) as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(2393)
    eta_sp, k_st, k_arm, placebo, eps, last = {}, {}, {}, {}, {}, {}
    for r in rows:
        sp, st, arm = r["ID"], r["STUDYID"], r["ARMID"]
        if sp not in eta_sp:
            eta_sp[sp] = rng.gauss(0.0, math.sqrt(OM_SP))
        if st not in k_st:
            k_st[st] = rng.gauss(0.0, math.sqrt(K_ST))
            placebo[st] = rng.gauss(0.0, SD_PLACEBO)
        if arm not in k_arm:
            k_arm[arm] = rng.gauss(0.0, math.sqrt(K_ARM / float(r["NARM"])))
            e = rng.gauss(0.0, math.sqrt(ADD))  # stationary start
        else:
            assert float(r["TIME"]) > last[arm], arm  # visits contiguous, in TIME order
            e = RHO * eps[arm] + rng.gauss(0.0, math.sqrt(ADD * (1.0 - RHO * RHO)))
        eps[arm], last[arm] = e, float(r["TIME"])
        t, dose = float(r["TIME"]), float(r["DOSE"])
        emax = TVEMAX * math.exp(eta_sp[sp] + k_st[st] + k_arm[arm])
        y = TVE0 + placebo[st] + emax * dose / (ED50 + dose) * t / (t + ET50)
        r["DV"] = f"{y + e:.5f}"
        r["EPS"] = e
    return rows


if __name__ == "__main__":
    rows = simulate()
    cols = [c for c in rows[0] if c != "EPS"]
    with open(os.path.join(here, "nested_live_rho.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
