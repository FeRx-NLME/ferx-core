# Vancomycin loading-dose + maintenance-titration reference (mrgsolve)

Frozen output of `vanco_loading_mrgsolve.R` (mrgsolve 2.0.1). External anchor for
the reactive `[adaptive_dosing]` **pre-scheduled base regimen** support (#702) in
`examples/adaptive_vanco_loading.ferx`. NONMEM has no feedback dosing, so mrgsolve
is the comparator; the R driver mirrors ferx's controller semantics exactly.
Regenerate with `Rscript vanco_loading_mrgsolve.R`.

1-cpt IV vancomycin. A fixed LOADING bolus rides on the subject (the base regimen);
the controller then titrates DAILY MAINTENANCE boluses on the measured trough.
Params (identical in the ferx model): CL=4 L/h, V=80 L. Loading bolus 1500 mg
at t=0. Controller: pre-dose trough (CENT/V) titrated +/-25% to keep trough
in [10, 15] mg/L; dose_bounds [250, 4000] mg; maintenance start_dose 750 mg;
6 daily decisions (t = 24..144 h).

The loading dose is what makes the day-1 trough non-trivial (the decayed loading
dose, not 0), so a dropped or mis-integrated base regimen would move every trough.

## Realized maintenance dose ladder

| decision | time (h) | trough (mg/L) | maintenance dose (mg) | action |
|---:|---:|---:|---:|:---|
| 0 | 24 | 5.647391473 | 937.500000000 | increase |
| 1 | 48 | 5.230581295 | 1171.875000000 | increase |
| 2 | 72 | 5.987445399 | 1464.843750000 | increase |
| 3 | 96 | 7.318414634 | 1831.054687500 | increase |
| 4 | 120 | 9.098052548 | 2288.818359375 | increase |
| 5 | 144 | 11.357516291 | 2288.818359375 | reissue |

## Summary

- loading_dose = 1500 mg (pre-scheduled base regimen, not in the reactive ledger)
- cumulative_maintenance_dose = 9982.91 mg
- n_increases = 4, n_decreases = 0 (realized maintenance step-ups/-downs)

ferx (`tests/adaptive_vanco_loading_anchor.rs`), run at a supplied eta = 0 (the
typical-value patient simulated here), reproduces this maintenance ladder
dose-for-dose and the troughs to ~1e-9 mg/L (#2580).
The loading dose is integrated by ferx's base-regimen path (#702) and the controller
titrates on top of it -- exactly what this reference exercises.

## Per-window AUC24

The area under CENT/V over each closed inter-decision window, from an AUC compartment (dxdt_AUC = CENT/V) reset at the start of each window (mrgsolve 2.0.1). It is the external reference for `auc_target` (#943).

| window | days [t0,t1] (h) | AUC24 (mg*h/L) |
|---:|---:|---:|
| 0 | [24, 48] | 242.711203568 |
| 1 | [48, 72] | 277.831467910 |
| 2 | [72, 96] | 339.591552808 |
| 3 | [96, 120] | 422.170913604 |
| 4 | [120, 144] | 527.015314969 |
