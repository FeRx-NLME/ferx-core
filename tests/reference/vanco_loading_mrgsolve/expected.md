# Vancomycin loading-dose + maintenance-titration reference (mrgsolve)

Frozen output of `vanco_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
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
| 0 | 24 | 5.647391 | 937.500000 | increase |
| 1 | 48 | 5.230581 | 1171.875000 | increase |
| 2 | 72 | 5.987445 | 1464.844000 | increase |
| 3 | 96 | 7.318415 | 1831.055000 | increase |
| 4 | 120 | 9.098053 | 2288.818000 | increase |
| 5 | 144 | 11.357516 | 2288.818000 | reissue |

## Summary

- loading_dose = 1500 mg (pre-scheduled base regimen, not in the reactive ledger)
- cumulative_maintenance_dose = 9982.91 mg
- n_increases = 4, n_decreases = 0 (realized maintenance step-ups/-downs)

ferx (`tests/adaptive_vanco_loading_anchor.rs`) reproduces this maintenance ladder
dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA).
The loading dose is integrated by ferx's base-regimen path (#702) and the controller
titrates on top of it -- exactly what this reference exercises.
