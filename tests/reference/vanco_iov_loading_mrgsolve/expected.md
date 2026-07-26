# Vancomycin per-occasion IOV LOADING + maintenance-titration reference (mrgsolve)

Frozen output of `vanco_iov_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the reactive `[adaptive_dosing]` **pre-scheduled base regimen under inter-occasion
variability** (#931) in `examples/adaptive_vanco_iov_loading.ferx`. NONMEM has no feedback
dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller semantics
exactly and feeds CL per day from the occasion at the segment's end (the NONMEM
end-of-interval convention ferx's per-segment PK uses). Regenerate with
`Rscript vanco_iov_loading_mrgsolve.R`.

1-cpt IV vancomycin, bolus dosing. A fixed LOADING bolus rides on the subject (the base
regimen); the controller then titrates DAILY MAINTENANCE boluses on the measured trough,
while a fresh per-occasion clearance is drawn each decision window. Params (identical in
the ferx model):
TVCL=4 L/h at kappa=0, V=80 L, CL = TVCL * exp(eta + kappa). Loading bolus 1500 mg at t=0.
Per-occasion kappa swings CL 1.87919 -> 5.23931 L/h across the horizon (occasions 0..7).
Controller: pre-dose trough (CENT/V) titrated +/-25% to keep trough in [10, 15]
mg/L; dose_bounds [250, 4000] mg; maintenance start_dose 750 mg; 8 daily decisions
(t = 24..192 h).

The loading dose decayed under occasion 0's clearance (the end-of-interval value for the
(0, 24] segment) is what makes the day-1 trough non-trivial (a dropped base regimen, or one
frozen at kappa=0, would move every trough). This is the base-regimen path (#702) combined
with per-occasion IOV PK (#701).

## Realized maintenance dose ladder

| decision | time (h) | CL (L/h) | trough (mg/L) | maintenance dose (mg) | action |
|---:|---:|---:|---:|---:|:---|
| 0 | 24 | 3.359412 | 6.843985 | 937.500000 | increase |
| 1 | 48 | 3.606756 | 6.291058 | 1171.875000 | increase |
| 2 | 72 | 5.239310 | 4.348557 | 1464.844000 | increase |
| 3 | 96 | 2.894896 | 9.507614 | 1831.055000 | increase |
| 4 | 120 | 1.879191 | 18.435378 | 1373.291000 | decrease |
| 5 | 144 | 4.511665 | 9.197104 | 1716.614000 | increase |
| 6 | 168 | 2.914567 | 12.786866 | 1716.614000 | reissue |
| 7 | 192 | 3.506851 | 11.958851 | 1716.614000 | reissue |

## Summary

- loading_dose = 1500 mg (pre-scheduled base regimen, not in the reactive ledger)
- cumulative_maintenance_dose = 11928.4 mg
- increase rule fired 5 times, decrease rule fired 1 times
- n_increases = 4, n_decreases = 1 (realized maintenance step-ups/-downs)
- trough_in_window = 2/8 decisions in [10, 15] mg/L

ferx (`tests/adaptive_vanco_iov_loading_anchor.rs`) reproduces this maintenance ladder
dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA). The
loading dose is integrated by ferx's base-regimen path under the per-occasion clearance
(#931), and the controller titrates on top of it -- exactly what this reference exercises.
