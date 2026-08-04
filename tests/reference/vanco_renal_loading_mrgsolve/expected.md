# Vancomycin renal-decline LOADING + maintenance-titration reference (mrgsolve)

Frozen output of `vanco_renal_loading_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the reactive `[adaptive_dosing]` **pre-scheduled base regimen under a time-varying
covariate** (#930) in `examples/adaptive_vanco_renal_loading.ferx`. NONMEM has no feedback
dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller semantics
exactly and feeds CL per day from the covariate at the segment's end (the NONMEM
end-of-interval convention ferx's per-segment PK uses). Regenerate with
`Rscript vanco_renal_loading_mrgsolve.R`.

1-cpt IV vancomycin, bolus dosing. A fixed LOADING bolus rides on the subject (the base
regimen); the controller then titrates DAILY MAINTENANCE boluses on the measured trough,
while renal function declines. Params (identical in the ferx model):
TVCL=4 L/h at CRCL=100, V=80 L, CL = TVCL * CRCL / 100. Loading bolus 1500 mg at t=0.
Renal function declines CRCL 120 -> 41 mL/min, so CL falls 4.8 -> 1.64 L/h and drug
accumulates.
Controller: pre-dose trough (CENT/V) titrated +/-25% to keep trough in [10, 15]
mg/L; dose_bounds [250, 4000] mg; maintenance start_dose 750 mg; 8 daily decisions
(t = 24..192 h).

The loading dose decayed under the DECLINING clearance is what makes the day-1 trough
non-trivial (a dropped base regimen, or one frozen at the t=0 covariate, would move every
trough). This is the base-regimen path (#702) combined with per-event covariate PK (#700).

## Realized maintenance dose ladder

| decision | time (h) | CRCL (mL/min) | trough (mg/L) | maintenance dose (mg) | action |
|---:|---:|---:|---:|---:|:---|
| 0 | 24 | 100.000000 | 5.647391 | 937.500000 | increase |
| 1 | 48 | 85.000000 | 6.262143 | 1171.875000 | increase |
| 2 | 72 | 72.000000 | 8.813241 | 1464.843800 | increase |
| 3 | 96 | 62.000000 | 12.889476 | 1464.843800 | reissue |
| 4 | 120 | 54.000000 | 16.320448 | 1098.632800 | decrease |
| 5 | 144 | 48.000000 | 16.894268 | 823.974600 | decrease |
| 6 | 168 | 44.000000 | 16.038540 | 617.981000 | decrease |
| 7 | 192 | 41.000000 | 14.528939 | 617.981000 | reissue |

## Summary

- loading_dose = 1500 mg (pre-scheduled base regimen, not in the reactive ledger)
- cumulative_maintenance_dose = 8197.63 mg
- increase rule fired 3 times, decrease rule fired 3 times
- n_increases = 2, n_decreases = 3 (realized maintenance step-ups/-downs)
- trough_in_window = 2/8 decisions in [10, 15] mg/L

ferx (`tests/adaptive_vanco_renal_loading_anchor.rs`) reproduces this maintenance ladder
dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs LSODA). The
loading dose is integrated by ferx's base-regimen path under the declining per-segment
clearance (#930), and the controller titrates on top of it -- exactly what this reference
exercises.
