# Vancomycin renal-decline TDM titration reference (mrgsolve)

Frozen output of `vanco_renal_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the reactive `[adaptive_dosing]` *continuous* (percentage) titration with a
**time-varying covariate driving PK** (#700) in
`examples/adaptive_vanco_renal.ferx`. NONMEM has no feedback dosing, so mrgsolve
is the comparator; the R driver mirrors ferx's controller semantics exactly and
feeds CL per window from the covariate at the window's end (the NONMEM
end-of-interval convention ferx's per-segment PK uses). Regenerate with
`Rscript vanco_renal_mrgsolve.R`.

1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx
model): TVCL=4 L/h at CRCL=100, V=80 L, CL = TVCL * CRCL / 100.
Renal function declines over the horizon (a realistic AKI trajectory):
CRCL 120 -> 40 mL/min, so CL falls 4.8 -> 1.6 L/h and drug accumulates.
Controller: pre-dose trough (CENT/V) titrated +/-25% to keep trough in
[10, 15] mg/L; dose_bounds [250, 4000] mg; start_dose 500 mg; 14 daily
decisions (t = 0..312 h).

`auc_target` is intentionally absent: its exposure metric integrates a dense
grid from a single frozen PK snapshot, which would be silently wrong under a
changing covariate, so it is a typed error for a time-varying-covariate subject
(#700). The pct-in-window trough metric (`target_window`) is per-event
covariate-aware and retained.

## Realized dose ladder

| decision | time (h) | CRCL (mL/min) | trough (mg/L) | dose (mg) | action |
|---:|---:|---:|---:|---:|:---|
| 0 | 0 | 120.000000 | 0.000000 | 625.000000 | increase |
| 1 | 24 | 115.000000 | 2.020561 | 781.250000 | increase |
| 2 | 48 | 108.000000 | 3.291760 | 976.562500 | increase |
| 3 | 72 | 98.000000 | 4.860710 | 1220.703100 | increase |
| 4 | 96 | 88.000000 | 7.095079 | 1525.878900 | increase |
| 5 | 120 | 78.000000 | 10.378132 | 1525.878900 | reissue |
| 6 | 144 | 68.000000 | 13.124256 | 1525.878900 | reissue |
| 7 | 168 | 60.000000 | 15.768755 | 1144.409200 | decrease |
| 8 | 192 | 54.000000 | 15.797279 | 858.306900 | decrease |
| 9 | 216 | 49.000000 | 14.777775 | 858.306900 | reissue |
| 10 | 240 | 45.000000 | 14.911299 | 858.306900 | reissue |
| 11 | 264 | 42.000000 | 15.539585 | 643.730200 | decrease |
| 12 | 288 | 41.000000 | 14.465452 | 643.730200 | reissue |
| 13 | 312 | 40.000000 | 13.974498 | 643.730200 | reissue |

## Summary metrics

- cumulative_dose = 13831.7 mg
- increase rule fired 5 times, decrease rule fired 3 times
- n_increases = 4, n_decreases = 3 (realized dose step-ups/-downs; the
  increase rule fired 5 times, but decision 0 steps off the un-realized
  start_dose, so there is one fewer realized step-up -- ferx counts dose changes)
- trough_in_window = 6/14 decisions in [10, 15] mg/L

ferx (`tests/adaptive_vanco_renal_anchor.rs`) reproduces this ladder
dose-for-dose and the troughs to a small cross-solver tolerance (RK45 vs
LSODA). Both engines apply the covariate per record with the same
end-of-interval convention, so the piecewise-constant CL — hence the whole
trough trajectory and every dose decision — agrees.
