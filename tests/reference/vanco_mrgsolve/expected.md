# Vancomycin AUC-TDM titration reference (mrgsolve)

Frozen output of `vanco_mrgsolve.R` (mrgsolve 2.0.1). External anchor for the
reactive `[adaptive_dosing]` *continuous* (percentage) titration and the
metrics-only AUC machinery (`auc_target` -> `auc_target_attainment`) in
`examples/adaptive_vanco_auc.ferx` (epic #391 S2.5b). NONMEM has no feedback
dosing, so mrgsolve is the comparator; the R driver mirrors ferx's controller
semantics exactly. Regenerate with `Rscript vanco_mrgsolve.R`.

1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx
model): CL=4 L/h, V=80 L. Controller: pre-dose trough (CENT/V) titrated
+/-25% to keep trough in [10, 15] mg/L; dose_bounds [250, 4000] mg;
start_dose 500 mg; 14 daily decisions (t = 0..312 h).

The reported outcome is AUC24 target attainment, band [400, 600] mg*h/L
(metric only -- it never influences dosing).

## Realized dose ladder

| decision | time (h) | trough (mg/L) | dose (mg) | action |
|---:|---:|---:|---:|:---|
| 0 | 0 | 0.000000000 | 625.000000000 | increase |
| 1 | 24 | 2.412899604 | 781.250000000 | increase |
| 2 | 48 | 3.742875900 | 976.562500000 | increase |
| 3 | 72 | 4.897488188 | 1220.703125000 | increase |
| 4 | 96 | 6.187789635 | 1525.878906250 | increase |
| 5 | 120 | 7.754594597 | 1907.348632812 | increase |
| 6 | 144 | 9.699224226 | 2384.185791016 | increase |
| 7 | 168 | 12.125831719 | 2384.185791016 | reissue |
| 8 | 192 | 12.856711850 | 2384.185791016 | reissue |
| 9 | 216 | 13.076848715 | 2384.185791016 | reissue |
| 10 | 240 | 13.143152665 | 2384.185791016 | reissue |
| 11 | 264 | 13.163123031 | 2384.185791016 | reissue |
| 12 | 288 | 13.169137990 | 2384.185791016 | reissue |
| 13 | 312 | 13.170949660 | 2384.185791016 | reissue |

## Per-window AUC24 and attainment

| window | days [t0,t1] (h) | AUC24 (mg*h/L) | in [400,600] |
|---:|---:|---:|:--:|
| 0 | [0, 24] | 107.992007917 | no |
| 1 | [24, 48] | 168.712974085 | no |
| 2 | [48, 72] | 221.048379232 | no |
| 3 | [72, 96] | 279.369752324 | no |
| 4 | [96, 120] | 350.133627321 | no |
| 5 | [120, 144] | 437.944565617 | yes |
| 6 | [144, 168] | 547.514297892 | yes |
| 7 | [168, 192] | 581.428845133 | yes |
| 8 | [192, 216] | 591.643710453 | yes |
| 9 | [216, 240] | 594.720368765 | yes |
| 10 | [240, 264] | 595.647040435 | yes |
| 11 | [264, 288] | 595.926148572 | yes |
| 12 | [288, 312] | 596.010214337 | yes |

## Summary metrics

- cumulative_dose = 26110.2 mg
- n_increases = 6, n_decreases = 0 (realized dose step-ups/-downs; the
  increase rule fired 7 times, but decision 0 steps off the un-realized
  start_dose, so there are one fewer realized step-ups -- ferx counts dose changes)
- auc_target_attainment = 8/13 = 0.6153846154

ferx (`tests/adaptive_vanco_anchor.rs`), run at a supplied eta = 0 (the
typical-value patient simulated here), reproduces this ladder dose-for-dose,
the troughs to ~1e-9 mg/L (this table's own error: ferx sits on the exact
1-cpt closed form, #2582), and the AUC-target attainment fraction exactly.
ferx integrates the exposure by a 128-panel trapezoid per window (vs this AUC
compartment), which agrees to ~3e-6 relative -- far inside the margin from
each AUC24 to the band edges, so the in/out classification (hence
attainment) is identical. The model's omega ~ 1e-10 would otherwise draw an
eta of order 1e-5 on CL and offset every trough one-sidedly (#2580).
