# Platelet-ladder dose-modification reference (mrgsolve)

Frozen output of `platelet_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the reactive `[adaptive_dosing]` `levels` ladder in
`examples/adaptive_platelet_ladder.ferx` (epic #391 S2.5a). NONMEM has no
feedback dosing, so mrgsolve is the comparator; the R driver mirrors ferx's
controller semantics exactly. Regenerate with `Rscript platelet_mrgsolve.R`.

Friberg myelosuppression (n=3 transit) + 1-cpt IV drug. Params (identical in
the ferx model): CL=0.22, V=15, CIRC0=250, MTT=120, GAMMA=0.20, SLOPE=0.06.
Ladder levels [25,50,75,100] mg, start_dose 100; weekly decisions t=0..1512 h;
rules: PLT<30 -> stop, PLT<120 -> decrease one level.

## Realized dose ladder

| decision | time (h) | PLT (x10^9/L) | dose (mg) | action |
|---:|---:|---:|---:|:---|
| 0 | 0 | 250.000000 | 100 | continue |
| 1 | 168 | 169.562700 | 100 | continue |
| 2 | 336 | 94.266960 | 75 | decrease |
| 3 | 504 | 98.687490 | 50 | decrease |
| 4 | 672 | 151.280660 | 50 | continue |
| 5 | 840 | 178.503940 | 50 | continue |
| 6 | 1008 | 166.771510 | 50 | continue |
| 7 | 1176 | 158.404870 | 50 | continue |
| 8 | 1344 | 160.123630 | 50 | continue |
| 9 | 1512 | 162.540060 | 50 | continue |

## Summary metrics

- cumulative_dose = 625 mg
- n_doses = 10
- n_decreases = 2 (100 -> 75 -> 50, then platelets recover and hold)
- discontinued = false
- pct_time_in_window (PLT >= 100) = 0.8

ferx (`tests/adaptive_platelet_anchor.rs`, RK45) reproduces this ladder
exactly and the platelet signals to < 0.01 (x10^9/L) -- a cross-solver
difference ~3e-5 relative, far below the ~25-unit margin to either rule
threshold, so every dose decision is robust.
