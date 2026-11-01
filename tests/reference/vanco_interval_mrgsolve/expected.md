# Vancomycin interval-extension reference (mrgsolve)

Frozen output of `vanco_interval_mrgsolve.R` (mrgsolve 2.0.1). External
anchor for schedule adaptation (#945): a fixed review grid, a due mask, and an
interval ladder moved by the controller. NONMEM has no feedback dosing, so
mrgsolve is the comparator. Regenerate with `Rscript vanco_interval_mrgsolve.R`.

1-cpt IV vancomycin, 1000 mg by 1-h infusion, CL = 4 * RF L/h, V = 80 L.
Review grid every 12 h, 0..288 h. Interval ladder 12/24/36/48 h, start 12 h.
Due-point rules (first match): trough > 25 hold + lengthen; > 14 dose +
lengthen; < 9 dose + shorten; else dose. `margin` is the trough's distance to
the nearest threshold at a consulted decision.

## Subject 1 (RF = 0.15)

| decision | time (h) | trough (mg/L) | outcome | interval after (h) | margin |
|---:|---:|---:|:---|---:|---:|
| 0 | 0 | 0.000000 | dosed | 12 | 9.000000 |
| 1 | 12 | 11.467088 | dosed | 12 | 2.467088 |
| 2 | 24 | 21.947217 | dosed | 24 | 3.052783 |
| 3 | 36 | 31.525333 | not_due | 24 |  |
| 4 | 48 | 28.811985 | hold | 36 | 3.811985 |
| 5 | 60 | 26.332172 | not_due | 36 |  |
| 6 | 72 | 24.065793 | not_due | 36 |  |
| 7 | 84 | 21.994479 | dosed | 48 | 3.005521 |
| 8 | 96 | 31.568528 | not_due | 48 |  |
| 9 | 108 | 28.851462 | not_due | 48 |  |
| 10 | 120 | 26.368251 | not_due | 48 |  |
| 11 | 132 | 24.098767 | dosed | 48 | 0.901233 |
| 12 | 144 | 33.491702 | not_due | 48 |  |
| 13 | 156 | 30.609111 | not_due | 48 |  |
| 14 | 168 | 27.974621 | not_due | 48 |  |
| 15 | 180 | 25.566879 | hold | 48 | 0.566879 |
| 16 | 192 | 23.366368 | not_due | 48 |  |
| 17 | 204 | 21.355252 | not_due | 48 |  |
| 18 | 216 | 19.517231 | not_due | 48 |  |
| 19 | 228 | 17.837406 | dosed | 48 | 3.837406 |
| 20 | 240 | 27.769249 | not_due | 48 |  |
| 21 | 252 | 25.379183 | not_due | 48 |  |
| 22 | 264 | 23.194827 | not_due | 48 |  |
| 23 | 276 | 21.198475 | dosed | 48 | 3.801525 |
| 24 | 288 | 30.841035 | not_due | 48 |  |

## Subject 2 (RF = 0.5)

| decision | time (h) | trough (mg/L) | outcome | interval after (h) | margin |
|---:|---:|---:|:---|---:|---:|
| 0 | 0 | 0.000000 | dosed | 12 | 9.000000 |
| 1 | 12 | 9.376951 | dosed | 12 | 0.376951 |
| 2 | 24 | 16.323568 | dosed | 24 | 2.323568 |
| 3 | 36 | 21.469748 | not_due | 24 |  |
| 4 | 48 | 15.905180 | dosed | 36 | 1.905180 |
| 5 | 60 | 21.159799 | not_due | 36 |  |
| 6 | 72 | 15.675564 | not_due | 36 |  |
| 7 | 84 | 11.612744 | dosed | 36 | 2.387256 |
| 8 | 96 | 17.979883 | not_due | 36 |  |
| 9 | 108 | 13.319825 | not_due | 36 |  |
| 10 | 120 | 9.867569 | dosed | 36 | 0.867569 |
| 11 | 132 | 16.687026 | not_due | 36 |  |
| 12 | 144 | 12.362053 | not_due | 36 |  |
| 13 | 156 | 9.158034 | dosed | 36 | 0.158034 |
| 14 | 168 | 16.161390 | not_due | 36 |  |
| 15 | 180 | 11.972652 | not_due | 36 |  |
| 16 | 192 | 8.869559 | dosed | 24 | 0.130441 |
| 17 | 204 | 15.947682 | not_due | 24 |  |
| 18 | 216 | 11.814333 | dosed | 24 | 2.185667 |
| 19 | 228 | 18.129225 | not_due | 24 |  |
| 20 | 240 | 13.430460 | dosed | 24 | 0.569540 |
| 21 | 252 | 19.326481 | not_due | 24 |  |
| 22 | 264 | 14.317409 | dosed | 36 | 0.317409 |
| 23 | 276 | 19.983549 | not_due | 36 |  |
| 24 | 288 | 14.804177 | not_due | 36 |  |

Smallest margin at a consulted decision: 0.130441 mg/L.
