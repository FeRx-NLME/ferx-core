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
| 0 | 0 | 0.000000000 | dosed | 12 | 9.000000000 |
| 1 | 12 | 11.467087642 | dosed | 12 | 2.467087642 |
| 2 | 24 | 21.947216641 | dosed | 24 | 3.052783359 |
| 3 | 36 | 31.525333360 | not_due | 24 |  |
| 4 | 48 | 28.811985281 | hold | 36 | 3.811985281 |
| 5 | 60 | 26.332171855 | not_due | 36 |  |
| 6 | 72 | 24.065793032 | not_due | 36 |  |
| 7 | 84 | 21.994478748 | dosed | 48 | 3.005521252 |
| 8 | 96 | 31.568527673 | not_due | 48 |  |
| 9 | 108 | 28.851461911 | not_due | 48 |  |
| 10 | 120 | 26.368250779 | not_due | 48 |  |
| 11 | 132 | 24.098766685 | dosed | 48 | 0.901233315 |
| 12 | 144 | 33.491702042 | not_due | 48 |  |
| 13 | 156 | 30.609110941 | not_due | 48 |  |
| 14 | 168 | 27.974621040 | not_due | 48 |  |
| 15 | 180 | 25.566878562 | hold | 48 | 0.566878562 |
| 16 | 192 | 23.366367625 | not_due | 48 |  |
| 17 | 204 | 21.355252057 | not_due | 48 |  |
| 18 | 216 | 19.517230823 | not_due | 48 |  |
| 19 | 228 | 17.837405897 | dosed | 48 | 3.837405897 |
| 20 | 240 | 27.769249155 | not_due | 48 |  |
| 21 | 252 | 25.379182792 | not_due | 48 |  |
| 22 | 264 | 23.194826608 | not_due | 48 |  |
| 23 | 276 | 21.198475372 | dosed | 48 | 3.801524628 |
| 24 | 288 | 30.841035364 | not_due | 48 |  |

## Subject 2 (RF = 0.5)

| decision | time (h) | trough (mg/L) | outcome | interval after (h) | margin |
|---:|---:|---:|:---|---:|---:|
| 0 | 0 | 0.000000000 | dosed | 12 | 9.000000000 |
| 1 | 12 | 9.376951271 | dosed | 12 | 0.376951271 |
| 2 | 24 | 16.323567627 | dosed | 24 | 2.323567627 |
| 3 | 36 | 21.469747595 | not_due | 24 |  |
| 4 | 48 | 15.905180210 | dosed | 36 | 1.905180210 |
| 5 | 60 | 21.159798573 | not_due | 36 |  |
| 6 | 72 | 15.675564327 | not_due | 36 |  |
| 7 | 84 | 11.612743671 | dosed | 36 | 2.387256329 |
| 8 | 96 | 17.979883374 | not_due | 36 |  |
| 9 | 108 | 13.319825208 | not_due | 36 |  |
| 10 | 120 | 9.867569209 | dosed | 36 | 0.867569209 |
| 11 | 132 | 16.687026334 | not_due | 36 |  |
| 12 | 144 | 12.362053156 | not_due | 36 |  |
| 13 | 156 | 9.158034222 | dosed | 36 | 0.158034222 |
| 14 | 168 | 16.161389888 | not_due | 36 |  |
| 15 | 180 | 11.972652099 | not_due | 36 |  |
| 16 | 192 | 8.869558824 | dosed | 24 | 0.130441176 |
| 17 | 204 | 15.947682056 | not_due | 24 |  |
| 18 | 216 | 11.814333444 | dosed | 24 | 2.185666556 |
| 19 | 228 | 18.129224751 | not_due | 24 |  |
| 20 | 240 | 13.430460020 | dosed | 24 | 0.569539980 |
| 21 | 252 | 19.326480766 | not_due | 24 |  |
| 22 | 264 | 14.317409091 | dosed | 36 | 0.317409091 |
| 23 | 276 | 19.983548798 | not_due | 36 |  |
| 24 | 288 | 14.804177062 | not_due | 36 |  |

Smallest margin at a consulted decision: 0.130441176 mg/L.
