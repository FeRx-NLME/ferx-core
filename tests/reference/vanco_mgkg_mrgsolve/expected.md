# Weight-based (mg/kg) vancomycin reference (mrgsolve)

Frozen output of `vanco_mgkg_mrgsolve.R` (mrgsolve 2.0.1). External anchor
for per-subject dose expressions (#944): `start_dose = 15 * WT` and
`dose_bounds = [0, min(30 * WT, 4000)]`, evaluated once per subject. NONMEM has
no feedback dosing, so mrgsolve is the comparator. Regenerate with
`Rscript vanco_mgkg_mrgsolve.R`.

1-cpt IV vancomycin, 1-h infusions, CL = 6 * WT / 70 L/h, V = 0.7 * WT L.
Decisions every 12 h, 0..144 h. Rules (first match, confirm = 2): trough < 15
increase 25%; trough > 25 decrease 25%. `amt` is the dose issued at the
decision; `fired` names the rule that acted; `margin` is the trough's distance to
the nearer threshold.

## Subject 1 (WT = 50)

| decision | time (h) | trough (mg/L) | fired | amt (mg) | margin |
|---:|---:|---:|:---|---:|---:|
| 0 | 0 | 0.000000000 |  | 750.000000000 |  |
| 1 | 12 | 5.244534786 | increase | 937.500000000 | 9.755465214 |
| 2 | 24 | 7.762259191 |  | 937.500000000 | 7.237740809 |
| 3 | 36 | 8.341502727 | increase | 1171.875000000 | 6.658497273 |
| 4 | 48 | 10.113684264 |  | 1171.875000000 | 4.886315736 |
| 5 | 60 | 10.521403511 | increase | 1464.843750000 | 4.478596489 |
| 6 | 72 | 12.663852371 |  | 1464.843750000 | 2.336147629 |
| 7 | 84 | 13.156757651 | increase | 1500.000000000 | 1.843242349 |
| 8 | 96 | 13.515996112 |  | 1500.000000000 | 1.484003888 |
| 9 | 108 | 13.598644776 | increase | 1500.000000000 | 1.401355224 |
| 10 | 120 | 13.617659448 |  | 1500.000000000 | 1.382340552 |
| 11 | 132 | 13.622034083 | increase | 1500.000000000 | 1.377965917 |
| 12 | 144 | 13.623040539 |  | 1500.000000000 | 1.376959461 |

## Subject 2 (WT = 150)

| decision | time (h) | trough (mg/L) | fired | amt (mg) | margin |
|---:|---:|---:|:---|---:|---:|
| 0 | 0 | 0.000000000 |  | 2250.000000000 |  |
| 1 | 12 | 5.244534786 | increase | 2812.500000000 | 9.755465214 |
| 2 | 24 | 7.762259191 |  | 2812.500000000 | 7.237740809 |
| 3 | 36 | 8.341502727 | increase | 3515.625000000 | 6.658497273 |
| 4 | 48 | 10.113684264 |  | 3515.625000000 | 4.886315736 |
| 5 | 60 | 10.521403511 | increase | 4000.000000000 | 4.478596489 |
| 6 | 72 | 11.744237764 |  | 4000.000000000 | 3.255762236 |
| 7 | 84 | 12.025570715 | increase | 4000.000000000 | 2.974429285 |
| 8 | 96 | 12.090295945 |  | 4000.000000000 | 2.909704055 |
| 9 | 108 | 12.105187040 | increase | 4000.000000000 | 2.894812960 |
| 10 | 120 | 12.108612979 |  | 4000.000000000 | 2.891387021 |
| 11 | 132 | 12.109401172 | increase | 4000.000000000 | 2.890598828 |
| 12 | 144 | 12.109582508 |  | 4000.000000000 | 2.890417492 |

Smallest margin after t = 0: 1.376959461 mg/L.
