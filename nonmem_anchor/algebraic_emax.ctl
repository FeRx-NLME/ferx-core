$PROBLEM Emax time-course, no compartments - the $PRED analogue of a ferx
; compartment-free [structural_model] (issue #811).
;
; y = E0*exp(eta) - EMAX*t/(ET50 + t) + eps,  additive residual error.
; No doses, no ADVAN, no compartments: $PRED is exactly what a ferx
; `[structural_model]` with `y = <expr>` and no `pk`/`ode` line compiles to.
;
; Data simulated from THETA = (10, 6, 2), OMEGA = 0.09, SIGMA = 0.25 by
; `simulate_algebraic_emax.py` (seeded, 30 subjects x 8 times). The ferx twin is
; the shipped example `examples/emax_timecourse.ferx` + `data/emax_timecourse.csv`
; (a byte-identical copy of this dataset, pinned by the anchor test).
$INPUT ID TIME DV MDV
$DATA algebraic_emax.csv IGNORE=@

$PRED
E0   = THETA(1)*EXP(ETA(1))
EMAX = THETA(2)
ET50 = THETA(3)
Y    = E0 - EMAX*TIME/(ET50 + TIME) + EPS(1)

$THETA (0.1, 8.0,  100.0)   ; TVE0   - deliberately off the truth (10)
$THETA (0.1, 4.0,  100.0)   ; TVEMAX - deliberately off the truth (6)
$THETA (0.01, 1.0, 100.0)   ; TVET50 - deliberately off the truth (2)
$OMEGA 0.04                 ; IIV on the baseline (truth 0.09)
$SIGMA 1.0                  ; additive residual variance (truth 0.25)

$ESTIMATION METHOD=1 MAXEVAL=9999 PRINT=5 NOABORT SIGDIGITS=4
$COVARIANCE MATRIX=R UNCONDITIONAL
$TABLE ID TIME DV PRED CWRES NOPRINT ONEHEADER FILE=algebraic_emax.tab
