$PROBLEM Circadian baseline with a phase random effect inside COS and SIN (issue #2720)
; CIRC = AMP1*COS(2*PI*(TIME+SHIFT)/24 - PHS1) + AMP2*SIN(4*PI*(TIME+SHIFT)/24 - PHS2)
; Y    = BASE*(1 + CIRC)*(1 + EPS(1))
; BASE = THETA(1)*EXP(ETA(1)), AMP1 = THETA(2)*EXP(ETA(2)), SHIFT = ETA(3) (hours)
;
; The ferx twin is circadian_cos_fit.ferx, which spells the trigonometric terms
; with the DSL's `cos(...)` / `sin(...)`. ETA(3) sits inside both arguments, so
; eta-hat and the FOCE-I objective depend on d/dETA(3) through COS and SIN.
;
; The R-matrix covariance step alone, at the optimum circadian_cos_est.ctl
; reached (its estimates to the 6-7 digits ferx prints), MAXEVAL=0, with the
; objective and the $COV finite differences run tight (SIGL=14 / SIGL=12 TOL=12).
; At the default precision circadian_cos_est.ctl's own $COV put the SEs of
; PHS1/PHS2 ~30% (and of OMEGA(1,1)/SIGMA 14-18%) above this run's, which is
; within ~3% of ferx's exact R-matrix and its FD Hessian on every element.
; Data: circadian_cos.csv (simulate_circadian_cos.py, seed 2720).
$INPUT ID TIME DV MDV
$DATA circadian_cos.csv IGNORE=@

$PRED
PI    = 3.141592653589793
BASE  = THETA(1)*EXP(ETA(1))
AMP1  = THETA(2)*EXP(ETA(2))
PHS1  = THETA(3)
AMP2  = THETA(4)
PHS2  = THETA(5)
SHIFT = ETA(3)
CIRC  = AMP1*COS(2*PI*(TIME + SHIFT)/24 - PHS1) + AMP2*SIN(4*PI*(TIME + SHIFT)/24 - PHS2)
IPRED = BASE*(1 + CIRC)
Y     = IPRED*(1 + EPS(1))

$THETA (1, 118.978735, 1000)
$THETA (0.001, 0.122026, 1)
$THETA (-5, 0.941426, 5)
$THETA (0.001, 0.039702, 1)
$THETA (-5, 0.440610, 5)
$OMEGA 0.021898
$OMEGA 0.086498
$OMEGA 4.831540
$SIGMA 0.000807688

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 SIGL=14 POSTHOC
$COVARIANCE MATRIX=R UNCONDITIONAL SIGL=12 TOL=12
