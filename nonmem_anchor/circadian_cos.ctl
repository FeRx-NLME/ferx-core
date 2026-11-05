$PROBLEM Circadian baseline with a phase random effect inside COS and SIN (issue #2720)
; CIRC = AMP1*COS(2*PI*(TIME+SHIFT)/24 - PHS1) + AMP2*SIN(4*PI*(TIME+SHIFT)/24 - PHS2)
; Y    = BASE*(1 + CIRC)*(1 + EPS(1))
; BASE = THETA(1)*EXP(ETA(1)), AMP1 = THETA(2)*EXP(ETA(2)), SHIFT = ETA(3) (hours)
;
; The ferx twin is circadian_cos_fit.ferx, which spells the trigonometric terms
; with the DSL's `cos(...)` / `sin(...)`. ETA(3) sits inside both arguments, so
; eta-hat and the FOCE-I objective depend on d/dETA(3) through COS and SIN.
;
; Evaluated, not fitted: MAXEVAL=0 at the initial estimates below (off the
; simulation truth), FOCE-I, with POSTHOC. Data: circadian_cos.csv
; (simulate_circadian_cos.py, seed 2720). circadian_cos_est.ctl fits the same
; model to convergence.
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

$THETA (1, 110, 1000)   ; TVBASE (truth 120)
$THETA (0.001, 0.10, 1) ; TVAMP1 (truth 0.12)
$THETA (-5, 0.8, 5)     ; PHS1 [rad] (truth 1.0)
$THETA (0.001, 0.05, 1) ; TVAMP2 (truth 0.04)
$THETA (-5, 0.6, 5)     ; PHS2 [rad] (truth 0.5)
$OMEGA 0.03             ; ETA_BASE (truth 0.02)
$OMEGA 0.08             ; ETA_AMP1 (truth 0.10)
$OMEGA 3.0              ; ETA_SHIFT [h^2] (truth 4.0)
$SIGMA 0.0009           ; proportional variance, ferx SD 0.03

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC
$TABLE ID TIME DV PRED IPRED NOPRINT ONEHEADER NOAPPEND FORMAT=s1PE23.16 FILE=circadian_cos.tab
