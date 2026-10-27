$PROBLEM Two level blocks, hand-coded sum-to-zero, no intercept (issue #1963)
; E0 = TVE0*(1 + 0.01*TIME^2) + B2[OCC] + PLACEBO[STUDY] + ETA(1)
; Y  = E0 + TVEMAX*TIME/(TIME + TVET50) + EPS(1)
;
; The ferx twin is level_two_block_fit.ferx: `theta B2[OCC]` and
; `theta PLACEBO[STUDY]` as level blocks, both resolving to sum-to-zero. Here
; both are written out as THETAs of that coding: the last level of each block is
; minus the sum of the others, in the order ferx sums them. One study per
; subject, so PLACEBO's levels are the study random effect's up to the levels'
; mean, which nothing else carries (TVE0 is read through a factor of TIME).
;
; Evaluated, not fitted: MAXEVAL=0 at the initial estimates below, FOCE-I, with
; POSTHOC. Data: level_two_block.csv (simulate_level_two_block.py, seed 1963).
$INPUT ID STUDY OCC TIME DV MDV
$DATA level_two_block.csv IGNORE=@

$PRED
B = -(THETA(4) + THETA(5))
IF (OCC.EQ.1) B = THETA(4)
IF (OCC.EQ.2) B = THETA(5)
P = -((((THETA(6) + THETA(7)) + THETA(8)) + THETA(9)) + THETA(10))
IF (STUDY.EQ.1) P = THETA(6)
IF (STUDY.EQ.2) P = THETA(7)
IF (STUDY.EQ.3) P = THETA(8)
IF (STUDY.EQ.4) P = THETA(9)
IF (STUDY.EQ.5) P = THETA(10)
E0 = THETA(1)*(1 + 0.01*TIME*TIME) + B + P + ETA(1)
IPRED = E0 + THETA(2)*TIME/(TIME + THETA(3))
Y = IPRED + EPS(1)

$THETA (-10, 1.4, 10)    ; TVE0
$THETA (0.1, 2.8, 20)    ; TVEMAX
$THETA (0.1, 1.6, 20)    ; TVET50
$THETA (-10, 0.25, 10)   ; B2[OCC=1]
$THETA (-10, -0.15, 10)  ; B2[OCC=2]
$THETA (-10, 0.35, 10)   ; PLACEBO[STUDY=1]
$THETA (-10, -0.25, 10)  ; PLACEBO[STUDY=2]
$THETA (-10, 0.15, 10)   ; PLACEBO[STUDY=3]
$THETA (-10, 0.05, 10)   ; PLACEBO[STUDY=4]
$THETA (-10, -0.40, 10)  ; PLACEBO[STUDY=5]
$OMEGA 0.1                ; ETA_E0
$SIGMA 0.25               ; additive variance, ferx SD 0.5

$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC PRINT=1 FORMAT=s1PE23.16
$TABLE ID STUDY OCC TIME DV PRED IPRED ETA1 NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=level_two_block.tab
