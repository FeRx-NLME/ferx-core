$PROBLEM Level block on the residual SD next to IIV on RUV, within-coded (issue #1879)
; IPRED = TVA*EXP(-K*TIME), K = TVK*EXP(ETA(1))
; Y     = IPRED + IPRED*EXP(L)*EXP(ETA(2))*EPS(1), L = ERRSCALE[STUDY, OCC]
;
; The ferx twin is level_ruv_fit.ferx: `theta ERRSCALE[STUDY, OCC]` read in
; `proportional(PROP * exp(ERRSCALE))` next to `iiv_on_ruv = ETA_RUV`, one study
; per subject. ETA(2) scales every record of its subject alike, which is the
; common shift of the subject's three levels, so the block binds
; sum-to-zero within each study. Here that coding is written out as THETAs: the
; third occasion's level is minus the sum of the first two, in the order ferx
; sums them.
;
; Evaluated, not fitted: MAXEVAL=0 at the initial estimates below, FOCE-I, with
; POSTHOC. Data: level_ruv.csv (simulate_level_ruv.py, seed 1879).
$INPUT ID STUDY OCC TIME DV MDV
$DATA level_ruv.csv IGNORE=@

$PRED
L = 0
IF (STUDY.EQ.1.AND.OCC.EQ.1) L = THETA(3)
IF (STUDY.EQ.1.AND.OCC.EQ.2) L = THETA(4)
IF (STUDY.EQ.1.AND.OCC.EQ.3) L = -(THETA(3) + THETA(4))
IF (STUDY.EQ.2.AND.OCC.EQ.1) L = THETA(5)
IF (STUDY.EQ.2.AND.OCC.EQ.2) L = THETA(6)
IF (STUDY.EQ.2.AND.OCC.EQ.3) L = -(THETA(5) + THETA(6))
IF (STUDY.EQ.3.AND.OCC.EQ.1) L = THETA(7)
IF (STUDY.EQ.3.AND.OCC.EQ.2) L = THETA(8)
IF (STUDY.EQ.3.AND.OCC.EQ.3) L = -(THETA(7) + THETA(8))
IF (STUDY.EQ.4.AND.OCC.EQ.1) L = THETA(9)
IF (STUDY.EQ.4.AND.OCC.EQ.2) L = THETA(10)
IF (STUDY.EQ.4.AND.OCC.EQ.3) L = -(THETA(9) + THETA(10))
K = THETA(2)*EXP(ETA(1))
IPRED = THETA(1)*EXP(-K*TIME)
Y = IPRED + IPRED*EXP(L)*EXP(ETA(2))*EPS(1)

$THETA (0.1, 10.0, 100)   ; TVA
$THETA (0.01, 0.15, 2)    ; TVK
$THETA (-5, 0.30, 5)      ; ERRSCALE[STUDY=1,OCC=1]
$THETA (-5, -0.10, 5)     ; ERRSCALE[STUDY=1,OCC=2]
$THETA (-5, -0.25, 5)     ; ERRSCALE[STUDY=2,OCC=1]
$THETA (-5, 0.40, 5)      ; ERRSCALE[STUDY=2,OCC=2]
$THETA (-5, 0.20, 5)      ; ERRSCALE[STUDY=3,OCC=1]
$THETA (-5, 0.10, 5)      ; ERRSCALE[STUDY=3,OCC=2]
$THETA (-5, 0.50, 5)      ; ERRSCALE[STUDY=4,OCC=1]
$THETA (-5, -0.05, 5)     ; ERRSCALE[STUDY=4,OCC=2]
$OMEGA 0.09               ; ETA_K
$OMEGA 0.16               ; ETA_RUV
$SIGMA 0.01               ; proportional variance, ferx SD 0.1

$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC PRINT=1 FORMAT=s1PE23.16
$TABLE ID STUDY OCC TIME DV PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=level_ruv.tab
