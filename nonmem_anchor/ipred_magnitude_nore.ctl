$PROBLEM ferx #3121 anchor, no random effects (IPRED = PRED): SD = A*EXP(B*IPRED)/SQRT(NIND)
; The arm-level form an MBMA summary-data model needs: no ETA, so the objective is the exact
; extended-least-squares likelihood and the SD moves with theta through the prediction.
$INPUT ID ARM TIME DV TRT FLARE NIND MDV
$DATA ipred_magnitude.csv IGNORE=@
$PRED
 E0    = THETA(1) + THETA(4)*FLARE
 EMAX  = THETA(2) + THETA(5)*FLARE + THETA(6)*TRT
 ET50  = THETA(3) * EXP(THETA(7)*TRT)
 IPRED = E0 - EMAX*TIME/(TIME + ET50)
 W     = THETA(8)*EXP(THETA(9)*IPRED)/SQRT(NIND)
 Y     = IPRED + W*EPS(1)
$THETA
 (1.0, 5.19, 20.0)    ; TVE0
 (-5.0, 0.88, 20.0)   ; TVEMAX
 (0.01, 0.70, 20.0)   ; TVET50
 (-10.0, 0.97, 10.0)  ; B_E0_FLARE
 (-10.0, 1.10, 10.0)  ; B_EMAX_FLARE
 (-10.0, 0.79, 10.0)  ; B_EMAX_TRT
 (-10.0, -1.18, 10.0) ; B_LET50_TRT
 (0.01, 1.0, 100.0)   ; A
 (-5.0, 0.1, 5.0)     ; B
$SIGMA 1 FIX
$ESTIMATION METHOD=0 MAXEVAL=9999 PRINT=10 SIGL=12 NSIG=4 FORMAT=s1PE23.16
$COVARIANCE MATRIX=R SIGL=12
