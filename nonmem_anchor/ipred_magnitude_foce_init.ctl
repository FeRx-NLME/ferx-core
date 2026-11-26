$PROBLEM ferx #3121 anchor, FOCE without INTER (MAXEVAL=0): SD = A*EXP(B*IPRED)/SQRT(NIND), EMAX log-normal
; Natural-scale naproxen arm means (data/mbma_naproxen.csv, DV*WPSE), study-level random
; effects. The within-arm SD model A*EXP(B*IPRED) divided by SQRT(NIND) is the SE of the arm
; mean; no combined / power form expresses it (d2SD/dIPRED2 != 0).
$INPUT ID ARM TIME DV TRT FLARE NIND MDV
$DATA ipred_magnitude.csv IGNORE=@
$PRED
 E0    = THETA(1) + THETA(4)*FLARE + ETA(1)
 EMAX  = (THETA(2) + THETA(5)*FLARE + THETA(6)*TRT) * EXP(ETA(2))
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
$OMEGA 0.393129       ; ETA_E0   (0.627 sd)
$OMEGA 0.3            ; ETA_EMAX, log-normal here so f(eta) is nonlinear in eta
$SIGMA 1 FIX
$ESTIMATION METHOD=1 MAXEVAL=0 SIGL=12 NSIG=4 FORMAT=s1PE23.16
