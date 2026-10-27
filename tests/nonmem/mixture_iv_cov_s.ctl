$PROBLEM 2-class CL mixture, 1-cpt IV: MATRIX=S at the FOCEI finals, MAXEVAL=0 (#1956)
; mixture_iv_cov.ctl with the THETA initials set to its .ext finals (as printed, 6 sig. fig.)
; and MAXEVAL=0, so NONMEM evaluates exactly the point ferx's outer_maxiter = 0 does.
; MATRIX=S is the inverse score cross-product S^-1, ferx's covariance_method = s.
$INPUT ID TIME DV AMT EVID CMT WT
$DATA mixture_iv.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  IF (MIXNUM.EQ.1) TVCL = THETA(1)
  IF (MIXNUM.EQ.2) TVCL = THETA(2)
  CL = TVCL * EXP(ETA(1))
  V  = THETA(3)
  S1 = V

$MIX
  NSPOP = 2
  P(1) = THETA(4)
  P(2) = 1.0 - THETA(4)

$ERROR
  IPRED = F
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 0.980833, 100.0)   ; TVCL1
  (0.01, 2.84198, 100.0)    ; TVCL2
  (0.1, 9.97859, 1000.0)    ; V
  (0.001, 0.471970, 0.999)  ; P(1)
$OMEGA 0.09 FIX
$SIGMA 0.04 FIX

$ESTIMATION METHOD=1 INTER MAXEVAL=0 NOABORT
$COVARIANCE MATRIX=S PRINT=E
