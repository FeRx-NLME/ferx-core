$PROBLEM #1716 ETA shape anchor: Box-Cox on CL, lambda = +0.5
; Karlsson's / PsN `transform` Box-Cox, Petersson et al. 2009 form A:
; ETATR = (EXP(ETA)**LAMBDA - 1)/LAMBDA, CL = TVCL*EXP(ETATR).
; Differs from eta_shape_null.ctl by this transform alone.
; MAXEVAL=0 POSTHOC INTERACTION on the multi-dose eta_shape.csv.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  ETATR = (EXP(ETA(1))**THETA(3) - 1)/THETA(3)
  CL = THETA(1)*EXP(ETATR)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
  0.5 FIX              ; LAMBDA
$OMEGA
  0.2   ; ETA_CL
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FILE=eta_shape_bc_pos.tab FORMAT=s1PE23.16
