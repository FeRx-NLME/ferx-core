$PROBLEM IIV on residual error (ferx #409): Y = IPRED + EPS*EXP(ETA)
; 1-cpt oral (ADVAN2 TRANS2). Data simulated by tests/gen_iiv_anchor.rs from the
; ferx model nonmem_anchor/iiv_on_ruv_fit.ferx (true ETA_RUV variance = 0.30).
; Reproduce: nmfe75 iiv_on_ruv.ctl iiv_on_ruv.lst
$INPUT ID TIME DV EVID AMT CMT RATE MDV
$DATA iiv_on_ruv.csv IGNORE=@
$SUBROUTINE ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  IPRED = F
  ; proportional residual error scaled per-subject by EXP(ETA(4)) — IIV on RUV.
  Y = IPRED + IPRED*EPS(1)*EXP(ETA(4))
$THETA
  (0.001, 0.13, 10.0)   ; TVCL
  (0.1,   8.0,  500.0)  ; TVV
  (0.01,  1.0,  50.0)   ; TVKA
$OMEGA
  0.09   ; ETA_CL
  0.04   ; ETA_V
  0.30   ; ETA_KA
  0.30   ; ETA_RUV  (IIV on residual error)
$SIGMA
  0.01   ; PROP_ERR variance (sd 0.1)
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 NOABORT PRINT=5
$COVARIANCE UNCONDITIONAL
