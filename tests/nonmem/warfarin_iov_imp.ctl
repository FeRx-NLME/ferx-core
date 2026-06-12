$PROBLEM Warfarin IOV (CL) - IMP validation
$DATA warfarin_iov.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV OCC
$SUBROUTINES ADVAN2 TRANS2
$PK
  ; Inter-occasion variability on CL via per-occasion etas sharing one variance
  OCC1 = 0
  OCC2 = 0
  IF(OCC.EQ.1) OCC1 = 1
  IF(OCC.EQ.2) OCC2 = 1
  IOVCL = OCC1*ETA(4) + OCC2*ETA(5)
  CL = THETA(1)*EXP(ETA(1) + IOVCL)
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.2)   ; TVCL
$THETA (0, 10.0)  ; TVV
$THETA (0, 1.5)   ; TVKA
$OMEGA 0.09       ; ETA_CL  (BSV)
$OMEGA 0.04       ; ETA_V   (BSV)
$OMEGA 0.30       ; ETA_KA  (BSV)
$OMEGA BLOCK(1) 0.01   ; KAPPA_CL occasion 1
$OMEGA BLOCK(1) SAME   ; KAPPA_CL occasion 2 (same variance => IOV)
$SIGMA 0.04       ; proportional variance
$ESTIMATION METHOD=IMP INTER NITER=5000 SEED=42 PRINT=5 NOABORT
$TABLE ID TIME OCC NOPRINT ONEHEADER FILE=sdtab_iov_imp
