$PROBLEM ferx #1031 anchor -- sample-size-weighted IOV, hand-written KAPPA/SQRT(NARM)
; NONMEM has no `weight =` on an OMEGA, so the equivalent model writes the
; scaling into $PK by hand -- which is exactly what ferx's
; `kappa KAPPA_CL ~ 0.01 weight = NARM` desugars into. This control stream is
; the anchor for that declaration: same data, same 1-cpt oral structure, same
; FOCE objective.
;
; NARM is the arm size carried per occasion: 400 on OCC 1, 25 on OCC 2, so the
; occasion-2 kappa acts four times as strongly as the occasion-1 one. CMT is
; dropped so NONMEM routes doses to the depot and observations to the default
; observation compartment (2) under ADVAN2.
$INPUT ID TIME DV EVID AMT CMT=DROP RATE=DROP MDV OCC NARM
$DATA kappa_weight_iov.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  KAPPA = 0
  IF(OCC.EQ.1) KAPPA = ETA(4)
  IF(OCC.EQ.2) KAPPA = ETA(5)
  CL = THETA(1)*EXP(ETA(1) + KAPPA/SQRT(NARM))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))

$THETA
  (0.001, 0.2,  10.0)   ; TVCL
  (0.1,   10.0, 500.0)  ; TVV
  (0.01,  1.5,  50.0)   ; TVKA

$OMEGA
  0.09   ; ETA(1) CL BSV
  0.04   ; ETA(2) V  BSV
  0.30   ; ETA(3) KA BSV
$OMEGA BLOCK(1)
  0.01   ; ETA(4) IOV variance (the unweighted gamma^2)
$OMEGA BLOCK(1) SAME  ; ETA(5) occasion 2

$SIGMA
  0.04   ; proportional residual variance (SD 0.2)

$ESTIMATION METHOD=1 MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE MATRIX=R
$TABLE ID TIME OCC IPRED CWRES NOPRINT NOAPPEND ONEHEADER FILE=kappa_weight_iov.tab
