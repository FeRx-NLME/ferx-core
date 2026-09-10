$PROBLEM ferx #1312 anchor -- `categorical` (MFL `cat`), the reparameterization twin
; The `1 + THETA(4)` shape of the same covariate effect on the same data as
; `covariate_cat2.ctl`, started at the corresponding value (0.999 - 1 = -0.001,
; the PsN / ferx `categorical` default init). The pair exists so the identity
; theta_cat2 = 1 + theta_cat is anchored *in NONMEM* and not only inside ferx:
; both runs must land on the same OFV, and NONMEM's two thetas must differ by 1.
;
; CMT is dropped so NONMEM routes doses to the depot and observations to the
; default observation compartment (2) under ADVAN2.
$INPUT ID TIME DV EVID AMT CMT=DROP RATE=DROP MDV SEX
$DATA covariate_cat2.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  ; The `cat` shape: the reference level contributes 1, each non-reference level
  ; 1 + its own THETA -- an offset from the reference, not the factor itself.
  COVCL = 1
  IF (SEX.EQ.1) COVCL = 1 + THETA(4)
  CL = THETA(1)*COVCL*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))

$THETA
  (0.01,  1.0,    100.0)  ; TVCL
  (1.0,   20.0,   500.0)  ; TVV
  (0.01,  1.0,    20.0)   ; TVKA
  (-1.0, -0.001,  5.0)    ; THETA_CL_SEX_1 -- the PsN / ferx `categorical`
                          ; default init and bounds.

$OMEGA
  0.09   ; ETA(1) CL BSV
  0.09   ; ETA(2) V  BSV

$SIGMA
  0.04   ; proportional residual variance

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE MATRIX=R
$TABLE ID TIME SEX IPRED CWRES NOPRINT NOAPPEND ONEHEADER FILE=covariate_cat.tab
