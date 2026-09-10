$PROBLEM ferx #1312 anchor -- `categorical2` (Pharmpy MFL `cat2`) covariate form
; ferx's `CL ~ SEX categorical2(ref = 0)` desugars to a factor of 1 at the
; reference level and THETA(4) at the contrast level. That has an exact NONMEM
; spelling, written out by hand below, so this is a direct anchor: same data,
; same 1-cpt oral structure, same FOCEI objective, one declaration versus three
; lines of $PK.
;
; The companion run `covariate_cat.ctl` is the `1 + THETA(4)` (`cat`) shape on
; the same data, started at the corresponding value. The two must reach the same
; OFV at theta_cat2 = 1 + theta_cat -- the reparameterization identity.
;
; CMT is dropped so NONMEM routes doses to the depot and observations to the
; default observation compartment (2) under ADVAN2; ferx's `one_cpt_oral` keeps
; the depot implicit and numbers central 1.
$INPUT ID TIME DV EVID AMT CMT=DROP RATE=DROP MDV SEX
$DATA covariate_cat2.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  ; The `cat2` shape: the reference level contributes 1, each non-reference
  ; level its own THETA -- the factor itself, not an offset from 1.
  COVCL = 1
  IF (SEX.EQ.1) COVCL = THETA(4)
  CL = THETA(1)*COVCL*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)
  S2 = V

$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))

$THETA
  (0.01, 1.0,  100.0)   ; TVCL
  (1.0,  20.0, 500.0)   ; TVV
  (0.01, 1.0,  20.0)    ; TVKA
  (0.0,  0.999, 6.0)    ; THETA_CL_SEX_1 -- the `categorical2` default init and
                        ; bounds: the image of the `categorical` (-0.001, -1, 5)
                        ; row under theta_cat2 = 1 + theta_cat.

$OMEGA
  0.09   ; ETA(1) CL BSV
  0.09   ; ETA(2) V  BSV

$SIGMA
  0.04   ; proportional residual variance

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
; MATRIX=R gives the pure R^-1 covariance -- the same estimator ferx computes.
$COVARIANCE MATRIX=R
$TABLE ID TIME SEX IPRED CWRES NOPRINT NOAPPEND ONEHEADER FILE=covariate_cat2.tab
