$PROBLEM #1716 density anchor: `density lognormal` as its affine Gaussian twin
; ferx: `omega ETA_CL ~ 0.2 density lognormal`, CL = TVCL * ETA_CL: the effect has
; mean 1 and variance 0.2, so ln(effect) ~ N(-S/2, S), S = ln(1 + 0.2). NONMEM has no
; such prior, but the same marginal model is this Gaussian stream:
;   CL = TVCL * EXP(SQRT(S)*ETA(1) - S/2), ETA(1) ~ N(0, 1) FIX.
; ferx's latent z = SQRT(S)*ETA(1) - S/2 is an affine map of ETA(1), and the
; Laplace/FOCEI objective is invariant under an affine map of eta, so the two engines
; must agree on the OFV to arithmetic, and ETA1 = (z + S/2)/SQRT(S).
; SLN, not S: NM-TRAN reads `S` as a compartment scale (WARNING 7). results/ was run
; with the name S; the arithmetic is the same.
; MAXEVAL=0 POSTHOC INTERACTION, the dataset and the rest of the model as
; eta_shape_null.ctl.
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA eta_shape.csv IGNORE=@
$SUBROUTINE ADVAN1 TRANS2
$PK
  SLN = THETA(3)
  CL  = THETA(1)*EXP(SQRT(SLN)*ETA(1) - SLN/2)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V
$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))
$THETA
  (0.01, 2.0, 100.0)   ; TVCL
  (0.1, 20.0, 1000.0)  ; TVV
  0.18232155679395462 FIX   ; S = ln(1 + 0.2)
$OMEGA
  1 FIX ; ETA_CL, standard normal
  0.1   ; ETA_V
$SIGMA
  0.01  ; PROP_ERR
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT FORMAT=s1PE23.16
$TABLE ID TIME PRED IPRED ETA1 ETA2 NOPRINT ONEHEADER FILE=eta_density_lognormal.tab FORMAT=s1PE23.16
