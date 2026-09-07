$PROBLEM 2-class CL mixture + IOV on CL, 1-cpt IV (#985 ferx IOV+mixture anchor)
; Two latent subpopulations differing in clearance (TVCL1 vs TVCL2), constant
; mixing fraction P(1), plus inter-occasion variability on CL: each subject has
; two dosing occasions (OCC 1/2), and CL carries a per-occasion kappa drawn from
; a shared IOV omega ($OMEGA BLOCK(1) ... SAME). FOCE-I estimation. IIV + IOV
; omegas are free; proportional sigma fixed at the data-generating value so the
; MATRIX=R standard errors compare cleanly against ferx (pure R^-1). IIV + IOV
; omegas and sigma are FIXED at their data-generating values (mirroring the
; mixture_iv_cov anchor): with only two occasions the IOV omega is weakly
; identified, so fixing the variances keeps the 4-theta R matrix well-conditioned
; while the IOV kappa still enters the FOCE-I marginal.
$INPUT ID TIME DV AMT EVID CMT OCC WT
$DATA mixture_iv_iov.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  IF (MIXNUM.EQ.1) TVCL = THETA(1)
  IF (MIXNUM.EQ.2) TVCL = THETA(2)
  OCC1 = 0
  OCC2 = 0
  IF (OCC.EQ.1) OCC1 = 1
  IF (OCC.EQ.2) OCC2 = 1
  KACL = OCC1*ETA(2) + OCC2*ETA(3)
  CL = TVCL * EXP(ETA(1) + KACL)
  V  = THETA(3)
  S1 = V

$MIX
  NSPOP = 2
  P(1) = THETA(4)
  P(2) = 1.0 - THETA(4)

$ERROR
  MIXE = MIXEST
  IPRED = F
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 1.0, 100.0)     ; TVCL1  (class 1, low CL)
  (0.01, 3.0, 100.0)     ; TVCL2  (class 2, high CL)
  (0.1, 10.0, 1000.0)    ; V
  (0.001, 0.5, 0.999)    ; P(1)   mixing fraction of class 1
$OMEGA 0.05 FIX          ; IIV on CL (fixed at data-generating value)
$OMEGA BLOCK(1) 0.02 FIX ; IOV on CL, occasion 1 (fixed)
$OMEGA BLOCK(1) SAME     ; IOV on CL, occasion 2 (same variance, inherits FIX)
$SIGMA 0.04 FIX          ; proportional RUV (fixed at data-generating value)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE MATRIX=R
$TABLE ID TIME MIXE NOPRINT ONEHEADER FILE=mixture_iv_iov.sdtab
