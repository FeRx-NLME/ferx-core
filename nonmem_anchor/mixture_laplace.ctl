$PROBLEM 2-class CL mixture, 1-cpt IV, LAPLACIAN (#3025 ferx Laplace mixture anchor)
; The `tests/nonmem/mixture_iv.ctl` model (two latent subpopulations differing only in
; clearance, constant mixing fraction, OMEGA / SIGMA fixed at the data-generating values)
; under NONMEM's Laplacian conditional estimation. ferx: `method = laplace`.
; Data: mixture_iv.csv next to this file (a copy of tests/nonmem/mixture_iv.csv). Run with
; NONMEM 7.5.1 (nmfe75); outputs in results/mixture_laplace.{lst,ext}, compared with the
; ferx twin mixture_laplace.ferx in tests/mixture_laplace_nonmem_anchor.rs.
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
  MIXE = MIXEST
  IPRED = F
  Y = IPRED * (1.0 + EPS(1))

$THETA
  (0.01, 1.2, 100.0)     ; TVCL1  (class 1, low CL)
  (0.01, 2.5, 100.0)     ; TVCL2  (class 2, high CL)
  (0.1, 10.0, 1000.0)    ; V
  (0.001, 0.5, 0.999)    ; P(1)   mixing fraction of class 1
$OMEGA 0.09 FIX           ; IIV on CL (fixed at data-generating value)
$SIGMA 0.04 FIX           ; proportional RUV (fixed at data-generating value)

$ESTIMATION METHOD=COND LAPLACIAN INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE MATRIX=R
$TABLE ID TIME MIXE NOPRINT ONEHEADER FILE=mixture_laplace.sdtab
