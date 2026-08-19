$PROBLEM 2-class CL mixture, 1-cpt IV (#985 ferx SAEM anchor)
; Two latent subpopulations differing only in clearance (TVCL1 vs TVCL2),
; constant (covariate-free) mixing fraction P(1). SAEM estimation — the
; NONMEM METHOD=SAEM reference for the ferx mixture SAEM path (#985). Same
; model / data as mixture_iv.ctl (which is FOCE-I); Omega/Sigma FIXed at the
; data-generating values so the estimated quantities are the two class
; clearances, V, and the mixing fraction.
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

; SAEM: stochastic-approximation EM. The class indicator is sampled each
; E-step, exactly the scheme the ferx SAEM mixture path implements (#985).
$ESTIMATION METHOD=SAEM INTERACTION NBURN=1000 NITER=1000 ISAMPLE=2
            SEED=20250818 CTYPE=3 PRINT=100 NOABORT
; Importance-sampling step to report a comparable objective function value.
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=5 ISAMPLE=1000
            SEED=20250818 PRINT=1 NOABORT
$TABLE ID TIME MIXE NOPRINT ONEHEADER FILE=mixture_iv_saem.sdtab
