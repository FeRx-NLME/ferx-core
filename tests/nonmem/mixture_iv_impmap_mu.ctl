$PROBLEM 2-class CL mixture, 1-cpt IV, MU-referenced IMPMAP (#996 ferx anchor)
; Same model / data as mixture_iv.ctl (FOCE-I) and mixture_iv_saem.ctl (SAEM):
; two latent subpopulations differing only in clearance (TVCL1 vs TVCL2) with a
; constant mixing fraction P(1), Omega/Sigma FIXed at the data-generating values.
;
; What is new here (#996): the class-switched typical value is written as a
; MU_ reference *inside* the MIXNUM branch, so NONMEM's EM machinery applies its
; mu-referenced conjugate theta move per class. This is the NONMEM counterpart
; of the ferx class-aware mu-ref shift for estimating IMP/IMPMAP; the existing
; anchors both use a plain `IF (MIXNUM.EQ.1) TVCL = THETA(1)` with no MU_.
$INPUT ID TIME DV AMT EVID CMT WT
$DATA mixture_iv.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  IF (MIXNUM.EQ.1) MU_1 = LOG(THETA(1))
  IF (MIXNUM.EQ.2) MU_1 = LOG(THETA(2))
  CL = EXP(MU_1 + ETA(1))
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

; IMPMAP (MAP-centred importance-sampling EM) at the same settings the ferx
; test uses: 50 iterations x 1500 samples per subject, seed 20250818.
$ESTIMATION METHOD=IMPMAP INTERACTION NITER=50 ISAMPLE=1500
            SEED=20250818 PRINT=5 NOABORT
$TABLE ID TIME MIXE NOPRINT ONEHEADER FILE=mixture_iv_impmap_mu.sdtab
