$PROBLEM NPDE/NPD + IOV anchor, Cholesky decorrelation (ferx #734)
; Identical to npde_iov_anchor.ctl except for WRESCHOL on $TABLE, which
; switches NONMEM's NPDE decorrelation from its default (symmetric square
; root, the CWRES convention) to the Cholesky factor of the simulated
; covariance -- the Brendel/Comets procedure ferx implements. NPD carries no
; decorrelation and is bit-identical between the two runs (same ESAMPLE/SEED),
; so the pair isolates the decorrelation convention and nothing else.
; Twin of tests/fixtures/npde_iov_anchor.ferx on the identical dataset
; (npde_iov_anchor.csv, written by tests/gen_npde_iov_anchor.rs).
;
; Every parameter is FIXed at the value the data was simulated under, so the
; reference distribution NONMEM builds for $TABLE NPDE/NPD is the same
; population model ferx's compute_npde_npd builds -- the only difference is the
; Monte-Carlo stream. ESAMPLE/SEED mirror the ferx nsim/seed.
;
; ferx's `one_cpt_iv(cl=CL, v=V)` predicts a concentration A1/V, which is
; ADVAN1 TRANS2 with S1 = V.
$INPUT ID TIME DV EVID AMT CMT MDV OCC
$DATA npde_iov_anchor.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  OC1 = 0
  OC2 = 0
  OC3 = 0
  IF(OCC.EQ.1) OC1 = 1
  IF(OCC.EQ.2) OC2 = 1
  IF(OCC.EQ.3) OC3 = 1
  KAPCL = OC1*ETA(3) + OC2*ETA(4) + OC3*ETA(5)
  CL = THETA(1)*EXP(ETA(1) + KAPCL)
  V  = THETA(2)*EXP(ETA(2))
  S1 = V

$ERROR
  IPRED = F
  Y = F*(1 + EPS(1))

$THETA
  5.0  FIX   ; TVCL
  50.0 FIX   ; TVV

$OMEGA 0.09 FIX          ; ETA_CL  (BSV)
$OMEGA 0.04 FIX          ; ETA_V   (BSV)
$OMEGA BLOCK(1) 0.09 FIX ; KAPPA_CL, occasion 1
$OMEGA BLOCK(1) SAME     ; occasion 2
$OMEGA BLOCK(1) SAME     ; occasion 3

$SIGMA 0.01 FIX          ; proportional, sd 0.1

$ESTIMATION MAXEVAL=0 METHOD=1 INTER NOABORT POSTHOC PRINT=1
$TABLE ID TIME DV NPDE NPD ESAMPLE=2000 SEED=734 WRESCHOL
       ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=npde_iov_chol.tab
