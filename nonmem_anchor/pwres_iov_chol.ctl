$PROBLEM EPRED + EWRES anchor (ferx #2871) -- 1-cpt IV bolus, IOV on CL
; Same model, data and FIXed parameters as the #734 NPDE anchor
; (npde_iov_anchor.ctl, twin of tests/fixtures/npde_iov_anchor.ferx, dataset
; npde_iov_anchor.csv written by tests/gen_npde_iov_anchor.rs), tabling the
; Monte-Carlo EPRED and EWRES instead of NPDE/NPD.
;
; EWRES = Sigma_sim^(-1/2) (DV - EPRED), with EPRED and Sigma_sim the mean and
; covariance of ESAMPLE simulated replicates of the subject. NONMEM's default
; decorrelation is the symmetric inverse square root; WRESCHOL switches it to
; the Cholesky factor. This is the WRESCHOL run: identical to
; pwres_iov_anchor.ctl but for that flag, so EPRED is bit-identical between the
; two and EWRES differs only by the decorrelation factor.
;
; Every parameter is FIXed, so the reference distribution is ferx's
; compute_npde_npd reference -- the only difference is the Monte-Carlo stream.
; ESAMPLE/SEED mirror the ferx nsim/seed (tests/pwres_iov_nonmem_anchor.rs).
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
$TABLE ID TIME DV EPRED EWRES ESAMPLE=2000 SEED=2871 WRESCHOL
       ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=pwres_iov_chol.tab
