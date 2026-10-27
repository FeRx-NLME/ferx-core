$PROBLEM NPDE/NPD with a correlated combined residual (ferx #1733, arm A)
; Twin of tests/fixtures/npde_block_sigma_a.ferx on npde_block_sigma_a.csv
; (simulate_npde_block_sigma_data.py). The ferx twin DECLARES rho = 0.5 and is
; scored at a parameter set carrying the FITTED rho = -0.9; this stream FIXes
; the fitted value, so it is the reference for what the ferx NPDE must
; simulate. Before #1733 ferx drew at the declaration.
;
; Y = F + F*EPS(1) + EPS(2) is ferx's combined(PROP, ADD):
;   Var = F^2*s1^2 + s2^2 + 2*F*cov,  cov = rho*s1*s2 = -0.9*0.2*1.0 = -0.18.
; Two doses, the second landing on residual drug, so F spans ~1..20 and both
; the proportional and the cross term are live across the records.
;
; ESAMPLE/SEED mirror ferx's nsim/seed; WRESCHOL selects the Cholesky
; decorrelation ferx implements (see npde_iov_chol.ctl).
$INPUT ID TIME DV EVID AMT CMT MDV
$DATA npde_block_sigma_a.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  Y = F + F*EPS(1) + EPS(2)

$THETA
  1.0  FIX   ; TVCL
  10.0 FIX   ; TVV

$OMEGA 0.09 FIX          ; ETA_CL

$SIGMA BLOCK(2) 0.04 -0.18 1.0 FIX   ; PROP, ADD at the fitted rho = -0.9

$ESTIMATION MAXEVAL=0 METHOD=1 INTER NOABORT POSTHOC PRINT=1
$TABLE ID TIME DV MDV NPDE NPD ESAMPLE=2000 SEED=1733 WRESCHOL
       ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=npde_block_sigma_a.tab
