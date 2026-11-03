$PROBLEM NPDE/NPD with a cross-endpoint correlated residual via L2 (ferx #1733, arm B)
; Twin of tests/fixtures/npde_block_sigma_b.ferx on npde_block_sigma_b.csv
; (simulate_npde_block_sigma_data.py). Each sample is a total (FREE = 0) /
; unbound (FREE = 1) pair sharing an L2 id; EPS(1) acts on the total row,
; EPS(2) on the unbound row, and $SIGMA BLOCK(2) correlates them across the
; pair: cov = rho*s1*s2 = -0.8*0.05*0.30 = -0.012. The ferx twin DECLARES
; rho = 0.5 and is scored at the FITTED -0.8.
;
; Whether NONMEM's $TABLE NPDE simulation honours L2 is itself measured by
; this run (ferx #1733 plan, row 6): the anchor test pins either answer.
$INPUT ID TIME DV EVID AMT CMT MDV FREE L2
$DATA npde_block_sigma_b.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  Y = F + F*((1 - FREE)*EPS(1) + FREE*EPS(2))

$THETA
  1.0  FIX   ; TVCL
  10.0 FIX   ; TVV

$OMEGA 0.09 FIX          ; ETA_CL

$SIGMA BLOCK(2) 0.0025 -0.012 0.09 FIX   ; PROP_TOTAL, PROP_UNBOUND at rho = -0.8

$ESTIMATION MAXEVAL=0 METHOD=1 INTER NOABORT POSTHOC PRINT=1
$TABLE ID TIME DV MDV FREE NPDE NPD ESAMPLE=2000 SEED=1733 WRESCHOL
       ONEHEADER NOAPPEND NOPRINT FORMAT=s1PE23.16 FILE=npde_block_sigma_b.tab
