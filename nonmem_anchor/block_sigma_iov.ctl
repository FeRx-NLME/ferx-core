$PROBLEM block_sigma x IOV: total/unbound pair correlated via L2, kappa on CL (ferx #979)
; Twin of block_sigma_iov.ferx on block_sigma_iov.csv (simulate_block_sigma_iov_data.py).
; Each sample is a total (FREE = 0) / unbound (FREE = 1) pair sharing an L2 id;
; EPS(1) acts on the total row, EPS(2) on the unbound row, and $SIGMA BLOCK(2)
; correlates them across the pair. Three occasions, a dose opening each, so
; occasions 2 and 3 start on residual drug (the kappa boundary is live).
$INPUT ID TIME DV EVID AMT CMT MDV OCC FREE L2
$DATA block_sigma_iov.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2

$PK
  OCC1 = 0
  OCC2 = 0
  OCC3 = 0
  IF(OCC.EQ.1) OCC1 = 1
  IF(OCC.EQ.2) OCC2 = 1
  IF(OCC.EQ.3) OCC3 = 1
  KAPPA = OCC1*ETA(2) + OCC2*ETA(3) + OCC3*ETA(4)
  CL = THETA(1)*EXP(ETA(1) + KAPPA)
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  Y = F + F*((1 - FREE)*EPS(1) + FREE*EPS(2))

$THETA
  (0.1, 1.2, 10)     ; TVCL
  (1, 12, 100)       ; TVV

$OMEGA 0.1                ; ETA_CL
$OMEGA BLOCK(1) 0.03      ; KAPPA_CL occasion 1
$OMEGA BLOCK(1) SAME      ; occasion 2
$OMEGA BLOCK(1) SAME      ; occasion 3

$SIGMA BLOCK(2) 0.0144 0.006 0.0324   ; PROP_TOTAL, PROP_UNBOUND (rho = 0.2778)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 NSIG=3 SIGL=9 PRINT=5 NOABORT
$TABLE ID TIME OCC ETA1 ETA2 ETA3 ETA4 FIRSTONLY NOAPPEND NOPRINT ONEHEADER
       FORMAT=s1PE23.16 FILE=block_sigma_iov.tab
