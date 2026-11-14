$PROBLEM #2368: an endpoint row, or a row skipped for its missing DV, governs an ADDL-implied dose (ADVAN1, CL = TVCL*X)
; NONMEM has no endpoint kinds, so the ferx twins' binary rows (CMT 3) and missing-DV rows
; are plain records here. The ferx MDV and CMT columns are dropped. NMMDV is MDV=1 on those
; rows, so the .tab scores only the Gaussian observations both sides compare.
; 1 endpoint row at an implied dose (12) / 2 endpoint row after it (15) / 3 missing-DV
; observation at 15 / 4 endpoint row at 15 with a missing DV / 5 constant X /
; 6 endpoint rows at 15 and at the implied dose at 24 / 7 case 2 led by an endpoint row at 0
$INPUT ID TIME DV MDVF=DROP EVID AMT RATE CMTF=DROP ADDL II X MDV
$DATA addl_endpoint.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  CL = THETA(1)*X*EXP(ETA(1))
  V  = THETA(2)
  S1 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0.5 FIX) (10 FIX)
$OMEGA 0.01 FIX
$SIGMA 0.01 FIX
$ESTIMATION METHOD=0 MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME EVID MDV X PRED NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=addl_endpoint.tab
