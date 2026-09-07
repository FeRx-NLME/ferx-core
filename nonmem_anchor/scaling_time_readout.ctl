$PROBLEM ferx #1028 anchor -- TIME in the structural readout ([scaling] y vs $ERROR IPRED)
$INPUT ID TIME DV MDV EVID AMT CMT
$DATA scaling_time_readout.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL, DEFDOSE)

$PK
  CL   = THETA(1)
  V    = THETA(2)
  BETA = THETA(3)
  T50  = THETA(4)
  K    = CL/V

$DES
  DADT(1) = -K*A(1)

$ERROR
  IPRED = A(1)/V + BETA*TIME/(TIME + T50)
  Y = IPRED*(1+EPS(1))

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (3.0 FIX)   ; BETA  -- Emax of the response-vs-time term
  (4.0 FIX)   ; T50   -- half-maximal time

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sdtab.tab
