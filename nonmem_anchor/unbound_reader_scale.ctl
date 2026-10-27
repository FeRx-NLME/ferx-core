$PROBLEM ferx #1898 anchor B -- scale S2 = V*SCF, SCF an ordinary $PK variable no PK role uses
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA unbound_reader.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  CL = THETA(1)
  V  = THETA(2)
  KA = THETA(3)
  SCF = THETA(4)*WT/70 ; 0.857 (WT 60) / 1.143 (WT 80): read only by the scale
  S2 = V*SCF

$ERROR
  IPRED = F
  Y = IPRED + EPS(1)   ; additive: IPRED(0) = 0 here (no init)

$THETA
  (5.0 FIX)   ; CL
  (50.0 FIX)  ; V
  (1.5 FIX)   ; KA
  (1.0 FIX)   ; SCF at 70 kg

$OMEGA 0 FIX
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME WT IPRED NOPRINT ONEHEADER FILE=unbound_reader_scale.tab
