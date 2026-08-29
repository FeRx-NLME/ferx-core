$PROBLEM #1121 review F2 - SS bolus with ALAG >= II; pre-arrival window holds a virtual pulse
$INPUT ID TIME DV AMT RATE EVID CMT MDV WT SS II LG
$DATA ss_lag_ge_ii_flatwt.csv IGNORE=@

$SUBROUTINE ADVAN13 TOL=9

$MODEL
  COMP=(CENTRAL,DEFDOSE,DEFOBS)

$PK
  CL    = THETA(1)*(WT/70)**THETA(3)*EXP(ETA(1))
  V     = THETA(2)*EXP(ETA(2))
  ALAG1 = LG
  S1    = V

$DES
  DADT(1) = -(CL/V)*A(1)

$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))

$THETA
  (0, 10.0)  ; 1 TVCL
  (0, 50.0)  ; 2 TVV
  (0, 0.75)  ; 3 WTEXP

$OMEGA
  0.09  ; ETA_CL
  0.09  ; ETA_V

$SIGMA
  0.0025 ; PROP variance (sd 0.05)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME WT DV IPRED CL V ALAG1
       ONEHEADER NOPRINT FILE=ss_lag_ge_ii_flatwt.tab
