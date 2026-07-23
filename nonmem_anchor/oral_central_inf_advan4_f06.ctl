$PROBLEM oral_central_inf_advan4_f06
; #376: 2-cpt ORAL model (ADVAN4), infusion (RATE>0) into CMT=2 (central,
; depot bypass), F2=0.6. F2 duration-scaling (#419) on a 2-cpt oral central
; infusion: duration = F*AMT/RATE = 0.6*100/25 = 2.4 h at held rate 25.
$INPUT ID TIME DV AMT CMT EVID RATE SS II
$DATA oral_central_inf_advan4.csv IGNORE=@
$SUBROUTINES ADVAN4 TRANS4
$PK
  CL = 5
  V2 = 50
  Q  = 3
  V3 = 80
  KA = 1
  F2 = 0.6
  S2 = V2
$ERROR
  Y = F * (1 + EPS(1))
$THETA (5.0 FIX)   ; CL (unused; NONMEM needs >=1 THETA)
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME CMT PRED NOPRINT ONEHEADER FILE=oral_central_inf_advan4_f06.tab FORMAT=,1PE17.10
