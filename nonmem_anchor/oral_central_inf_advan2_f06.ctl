$PROBLEM oral_central_inf_advan2_f06
; #376: 1-cpt ORAL model (ADVAN2), infusion (RATE>0) into CMT=2 (central,
; depot bypass), F2=0.6. Guards the F-on-infusion interaction (#327/#349/#419)
; on the depot-bypass central path: NONMEM holds the data RATE and scales the
; *duration* to F*AMT/RATE = 0.6*100/25 = 2.4 h (the full F*AMT delivered at
; rate 25 over a shortened window). The central compartment carries the whole
; dose (F2), so this is F2 (not F1) — F1 is irrelevant with an empty depot.
$INPUT ID TIME DV AMT CMT EVID RATE SS II
$DATA oral_central_inf_advan2.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = 5
  V  = 50
  KA = 1
  F2 = 0.6
  S2 = V
$ERROR
  Y = F * (1 + EPS(1))
$THETA (5.0 FIX)   ; CL (unused; NONMEM needs >=1 THETA)
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 PRINT=1 NOABORT
$TABLE ID TIME CMT PRED NOPRINT ONEHEADER FILE=oral_central_inf_advan2_f06.tab FORMAT=,1PE17.10
