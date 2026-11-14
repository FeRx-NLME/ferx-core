$PROBLEM SS=1 infusions whose duration exceeds II (overlapping pulses), 1cpt
; T_inf = 1.5 II (ID 1), 2.5 II (IDs 2, 4, 5), exactly 2 II (ID 3); ALAG 3 (IDs 4, 6);
; WT steps at t = 10 (IDs 5, 6). Every subject takes a non-SS infusion at t = 24 onto
; residual drug. Run with NONMEM 7.6.0 (results/ss_infusion_overlap_1cpt.tab); its PRED
; matches ss_infusion_overlap_ref_1cpt.csv (exact pulse-train superposition, ss_infusion_overlap_ref.py).
$INPUT ID TIME DV AMT RATE EVID CMT MDV WT SS II LG
$DATA ss_infusion_overlap.csv IGNORE=@

$SUBROUTINE ADVAN1 TRANS2

$PK
  CL    = THETA(1)*(WT/70)**0.75*EXP(ETA(1))
  V     = THETA(2)*EXP(ETA(2))
  ALAG1 = LG
  S1    = V

$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))

$THETA
  (0, 2.0)   ; 1 TVCL
  (0, 20.0)  ; 2 TVV

$OMEGA
  0.09  ; ETA_CL
  0.09  ; ETA_V

$SIGMA
  0.0025

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME WT DV PRED IPRED ONEHEADER NOPRINT FORMAT=s1PE23.16
       FILE=ss_infusion_overlap_1cpt.tab
