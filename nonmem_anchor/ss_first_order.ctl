$PROBLEM Steady-state (SS=1) first-order oral absorption reference for ferx-core #719 (gap 1)
; Anchors ferx's SS-into-built-in-absorption support (a pulse-train equilibration
; trough + a periodic-sum forward R_in) against NONMEM's exact analytic ADVAN2 SS.
; KA is deliberately slow (t1/2,abs ~ 4.6 h) relative to II = 8 h, so each dose's
; absorption tail spills into the next interval — exactly the periodic R_in carryover.
; Pure evaluation at fixed thetas (MAXEVAL=0); PRED is the reference for the ferx test.
$DATA ss_first_order.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT MDV II SS
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)
  V  = THETA(2)
  KA = THETA(3)
  S2 = V
$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))
$THETA 1.0 FIX  20.0 FIX  0.15 FIX   ; CL V KA
$OMEGA 0 FIX
$SIGMA 0.01 FIX
$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
$TABLE ID TIME EVID PRED IPRED NOPRINT ONEHEADER FILE=sdtab1
