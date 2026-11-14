$PROBLEM SS=1 dose into a zero-order absorption window, with and without a lag (#2786)
; ferx: d/dt(central) = zero_order(dur=DUR, lag=LAG) - CL/V*central (per-route lag), and the
;       same with ALAG1 = LAG on the compartment; both compared with this table.
; NONMEM: ADVAN1, the zero-order input is NONMEM's own duration-modeled dose (RATE=-2,
;         D1 = DR) shifted by ALAG1 = LG; closed form, so PRED is exact to round-off.
; IDs 1-9 of ss_absorption_lag.csv (ss_absorption_lag_gen.py lists each case): DR below, at
; and above II = 8; LG 0, below II, lag + DR past II, LG = II and LG > II; WT steps at
; t = 10 on IDs 8-9. Every subject takes a non-SS dose at t = 24 onto residual drug.
$INPUT ID TIME DV AMT RATE EVID CMT MDV WT SS II LG DR
$DATA ss_absorption_lag.csv IGNORE=@ IGNORE=(ID.GT.10)

$SUBROUTINE ADVAN1 TRANS2

$PK
  CL    = THETA(1)*(WT/70)**0.75*EXP(ETA(1))
  V     = THETA(2)*EXP(ETA(2))
  D1    = DR
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
$TABLE ID TIME WT LG DR DV PRED IPRED ONEHEADER NOPRINT FORMAT=s1PE23.16
       FILE=ss_zero_order_lag.tab
