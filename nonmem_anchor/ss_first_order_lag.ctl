$PROBLEM SS=1 dose into a lagged first-order absorption compartment (#2786)
; ferx: d/dt(central) = first_order(ka=KA, lag=LAG) - CL/V*central (per-route lag), and an
;       explicit depot with ALAG1 = LAG on it; both compared with this table.
; NONMEM: ADVAN2 (depot = CMT 1, central = CMT 2), ALAG1 = LG on the depot; closed form, so
;         PRED is exact to round-off.
; IDs 11-17 of ss_absorption_lag.csv (ss_absorption_lag_gen.py lists each case): bolus with
; LG 0, 2, 5, = II (8) and > II (11); a WT step at t = 10 (ID 16); an infusion into the depot
; longer than II (RATE 100, T_inf 10) under LG 3 (ID 17). Every subject takes a non-SS dose
; at t = 24 onto residual drug.
$INPUT ID TIME DV AMT RATE EVID CMT MDV WT SS II LG DR
$DATA ss_absorption_lag.csv IGNORE=@ IGNORE=(ID.LT.10)

$SUBROUTINE ADVAN2 TRANS2

$PK
  CL    = THETA(1)*(WT/70)**0.75*EXP(ETA(1))
  V     = THETA(2)*EXP(ETA(2))
  KA    = THETA(3)
  ALAG1 = LG
  S2    = V

$ERROR
  IPRED = F
  Y     = IPRED*(1+EPS(1))

$THETA
  (0, 2.0)   ; 1 TVCL
  (0, 20.0)  ; 2 TVV
  (0, 0.7)   ; 3 TVKA

$OMEGA
  0.09  ; ETA_CL
  0.09  ; ETA_V

$SIGMA
  0.0025

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1 NOABORT
$TABLE ID TIME WT LG DV PRED IPRED ONEHEADER NOPRINT FORMAT=s1PE23.16
       FILE=ss_first_order_lag.tab
