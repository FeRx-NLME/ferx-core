$PROBLEM first_order(ka) + estimated lag (ALAG1) -- NONMEM anchor for PR2 of #486
; ferx's built-in first_order(ka) input-rate function feeding `central` directly
; is mathematically identical to the classic depot->central first-order
; absorption (the "input into central" from an exponentially-decaying depot IS
; R_in(tad) = dose*ka*exp(-ka*tad)) -- so this is a plain, standard ADVAN2 TRANS2
; model with ALAG1, no $DES/PODO tricks needed. Validates the event-driven ODE
; sensitivity walk's new rate-on onset saltation for the lagtime + input-rate
; combination (the highest-risk new math in PR2, since first_order's onset
; R_in(0+) = dose*ka is always finite and nonzero, unlike igd's which vanishes).

$INPUT ID TIME DV AMT EVID CMT MDV
$DATA first_order_alag_nm.csv IGNORE=@

$SUBROUTINES ADVAN2 TRANS2

$PK
  CL    = THETA(1)*EXP(ETA(1))
  V     = THETA(2)
  KA    = THETA(3)*EXP(ETA(2))
  ALAG1 = THETA(4)
  S2    = V

$ERROR
  IPRED = F
  Y = IPRED*(1.0 + EPS(1))

$THETA
  (0.1,  5.0,  100)   ; 1 CL
  (5.0,  50.0, 500)   ; 2 V
  (0.05, 1.0,  20)    ; 3 KA
  (0.01, 0.5,  5)     ; 4 ALAG1

$OMEGA
  0.09    ; IIV CL
  0.04    ; IIV KA

$SIGMA
  0.01    ; proportional residual variance (0.10^2)

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE
$TABLE ID TIME DV IPRED CWRES MDV NOPRINT ONEHEADER FILE=first_order_alag.tab
