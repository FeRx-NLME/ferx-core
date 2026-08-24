$PROBLEM ferx #1001 anchor SMALL -- proportional residual error, sigma = 0.05 SD
; The fit a single-endpoint `[error_model] DV ~ proportional(S_SMALL)` reads as
; describing. Identical to sigma_order_big.ctl except for the $SIGMA line.
$INPUT ID TIME AMT EVID MDV CMT DV
$DATA sigma_order.csv IGNORE=@

$SUBROUTINE ADVAN1 TRANS1

$PK
  KE = THETA(1)*EXP(ETA(1))
  V  = THETA(2)
  K  = KE
  S1 = V

$ERROR
  IPRED = F
  Y = IPRED*(1+EPS(1))

$THETA
  (0.1 FIX)   ; TVKE
  (10.0 FIX)  ; TVV

$OMEGA 0.09 FIX
$SIGMA 0.0025 FIX   ; 0.05**2 -- S_SMALL

$ESTIMATION MAXEVAL=0 METHOD=1 INTER
$TABLE ID TIME IPRED NOPRINT ONEHEADER FILE=sigma_order_small.tab
