$PROBLEM TTE Exponential – Phase 1 ferx validation
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_exp.csv IGNORE=#

$PRED
  LAMBDA = EXP(THETA(1) + ETA(1))
  CHZ    = LAMBDA * TIME        ; H(t) = lambda * t  (Exponential)
  HAZNOW = LAMBDA               ; h(t) = lambda      (constant)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)              ; right-censored: S(T)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)    ; exact event:    h(T)*S(T)

$THETA (-10, -2.3, 5)   ; log(lambda): init log(0.1) = -2.303
$OMEGA 0.25             ; var(eta.lambda): init 0.25

$ESTIMATION METHOD=CONDITIONAL LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=1 NOABORT
$COVARIANCE PRINT=E

$TABLE ID TIME DV ETA1 IPRED PRED NOPRINT NOAPPEND FILE=tte_exp.sdtab
