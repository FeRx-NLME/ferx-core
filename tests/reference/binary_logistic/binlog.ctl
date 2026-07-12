$PROBLEM Binary logistic (F_FLAG=1 / LIKE) fixed-effects — ferx #760 anchor
$INPUT ID TIME DV CMT EVID MDV X
$DATA binary_logistic.csv IGNORE=@
$PRED
  LOGIT = THETA(1) + THETA(2)*X + THETA(3)*TIME + ETA(1)
  P1 = 1/(1 + EXP(-LOGIT))
  Y = 1 - P1
  IF (DV.EQ.1) Y = P1
$THETA
  (-10, 0.1, 10)   ; TH0 intercept
  (-10, 0.1, 10)   ; THX
  (-10, 0.1, 10)   ; THT
$OMEGA 0 FIX       ; dummy ETA(1) fixed to 0 -> fixed-effects logistic
$ESTIMATION METHOD=COND LAPLACE LIKE MAXEVAL=9999 PRINT=5 NSIG=6 SIGL=9
$COVARIANCE
