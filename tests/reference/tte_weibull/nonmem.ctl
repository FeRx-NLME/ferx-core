$PROBLEM TTE Weibull (scale param) – Phase 1 ferx validation
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_weibull.csv IGNORE=#

$PRED
  SCALE  = EXP(THETA(1))           ; scale (time units): H=(t/scale)^shape
  SHAPE  = EXP(THETA(2) + ETA(1))  ; shape (dimensionless)
  CHZ    = (TIME/SCALE)**SHAPE      ; H(t)
  HAZNOW = (SHAPE/SCALE)*(TIME/SCALE)**(SHAPE-1)  ; h(t)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)

$THETA (0.001, 3.0, 10)   ; log(scale): init log(20) = 2.996
$THETA (0.001, 0.693, 5)  ; log(shape): init log(2.0) = 0.693
$OMEGA 0.20               ; var(eta.shape)

$ESTIMATION METHOD=CONDITIONAL LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=1 NOABORT
$COVARIANCE PRINT=E

$TABLE ID TIME DV ETA1 NOPRINT NOAPPEND FILE=tte_weibull.sdtab
