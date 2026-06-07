$PROBLEM TTE Gompertz (2-arm RCT, fixed effects) – Phase 1 ferx validation
$INPUT ID TIME DV TRT EVID CMT MDV
$DATA tte_gompertz.csv IGNORE=#

$PRED
  ALPHA  = EXP(THETA(1))     ; baseline hazard rate at t=0
  GAMMA  = EXP(THETA(2))     ; hazard acceleration (growth rate)
  LOGHR  = THETA(3)          ; log hazard ratio (treatment effect)
  H      = ALPHA * EXP(GAMMA * TIME) * EXP(LOGHR * TRT)
  CHZ    = (ALPHA/GAMMA) * (EXP(GAMMA*TIME) - 1) * EXP(LOGHR * TRT)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)
  IF (DV.EQ.1) Y = H * EXP(-CHZ)

$THETA (-15, -6.0, 0)    ; log_alpha: init -6.0
$THETA (-15, -5.4, 0)    ; log_gamma: init -5.4
$THETA (-5, -0.8, 5)     ; log_hr:    init -0.8

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION MAXEVAL=9999 PRINT=1 NOABORT
$COVARIANCE PRINT=E

$TABLE ID TIME DV TRT IPRED PRED NOPRINT NOAPPEND FILE=tte_gompertz.sdtab
