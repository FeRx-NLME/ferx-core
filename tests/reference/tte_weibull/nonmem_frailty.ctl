$PROBLEM TTE Weibull (FRAILTY on shape) – Phase 1 ferx validation
; Mixed-effects (frailty on shape) via CONDITIONAL LAPLACIAN INTERACTION; fills the
; "NONMEM LAPLACIAN" column / the #440 nonlinear-frailty omega^2 cross-check
; (ferx 0.204 vs nlmixr2 0.173).
;
; KEY for F_FLAG=1 likelihood models: a dummy **$SIGMA 1 FIX is REQUIRED**. Without an
; EPS, NM-TRAN infers the data are single-subject and rejects CONDITIONAL/LAPLACIAN
; (error 350). The dummy must ALSO be REFERENCED in $PRED (DUMMY = EPS(1) below) —
; declaring $SIGMA alone is not enough. DUMMY is unused, so the likelihood is unchanged.
; (Records-per-subject was NOT the trigger — the plain 1-row/subject file is used.)
;
; Also: HAZNOW (not the reserved name H); IGNORE=@; clean $TABLE; NUMERICAL SLOW on $EST
; + SLOW on $COV (NONMEM requires these with LAPLACIAN INTERACTION).
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_weibull.csv IGNORE=@

$PRED
  SCALE  = EXP(THETA(1))           ; scale (time units): H=(t/scale)^shape
  SHAPE  = EXP(THETA(2) + ETA(1))  ; shape (dimensionless), frailty on shape
  CHZ    = (TIME/SCALE)**SHAPE      ; H(t)
  HAZNOW = (SHAPE/SCALE)*(TIME/SCALE)**(SHAPE-1)  ; h(t)  (NOT H — reserved $PRED arg)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)
  DUMMY = EPS(1)
$THETA (0.001, 3.0, 10)   ; log(scale): init log(20) = 2.996
$THETA (0.001, 0.693, 5)  ; log(shape): init log(2.0) = 0.693
$OMEGA 0.20               ; var(eta.shape): ESTIMATED frailty, init 0.20
$SIGMA 1 FIX              ; dummy EPS — required so NONMEM treats this as population data

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION NUMERICAL SLOW MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E SLOW
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_weibull_frailty.sdtab
