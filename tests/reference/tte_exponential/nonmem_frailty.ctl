$PROBLEM TTE Exponential (FRAILTY) – Phase 1 ferx validation
; Mixed-effects (frailty) fit via CONDITIONAL LAPLACIAN INTERACTION; fills the
; "NONMEM LAPLACIAN" column of expected.md.
;
; KEY for F_FLAG=1 likelihood models: a dummy **$SIGMA 1 FIX is REQUIRED**. Without an
; EPS/residual level, NM-TRAN infers the data are single-subject and rejects
; CONDITIONAL/LAPLACIAN ("350 METHOD=CONDITIONAL INVALID WITH SINGLE-SUBJECT DATA").
; The dummy must ALSO be REFERENCED in $PRED (DUMMY = EPS(1) below) — declaring $SIGMA
; alone is not enough; an *unreferenced* EPS still triggers the single-subject inference.
; DUMMY is itself unused, so the likelihood is unchanged. (Records-per-subject was NOT
; the trigger; the data is the plain 1-row/subject file the other tools fit.)
;
; Also: HAZNOW (not the reserved $PRED name H); $TABLE without IPRED/PRED (undefined for
; F_FLAG=1); IGNORE=@ (skip the header); NUMERICAL SLOW on $EST + SLOW on $COV (NONMEM
; requires these with LAPLACIAN INTERACTION — given explicitly to silence the warning).
$INPUT ID TIME DV EVID CMT MDV
$DATA tte_exp.csv IGNORE=@

$PRED
  LAMBDA = EXP(THETA(1) + ETA(1))
  CHZ    = LAMBDA * TIME        ; H(t) = lambda * t  (Exponential)
  HAZNOW = LAMBDA               ; h(t) = lambda  (do NOT name this H — reserved $PRED arg)
  F_FLAG = 1
  IF (DV.EQ.0) Y = EXP(-CHZ)            ; right-censored: S(T)
  IF (DV.EQ.1) Y = HAZNOW * EXP(-CHZ)   ; exact event:    h(T)*S(T)
  DUMMY = EPS(1)

$THETA (-10, -2.3, 5)   ; log(lambda): init log(0.1) = -2.303
$OMEGA 0.25             ; var(eta.lambda): ESTIMATED frailty, init 0.25
$SIGMA 1 FIX            ; dummy EPS — required so NONMEM treats this as population data

$ESTIMATION METHOD=COND LAPLACIAN INTERACTION NUMERICAL SLOW MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE PRINT=E SLOW
$TABLE ID TIME DV NOPRINT NOAPPEND FILE=tte_exp_frailty.sdtab
