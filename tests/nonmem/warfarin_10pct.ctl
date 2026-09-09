$PROBLEM Warfarin 1-cpt oral, 10% proportional residual - VI Anchor B/D realistic arm
; Companion to warfarin_imp.ctl, which is the same model on data/warfarin.csv (~1% residual).
;
; Why a second dataset. The ~1% residual of data/warfarin.csv is not a realistic PK error, and
; it is the one regime where ferx's VI misses sigma (+5.5%): the variational q initialises at the
; prior and must contract ~1400x in variance to reach the posterior. At 10% the contraction is
; ~14x and VI lands on AGQ to 0.04%. The VI validation therefore needs both arms, and this arm
; needs a NONMEM column for the same reason the 1% arm has one: to corroborate the AGQ reference
; independently of either implementation being validated against it.
;
; Data: warfarin_10pct.csv, simulated from the 1% file's own converged estimates with sigma
; swapped to 0.10 - same 10 subjects, same 11-point grid, same 100 mg dose, one variable changed.
; Generated deterministically by tools/vi-emvi-comparison/make-wide-residual-data.sh (seed 1).
;
; Initial estimates deliberately match the ferx harness models (tools/vi-emvi-comparison/*.ferx)
; so the engines start from the same point, not just fit the same file.
$DATA warfarin_10pct.csv IGNORE=@
$INPUT ID TIME DV EVID AMT CMT=DROP RATE MDV
$SUBROUTINES ADVAN2 TRANS2
$PK
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2))
  KA = THETA(3)*EXP(ETA(3))
  S2 = V
$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1))
$THETA (0, 0.13)  ; TVCL
$THETA (0, 8.0)   ; TVV
$THETA (0, 1.0)   ; TVKA
$OMEGA 0.09       ; ETA_CL
$OMEGA 0.04       ; ETA_V
$OMEGA 0.30       ; ETA_KA
$SIGMA 0.01       ; proportional VARIANCE (= 0.10 SD, matching the .ferx init)
$ESTIMATION METHOD=COND INTERACTION MAXEVAL=9999 PRINT=5 NOABORT
$COVARIANCE UNCONDITIONAL
$TABLE ID TIME DV PRED IPRED CWRES NOPRINT ONEHEADER FILE=sdtab_10pct
