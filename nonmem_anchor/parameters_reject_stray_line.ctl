; ferx-core #1377 - NM-TRAN reject control, arm B: a stray line of its own in the
; $OMEGA region.
;
; NOT a fit anchor - see arm A (`parameters_reject_trailing_token.ctl`) for why a
; parse reject has no number to anchor. Arm B is the own-line half of the same
; question: ferx's `[parameters]` used to drop a line matching no declaration
; form without a trace, so `banana` on a line by itself validated clean.
;
; Identical to arm A except that the offending text sits on its own line rather
; than at the end of the $OMEGA record, and the $OMEGA record itself is clean.
; Delete the `BANANA` line and NM-TRAN accepts the stream.
;
; NM-TRAN (nmfe76, nonmemdocker:V0.1):
;   AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.
;   ... THE CHARACTERS IN ERROR ARE: BANANA
;
; The ferx equivalent is a lone `banana` line in [parameters], which reports
; E_PARSE quoting the line. A bare `(sd)` line after a block's closing `]` is
; folded back onto the block and reports E_BLOCK_VARIANCE_ONLY instead - see
; `unrecognized_parameters_line_is_rejected` in src/parser/model_parser_tests.rs.
$PROBLEM #1377 arm B - stray line in the $OMEGA region (NM-TRAN must reject)

$INPUT ID TIME DV EVID AMT DROP RATE MDV
$DATA ../data/warfarin.csv IGNORE=@

$SUBROUTINE ADVAN2 TRANS2

$PK
  CL = THETA(1) * EXP(ETA(1))
  V  = THETA(2) * EXP(ETA(2))
  KA = THETA(3)
  S2 = V

$ERROR
  IPRED = F
  Y     = IPRED * (1 + EPS(1))

$THETA 0.134 FIX  8.11 FIX  1.35 FIX   ; CL V KA
$OMEGA BLOCK(2) 0.09 0.02 0.04
BANANA
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
