; ferx-core #1377 - NM-TRAN reject control, arm B: a stray line of its own in the
; $OMEGA region.
;
; NOT a fit anchor - see arm A (`parameters_reject_trailing_token.ctl`) for why a
; parse reject has no number to anchor. Arm B is the own-line half of the
; question *ferx* asks - NM-TRAN does not split it that way, see below: ferx's
; `[parameters]` used to drop a line matching no declaration form without a
; trace, so `banana` on a line by itself validated clean.
;
; Identical to arm A except that the offending text sits on its own line rather
; than at the end of the $OMEGA record, and the $OMEGA record itself is clean.
; Delete the `BANANA` line and NM-TRAN accepts the stream.
;
; READ THE ASYMMETRY BEFORE QUOTING THIS STREAM. The two arms are two cases for
; ferx and ONE case for NM-TRAN. A NONMEM record runs to the next `$`, so
; `BANANA` on its own line is simply one more option on the $OMEGA record and
; fails with the same error 20 as arm A's trailing `FOO`. NM-TRAN has no notion
; of "a line matching no declaration form", because it has no notion of a line
; here at all. So what this pair supports is the narrower claim that an unknown
; token anywhere in the $OMEGA region is an error - NOT that the comparator
; treats the own-line half as its own case. ferx's `[parameters]` is
; line-oriented and does, which is a difference between the two parsers rather
; than NONMEM backing for ferx's split.
;
; Measured first-hand: nmfe76 (/opt/NONMEM/nm760/run/nmfe76), NONMEM VERSION
; 7.6.0 (nm760), docker image nonmemdocker:V0.1 (sha256:2ee46060f1e0). Rejected,
; nmfe exit 107. Verbatim, leading spaces as printed (the omitted
; `AN ERROR WAS FOUND ON LINE <N> ...` line counts this comment header, so it
; moves with it):
;
;  AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.
;
;  BANANA
;  X
;  THE CHARACTERS IN ERROR ARE: BANANA
;    20  UNKNOWN OPTION.
;
; The control: deleting the `BANANA` line and nothing else is ACCEPTED, exit 0,
; OBJECTIVE FUNCTION VALUE: 213.51174651518426 - the same value arm A's control
; gives, as it must, both being the same clean stream.
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
