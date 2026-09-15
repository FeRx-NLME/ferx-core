; ferx-core #1390 - NM-TRAN comparator, arm A: an option keyword written
; WITHOUT its `=`. This is a MEASURED NEGATIVE RESULT. Read it before quoting
; NONMEM in support of ferx's `[fit_options]` rule, because it does not support
; it - it says the comparator cannot express the object.
;
; NOT a fit anchor. A parse reject has no number to anchor; what is compared is
; the parse OUTCOME, the same thing `parameters_reject_*.ctl` compares for
; #1377.
;
; The ferx side of the question: `[fit_options]` is uniformly `key = value`, and
; before #1390 a line without the `=` was DROPPED - `method saem` vanished, the
; fit ran the default estimator, and `ferx check` reported the file VALID. The
; natural comparator question is "does NM-TRAN require the `=` on a $ESTIMATION
; option?".
;
; It does not. This stream writes `MAXEVAL 0` in place of `MAXEVAL=0` and
; NM-TRAN ACCEPTS it, exit 0, and HONOURS it: the .lst reports
;
;   ESTIMATION STEP OMITTED:                 YES
;
; so the space is read as the `=`. A NONMEM record is a token stream running to
; the next `$`, in which whitespace and `=` are both separators; ferx's
; `[fit_options]` is a line-oriented `key = value` block. The two grammars
; differ, and this arm is here to record that measurement rather than to leave a
; reviewer to assume NM-TRAN backs the ferx rule.
;
; So #1390's `[fit_options]` half is anchored on its own terms - an exact,
; hand-checked message and the `parse_full_model` control that the accepted
; spelling still selects SAEM (`a_model_file_with_a_missing_equals_no_longer_-
; parses_clean`, src/parser/model_parser_tests.rs) - not on a NONMEM run. Arm B
; (`error_model_reject_stray_line.ctl`) is the half NM-TRAN *does* speak to.
;
; The control is arm B's control stream, which is this stream with `MAXEVAL=0`:
; accepted, exit 0, OBJECTIVE FUNCTION VALUE: 215.06724369784769.
;
; Measured first-hand: nmfe75 (/opt/NONMEM/nm751/run/nmfe75), NONMEM 7.5.1,
; licensed `pmx` container image a8b78e6253e5, 2026-09-15.
$PROBLEM #1390 arm A - $ESTIMATION option with its `=` left out (NM-TRAN ACCEPTS)

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

$THETA 0.134 FIX  8.11 FIX  1.35 FIX
$OMEGA 0.09 0.04
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL 0 METHOD=0 POSTHOC NOABORT
