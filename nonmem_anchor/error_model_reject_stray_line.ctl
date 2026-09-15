; ferx-core #1390 - NM-TRAN comparator, arm B: a stray line inside the residual
; error definition, sitting beside a COMPLETE `Y = ...` statement.
;
; NOT a fit anchor - a parse reject has no number to anchor, the same reason
; `parameters_reject_*.ctl` gives for #1377. What is compared is the parse
; OUTCOME.
;
; The ferx side: before #1390 an `[error_model]` line matching no statement form
; was dropped without a trace, so
;
;   [error_model]
;     DV ~ proportional(PROP_ERR)
;     banana
;
; validated clean with `ferx check` reporting VALID. The stray line beside a
; complete statement is the shape that matters - a block with ONLY the stray
; line already failed, late and unhelpfully, with "No error model found in
; [error_model] block", so the defect is only visible when a valid statement is
; there to be fitted instead.
;
; NM-TRAN rejects the same shape. `BANANA` on its own line in $ERROR, with a
; complete `Y = IPRED * (1 + EPS(1))` above it, fails with nmfe exit 107.
; Verbatim, leading spaces as printed. The line number counts this comment
; header, so it moves whenever the header does - it was 18 on the bare stream
; and is 64 as committed, which is the number measured from the file in this
; directory:
;
;  AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.
;
; AN ERROR WAS FOUND ON LINE 64 AT THE APPROXIMATE POSITION NOTED:
;    BANANA
;    X
;   202  FORTRAN SYNTAX IS INCORRECT OR INAPPROPRIATE IN THIS CONTEXT.
;
; READ THE ASYMMETRY BEFORE QUOTING THIS STREAM. $ERROR is Fortran, so NM-TRAN
; rejects `BANANA` as a *syntax* error (202) rather than as "a line matching no
; error-model form" - it has no notion of a declaration form here. So what this
; supports is the narrower claim that a stray line in the residual-error region
; is an error and not something to drop, NOT that NM-TRAN classifies it the way
; ferx does. ferx's `[error_model]` is a small fixed grammar of statements and
; can say which forms it accepts, which is a difference between the two parsers
; rather than NONMEM backing for ferx's message.
;
; The control: deleting the `BANANA` line and nothing else is ACCEPTED, exit 0,
; OBJECTIVE FUNCTION VALUE: 215.06724369784769.
;
; Measured first-hand: nmfe75 (/opt/NONMEM/nm751/run/nmfe75), NONMEM 7.5.1,
; licensed `pmx` container image a8b78e6253e5, 2026-09-15.
$PROBLEM #1390 arm B - stray line in $ERROR beside a complete Y (NM-TRAN rejects)

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
  BANANA

$THETA 0.134 FIX  8.11 FIX  1.35 FIX
$OMEGA 0.09 0.04
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
