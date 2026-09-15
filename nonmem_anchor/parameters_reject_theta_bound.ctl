; ferx-core #1377 / PR #1388 review round 3 #1 - NM-TRAN reject control: a
; $THETA bound that is present but not a number.
;
; NOT a fit anchor, like the two parameters_reject_* streams beside it: the
; object under test is a PARSE OUTCOME. ferx's `theta` bound groups are the
; character class `[0-9eE.+-]+`, which admits `0.001-10.0`, `-` and `1e`, and
; each was read with `.parse().unwrap_or(1e-9 / 1e9)` - so
; `theta TVCL(50, 0.001-10.0)`, a mistyped comma, fitted in the box (1e-9, 1e9)
; with `ferx check` VALID. ferx now reports `Bad theta lower bound`. What this
; stream records is that the comparator refuses the same shape instead of
; substituting a default.
;
; The stream is the warfarin ADVAN2 MAXEVAL=0 evaluation of
; parameters_reject_trailing_token.ctl with a clean $OMEGA BLOCK and one edit:
; the first $THETA is written with bounds, and its lower bound is the joined
; `0.001-10.0`.
;
; Measured first-hand: nmfe76 (/opt/NONMEM/nm760/run/nmfe76), NONMEM VERSION
; 7.6.0 (nm760), docker image nonmemdocker:V0.1 (sha256:2ee46060f1e0), run from
; nonmem_anchor/ against ../data/warfarin.csv. Rejected, nmfe exit 107.
; Verbatim, leading spaces as printed:
;
;  AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.
;
; AN ERROR WAS FOUND ON LINE <N> AT THE APPROXIMATE POSITION NOTED:
;  $THETA (0.001-10.0, 0.134) FIX  8.11 FIX  1.35 FIX   ; CL V KA
;          X
;  THE CHARACTERS IN ERROR ARE: 0.001-
;    18  INCORRECTLY FORMED VALUE, OPTION, OR RESERVED WORD.
;
; <N> counts every line of this file, the header included, so it is not pinned.
;
; The control, one edit away: `(0.001, 0.134, 10.0) FIX` in place of
; `(0.001-10.0, 0.134) FIX` is ACCEPTED, exit 0, OBJECTIVE FUNCTION VALUE:
; 213.51174651518426 - identical to the stream with no bounds at all, since the
; theta is fixed at 0.134 inside either box.
;
; Two more spellings measured the same way, each replacing only the lower bound:
;   `(-, 0.134, 10.0)`   -> rejected, `THE CHARACTERS IN ERROR ARE: -`,
;                           `22  UNKNOWN SYMBOL.`
;   `(1e, 0.134, 10.0)`  -> rejected, `THE CHARACTERS IN ERROR ARE: 1E`,
;                           `18  INCORRECTLY FORMED VALUE, OPTION, OR RESERVED WORD.`
; and `(0.001, 0.134, 1e)` on the upper bound -> rejected, error 18 on `1E`.
;
; The ferx equivalents are `theta TVCL(50, 0.001-10.0)`, `theta TVCL(0.2, -, 10.0)`
; and `theta TVCL(0.2, 1e, 10.0)`, each reporting `Bad theta lower bound` - see
; `a_malformed_theta_bound_is_an_error_not_the_default` in
; src/parser/model_parser_tests.rs.
$PROBLEM #1377 round 3 - a non-numeric $THETA bound (NM-TRAN must reject)

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

$THETA (0.001-10.0, 0.134) FIX  8.11 FIX  1.35 FIX   ; CL V KA
$OMEGA BLOCK(2) 0.09 0.02 0.04
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
