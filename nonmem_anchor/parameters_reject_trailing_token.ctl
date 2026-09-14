; ferx-core #1377 - NM-TRAN reject control, arm A: a trailing token on a $OMEGA
; BLOCK record.
;
; NOT a fit anchor, and deliberately so. #1377 makes ferx REJECT a scale tag -
; and any other trailing text - on a `block_omega` / `block_sigma` /
; `block_kappa` declaration instead of dropping it in silence, so the object
; under test is a PARSE OUTCOME, not a number: there is no OFV to compare. What
; this stream anchors is that the comparator treats the same shape as an error
; rather than as something to ignore, which is the whole question #1377 asks.
;
; The stream is a valid warfarin ADVAN2 evaluation in every respect except the
; single token `FOO` appended to the $OMEGA BLOCK record. Delete that token and
; NM-TRAN accepts the stream - that one-token difference is the experiment, in
; the repo's "vary one input at a time" discipline, so no separate baseline file
; is needed.
;
; Measured first-hand, not inherited: nmfe76 (/opt/NONMEM/nm760/run/nmfe76),
; NONMEM VERSION 7.6.0 (nm760), docker image nonmemdocker:V0.1
; (sha256:2ee46060f1e0). Run from nonmem_anchor/ against ../data/warfarin.csv.
; Rejected, nmfe exit 107. Verbatim, leading spaces as printed:
;
;  AN ERROR WAS FOUND IN THE CONTROL STATEMENTS.
;
; AN ERROR WAS FOUND ON LINE <N> AT THE APPROXIMATE POSITION NOTED:
;  $OMEGA BLOCK(2) 0.09 0.02 0.04 FOO
;                                 X
;  THE CHARACTERS IN ERROR ARE: FOO
;    20  UNKNOWN OPTION.
;
; <N> is spelled out as a number in the real output. It counts every line of the
; file, this comment header included, so it moves whenever the header does - the
; one part of the message not worth pinning here.
;
; The control: deleting `FOO` and nothing else is ACCEPTED, exit 0, OBJECTIVE
; FUNCTION VALUE: 213.51174651518426.
;
; The ferx equivalent is
;   block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04] banana
; which reports E_PARSE quoting the line. A `(sd)` in the same position reports
; E_BLOCK_VARIANCE_ONLY instead - see tests/check_command.rs.
$PROBLEM #1377 arm A - trailing token on a $OMEGA BLOCK record (NM-TRAN must reject)

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
$OMEGA BLOCK(2) 0.09 0.02 0.04 FOO
$SIGMA 0.01 FIX

$ESTIMATION MAXEVAL=0 METHOD=0 POSTHOC NOABORT
