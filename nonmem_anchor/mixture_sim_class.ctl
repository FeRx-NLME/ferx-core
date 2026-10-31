$PROBLEM #2163: a [mixture] model under $SIM draws each subject's class
; Two classes, class 2 clearing 4x faster, with its own IIV and residual SD (ferx
; `omega(2)` / `sigma(2)`: here ETA(2) and SDP). The class-1 probability depends on
; WT, which spans 50-110 kg, so P(1) runs from 0.14 to 0.76 across the population.
; ONLYSIM: no estimation, every THETA/OMEGA/SIGMA FIX. Data from
; make_mixture_sim_class.py, summarized by the same script into
; mixture_sim_class_summary.csv; compared in tests/mixture_simulate_nonmem_anchor.rs.
$INPUT ID TIME DV AMT EVID CMT WT
$DATA mixture_sim_class.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
; $MIX reads data items only through $CONTR DATA; WT is the value on the subject's
; first observation record, and WT is constant within a subject.
$CONTR DATA=(WT)

$MIX
  NSPOP = 2
  P(1) = 1 / (1 + EXP(-(THETA(3) + THETA(4) * (WT - 70))))
  P(2) = 1 - P(1)

$PK
  MIXN = MIXNUM
  IF (MIXNUM.EQ.1) THEN
    CL  = THETA(1) * EXP(ETA(1))
    SDP = SQRT(0.04)
  ELSE
    CL  = 4 * THETA(1) * EXP(ETA(2))
    SDP = SQRT(0.09)
  ENDIF
  V  = THETA(2)
  S1 = V

$ERROR
  IPRED = F
  Y = IPRED * (1 + SDP * EPS(1))

$THETA
  1.0   FIX ; TVCL
  10.0  FIX ; TVV
  -0.85 FIX ; MIXL, the class-1 logit at WT = 70
  0.05  FIX ; BWT

$OMEGA 0.09 FIX ; ETA_CL, class 1
$OMEGA 0.36 FIX ; ETA_CL, class 2 (ferx `omega(2) ETA_CL ~ 0.36`)
$SIGMA 1 FIX    ; scaled by SDP: Var = 0.04 (class 1), 0.09 (class 2)

$SIMULATION (2163) (2164 UNIFORM) ONLYSIM NSUB=1

$TABLE ID TIME EVID WT MIXN IPRED DV NOPRINT NOAPPEND ONEHEADER
  FILE=mixture_sim_class.tab FORMAT=s1PE15.8
