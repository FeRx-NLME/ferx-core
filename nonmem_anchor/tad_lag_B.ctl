$PROBLEM #1070 TAD-dependent DES x estimated lagtime with IIV -- two doses
; Multi-dose companion to runA. Two doses (t=0, t=12), each arriving at
; t_dose+ALAG1, so time-after-dose switches anchor at the SECOND arrival --
; strictly inside a record-to-record interval, and carrying an eta.
;
; NONMEM has no TAD builtin in $DES, so the anchor is carried explicitly. Dose
; times are fixed and known for this dataset, so TDOS is derived directly from T
; and MTIME(1) exists only to place an integration break exactly at the second
; arrival (the point where TDOS jumps).
;
; MTIME(1) is set UNCONDITIONALLY. Two earlier constructions were measured wrong
; against an independent RK4 reference (runBdet/truth.py, OMEGA 0 FIX so IPRED is
; a pure function) and are kept in runBdet/ as det_B*.ctl:
;   - TDP/TDN + MPAST(1) with MTIME set at every record: 40% IPRED error.
;   - the same with MTIME set only inside IF(EVID.EQ.1): NM WARNING 3, MTIME is
;     zero whenever the condition fails, and the run does not execute at all.
; This construction reproduces the reference to table precision (3.3e-5).
;
$INPUT ID TIME DV MDV EVID AMT CMT WT
$DATA tad_lag_B.csv IGNORE=@
;
; WT carries a FIXED-at-zero exponent: (WT/70)**THETA(5) with THETA(5)=0 FIX is
; exactly 1, so WT has no effect on any prediction. It is present only because
; ferx routes a subject to its event-driven predictor when a covariate column
; actually varies within the subject, and the plain (dense) predictor currently
; returns NaN for a TAD-reading RHS under a lagtime (ferx #1110). Because the
; exponent is zero, every per-event covariate snapshot is identical, so this
; anchor does not exercise the time-varying-covariate snapshot convention and
; cannot be invalidated by ferx #1073's change to where the timeline breaks.
$SUBROUTINES ADVAN13 TOL=9
$MODEL COMP=(CENTRAL,DEFDOSE)
$PK
  CL    = THETA(1)*(WT/70)**THETA(5)*EXP(ETA(1))
  V     = THETA(2)
  ALAG1 = THETA(4)*EXP(ETA(2))
  MTIME(1) = 12.0 + ALAG1
  MTDIFF   = 1
$DES
  TDOS = ALAG1
  IF (T.GE.12.0+ALAG1) TDOS = 12.0 + ALAG1
  TADX = T - TDOS
  DADT(1) = -(CL/V)*A(1)*(1.0 + THETA(3)*TADX)
$ERROR
  IPRED = A(1)/V
  Y     = IPRED*(1.0 + EPS(1))
$THETA
 (0.001, 1.0)   ; TVCL
 (0.001, 20.0)  ; TVV
 (0.0,   0.3)   ; THETA_TAD
 (0.001, 0.5)   ; TVLAG
 (0.0 FIX)      ; WT exponent -- zero, see header
$OMEGA
 0.09   ; ETA_CL
 0.02   ; ETA_LAG
$SIGMA
 0.01   ; proportional, variance
$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC PRINT=1
$TABLE ID TIME IPRED CWRES NOPRINT ONEHEADER FILE=tad_lag_B.tab
