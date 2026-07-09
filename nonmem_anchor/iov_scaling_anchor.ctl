$PROBLEM IOV + [scaling] occasion-less anchor (#723 review): obs_scale applied with no active occasions
; An IOV-on-V model run OCCASION-LESS: OCC=0 for every record, so IF(OCC.EQ.k)
; never fires and KAPPA=0. This is exactly the case ferx hits when an IOV dataset
; is read without an iov_column (kappa collapses to 0). obs at t=0 after an EVID=4
; reset+bolus => IPRED = AMT/V exactly; $ERROR divides by 1000, mirroring ferx
; [scaling] obs_scale = 1000. Hence
;   log IPRED = log(AMT) - log(TVV) - ETA_V   (KAPPA=0),  minus log(1000) from scaling
; so mean(log IPRED) = log(AMT/TVV/1000) = log(0.01) = -4.60517 and
; Var(log IPRED) = OMEGA(2,2) = 0.04 (BSV on V only; no IOV component).
$INPUT ID TIME DV AMT EVID CMT MDV OCC
$DATA iov_scaling_anchor.csv IGNORE=@
$SUBROUTINES ADVAN1 TRANS2
$PK
  KAPPA = 0
  IF(OCC.EQ.1) KAPPA = ETA(3)
  IF(OCC.EQ.2) KAPPA = ETA(4)
  IF(OCC.EQ.3) KAPPA = ETA(5)
  CL = THETA(1)*EXP(ETA(1))
  V  = THETA(2)*EXP(ETA(2) + KAPPA)
  S1 = V
$ERROR
  ; obs_scale = 1000: divide the prediction by 1000 (ferx [scaling] obs_scale = 1000)
  IPRED = F/1000
  Y = IPRED*(1 + EPS(1))
$THETA
  1.0    ; TVCL
  10.0   ; TVV
$OMEGA
  0.09   ; ETA(1) CL BSV
  0.04   ; ETA(2) V  BSV
$OMEGA BLOCK(1)
  0.04   ; ETA(3) IOV variance (inert here: OCC=0 => KAPPA=0)
$OMEGA BLOCK(1) SAME  ; ETA(4) occasion 2
$OMEGA BLOCK(1) SAME  ; ETA(5) occasion 3
$SIGMA
  0.0025 ; proportional residual, SD 0.05 (IPRED is noise-free; kept for parity)
$SIMULATION (20260708) ONLYSIMULATION SUBPROBLEMS=1
$TABLE ID TIME OCC EVID MDV IPRED Y NOPRINT NOAPPEND ONEHEADER FILE=iov_scaling_anchor.tab
