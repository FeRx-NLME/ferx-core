$PROBLEM two_cpt_oral_cov: base model for the PsN scm anchor (#1180)
; Same model as base.ferx: two-compartment oral, eta on CL and V1, FOCEI,
; proportional error. CMT and RATE are dropped so NONMEM uses ADVAN4's
; defaults (dose into 1 = depot, observation from 2 = central), which is
; what CMT=1 on every row means to ferx's `two_cpt_oral` template.
$INPUT ID TIME DV EVID AMT CMT=DROP RATE=DROP MDV WT CRCL
$DATA ../../../../../data/two_cpt_oral_cov.csv IGNORE=@
$SUBROUTINES ADVAN4 TRANS4
$PK
TVCL = THETA(1)
TVV1 = THETA(2)
TVQ  = THETA(3)
TVV2 = THETA(4)
TVKA = THETA(5)
CL = TVCL*EXP(ETA(1))
V2 = TVV1*EXP(ETA(2))
Q  = TVQ
V3 = TVV2
KA = TVKA
S2 = V2
$ERROR
IPRED = F
Y = IPRED*(1+EPS(1))
$THETA (0.1, 4.0, 100)
$THETA (1, 40, 500)
$THETA (0.1, 8, 100)
$THETA (1, 80, 500)
$THETA (0.01, 1.0, 10)
$OMEGA 0.15
$OMEGA 0.15
$SIGMA 0.0016
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=10 NOABORT
