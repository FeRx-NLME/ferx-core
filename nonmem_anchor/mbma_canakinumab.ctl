$PROBLEM canakinumab RA MBMA, Bracis et al. 2026 Case Study 2 final model (r03), FOCE-I (#1913)
; The NONMEM twin of tests/data/mbma_canakinumab/mbma_canakinumab.ferx.
; The data are NOT in the repository (CC BY-NC-ND 4.0) and must never be. Copy this
; file to a run folder OUTSIDE the repository and provision the data there:
;   python3 tools/mbma_canakinumab_fetch.py --out <run dir>/mbma_canakinumab.csv
; Written for the anchor image: nm3 mbma_canakinumab.ctl
;
; NONMEM has no weight on an $OMEGA, so the between-arm kappa is written out by
; hand: one ETA per arm (ARM 0 / ARM 1, OMEGA SAME), divided by SQRT(NARM). The
; residual SD is THETA(7) scaled by 1/SQRT(NARM) with SIGMA fixed to 1 -- the same
; likelihood as ferx's `additive(ADD_ERR) weight = 1 / sqrt(NARM)`, Jacobian included,
; so the two objective values are directly comparable.
$INPUT ID ARM TIME DV NARM ABATA ADALI CANAKI MDV
$DATA mbma_canakinumab.csv IGNORE=@

$PRED
OC1 = 0
IF (ARM.EQ.0) OC1 = 1
OC2 = 1 - OC1
KAP = (OC1*ETA(2) + OC2*ETA(3)) / SQRT(NARM)

LTV = LOG(THETA(1)/(1 - THETA(1)))
LEM = LTV + THETA(2)*ABATA + THETA(3)*ADALI + THETA(4)*CANAKI + ETA(1) + KAP
EM  = EXP(LEM)/(1 + EXP(LEM))
T50 = THETA(5)*EXP(THETA(6)*ADALI)

ACR = EM*TIME/(TIME + T50)
IF (ACR.LT.0.01) ACR = 0.01
IF (ACR.GT.0.99) ACR = 0.99
IPRED = LOG(ACR/(1 - ACR))

W = THETA(7)/SQRT(NARM)
Y = IPRED + W*EPS(1)

$THETA
(0.001, 0.40, 0.999)  ; EMAX_POP
(-10, 1.5, 10)        ; B_EMAX_ABA
(-10, 1.0, 10)        ; B_EMAX_ADA
(-10, 1.0, 10)        ; B_EMAX_CAN
(0.1, 4.0, 100)       ; T50_POP
(-10, -0.5, 10)       ; B_T50_ADA
(0.01, 2.5)           ; ADD_ERR (SD)

$OMEGA 0.09                 ; ETA_BSV_EMAX
$OMEGA BLOCK(1) 4.0         ; KAPPA_EMAX, ARM 0
$OMEGA BLOCK(1) SAME        ; KAPPA_EMAX, ARM 1
$SIGMA 1 FIX

$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 NOABORT SIGDIG=5 FORMAT=s1PE23.16
$COVARIANCE MATRIX=R UNCONDITIONAL
