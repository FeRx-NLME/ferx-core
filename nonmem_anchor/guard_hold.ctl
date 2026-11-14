$PROBLEM Held covariance: ETA_KA FIX at ferx's internal rail (#2797)
; ferx holds a variance collapsed onto its internal lower guard
; (omega = exp(-12) = 6.1442123533282098E-6) in the covariance step, so every
; other standard error is conditional on it at its estimate. "Held" means FIX,
; and this is the NONMEM run of exactly that: fixture A
; (tests/data/guard_hold/sim_oral_no_ka_iiv.csv, simulated with no KA
; variability, generator beside it) at ferx's FOCEI estimate, with the KA
; variance FIXed on the rail, MAXEVAL=0 and $COV MATRIX=R. The data here is the
; same file with the observations keyed to CMT=2 (ADVAN2's central), which is
; NONMEM's convention for the compartment ferx reads with one_cpt_oral.
;
; $COV runs at SIGL=12 (reference_nonmem_cov_precision: $COV SEs move with the
; estimation precision). ferx's comparison is in tests/guard_hold_nonmem_anchor.rs.
$INPUT ID TIME DV EVID AMT CMT RATE MDV OCC
$DATA guard_hold.csv IGNORE=@
$SUBROUTINES ADVAN2 TRANS2

$PK
CL = THETA(1)*EXP(ETA(1))
V  = THETA(2)*EXP(ETA(2))
KA = THETA(3)*EXP(ETA(3))
S2 = V

$ERROR
IPRED = F
Y = F*(1 + EPS(1))

$THETA (0.001, 0.12250099819205257, 10.0)  ; TVCL
$THETA (0.1, 7.753670975867449, 500.0)     ; TVV
$THETA (0.01, 0.9982755334099482, 50.0)    ; TVKA
$OMEGA 0.0792688562315328                  ; ETA_CL
$OMEGA 0.05944155243978235                 ; ETA_V
$OMEGA 6.1442123533282098E-6 FIX           ; ETA_KA, on ferx's rail
$SIGMA 0.01013886825572548                 ; PROP_ERR^2 (ferx: SD 0.10069194732313741)

$ESTIMATION METHOD=1 INTERACTION MAXEVAL=0 POSTHOC SIGL=12 NSIG=4 PRINT=1
$COVARIANCE MATRIX=R SIGL=12 PRINT=E
$TABLE ID TIME DV PRED IPRED NOPRINT ONEHEADER FORMAT=s1PE23.16 FILE=guard_hold.tab
