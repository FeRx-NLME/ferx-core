; #1445 anchor: sparse 2-cpt IV infusion, BLOCK(3) IIV, combined residual error.
;
; WHAT THIS ANCHORS. The SAEM estimate of the *additive* half of a
; `combined(PROP, ADD)` residual error on sparse data — the object #1445 is
; about. The equivalent NONMEM run is `METHOD=SAEM` with the same combined
; error, so no AGENTS.md anchor exception applies and none is claimed.
;
; DATA. `data/saem_sparse_combined.csv` is the byte-for-byte fixture the ferx
; Tier-3 test builds in-process (tests/saem_combined_error.rs, SPARSE_MODEL +
; `sparse_simulated_population`), written out to six decimals;
; `committed_nonmem_csv_still_describes_the_fixture` fails if the two drift.
; 300 subjects, 464 observations, median 1 obs/subject, q8h 0.5 h infusions.
;
; SIMULATION TRUTH. CL 4, V1 22, Q 12 (fixed), V2 13; Omega diag
; 0.17 / 0.15 / 0.69 with 0.02 off-diagonals; residual combined, proportional
; SD 0.13 and additive SD 1.8. Inits below are those truths, so the run measures
; where the estimator settles, not whether it can travel.
;
; RESULTS (NONMEM 7.5.1, nm751, read from the .ext — a SAEM+IMP chain truncates
; its .lst while printing):
;   nonmem_anchor/results/saem_sparse_combined_saem.ext          ISAMPLE=10
;     SIGMA(1,1) 1.87565E-02 -> prop SD 0.13696;  SIGMA(2,2) 2.37945 -> add SD 1.54255
;     IMP -2 log L 3180.797
;   nonmem_anchor/results/saem_sparse_combined_saem_isample2.ext ISAMPLE=2 (NONMEM default)
;     prop SD 0.14912; add SD 1.64628; IMP -2 log L 3185.844
;     — and OMEGA(2,2) 8.8e-4 / OMEGA(3,3) 0.0319 against truths 0.15 / 0.69:
;       ISAMPLE=2 collapses the weakly identified Omegas on this design, which is
;       why the ISAMPLE=10 arm is the headline and both are kept.
; The additive SD is the point: NONMEM SAEM lands at 1.54, ferx SAEM after #1445
; at 1.42 [1.11, 1.75] over 8 seeds, and ferx SAEM before #1445 at 0.0025-0.011.
;
; REPRODUCE (licensed NONMEM in the `pmx` container):
;   docker exec pmx sh -lc 'mkdir -p /tmp/a && cd /tmp/a && \
;     /opt/NONMEM/nm_current/run/nmfe75 saem_sparse_combined_saem.ctl out.lst'
; after copying this file and the CSV in beside each other (adjust $DATA).
$PROBLEM 1445 sparse combined-error SAEM anchor
$INPUT ID TIME AMT RATE DV MDV EVID CMT
$DATA saem_sparse_combined.csv IGNORE=@
$SUBROUTINE ADVAN3 TRANS4

$PK
  MU_1 = LOG(THETA(1))
  MU_2 = LOG(THETA(2))
  MU_3 = LOG(THETA(3))
  CL = EXP(MU_1 + ETA(1))
  V1 = EXP(MU_2 + ETA(2))
  Q  = 12.0
  V2 = EXP(MU_3 + ETA(3))
  S1 = V1

$ERROR
  IPRED = F
  Y = IPRED*(1 + EPS(1)) + EPS(2)

$THETA
  (0.2,  4.0, 40.0)   ; 1 TVCL
  (2.0, 22.0, 200.0)  ; 2 TVV1
  (1.0, 13.0, 200.0)  ; 3 TVV2

$OMEGA BLOCK(3)
  0.17
  0.02  0.15
  0.02  0.02  0.69

$SIGMA
  0.0169   ; EPS(1) proportional, variance = 0.13^2
$SIGMA
  3.24     ; EPS(2) additive,     variance = 1.8^2

$ESTIMATION METHOD=SAEM INTERACTION NBURN=1500 NITER=2500 ISAMPLE=10 SEED=1445 PRINT=250 CTYPE=0
$ESTIMATION METHOD=IMP INTERACTION EONLY=1 NITER=10 ISAMPLE=3000 PRINT=1 SEED=1445
