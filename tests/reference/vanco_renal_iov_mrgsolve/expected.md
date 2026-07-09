# Vancomycin renal-decline x IOV (CRCL covariate x per-occasion kappa on CL) TDM titration reference (mrgsolve)

Frozen output of `vanco_renal_iov_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the COMPOSITION of the reactive `[adaptive_dosing]` time-varying-covariate (#700)
and inter-occasion-variability (#701) paths in `examples/adaptive_vanco_renal_iov.ferx`.
NONMEM has no feedback dosing, so mrgsolve is the comparator.

## Anchor form: deterministic dose-for-dose replay with declining CRCL + reconstructed kappa

Clearance depends on BOTH a declining renal covariate and a per-occasion kappa:
CL = TVCL * (CRCL/100) * exp(eta_CL + kappa_CL), TVCL=4 L/h, V=80 L, Omega_IOV=0.09.
The kappa is *random* (drawn per decision window on a seeded RNG substream) and its
substream is model-independent, so it is IDENTICAL to the #701 IOV anchor's kappa
(seed 20260708; the Rust reconstruction is pinned by the
`adaptive_iov_matches_predict_iov_with_reconstructed_kappa` unit test). This anchor
feeds mrgsolve the SAME declining CRCL AND the SAME per-occasion clearance and
replays ferx's realized dose ladder, so the cross-validated thing is the
covariate x occasion -> CL -> trajectory *mechanism* composed: an independent ODE
engine (LSODA) integrating the identical piecewise-constant system ferx's RK45 does.

This is the composition of the two single-effect anchors (renal #700, IOV #701):
same two-segment-per-day integration, but the piecewise-constant CL now folds in
BOTH the covariate decline and the per-occasion kappa. Each day's infusion runs on
THIS occasion's (CRCL, kappa) and the decay on the NEXT occasion's (CRCL, kappa)
(end-of-interval convention ferx's per-segment PK uses).

`auc_target` is intentionally absent: its exposure metric integrates a dense grid
from a single frozen PK snapshot, which is silently wrong when CL changes across the
horizon (a drifting covariate OR a per-occasion kappa), so it is a typed error for a
time-varying-covariate / IOV subject (#700/#701).

## Realized ladder (ferx doses + declining CRCL + reconstructed kappa, mrgsolve troughs)

| decision | time (h) | CRCL | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 120 | 0.132745 | 5.481396 | 0.000000 | 625.000000 |
| 1 | 24 | 115 | -0.199609 | 3.767629 | 2.556064 | 781.250000 |
| 2 | 48 | 108 | 0.051585 | 4.548687 | 3.239114 | 976.562500 |
| 3 | 72 | 98 | 0.317164 | 5.383046 | 3.175663 | 1220.703125 |
| 4 | 96 | 88 | 0.004488 | 3.535827 | 6.413905 | 1525.878906 |
| 5 | 120 | 78 | -0.365059 | 2.165763 | 13.302792 | 1525.878906 |
| 6 | 144 | 68 | 0.324978 | 3.764475 | 10.762650 | 1525.878906 |
| 7 | 168 | 60 | -0.222152 | 1.921902 | 16.631220 | 1144.409180 |
| 8 | 192 | 54 | 0.373939 | 3.139445 | 12.316018 | 1144.409180 |
| 9 | 216 | 49 | -0.204405 | 1.597657 | 16.342389 | 858.306885 |
| 10 | 240 | 45 | 0.048067 | 1.888630 | 15.479172 | 643.730164 |
| 11 | 264 | 42 | -0.231695 | 1.332554 | 15.727982 | 482.797623 |
| 12 | 288 | 41 | -0.339564 | 1.167811 | 15.334823 | 362.098217 |
| 13 | 312 | 40 | -0.179133 | 1.337589 | 13.346793 | 362.098217 |

ferx (`tests/adaptive_vanco_renal_iov_anchor.rs`) runs the reactive driver live at
the same seed (declining CRCL x reconstructed kappa), computes its troughs (RK45),
and asserts they match these mrgsolve troughs (LSODA) to a small cross-solver
tolerance. Both engines integrate the identical composed piecewise-constant CL, so
the whole trough trajectory agrees.
