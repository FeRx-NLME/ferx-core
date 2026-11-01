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

`auc_target` is supported here since #943: each window is integrated under the
covariate- and occasion-driven CL its trajectory runs under. The per-window AUC24 below is its
reference; the pct-in-window trough metric (`target_window`) is retained.

## Realized ladder (ferx doses + declining CRCL + reconstructed kappa, mrgsolve troughs)

| decision | time (h) | CRCL | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 120 | 0.132745 | 5.481396 | 0.000000 | 625.000000 |
| 1 | 24 | 115 | -0.199609 | 3.767629 | 2.583322 | 781.250000 |
| 2 | 48 | 108 | 0.051585 | 4.548687 | 3.227217 | 976.562500 |
| 3 | 72 | 98 | 0.317164 | 5.383046 | 3.153547 | 1220.703125 |
| 4 | 96 | 88 | 0.004488 | 3.535827 | 6.492767 | 1525.878906 |
| 5 | 120 | 78 | -0.365059 | 2.165763 | 13.486416 | 1525.878906 |
| 6 | 144 | 68 | 0.324978 | 3.764475 | 10.672210 | 1525.878906 |
| 7 | 168 | 60 | -0.222152 | 1.921902 | 16.841537 | 1144.409180 |
| 8 | 192 | 54 | 0.373939 | 3.139445 | 12.255352 | 1144.409180 |
| 9 | 216 | 49 | -0.204405 | 1.597657 | 16.535773 | 858.306885 |
| 10 | 240 | 45 | 0.048067 | 1.888630 | 15.543945 | 643.730164 |
| 11 | 264 | 42 | -0.231695 | 1.332554 | 15.862112 | 482.797623 |
| 12 | 288 | 41 | -0.339564 | 1.167811 | 15.456498 | 362.098217 |
| 13 | 312 | 40 | -0.179133 | 1.337589 | 13.403205 | 362.098217 |

ferx (`tests/adaptive_vanco_renal_iov_anchor.rs`) runs the reactive driver live at
the same seed (declining CRCL x reconstructed kappa), computes its troughs (RK45),
and asserts they match these mrgsolve troughs (LSODA) to a small cross-solver
tolerance. Both engines integrate the identical composed piecewise-constant CL, so
the whole trough trajectory agrees.

## Per-window AUC24

The area under CENT/V over each closed inter-decision window, from an AUC compartment (dxdt_AUC = CENT/V) reset at the start of each window (mrgsolve 2.0.1). It is the external reference for `auc_target` (#943).

| window | days [t0,t1] (h) | AUC24 (mg*h/L) |
|---:|---:|---:|
| 0 | [0, 24] | 111.033800 |
| 1 | [24, 48] | 160.428342 |
| 2 | [48, 72] | 182.509348 |
| 3 | [72, 96] | 269.686668 |
| 4 | [96, 120] | 446.210865 |
| 5 | [120, 144] | 465.141934 |
| 6 | [144, 168] | 537.141299 |
| 7 | [168, 192] | 481.392114 |
| 8 | [192, 216] | 501.969692 |
| 9 | [216, 240] | 496.472543 |
| 10 | [240, 264] | 463.978639 |
| 11 | [264, 288] | 441.207326 |
| 12 | [288, 312] | 393.515239 |
