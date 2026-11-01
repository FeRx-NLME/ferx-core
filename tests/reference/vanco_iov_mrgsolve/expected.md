# Vancomycin IOV (per-occasion κ on CL) TDM titration reference (mrgsolve)

Frozen output of `vanco_iov_mrgsolve.R` (mrgsolve 1.7.2). External anchor for
the reactive `[adaptive_dosing]` **inter-occasion variability** path (a fresh
per-occasion κ on clearance, #701) in `examples/adaptive_vanco_iov.ferx`. NONMEM
has no feedback dosing, so mrgsolve is the comparator.

## Anchor form: deterministic dose-for-dose replay with reconstructed κ

κ is *random* (drawn per decision window on a seeded RNG substream), so an
independent mrgsolve draw could never match ferx. Instead this anchor
reconstructs ferx's EXACT per-occasion κ (seed 20260708; the Rust reconstruction
is pinned by the `adaptive_iov_matches_predict_iov_with_reconstructed_kappa` unit
test) and injects the SAME per-occasion clearance into mrgsolve, replaying ferx's
realized dose ladder. What is cross-validated is therefore the occasion → CL →
trajectory *mechanism*: an independent ODE engine (LSODA) integrating the
identical piecewise-constant system ferx's RK45 does. This is the #701 analogue
of the #700 renal-covariate anchor — same two-segment-per-day integration, but
the piecewise-constant CL comes from the per-occasion κ instead of a covariate.

1-cpt IV vancomycin, once-daily 1-h infusion. Params (identical in the ferx
model): TVCL=4 L/h at κ=0, V=80 L, CL = TVCL * exp(η_CL + κ_CL), Ω_IOV=0.09.
Each day's infusion runs on THIS occasion's CL and the decay runs on the NEXT
occasion's CL (end-of-interval convention ferx's per-segment IOV PK uses). The
per-occasion κ swings CL across the horizon, so the controller re-titrates each
day off the trough it observes.

`auc_target` is supported here since #943: each window is integrated under the
per-occasion CL its trajectory runs under. The per-window AUC24 below is its
reference; the pct-in-window trough metric (`target_window`) is retained.

## Realized ladder (ferx doses + reconstructed κ, mrgsolve troughs)

| decision | time (h) | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.132745 | 4.567830 | 0.000000 | 625.000000 |
| 1 | 24 | -0.199609 | 3.276200 | 2.984437 | 781.250000 |
| 2 | 48 | 0.051585 | 4.211748 | 3.677836 | 976.562500 |
| 3 | 72 | 0.317164 | 5.492904 | 3.139722 | 1220.703125 |
| 4 | 96 | 0.004488 | 4.017986 | 5.628443 | 1525.878906 |
| 5 | 120 | -0.365059 | 2.776619 | 10.884727 | 1525.878906 |
| 6 | 144 | 0.324978 | 5.535993 | 5.820004 | 1907.348633 |
| 7 | 168 | -0.222152 | 3.203169 | 11.531568 | 1907.348633 |
| 8 | 192 | 0.373939 | 5.813787 | 6.338297 | 2384.185791 |
| 9 | 216 | -0.204405 | 3.260525 | 13.820478 | 2384.185791 |
| 10 | 240 | 0.048067 | 4.196956 | 12.610967 | 2384.185791 |
| 11 | 264 | -0.231695 | 3.172749 | 16.604240 | 1788.139343 |
| 12 | 288 | -0.339564 | 2.848319 | 16.747015 | 1341.104507 |
| 13 | 312 | -0.179133 | 3.343973 | 12.418945 | 1341.104507 |

ferx (`tests/adaptive_vanco_iov_anchor.rs`) reconstructs the same per-occasion κ,
computes its troughs live (RK45), and asserts they match these mrgsolve troughs
(LSODA) to a small cross-solver tolerance. Both engines integrate the identical
piecewise-constant CL, so the whole trough trajectory agrees.

## Per-window AUC24

The area under CENT/V over each closed inter-decision window, from an AUC compartment (dxdt_AUC = CENT/V) reset at the start of each window (mrgsolve 2.0.1). It is the external reference for `auc_target` (#943).

| window | days [t0,t1] (h) | AUC24 (mg*h/L) |
|---:|---:|---:|
| 0 | [0, 24] | 117.894224 |
| 1 | [24, 48] | 172.322318 |
| 2 | [48, 72] | 185.623432 |
| 3 | [72, 96] | 254.258103 |
| 4 | [96, 120] | 398.101469 |
| 5 | [120, 144] | 348.818476 |
| 6 | [144, 168] | 452.808855 |
| 7 | [168, 192] | 399.534846 |
| 8 | [192, 216] | 547.645310 |
| 9 | [216, 240] | 591.129975 |
| 10 | [240, 264] | 650.768225 |
| 11 | [264, 288] | 623.777441 |
| 12 | [288, 312] | 504.594452 |
