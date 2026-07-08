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

`auc_target` is intentionally absent: its exposure metric integrates a dense grid
from a single frozen PK snapshot, which is silently wrong when CL switches per
occasion, so it is a typed error for an IOV (`kappa`) subject (#701). The
pct-in-window trough metric (`target_window`) is per-occasion aware and retained.

## Realized ladder (ferx doses + reconstructed κ, mrgsolve troughs)

| decision | time (h) | kappa_CL | CL (L/h) | trough (mg/L) | dose (mg) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.132745 | 4.567830 | 0.000000 | 625.000000 |
| 1 | 24 | -0.199609 | 3.276200 | 2.960637 | 781.250000 |
| 2 | 48 | 0.051585 | 4.211748 | 3.697443 | 976.562500 |
| 3 | 72 | 0.317164 | 5.492904 | 3.174336 | 1220.703125 |
| 4 | 96 | 0.004488 | 4.017986 | 5.578850 | 1525.878906 |
| 5 | 120 | -0.365059 | 2.776619 | 10.761070 | 1525.878906 |
| 6 | 144 | 0.324978 | 5.535993 | 5.932958 | 1907.348633 |
| 7 | 168 | -0.222152 | 3.203169 | 11.376082 | 1907.348633 |
| 8 | 192 | 0.373939 | 5.813787 | 6.447514 | 2384.185791 |
| 9 | 216 | -0.204405 | 3.260525 | 13.606025 | 2384.185791 |
| 10 | 240 | 0.048067 | 4.196956 | 12.646158 | 2384.185791 |
| 11 | 264 | -0.231695 | 3.172749 | 16.481410 | 1788.139343 |
| 12 | 288 | -0.339564 | 2.848319 | 16.646880 | 1341.104507 |
| 13 | 312 | -0.179133 | 3.343973 | 12.439516 | 1341.104507 |

ferx (`tests/adaptive_vanco_iov_anchor.rs`) reconstructs the same per-occasion κ,
computes its troughs live (RK45), and asserts they match these mrgsolve troughs
(LSODA) to a small cross-solver tolerance. Both engines integrate the identical
piecewise-constant CL, so the whole trough trajectory agrees.
