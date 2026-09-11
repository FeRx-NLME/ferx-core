#!/usr/bin/env bash
#
# A/B benchmark for the Laplace/AGQ and FOCEI-AGQ grid response (#251).
#
#   baseline  FERX_AGQ_GRID_RESPONSE=fd        — rebuild the anchor at x ± h per free coord
#   candidate FERX_AGQ_GRID_RESPONSE=analytic  — analytic dH/dx (Exact) or dH̃/dx (GaussNewton)
#
# Both anchors are covered:
#
#   * `laplace, n_agq=1` (`HessianAnchor::Exact`) — `estimation::laplace_h_deriv`, needs the
#     covariance provider's third-order jet. `analytic` is the DEFAULT here as of #1335 (see
#     `agq::use_analytic_grid_response`); `fd` forces the old perturbed-anchor sweep for
#     comparison. It was opt-in until the block-Ω fixture was measured — the win is
#     representation-dependent, and the diagonal fixture alone could not settle it.
#
#     Session 2 (5 interleaved reps, `f2434247`, 2026-09-10, RAYON_NUM_THREADS=1) — the run
#     that flipped the default. Every arm matched OFV to four decimals:
#       diagonal-Ω  (`warfarin_laplace.ferx`):             6840→6460 calls, 0.0586→0.0556s
#         (−5%, 4/5 reps), iters 37→37 — still near parity, as below
#       block-Ω     (`warfarin_block_omega_laplace.ferx`): 11520→8160 calls, 0.0880→0.0670s
#         (−24%, 5/5 reps), iters 47→47 — the DECIDING case: a wide call-count gap
#         (20 FD rebuilds vs 13 provider evaluations) and a like-for-like iteration count
#       ODE         (`warfarin_ode_laplace.ferx`):         10270→6470 calls, 1.581→1.029s
#         (−35%, 5/5 reps), iters 56→37 — bigger, but NOT like-for-like; read the call count
#
#     Session 1 (3 interleaved reps, `8e03e076`, 2026-09-09) — block-Ω not yet run —
#       diagonal-Ω, analytic 1cpt (`warfarin_laplace.ferx`):  6840→6460 calls,  provider time
#         0.070/0.061/0.055s → 0.053/0.057/0.050s (~14% faster, all 3 reps)
#       ODE (`warfarin_ode_laplace.ferx`):                    10270→7660 calls, provider time
#         ~1.66s → ~1.20s avg (~27% faster, all 3 reps; also fewer outer iterations, 56→44,
#         so the two routes do not walk the identical path — see CHANGELOG)
#     Both cases favored `analytic` in session 1 too, but block-Ω had not been run, so
#     the default was left alone: an earlier pass on the same non-ODE case had recorded the
#     opposite sign (+34% slower). Both were real — numbers this close to parity are
#     sensitive to machine/toolchain state, which is exactly why this script exists rather
#     than a one-line claim, and why the diagonal fixture alone was never going to settle
#     it. Session 2 added block-Ω and that is what flipped the default (#1335).
#   * `focei, n_agq=3` (`HessianAnchor::GaussNewton`) — `estimation::focei_htilde_dx`, needs
#     no third order at all (`H̃` is bilinear in first-order sensitivities), so it costs one
#     extra plain `subject_sensitivities` call, independent of `n_free`. `analytic` is the
#     unconditional DEFAULT here; `fd` forces the old perturbed-anchor sweep for comparison.
#
# Both routes come out of the SAME binary, so inlining and cache state cannot differ between
# arms; runs are interleaved (baseline, candidate, …) so a machine that warms up or throttles
# biases neither arm.
#
# ODE models are benchmarked for both anchors: `laplace_h_deriv::subject_h_inner_dx` admits
# ODE (the covariance provider's third-order sweep already reaches it via
# `ode_analytical_supported`), and `focei_htilde_dx::subject_htilde_dx` does too (the base
# provider is representation-agnostic). This is the configuration where the `Exact` route is
# most likely to win: each FD-perturbed anchor rebuild is a full re-integration — confirmed
# above, ~27% less provider time on `warfarin_ode_laplace.ferx`.
#
# `FERX_PROFILE=1` prints the analytic provider's call count and total time — the
# primary metrics (see `estimation/agq.rs`). Call counts are deterministic for a given
# trajectory; provider time remains timing-sensitive. Wall time on these small fits is
# dominated by process/parse overhead and should be read as corroborating, not primary.
#
# Usage:  plans/laplace-sensitivity-speed/bench.sh [REPS]
#
# Build once, outside the timed region (a release build here is fat-LTO and ~45 min;
# `ci-test` is release-level optimisation without it):
#
#   cargo build --profile ci-test -p ferx-cli
#
set -euo pipefail

cd "$(dirname "$0")/../.."
REPS="${1:-5}"
BIN="target/ci-test/ferx"
DATA="data/warfarin.csv"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

if [[ ! -x "$BIN" && ! -x "$BIN.exe" ]]; then
  echo "build first:  cargo build --profile ci-test -p ferx-cli" >&2
  exit 1
fi
[[ -x "$BIN" ]] || BIN="$BIN.exe"

echo "commit:    $(git rev-parse --short HEAD 2>/dev/null || echo '(dirty worktree)')"
echo "toolchain: $(rustc --version)"
echo "threads:   ${RAYON_NUM_THREADS:-default}"
echo "reps:      $REPS"
echo

run() { # run <label> <model> <env-assignment...>
  local label="$1" model="$2"; shift 2
  local log="$OUT/$label.log"
  local t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  ( cd "$OUT" && FERX_PROFILE=1 "$@" "$OLDPWD/$BIN" "$OLDPWD/$model" --data "$OLDPWD/$DATA" ) \
    >"$log" 2>&1 || { echo "FAILED: $label"; sed -n '1,40p' "$log"; return 1; }
  t1=$(python3 -c 'import time;print(time.time())')
  local ofv iters provider
  # The console prints OFV to four decimals. Equality here is only at displayed
  # precision; use derivative-parity tests to assess gradient agreement.
  ofv=$(grep -Eo 'OFV[^0-9-]*(-?[0-9.]+)' "$log" | tail -1 | grep -Eo '\-?[0-9.]+$' || echo NA)
  iters=$(grep -Eio 'iterations?[^0-9]*([0-9]+)' "$log" | tail -1 | grep -Eo '[0-9]+$' || echo NA)
  provider=$(grep -o 'subject_sensitivities): [0-9]* calls, [0-9.]*s' "$log" | tail -1 || echo "NA")
  printf '%-28s %8.2fs  OFV=%-14s iters=%-5s %s\n' \
    "$label" "$(python3 -c "print($t1-$t0)")" "$ofv" "$iters" "$provider"
}

for model in plans/laplace-sensitivity-speed/warfarin_laplace.ferx \
             plans/laplace-sensitivity-speed/warfarin_block_omega_laplace.ferx \
             plans/laplace-sensitivity-speed/warfarin_ode_laplace.ferx \
             plans/laplace-sensitivity-speed/warfarin_focei_agq.ferx \
             plans/laplace-sensitivity-speed/warfarin_block_omega_focei_agq.ferx \
             plans/laplace-sensitivity-speed/warfarin_ode_focei_agq.ferx; do
  echo "=== $(basename "$model") ==="
  for _ in $(seq "$REPS"); do
    run "baseline (fd)"        "$model" env FERX_AGQ_GRID_RESPONSE=fd
    run "candidate (analytic)" "$model" env FERX_AGQ_GRID_RESPONSE=analytic
  done
  echo
done
