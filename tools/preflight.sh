#!/usr/bin/env bash
#
# Run the fast CI gates locally, so "green on my machine" means what CI means.
#
#   tools/preflight.sh              # every fast gate, cheapest first
#   tools/preflight.sh check        # just the cargo check matrix (tight edit loop)
#   tools/preflight.sh fmt clippy   # any subset, in the order you name them
#   tools/preflight.sh --list       # print the commands without running them
#
# This is #1157. `.github/workflows/ci.yml` invokes THIS SCRIPT for its `Check`,
# `Clippy` and `Format` jobs, which is the entire point: a script that merely
# *documents* the commands drifts the first time somebody edits the workflow, so
# CI has to actually execute it. Same contract as `tools/update-public-api.sh`,
# whose header makes the same argument for the public-API baseline.
#
# It exists because a green test suite is NOT a green build in this repo. The
# feature sets below are not nested: `cargo test --features ci,survival` compiles
# neither the `nn`-gated nor the `slow-tests`-gated source, so a file behind those
# cfgs can be broken while the whole suite passes. On #1133 that let one struct
# literal in `src/estimation/nn_theta_gradient_tests.rs` turn three CI jobs red
# after a local run of 158 binaries / 4634 tests reported clean:
#
#     error[E0063]: missing fields `reset_covariates` and `reset_occasions`
#                   in initializer of `types::Subject`
#
# The whole matrix is compile-only or near it, so it costs a couple of minutes
# warm — far less than a push/wait/diagnose cycle.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

LIST_ONLY=0
SELECTED=()

for arg in "$@"; do
  case "$arg" in
    --list) LIST_ONLY=1 ;;
    -h|--help)
      # POSIX class, not `\s`: BSD sed (macOS, where most of us edit) does not
      # understand `\s` and would silently leave every `#` in place.
      sed -n '3,8p' "${BASH_SOURCE[0]}" | sed 's/^#[[:space:]]\{0,1\}//'
      echo
      echo "Groups: fmt, check, clippy, public-api (default: all four, in that order)"
      exit 0
      ;;
    fmt|check|clippy|public-api) SELECTED+=("$arg") ;;
    -*)
      echo "preflight: unknown flag '$arg' (try --help)" >&2
      exit 2
      ;;
    *)
      echo "preflight: unknown group '$arg' — expected one of: fmt check clippy public-api" >&2
      exit 2
      ;;
  esac
done

# Cheapest first, so a formatting slip fails in seconds rather than after a
# full compile. CI calls one group per job, so this order only affects local runs.
if [ ${#SELECTED[@]} -eq 0 ]; then
  SELECTED=(fmt check clippy public-api)
fi

# ── Plumbing ────────────────────────────────────────────────────────────────
# `CI_JOB` names the workflow job a failure would have shown up in, so the error
# teaches the local-command → CI-job mapping instead of just failing.
CI_JOB=""
FAILED_CMD=""

run() {
  echo
  echo "  \$ $*"
  if [ "$LIST_ONLY" -eq 1 ]; then
    return 0
  fi
  if ! "$@"; then
    FAILED_CMD="$*"
    return 1
  fi
}

on_failure() {
  echo >&2
  echo "────────────────────────────────────────────────────────────────────────────" >&2
  echo "preflight FAILED in group '$CURRENT_GROUP'." >&2
  echo "  command:  $FAILED_CMD" >&2
  echo "  CI job:   $CI_JOB — this is what would have gone red on the PR." >&2
  echo "────────────────────────────────────────────────────────────────────────────" >&2
}

# ── Groups ──────────────────────────────────────────────────────────────────

group_fmt() {
  CI_JOB="Format"
  # `--all`: without it `cargo fmt` touches only the root package, so
  # `crates/ferx-tools` and `crates/ferx-cli` go unchecked (#1114).
  # `.githooks/pre-commit` uses the same flag.
  run cargo fmt --all -- --check
}

group_check() {
  CI_JOB="Check"

  run cargo check --tests --no-default-features --features ci

  # `tests/tte_convergence.rs` is gated on BOTH `survival` and `slow-tests`, and no
  # other per-PR job sets both — so those Tier-3 TTE tests compile in *neither*.
  # Without this a renamed public API or changed `FitResult` field lands green and
  # only breaks the nightly slow-tests run, decoupled from the offending PR.
  # Compile-only: the tests themselves stay nightly per the tier rules.
  # (#441 review finding #3.)
  run cargo check --tests --no-default-features --features ci,survival,slow-tests

  # This is also what COMPILE-GATES the whole feature-gated surface. The
  # `Tests + coverage (TTE/CTMM endpoints)` job builds only the handful of endpoint
  # test binaries, so a break in any *other* file under `--features ci,markov`
  # (which implies `survival`) is caught here in ~1 min, rather than by an 18-min
  # instrumented build.
  run cargo check --tests --no-default-features --features ci,markov

  # Same rationale for `nn`, with `slow-tests` folded in for the same reason as the
  # survival line: `tests/nn_fit_smoke.rs` and `tests/nn_fit_convergence.rs` are
  # gated on BOTH `nn` and `slow-tests`, so their bodies compile in no other per-PR
  # job. One cheap check covers the whole `nn` surface — `src/nn`, the
  # `[covariate_nn]` parser, the θ-gradient module, and every nn-gated test file.
  # This is the exact line that would have caught #1133's E0063 before the push.
  run cargo check --tests --no-default-features --features ci,nn,slow-tests

  # The workspace members (#1114). `-p` rather than `--workspace` so the `ferx-core`
  # targets above are not rebuilt, and the feature is package-qualified
  # (`ferx-core/ci`) because `ci` belongs to `ferx-core`, not to the members — a
  # bare `--features ci` here fails outright with "none of the selected packages
  # contains this feature". Qualified this way it resolves to the SAME feature set
  # as the lines above, so the members link the rlib already built instead of
  # forcing a second `ferx-core` compile under a different feature hash.
  run cargo check -p ferx-tools -p ferx-cli --tests --no-default-features --features ferx-core/ci
}

group_clippy() {
  CI_JOB="Clippy"

  # `markov` and `nn` so the CTMM module and the covariate-NN / DCM stack are linted
  # too (both compiled out of the base `ci` build). `markov = ["survival"]`, so this
  # covers the whole non-Gaussian endpoint surface as well.
  #
  # `--all-targets` is load-bearing. Without it the default target selection covers
  # only lib and bins, and roughly a third of this repo — every `#[cfg(test)]`
  # module, every sibling `*_tests.rs`, and all of `tests/` — gets no lint coverage.
  # That gap hid 6 `approx_constant` ERRORS (not warnings) that the default
  # invocation exits 0 on (#1023).
  #
  # NOTE: CI installs a FRESH nightly every run, so its lint set is newer than a
  # local toolchain unless you have updated recently — one of those 6
  # (`EULER_GAMMA`, `src/stats/special.rs`) is invisible to an old clippy. Run
  # `rustup update nightly` before trusting a green result here.
  run cargo clippy --no-default-features --features ci,markov,nn --all-targets

  # Same package-qualified-feature trick as `check`, so the members reuse the
  # `ferx-core` build the line above just produced. `--tests` here (unlike the
  # ferx-core line) because the members' test code is a meaningful share of their
  # line count while they are still small.
  run cargo clippy -p ferx-tools -p ferx-cli --tests --no-default-features \
    --features ferx-core/ci,ferx-core/markov,ferx-core/nn
}

group_public_api() {
  CI_JOB="Public API baseline"
  # Owns its own pinned nightly and pinned `cargo-public-api`, and self-installs the
  # latter. Regenerate with `tools/update-public-api.sh` (no `--check`) when a
  # widening is intended, and say in the PR which tool needs each added item.
  run tools/update-public-api.sh --check
}

# ── Drive ───────────────────────────────────────────────────────────────────

if [ "$LIST_ONLY" -eq 1 ]; then
  echo "preflight would run (in order):"
fi

for g in "${SELECTED[@]}"; do
  CURRENT_GROUP="$g"
  echo
  echo "══ $g ═══════════════════════════════════════════════════════════════════"
  case "$g" in
    fmt)        group_fmt ;;
    check)      group_check ;;
    clippy)     group_clippy ;;
    public-api) group_public_api ;;
  esac || { on_failure; exit 1; }
done

echo
if [ "$LIST_ONLY" -eq 1 ]; then
  echo "(--list: nothing was executed)"
else
  echo "preflight OK — ${SELECTED[*]}"
  if [ ${#SELECTED[@]} -lt 4 ]; then
    echo "note: this was a SUBSET. A full run is 'tools/preflight.sh' with no arguments."
  fi
fi
