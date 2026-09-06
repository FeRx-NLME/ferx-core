#!/usr/bin/env bash
#
# Run the fast CI gates locally, so "green on my machine" means what CI means.
# `--help` prints usage; this header is the why.
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
# Cost: the matrix is compile-only or near it, so warm it is fast — measured on a
# warm `target/` here, the full default run was 34s (fmt 13 · check 8 · clippy 2 ·
# public-api 15), and that was with two other cargo jobs competing for the lock.
# Cold it is a dependency build like any other; run `check` alone while iterating.
#
# If you ever time this and get minutes for the same work, check for a concurrent
# cargo FIRST. Cargo serialises on a per-`target/` lock, and a build running in a
# sibling worktree blocks this one behind a single line of output — that is not
# preflight being slow, and it inflated an early measurement of this script
# several-fold. `CARGO_TARGET_DIR` isolates it if you need a clean number.
set -euo pipefail

# Resolved BEFORE the `cd`, so `--help` and the group table work from any cwd.
script_path="${BASH_SOURCE[0]}"
repo_root="$(cd "$(dirname "$script_path")/.." && pwd)"
cd "$repo_root"

# Mirror `ci.yml`'s workflow-level `RUSTUP_TOOLCHAIN: nightly`.
#
# `rust-toolchain.toml` says `nightly`, but it is NOT the last word: an inherited
# `RUSTUP_TOOLCHAIN`, or a `rustup override` on this directory or any parent,
# both beat it. Measured in this repo — `RUSTUP_TOOLCHAIN=stable cargo fmt
# --version` reports `rustfmt 1.9.0-stable` where a plain `cargo fmt --version`
# reports `1.10.0-nightly`. Running the fast gates on a channel CI never uses is
# the exact local/CI split this script exists to remove, and clippy is where it
# bites: whole lint groups are nightly-only, so a stable run is quietly laxer.
#
# `:-` so a caller who sets it deliberately (bisecting a toolchain regression,
# say) still wins.
export RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-nightly}"

# ── The group list ──────────────────────────────────────────────────────────
# The ONE place groups are enumerated. Argument validation, the default
# selection, `--help` and the driver all read this array, so adding a group is
# one entry here plus its `group_<name>` function — and a mismatch between the
# two is a startup error, not a group that silently never runs.
#
# NOT named `GROUPS`: bash maintains a special array of that name holding the
# current user's numeric group IDs, and it wins. The first draft used it and
# every group name silently became a GID — `unknown group 'check' — expected one
# of: 20 12 61 ...`.
ALL_GROUPS=(fmt check clippy rustdoc docs public-api debug-assertions)

# Groups that are NOT in the default selection.
#
# Every other group is a compile or a lint — seconds on a warm `target/`, which
# is what makes a no-argument run something you do before every push. A group
# that RUNS the unit suite is a different order of cost (its own profile hash,
# so its own full build, and then the tests), and folding it into the default
# would turn `tools/preflight.sh` from a pre-push habit into a coffee break. So
# it is opt-in by name — `tools/preflight.sh debug-assertions` — and CI runs it
# as its own job, which is where it has to be green.
#
# `--help` and `--list` still enumerate it, and
# `tests/preflight_owns_the_fast_gates.rs` pins both this array and the fact that
# the corresponding CI job delegates here, so an opt-in group cannot quietly
# become a group that nothing ever runs.
OPT_IN_GROUPS=(debug-assertions)

is_opt_in() {
  local candidate="$1" g
  for g in "${OPT_IN_GROUPS[@]}"; do
    if [ "$g" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

# `ALL_GROUPS` minus `OPT_IN_GROUPS`, in `ALL_GROUPS` order. Derived rather than
# written out a second time: a group added to one list and forgotten in the other
# is the failure mode this whole script exists to avoid.
DEFAULT_GROUPS=()
for g in "${ALL_GROUPS[@]}"; do
  if ! is_opt_in "$g"; then
    DEFAULT_GROUPS+=("$g")
  fi
done


usage() {
  cat <<EOF
Run the fast CI gates locally, so "green on my machine" means what CI means.

  tools/preflight.sh              every fast gate, cheapest first
  tools/preflight.sh check        just the cargo check matrix (tight edit loop)
  tools/preflight.sh fmt clippy   any subset, in the order you name them
  tools/preflight.sh --list       print the commands without running them

Groups: ${ALL_GROUPS[*]}
Default: ${DEFAULT_GROUPS[*]}  (in that order)

Opt-in, i.e. NOT in a no-argument run: ${OPT_IN_GROUPS[*]}. \`debug-assertions\`
RUNS the Tier-1 suite on the default \`dev\` profile (#344). Every other test job
builds with \`ci-test\`/\`ci-fast\`, which \`inherits = "release"\` — so every
\`debug_assert!\` in the crate is compiled to nothing and its invariant goes
unchecked. It is a build plus a test run rather than seconds of compile, so you
name it when you want it; CI runs it on every PR.

NOT covered by the DEFAULT set: the test jobs. \`cargo check --tests\` COMPILES
the test targets but runs nothing, so \`Tests + coverage (core)\` can still go red
after a green preflight. The default set gates compilation and lint, not
behaviour. Two exceptions: \`docs\`, whose gate IS its test run (a filesystem walk
over \`docs/\`, not a fit), and the opt-in \`debug-assertions\` above.
EOF
}

is_group() {
  local candidate="$1" g
  for g in "${ALL_GROUPS[@]}"; do
    if [ "$g" = "$candidate" ]; then
      return 0
    fi
  done
  return 1
}

LIST_ONLY=0
SELECTED=()

for arg in "$@"; do
  case "$arg" in
    --list)
      LIST_ONLY=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*)
      echo "preflight: unknown flag '$arg' (try --help)" >&2
      exit 2
      ;;
    *)
      if is_group "$arg"; then
        SELECTED+=("$arg")
      else
        echo "preflight: unknown group '$arg' — expected one of: ${ALL_GROUPS[*]}" >&2
        exit 2
      fi
      ;;
  esac
done

# Cheapest first, so a formatting slip fails in seconds rather than after a
# full compile. CI calls one group per job, so this order only affects local runs.
if [ ${#SELECTED[@]} -eq 0 ]; then
  SELECTED=("${DEFAULT_GROUPS[@]}")
fi

# ── Plumbing ────────────────────────────────────────────────────────────────
# `CI_JOB` names the workflow job a failure would have shown up in, so the error
# teaches the local-command → CI-job mapping instead of just failing.
CI_JOB=""
CURRENT_GROUP=""

die() {
  echo >&2
  echo "────────────────────────────────────────────────────────────────────────────" >&2
  echo "preflight FAILED in group '$CURRENT_GROUP'." >&2
  echo "  command:  $*" >&2
  # The driver blanks CI_JOB before each group, so an empty one here means the
  # group function never set it — say so rather than name a job that is not the
  # one that would go red. Before the driver blanked it, a group that forgot the
  # assignment silently inherited the PREVIOUS group's job name, which is a
  # diagnostic that confidently points at the wrong place.
  if [ -n "$CI_JOB" ]; then
    echo "  CI job:   $CI_JOB — this is what would have gone red on the PR." >&2
  else
    echo "  CI job:   (group '$CURRENT_GROUP' sets no CI_JOB — fix that in the script)" >&2
  fi
  echo "────────────────────────────────────────────────────────────────────────────" >&2
  exit 1
}

# `run` NEVER returns to its caller on failure — it exits the script.
#
# That is load-bearing, not style. The first version of this file returned a
# status and left the caller to propagate it, through a group function, through
# a `case ... esac || { ...; exit 1; }` in the driver. Bash suspends `errexit`
# for every command of an AND-OR list but the last, and propagates that
# suspension into the whole body of a function called in that context — so the
# `return 1` stopped nothing, each group reported its LAST command's status, and
# four of the five `cargo check` lines (including `ci,nn,slow-tests`, the one
# that would have caught #1133) could fail with the script printing
# "preflight OK" and exiting 0. Exiting from here deletes the propagation path
# instead of fixing one link in it: there is no caller left that can drop the
# status, whatever `errexit` is doing at the call site.
#
# `tests/preflight_owns_the_fast_gates.rs` pins this by failing each command
# position in turn behind a fake `cargo` and asserting a non-zero exit.
run() {
  echo
  echo "  \$ $*"
  if [ "$LIST_ONLY" -eq 1 ]; then
    return 0
  fi
  "$@" || die "$*"
}

# ── Groups ──────────────────────────────────────────────────────────────────

group_fmt() {
  CI_JOB="Format"
  # `--all`: without it `cargo fmt` touches only the root package, so
  # `crates/ferx-tools` and `crates/ferx-cli` go unchecked (#1114).
  #
  # No `+nightly`: `rust-toolchain.toml` pins the channel locally and the
  # workflow exports `RUSTUP_TOOLCHAIN: nightly`. `.githooks/pre-commit` calls
  # THIS group rather than repeating the command, so the hook, the local gate
  # and CI are one owner.
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

group_docs() {
  CI_JOB="Docs lint"
  # The structural gate on `docs/**/*.qmd` (#1163): section size, heading levels,
  # duplicate anchors, dead internal links. `docs-lint` depends on nothing — not
  # even `ferx-core` — so this group is a seconds-long compile and never touches
  # the cache the other groups warm.
  #
  # This is the one group that also RUNS its tests rather than only compiling
  # them: the corpus check IS the gate, and it is a filesystem walk, not a fit.
  run cargo test -p docs-lint

  # `docs-lint` is outside the `Clippy` group's package list (which is scoped to
  # `ferx-core` and its two members), so without this line the crate would be
  # linted by nothing. Cheap: the compile above already warmed it.
  run cargo clippy -p docs-lint --all-targets
}

group_rustdoc() {
  CI_JOB="Rustdoc"
  # `env RUSTDOCFLAGS=...` rather than a `RUSTDOCFLAGS=... run cargo ...` prefix.
  # Both reach rustdoc — bash exports an assignment prefix into the function and
  # on to the child — but `run` echoes `$*`, and the prefix form is not part of
  # `$*`. `--list` would then print a `cargo doc` line that does NOT fail on
  # warnings, which is the one thing `--list` exists to tell you truthfully. The
  # `env` form is in the argument vector, so what is printed is what runs.
  #
  # `-Dwarnings` unspaced for the same reason: it survives copy-paste without
  # quoting.
  #
  # One feature set, unlike `check`'s five, and that is not an oversight:
  # `markov` implies `survival` and `slow-tests` gates only `#[cfg_attr(...)]`
  # test attributes, so `ci,markov,nn` IS the union of the production cfgs. The
  # non-nesting trap that makes `check` a matrix does not exist here. (Clippy
  # takes this same set for the same reason.)
  #
  # `cargo doc` documents the library only — `#[cfg(test)]` modules and sibling
  # `*_tests.rs` files are not rustdoc'd, so a doc link in test code is not
  # gated. That is fine: it renders nowhere.
  run env RUSTDOCFLAGS=-Dwarnings cargo doc --no-deps --no-default-features \
    --features ci,markov,nn

  # The members, for the same reason clippy checks them separately (#1114):
  # a workspace-root `cargo doc` documents only the root package, so
  # `ferx-tools` and `ferx-cli` would be gated by nothing.
  run env RUSTDOCFLAGS=-Dwarnings cargo doc -p ferx-tools -p ferx-cli --no-deps \
    --no-default-features --features ferx-core/ci,ferx-core/markov,ferx-core/nn
}

group_public_api() {
  CI_JOB="Public API baseline"
  # Owns its own pinned nightly and pinned `cargo-public-api`, and self-installs
  # both — so the FIRST run of this group downloads a toolchain and builds
  # rustdoc JSON from cold. For a tight edit loop name the groups you want
  # (`tools/preflight.sh check clippy`); this one is in the default set because
  # forgetting it is how a widened surface reaches the PR.
  #
  # Regenerate with `tools/update-public-api.sh` (no `--check`) when a widening
  # is intended, and say in the PR which tool needs each added item.
  run tools/update-public-api.sh --check
}

group_debug_assertions() {
  CI_JOB="Tests (debug-assertions)"
  # #344. Every OTHER test job in this repo builds with `--profile ci-test` or
  # `--profile ci-fast`, both of which `inherits = "release"` — so
  # `debug-assertions` is OFF and all 177 `debug_assert!` guards in the crate are
  # compiled to nothing. They are invariant checks that CI was not running.
  #
  # That is not theoretical: a real off-by-one in `compute_max_stack`
  # (`src/parser/model_parser.rs`) tripped its own `debug_assert!` on any
  # expression containing an `if`, and four lib tests panicked under the dev
  # profile while CI stayed green through the whole thing. It was fixed in #342;
  # the CI blind spot that hid it is what this group closes.
  #
  # The DEFAULT (dev) profile, deliberately — no `--profile` flag:
  #
  #  * it is the profile the bug actually reproduced under, so this gate is the
  #    literal repro rather than an approximation of it;
  #  * it also turns on `overflow-checks`, which `release` leaves off, so integer
  #    overflow panics instead of wrapping silently — a second class of guard the
  #    other jobs cannot see;
  #  * a `ci-test`-with-debug-assertions profile would be a distinct profile hash
  #    and therefore a whole extra OPTIMISED build of the lib, which in this
  #    compile-bound repo (93% of the build is LLVM, #969/#971) costs far more
  #    than the unoptimised tests do. Measured here on the dev profile: 80s of
  #    tests for `ci` (4035) and 198s for `ci,markov,nn` (4366), all green.
  #
  # `--lib` only: Tier-1 unit tests. The `tests/` binaries are run by
  # `Tests + coverage (core)`, and compiling 120-odd of them again under a second
  # profile is the cost this group is shaped to avoid. Tier-1 is also where the
  # guards live — `debug_assert!` sits next to the invariant, in `src/`.
  # `env FERX_REQUIRE_DEBUG_ASSERTIONS=1` in the ARGUMENT VECTOR, not as a shell
  # assignment prefix and not an `export` up in this function. Both of those would
  # reach the child just the same, but `run` echoes `$*`, so only this form appears
  # in `--list` — and what `--list` prints is what the guard test can assert and
  # what a developer can copy. Same reasoning as `RUSTDOCFLAGS` in `group_rustdoc`.
  #
  # It arms `debug_assertion_canary` (`src/lib_tests.rs`), which fails the run if
  # `debug_assert!` turns out to compile to nothing. Without it this whole group is
  # an invariant held by ABSENCE — nothing in `cargo test --lib --features ci`
  # distinguishes a build with the guards live from one without, so a
  # `[profile.dev] debug-assertions = false` or a `CARGO_PROFILE_DEV_DEBUG_ASSERTIONS`
  # in the environment would neuter it with every test still green. The canary is
  # opt-in precisely because every OTHER job legitimately runs with the guards off.
  run env FERX_REQUIRE_DEBUG_ASSERTIONS=1 cargo test --lib --no-default-features --features ci

  # The feature-gated surface, for the same non-nesting reason `check` is a matrix
  # rather than one line: `--features ci` compiles neither `src/survival`,
  # `src/markov`, `src/categorical` nor `src/nn`, and all four carry
  # `debug_assert!`s. Without this line their guards would be as dead in this job
  # as they are in every other one. `markov` implies `survival`, so
  # `ci,markov,nn` is the union of the production cfgs — the same set `clippy` and
  # `rustdoc` take, and for the same reason.
  run env FERX_REQUIRE_DEBUG_ASSERTIONS=1 cargo test --lib --no-default-features \
    --features ci,markov,nn
}

# ── Drive ───────────────────────────────────────────────────────────────────

# A group named in `ALL_GROUPS` with no `group_<name>` function would print its
# banner and silently run nothing. Fail at startup instead, before any compile.
for g in "${ALL_GROUPS[@]}"; do
  fn="group_${g//-/_}"
  if ! declare -F "$fn" >/dev/null; then
    echo "preflight: internal error: group '$g' has no $fn function" >&2
    exit 3
  fi
done

if [ "$LIST_ONLY" -eq 1 ]; then
  echo "preflight would run (in order):"
fi

for g in "${SELECTED[@]}"; do
  CURRENT_GROUP="$g"
  # Blank per group: see `die`. Leaving the previous group's value in place makes
  # a forgotten assignment invisible instead of loud.
  CI_JOB=""
  echo
  echo "══ $g ═══════════════════════════════════════════════════════════════════"
  # A plain command, deliberately: no `||` here, so nothing suspends `errexit`
  # for the function body. `run` exits on failure anyway; this keeps the driver
  # from re-introducing the swallow if that ever changes.
  "group_${g//-/_}"
done

echo
if [ "$LIST_ONLY" -eq 1 ]; then
  echo "(--list: nothing was executed)"
else
  echo "preflight OK — ${SELECTED[*]}"
  if [ ${#SELECTED[@]} -lt ${#DEFAULT_GROUPS[@]} ]; then
    echo "note: this was a SUBSET. A full run is 'tools/preflight.sh' with no arguments."
  fi
fi
