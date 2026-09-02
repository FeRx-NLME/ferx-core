#!/usr/bin/env bash
#
# Regenerate (or verify) the committed `ferx-core` public-API baseline.
#
#   tools/update-public-api.sh          # rewrite api/ferx-core-public-api.txt
#   tools/update-public-api.sh --check  # diff against it; non-zero on drift
#
# This is A2 of #1114: `ferx-tools` can only touch `pub` items, so the way the
# boundary rots is `ferx-core` growing an ad-hoc `pub` in the same PR that needs
# it. Committing the surface and diffing it in CI turns every widening into an
# explicit reviewable line in the diff instead of something buried in a 400-line
# change. It doubles as semver protection for the crates.io release and for
# ferx-r, which consumes `ferx-core` as an ordinary external crate.
#
# CI runs this exact script (`--check`) so the generating command cannot drift
# from the verifying one.
set -euo pipefail

# ── Pinned toolchain ────────────────────────────────────────────────────────
# The output is rustdoc-JSON-derived, so it moves whenever rustdoc's JSON format
# or rendering does — i.e. potentially every night. `rust-toolchain.toml` says a
# bare `nightly`, which would make this baseline drift on its own and produce
# red diffs unrelated to any PR. `RUSTUP_TOOLCHAIN` takes precedence over
# `rust-toolchain.toml` (the same override the coverage jobs use to pin
# `stable`), so pin a *dated* nightly here. Bumping it is a deliberate act: bump
# the date, rerun without `--check`, and commit the regenerated baseline in the
# same PR.
PINNED_TOOLCHAIN="${FERX_API_TOOLCHAIN:-nightly-2026-05-29}"

# Pinned for the same reason — `cargo-public-api`'s own rendering changes across
# releases.
PUBLIC_API_VERSION="0.51.0"

# ── Feature set ─────────────────────────────────────────────────────────────
# Deliberately NOT the default (empty) feature set. `survival`, `markov` and
# `nn` each add public items, so a default-features baseline would let a `pub`
# added behind one of those cfgs through the gate unseen. `markov = ["survival"]`,
# so `ci,markov,nn` is the full surface in one build — the same feature set the
# Clippy and endpoint-coverage jobs use.
FEATURES="ci,markov,nn"

# `-sss` = --omit blanket-impls,auto-trait-impls,auto-derived-impls. Without it
# ~3.6k of the ~9.2k lines are `impl core::marker::Freeze for …` and friends,
# and adding one `pub struct` moves a dozen lines. A2 exists so a new public
# item is ONE reviewable line; the auto impls defeat that and carry no
# information a reviewer of this repo acts on.
SIMPLIFY="-sss"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
baseline="$repo_root/api/ferx-core-public-api.txt"

# Validate the argument BEFORE doing any work. The check/write choice below is a
# bare `if --check ... else write`, so without this any unrecognised argument —
# `--chek`, `-check`, `check` — falls into the write arm and OVERWRITES the
# committed baseline with whatever the current tree exposes. That is the one
# outcome this gate exists to prevent, reached by a typo, and it looks like a
# pass: the script prints "wrote ..." and exits 0.
usage() {
  echo "usage: tools/update-public-api.sh [--check]"
  echo "  (no argument) regenerate $baseline"
  echo "  --check       diff against it; non-zero on drift (what CI runs)"
}

case "${1:-}" in
  "" | --check) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    echo "refusing to run with unknown argument '$1' — writing the baseline by" >&2
    echo "accident would silently accept whatever the tree currently exposes." >&2
    exit 2
    ;;
esac

if ! cargo public-api --version 2>/dev/null | grep -q "$PUBLIC_API_VERSION"; then
  echo "installing cargo-public-api $PUBLIC_API_VERSION ..." >&2
  cargo install cargo-public-api --locked --version "$PUBLIC_API_VERSION"
fi

generated="$(mktemp)"
trap 'rm -f "$generated"' EXIT

RUSTUP_TOOLCHAIN="$PINNED_TOOLCHAIN" cargo public-api \
  --manifest-path "$repo_root/Cargo.toml" \
  --package ferx-core \
  $SIMPLIFY \
  --no-default-features \
  --features "$FEATURES" \
  >"$generated"

if [ "${1:-}" = "--check" ]; then
  if diff -u "$baseline" "$generated"; then
    echo "public API matches the committed baseline."
  else
    cat >&2 <<'MSG'

────────────────────────────────────────────────────────────────────────────
The public API of `ferx-core` changed (diff above; `-` = removed, `+` = added).

If the change is intended, regenerate the baseline IN THIS PR:

    tools/update-public-api.sh

and say in the PR description which tool needs each added item and why the
existing surface does not suffice (A6). Widening is a design step, not
paperwork — and no `#[doc(hidden)] pub` escape hatch (A3): an item is either
public API (documented, in this baseline, usable from ferx-r) or it stays
`pub(crate)` and the caller does without.
────────────────────────────────────────────────────────────────────────────
MSG
    exit 1
  fi
else
  cp "$generated" "$baseline"
  echo "wrote $baseline"
fi
