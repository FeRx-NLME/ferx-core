#!/usr/bin/env bash
# Anchor C driver (VI_VALIDATION.md §5). Run from the repo root.
#
# The isolated venv mirrors the R-library arrangement in tools/vi-emvi-comparison: everything
# persistent lives outside the repo under $FERX_VI_STATE, the system Python is untouched, and
# deleting that directory undoes the install. Not wired into CI -- no CI image here carries jax.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
STATE="${FERX_VI_STATE:-$HOME/.local/share/ferx-vi-validation}"
PYENV="$STATE/pyenv"
RESULTS="$STATE/results"
mkdir -p "$RESULTS"

if [ ! -x "$PYENV/bin/python" ]; then
  echo "creating an isolated venv at $PYENV"
  # Python 3.11: jax publishes arm64 wheels for it, and the system python3 here is 3.14 with
  # no numpy at all.
  "${FERX_PY:-/opt/anaconda3/bin/python3.11}" -m venv "$PYENV"
  "$PYENV/bin/python" -m pip install --quiet --upgrade pip
  "$PYENV/bin/python" -m pip install --quiet jax numpyro
fi

# The ferx side: q at the AGQ estimate, every population parameter FIXed so both sides sit at
# the same theta/Omega/sigma and the comparison is about q alone. Needs many draws -- the point
# is to measure the approximation, not the optimizer.
echo "=== ferx: variational q at the AGQ estimate ==="
cargo build --profile ci-fast --bin ferx
( cd "$RESULTS" && "$REPO/target/ci-fast/ferx" \
    "$REPO/tools/vi-nuts-anchor/q_at_agq.ferx" --data "$REPO/data/warfarin.csv" | tail -n 12 )

echo
echo "=== NUTS reference ==="
FERX_REPO="$REPO" FERX_VI_OUT="$RESULTS" "$PYENV/bin/python" tools/vi-nuts-anchor/anchor_c.py
