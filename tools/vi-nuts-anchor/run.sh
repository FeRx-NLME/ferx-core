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
# Honour the caller's FERX_VI_OUT / FERX_DATA / FERX_Q_MODEL: Anchor C now has two arms, the
# ~1% residual of data/warfarin.csv and the realistic 10% one (VI_VALIDATION.md 4.15), and each
# needs its own results directory or the second run overwrites the first arm's reference. The
# q model must be FIXed at the AGQ estimate OF THE ARM being run -- q_at_agq.ferx carries the 1%
# estimate, q_at_agq_10pct.ferx the 10% one.
RESULTS="${FERX_VI_OUT:-$STATE/results}"
DATA="${FERX_DATA:-$REPO/data/warfarin.csv}"
Q_MODEL="${FERX_Q_MODEL:-$REPO/tools/vi-nuts-anchor/q_at_agq.ferx}"
[ -f "$DATA" ] || { echo "data file not found: $DATA" >&2; exit 1; }
[ -f "$Q_MODEL" ] || { echo "q model not found: $Q_MODEL" >&2; exit 1; }
mkdir -p "$RESULTS"
echo "data:    $DATA"
echo "q model: $Q_MODEL"
echo "results: $RESULTS"

if [ ! -x "$PYENV/bin/python" ]; then
  echo "creating an isolated venv at $PYENV"
  # Python 3.11: jax publishes arm64 wheels for it, and the system python3 here is 3.14 with
  # no numpy at all.
  # jax publishes arm64 wheels for 3.11/3.12; set FERX_PY if your python3 is outside that
  # range (this was developed against a 3.14 system python, which has no jax wheels at all).
  "${FERX_PY:-python3.11}" -m venv "$PYENV"
  "$PYENV/bin/python" -m pip install --quiet --upgrade pip
  "$PYENV/bin/python" -m pip install --quiet jax numpyro
fi

# The ferx side: q at the AGQ estimate, every population parameter FIXed so both sides sit at
# the same theta/Omega/sigma and the comparison is about q alone. Needs many draws -- the point
# is to measure the approximation, not the optimizer.
echo "=== ferx: variational q at the AGQ estimate ==="
cargo build --profile ci-fast -p ferx-cli
( cd "$RESULTS" && "$REPO/target/ci-fast/ferx" "$Q_MODEL" --data "$DATA" | tail -n 12 )

echo
echo "=== NUTS reference ==="
FERX_REPO="$REPO" FERX_VI_OUT="$RESULTS" FERX_DATA="$DATA" \
  FERX_Q_FILE="${FERX_Q_FILE:-$RESULTS/$(basename "$Q_MODEL" .ferx)-fit.yaml}" \
  "$PYENV/bin/python" tools/vi-nuts-anchor/anchor_c.py

echo
echo "figures: Rscript tools/vi-nuts-anchor/plots.R   (FERX_VI_FIGS=<dir> to place them)"
