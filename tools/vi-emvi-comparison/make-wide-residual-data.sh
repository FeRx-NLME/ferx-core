#!/usr/bin/env bash
# Build a warfarin dataset with a REALISTIC residual error, for the 10% arm of Anchor B.
#
# Why this exists. `data/warfarin.csv` carries a ~1% proportional residual, which is not a
# realistic PK error and turns out to be the one regime where ferx's VI misbehaves: q initialises
# at the prior and has to contract ~1400x in variance to reach the posterior, and sigma stalls on
# the way (VI_VALIDATION.md 4.15). At 10% the contraction is ~14x and VI lands on AGQ to 0.04%.
# So the comparison needs both arms: 1% (hostile, and the one with a NONMEM column) and 10%
# (realistic, and where the user-facing claims should be calibrated).
#
# The data is SIMULATED from the 1% file's own converged estimates with sigma swapped to 0.10 --
# same design, same theta/Omega, one variable changed. ferx's simulator is the generator, which
# is fine here because every downstream claim is cross-tool agreement or agreement with AGQ on
# THIS file; nothing depends on recovering the generating vector.
#
# Deterministic: the [simulation] seed is fixed, so re-running reproduces the file byte for byte.
#
# Usage:  tools/vi-emvi-comparison/make-wide-residual-data.sh
# Writes: $FERX_VI_STATE/warfarin_10pct.csv  (outside the repo, like the rest of the harness state)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="${FERX_VI_STATE:-$HOME/.local/share/ferx-vi-validation}"
OUT="$STATE/warfarin_10pct.csv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$STATE"

BIN="$REPO/target/ci-fast/ferx"
[ -x "$BIN" ] || { echo "build first: cargo build --profile ci-fast -p ferx-cli" >&2; exit 1; }

cat > "$WORK/sim.ferx" <<'FERX'
[parameters]
  theta TVCL(0.132687, 0.001, 10.0)
  theta TVV(7.737464, 0.1, 500.0)
  theta TVKA(0.810901, 0.01, 50.0)
  omega ETA_CL ~ 0.028592
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.336036
  sigma PROP_ERR ~ 0.10 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = laplace
  n_agq = 1
  maxiter = 1
  covariance = false
[simulation]
  n_subjects = 10
  dose_amt   = 100.0
  dose_cmt   = 1
  times      = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0, 120.0]
  seed       = 1
FERX

# --simulate draws the dataset and fits it; the draw is what we want, and sdtab is where the
# realised DV lands. The fit is capped at one iteration because its estimates are discarded.
( cd "$WORK" && "$BIN" sim.ferx --simulate > sim.log 2>&1 ) || { cat "$WORK/sim.log" >&2; exit 1; }

# sdtab carries observation rows only (ID, TIME, DV, ...). Re-add the dose record each subject
# was simulated with, in NONMEM order, to make a file the other tools can read.
awk -F, -v OFS=, '
  NR == 1 { print "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV"; next }
  { id = $1 + 0; t = $2 + 0; dv = $3
    if (id != last) { print id, 0, ".", 1, 100, 1, 0, 1; last = id }
    printf "%d,%g,%.4f,0,.,1,0,0\n", id, t, dv }
' "$WORK/sim-sdtab.csv" > "$OUT"

echo "wrote $OUT"
awk -F, 'NR>1 && $4==0 {n++; s+=$3} END {printf "  %d observations, mean DV %.3f\n", n, s/n}' "$OUT"
echo "  10 subjects, 11 timepoints each, 100 mg single oral dose, 10% proportional residual"
