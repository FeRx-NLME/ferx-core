#!/usr/bin/env bash
# Drive both sides of the VI cross-implementation comparison (VI_VALIDATION.md Anchor B).
#
# Run from the repo root:
#     tools/vi-emvi-comparison/run.sh
#
# Not wired into CI, and should not be: it needs an R stack that no CI image here carries,
# and the fits take minutes. This is an on-demand harness.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Persistent state lives outside the repo: 110 MB of R binaries has no business in a git
# tree, and the run outputs are regenerable.
STATE="${FERX_VI_STATE:-$HOME/.local/share/ferx-vi-validation}"
RLIB="$STATE/Rlib"
# FERX_VI_OUT is honoured HERE, not just exported below: it was hardcoded to $STATE/results and
# then exported over the caller's value, so `FERX_VI_OUT=... run.sh` silently wrote the second
# arm's fits on top of the first one's. Exactly the failure the FERX_DATA note below warns about.
RESULTS="${FERX_VI_OUT:-$STATE/results}"
mkdir -p "$RLIB" "$RESULTS"

# ---- The linker workaround ---------------------------------------------------------------
# R's FLIBS points at /opt/gfortran/lib/... for -lgfortran -lquadmath. If the CRAN gfortran
# was never installed, that path does not exist and *every* rxode2 model fails at the LINK
# step -- the C compile succeeds, which makes it look like a code problem. Emptying FLIBS
# works because macOS links with -undefined dynamic_lookup, so the Fortran symbols reached
# through -lRblas/-lRlapack resolve from R's own bundled libraries at load time.
#
# Scoped to this process via R_MAKEVARS_USER; nothing global is modified.
MAKEVARS="$STATE/Makevars.nofortran"
if [ ! -f "$MAKEVARS" ]; then printf 'FLIBS=\n' > "$MAKEVARS"; fi
export R_MAKEVARS_USER="$MAKEVARS"

# ---- Install into the isolated library, once ---------------------------------------------
# Prebuilt arm64 binaries, so no compilation and no gfortran needed for the install itself.
# The system R library is left untouched -- deleting $STATE undoes everything.
if [ ! -d "$RLIB/nlmixr2est" ]; then
  echo "installing nlmixr2est into $RLIB (isolated; system library untouched)"
  Rscript -e "install.packages(c('nlmixr2est','RcppParallel'), lib='$RLIB', \
    repos='https://cloud.r-project.org', type='binary', \
    dependencies=c('Depends','Imports'))"
fi

export FERX_RLIB="$RLIB"
export FERX_VI_OUT="$RESULTS"

# FERX_DATA points both sides at the same file. It exists because the comparison now has two
# arms: data/warfarin.csv (~1% proportional residual -- the arm with a NONMEM column, and the
# regime where VI's sigma misses by 5.5%) and warfarin_10pct.csv (10%, realistic, where VI lands
# on AGQ to 0.04%). See VI_VALIDATION.md 4.15 and make-wide-residual-data.sh.
#
#   FERX_DATA=$FERX_VI_STATE/warfarin_10pct.csv FERX_VI_OUT=$FERX_VI_STATE/results-10pct \
#     tools/vi-emvi-comparison/run.sh
#
# Point FERX_VI_OUT somewhere separate when switching arms, or the second run overwrites the
# first arm's fits and the figures silently mix two datasets.
DATA="${FERX_DATA:-$REPO/data/warfarin.csv}"
[ -f "$DATA" ] || { echo "data file not found: $DATA" >&2; exit 1; }
export FERX_DATA="$DATA"
echo "data: $DATA"
echo "results: $RESULTS"

# ---- nlmixr2 side ------------------------------------------------------------------------
echo "=== nlmixr2: FOCEI + emvi ==="
Rscript tools/vi-emvi-comparison/emvi-compare.R

# ---- ferx side --------------------------------------------------------------------------
# ci-fast is release-level optimisation without LTO. The shipped `release` profile uses fat
# LTO, whose whole-program link dominates the wall clock for a one-off run like this.
echo
echo "=== ferx: FOCEI + VI (diagonal omega, both omega routes; then mixed omega) ==="
cargo build --profile ci-fast -p ferx-cli
# agq_ref first: it is the arbiter the rest are read against, and it is the cheapest arm.
MODELS="agq_ref warfarin_cmp vi_adam vi_closed_form"
# Part 2 (VI_VALIDATION.md 4.9): the mixed-omega arms. FERX_VI_DIAG_ONLY=1 skips them when
# only the section-4.11 diagonal numbers are wanted -- they are four more multi-minute fits.
if [ "${FERX_VI_DIAG_ONLY:-0}" != "1" ]; then
  MODELS="$MODELS warfarin_block_cmp vi_block_adam vi_block_closed_form vi_block_mean_field"
fi
for m in $MODELS; do
  echo "--- $m ---"
  ( cd "$RESULTS" && "$REPO/target/ci-fast/ferx" \
      "$REPO/tools/vi-emvi-comparison/$m.ferx" --data "$DATA" \
      | tail -n 20 )
done

# ---- Claim (a): the structural zeros, read off the ferx fits -----------------------------
# io/output.rs emits an `ETA_x__ETA_y:` entry for every omega pair with |cov| > 1e-15, so
# ABSENCE of the ETA_KA pairs is a positive result, not missing output -- and the block's own
# ETA_V__ETA_CL entry being present is what proves the emitter ran at all.
if [ "${FERX_VI_DIAG_ONLY:-0}" != "1" ]; then
  echo
  echo "=== claim (a): structural zeros in the ferx mixed-omega fits ==="
  # warfarin_block_cmp is in this loop on purpose: ferx's FOCEI does NOT honour the declared
  # structure (it estimates the full lower triangle, n_parameters = 10 rather than 8), so it is
  # EXPECTED to report NOT HELD while every VI arm reports HELD. Surfacing that every run beats
  # burying it in a doc -- see warfarin_block_cmp.ferx and VI_VALIDATION.md 4.13.
  for m in vi_block_adam vi_block_closed_form vi_block_mean_field warfarin_block_cmp; do
    y="$RESULTS/$m-fit.yaml"
    [ -f "$y" ] || { echo "$m: no fit YAML -- the fit did not finish"; continue; }
    blk=$(grep -c 'ETA_V__ETA_CL:' "$y" || true)
    ka=$(grep -cE 'ETA_KA__ETA_(CL|V):' "$y" || true)
    if [ "$blk" -ge 1 ] && [ "$ka" -eq 0 ]; then
      verdict="HELD (block covariance estimated, ETA_KA pairs absent)"
    elif [ "$blk" -eq 0 ]; then
      verdict="INCONCLUSIVE -- the block covariance is absent too, so nothing was emitted"
    else
      verdict="NOT HELD -- an ETA_KA off-diagonal was emitted"
      [ "$m" = "warfarin_block_cmp" ] && verdict="$verdict (EXPECTED: FOCEI, see the .ferx header)"
    fi
    printf '  %-24s %s\n' "$m" "$verdict"
  done
fi

echo
echo "results in $RESULTS"
echo "  emvi-results.rds        nlmixr2 estimates + per-subject q; block_* entries are Part 2"
echo "  *-fit.yaml              ferx estimates; the vi: block carries eta_posterior"
echo "  vi_block_*-fit.yaml     mixed omega -- structural zeros and the mean_field bound"
echo "  agq_ref-fit.yaml        near-exact AGQ reference -- the arbiter for both tools"
echo "interpretation: VI_VALIDATION.md section 4.11 (diagonal), 4.13 (mixed omega)"
echo
echo "figures:      Rscript tools/vi-emvi-comparison/plots.R   (FERX_VI_FIGS=<dir> to place them)"
echo "off-diagonals: Rscript tools/vi-emvi-comparison/tier2-offdiag.R"
echo "  Tier 2b -- the per-subject COVARIANCE, both tools against the Anchor C NUTS"
echo "  reference. Needs tools/vi-nuts-anchor/run.sh to have produced anchor-c.json."
