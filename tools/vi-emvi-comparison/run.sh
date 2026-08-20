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
RESULTS="$STATE/results"
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

# ---- nlmixr2 side ------------------------------------------------------------------------
echo "=== nlmixr2: FOCEI + emvi ==="
Rscript tools/vi-emvi-comparison/emvi-compare.R

# ---- ferx side --------------------------------------------------------------------------
# ci-fast is release-level optimisation without LTO. The shipped `release` profile uses fat
# LTO, whose whole-program link dominates the wall clock for a one-off run like this.
echo
echo "=== ferx: FOCEI + VI (both omega routes) ==="
cargo build --profile ci-fast --bin ferx
for m in warfarin_cmp vi_adam vi_closed_form; do
  echo "--- $m ---"
  ( cd "$RESULTS" && "$REPO/target/ci-fast/ferx" \
      "$REPO/tools/vi-emvi-comparison/$m.ferx" --data "$REPO/data/warfarin.csv" \
      | tail -n 20 )
done

echo
echo "results in $RESULTS"
echo "  emvi-results.rds        nlmixr2 estimates + per-subject q"
echo "  *-fit.yaml              ferx estimates; the vi: block carries eta_posterior"
echo "interpretation: VI_VALIDATION.md section 4.11"
