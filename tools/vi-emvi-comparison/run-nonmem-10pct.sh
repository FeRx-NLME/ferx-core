#!/usr/bin/env bash
# Run the NONMEM arm of the 10%-residual comparison, locally or in a container.
#
# NONMEM is licensed and is not installed on every machine that runs the rest of this harness,
# so this script does one job: put the control stream and its data in a scratch directory, run
# whatever nmfe is available, and copy the outputs back next to the control stream where the
# comparison expects them (.lst / .ext / sdtab).
#
# Usage
#   tools/vi-emvi-comparison/run-nonmem-10pct.sh                    # local nmfe on PATH
#   NMFE=nmfe751 tools/vi-emvi-comparison/run-nonmem-10pct.sh       # pick the wrapper
#   DOCKER_IMAGE=my/nonmem:7.5.1 tools/vi-emvi-comparison/run-nonmem-10pct.sh
#
# The run takes seconds - 10 subjects, 110 observations, one FOCEI step.
#
# What comes back matters more than how it is run: commit warfarin_10pct.lst and .ext next to
# the .ctl, exactly as warfarin_imp.lst/.ext are committed for the 1% arm, and the FOCEI column
# of VI_VALIDATION.md 4.15 can be filled from `TABLE NO. 1` of the .ext.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NM_DIR="$REPO/tests/nonmem"
CTL="warfarin_10pct.ctl"
LST="warfarin_10pct.lst"
NMFE="${NMFE:-nmfe75}"

[ -f "$NM_DIR/$CTL" ] || { echo "missing $NM_DIR/$CTL" >&2; exit 1; }
[ -f "$NM_DIR/warfarin_10pct.csv" ] || {
  echo "missing $NM_DIR/warfarin_10pct.csv -- regenerate it with" >&2
  echo "  tools/vi-emvi-comparison/make-wide-residual-data.sh" >&2
  exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp "$NM_DIR/$CTL" "$NM_DIR/warfarin_10pct.csv" "$WORK/"

if [ -n "${DOCKER_IMAGE:-}" ]; then
  echo "running $CTL in $DOCKER_IMAGE"
  docker run --rm -v "$WORK:/work" -w /work "$DOCKER_IMAGE" "$NMFE" "$CTL" "$LST"
else
  command -v "$NMFE" >/dev/null || {
    echo "$NMFE not found on PATH. Either install/point NMFE at your wrapper, or set" >&2
    echo "DOCKER_IMAGE=<image> to run it in a container." >&2
    exit 1; }
  echo "running $CTL with $NMFE"
  ( cd "$WORK" && "$NMFE" "$CTL" "$LST" )
fi

# .ext is the one that matters -- the .lst prints 3 significant figures, which is not enough to
# compare against an engine that agrees to six.
for f in "$LST" warfarin_10pct.ext warfarin_10pct.phi sdtab_10pct; do
  [ -f "$WORK/$f" ] && cp "$WORK/$f" "$NM_DIR/" && echo "  -> tests/nonmem/$f"
done

echo
echo "final estimates (TABLE NO. 1, last iteration):"
awk '/TABLE NO./ {t++} t==1 && $1 == -1000000000 {print; exit}' "$NM_DIR/warfarin_10pct.ext" 2>/dev/null \
  || echo "  (no .ext produced - check $NM_DIR/$LST)"
echo
echo "compare against ferx on the same file:"
echo "  FERX_DATA=\$FERX_VI_STATE/warfarin_10pct.csv tools/vi-emvi-comparison/run.sh"
