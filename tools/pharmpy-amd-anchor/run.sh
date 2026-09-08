#!/usr/bin/env bash
# Regenerate crates/ferx-tools/tests/data/amd_pharmpy_anchor.json (#1184).
#
# Runs inside the licensed `pmx` container, where Pharmpy 2.2.0 is
# pip-installed (see tools/pharmpy-variability-anchor/README.md). Nothing is
# fitted — the dump is Pharmpy's subtool order and its search-space split — so
# NONMEM is not touched and the run takes a second.
#
#   tools/pharmpy-amd-anchor/run.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
container="${FERX_PHARMPY_CONTAINER:-pmx}"
inside="/home/rstudio/${here#/Users/ron/}"

docker exec -w "$inside" "$container" python3 \
  tools/pharmpy-amd-anchor/dump.py \
  tools/pharmpy-amd-anchor/corpus.txt \
  crates/ferx-tools/tests/data/amd_pharmpy_anchor.json
