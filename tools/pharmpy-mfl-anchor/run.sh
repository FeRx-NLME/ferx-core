#!/usr/bin/env bash
# Regenerate crates/ferx-tools/tests/data/mfl_pharmpy_anchor.json from Pharmpy (#1179).
#
# Needs a container image with Pharmpy at /app/venv (the local `pharma_ai_r`
# image does). Override with FERX_PHARMPY_IMAGE / FERX_PHARMPY_PYTHON.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
image="${FERX_PHARMPY_IMAGE:-pharma_ai_r:latest}"
python="${FERX_PHARMPY_PYTHON:-/app/venv/bin/python}"
out="crates/ferx-tools/tests/data/mfl_pharmpy_anchor.json"

docker run --rm -v "$root:/work" -w /work "$image" "$python" \
  tools/pharmpy-mfl-anchor/dump.py \
  tools/pharmpy-mfl-anchor/corpus.txt \
  data/two_cpt_oral_cov.csv \
  "$out"
