#!/usr/bin/env bash
# Phase 1a benchmark (#1965): impmap_map_recheck periods on two_cpt_oral_cov,
# and the default-period recheck on the t4 proposal across all four models.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
python3 bench.py run --models two_cpt_oral_cov \
  --arms impmap_prng_recheck5,impmap_prng_recheck10,impmap_prng_recheck25 \
  --seeds 10 --jobs 4 --eval
python3 bench.py run --models two_cpt_oral_cov,warfarin,three_cpt_iv,sparse_combined \
  --arms impmap_t4_recheck10 --seeds 10 --jobs 4 --eval
python3 bench.py report > /dev/null
echo PHASE1A_DONE
