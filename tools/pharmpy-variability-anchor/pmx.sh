#!/usr/bin/env bash
# Run a command inside the licensed `pmx` NONMEM container (Pharmpy 2.2.0
# pip-installed there), from one of the Pharmpy anchor directories under
# crates/ferx-tools/tests/pharmpy/ — the host repo is mounted at the same
# path under /home/rstudio (see the README next to this script).
#
#   tools/pharmpy-variability-anchor/pmx.sh iivsearch_anchor python3 simulate_iiv.py
#   tools/pharmpy-variability-anchor/pmx.sh iovsearch_anchor python3 run_iovsearch.py
set -euo pipefail
sub="$1"
shift
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
inside="/home/rstudio/${here#/Users/ron/}/crates/ferx-tools/tests/pharmpy/${sub}"
exec docker exec -w "$inside" pmx "$@"
