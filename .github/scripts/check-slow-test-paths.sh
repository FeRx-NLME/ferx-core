#!/usr/bin/env bash
#
# Guard for the `push:` path filter in .github/workflows/slow-tests.yml.
#
# The filter decides whether a push to `main` re-runs the heavy fit suite. Two
# ways it can silently stop covering something, both of which look identical to
# a correct filter when you read the diff:
#
#   1. A listed path is deleted, renamed, or split, leaving a glob that matches
#      nothing. `src/api.rs` did exactly this after the `src/api/` split and the
#      push leg went dark for months (#949).
#   2. A new top-level `src/` module appears and nobody adds it to the list, so
#      changes to it never trigger the suite that is supposed to cover them.
#
# This script fails on both. Run it locally with:
#
#     .github/scripts/check-slow-test-paths.sh
#
set -euo pipefail

WORKFLOW='.github/workflows/slow-tests.yml'

# Top-level `src/` entries deliberately NOT in the filter, because a change to
# them cannot move a heavy fit's objective or break one. Adding an entry here is
# a claim you have to be able to defend — prefer listing the path in the
# workflow.
ALLOWLIST=(
  'bin'                    # dev-only tooling (generate_data), not linked into a fit
  'build_info.rs'          # version/commit strings surfaced in output
  'cancel.rs'              # cooperative cancellation flag
  'categorical'            # no Tier-3 fit today; this guard fires when one is added
  'diagnostics.rs'         # warning-code registry; text only
  'environment.rs'         # thread/env probing
  'lib.rs'                 # module wiring and re-exports only
  'propensity_match.rs'    # standalone matching utility, not on any fit path
  'serde_nalgebra.rs'      # (de)serialisation helpers for output
  'types_test_helpers.rs'  # test scaffolding
  'types_tests.rs'         # test scaffolding
)

if [[ ! -f "$WORKFLOW" ]]; then
  echo "error: $WORKFLOW not found (run from the repository root)" >&2
  exit 1
fi

# Collect the quoted globs from the `paths:` block, stopping at the first line
# that leaves it (a key at column 0, e.g. `env:`).
# (a `while read` loop rather than `mapfile`, so this also runs on the bash 3.2
# that ships with macOS)
GLOBS=()
while IFS= read -r line; do
  GLOBS+=("$line")
done < <(
  awk '
    /^    paths:$/       { inblock = 1; next }
    inblock && /^[^ ]/   { inblock = 0 }
    inblock && /^ *- '"'"'/ {
      match($0, /'"'"'[^'"'"']*'"'"'/)
      print substr($0, RSTART + 1, RLENGTH - 2)
    }
  ' "$WORKFLOW"
)

if (( ${#GLOBS[@]} == 0 )); then
  echo "error: parsed zero path globs out of $WORKFLOW — the block moved or changed shape." >&2
  echo "       Fix this script rather than deleting it: a silently-empty parse would make the" >&2
  echo "       guard pass on any filter at all." >&2
  exit 1
fi

status=0

# ---- 1. every listed glob must match at least one tracked file ---------------
for glob in "${GLOBS[@]}"; do
  probe="${glob%/\*\*}"
  if [[ -z "$(git ls-files -- "$probe")" ]]; then
    echo "error: $WORKFLOW lists '$glob', which matches no tracked file." >&2
    echo "       A glob that matches nothing silently skips the slow-test push leg (#949)." >&2
    status=1
  fi
done

# ---- 2. every top-level src/ entry must be listed or explicitly allowlisted --
covered() {
  local entry="$1" glob allowed
  for glob in "${GLOBS[@]}"; do
    [[ "$glob" == "src/$entry" || "$glob" == "src/$entry/**" ]] && return 0
  done
  for allowed in "${ALLOWLIST[@]}"; do
    [[ "$allowed" == "$entry" ]] && return 0
  done
  return 1
}

while read -r entry; do
  [[ -z "$entry" ]] && continue
  if ! covered "$entry"; then
    echo "error: src/$entry is neither listed in $WORKFLOW nor on this script's allowlist." >&2
    echo "       Add it to the workflow's push paths if a change there can move a heavy fit's" >&2
    echo "       objective; otherwise add it to ALLOWLIST here, with a reason." >&2
    status=1
  fi
done < <(git ls-files 'src/*' | cut -d/ -f2 | sort -u)

if (( status == 0 )); then
  echo "slow-test path filter OK: ${#GLOBS[@]} globs, all match; all src/ modules accounted for."
fi

exit "$status"
