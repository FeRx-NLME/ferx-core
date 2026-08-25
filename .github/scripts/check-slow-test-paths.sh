#!/usr/bin/env bash
#
# Guard for the `push:` path filter in .github/workflows/slow-tests.yml.
#
# The filter decides whether a push to `main` re-runs the heavy fit suite, and
# it has exactly one way to fail silently: a listed path is deleted, renamed, or
# split, leaving a glob that matches nothing. `src/api.rs` did precisely this
# after the `src/api/` split and the push leg went dark for months, because a
# stale glob and a correct glob are indistinguishable when you read the diff
# (#949).
#
# This script fails when any listed glob matches no tracked file.
#
# There is deliberately no "is every module covered?" check here. The filter
# matches `src/**` rather than enumerating modules, so there is nothing left to
# be incomplete about — and a per-module allowlist would reintroduce exactly the
# standing judgement call this guard exists to remove.
#
# Run it locally with:
#
#     .github/scripts/check-slow-test-paths.sh
#
set -euo pipefail

WORKFLOW='.github/workflows/slow-tests.yml'

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

# ---- every listed glob must match at least one tracked file ------------------
for glob in "${GLOBS[@]}"; do
  probe="${glob%/\*\*}"
  if [[ -z "$(git ls-files -- "$probe")" ]]; then
    echo "error: $WORKFLOW lists '$glob', which matches no tracked file." >&2
    echo "       A glob that matches nothing silently skips the slow-test push leg (#949)." >&2
    status=1
  fi
done

if (( status == 0 )); then
  echo "slow-test path filter OK: ${#GLOBS[@]} globs, all match at least one tracked file."
fi

exit "$status"
