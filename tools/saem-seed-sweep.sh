#!/usr/bin/env bash
#
# Is a SAEM objective change between two builds a regression, or a different draw?
# `--help` prints usage; this header is the why (#2602).
#
# SAEM is deterministic at a fixed `seed`, so two builds that agree to the last digit
# and then stop agreeing have changed *something* on the chain's path — often only
# the order in which random numbers are consumed. One seed per build cannot tell a
# systematically worse trajectory from a different, equally good one: on #2602's
# public stand-ins (a 450-subject 2-cpt block-Ω closed form and a 3-state ODE), the
# importance-sampled −2 log L of a single build has a seed-to-seed sd of 1.5 and
# 3.4–4.1 (range 7 and 14–15) over 40 seeds, against an IS-draw-only sd of 0.55 and
# 0.9. A +4 or +7 at one seed is therefore inside the chain's own spread. The
# comparison has to be of distributions, which is what this script runs:
#
#   - every REV is exported with `git archive` (no worktree, no checkout) and built
#     once under the `ci-test` profile into its own target dir under WORKDIR, which
#     defaults to `target/saem-seed-sweep` — NOT /tmp, which is a small tmpfs on
#     some dev boxes and fills up mid-build;
#   - the model's `seed` and `imp_seed` lines are replaced (or added) per run, with
#     imp_seed = seed + 1000 unless --imp-seed pins it;
#   - the objective read is the `-2 log L (IS)` line, so the model must end in an
#     importance-sampling stage (`method = [saem, imp]` with `imp_eval_only = true`);
#   - the summary is per-REV n / mean / sd / min / max, plus the mean shift of each
#     later REV against the first with its Welch standard error.
#
# A shift whose |t| stays below ~2 with both builds' ranges overlapping is chain
# variation; a consistent upward shift is a regression worth bisecting. With
# --imp-seed fixed and --fixed-saem-seed, the sweep instead varies only the IS draw
# and measures the IS Monte-Carlo noise floor on its own.

set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/saem-seed-sweep.sh --model M.ferx --data D.csv --rev REV [--rev REV ...]
                                [--seeds "1 2 3 4 5"] [--imp-seed N] [--fixed-saem-seed N]
                                [--jobs N] [--threads N] [--workdir DIR]

  --model            model file; must end in an IS stage (method = [saem, imp],
                     imp_eval_only = true) so the run prints `-2 log L (IS)`
  --data             dataset passed to `ferx --data`
  --rev              git revision to build and run (repeatable; the first is the baseline)
  --seeds            SAEM seeds to sweep (default "1 2 3 4 5 6 7 8 9 10")
  --imp-seed         pin imp_seed for every run (default: seed + 1000)
  --fixed-saem-seed  hold the SAEM seed at N and treat --seeds as imp_seed values,
                     to measure the IS Monte-Carlo noise alone
  --jobs             concurrent fits (default 4)
  --threads          --threads passed to each fit (default 4)
  --workdir          builds and runs go here (default ${CARGO_TARGET_DIR:-target}/saem-seed-sweep)

Prints a TSV of every run (rev, seed, imp_seed, objective) and a per-rev summary.
EOF
}

model="" data="" revs=() seeds="1 2 3 4 5 6 7 8 9 10" imp_seed="" fixed_saem=""
jobs=4 threads=4 workdir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) model=$2; shift 2 ;;
    --data) data=$2; shift 2 ;;
    --rev) revs+=("$2"); shift 2 ;;
    --seeds) seeds=$2; shift 2 ;;
    --imp-seed) imp_seed=$2; shift 2 ;;
    --fixed-saem-seed) fixed_saem=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --workdir) workdir=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -z "$model" ] || [ -z "$data" ] || [ ${#revs[@]} -eq 0 ]; then
  usage >&2; exit 2
fi
# Seeds become run-directory names that are `rm -rf`-ed before each run, so they
# must be plain non-negative integers: anything else (`../..`, a slash) could name
# a directory outside the sweep.
is_uint() { [[ $1 =~ ^[0-9]+$ ]]; }
for v in $seeds $imp_seed $fixed_saem; do
  is_uint "$v" || { echo "error: seed '$v' is not a non-negative integer" >&2; exit 2; }
done
for v in "$jobs" "$threads"; do
  is_uint "$v" || { echo "error: '$v' is not a non-negative integer" >&2; exit 2; }
done
grep -Eq '^[[:space:]]*\[fit_options\]' "$model" || {
  echo "error: $model has no [fit_options] block to put the seeds in" >&2; exit 2; }

repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
workdir=${workdir:-${CARGO_TARGET_DIR:-$repo/target}/saem-seed-sweep}
mkdir -p "$workdir"
workdir=$(cd "$workdir" && pwd)
model=$(cd "$(dirname "$model")" && pwd)/$(basename "$model")
data=$(cd "$(dirname "$data")" && pwd)/$(basename "$data")

# Build each rev once; a rev already built (same full SHA) is reused.
shas=()
for rev in "${revs[@]}"; do
  sha=$(git -C "$repo" rev-parse --verify "$rev^{commit}")
  shas+=("$sha")
  bin=$workdir/target-$sha/ci-test/ferx
  if [ ! -x "$bin" ]; then
    echo "building $rev ($sha)..." >&2
    rm -rf "$workdir/src-$sha"; mkdir -p "$workdir/src-$sha"
    git -C "$repo" archive "$sha" | tar -x -C "$workdir/src-$sha"
    (cd "$workdir/src-$sha" &&
      CARGO_TARGET_DIR=$workdir/target-$sha cargo build --profile ci-test -p ferx-cli) \
      > "$workdir/build-$sha.log" 2>&1 ||
      { echo "error: build of $rev failed, see $workdir/build-$sha.log" >&2; exit 1; }
  fi
done

# One fit. Arguments: sha seed imp_seed. Writes <rundir>/objective.
run_one() {
  local sha=$1 s=$2 is=$3
  [[ $sha =~ ^[0-9a-f]{40}$ && $s =~ ^[0-9]+$ && $is =~ ^[0-9]+$ ]] ||
    { echo "error: refusing run directory for '$sha' '$s' '$is'" >&2; return 1; }
  local d=$workdir/runs/$sha/s$s-i$is
  rm -rf "$d"; mkdir -p "$d"
  # Drop any existing seed / imp_seed line, then add both right under [fit_options].
  awk -v s="$s" -v is="$is" '
    /^[[:space:]]*(seed|imp_seed)[[:space:]]*=/ { next }
    { print }
    /^[[:space:]]*\[fit_options\]/ { print "  seed = " s; print "  imp_seed = " is }
  ' "$model" > "$d/model.ferx"
  # The CLI writes model-fit.yaml, model-sdtab.csv and its model.tmp checkpoint to
  # its WORKING directory, so each run has to start in its own: run from the
  # caller's directory, parallel fits overwrite one another's outputs, race on
  # one checkpoint file, and clobber any model-* files the caller already had.
  (cd "$d" && "$workdir/target-$sha/ci-test/ferx" "$d/model.ferx" --data "$data" \
    --threads "$threads" --clean) > "$d/stdout.log" 2>&1 || true
  # Keep only a finite decimal number. A collapsed IS run can print `inf` or
  # `NaN`, which awk reads as 0 and would fold into the mean as a real value;
  # those, like a run with no IS line at all, become NA.
  sed -n 's/.*-2 log L (IS): *\([^ ]*\).*/\1/p' "$d/stdout.log" | head -1 |
    grep -E '^[-+]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][-+]?[0-9]+)?$' > "$d/objective" || true
}
export -f run_one
export workdir model data threads

for sha in "${shas[@]}"; do
  for s in $seeds; do
    if [ -n "$fixed_saem" ]; then echo "$sha $fixed_saem $s"
    elif [ -n "$imp_seed" ]; then echo "$sha $s $imp_seed"
    else echo "$sha $s $((s + 1000))"
    fi
  done
done | xargs -P "$jobs" -L 1 bash -c 'run_one "$@"' _

table=$workdir/results.tsv
printf 'rev\tsha\tseed\timp_seed\tobjective\n' > "$table"
for i in "${!shas[@]}"; do
  sha=${shas[$i]}
  for s in $seeds; do
    if [ -n "$fixed_saem" ]; then ss=$fixed_saem is=$s
    elif [ -n "$imp_seed" ]; then ss=$s is=$imp_seed
    else ss=$s is=$((s + 1000))
    fi
    obj=$(cat "$workdir/runs/$sha/s$ss-i$is/objective")
    printf '%s\t%s\t%s\t%s\t%s\n' "${revs[$i]}" "${sha:0:8}" "$ss" "$is" "${obj:-NA}" >> "$table"
  done
done
cat "$table"
echo
# A run with no `-2 log L (IS)` line (a failed fit, or a model with no IS stage)
# reads NA and is counted, not silently dropped.
awk -F'\t' '
  NR == 1 { next }
  !($1 in seen) { seen[$1] = 1; order[++k] = $1 }
  $5 == "NA" { na[$1]++; next }
  { n[$1]++; s[$1] += $5; ss[$1] += $5 * $5
    if (!($1 in lo) || $5 < lo[$1]) lo[$1] = $5
    if (!($1 in hi) || $5 > hi[$1]) hi[$1] = $5 }
  END {
    printf "%-14s %4s %4s %12s %8s %12s %12s\n", "rev", "n", "NA", "mean", "sd", "min", "max"
    for (j = 1; j <= k; j++) {
      r = order[j]; m[r] = n[r] ? s[r] / n[r] : 0
      v[r] = n[r] > 1 ? (ss[r] - n[r] * m[r] * m[r]) / (n[r] - 1) : 0
      if (v[r] < 0) v[r] = 0
      printf "%-14s %4d %4d %12.3f %8.3f %12.3f %12.3f\n", r, n[r], na[r], m[r], sqrt(v[r]), lo[r], hi[r]
    }
    b = order[1]
    for (j = 2; j <= k; j++) {
      r = order[j]
      if (n[b] < 2 || n[r] < 2) continue
      se = sqrt(v[b] / n[b] + v[r] / n[r])
      t = se > 0 ? sprintf("%+.2f", (m[r] - m[b]) / se) : "inf"
      printf "%s - %s: %+.3f (Welch SE %.3f, t = %s)\n", r, b, m[r] - m[b], se, t
    }
  }' "$table"
