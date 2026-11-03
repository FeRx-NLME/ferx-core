#!/usr/bin/env bash
#
# Nightly cargo-mutants over the previous day's merges to `main` (#2633, part of
# #2466). `--help` prints usage; this header is the why.
#
# AGENTS.md asks for mutation evidence on every new assertion, and that step is
# done by hand and is the one most often incomplete. cargo-mutants' function-body
# replacement is exactly the edit that survived review twice: dropping the
# `push_route_lag_break_times` calls (#1171 / #1174) and deleting the dense fill
# (#1223). Running it nightly over what merged yesterday turns "did anyone mutate
# this?" into a job summary listing the survivors as `file:line` + mutation.
#
# What it does NOT cover: cargo-mutants never mutates string literals, so a false
# diagnostic sentence (#1405 / #1456) still needs the manual sentence-deletion
# check AGENTS.md describes. A clean summary says nothing about message text.
#
# Report-only by design. A surviving mutant is a question for a human — "add a
# test, or write down why not" — not a red build, so the workflow never fails on
# one. It fails only when the tool itself cannot run (bad diff, baseline red).
#
# Subcommands (all used by `.github/workflows/mutants.yml`; each is a pure function
# of its inputs so `tests/nightly_mutants_script.rs` can drive it without cargo):
#
#   range  [--until ISO] [--hours N]  print "BASE HEAD": the first-parent `main`
#                                     commits bounding the window (until-N h, until].
#   diff   BASE HEAD OUT              write `git diff BASE HEAD -- src/` minus test
#                                     files to OUT; print the number of changed files.
#   select LIST OUT [--all-genres|--exact]
#                                     reduce a `cargo mutants --list` to the mutants
#                                     worth a rebuild each (see `cmd_select`); write
#                                     them to OUT as a cargo-mutants config whose
#                                     `examine_re` names each one exactly; print the
#                                     count.
#   untested SELECTED OUT DIR...      write the names in SELECTED that no DIR has an
#                                     outcome for to OUT; print the count.
#   summary [--selected N] [--window R] [--untested FILE [--continued]] DIR...
#                                     merge one or more `mutants.out` dirs (one per
#                                     shard, plus earlier passes) into a markdown
#                                     report on stdout: survivors, timeouts, measured
#                                     per-mutant time, and any mutant left untested.
set -euo pipefail

usage() {
  sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed -e 's/^# \{0,1\}//' -e '/^set -euo/d'
}

die() {
  echo "nightly-mutants: $*" >&2
  exit 1
}

# The window is anchored on COMMITTER date of first-parent commits, because that is
# when a merge landed on `main` (the author date of a merge commit is when the PR
# author last rebased, which can be days earlier). `--first-parent` so a merged
# branch's own commits are not window boundaries — the merge commit is.
cmd_range() {
  local until="" hours=24 ref="${MUTANTS_MAIN_REF:-origin/main}"
  while (($#)); do
    case "$1" in
      --until) until="$2"; shift 2 ;;
      --hours) hours="$2"; shift 2 ;;
      --ref) ref="$2"; shift 2 ;;
      *) die "range: unknown argument '$1'" ;;
    esac
  done
  [[ "$hours" =~ ^[0-9]+$ ]] || die "range: --hours must be a whole number, got '$hours'"
  if [[ -z "$until" ]]; then
    # Midnight UTC today: a run at any time of day covers the same calendar day.
    until="$(date -u +%Y-%m-%dT00:00:00Z)"
  fi
  local until_epoch since_epoch
  until_epoch="$(date_to_epoch "$until")"
  since_epoch=$((until_epoch - hours * 3600))

  local head base
  head="$(git rev-list --first-parent -1 --before="@$until_epoch" "$ref")"
  base="$(git rev-list --first-parent -1 --before="@$since_epoch" "$ref")"
  [[ -n "$head" ]] || die "range: no commit on $ref before $until"
  # A repo younger than the window has no commit before `since`: diff from the root.
  if [[ -z "$base" ]]; then
    base="$(git rev-list --first-parent --max-parents=0 "$ref" | tail -1)"
  fi
  echo "$base $head"
}

# GNU date and BSD date disagree on parsing; git does not, and it is already a
# dependency. `git log --since` would need a commit; `rev-parse --since` prints the
# parsed epoch as `--max-age=<epoch>` for any date string git understands.
date_to_epoch() {
  local out
  out="$(git rev-parse --since="$1")"
  echo "${out#--max-age=}"
}

# Which files are fair game. Test code is not mutated: a mutant in a test is a
# question about the test, not about coverage of production code, and cargo-mutants
# would otherwise rebuild the lib for every line of a large `*_tests.rs` sibling.
is_test_path() {
  case "$1" in
    tests/* | */tests/* | *_tests.rs | *test_helpers.rs | */test_helpers/*) return 0 ;;
    *) return 1 ;;
  esac
}

cmd_diff() {
  (($# == 3)) || die "diff: usage: diff BASE HEAD OUT"
  local base="$1" head="$2" out="$3"
  local files=() f
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    is_test_path "$f" && continue
    files+=("$f")
  done < <(git diff --name-only --diff-filter=AMR "$base" "$head" -- 'src/*.rs')
  if ((${#files[@]} == 0)); then
    : >"$out"
  else
    git diff "$base" "$head" -- "${files[@]}" >"$out"
  fi
  echo "${#files[@]}"
}

# Why a selection step at all: cargo-mutants generates one FnValue mutant per
# *combination* of default values for a function's return type, so a tuple return
# multiplies. Measured on the 2026-11-02 merges (#2633): 58,192 mutants listed, of
# which 34,848 were ONE function (`run_inner_loop_and_nll_prepared`, a 7-tuple) and
# another 17,424 were six more tuple-returning functions. Every one is a lib rebuild.
#
# The rule (`cover`), applied per function body (one source location):
#
#   Split each generated value into its SLOTS — the innermost comma-separated
#   elements, keyed by position, so `Ok((vec![], 0.0))` has slots `vec![]` and
#   `0.0` — and keep the fewest variants (greedy, list order on ties) such that
#   every value cargo-mutants generated for every slot appears in at least one.
#
# For a non-tuple return the whole value is the one slot, so EVERY variant is kept:
# `-> bool` runs both `true` and `false`, `-> Option<f64>` all of `None`, `Some(0.0)`,
# `Some(1.0)`, `Some(-1.0)`. Only tuple products are pruned, and then to the
# per-slot alternatives: `run_inner_loop_and_nll_prepared` runs 12 variants instead
# of 34,848, and every replacement of each of its slots still runs at least once.
# The 2026-11-02 merges select 1,261 mutants this way (447 function bodies).
# A survivor hidden by the pruning would need a tuple combination to survive while
# every variant carrying each of its slot values is caught — a cross-slot
# interaction, which is the one thing this does not test.
#
# Operators, match arms/guards and struct fields: only with --all-genres (then
# every one, since each is a distinct edit). The default is function bodies because
# that is the class #1171 and #1223 survived (dropping a call, deleting a fill).
#
# A FnValue line reads `FILE:LINE:COL: replace FN[ -> T] with VALUE`; every other
# genre's line ends ` in FN` (or names a struct field), which is how they are told
# apart — `VALUE` is a Rust expression and never contains ` in `.
#
# --exact skips all of that and names every line of LIST verbatim: the follow-up
# pass uses it for the mutants a budget-hit pass left untested.
cmd_select() {
  local list="" out="" mode=default
  while (($#)); do
    case "$1" in
      --all-genres) mode=all; shift ;;
      --exact) mode=exact; shift ;;
      *) if [[ -z "$list" ]]; then list="$1"; elif [[ -z "$out" ]]; then out="$1"; else die "select: unexpected '$1'"; fi; shift ;;
    esac
  done
  [[ -n "$list" && -n "$out" ]] || die "select: usage: select LIST OUT [--all-genres|--exact]"
  [[ -f "$list" ]] || die "select: no such list '$list'"
  python3 -I - "$list" "$out" "$mode" <<'PY'
import re, sys

list_path, out_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
fn_value = re.compile(r"^([^:]+:\d+:\d+): replace (.+?) (?:-> .+ )?with (.+)$")
OPEN, CLOSE = "([{", ")]}"

def is_fn_value(line):
    m = fn_value.match(line)
    return m if m and " in " not in m.group(3) and not m.group(2).startswith("match ") else None

def split_top(s, sep=","):
    """Split at depth-0 separators; None if the brackets do not balance."""
    parts, depth, cur = [], 0, ""
    for c in s:
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth -= 1
            if depth < 0:
                return None
        if c == sep and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += c
    if depth != 0:
        return None
    parts.append(cur.strip())
    return parts

def groups(s):
    """Contents of the depth-0 bracket groups of s, in order."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c in OPEN:
            if depth == 0:
                start = i + 1
            depth += 1
        elif c in CLOSE:
            depth -= 1
            if depth == 0:
                out.append(s[start:i])
    return out

def has_comma(s):
    parts = split_top(s)
    if parts is None:
        return False
    return len(parts) > 1 or any(has_comma(g) for g in groups(s))

def slots(s, path=()):
    """(position, text) of the innermost comma-separated elements of a value."""
    parts = split_top(s)
    if parts is None:
        return {(path, s)}
    if len(parts) > 1:
        out = set()
        for i, p in enumerate(parts):
            out |= slots(p, path + (i,))
        return out
    for j, g in enumerate(groups(s)):
        if has_comma(g):
            return slots(g, path + ("g", j))
    return {(path, s)}

def cover(variants):
    """Fewest variants (greedy, list order on ties) covering every slot value."""
    sl = [slots(v) for _, v in variants]
    need = set().union(*sl)
    picked = []
    while need:
        best = max(range(len(variants)), key=lambda i: (len(sl[i] & need), -i))
        picked.append(best)
        need -= sl[best]
    return [variants[i][0] for i in sorted(picked)]

# The Rust `regex` crate's own escape set (regex_syntax::is_meta_character). Python's
# re.escape escapes more (e.g. space), and Rust rejects an escaped space.
def rust_escape(s):
    return "".join("\\" + c if c in "\\.+*?()|[]{}^$#&-~" else c for c in s)

lines = [l.rstrip("\n") for l in open(list_path, encoding="utf-8")]
lines = list(dict.fromkeys(l for l in lines if l))
if mode == "exact":
    picked = lines
else:
    by_loc, order, others = {}, [], []
    for line in lines:
        m = is_fn_value(line)
        if m:
            if m.group(1) not in by_loc:
                by_loc[m.group(1)] = []
                order.append(m.group(1))
            by_loc[m.group(1)].append((line, m.group(3)))
        elif mode == "all":
            others.append(line)
    keep = set(others)
    for loc in order:
        keep.update(cover(by_loc[loc]))
    picked = [l for l in lines if l in keep]

with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Generated by tools/nightly-mutants.sh select; do not edit.\n")
    f.write("examine_re = [\n")
    for line in picked:
        pat = "^" + rust_escape(line) + "$"
        f.write("  " + '"' + pat.replace("\\", "\\\\").replace('"', '\\"') + '"' + ",\n")
    f.write("]\n")
print(len(picked))
PY
}

# Counts lines across the same-named file in every shard dir; a shard that ran
# zero mutants has no such file, which counts as zero.
count_in() {
  local name="$1" total=0 d n
  shift
  for d in "$@"; do
    if [[ -f "$d/$name" ]]; then
      n="$(grep -c . "$d/$name" || true)"
      total=$((total + n))
    fi
  done
  echo "$total"
}

cat_in() {
  local name="$1" d
  shift
  for d in "$@"; do
    [[ -f "$d/$name" ]] && grep . "$d/$name" || true
  done | LC_ALL=C sort -t: -k1,1 -k2,2n -u
}

# The mutants of SELECTED (a `cargo mutants --list` output, one name per line) that
# no shard dir records an outcome for: what a budget-hit pass left untested. Written
# to OUT; the count is printed. The workflow hands OUT to the next pass.
cmd_untested() {
  (($# >= 2)) || die "untested: usage: untested SELECTED OUT [MUTANTS_OUT_DIR...]"
  local selected="$1" out="$2"
  shift 2
  [[ -f "$selected" ]] || die "untested: no such list '$selected'"
  local d
  for d in "$@"; do
    for f in caught.txt missed.txt timeout.txt unviable.txt; do
      [[ -f "$d/$f" ]] && grep . "$d/$f" || true
    done
  done | LC_ALL=C sort -u >"$out.examined"
  grep . "$selected" | LC_ALL=C sort -u | LC_ALL=C comm -23 - "$out.examined" >"$out" || true
  rm -f "$out.examined"
  grep -c . "$out" || true
}

# Per-mutant wall time from each shard's outcomes.json, so every run records its own
# measured CI cost in the summary (#2633 asks for it; the budget is sized from it).
timing_line() {
  python3 -I - "$@" <<'PY'
import json, os, statistics, sys
build, test = [], []
for d in sys.argv[1:]:
    p = os.path.join(d, "outcomes.json")
    try:
        doc = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        continue
    for o in doc.get("outcomes", []):
        if not isinstance(o.get("scenario"), dict):
            continue  # the baseline
        ph = {r["phase"]: r["duration"] for r in o.get("phase_results", [])}
        if "Build" in ph:
            build.append(ph["Build"])
            test.append(ph.get("Test", 0.0))
if build:
    tot = [b + t for b, t in zip(build, test)]
    print(f"Measured on this run: {len(build)} mutants, median {statistics.median(build):.0f} s build "
          f"+ {statistics.median(test):.0f} s test = {statistics.median(tot):.0f} s per mutant "
          f"(max {max(tot):.0f} s).")
PY
}

cmd_summary() {
  local selected="" window="" untested="" continued=0 dirs=()
  while (($#)); do
    case "$1" in
      --selected) selected="$2"; shift 2 ;;
      --window) window="$2"; shift 2 ;;
      --untested) untested="$2"; shift 2 ;;
      --continued) continued=1; shift ;;
      *) dirs+=("$1"); shift ;;
    esac
  done
  ((${#dirs[@]} >= 1)) || die "summary: usage: summary [--selected N] [--window RANGE] [--untested FILE [--continued]] MUTANTS_OUT_DIR..."
  [[ -z "$selected" || "$selected" =~ ^[0-9]+$ ]] || die "summary: --selected must be a count, got '$selected'"
  set -- "${dirs[@]}"
  local caught missed timeout unviable total n_untested=0
  caught="$(count_in caught.txt "$@")"
  missed="$(count_in missed.txt "$@")"
  timeout="$(count_in timeout.txt "$@")"
  unviable="$(count_in unviable.txt "$@")"
  total=$((caught + missed + timeout + unviable))
  [[ -n "$untested" && -f "$untested" ]] && n_untested="$(grep -c . "$untested" || true)"

  echo "### Mutants: ${missed} survived of ${total} examined"
  echo
  [[ -n "$window" ]] && echo "Window: \`${window}\`" && echo
  # A shard that hit its wall-clock budget stops early. Say so in the headline
  # numbers, not in a log: "0 survived" over half the selection is not a clean day.
  if [[ -n "$selected" ]]; then
    if ((total < selected)); then
      echo -n "**Budget hit: examined ${total} of ${selected} selected mutants;** the other $((selected - total))"
      if ((continued)); then
        echo " carry over to a follow-up pass on the same commit, whose summary includes this one's."
      else
        echo " were not tested and are listed below."
      fi
    else
      echo "Examined all ${selected} selected mutants."
    fi
    echo
  fi
  echo "| caught | missed | timeout | unviable | total |"
  echo "|---:|---:|---:|---:|---:|"
  echo "| ${caught} | ${missed} | ${timeout} | ${unviable} | ${total} |"
  echo
  timing_line "$@"
  echo
  if ((missed > 0)); then
    echo "#### Surviving mutants"
    echo
    echo "Each one is an edit to production code that every \`--lib\` test still passes."
    echo "Triage: add a test that kills it, or record why it cannot matter (#2633)."
    echo
    echo '```'
    cat_in missed.txt "$@"
    echo '```'
    echo
  fi
  if ((timeout > 0)); then
    echo "#### Timed out"
    echo
    echo "Usually a mutant that turned a loop infinite — read as caught, but check."
    echo
    echo '```'
    cat_in timeout.txt "$@"
    echo '```'
    echo
  fi
  if ((n_untested > 0 && !continued)); then
    echo "#### Not examined"
    echo
    echo "Selected but never run: the passes ran out. Re-dispatch this window to test them."
    echo
    echo '```'
    LC_ALL=C sort -t: -k1,1 -k2,2n "$untested"
    echo '```'
    echo
  fi
  echo "_cargo-mutants does not mutate string literals: a false diagnostic sentence"
  echo "still needs the manual sentence-deletion check (AGENTS.md, #1405)._"
}

main() {
  (($#)) || { usage; exit 2; }
  local sub="$1"
  shift
  case "$sub" in
    -h | --help) usage ;;
    range) cmd_range "$@" ;;
    diff) cmd_diff "$@" ;;
    select) cmd_select "$@" ;;
    untested) cmd_untested "$@" ;;
    summary) cmd_summary "$@" ;;
    *) die "unknown subcommand '$sub' (try --help)" ;;
  esac
}

main "$@"
