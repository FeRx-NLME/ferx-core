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
#   select LIST OUT [--all-genres]   reduce a `cargo mutants --list` to the mutants
#                                     worth a rebuild each (see `cmd_select`); write
#                                     them to OUT as a cargo-mutants config whose
#                                     `examine_re` names each one exactly; print the
#                                     count.
#   summary [--selected N] [--window R] DIR...
#                                     merge one or more `mutants.out` dirs (one per
#                                     shard) into a markdown report on stdout; with
#                                     --selected, say how many the budget left untested.
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
# another 17,424 were six more tuple-returning functions. Every one of those is a
# lib rebuild. The variants differ only in which default each tuple slot gets;
# the question the issue asks — "is this body pinned at all?" — is answered by the
# first. So:
#
#   - function-body replacements (FnValue): ONE per function body, the first listed;
#   - everything else (operators, match arms/guards, struct fields): only with
#     --all-genres, and then every one, since each is a distinct edit.
#
# The default is FnValue only because that is the class #1171 and #1223 survived
# (dropping a call, deleting a fill) and because the day measured above still had
# 2,365 mutants with operators included against 423 without.
#
# A FnValue line reads `FILE:LINE:COL: replace FN[ -> T] with VALUE`; every other
# genre's line ends ` in FN` (or names a struct field), which is how they are told
# apart — `VALUE` is a Rust expression and never contains ` in `.
cmd_select() {
  local list="" out="" all=0
  while (($#)); do
    case "$1" in
      --all-genres) all=1; shift ;;
      *) if [[ -z "$list" ]]; then list="$1"; elif [[ -z "$out" ]]; then out="$1"; else die "select: unexpected '$1'"; fi; shift ;;
    esac
  done
  [[ -n "$list" && -n "$out" ]] || die "select: usage: select LIST OUT [--all-genres]"
  [[ -f "$list" ]] || die "select: no such list '$list'"
  python3 -I - "$list" "$out" "$all" <<'PY'
import re, sys

list_path, out_path, all_genres = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
fn_value = re.compile(r"^([^:]+:\d+:\d+): replace (.+?) (?:-> .+ )?with (.+)$")

def is_fn_value(line):
    m = fn_value.match(line)
    return m if m and " in " not in m.group(3) and not m.group(2).startswith("match ") else None

# The Rust `regex` crate's own escape set (regex_syntax::is_meta_character). Python's
# re.escape escapes more (e.g. space), and Rust rejects an escaped space.
def rust_escape(s):
    return "".join("\\" + c if c in "\\.+*?()|[]{}^$#&-~" else c for c in s)

seen, picked = set(), []
for raw in open(list_path, encoding="utf-8"):
    line = raw.rstrip("\n")
    if not line:
        continue
    m = is_fn_value(line)
    if m:
        key = m.group(1)
    elif all_genres:
        key = line
    else:
        continue
    if key in seen:
        continue
    seen.add(key)
    picked.append(line)

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

cmd_summary() {
  local selected="" window="" dirs=()
  while (($#)); do
    case "$1" in
      --selected) selected="$2"; shift 2 ;;
      --window) window="$2"; shift 2 ;;
      *) dirs+=("$1"); shift ;;
    esac
  done
  ((${#dirs[@]} >= 1)) || die "summary: usage: summary [--selected N] [--window RANGE] MUTANTS_OUT_DIR..."
  [[ -z "$selected" || "$selected" =~ ^[0-9]+$ ]] || die "summary: --selected must be a count, got '$selected'"
  set -- "${dirs[@]}"
  local caught missed timeout unviable total
  caught="$(count_in caught.txt "$@")"
  missed="$(count_in missed.txt "$@")"
  timeout="$(count_in timeout.txt "$@")"
  unviable="$(count_in unviable.txt "$@")"
  total=$((caught + missed + timeout + unviable))

  echo "### Mutants: ${missed} survived of ${total} examined"
  echo
  [[ -n "$window" ]] && echo "Window: \`${window}\`" && echo
  # A shard that hit its wall-clock budget stops early. Say so in the headline
  # numbers, not in a log: "0 survived" over half the selection is not a clean day.
  if [[ -n "$selected" ]]; then
    if ((total < selected)); then
      echo "**Budget hit: examined ${total} of ${selected} selected mutants;** the other $((selected - total)) were not tested."
    else
      echo "Examined all ${selected} selected mutants."
    fi
    echo
  fi
  echo "| caught | missed | timeout | unviable | total |"
  echo "|---:|---:|---:|---:|---:|"
  echo "| ${caught} | ${missed} | ${timeout} | ${unviable} | ${total} |"
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
    summary) cmd_summary "$@" ;;
    *) die "unknown subcommand '$sub' (try --help)" ;;
  esac
}

main "$@"
