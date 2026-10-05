#!/usr/bin/env bash
#
# Changelog fragments (#1545). `--help` prints usage; this header is the why.
#
# Every user-facing PR used to add a bullet under `## [Unreleased]` in
# CHANGELOG.md, at the same insertion point. With several PRs open and green at
# once, merging the first made every other one conflict on CHANGELOG.md, so each
# needed a local rebase, a conflict resolution and a fresh CI run before it could
# merge. That is structural: the convention forced N concurrent PRs to edit the
# same lines of the same file. `merge=union` does not fix it — GitHub's merge
# button and conflict detection ignore custom merge drivers.
#
# So each PR adds a NEW file instead, `changelog.d/<N>.<category>.md`, and the
# release step assembles them into CHANGELOG.md (the towncrier / scriv / changie
# pattern). Two PRs adding two different files cannot conflict.
#
# Plain bash + awk + sed on purpose: no new runtime dependency, and it runs
# unchanged under macOS's bash 3.2 and CI's bash 5.
#
# `CHANGELOG_ROOT` (default: the repo root) points every subcommand at another
# tree; the tests in `tests/changelog_fragments.rs` use it to run against a
# scratch copy instead of the real CHANGELOG.md.
set -euo pipefail

script_path="${BASH_SOURCE[0]}"
repo_root="$(cd "$(dirname "$script_path")/.." && pwd)"
root="${CHANGELOG_ROOT:-$repo_root}"
frag_dir="$root/changelog.d"
changelog="$root/CHANGELOG.md"

# Keep a Changelog's order, plus `Performance`, which this project has used
# since 0.2.0. The order here is the order of the `###` headings in a release.
CATEGORIES=(added changed deprecated removed fixed security performance)

# Files in changelog.d/ that are not fragments.
NON_FRAGMENTS=(README.md .gitkeep)

# Paths whose change is presumed user-facing. Test-only files under them are
# excluded by `is_test_path` below.
USER_FACING_PREFIXES=(src/ crates/ferx-cli/ crates/ferx-tools/)

OPT_OUT_LABEL="no-changelog"

usage() {
  cat <<EOF
Changelog fragments (#1545): each user-facing PR adds a file under changelog.d/
instead of editing CHANGELOG.md, so concurrent PRs never conflict.

  tools/changelog.sh check                     validate every fragment, and that
                                               CHANGELOG.md's [Unreleased] is empty
  tools/changelog.sh preview                   print the section the fragments
                                               would assemble into
  tools/changelog.sh assemble <version> [--date YYYY-MM-DD]
                                               release time only: write the
                                               fragments into CHANGELOG.md as
                                               ## [<version>] - <date>, update the
                                               compare links, delete the fragments
  tools/changelog.sh require <base-ref> [--pr N] [--opt-out true|false]
                                               fail if the diff base...HEAD touches
                                               user-facing code and adds no fragment,
                                               unless --opt-out true (CI passes
                                               whether the '$OPT_OUT_LABEL' label is set)
  tools/changelog.sh released <tag>            fail unless CHANGELOG.md has the tag's
                                               section and no fragment is pending
                                               (release.yml runs it on every tag)

Fragment: changelog.d/<N>.<category>.md, where <N> is the issue or PR number
(append -2, -3, ... for a second entry on the same number, e.g. 1541-2.fixed.md)
and <category> is one of:
  ${CATEGORIES[*]}
Its body is exactly one Markdown bullet, as it will appear in CHANGELOG.md: the
first line starts with '- ', every further line is blank or indented.
EOF
}

die() {
  echo "changelog: $*" >&2
  exit 1
}

is_category() {
  local c
  for c in "${CATEGORIES[@]}"; do
    [ "$c" = "$1" ] && return 0
  done
  return 1
}

is_non_fragment() {
  local n
  for n in "${NON_FRAGMENTS[@]}"; do
    [ "$n" = "$1" ] && return 0
  done
  return 1
}

# `Added` from `added`. Spelled out rather than `${c^}`, which bash 3.2 lacks.
heading_of() {
  local c="$1"
  printf '%s%s' "$(printf '%s' "${c:0:1}" | tr '[:lower:]' '[:upper:]')" "${c:1}"
}

# Every fragment file name, sorted by its leading number (then lexically, so
# 1541.fixed.md precedes 1541-2.fixed.md).
fragment_names() {
  [ -d "$frag_dir" ] || return 0
  # `find`, not a glob: every glob spelling misses some dotfile shape (`..x`),
  # and a file that `check` never sees but `require` counts is a hole in both.
  #
  # A reserved name (README.md, .gitkeep) is skipped only when it is a plain
  # file: a `.gitkeep/` DIRECTORY is listed, so `check` rejects it rather than
  # letting `require` count a path nested inside it.
  local name
  find "$frag_dir" -mindepth 1 -maxdepth 1 | sed 's#.*/##' | while IFS= read -r name; do
    if is_non_fragment "$name" && [ -f "$frag_dir/$name" ] && [ ! -L "$frag_dir/$name" ]; then
      continue
    fi
    printf '%s\n' "$name"
  done | LC_ALL=C sort -t. -k1,1n -k1,1
}

# The fragment manifest, read ONCE per run into FRAGMENTS. Every consumer —
# check, render, verify, delete — walks this same list, so a file appearing
# mid-assemble is neither rendered nor deleted.
#
# Split on newlines with globbing off, not a `while read ... <<EOF $(...)` loop:
# a here-document needs a temp file, and when bash cannot create one the loop
# silently runs zero times — `check` then reports every fragment OK having read
# none, and exits 0.
FRAGMENTS=()
load_fragments() {
  local names
  names="$(fragment_names)" || die "could not list changelog.d/"
  FRAGMENTS=()
  [ -n "$names" ] || return 0
  local IFS=$'\n'
  set -f
  # shellcheck disable=SC2206 # word-splitting on newlines is the point
  FRAGMENTS=($names)
  set +f
}

# Fragment body with trailing blank lines removed and a final newline ensured.
fragment_body() {
  awk '{ lines[NR] = $0 } NF { last = NR }
       END { for (i = 1; i <= last; i++) print lines[i] }' "$1"
}

# Print the lines of CHANGELOG.md's [Unreleased] section (excluding its heading)
# that are not blank.
unreleased_content() {
  awk '
    /^## \[Unreleased\]/ { inside = 1; next }
    inside && /^## \[/   { exit }
    inside && NF         { print FNR ": " $0 }
  ' "$changelog"
}

cmd_check() {
  local errors=0 name stem cat body first bad headings
  [ -f "$changelog" ] || die "no CHANGELOG.md at $changelog"

  # Exactly one `## [Unreleased]` heading. Without one, an empty section and a
  # missing section look the same to everything below, and `assemble` would
  # delete the fragments having written them nowhere.
  headings="$(grep -c '^## \[Unreleased\][[:space:]]*$' "$changelog" || true)"
  if [ "$headings" != "1" ]; then
    echo "CHANGELOG.md: expected exactly one '## [Unreleased]' heading, found $headings" >&2
    errors=$((errors + 1))
  fi

  load_fragments
  for name in ${FRAGMENTS[@]+"${FRAGMENTS[@]}"}; do
    local path="$frag_dir/$name"
    if [ -L "$path" ] || [ ! -f "$path" ]; then
      echo "changelog.d/$name: not a regular file; fragments are plain files directly" \
        "under changelog.d/ (no directories, no symlinks)" >&2
      errors=$((errors + 1))
      continue
    fi
    # <N>[-<k>].<category>.md, k >= 2: the second entry on one number is `-2`.
    if ! printf '%s' "$name" | grep -Eq '^[0-9]+(-([2-9]|[1-9][0-9]+))?\.[a-z]+\.md$'; then
      echo "changelog.d/$name: name must be <N>.<category>.md (e.g. 1541.fixed.md;" \
        "1541-2.fixed.md for a second entry)" >&2
      errors=$((errors + 1))
      continue
    fi
    stem="${name%.md}"
    cat="${stem##*.}"
    if ! is_category "$cat"; then
      echo "changelog.d/$name: unknown category '$cat' — expected one of: ${CATEGORIES[*]}" >&2
      errors=$((errors + 1))
      continue
    fi
    body="$(fragment_body "$path")"
    if [ -z "$(printf '%s' "$body" | tr -d '[:space:]')" ]; then
      echo "changelog.d/$name: empty — write the bullet as it should read in CHANGELOG.md" >&2
      errors=$((errors + 1))
      continue
    fi
    first="$(printf '%s\n' "$body" | head -n 1)"
    if ! printf '%s' "$first" | grep -Eq '^- [^[:space:]]'; then
      echo "changelog.d/$name: line 1 must start a bullet ('- ...'), found: $first" >&2
      errors=$((errors + 1))
      continue
    fi
    # Every later non-blank line must be indented by at least two spaces (or a
    # tab) — the content column of `- `, so Markdown keeps it inside the one
    # bullet. A column-0 line is a second bullet, a heading, or prose outside
    # the list; and ` - x` with ONE space is still a sibling bullet, not a
    # nested one.
    bad="$(printf '%s\n' "$body" | awk 'NR > 1 && NF && !/^(  |\t)/ { print NR ": " $0; exit }')"
    if [ -n "$bad" ]; then
      echo "changelog.d/$name: must be a single bullet; line $bad is not indented" \
        "by two spaces (continuation lines need at least two; one entry per file)" >&2
      errors=$((errors + 1))
      continue
    fi
    if printf '%s' "$body" | grep -q $'\r'; then
      echo "changelog.d/$name: contains CR line endings; save it with LF" >&2
      errors=$((errors + 1))
      continue
    fi
    if ! printf '%s' "$body" | grep -Eq '#[0-9]+'; then
      echo "changelog.d/$name: no issue/PR reference — cite it as (#NN)" >&2
      errors=$((errors + 1))
      continue
    fi
  done

  local stray
  stray="$(unreleased_content)"
  if [ -n "$stray" ]; then
    {
      echo "CHANGELOG.md: [Unreleased] must stay empty between releases (#1545) — it is"
      echo "assembled from changelog.d/ at release time. Move each entry into its own"
      echo "changelog.d/<N>.<category>.md. Offending lines:"
      printf '%s\n' "$stray" | head -n 5 | sed 's/^/  /'
    } >&2
    errors=$((errors + 1))
  fi

  if [ "$errors" -gt 0 ]; then
    echo "changelog: $errors problem(s) found" >&2
    exit 1
  fi
  echo "changelog: ${#FRAGMENTS[@]} fragment(s) OK"
}

# The `### Category` blocks for every fragment, in category order.
render_sections() {
  local cat name first=1 matched
  for cat in "${CATEGORIES[@]}"; do
    matched=()
    for name in ${FRAGMENTS[@]+"${FRAGMENTS[@]}"}; do
      case "$name" in *."$cat".md) matched+=("$name") ;; esac
    done
    [ ${#matched[@]} -gt 0 ] || continue
    [ "$first" -eq 1 ] || echo
    first=0
    echo "### $(heading_of "$cat")"
    for name in "${matched[@]}"; do
      fragment_body "$frag_dir/$name"
    done
  done
}

cmd_preview() {
  cmd_check >/dev/null # also loads FRAGMENTS
  echo "## [Unreleased]"
  echo
  render_sections
}

cmd_assemble() {
  local version="" date=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --date)
        [ $# -ge 2 ] || die "--date needs a value"
        date="$2"
        shift 2
        ;;
      -*) die "assemble: unknown flag '$1'" ;;
      *)
        [ -z "$version" ] || die "assemble: more than one version given"
        version="$1"
        shift
        ;;
    esac
  done
  [ -n "$version" ] || die "assemble: usage: tools/changelog.sh assemble <version> [--date YYYY-MM-DD]"
  version="${version#v}"
  # grep matches per LINE, so `$'junk\n1.2.3'` would pass the regexes below and
  # write a split heading. Refuse any line break first.
  case "$version$date" in
    *$'\n'* | *$'\r'*) die "assemble: the version and date must be single-line" ;;
  esac
  # SemVer core plus an optional pre-release, no leading zeros — in a numeric
  # pre-release identifier too (`rc.01`), per SemVer §9. Build metadata (`+...`)
  # is deliberately not accepted: it is not part of a release tag here.
  local num='(0|[1-9][0-9]*)' ident='(0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)'
  printf '%s' "$version" |
    grep -Eq "^$num\\.$num\\.$num(-$ident(\\.$ident)*)?\$" ||
    die "assemble: '$version' is not a release version (X.Y.Z or X.Y.Z-pre, no leading zeros)"
  [ -n "$date" ] || date="$(date -u +%Y-%m-%d)"
  # Shape, then the calendar: `2026-02-31` has the right shape and is no date.
  printf '%s' "$date" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' &&
    printf '%s\n' "$date" | awk -F- '{
      y = $1 + 0; m = $2 + 0; d = $3 + 0
      split("31 28 31 30 31 30 31 31 30 31 30 31", days, " ")
      if (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)) days[2] = 29
      exit !(m >= 1 && m <= 12 && d >= 1 && d <= days[m])
    }' ||
    die "assemble: --date '$date' is not a calendar date (YYYY-MM-DD)"

  cmd_check >/dev/null # also loads FRAGMENTS: the one manifest used below
  [ ${#FRAGMENTS[@]} -gt 0 ] || die "assemble: no fragments in changelog.d/ — nothing to release"
  if grep -Fq "## [$version]" "$changelog"; then
    die "assemble: CHANGELOG.md already has a ## [$version] section"
  fi

  # `[Unreleased]: <base>/compare/<prev>...HEAD` gives both the compare base URL
  # and the previous tag.
  local link base prev
  link="$(grep -E '^\[Unreleased\]: .*/compare/[^/]+\.\.\.HEAD$' "$changelog" || true)"
  [ -n "$link" ] || die "assemble: no '[Unreleased]: <url>/compare/<tag>...HEAD' link in CHANGELOG.md"
  [ "$(printf '%s\n' "$link" | grep -c .)" = "1" ] ||
    die "assemble: more than one '[Unreleased]:' compare link in CHANGELOG.md"
  base="$(printf '%s' "$link" | sed -E 's#^\[Unreleased\]: (.*)/compare/[^/]+\.\.\.HEAD$#\1#')"
  prev="$(printf '%s' "$link" | sed -E 's#^.*/compare/([^/]+)\.\.\.HEAD$#\1#')"

  # The rewrite goes to a copy beside CHANGELOG.md (`cp -p`, so the `mv` keeps
  # its mode — a `mktemp` file is 0600) and replaces it only once verified.
  # Globals, not locals: the EXIT trap runs after this function has returned.
  section="$(mktemp)"
  tmp="$changelog.assemble.$$"
  cp -p "$changelog" "$tmp"
  trap 'rm -f "$section" "$tmp"' EXIT
  {
    echo "## [$version] - $date"
    echo
    render_sections
  } >"$section"

  # Swallow the (empty) [Unreleased] body, then write the new release section
  # right before the previous release's heading.
  awk -v section="$section" -v version="$version" -v base="$base" -v prev="$prev" '
    function emit_section(   line) {
      while ((getline line < section) > 0) print line
      close(section)
      print ""
    }
    /^## \[Unreleased\]/ { print; print ""; inside = 1; next }
    inside && /^## \[/   { emit_section(); inside = 0; done = 1 }
    inside               { next }
    /^\[Unreleased\]: /  {
      print "[Unreleased]: " base "/compare/v" version "...HEAD"
      print "[" version "]: " base "/compare/" prev "...v" version
      next
    }
    { print }
    END {
      # No earlier release heading: the new section closes the file body.
      if (inside) emit_section()
    }
  ' "$changelog" >"$tmp"

  # Never delete a fragment that did not make it into the file: the new heading
  # and every fragment's first line must be present before anything is removed.
  grep -Fqx "## [$version] - $date" "$tmp" ||
    die "assemble: internal error: the new section was not written; nothing changed"
  local name
  for name in ${FRAGMENTS[@]+"${FRAGMENTS[@]}"}; do
    grep -Fqx -- "$(sed -n 1p "$frag_dir/$name")" "$tmp" ||
      die "assemble: internal error: changelog.d/$name was not written; nothing changed"
  done
  mv "$tmp" "$changelog"

  local n=0
  for name in ${FRAGMENTS[@]+"${FRAGMENTS[@]}"}; do
    rm -f "$frag_dir/$name"
    n=$((n + 1))
  done
  echo "changelog: assembled $n fragment(s) into ## [$version] - $date and removed them."
  echo "Review CHANGELOG.md — add any release-level prose (e.g. upgrade notes) under the"
  echo "new heading by hand — then commit it together with the deleted fragments."
}

# Test-only files under a user-facing prefix: `*_tests.rs` siblings, `tests/`
# directories, and the benches.
is_test_path() {
  case "$1" in
    *_tests.rs | */tests/* | */benches/*) return 0 ;;
  esac
  return 1
}

# Each newline-separated path of $1 that is user-facing, indented. Same
# newline split as `load_fragments`, for the same here-document reason; a
# function of its own so the IFS change cannot leak into the caller.
user_facing_of() {
  local IFS=$'\n' f
  set -f
  for f in $1; do
    if is_user_facing "$f"; then
      printf '  %s\n' "$f"
    fi
  done
  set +f
}

# $1 as `git diff --name-only` prints it under `core.quotePath=false`: plain
# for UTF-8, but a path with a quote, backslash or control character comes out
# C-quoted (`"src/a\"b.rs"`). Dropping a leading quote is enough for the prefix
# test, and errs toward "user-facing", which is the safe direction.
is_user_facing() {
  local p path="${1#\"}"
  is_test_path "$path" && return 1
  for p in "${USER_FACING_PREFIXES[@]}"; do
    case "$path" in "$p"*) return 0 ;; esac
  done
  return 1
}

# 0 if fragment $2, as of commit $1, appears as ONE contiguous run — same lines,
# same order, same multiplicity — among the lines this PR ADDED to CHANGELOG.md
# ($3): i.e. `assemble` wrote it, continuation and all. Not a first-line match
# anywhere in the file (a dropped continuation, or text that already existed,
# would pass) and not set membership (reordered or deduplicated lines would).
#
# Blank lines are dropped from both sides first: git may align a blank line
# inside the inserted section with a pre-existing one as context, which splits
# the `+` run there without the text having changed.
fragment_assembled() {
  local body needle haystack
  body="$(git -C "$root" show "$1:$2")" || return 1
  needle="$(printf '%s\n' "$body" | grep -v '^[[:space:]]*$' || true)"
  haystack="$(printf '%s\n' "$3" | grep -v '^[[:space:]]*$' || true)"
  [ -n "$needle" ] || return 1
  # Newline-delimited on both ends, so a match is whole lines; quoted, so the
  # needle is literal text, not a glob.
  case $'\n'"$haystack"$'\n' in
    *$'\n'"$needle"$'\n'*) return 0 ;;
  esac
  return 1
}

cmd_require() {
  local base="" pr="" opt_out="false"
  while [ $# -gt 0 ]; do
    case "$1" in
      --pr)
        [ $# -ge 2 ] || die "--pr needs a value"
        pr="$2"
        shift 2
        ;;
      --opt-out)
        [ $# -ge 2 ] || die "--opt-out needs true or false"
        opt_out="$2"
        shift 2
        ;;
      -*) die "require: unknown flag '$1'" ;;
      *)
        [ -z "$base" ] || die "require: more than one base ref given"
        base="$1"
        shift
        ;;
    esac
  done
  [ -n "$base" ] || die "require: usage: tools/changelog.sh require <base-ref> [--pr N] [--opt-out true|false]"

  # Pending fragments are other PRs' entries: deleting one — or renaming it
  # away, which `--no-renames` reports as a deletion — drops it from the next
  # release. The one legitimate deletion is `assemble`'s, so a deleted fragment
  # passes only if this PR added all of it to CHANGELOG.md. Checked before the
  # opt-out: the label waives this PR's own entry, not anyone else's.
  local mb deleted lost="" added_lines renamed
  mb="$(git -C "$root" merge-base "$base" HEAD)" ||
    die "require: no merge base between $base and HEAD (CI needs fetch-depth: 0)"

  # A rename is never a release step: `assemble` deletes, it does not move. And
  # copying the old body into CHANGELOG.md must not excuse one — the renamed
  # file would be released a second time.
  renamed="$(git -C "$root" diff -M --name-status --diff-filter=R "$mb" HEAD -- changelog.d/ || true)"
  if [ -n "$renamed" ]; then
    {
      echo "changelog: this PR renames pending changelog fragment(s):"
      printf '%s\n' "$renamed" | sed 's/^/  /'
      echo
      echo "Leave other PRs' fragments where they are (editing one to reword it is fine)."
    } >&2
    exit 1
  fi
  deleted="$(git -C "$root" diff --no-renames --name-only --diff-filter=D "$mb" HEAD -- changelog.d/ |
    grep -Ev '^changelog\.d/(README\.md|\.gitkeep)$' || true)"
  if [ -n "$deleted" ]; then
    added_lines="$(git -C "$root" diff --no-renames "$mb" HEAD -- CHANGELOG.md |
      sed -n -e '/^+++ /d' -e 's/^+//p')"
    local IFS_SAVE="$IFS" f
    IFS=$'\n'
    set -f
    for f in $deleted; do
      if ! fragment_assembled "$mb" "$f" "$added_lines"; then
        lost="$lost  $f"$'\n'
      fi
    done
    set +f
    IFS="$IFS_SAVE"
  fi
  if [ -n "$lost" ]; then
    {
      echo "changelog: this PR deletes pending changelog fragment(s) that are not in"
      echo "CHANGELOG.md, so their entries would be missing from the next release:"
      printf '%s' "$lost"
      echo
      echo "Restore them (edit one to reword it). Only 'tools/changelog.sh assemble'"
      echo "removes fragments, at release time."
    } >&2
    exit 1
  fi

  # A release PR (it assembled fragments) must leave none behind. In CI HEAD is
  # the merge with `main`, so this also catches a fragment that landed on `main`
  # after the release branch was assembled — its entry would miss the release.
  load_fragments
  if [ -n "$deleted" ] && [ ${#FRAGMENTS[@]} -gt 0 ]; then
    {
      echo "changelog: this release PR assembled fragments but leaves pending ones:"
      printf '  changelog.d/%s\n' "${FRAGMENTS[@]}"
      echo
      echo "Merge main into the release branch and run 'tools/changelog.sh assemble'"
      echo "again, so every entry is in the release section."
    } >&2
    exit 1
  fi

  # A boolean the workflow computes with `contains(labels.*.name, ...)`, not a
  # joined label list: joining loses the boundaries, so a single label named
  # `x,no-changelog` would read as the opt-out.
  case "$opt_out" in
    true)
      echo "changelog: '$OPT_OUT_LABEL' label set — no fragment required."
      return 0
      ;;
    false) ;;
    *) die "require: --opt-out must be true or false, got '$opt_out'" ;;
  esac

  local changed facing="" added=""
  # `--no-renames`: a rename reports only its destination, so
  # `src/x.rs -> src/x_tests.rs` would hide the production file it removed.
  changed="$(git -C "$root" -c core.quotePath=false diff --no-renames --name-only "$base"...HEAD)" ||
    die "require: git diff $base...HEAD failed (is the base fetched? CI needs fetch-depth: 0)"

  facing="$(user_facing_of "$changed")"

  # Only an ADDED fragment counts: rewording one of the pending fragments from
  # another PR does not describe this one. The opposite rename setting from the
  # diff above, and deliberately: with `-M` a renamed fragment is an `R`, not an
  # `A`, so renaming someone else's entry cannot pass for adding one.
  # Directly under changelog.d/ only: `changelog.d/.gitkeep/x.md` is no fragment.
  added="$(git -C "$root" diff -M --name-only --diff-filter=A "$base"...HEAD -- changelog.d/ |
    grep -E '^changelog\.d/[^/]+$' | grep -Ev '^changelog\.d/(README\.md|\.gitkeep)$' || true)"

  if [ -z "$facing" ]; then
    echo "changelog: no user-facing paths changed — no fragment required."
    return 0
  fi
  if [ -n "$added" ]; then
    echo "changelog: fragment(s) present:"
    printf '%s\n' "$added" | sed 's/^/  /'
    return 0
  fi

  local want="changelog.d/${pr:-<N>}.<category>.md"
  {
    echo "changelog: this PR changes user-facing code but adds no changelog fragment."
    echo
    echo "Missing: $want"
    echo "  <category> is one of: ${CATEGORIES[*]}"
    echo "  e.g.  changelog.d/${pr:-<N>}.fixed.md  containing one bullet:"
    echo "        - **What changed, in user-facing words** (#${pr:-<N>}). ..."
    echo
    echo "If the change is not user-facing (refactor, tests, CI), apply the"
    echo "'$OPT_OUT_LABEL' label to the PR instead."
    echo
    echo "User-facing paths changed:"
    printf '%s\n' "$facing"
  } >&2
  exit 1
}

# The tag-time backstop for `require`'s release check: a merge to `main` after
# the release PR's last CI run (a base-only advance re-runs nothing) could still
# leave a fragment pending at tag time.
cmd_released() {
  [ $# -eq 1 ] || die "released: usage: tools/changelog.sh released <tag>"
  local version="${1#refs/tags/}"
  version="${version#v}"
  cmd_check >/dev/null # also loads FRAGMENTS
  grep -Eq "^## \[$(printf '%s' "$version" | sed 's/[.]/\\./g')\] - " "$changelog" ||
    die "released: CHANGELOG.md has no '## [$version] - <date>' section — run 'tools/changelog.sh assemble $version' before tagging"
  if [ ${#FRAGMENTS[@]} -gt 0 ]; then
    {
      echo "changelog: tag $1 has pending fragments, so their entries are not in"
      echo "the [$version] section:"
      printf '  changelog.d/%s\n' "${FRAGMENTS[@]}"
    } >&2
    exit 1
  fi
  echo "changelog: [$version] is assembled and nothing is pending."
}

[ $# -ge 1 ] || {
  usage >&2
  exit 2
}
sub="$1"
shift
case "$sub" in
  check) cmd_check "$@" ;;
  preview) cmd_preview "$@" ;;
  assemble) cmd_assemble "$@" ;;
  require) cmd_require "$@" ;;
  released) cmd_released "$@" ;;
  -h | --help | help) usage ;;
  *)
    echo "changelog: unknown subcommand '$sub' (try --help)" >&2
    exit 2
    ;;
esac
