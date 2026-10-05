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
  tools/changelog.sh require <base-ref> [--pr N] [--labels a,b,...]
                                               fail if the diff base...HEAD touches
                                               user-facing code and adds no fragment,
                                               unless the '$OPT_OUT_LABEL' label is set

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
  local f name
  for f in "$frag_dir"/* "$frag_dir"/.[!.]*; do
    [ -e "$f" ] || continue
    name="${f##*/}"
    is_non_fragment "$name" && continue
    printf '%s\n' "$name"
  done | LC_ALL=C sort -t. -k1,1n -k1,1
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
  local errors=0 name stem cat body first bad
  [ -f "$changelog" ] || die "no CHANGELOG.md at $changelog"

  while IFS= read -r name; do
    [ -n "$name" ] || continue
    local path="$frag_dir/$name"
    if [ -d "$path" ]; then
      echo "changelog.d/$name: a directory; fragments are files directly under changelog.d/" >&2
      errors=$((errors + 1))
      continue
    fi
    # <N>[-<suffix>].<category>.md
    if ! printf '%s' "$name" | grep -Eq '^[0-9]+(-[0-9a-z]+)*\.[a-z]+\.md$'; then
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
    # Every later non-blank line must be indented: a continuation of the one
    # bullet. A second column-0 line is a second bullet, a heading, or prose
    # that would land outside the list.
    bad="$(printf '%s\n' "$body" | awk 'NR > 1 && NF && !/^[ \t]/ { print NR ": " $0; exit }')"
    if [ -n "$bad" ]; then
      echo "changelog.d/$name: must be a single bullet; line $bad is not indented" \
        "(continuation lines need leading spaces; one entry per file)" >&2
      errors=$((errors + 1))
      continue
    fi
    if printf '%s' "$body" | grep -q $'\r'; then
      echo "changelog.d/$name: contains CR line endings; save it with LF" >&2
      errors=$((errors + 1))
      continue
    fi
  done <<EOF
$(fragment_names)
EOF

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
  echo "changelog: $(fragment_names | grep -c . || true) fragment(s) OK"
}

# The `### Category` blocks for every fragment, in category order.
render_sections() {
  local cat names name first=1
  for cat in "${CATEGORIES[@]}"; do
    names="$(fragment_names | grep -E "\.${cat}\.md$" || true)"
    [ -n "$names" ] || continue
    [ "$first" -eq 1 ] || echo
    first=0
    echo "### $(heading_of "$cat")"
    while IFS= read -r name; do
      fragment_body "$frag_dir/$name"
    done <<EOF
$names
EOF
  done
}

cmd_preview() {
  cmd_check >/dev/null
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
  printf '%s' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$' ||
    die "assemble: '$version' is not a semantic version (X.Y.Z)"
  [ -n "$date" ] || date="$(date -u +%Y-%m-%d)"
  printf '%s' "$date" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' ||
    die "assemble: --date '$date' is not YYYY-MM-DD"

  cmd_check >/dev/null
  [ -n "$(fragment_names)" ] || die "assemble: no fragments in changelog.d/ — nothing to release"
  if grep -Fq "## [$version]" "$changelog"; then
    die "assemble: CHANGELOG.md already has a ## [$version] section"
  fi

  # `[Unreleased]: <base>/compare/<prev>...HEAD` gives both the compare base URL
  # and the previous tag.
  local link base prev
  link="$(grep -E '^\[Unreleased\]: .*/compare/[^/]+\.\.\.HEAD$' "$changelog" || true)"
  [ -n "$link" ] || die "assemble: no '[Unreleased]: <url>/compare/<tag>...HEAD' link in CHANGELOG.md"
  base="$(printf '%s' "$link" | sed -E 's#^\[Unreleased\]: (.*)/compare/[^/]+\.\.\.HEAD$#\1#')"
  prev="$(printf '%s' "$link" | sed -E 's#^.*/compare/([^/]+)\.\.\.HEAD$#\1#')"

  local section tmp
  section="$(mktemp)"
  tmp="$(mktemp)"
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
  mv "$tmp" "$changelog"
  rm -f "$section"

  local name n=0
  while IFS= read -r name; do
    [ -n "$name" ] || continue
    rm -f "$frag_dir/$name"
    n=$((n + 1))
  done <<EOF
$(fragment_names)
EOF
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

is_user_facing() {
  local p
  is_test_path "$1" && return 1
  for p in "${USER_FACING_PREFIXES[@]}"; do
    case "$1" in "$p"*) return 0 ;; esac
  done
  return 1
}

cmd_require() {
  local base="" pr="" labels=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --pr)
        [ $# -ge 2 ] || die "--pr needs a value"
        pr="$2"
        shift 2
        ;;
      --labels)
        [ $# -ge 2 ] || die "--labels needs a value"
        labels="$2"
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
  [ -n "$base" ] || die "require: usage: tools/changelog.sh require <base-ref> [--pr N] [--labels a,b]"

  case ",$labels," in
    *",$OPT_OUT_LABEL,"*)
      echo "changelog: '$OPT_OUT_LABEL' label set — no fragment required."
      return 0
      ;;
  esac

  local changed facing="" added=""
  changed="$(git -C "$root" diff --name-only "$base"...HEAD)" ||
    die "require: git diff $base...HEAD failed (is the base fetched? CI needs fetch-depth: 0)"

  local f
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if is_user_facing "$f"; then
      facing="$facing  $f"$'\n'
    fi
  done <<EOF
$changed
EOF

  # A fragment counts if it is ADDED or MODIFIED by this PR (a deleted one does
  # not describe it).
  added="$(git -C "$root" diff --name-only --diff-filter=AMR "$base"...HEAD -- changelog.d/ |
    grep -Ev '^changelog\.d/(README\.md|\.gitkeep)$' || true)"

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
    printf '%s' "$facing"
  } >&2
  exit 1
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
  -h | --help | help) usage ;;
  *)
    echo "changelog: unknown subcommand '$sub' (try --help)" >&2
    exit 2
    ;;
esac
