#!/usr/bin/env bash
#
# One-call status of the current checkout against its base. `--help` prints
# usage; this header is the why.
#
# An agent session isolated in a worktree (Claude Code's `EnterWorktree`) runs
# behind a fence that refuses any shell command it cannot prove keeps git inside
# that worktree: `git -C`, `cd … && git`, `$(git …)`, pipes and `;`/`&&` chains
# around git, heredocs, loops. Measured 2026-10-07 over 205 worktree sessions:
# 1,808 commands refused, a median of 7 per session, each one a wasted round
# trip. A plain single command, a `bash FILE` and a `tools/*.sh` call were refused
# zero times.
#
# The compounds that kept being refused were the same few questions asked by
# hand: how far behind the base am I, does it conflict, what is uncommitted, is
# the branch pushed. This script asks them all from the inside, as one plain
# command. It only ever runs git on the checkout it is started in, never `-C`
# elsewhere, so it does not widen what the fence is there to stop.
#
# Read-only apart from `--fetch` (which updates remote-tracking refs, exactly
# like a bare `git fetch`). `merge-tree --write-tree` writes loose objects but
# touches no ref, index or file. Needs git >= 2.38.
#
# Plain bash on purpose: it runs unchanged under macOS's bash 3.2 and CI's bash 5.
# Tested by `tests/wt_status_script.rs` against scratch repositories.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: tools/wt-status.sh [--fetch] [--base <ref>]

Prints, one `key: value` per line, for the checkout it is run in:
  branch, head        current branch (or "(detached)") and HEAD sha
  base, merge-base    the base ref's sha (default origin/main) and the fork point
  ahead, behind       commits on HEAD not on base, and on base not on HEAD
  upstream            "<ref> <sha> (in sync|ahead N|behind N|diverged N/M)" or "none"
  conflicts           "none", or "yes" followed by "  <path>" lines: what a merge
                      of base into HEAD would conflict on
  changed             files changed on HEAD since the merge-base (committed only)
  dirty               "clean", or a count followed by `git status --porcelain` lines

  --fetch       `git fetch origin` first (quietly)
  --base <ref>  compare against <ref> instead of origin/main

Exit status: 0 when the report was produced, 2 on usage error or when a ref is
missing. A conflict is reported, not an error.
EOF
}

fetch=0
base_ref=origin/main
while [ $# -gt 0 ]; do
    case "$1" in
    --fetch) fetch=1 ;;
    --base)
        [ $# -ge 2 ] || { echo "wt-status: --base needs a ref" >&2; exit 2; }
        base_ref="$2"
        shift
        ;;
    -h | --help) usage; exit 0 ;;
    *) echo "wt-status: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

git rev-parse --git-dir >/dev/null 2>&1 || { echo "wt-status: not inside a git checkout" >&2; exit 2; }

if [ "$fetch" = 1 ]; then
    git fetch --quiet origin
fi

base=$(git rev-parse --verify --quiet "$base_ref^{commit}") ||
    { echo "wt-status: base ref '$base_ref' not found (try --fetch)" >&2; exit 2; }
head=$(git rev-parse HEAD)
branch=$(git symbolic-ref --quiet --short HEAD || echo "(detached)")
mb=$(git merge-base HEAD "$base")
read -r ahead behind < <(git rev-list --left-right --count "HEAD...$base")

echo "branch: $branch"
echo "head: $head"
echo "base: $base_ref $base"
echo "merge-base: $mb"
echo "ahead: $ahead"
echo "behind: $behind"

if up=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null); then
    up_sha=$(git rev-parse "$up")
    read -r u_ahead u_behind < <(git rev-list --left-right --count "HEAD...$up")
    if [ "$u_ahead" = 0 ] && [ "$u_behind" = 0 ]; then
        state="in sync"
    elif [ "$u_behind" = 0 ]; then
        state="ahead $u_ahead"
    elif [ "$u_ahead" = 0 ]; then
        state="behind $u_behind"
    else
        state="diverged $u_ahead/$u_behind"
    fi
    echo "upstream: $up $up_sha ($state)"
else
    echo "upstream: none"
fi

# Exit 1 means "conflicts", and the first output line is then the tree id;
# anything above 1 is a real failure, which `set -e` would otherwise hide here.
set +e
mt=$(git merge-tree --write-tree --name-only --no-messages HEAD "$base" 2>&1)
mt_status=$?
set -e
case "$mt_status" in
0) echo "conflicts: none" ;;
1)
    echo "conflicts: yes"
    printf '%s\n' "$mt" | sed '1d; /^$/d; s/^/  /'
    ;;
*)
    echo "wt-status: git merge-tree failed ($mt_status): $mt" >&2
    exit 2
    ;;
esac

echo "changed: $(git diff --name-only "$mb" HEAD | wc -l | tr -d ' ')"

porcelain=$(git status --porcelain)
if [ -z "$porcelain" ]; then
    echo "dirty: clean"
else
    echo "dirty: $(printf '%s\n' "$porcelain" | wc -l | tr -d ' ')"
    printf '%s\n' "$porcelain" | sed 's/^/  /'
fi
