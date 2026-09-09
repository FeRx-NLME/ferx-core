#!/usr/bin/env bash
#
# Render the Quarto site into docs/_site, from any checkout.
#
#   tools/render-docs.sh          # render docs/ -> docs/_site/
#
# Use this instead of a bare `quarto render docs` (#1190).
#
# Quarto's project input discovery skips every path with a **hidden** (dot-
# prefixed) directory component. A worktree under `.claude/worktrees/<name>/`
# — which CLAUDE.md asks for on any branch that is not `main` — is exactly
# that, so `quarto render docs` there finds *zero* inputs: it writes
# `robots.txt` and `sitemap.xml`, renders not one page, prints no warning and
# exits **0**. Nothing tells you the render did not happen; you notice when a
# check against `docs/_site` reads a stale site, or none.
#
# The confirmed cause is the dot, not `.gitignore`: the same tree renders as
# soon as the ancestor directory is renamed to something not starting with `.`.
# So when the checkout sits under a dot directory this stages `docs/` outside
# it, renders there, and copies `_site` back.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docs="$root/docs"
[ -d "$docs" ] || { echo "render-docs: no docs/ at $docs" >&2; exit 1; }

# A dot-prefixed component anywhere above docs/ is what Quarto refuses to walk.
if [[ "$docs" == *"/."* ]]; then
    stage="$(mktemp -d)"
    trap 'rm -rf "$stage"' EXIT
    echo "render-docs: $root is under a hidden directory — staging in $stage"
    cp -R "$docs" "$stage/docs"
    rm -rf "$stage/docs/_site" "$stage/docs/.quarto"
    quarto render "$stage/docs"
    rm -rf "$docs/_site"
    cp -R "$stage/docs/_site" "$docs/_site"
else
    quarto render "$docs"
fi

pages="$(find "$docs/_site" -name '*.html' | wc -l | tr -d ' ')"
[ "$pages" -gt 0 ] || { echo "render-docs: rendered 0 pages — nothing was discovered" >&2; exit 1; }
echo "render-docs: $pages page(s) in docs/_site"
