#!/usr/bin/env bash
# Fails when `ferx-cli` built without its `search` feature still links `ferx-search`
# (#3187). Run by `tools/preflight.sh no-search`.
#
# A green `cargo test` of the no-search build cannot see this: an unconditional
# `ferx-search` dependency, or a member feature that re-enables it, compiles and passes
# the same tests while the binary still carries the whole search layer.
#
# `--edges normal,build`: what the binary links. A dev-dependency on `ferx-search` would
# not reach the binary and is not this check's business. The feature set is the one the
# no-search CI lines build, so this asserts the graph those lines actually compile.
set -euo pipefail

tree=$(cargo tree -p ferx-cli --no-default-features --features ferx-core/ci \
  --edges normal,build --prefix none)

if grep -q '^ferx-search ' <<<"$tree"; then
  echo "error: \`ferx-cli\` without its \`search\` feature still depends on \`ferx-search\`:" >&2
  cargo tree -p ferx-cli --no-default-features --features ferx-core/ci \
    --edges normal,build --invert ferx-search >&2 || true
  exit 1
fi
echo "ok: ferx-cli without \`search\` does not depend on ferx-search"
