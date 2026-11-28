#!/usr/bin/env bash
#
# License gate (#3183), run by `tools/preflight.sh licenses`.
#
# ferx-search is LGPL-3.0-or-later; ferx-core, ferx-tools and ferx-cli are MIT,
# and the `ferx` binary links ferx-search, so it carries the LGPL texts and the
# third-party notices (`ferx --licenses`). Each crate tarball has to contain its
# own license texts: ferx-tools 0.4.0 and ferx-cli 0.4.0 shipped with none. This
# checks, and fails on the first miss:
#
#   1. each crate's `license` field;
#   2. an SPDX header on the first line of every ferx-search `.rs` file;
#   3. the license files in `cargo package --list` of each crate;
#   4. the copies of a license text that live in two crates are identical (a
#      crate cannot include a file outside its own directory);
#   5. crates/ferx-cli/THIRD-PARTY-NOTICES.txt matches Cargo.lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail() {
  echo "check-licenses: $*" >&2
  exit 1
}

# 1. License fields.
metadata="$(cargo metadata --format-version 1 --no-deps --locked)"
for expect in ferx-core=MIT ferx-tools=MIT ferx-cli=MIT ferx-search=LGPL-3.0-or-later; do
  name="${expect%%=*}"
  want="${expect#*=}"
  got="$(printf '%s' "$metadata" | python3 -c '
import json, sys
name = sys.argv[1]
print(next(p.get("license") for p in json.load(sys.stdin)["packages"] if p["name"] == name))
' "$name")"
  [ "$got" = "$want" ] || fail "$name has license = \"$got\", expected \"$want\""
done

# 2. SPDX headers.
spdx='// SPDX-License-Identifier: LGPL-3.0-or-later'
while IFS= read -r f; do
  [ "$(head -n 1 "$f")" = "$spdx" ] || fail "$f does not start with \`$spdx\`"
done < <(git ls-files 'crates/ferx-search/*.rs')

# 3. License files in each packaged crate.
need_in_package() {
  local pkg="$1"
  shift
  local list
  list="$(cargo package --list --allow-dirty --locked -p "$pkg")"
  for f in "$@"; do
    grep -qx "$f" <<<"$list" || fail "the $pkg package does not contain $f"
  done
}
need_in_package ferx-core LICENSE
need_in_package ferx-tools LICENSE
need_in_package ferx-search COPYING COPYING.LESSER LICENSE-MIT NOTICE
need_in_package ferx-cli LICENSE THIRD-PARTY-NOTICES.txt \
  licenses/GPL-3.0.txt licenses/LGPL-3.0.txt licenses/LGPL-2.1.txt

# 4. Copies agree. The member MIT files are the root LICENSE's MIT grant, which
# is everything above its first `-----` rule.
mit="$(sed '/^-----/,$d' LICENSE)"
for f in crates/ferx-tools/LICENSE crates/ferx-cli/LICENSE crates/ferx-search/LICENSE-MIT; do
  [ "$(cat "$f")" = "$mit" ] || fail "$f differs from the MIT grant at the top of LICENSE"
done
cmp -s crates/ferx-search/COPYING crates/ferx-cli/licenses/GPL-3.0.txt \
  || fail "crates/ferx-cli/licenses/GPL-3.0.txt differs from crates/ferx-search/COPYING"
cmp -s crates/ferx-search/COPYING.LESSER crates/ferx-cli/licenses/LGPL-3.0.txt \
  || fail "crates/ferx-cli/licenses/LGPL-3.0.txt differs from crates/ferx-search/COPYING.LESSER"
pharmpy='Parts of this crate follow the behaviour of Pharmpy'
for f in crates/ferx-search/NOTICE crates/ferx-cli/src/licenses.rs; do
  grep -q "$pharmpy" "$f" || fail "$f lacks the Pharmpy attribution line"
done

# 5. Third-party notices.
python3 tools/third-party-notices.py --check

echo "check-licenses: OK"
