#!/usr/bin/env python3
"""The part of ferx-model's public API that is ferx-core's public API (#3041).

`cargo public-api` lists an item ferx-core re-exports from ferx-model only as
`pub use ferx_core::<path>`: its fields, variants, methods and impls are not in
ferx-core's listing. This filter keeps, from `cargo public-api -p ferx-model`, exactly
the items ferx-core's facade re-exports publicly (read from its `pub use ferx_model::…`
lines) and everything that belongs to them, so the second baseline gates their shape
the way the first gates their names.

usage: public-api-model-filter.py REPO_ROOT FERX_CORE_LISTING FERX_MODEL_LISTING
Prints the filtered ferx-model listing; exits 1 if a `pub use ferx_core::…` line of
the ferx-core listing has no facade source (an unaccounted re-export)."""
import collections
import os
import re
import sys

root, core_listing, model_listing = sys.argv[1:4]

# ferx-model's public modules, from its own listing.
model_mods = {m.group(1) for m in (re.match(r'^pub mod ferx_model::([\w:]+)$', l.strip())
                                   for l in open(model_listing)) if m}

# 1. Every ferx_model item ferx-core re-exports publicly: `pub use ferx_model::…` in the facade,
#    and `pub use crate::<m>::…` where `<m>` is a module ferx-core takes from ferx-model (the
#    facade mirrors ferx-model's module paths, so `crate::<m>::x` is `ferx_model::<m>::x`).
sources = set()
aliases = set()  # `x as y`: the public name is y
for d, _, fs in os.walk(os.path.join(root, 'src')):
    for f in fs:
        if not f.endswith('.rs'):
            continue
        text = open(os.path.join(d, f)).read()
        for m in re.finditer(r'^\s*pub use (ferx_model|crate)::((?:\w+::)*)(\{[^}]*\}|\w+(?: as \w+)?);',
                             text, re.M | re.S):
            origin, prefix, tail = m.group(1), m.group(2), m.group(3)
            if origin == 'crate' and prefix.rstrip(':') not in model_mods:
                continue
            for item in (re.split(r',', tail.strip('{}')) if tail.startswith('{') else [tail]):
                item = item.strip()
                if not item or item == 'self' or '::' in item:
                    continue
                src = item.split(' as ')[0].strip()
                sources.add('ferx_model::' + prefix + src)
                if ' as ' in item:
                    aliases.add(item.split(' as ')[1].strip())

# 2. Each `pub use ferx_core::X` of ferx-core's listing must have a facade source named X.
names = {s.rsplit('::', 1)[1] for s in sources} | aliases
missing = []
for line in open(core_listing):
    m = re.match(r'^pub use ferx_core::(?:\w+::)*(\w+)$', line.strip())
    if m and m.group(1) not in names:
        missing.append(line.strip())

# 3. ferx-model lines about those items: the item, its members, its impls.
SUBJECT = [
    re.compile(r'^(?:#\[[^\]]*\]\s*)*pub (?:[a-z ]+ )?(ferx_model::\w+(?:::\w+)*)'),  # items and members
    re.compile(r'^(?:#\[[^\]]*\]\s*)*pub (ferx_model::\w+(?:::\w+)*)'),               # fields, variants
    re.compile(r'^(?:unsafe )?impl(?:<[^>]*(?:<[^>]*>[^>]*)*>)? (?:.* for )?(ferx_model::\w+(?:::\w+)*)'),
]


def subject(line):
    for p in SUBJECT:
        m = p.match(line)
        if m:
            return m.group(1)
    return None


# Members of ferx-model's trait impls on primitives (`impl PkNum for f64` lists as
# `pub fn f64::exp(…)`): the traits are ferx-core API (`sens::num`), and the listing is sorted,
# not grouped by impl, so these are kept by their primitive subject.
PRIMITIVE_MEMBER = re.compile(r'^pub (?:[a-z ]+ )?(?:f64|f32|usize|u64|u32|i64|i32|bool)::')

# An item ferx-model itself re-exports (`pub use activation::Activation` in `nn`) is listed at
# the re-export path, but its impls and methods at its definition path: add the definition
# path of every re-exported item by name, from the item lines of the listing.
lines = open(model_listing).read().split('\n')
defined = collections.defaultdict(set)
for l in lines:
    m = re.match(r'^(?:#\[[^\]]*\]\s*)*pub (?:struct|enum|trait|union|type) (ferx_model::\w+(?:::\w+)*)', l)
    if m:
        defined[m.group(1).rsplit('::', 1)[1]].add(m.group(1))
    m = re.match(r'^(?:unsafe )?impl(?:<[^>]*(?:<[^>]*>[^>]*)*>)? (?:.* for )?(ferx_model::\w+(?:::\w+)*)', l)
    if m:
        defined[m.group(1).rsplit('::', 1)[1]].add(m.group(1))
for src in list(sources):
    sources |= defined.get(src.rsplit('::', 1)[1], set())

ordered = sorted(sources, key=len, reverse=True)
for line in lines:
    line = line + '\n' if line else ''
    s = subject(line.rstrip('\n'))
    if (s and any(s == src or s.startswith(src + '::') for src in ordered)) or PRIMITIVE_MEMBER.match(line):
        sys.stdout.write(line)
if missing:
    sys.stderr.write('re-exports with no `pub use ferx_model::…` facade source:\n  '
                     + '\n  '.join(missing) + '\n')
    sys.exit(1)
