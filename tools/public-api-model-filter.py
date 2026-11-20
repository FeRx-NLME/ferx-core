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
the ferx-core listing has no facade source (an unaccounted re-export), or if a facade
`pub use ferx_model::…` line has a shape this filter cannot read."""
import collections
import os
import re
import sys

# The listings and sources are UTF-8 (`→`, `σ` in doc text); the platform default encoding
# (CP1252 on Windows) cannot read them, and the baseline is redirected from stdout.
sys.stdout.reconfigure(encoding='utf-8', newline='\n')
sys.stderr.reconfigure(encoding='utf-8')

root, core_listing, model_listing = sys.argv[1:4]

# ferx-model's public modules, from its own listing.
model_mods = {m.group(1) for m in (re.match(r'^pub mod ferx_model::([\w:]+)$', l.strip())
                                   for l in open(model_listing, encoding='utf-8')) if m}

# 1. Every ferx_model item ferx-core re-exports publicly: `pub use ferx_model::…` in the facade,
#    and `pub use crate::<m>::…` where `<m>` is a module ferx-core takes from ferx-model (the
#    facade mirrors ferx-model's module paths, so `crate::<m>::x` is `ferx_model::<m>::x`).
sources = set()
aliases = set()  # `x as y`: the public name is y
# A facade line this reader cannot parse would drop its items from the baseline without a
# word, leaving their shape ungated. Nothing like that exists today, so refuse it outright
# rather than guess: a nested group, a glob, or a path inside a group.
unreadable = []
FACADE = re.compile(r'^\s*pub use (ferx_model|crate)::((?:\w+::)*)(\{[^}]*\}|\w+(?: as \w+)?);',
                    re.M | re.S)
ANY_MODEL_USE = re.compile(r'^\s*pub use ferx_model::[^;]*;', re.M | re.S)
for d, _, fs in os.walk(os.path.join(root, 'src')):
    for f in fs:
        if not f.endswith('.rs'):
            continue
        path = os.path.join(d, f)
        text = open(path, encoding='utf-8').read()
        read = {m.start() for m in FACADE.finditer(text)}
        for m in ANY_MODEL_USE.finditer(text):
            if m.start() not in read:
                unreadable.append(f'{os.path.relpath(path, root)}: {" ".join(m.group(0).split())}')
        for m in FACADE.finditer(text):
            origin, prefix, tail = m.group(1), m.group(2), m.group(3)
            if origin == 'crate' and prefix.rstrip(':') not in model_mods:
                continue
            for item in (re.split(r',', tail.strip('{}')) if tail.startswith('{') else [tail]):
                item = item.strip()
                if not item or item == 'self':
                    continue
                if '::' in item:
                    unreadable.append(f'{os.path.relpath(path, root)}: `{item}` inside a group')
                    continue
                src = item.split(' as ')[0].strip()
                sources.add('ferx_model::' + prefix + src)
                if ' as ' in item:
                    aliases.add(item.split(' as ')[1].strip())

# 2. Each `pub use ferx_core::X` of ferx-core's listing must have a facade source named X.
names = {s.rsplit('::', 1)[1] for s in sources} | aliases
missing = []
for line in open(core_listing, encoding='utf-8'):
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
lines = open(model_listing, encoding='utf-8').read().split('\n')
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
# The sealed traits of `ferx_model::internal` are ferx-core's private access to crate-private
# members (#3041), not API: their impls on public types, and the members listed under each
# such impl header (`cargo public-api` prints an impl's members right after its header, as
# `pub fn <Type>::member`), are left out.
# More generally, an impl of a ferx-model trait that ferx-core does not re-export (one that was
# crate-private before the split, such as `sens::num::RecHooks`) is not reachable from ferx-core:
# a caller cannot name the trait. Those impls and their members are left out the same way.
TRAIT_IMPL = re.compile(r'^(?:unsafe )?impl(?:<.*?>)? (ferx_model::(?:\w+::)*\w+)(?:<[^>]*>)? for (\S+?)(?:<.*)?$')
internal_owner = None
for line in lines:
    mi = TRAIT_IMPL.match(line)
    if mi and mi.group(1) not in sources:
        internal_owner = mi.group(2)
        continue
    if internal_owner:
        if re.match(r'^pub (?:[a-z ]+ )?' + re.escape(internal_owner) + r'(?:<[^>]*>)?::', line):
            continue
        internal_owner = None
    if line.startswith(('pub trait ferx_model::internal::', 'pub fn ferx_model::internal::',
                        'pub mod ferx_model::internal', 'pub const ferx_model::internal::')):
        continue
    line = line + '\n' if line else ''
    s = subject(line.rstrip('\n'))
    if (s and any(s == src or s.startswith(src + '::') for src in ordered)) or PRIMITIVE_MEMBER.match(line):
        sys.stdout.write(line)
if unreadable:
    sys.stderr.write('facade re-exports this filter cannot read (write them as one flat '
                     '`pub use ferx_model::path::{A, B};` per module):\n  '
                     + '\n  '.join(unreadable) + '\n')
    sys.exit(1)
if missing:
    sys.stderr.write('re-exports with no `pub use ferx_model::…` facade source:\n  '
                     + '\n  '.join(missing) + '\n')
    sys.exit(1)
