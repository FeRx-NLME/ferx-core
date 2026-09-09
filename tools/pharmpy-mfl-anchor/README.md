# Pharmpy MFL anchor

The external reference for ferx's Model Feature Language parser and `@`-symbol
resolver (`crates/ferx-tools/src/search/{mfl,resolve}.rs`, #1179). MFL is
Pharmpy's grammar, so Pharmpy's own parser is the oracle: every line of
`corpus.txt` is fed to `pharmpy.tools.mfl.parse.parse`, and a set of
symbol-bearing statements is `expand()`ed on a reference model. The dump is
committed as `crates/ferx-tools/tests/data/mfl_pharmpy_anchor.json` and
replayed by the Tier-1 test module `mfl_pharmpy_anchor.rs`, which parses the
same strings with ferx and compares normalised structures.

```bash
tools/pharmpy-mfl-anchor/run.sh     # regenerate the JSON (needs docker + a Pharmpy image)
cargo test -p ferx-tools --lib search::mfl::pharmpy_anchor
```

The container is only needed to *regenerate*; CI replays the committed JSON.

## What the test asserts

For each corpus line, one of:

- **agree** — both parse, and the normalised structures are equal; or both
  reject it.
- **known divergence** — listed by name in the Rust test with a reason, and
  the test asserts the two sides really *do* differ (a stale entry fails).

The normalisation is where ferx's documented deviations live, and nowhere
else: ferx keeps the case of parameter / covariate names (Pharmpy upper-cases),
ferx de-duplicates a mode list (Pharmpy keeps `[ON,ON,OFF]`), and Pharmpy fills
`PERIPHERALS(n)` → `DRUG`, `TRANSITS(n)` → `DEPOT`, `TRANSITS(N)` → `NODEPOT`
at parse time where ferx keeps "unspecified".

The divergences, and why each is kept:

| corpus line | Pharmpy 2.0.0 | ferx | why |
|---|---|---|---|
| `PERIPHERALS(2..0)` | empty count set, silently | error "runs backwards" | a search over no compartments is a typo, not a space |
| `PERIPHERALS(0..100)` | 101 counts | error "above the largest compartment count accepted (64)" | a range is materialised at load; an unbounded one is an allocation of the user's choosing |
| `ALLOMETRY(WT)` | `IndexError` | reference optional | the grammar documents `ALLOMETRY(c[, ref])`; the crash is a Pharmpy bug |
| `COVARIATE?(*, …)`, `COVARIATE?(…, *, …)` | `TypeError` | accepted | the grammar has `parameter_wildcard` / `covariate_wildcard`; the interpreter crashes on them — a Pharmpy bug |
| `# comment`, empty program, `;;` | rejected | accepted | conveniences for a `"""…"""` TOML string |
| `IIV`, `IOV`, `COVARIANCE` | unknown keyword | parsed | in Pharmpy's documentation, not (yet) in its 2.0.0 grammar |
| unknown `@FOO` | resolves to `()` silently | error naming the symbol | a typo must not empty a space |
| `@PD` on a PK-only model | `ValueError` | statement dropped with a note | an empty symbol is an empty space, not a mistake — Pharmpy's own rule for `@CATEGORICAL` |
| `cat` on a covariate declared continuous | accepted | error naming covariate and form | the candidate's parser would refuse it anyway; refusing once at resolution is cheaper |
| explicit pair overriding a symbol pair | drops the **parameter** from the symbol statement (`CL-CRCL` vanishes too) | drops only the pair | Pharmpy's statements are `parameter × covariate` products and cannot remove one cell; ferx works per tuple |

Two Pharmpy rules were *adopted* because of this anchor: a mandatory
`COVARIATE` may not use `*` for its effects, and `TRANSITS(N)` takes no depot
option.

## Contents

| file | role |
|---|---|
| `corpus.txt` | one MFL program per line; `\n` is a literal escape for the multi-line case |
| `dump.py` | runs inside the container; writes the JSON |
| `run.sh` | docker wrapper |
