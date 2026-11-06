# Phase 0 scripts

These produced the numbers in `plans/fitrx-consolidation-phase0.md`. They are one-off and are not part of any build or CI job.

Each takes a scratch directory `S` holding `repos/ferx-r` (a clone) and `rlib` (a library with ferx installed from it).

| Script | Produces |
|---|---|
| `extract.py <S> <ferx-core>` | Key sets for `FitWire` and its sub-structs, the R writer and reader, and the R list, in `S/an/keys.json` |
| `matrix.py <S>` | The top-level field matrix |
| `mutators.R <R-dir> <out.csv>` | Every compound assignment under a directory, from the parsed AST |
| `rt.R`, `rt2.R <S>` | Live fit against R round trip against core re-save, on eight fixtures |
| `leaf.py <S>` | Leaf-level key differences between R-written and core-written `fit.json` |
| `cost.R`, `many.R <S>` | `.rds` size, timing and many-fit cost |
| `harness/` | Times core `save_fit` and `load_fit`. Rename `Cargo.toml.txt` to `Cargo.toml` to build it |

Do not build large datasets by copying subjects: deflate then shrinks the bundle about fifteenfold. Add per-subject noise.
