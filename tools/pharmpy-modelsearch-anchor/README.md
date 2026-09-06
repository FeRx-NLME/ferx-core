# Pharmpy modelsearch anchor

The external reference for ferx's structural-search candidate enumeration
(`crates/ferx-tools/src/modelsearch/`, #1181). The three algorithms and the
rules for which feature may follow which are Pharmpy's
(`pharmpy/tools/modelsearch/algorithms.py`), so Pharmpy's own workflow is the
oracle: for every case in `dump.py`'s `CASES`, the script builds Pharmpy's
base model (`create_base_model`), filters the space against it
(`filter_mfl_statements`), builds the algorithm's task graph **without fitting
anything**, and records every `create_candidate` task with its feature set and
its nearest candidate ancestors. The dump is committed as
`crates/ferx-tools/tests/data/modelsearch_pharmpy_anchor.json` and replayed by
the Tier-1 module `pharmpy_anchor.rs`, which runs the same cases through
ferx's `search` on a fitter that returns one OFV for everything.

```bash
tools/pharmpy-modelsearch-anchor/run.sh   # regenerate the JSON (needs docker + a Pharmpy image)
cargo test -p ferx-tools --lib modelsearch::pharmpy_anchor
```

The container is only needed to *regenerate*; CI replays the committed JSON.

## What the test asserts

For each case:

- the candidate moves (`funcs`) are the same features in the same
  dictionary order, and the base is derived by the same least number of
  transformations;
- the candidates, as `(feature set, parent's feature set)` edges, are the
  same multiset — through `reduced_stepwise`'s collector nodes, whose
  children ferx attaches to the group member it kept;
- for `exhaustive`, the same combinations in the same order, so `run{n}`
  matches `modelsearch_run{n}`.

Where ferx deliberately differs, the case is named in the Rust test's
`DIVERGENCES` with the reason, and the test asserts the two sides really
*do* differ (a stale entry fails):

| case | Pharmpy 2.0.0 | ferx | why |
|---|---|---|---|
| `PERIPHERALS(0..2); LAGTIME([OFF,ON])` on an oral base | fills the unnamed `ABSORPTION` with its default and derives an **INST** base | keeps the base's `FO` | a category the space does not name is not a request to change it |
| `ABSORPTION([INST,FO]); LAGTIME([OFF,ON])` on an IV base | `LAGTIME(ON)` is a candidate from the bolus base (the pair table sees only *applied* features) | never builds a lagged bolus | the pair rule is about the model, not the path that reached it |
| `ABSORPTION(FO); PERIPHERALS(0..2); LAGTIME([OFF,ON])`, `reduced_stepwise` | does not collapse the one duplicate group (`len(groups) > 1`), so it runs the full stepwise tree | collapses every group | the reduction is the algorithm's whole point |
| `ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])`, `exhaustive` | builds `INST + LAGTIME(ON)` | applies the pair table to combinations too | the issue asks that no fit be spent on a meaningless model |

## Why there are no transit cases

Two Pharmpy behaviours make `TRANSITS` unusable as an enumeration oracle:

- `create_base_model` raises on a space that lists `TRANSITS(0, NODEPOT)`
  against a first-order base ("Cannot set the number of transits to 1 for
  model with instantaneous absorption"), so the natural way to keep the
  base on the space cannot be written.
- `get_model_features` reports a transit chain by its **compartment** count
  (`TRANSITS(2,NODEPOT)` for `TRANSITS(1, NODEPOT)`), so a base derived
  with `TRANSITS(1)` does not filter `TRANSITS(1, NODEPOT)` out of the
  moves and Pharmpy re-applies the base's own chain as a candidate.

Both were observed with the dump script (Pharmpy 2.0.0). The transit rules
of the pair table — `ABSORPTION(FO)` with `TRANSITS(1, NODEPOT)`, a lag time
with any chain, a chain on a bolus — are pinned by the unit tests in
`structure_tests.rs`, transcribed from `_is_allowed`.

## Contents

| file | role |
|---|---|
| `dump.py` | runs inside the container; writes the JSON |
| `run.sh` | docker wrapper |
