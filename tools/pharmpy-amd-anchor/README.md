# Pharmpy `amd` anchor (#1184)

`ferx amd` adds no numerics. Every estimate it reports comes out of a tool
that carries its own external anchor — covsearch (#1180), modelsearch (#1181),
ruvsearch (#1182), iivsearch / iovsearch (#1183) — each of them against
Pharmpy driving NONMEM. What the pipeline itself owns is the **orchestration**,
and that is what this anchor covers:

* `get_subtool_order(strategy)` for every strategy Pharmpy offers, against
  `Strategy::order`, and
* for a corpus of AMD-shaped MFL spaces, the subspace Pharmpy hands its
  structural search and its covariate search, against `amd::space::subspace`.

Nothing is fitted, so the dump takes a second and never touches NONMEM.

```bash
tools/pharmpy-amd-anchor/run.sh          # regenerate the JSON
cargo test -p ferx-tools --lib amd::pharmpy_anchor
```

It runs inside the licensed `pmx` container, where Pharmpy 2.2.0 is
pip-installed — see `tools/pharmpy-variability-anchor/README.md` for how that
container is set up. `FERX_PHARMPY_CONTAINER` overrides the container name.
The committed `crates/ferx-tools/tests/data/amd_pharmpy_anchor.json` is what
CI replays; the container is only needed to regenerate it.

## What the corpus can and cannot cover

Pharmpy 2.2.0's MFL grammar has no `IIV`, `IOV` or `COVARIANCE` statements —
the variability half of a ferx search space is ferx's own (#1183) — so the
corpus covers the structural, allometry and covariate statements. The
variability split is anchored by `amd::space`'s own unit tests, whose oracle
is the tools themselves: each refuses a foreign statement by name, so a
subspace that is wrong is a subspace its tool rejects.

## Two divergences, asserted rather than hidden

* **Default fill.** Where the space says nothing about a step, Pharmpy
  substitutes a whole default search space (`ABSORPTION([FO,ZO,SEQ-ZO-FO]);
  ELIMINATION(FO);TRANSITS([0,1,3,10],*);PERIPHERALS(0..1);LAGTIME([OFF,ON])`
  for the structural step, two exploratory `COVARIATE?` statements for the
  covariate step). ferx **skips the step**, with the reason in `steps.csv`. A
  search the user did not ask for is not a default.
* **`LET` resolution.** Pharmpy substitutes a `LET` when it parses the space;
  ferx carries the definition into the subspace and resolves it against the
  model that step actually starts from, so `@IIV` means the η the *parent* has
  rather than the η the file's author had in mind.

Both are asserted as differences in `amd::pharmpy_anchor`, so one that quietly
disappears is a red test just as a new one would be.
