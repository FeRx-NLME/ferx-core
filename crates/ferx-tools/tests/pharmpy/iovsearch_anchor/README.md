# Pharmpy `iovsearch` trajectory anchor (#1183)

Pharmpy 2.2.0's `iovsearch`, driven by NONMEM 7.5.1, on a simulated
one-compartment oral dataset over three dosing occasions with
inter-occasion variability on CL only, so the search has a structure to
find. Exercised by `crates/ferx-tools/tests/iovsearch_pharmpy_anchor.rs`
(slow-gated).

## The data

`iov_sim.csv` — 40 subjects, a 100 mg oral dose at 0, 48 and 96 h (three
occasions, `OCC` = 1, 2, 3), seven samples at 0.5–24 h after each, from
`simulate_iov.py` (numpy only, seed 1184): `CL = 0.5·e^(η₁+κ)`,
`V = 10·e^η₂`, `KA = 1.2·e^η₃` with ω² = (0.09, 0.04, 0.30), κ² = 0.04 on
CL alone (a fresh κ per occasion; the carried-over amount eliminates at the
occasion's clearance, solved piecewise in closed form), and a 15%
proportional residual error. NONMEM format plus `OCC`.

## The input model

`base.ctl` (NONMEM) and `base.ferx` (ferx) are the same diagonal three-η
model with no κ; the ferx file declares `iov_column = OCC` so the occasions
are read. Both engines fit it to OFV 1290.157.

## How Pharmpy was run

`run_iovsearch.py`, inside the `pmx` container
(`tools/pharmpy-variability-anchor/pmx.sh iovsearch_anchor python3 run_iovsearch.py`):

```python
res = fit(model, esttool="nonmem", name="base_fit")
run_iovsearch(model=model, results=res, column="OCC", rank_type="bic",
              distribution=..., esttool="nonmem")
```

for `disjoint` and `same-as-iiv`, serialised to `pharmpy_iovsearch.json`
(`summary_tool`, `summary_models`, the final model).

## What Pharmpy did

Step 1, ranked on the BIC(random) (`rank_type="bic"` is `bic_random` in this
tool), `disjoint`:

| model | description | BIC(random) | rank |
|---|---|---:|---:|
| run7 | `IIV([CL]+[V]+[KA]);IOV([CL])` | 1219.577 | 1 |
| run3 | `IOV([CL]+[V])` | 1222.053 | – |
| run4 | `IOV([CL]+[KA])` | 1223.377 | – |
| run1 | `IOV([CL]+[KA]+[V])` (full) | 1225.937 | – |
| input | no IOV | 1315.980 | – |
| run5, run6, run2 | `IOV([V])`, `IOV([KA])`, `IOV([KA]+[V])` | 1319–1324 | – |

Step 2 from run7: `run8` `IIV([V]+[KA]);IOV([CL])` at 1273.951 is worse;
final `IOV([CL])`, OFV 1190.066. `same-as-iiv` (the η are diagonal, so the
κ are too) is the same search with the candidates numbered in a different
set order; same numbers, same final model.

## What ferx does on the same input

The same seven step-1 candidates, `IOV([CL])` best at BIC(random) 1219.577,
`IOV([CL]+[V])` 1222.048, the full model 1225.738, the input 1315.980; the
same step-2 candidate at 1273.951; the same final model at OFV 1190.066 —
to three decimals on every model both engines fitted to a proper optimum.
The subsets that remove CL's κ (`IOV([V])`, `IOV([KA])`, `IOV([KA]+[V])`)
are seeded from the full model's fit, whose κ on KA collapsed onto the
optimizer's rail; the search seed floors a collapsed κ as it floors a
collapsed ω, which is what lets those three candidates start.

## Contents

| file | role |
|---|---|
| `simulate_iov.py` | writes `iov_sim.csv` |
| `base.ctl`, `base.ferx` | the input model, both engines |
| `run_iovsearch.py` | runs Pharmpy, writes `pharmpy_iovsearch.json` |
