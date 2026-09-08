# Pharmpy `iivsearch` trajectory anchor (#1183)

Pharmpy 2.2.0's `iivsearch`, driven by NONMEM 7.5.1, on a simulated
one-compartment oral dataset whose variability structure has something to
find: IIV on CL and V only, correlated, and none on KA. Exercised by
`crates/ferx-tools/tests/iivsearch_pharmpy_anchor.rs` (slow-gated).

## The data

`iiv_sim.csv` — 40 subjects, a 100 mg oral dose at time 0, eleven samples at
0.5–120 h, from `simulate_iiv.py` (numpy only, seed 1183): `CL = 0.13·e^η₁`,
`V = 8·e^η₂` with ω² = (0.09, 0.04) and r = 0.6, `KA = 1` with no η, and a
15% proportional residual error. NONMEM format, the `data/warfarin.csv`
columns.

## The input model

`base.ctl` (NONMEM) and `base.ferx` (ferx) are the same diagonal three-η
model (`ETA_CL`, `ETA_V`, `ETA_KA`), `METHOD=COND INTERACTION`. Both engines
fit it to OFV 690.16 / 690.03 (`base_ofv` in the JSON is NONMEM's).

## How Pharmpy was run

`run_iivsearch.py`, inside the `pmx` container with Pharmpy pip-installed
(`tools/pharmpy-variability-anchor/pmx.sh iivsearch_anchor python3 run_iivsearch.py`):

```python
res = fit(model, esttool="nonmem", name="base_fit")
run_iivsearch(model=model, results=res, rank_type="bic", esttool="nonmem", **kwargs)
```

three times — `td` (`top_down_exhaustive`, the default space
`IIV?(@IIV,EXP);COVARIANCE?(IIV,@IIV)`), `bu` (`bottom_up_stepwise` on
`IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,@IIV)`) and `sim`
(`simultaneous_stepwise` on the same space) — and the trajectory serialised
to `pharmpy_iivsearch.json`: `summary_tool` (each step's ranking: model,
description, BIC(iiv), rank), `summary_models` (every model's OFV and
minimization status), `final_description`, `final_ofv`, `final_estimates`.

## What Pharmpy did

Top-down, ranked on the BIC(iiv) (`rank_type="bic"` is `bic_iiv` in this
tool):

| step | model | description | BIC(iiv) | rank |
|---|---|---|---:|---:|
| 1 | run2 | `[CL]+[V]` | 697.411 | 1 |
| 1 | input | `[CL]+[KA]+[V]` | 701.229 | – |
| 1 | run1, run3–run7 | the other subsets | 974–1243 | |
| 2 | run8 | `[CL,V]` | 666.086 | 1 |
| 3 | run8 vs input | | dBIC 35.14 | |

Final: `[CL,V]`, OFV 655.020. Bottom-up from the base `[CL]` (1002.664):
`[CL]+[V]` (697.411) beats `[CL]+[KA]`; adding KA does not improve; the
block stage gives `[CL,V]` (666.086); final `[CL,V]`. Simultaneous from
`[CL]`: `[CL,V]` wins step 1 (666.086); in step 2 `[CL,V]+[KA]` (run5)
terminated with unreportable significant digits at OFV 655.47 and
`[CL,KA,V]` was mis-built by Pharmpy as `[KA,V]+[CL]` (746.31); final
`[CL,V]`.

Rows with `rank: None` failed Pharmpy's strictness (NONMEM's minimization
did not succeed) — most of the subsets that drop CL or V.

## What ferx does on the same input

Recorded by the test; the summary:

- **Top-down**: the same seven candidates in the same numbering (`run1`
  `[CL]+[KA]` … `run7` no η), `run2` `[CL]+[V]` at BIC(iiv) 697.411,
  `run8` `[CL,V]` at 666.086, final `[CL,V]` at OFV 655.020. Same
  trajectory to three decimals.
- **Bottom-up**: base `[CL]` 1002.664, `run2` `[CL]+[V]` 697.411, `run3`
  `[CL]+[KA]+[V]` 701.102 (not better), `run4` `[CL,V]` 666.086, final
  `[CL,V]`.
- **Simultaneous**: step 1 identical (`run4` `[CL,V]` 666.086). Step 2
  diverges: `run5` `[CL,V]+[KA]` is a mixed block + diagonal ω, which
  ferx's FOCEI estimates with its cross-block covariances free (#1018) —
  OFV 647.73, i.e. the full block `[CL,V,KA]` (`run6` lands on the same
  647.73). `verify_run5.py` refits Pharmpy's `run5` with NONMEM from ferx's
  estimates: NONMEM evaluates that point at 658.50 (as ferx does when
  *evaluating* the declared model) and ends at 655.42, so the gap is the
  structure, not the optimizer. ferx ends on `run5` where Pharmpy ends on
  `[CL,V]`; the test asserts that divergence by name, and the search
  carries a note naming #1018 on the run. (The strictness gate does not
  catch it: the collapsing ω_KA is an internal-guard warning, not a
  declared bound.)

## Contents

| file | role |
|---|---|
| `simulate_iiv.py` | writes `iiv_sim.csv` |
| `base.ctl`, `base.ferx` | the input model, both engines |
| `run_iivsearch.py` | runs Pharmpy, writes `pharmpy_iivsearch.json` |
| `verify_run5.py` | the NONMEM refit of `run5` from ferx's estimates (`verify_run5.json`) |
