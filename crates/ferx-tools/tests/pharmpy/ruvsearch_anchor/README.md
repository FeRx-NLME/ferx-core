# Pharmpy `ruvsearch` trajectory anchor (#1182)

Pharmpy 2.2.0's `ruvsearch`, driven by NONMEM 7.5.1, on a simulated
one-compartment oral dataset whose residual error is a **power** model, so the
search has a feature to find. Exercised by
`crates/ferx-tools/tests/ruvsearch_pharmpy_anchor.rs` (slow-gated).

## The data

`power_sim.csv` — 40 subjects, a 100 mg oral dose at time 0, eleven samples at
0.5–120 h, from `simulate_power.py` (numpy only, seed 1182):
`CL = 0.13·e^η₁`, `V = 8·e^η₂`, `KA = 1·e^η₃` with ω² = (0.09, 0.04, 0.30), and
`DV = f + 0.25·f^0.5·ε` with ε ~ N(0, 1). NONMEM format, the `data/warfarin.csv`
columns.

## The input model

`base.ctl` (NONMEM) and `base.ferx` (ferx) are the same proportional-error
model: `Y = IPRED + IPRED·EPS(1)`, `$SIGMA 0.01`, `METHOD=COND INTERACTION`,
`$TABLE ID TIME DV MDV IPRED CWRES` (what ruvsearch needs). Both engines fit
it to OFV **503.002** (`base_ofv` in the JSON).

## How Pharmpy was run

`run_ruvsearch.py`, inside the `pmx` container with Pharmpy pip-installed
(`~/.config/Pharmpy/pharmpy.conf` pointing at `/opt/NONMEM/nm751`):

```python
res = fit(model, esttool="nonmem", name="base_fit")
run_ruvsearch(model=model, results=res, groups=4, p_value=0.001, skip=skip,
              max_iter=3, esttool="nonmem", name=...)
```

twice — `all` (no skip) and `no_iiv` (`skip=["IIV_on_RUV"]`) — and the
trajectory serialised to `pharmpy_ruvsearch.json`: `cwres_models` (the CWRES
screening dOFV and shape estimate of every candidate, per iteration),
`summary_tool` (the refits ranked by `modelrank`), `final_model_code`,
`final_ofv`, `final_estimates`.

## What Pharmpy did

Iteration 1, CWRES screen (dOFV over the CWRES base):

| candidate | dOFV | shape estimate |
|---|---:|---|
| `IIV_on_RUV` | 9.40 | ω = 0.036 |
| `combined` | **67.38** | σ_prop 0.598, σ_add 3.88 (CWRES scale) |
| `power` | 66.40 | θ = −0.332 → power 0.668 |
| `time_varying1` | 0.00 | θ = 1.003 |
| `time_varying2` | 9.96 | θ = 0.800 |
| `time_varying3` | 33.02 | θ = 0.654 |

`combined` is picked (largest dOFV, by 1.0 over `power` — the two are the
near-equivalent pair Pharmpy never tests one after the other) and refitted:
OFV **402.894**, dOFV 100.11 over the input, accepted. Pharmpy also fits the
additive twin (OFV 458.90) and ranks it below. Iteration 2 screens
`IIV_on_RUV` and the time-varying cuts on the new parent: nothing beats the
cutoff (largest dOFV 2.35). Final model: `combined`, OFV 402.894.

## What ferx does on the same input

Recorded by the test; the summary:

- **Full refit** (`cwres_prescreen = false`): every candidate fitted to the
  data. `power` 401.688 (dOFV 101.3) and `combined` 402.894 (dOFV 100.1) —
  ferx's `combined` fit lands on Pharmpy's to 1e-3 — `time_varying3` 454.43,
  `time_varying2` 487.32, `IIV_on_RUV` 501.96. `power` is selected (the lower
  OFV of the pair; the data were simulated from a power model). Iteration 2
  accepts nothing. Final: `power`.
- **CWRES pre-screen** (`cwres_prescreen = true`), ferx's `CWRES` on NONMEM's
  recipe (#1182), iteration 1 (dOFV over the CWRES base):

  | candidate | Pharmpy | ferx |
  |---|---:|---:|
  | `IIV_on_RUV` | 9.40 | 10.00 |
  | `combined` | **67.38** | **67.96** |
  | `power` | 66.40 | 66.98 |
  | `time_varying1` | 0.00 | 0.00 |
  | `time_varying2` | 9.96 | 9.87 |
  | `time_varying3` | 33.02 | 32.89 |

  Same pick (`combined`), refit OFV 402.894 (Pharmpy 402.894), iteration 2
  nothing (Pharmpy 2.35 at most, ferx 2.50), same final model.

The two engines agree on every real-model objective they both evaluate; where
they differ is the screen's choice within the `power`/`combined` pair, which
sits on a 1-unit CWRES-dOFV margin.
