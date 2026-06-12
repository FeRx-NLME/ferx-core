# `[covariate_nn]` — Deep Compartment Models (DCM)

> **Maturity: experimental** — see [Feature Maturity](../maturity.md) for what this means.

A `[covariate_nn]` block replaces the **typical-value covariate model** with
a small feed-forward neural network. Instead of writing

```ferx
CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)
```

you write a network whose inputs are subject covariates and whose outputs
are PK typical values, then compose with etas the standard way:

```ferx
[covariate_nn TYPICAL_PK]
  inputs     = [WT, CRCL]
  outputs    = [CL, V1, Q, V2, KA]
  layers     = [8, 8]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  Q  = TYPICAL_PK.Q  * exp(ETA_Q)
  V2 = TYPICAL_PK.V2 * exp(ETA_V2)
  KA = TYPICAL_PK.KA * exp(ETA_KA)
```

The compartmental ODE / analytical solution downstream is unchanged; etas
attach to the *final* PK parameters (not the NN weights), so the inner
FOCEI loop runs exactly as it does for an analytical model. This is the
"mixed-effects DCM" variant.

> **Reference**: Janssen A. et al. (2022). *Deep compartment models: A deep
> learning approach for the reliable prediction of time-series data in
> pharmacokinetic modeling.* CPT Pharmacometrics Syst Pharmacol 11:934–945.
> DOI [10.1002/psp4.12808](https://doi.org/10.1002/psp4.12808).

## Status

Behind the `nn` cargo feature (off by default). Build with:

```sh
RUSTFLAGS="-Z autodiff=Enable" cargo build --release --features nn
```

A full runnable example lives at [`examples/warfarin_dcm.ferx`][ex] —
compare with [`examples/two_cpt_oral_cov.ferx`][analytical], which uses
the same data and eta structure but an analytical covariate model.

[ex]: https://github.com/FeRx-NLME/ferx-core/blob/main/examples/warfarin_dcm.ferx
[analytical]: https://github.com/FeRx-NLME/ferx-core/blob/main/examples/two_cpt_oral_cov.ferx

## Block syntax

```ferx
[covariate_nn NAME]
  inputs     = [COV_1, COV_2, ...]
  outputs    = [PK_PARAM_1, PK_PARAM_2, ...]
  layers     = [hidden_1, hidden_2, ...]
  activation = tanh | relu | sigmoid | softplus | exp | identity
  output     = tanh | relu | sigmoid | softplus | exp | identity   # optional, default `identity`
```

| Field         | Required | Meaning                                                           |
|---------------|----------|-------------------------------------------------------------------|
| `inputs`      | yes      | Covariate column names from the NONMEM CSV                        |
| `outputs`     | yes      | PK parameter names (`cl`, `v` / `v1`, `q` / `q2`, `v2`, `ka`, `f`, `q3`, `v3`, `lagtime` / `alag`) |
| `layers`      | yes      | Hidden layer widths, in order. At least one entry required        |
| `activation`  | yes      | Hidden-layer element-wise activation                              |
| `output`      | no       | Output-layer activation. Use `softplus` or `exp` for positive PK params |

The header form `[covariate_nn NAME]` requires the **NAME** (e.g.
`TYPICAL_PK`). NAME is the dot-access prefix in `[individual_parameters]`
expressions (`NAME.CL`, `NAME.V1`, …).

Multiple `[covariate_nn]` blocks per model are permitted; they're keyed
by NAME and ordered alphabetically when generating theta names.

## Auto-generated parameters

For each block, the parser generates one theta per weight and per bias.
Names follow the convention:

- `W_<NAME>_<l>_<i>_<j>` — weight from input unit `j` (in layer `l−1`) to
  output unit `i` (in layer `l`)
- `B_<NAME>_<l>_<i>` — bias of output unit `i` in layer `l`

Layers are 1-indexed (input layer is layer 0, doesn't have weights).
For a 2 → 8 → 8 → 5 network: 2·8 + 8 + 8·8 + 8 + 8·5 + 5 = **141 thetas**.

Initial values use a Glorot-style deterministic scheme seeded by the
block NAME, so builds are reproducible without pulling `rand` into the
parser. Weights are unbounded (identity-packed): the optimizer sees them
on the natural scale, no log transform. Biases initialise to 0.

## Activation reference

| `activation` value | Behavior                  | Typical use                        |
|--------------------|---------------------------|-------------------------------------|
| `identity`         | `f(x) = x`                | Output layer when you'll wrap your own positivity head |
| `relu`             | `f(x) = max(0, x)`        | Cheap hidden activation, but creates kinks; reduces smoothness for FOCEI |
| `tanh`             | `f(x) = tanh(x)`          | **Recommended hidden default** — smooth, bounded `(-1, 1)` |
| `sigmoid`          | `f(x) = 1 / (1 + exp(-x))`| Bounded `(0, 1)`; useful for `F` bioavailability outputs |
| `softplus`         | `f(x) = ln(1 + exp(x))`   | **Recommended output for PK params** — guarantees positivity, no exp blow-up |
| `exp`              | `f(x) = exp(x)`           | Strictly positive, but unbounded; prefer `softplus` |

All activations use `if`/`else` instead of `f64::max`/`f64::min` for AD
safety (see [CLAUDE.md] → "Autodiff-Safe Code in `ad/` Module").

[CLAUDE.md]: https://github.com/FeRx-NLME/ferx-core/blob/main/CLAUDE.md

## Mu-ref composition with etas

The pattern `TYPICAL_PK.CL * exp(ETA_CL)` is recognised by the parser as
a **lognormal mu-ref** with `TYPICAL_PK.CL` as the structured anchor
name. It shows up in `FitResult.mu_refs` as:

```
{ "ETA_CL": MuRef { theta_name: "TYPICAL_PK.CL", log_transformed: true } }
```

and in `eta_param_info[ETA_CL].param_type` as `LogNormal`. The
AD-aware re-centering fast path (`compute_mu_k`) currently silently
skips structured-name mu-refs and FOCEI falls back to its standard
inner-loop path — correct, just slower than a true mu-ref-aware AD
inner loop. That perf improvement lives on the Phase A M2 roadmap in
[`plans/dcm-and-low-dim-node.md`].

[`plans/dcm-and-low-dim-node.md`]: https://github.com/FeRx-NLME/ferx-core/blob/main/plans/dcm-and-low-dim-node.md

## Fit output

Fit YAML (`{model}-fit.yaml`) and `.fitrx` bundles emit a compact
`neural_networks:` section in addition to the regular `theta:` block.
The NN weights are summarised by shape, activations, weight count, and
basic statistics over the trained values rather than dumped one row
per weight — for a 141-weight network, the YAML stays scannable:

```yaml
neural_networks:
  TYPICAL_PK:
    shape: [2, 8, 8, 5]
    hidden_activation: tanh
    output_activation: softplus
    inputs: [WT, CRCL]
    outputs: [CL, V1, Q, V2, KA]
    n_weights: 141
    weights_summary:
      min:  -1.034521
      max:   0.987643
      mean:  0.014277
      std:   0.342118
```

Full per-weight values remain in `result.theta` at indices
`weights_offset..weights_offset + n_weights` (the `weights_offset` is
also exposed on each `FitResult.neural_networks[k]` entry). `ferx-r`
loaders round-trip them losslessly via `fit.json` inside the `.fitrx`
archive.

## What's not yet wired

These items are tracked against Phase A M2 / future PRs in the
[plan][`plans/dcm-and-low-dim-node.md`]:

- `method = nn_mse` — Janssen 2022's original fixed-effects MSE objective
  (FOCEI works as the default today).
- AD-aware mu-ref re-centering for NN-anchored etas — fits work today
  but the inner-loop AD fast path skips NN-anchored etas; performance
  improvement, not correctness.
- Phase B `[dynamics_nn]` block for neural-ODE-style RHS terms (Bräm
  et al. 2025).

## See also

- [`docs/src/model-file/neural-networks.md`](neural-networks.md) — landing
  page with the "which block do I need?" decision table.
- [`examples/warfarin_dcm.ferx`][ex] — runnable example.
- [`examples/two_cpt_oral_cov.ferx`][analytical] — analytical equivalent
  for comparison.
- [`plans/dcm-and-low-dim-node.md`] — full milestone roadmap.
