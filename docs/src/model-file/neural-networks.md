# Neural Networks

> **Maturity: experimental** — see [Feature Maturity](../maturity.md) for what this means.

ferx-core is gaining two complementary neural-network extensions. Both are
gated behind the `nn` cargo feature (off by default) and are being built
incrementally — see [`plans/dcm-and-low-dim-node.md`][plan] in the source
tree for the milestone roadmap.

This page is the landing point: it explains which extension fits which
modeling problem. Detailed reference pages for each block sit beneath it.

[plan]: https://github.com/FeRx-NLME/ferx-core/blob/main/plans/dcm-and-low-dim-node.md

## TL;DR — which one do I need?

| Known PK / PD structure? | Known covariate model? | Use                                  |
|--------------------------|------------------------|--------------------------------------|
| Yes (e.g. 2-cpt oral)    | Yes                    | Classical analytical — no NN needed  |
| Yes                      | No / partial           | [`[covariate_nn]`](covariate-nn.md) (DCM, Phase A) |
| No / partial             | Yes                    | [`[dynamics_nn]`](dynamics-nn.md) (low-dim NODE, Phase B) |
| No                       | No                     | Both — they coexist in the same model |

If you also need *structural uncertainty* — system noise that accumulates
between observations regardless of covariates or dynamics — see
[`[diffusion]`](diffusion.md) (SDE / Extended Kalman Filter). It is
orthogonal to either neural-network mechanism.

## Two extension points, two papers

ferx-core integrates neural networks at two different layers of the model
pipeline, motivated by two distinct papers in the pharmacometrics
literature:

### `[covariate_nn]` — Deep Compartment Models (DCM)

A neural network replaces the **typical-value covariate model**: instead of
writing

```
CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)
```

you write a network whose inputs are subject covariates and whose outputs
are PK typical values, then combine with etas as usual:

```
[covariate_nn TYPICAL_PK]
  inputs  = [WT, CRCL]
  outputs = [CL, V1, Q, V2, KA]
  layers  = [16]
  activation = tanh
  output     = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V1 = TYPICAL_PK.V1 * exp(ETA_V1)
  …
```

The compartmental ODE / analytical solution downstream is unchanged.
Etas attach to the *final* PK parameters (not the NN weights), so the
inner FOCEI loop runs exactly as it does for an analytical model.

Reference: **Janssen et al. 2022** — *Deep compartment models: a deep
learning approach for the reliable prediction of time-series data in
pharmacokinetic modeling.* CPT Pharmacometrics Syst Pharmacol 11:934–945.
DOI [10.1002/psp4.12808](https://doi.org/10.1002/psp4.12808).

### `[dynamics_nn]` — Low-dimensional Neural ODEs

A small neural network appears on the **right-hand side of an ODE**,
replacing mechanistic terms like Michaelis–Menten or linear elimination
when the dynamics are unknown:

```
[dynamics_nn ka_nn]
  inputs     = [depot]
  hidden     = 5
  activation = relu
  iiv        = additive

[odes]
  d/dt(depot)   = -ka_nn(depot)
  d/dt(central) =  ka_nn(depot) - cl_nn(central) * central
```

The compartmental *topology* (number of states, dose routing, observation
mapping) is still specified by the modeler. Random effects can attach
directly to the NN weights via `iiv = additive` — Bräm et al.'s key
contribution.

Reference: **Bräm et al. 2025** — *Low-dimensional neural ordinary
differential equations accounting for inter-individual variability
implemented in Monolix and NONMEM.* CPT Pharmacometrics Syst Pharmacol
14:5–16. DOI [10.1002/psp4.13265](https://doi.org/10.1002/psp4.13265).

## Guardrails

Because both blocks declare neural networks, the parser and runtime
enforce a few rules to keep the difference visible:

- **Block-name cross-check**: a `[covariate_nn]` referenced inside `[odes]`
  is a hard error pointing to `[dynamics_nn]` (and vice versa).
- **Fit-output reporting**: both `{model}-fit.yaml` (CLI) and `.fitrx`
  (programmatic round-trip) emit a `neural_networks:` section listing every
  active block, its shape, and its weight count.
- **Runtime warning on `[dynamics_nn]`**: a `FitResult.warnings` entry
  notes the expected 10–100× slowdown vs classical PK and the lack of
  reliable standard errors for NN weights.
- **Scoped covariance auto-disable**: when `[dynamics_nn]` is present and
  `covariance = true`, ferx-core overrides to `covariance = false` and
  warns. `[covariate_nn]` weights are reasonably identifiable and
  covariance estimation stays enabled there.

## Status

| Component | Status |
|-----------|--------|
| `nn` cargo feature, `MlpMapper`, `NamedMlpMapper` (forward + Jacobian) | Implemented in `src/nn/mod.rs`; behind `--features nn` |
| `[covariate_nn]` parser block + auto-generated thetas | Pending (Phase A M1) |
| `pk_param_fn` dispatch on `[covariate_nn]` | Pending (Phase A M1) |
| `method = nn_mse` fixed-effects objective | Pending (Phase A M1) |
| Mu-ref composition `TYPICAL_PK.X * exp(ETA_X)` | Pending (Phase A M2) |
| `[dynamics_nn]` block + NN-in-ODE-expression syntax | Pending (Phase B) |
| Activation-aware Bräm-style initializer | Pending (Phase B) |
| Outer-loop AD through NN weights (production speed) | Deferred (Phase A M3) |

The Rust-level API (the `CovariateMapper` trait and `MlpMapper`) is
already callable today — see `tests/nn_dcm_smoke.rs` for an end-to-end
example.
