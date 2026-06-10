# Error Model

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

The `[error_model]` block defines the residual error structure, specifying how observed data (DV) relates to model predictions.

## Syntax

```
DV ~ ERROR_TYPE(SIGMA_PARAMS)
```

## Available Error Models

### Additive

```
DV ~ additive(SIGMA_NAME)
```

The residual variance is constant across all predictions:

\\[ \text{Var}(DV) = \sigma^2 \\]

Use when measurement error is independent of concentration (e.g., assay with fixed precision).

### Proportional

```
DV ~ proportional(SIGMA_NAME)
```

The residual variance scales with the predicted value:

\\[ \text{Var}(DV) = (\sigma \cdot f)^2 \\]

where \\( f \\) is the model prediction. Use when measurement error increases with concentration (most common in PK).

### Combined

```
DV ~ combined(SIGMA_PROP, SIGMA_ADD)
```

Combines proportional and additive components:

\\[ \text{Var}(DV) = (\sigma_1 \cdot f)^2 + \sigma_2^2 \\]

Use when both proportional and additive error sources are present. Requires two sigma parameters defined in `[parameters]`.

## Log-transform-both-sides (LTBS)

For data whose residual error is *multiplicative* (a constant CV across the
concentration range), an alternative to the proportional model is to fit on the
**log scale**: both the observation and the prediction are log-transformed and an
**additive** error is applied on the log scale. This matches NONMEM's
`Y = LOG(F) + EPS(1)` convention and is the natural choice when importing a model
or dataset from NONMEM.

There are two ways to declare it, depending on the scale of the `DV` column in
your dataset:

### `log(DV) ~ additive(SIGMA)` — DV on the natural scale

```
[parameters]
  sigma ADD_LOG ~ 0.1     # additive SD on the LOG scale

[error_model]
  log(DV) ~ additive(ADD_LOG)
```

The engine log-transforms the `DV` column itself (once, at load), and compares it
to `log(prediction)`. Use this when your data holds concentrations on the natural
scale.

### `DV ~ log_additive(SIGMA)` — DV already log-transformed

```
[error_model]
  DV ~ log_additive(ADD_LOG)
```

Use this when the `DV` column is *already* log-transformed in the dataset (e.g.
exported from a NONMEM workflow that pre-logged the data). The engine takes `DV`
as-is and log-transforms only the prediction. `log_additive` is additive error on
the log scale.

### Output scale

Under LTBS, **everything is reported on the log scale**, matching NONMEM:
`IPRED`/`PRED`, `IWRES`/`CWRES`, and simulated `DV` are all on the log scale.
Back-transform with `exp()` if you need natural-scale values.

The likelihood term is the additive form on the log scale:

\\[ \text{Var}(\log DV) = \sigma^2, \qquad \text{IWRES} = \frac{\log DV - \log f}{\sigma} \\]

### Restrictions

- **Additive only.** LTBS pairs with additive error on the log scale; `log(DV) ~
  proportional(...)` / `combined(...)` are rejected at parse time.
- **Single endpoint.** Not supported with per-CMT (multi-endpoint) error models.
- **No SDE.** Not supported with a `[diffusion]` (SDE/EKF) model.
- **BLOQ/M3 is supported** — the LLOQ is log-transformed alongside `DV`.
- A non-positive `DV` under `log(DV) ~ additive(...)` cannot be log-transformed;
  it is floored to `log(1e-12)` and a warning is emitted (check your data scale,
  or use `DV ~ log_additive(...)` if the data is already log-transformed).

## Multiple endpoints (per-CMT error models)

For simultaneous PK/PD (and other multi-analyte) models, a single observed
compartment is not enough: plasma concentrations and a PD effect typically
need *different* residual error models in the *same* joint likelihood. Prefix
each error line with `CMT=N:` to assign a distinct error model to each observed
compartment, dispatched by the dataset's `CMT` column:

```
[error_model]
  CMT=2: DV ~ proportional(PROP_ERR_PK)   # plasma concentration (central)
  CMT=3: DV ~ additive(ADD_ERR_PD)        # PD effect (effect compartment)
```

Every observation row is matched to the endpoint whose `CMT=N` equals its
`CMT` value, and its residual variance is computed from that endpoint's error
model and sigma(s). All endpoints contribute to one FOCEI objective, so PK and
PD parameters — and their uncertainty — are estimated jointly. This is the
gold-standard alternative to the sequential workaround (fit PK, freeze IPRED,
then fit PD), which underestimates uncertainty.

Each endpoint's sigma parameters are declared once in `[parameters]`, as usual:

```
[parameters]
  sigma PROP_ERR_PK ~ 0.10 (sd)
  sigma ADD_ERR_PD  ~ 1.00 (sd)
```

Rules and restrictions:

- **ODE models only.** Per-CMT dispatch lives in the finite-difference
  likelihood path used by ODE models. An analytical PK model with `CMT=N:`
  lines is rejected at parse time.
- **No mixing styles.** An `[error_model]` block is either a single plain
  `DV ~ ...` line *or* all `CMT=N:` lines — not both.
- **Coverage is checked at fit time.** Every observed `CMT` in the dataset must
  have a matching `CMT=N:` entry, or `fit()` errors and names the missing
  compartments. Duplicate `CMT=N` entries are rejected at parse time.
- **Estimation method.** Supported with FOCE/FOCEI, the Gauss-Newton
  optimizers, and SAEM (optionally followed by `imp`).

A complete worked model lives in [`examples/emax_pkpd.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/emax_pkpd.ferx)
— an oral 1-compartment PK model with an effect-compartment Emax PD readout,
proportional error on the plasma endpoint and additive error on the PD
endpoint. The per-CMT *readout* (which compartment/expression each `CMT` maps
to) is configured in the [`[scaling]`](scaling.md) block.

## Sigma scale

**All sigma parameters are estimated and reported on the standard-deviation scale**, not the variance scale. This is true for both proportional and additive components, and for both elements of a combined error model.

In particular:

| Error model       | Initial value `sigma = X` means …                                       |
|-------------------|--------------------------------------------------------------------------|
| `proportional`    | The residual SD scales as `X · f`, i.e. **CV% = `X · 100`**             |
| `additive`        | The residual SD is `X` in the units of `DV` (no CV interpretation)     |
| `combined`        | First sigma is proportional (CV-style), second is additive (units of `DV`) |

So `sigma PROP_ERR ~ 0.1` for a proportional model is a **10% CV** initial value, not 1%. Likewise, the SE on a proportional sigma is on the SD scale — multiply by 100 for an SE in CV-percentage points.

The fitted YAML emits both the SD (`estimate`) and the variance (`variance: estimate²`); for proportional components it also emits `cv_pct = estimate · 100` so downstream tooling does not have to re-derive it.

## Examples

Proportional error (most common):
```
[parameters]
  sigma PROP_ERR ~ 0.01

[error_model]
  DV ~ proportional(PROP_ERR)
```

Additive error:
```
[parameters]
  sigma ADD_ERR ~ 1.0

[error_model]
  DV ~ additive(ADD_ERR)
```

Combined error:
```
[parameters]
  sigma PROP_ERR ~ 0.1
  sigma ADD_ERR  ~ 0.5

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
```

## Impact on Estimation

The error model affects:
- **Individual weighted residuals (IWRES)**: `(DV - IPRED) / sqrt(Var)`
- **Conditional weighted residuals (CWRES)**: Accounts for uncertainty in random effect estimates
- **Objective function value (OFV)**: The likelihood includes `log(Var)` terms, so the error model structure directly influences parameter estimates
