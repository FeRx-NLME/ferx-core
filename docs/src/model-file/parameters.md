# Parameters

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

The `[parameters]` block defines all model parameters: fixed effects (theta), between-subject variability (omega), and residual error (sigma).

## Theta (Fixed Effects)

```
theta NAME(initial_value, lower_bound, upper_bound)
```

- **NAME**: Parameter name (used in `[individual_parameters]` expressions)
- **initial_value**: Starting value for estimation
- **lower_bound**: Lower bound constraint. Thetas with `lower_bound >= 0`
  are log-transformed internally (so the optimiser sees `log(theta)`);
  thetas with `lower_bound < 0` are estimated on the natural scale.
  This lets covariate exponents like `theta THETA_AGE_CL(-0.01, -1, 1)`
  pass through unchanged while still allowing CL/V/KA to be positivity-
  constrained via the log transform. **Edge case:** `lower_bound = 0.0`
  still picks the log transform with an internal `1e-10` floor, so a
  parameter that genuinely needs to reach 0 should declare a small
  negative `lower_bound` (e.g. `-1e-6`) to switch to identity packing.
- **upper_bound**: Upper bound constraint

Example:
```
theta TVCL(0.134, 0.001, 10.0)
theta TVV(8.1, 0.1, 500.0)
theta TVKA(1.0, 0.01, 50.0)
```

## Omega (Between-Subject Variability)

### Diagonal omega

```
omega NAME ~ value          # value interpreted as variance (default)
omega NAME ~ value (sd)     # value interpreted as standard deviation
omega NAME ~ value (variance)   # explicit no-op equivalent to no annotation
```

- **NAME**: Random effect name (used in `[individual_parameters]` as `ETA_XXX`)
- **value**: Initial value. Default scale is **variance** (the diagonal element of the omega matrix). Append `(sd)` to specify a standard deviation instead — the parser squares it before storing. The optimizer always works on the variance scale internally.

Example (variance scale, default):
```
omega ETA_CL ~ 0.07
omega ETA_V  ~ 0.02
omega ETA_KA ~ 0.40
```

Equivalent declaration using SD coding (`(sd)` annotation):
```
omega ETA_CL ~ 0.265 (sd)    # ≡ ~ 0.0702
omega ETA_V  ~ 0.141 (sd)    # ≡ ~ 0.0200
omega ETA_KA ~ 0.632 (sd)    # ≡ ~ 0.400
```

Each variance represents the between-subject variability for that parameter. The coefficient of variation (CV%) is approximately `sqrt(variance) * 100` for log-normally distributed parameters. For example, `omega ETA_CL ~ 0.09` corresponds to ~30% CV.

The `(sd)` form is convenient when you're setting initial values from expected CV%, e.g. "I expect ~25% CV on CL" → `omega ETA_CL ~ 0.25 (sd)`. The fit result records which form you used so that downstream printers can annotate the estimate with `[initial specified as SD]`.

Block omega (`block_omega (...) = [...]`) is variance-only — `(sd)` is not accepted there because the lower-triangle list mixes variances and covariances and a single tag would be ambiguous.

### Block omega (correlated random effects)

To estimate correlations between random effects, use `block_omega`:

```
block_omega (NAME1, NAME2, ...) = [lower_triangle_values]
```

The values are the lower triangle of the covariance matrix, specified row-wise. For a 2x2 block:

```
block_omega (ETA_CL, ETA_V) = [var_CL, cov_CL_V, var_V]
```

For a 3x3 block:

```
block_omega (ETA_CL, ETA_V, ETA_KA) = [var_CL, cov_CL_V, var_V, cov_CL_KA, cov_V_KA, var_KA]
```

#### Visualising the lower triangle

For an N×N block you supply `N·(N+1)/2` values, walked row by row through the lower triangle (row 1 has 1 value, row 2 has 2, …, row N has N). Drawing it out as a matrix makes the layout obvious:

```
          ETA_1   ETA_2   ETA_3   ETA_4
        ┌                                ┐
ETA_1   │  v[0]                          │
ETA_2   │  v[1]   v[2]                   │
ETA_3   │  v[3]   v[4]    v[5]           │
ETA_4   │  v[6]   v[7]    v[8]    v[9]   │
        └                                ┘
```

The diagonal entries are the variances; the off-diagonals are the covariances (which the optimiser estimates — set them to `0.0` as initial values and the fit will recover correlations from the data).

#### Converting diagonal omegas to a block

A common pattern: you have all-diagonal omegas and want to replace them with a single fully-correlated block. Each `omega` variance becomes a diagonal entry; off-diagonals start at `0.0`. For example:

```
omega ETA_CL ~ 0.1
omega ETA_V1 ~ 0.1
omega ETA_Q  ~ 0.1
omega ETA_V2 ~ 0.1
```

becomes a 4×4 block (10 values: `1+2+3+4`):

```
block_omega (ETA_CL, ETA_V1, ETA_Q, ETA_V2) = [
  0.1,
  0.0, 0.1,
  0.0, 0.0, 0.1,
  0.0, 0.0, 0.0, 0.1
]
```

(Multi-line and single-line layouts are equivalent — line breaks inside `[ ... ]` are ignored.) Reading the rows: row 1 is `var(CL)`, row 2 is `cov(CL,V1) var(V1)`, row 3 is `cov(CL,Q) cov(V1,Q) var(Q)`, row 4 is `cov(CL,V2) cov(V1,V2) cov(Q,V2) var(V2)`.

Block omega regularises poorly-identified parameters by letting them share variability — see the [optimisation FAQ](../faq.md) for a worked example where switching from diagonal to `block_omega(3)` rescues a fit that diagonal-omega couldn't converge.

#### Mixing diagonal and block

You can mix diagonal and block omega specifications. Diagonal omegas specify uncorrelated random effects, while block omegas estimate the full covariance sub-matrix:

```
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
omega ETA_KA ~ 0.40
```

This estimates a 3x3 omega where ETA_CL and ETA_V are correlated (2x2 block), but ETA_KA is uncorrelated with both.

### Declaration order

The order of `omega` and `block_omega` lines in the `[parameters]` block determines the ETA indexing throughout the model: in the omega matrix and all output. For example:

```
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
omega ETA_KA ~ 0.40
```

produces ETA order `[ETA_CL, ETA_V, ETA_KA]` (indices 1, 2, 3), while:

```
omega ETA_KA ~ 0.40
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
```

produces `[ETA_KA, ETA_CL, ETA_V]` (indices 1, 2, 3). The `[individual_parameters]` block should list assignments in the same order for clarity, though the parameter mapping is by name, not position.

## Kappa (Inter-Occasion Variability)

Inter-Occasion Variability (IOV) is declared with `kappa` (independent per-parameter) or `block_kappa` (correlated across parameters). Kappa parameters must be paired with `iov_column` in `[fit_options]` and an occasion column in the dataset.

### Diagonal kappa — Option A

```
kappa NAME ~ value              # value interpreted as variance (default)
kappa NAME ~ value (sd)         # value interpreted as standard deviation
kappa NAME ~ value FIX
```

Each `kappa` line adds one diagonal element to the IOV omega matrix. Occasions are independent. The `(sd)` annotation is accepted with the same semantics as for `omega`. `block_kappa` is variance-only.

Example:
```
kappa KAPPA_CL ~ 0.05
kappa KAPPA_V  ~ 0.03
kappa KAPPA_CL ~ 0.02 FIX
```

### Block kappa — Option B (correlated IOV)

```
block_kappa (NAME1, NAME2, ...) = [lower_triangle_values]
block_kappa (NAME1, NAME2, ...) = [lower_triangle_values] FIX
```

Mirrors the `block_omega` syntax — values are the lower triangle of the IOV covariance matrix, row-wise. Use this when IOV effects are expected to be correlated (e.g. correlated occasion shifts in CL and V).

Example (2×2 block):
```
block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
```

where `0.05` = Var(KAPPA_CL), `0.01` = Cov(KAPPA_CL, KAPPA_V), `0.03` = Var(KAPPA_V).

### Using kappas in individual parameters

Reference kappa names exactly like BSV etas in `[individual_parameters]`:

```
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V  + KAPPA_V)
  KA = TVKA * exp(ETA_KA)           # no IOV on absorption
```

Kappas can be combined freely — a parameter can carry BSV only, IOV only, or both.

### Mixed diagonal and block kappa

You can mix `kappa` (uncorrelated) and `block_kappa` (correlated) declarations in the same model:

```
block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
kappa KAPPA_KA ~ 0.10
```

A name may not appear in both `kappa` and `block_kappa` — this is a parse error.

### Declaration order

Kappa declaration order determines the IOV omega matrix layout and the kappa column order in output. Kappas follow all BSV etas in the internal indexing: if a model has 3 BSV etas and 2 kappas, the kappas sit at indices 4 and 5.

## Parameter-level correlation in output

When a `block_omega` (or `block_kappa`) is estimated, ferx reports a **parameter-level correlation** for each off-diagonal pair in the console and YAML output. This differs from the eta-level (normal-scale) correlation `ω_ij / √(ω_ii · ω_jj)`:

| Both etas lognormal (`THETA * exp(ETA)`) | `(exp(ω_ij) − 1) / √((exp(ω_ii) − 1)(exp(ω_jj) − 1))` |
|---|---|
| Both etas additive (`THETA + ETA`) | `ω_ij / √(ω_ii · ω_jj)` (same as eta-level) |
| Mixed or complex expressions | Falls back to eta-level; a warning is added to `FitResult.warnings` |

The formula for lognormal pairs is the standard bivariate lognormal identity and reflects the correlation between the actual PK/PD parameters (e.g. CL and V) rather than their underlying normal variates.

The result is exposed as `FitResult.omega_param_corr` (BSV) and `FitResult.omega_iov_param_corr` (IOV), and is used wherever ferx prints a `corr` or `correlation` value for a block omega pair.

## Sigma (Residual Error)

```
sigma NAME ~ value              # value interpreted as variance (default)
sigma NAME ~ value (sd)         # value interpreted as standard deviation
sigma NAME ~ value (variance)   # explicit no-op equivalent to no annotation
```

- **NAME**: Residual error parameter name (referenced in `[error_model]`)
- **value**: Initial value. Default scale is **variance**, matching omega. Append `(sd)` to specify a standard deviation; the parser converts it to the internal representation. This unifies the user-facing scale across omega and sigma — see [issue #56](https://github.com/FeRx-NLME/ferx-core/issues/56).

Example (variance scale, default):
```
sigma PROP_ERR ~ 0.0004     # variance 0.0004  →  SD = 0.02  →  2% CV
sigma ADD_ERR  ~ 1.0        # variance 1.0     →  SD = 1.0
```

Equivalent declarations using SD coding:
```
sigma PROP_ERR ~ 0.02 (sd)
sigma ADD_ERR  ~ 1.0  (sd)
```

The interpretation of sigma's role in the residual-error model depends on the error model:

| Error Model | Sigma component |
|-------------|---------------|
| Additive | Variance (or SD with `(sd)`) of additive error |
| Proportional | Variance (or SD with `(sd)`) of the proportional coefficient |
| Combined | First sigma = proportional coefficient, second = additive component |

> **Migration note** (pre-issue-#56 models): `sigma NAME ~ value` was previously interpreted as a standard deviation. The new default is **variance**, so a pre-#56 value `v` becomes either `v² (variance)` or `v (sd)`. The `examples/` directory uses the `(sd)` form to preserve the prior initial values verbatim.

## Complete Examples

Diagonal omega (no correlations):
```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01
```

Block omega (correlated CL and V):
```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01
```
