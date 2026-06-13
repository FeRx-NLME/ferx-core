# Derived Columns (`[derived]`)

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

The optional `[derived]` block defines extra columns that are computed after the fit and appended to the sdtab output. Each line has the form:

```
name = expression
```

where `name` becomes a new sdtab column and `expression` is evaluated for every observation row.

## Quick example

```
[derived]
  CLi  = CL                          # individual CL echoed directly
  AUC  = integral(IPRED, from=0, to=24)   # AUC 0-24 from predicted curve
  Cmax = max(IPRED)                   # subject-level maximum prediction
  Tmax = tmax(IPRED)                  # time of maximum
```

## Available variables

| Name | Meaning |
|------|---------|
| Individual-parameter names (`CL`, `V`, …) | EBE-derived value for the subject |
| Theta names (`TVCL`, `TVV`, …) | Final population estimate |
| Eta names (`ETA_CL`, …) | Subject EBE |
| Covariate names (`WT`, `AGE`, …) | Subject covariate |
| `IPRED` | Individual prediction at the row's time point |
| `PRED` | Population prediction (eta = 0) at the row's time point |
| `DV` | Observed value |
| `TAFD` | Time after first dose |
| `TAD` | Time after most recent dose (SS-aware) |
| `TIME` | Nominal time |
| `compartments[i]` | Amount/concentration in ODE compartment `i` (0-based) or equivalent analytical state |
| *state name* (ODE) | Named ODE state variable directly, e.g. `Ce`, `depot`, `A_central` |

## Operators and functions

Standard arithmetic (`+`, `-`, `*`, `/`), comparison (`<`, `>`, `<=`, `>=`, `==`, `!=`), and logical (`&&`, `||`, `!`) operators are supported. Use the `mod` keyword for modulo (`a mod b`). As well as:

| Function | Description |
|----------|-------------|
| `exp(x)` | Exponential |
| `log(x)` | Natural logarithm |
| `sqrt(x)` | Square root |
| `abs(x)` | Absolute value |
| `floor(x)` / `ceil(x)` / `round(x)` | Rounding |
| `x ^ y` | Power (use `^`, not a function call) |
| `max(expr)` | Subject-level maximum of `expr` over all observations |
| `min(expr)` | Subject-level minimum |
| `tmax(expr)` | Time at which `expr` is maximised |
| `integral(expr, from=t0, to=t1)` | AUC from `t0` to `t1`; obs times for DV, fine grid (500 pts) for IPRED |
| `integral(expr, from=t0, to=t1, step=dt)` | As above with grid step `dt` hours (IPRED only; `step=` is ignored for DV) |
| `integral(expr, cond, from=t0, to=t1)` | As above, only time points where `cond` is true contribute |
| `integral(expr, window=P)` | Periodic AUC: one value per dosing window of length `P` hours |
| `integral(expr, window=P, anchor=A)` | Periodic AUC with windows starting at `A` (default 0) |
| `integral(expr, cond, window=P, step=dt)` | Periodic AUC, filtered by `cond`, with grid step `dt` |

The constant `MACHEPS` (machine epsilon, ≈ 2.22 × 10⁻¹⁶) is also available.

## Computation kinds

### Per-row (`PerRow`)

The default: the expression is evaluated independently at each observation time point and written to the corresponding sdtab row.

```
CLi = CL        # same value every row for a subject; still PerRow
```

### Aggregate (`Aggregate`)

Using `max(expr)`, `min(expr)`, or `tmax(expr)` produces a single value per subject that is broadcast to all of that subject's rows.

```
Cmax = max(IPRED)
Tmax = tmax(IPRED)
```

A filter can be applied as the second argument inside the parentheses:

```
CmaxAfter12 = max(IPRED, TIME > 12)
Ctrough      = min(IPRED, TAD < 1e-10)
CmaxD14      = max(IPRED, TAFD >= 312 && TAFD < 336)
```

The filter uses the same expression language. If no observation passes the filter the result is `NaN`.

### Integral (`Integral`)

`integral(expr, from=t0, to=t1)` computes the trapezoidal area under `expr` over the half-open interval `[t0, t1]`.

- When `expr` references `DV`, the observed values at the matching observation times are used.
- When `expr` references `IPRED`, values are evaluated at the observation times within the window. A `step=dt` keyword argument causes the integrand to be evaluated on a uniform grid of spacing `dt` using nearest-neighbour IPRED approximation; the `step=` argument is silently ignored for DV-based integrals (warning `W_DERIVED_STEP_IGNORED`).

```
AUC24 = integral(IPRED, from=0, to=24)
AUC24_obs = integral(DV, from=0, to=24)
AUC24_grid = integral(IPRED, from=0, to=24, step=0.5)
```

> **Limitation — obs-based time-above-threshold:** When using `integral(1.0, IPRED > threshold, ...)` without a `step=` argument, only observation time points are evaluated. If the design has sparse sampling, the trapezoidal rule can miss brief periods above threshold between observations. Use `step=` (e.g. `step=0.1`) to force grid evaluation at the cost of approximating IPRED via nearest-neighbour interpolation:
>
> ```
> TAM_TAU = integral(1.0, IPRED > MEC, window=24, anchor=0, step=0.1)
> ```

## Naming rules

A `[derived]` name must **not** clash with:

- Built-in sdtab columns: `ID`, `TIME`, `DV`, `PRED`, `IPRED`, `CWRES`, `IWRES`, `EBE_OFV`, `N_OBS`, `TAFD`, `TAD`, `CENS`, `OCC`, `CMT`
- Any theta, eta, or individual-parameter name in the model

Shadowing a **covariate** name is permitted (the covariate column is replaced) but will emit a `W_DERIVED_COVARIATE_SHADOW` warning.

## Referencing earlier derived columns

A derived expression may reference the name of any **preceding** derived column defined in the same `[derived]` block (sequential scoping). Forward references are not allowed.

```
[derived]
  CLi  = CL
  VDSS = CLi / 0.693    # ok — CLi defined on the line above
```

## Compartment states (ODE and analytical models)

After the fit, the state of every model compartment is available in `[derived]` expressions. There are two equivalent syntaxes:

- **Subscript**: `compartments[i]` — 0-based index into the state vector.
- **Named** (ODE models only): the ODE state variable name directly, e.g. `Ce`, `depot`, `A_central`.

### ODE example — effect compartment

```
[odes]
  dA_central/dt = -CL/V * A_central
  dCe/dt        = Ke0 * (A_central/V - Ce)

[derived]
  Ce_idx   = compartments[1]   # effect compartment by index
  Ce_named = Ce                # same value, by ODE state name
  AUC_Ce   = integral(Ce, from=0, to=24)
```

`Ce_idx` and `Ce_named` will be identical. Both refer to the raw ODE state
`u[1]` — whatever units the ODE produces.

### Analytical model compartment layout

Analytical PK models expose a fixed compartment layout:

| Model | `compartments[0]` | `compartments[1]` | `compartments[2]` | `compartments[3]` |
|-------|-------------------|-------------------|-------------------|-------------------|
| `one_cpt_iv` | `central` (conc) | — | — | — |
| `one_cpt_oral` | `depot` (amount) | `central` (conc) | — | — |
| `two_cpt_iv` | `central` (conc) | `peripheral` (conc) | — | — |
| `two_cpt_oral` | `depot` (amount) | `central` (conc) | `peripheral` (conc) | — |
| `three_cpt_iv` | `central` (conc) | `peripheral1` (conc) | `peripheral2` (conc) | — |
| `three_cpt_oral` | `depot` (amount) | `central` (conc) | `peripheral1` (conc) | `peripheral2` (conc) |

Named access works for analytical models too:

```
[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[derived]
  C_periph = peripheral           # named access
  C_periph_idx = compartments[1] # same value, by index
```

> **Validation.** The analytical peripheral/depot states are cross-checked
> against NONMEM 7.5.1. For 2-cpt IV/oral and 3-cpt IV/oral models, evaluating
> at fixed θ with η = 0 reproduces NONMEM's tabled compartment amounts `A(i)`
> (rescaled by the matching volume, since ferx reports central/peripheral
> compartments as concentrations) to within `1e-4` relative. See
> `tests/compartment_states_nonmem.rs` and the control files in
> `tests/nonmem/` (`iv2`, `iv3`, `oral2`, `oral3`).

### Scaling

`compartments[i]` is **never scaled**. It is the raw solver state or analytical
formula output:

- For analytical models: depot compartments are in **amounts** (dose units); central
  and peripheral compartments are in **concentrations** matching `IPRED`.
- For ODE models: the value is whatever the ODE produces. If your ODE tracks amounts
  (`dA/dt = ...`) then `compartments[i]` is an amount; if it tracks concentrations
  (`dC/dt = ...`) it is a concentration.

If a `[scaling]` block is active, `IPRED` is the scaled output but `compartments[i]`
for the observed compartment remains unscaled. Compute the scaled value explicitly if
needed:

```
[derived]
  C_periph_scaled = compartments[1] / V2
```

### `integral` over compartment states

When an `integral(...)` integrand references `compartments[i]` or a named ODE state
variable, ferx **re-runs the ODE solver** (or analytical formula) at the grid points
rather than interpolating from stored per-observation states. This gives an exact
result at the cost of a second solver pass per subject — typically negligible for
post-fit diagnostics.

```
[derived]
  AUC_Ce = integral(compartments[1], from=0, to=24, step=0.5)
```

> **Named state variables in `integral` (ODE models):** ferx detects whether the
> integrand references a compartment by name at parse time. For this detection to
> work, the model must have an `[odes]` block (named access is not available for
> integrals in analytical models — use the `compartments[i]` subscript form instead).

### Name priority

When a named ODE state variable shares a name with a covariate, individual
parameter, or other derived column:

1. Built-in columns (`IPRED`, `TIME`, `DV`, `TAFD`, `TAD`, etc.) — highest priority
2. Individual parameter names (`CL`, `V`, `KA`, …)
3. Theta / eta names
4. Covariate names
5. ODE state variable names — lowest priority

If a state name shadows a covariate you will receive `W_DERIVED_COVARIATE_SHADOW`.
If a state name coincides with an individual parameter, the **parameter value** wins
and the compartment state is inaccessible by name — use `compartments[i]` instead.

### Limitations (Phase 1)

- `compartments[i]` requires a **literal integer index**. Dynamic indexing
  (`compartments[N]` where `N` is a parameter) is not supported and will produce a
  parse error.
- Named access is not available for integrals in **analytical models**. Use
  `compartments[i]` in `integral(...)` for analytical models.
- **SDE models** (`[diffusion]` block): `compartments[i]` returns the
  **deterministic ODE state** at each observation time, not the EKF-filtered state.
  For most post-fit derived quantities this is sufficient, but values will differ
  from the EKF posterior mean when process noise is large.
- **IOV subjects**: `compartments[i]` evaluates to `NaN` — compartment states are
  not yet available on the IOV prediction path
  (`W_DERIVED_CMT_IOV_UNSUPPORTED` warning is emitted). Scalar references —
  individual parameters (`CL`, `V`, …), `ETA`/`KAPPA`, `IPRED`, `TAD`, etc. — *do*
  honour each observation's occasion kappa, so `[derived]` expressions and
  `[output]` columns built from them are occasion-correct; only compartment-state
  access is unavailable.
- **Analytical models with time-varying covariates**: `compartments[i]` evaluates
  to `NaN` for affected subjects — states would require time-varying PK params that
  the superposition path does not support
  (`W_DERIVED_CMT_TV_ANALYTICAL` warning is emitted).
- **ODE models with time-varying covariates**: `compartments[i]` is approximate —
  states are computed with first-observation PK parameters while `IPRED` uses the
  correct per-event PK (`W_DERIVED_CMT_TV_ODE` warning is emitted).
