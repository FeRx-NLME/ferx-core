# Derived Columns (`[derived]`)

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
