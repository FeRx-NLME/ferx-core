# Scaling

The optional `[scaling]` block declares how the structural model's raw
output maps to the observed `DV`. It exists so the user does not have to
fold unit conversion or amount-to-concentration arithmetic into the
structural model itself — keeping `[odes]` and `[structural_model]`
readable, and making mixed-unit data (e.g. data in ng/mL when the model
thinks in mg/L) straightforward.

The convention is **divisive**: `pred_scaled = pred_raw / scale`. This
matches the natural reading of `obs_scale = V/1000` as *"divide amount by
V/1000 to get concentration in the user's units."*

Three forms are supported. Each is optional; omitting `[scaling]` keeps
the historical "raw prediction equals DV" behaviour.

## Form A — scalar divisor

Use for fixed unit conversion (e.g. mg/L → mg/mL is a constant 1000).

```ferx
[scaling]
  obs_scale = 1000
```

Applies to analytical PK models and ODE models alike: every prediction
is divided by the constant before reaching the residual error model.

## Form B — expression divisor

Use when the scale depends on theta, eta, or a covariate.

```ferx
[scaling]
  obs_scale = WT / 70
```

In Phase 1, expressions may reference:

- thetas (e.g. `TVV`),
- etas (e.g. `ETA_CL`),
- covariates (e.g. `WT`, `CR`).

Phase 1 **does not** support references to individual parameters (e.g.
`obs_scale = 1000 / V`) — the parser will reject these with a clear
error. Use the underlying theta (`TVV`) for now, or switch to Form C if
you need an individual-parameter-driven scale on an ODE model.

The scale is evaluated **once per subject** with subject-level
covariates (matching the no-TV path). Time-varying covariate support for
expression scales is a Phase 1.5 follow-up.

## Form C — explicit output expression (ODE only)

Use when the ODE state is held as an amount and the observation is a
concentration. Form C replaces the default `obs_cmt` readout entirely.

```ferx
[structural_model]
  ode(states=[depot, central])     # no obs_cmt — Form C provides it

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot - CL/V * central   # central holds amount

[scaling]
  y = central / V
```

The right-hand side may reference state names (`depot`, `central`),
individual parameters (`CL`, `V`, `KA`), and covariates. Thetas and
etas are **not** directly in scope — they are folded into individual
parameters by `[individual_parameters]` before `y` is evaluated.

Form C is only valid for ODE models. The parser rejects `y = ...` on
analytical models with a clear error.

## Comparison with NONMEM and nlmixr2

| Need | NONMEM | nlmixr2 | ferx |
|---|---|---|---|
| Scalar unit conversion | `S1 = 1000` | (multiplier in `cmt`/`f`) | `[scaling] obs_scale = 1000` |
| Amount-state ODE with concentration DV | `S2 = V/1000` plus `Y = A(2)/S2` | `cmt(central); f = central/V/1000` | `[scaling] y = central / (V/1000)` |

The ferx form is divisive by convention, so an `obs_scale = V/1000`
reads as "divide raw by V/1000" — matching NONMEM's `S2`.

## Interaction with gradients (AD vs FD)

- Scalar `obs_scale = K` works with both `gradient = ad` and
  `gradient = fd`. The constant is threaded through the AD entry points
  as a `Const` argument.
- Expression `obs_scale = <expr>` is supported only with
  `gradient = fd` in Phase 1. The parser rejects the combination of
  expression scaling with AD gradients and prints the required flag.

## Interaction with SDE / `[diffusion]`

The EKF / SDE path requires a single observable compartment index for
the Kalman update. SDE models can use Forms A and B (post-multiplied at
the end of the prediction path) but cannot use Form C — the parser
rejects SDE models that omit `obs_cmt` from `[structural_model]`.

## Multi-analyte (forward compatibility)

`[scaling]` is per-model in Phase 1. When ferx adds CMT-keyed scaling
(NONMEM's `if (CMT==2) Y = A2/V2`-style routing), the same block will
grow `y[CMT=1] = <expr>` / `y[CMT=2] = <expr>` and
`obs_scale[CMT=...]` variants. Existing single-form usage will continue
to work unchanged.
