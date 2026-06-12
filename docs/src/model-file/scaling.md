# Scaling

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

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

Expressions may reference:

- thetas (e.g. `TVV`),
- etas (e.g. `ETA_CL`),
- covariates (e.g. `WT`, `CR`),
- individual parameters declared in `[individual_parameters]` (e.g.
  `V`, `CL`) — these are resolved from a subject-static evaluation of
  `pk_param_fn` at scale-evaluation time, so `obs_scale = 1000 / V`
  uses the per-subject V (typical value times the EBE eta).

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
individual parameters (`CL`, `V`, `KA`), thetas (`TVCL`), etas
(`ETA_CL`), and covariates (`WT`). All five name classes are looked up
at evaluation time — states from the ODE solver, individual parameters
from the subject-static `pk_param_fn`, and thetas/etas/covariates from
the values supplied by the caller.

Form C is only valid for ODE models. The parser rejects `y = ...` on
analytical models with a clear error.

## Runtime behaviour on bad scales

If an expression scale (Form B or C) evaluates to a non-positive or
non-finite value at runtime — for example `WT / 70` when `WT` is missing
(reads 0) or `1 / (TVV - x)` near a singularity — every prediction for
that subject is set to `NaN`. The outer NLL then evaluates to `NaN` and
the optimizer rejects the step. This matches established NLM convention
(NONMEM's `OBJFN = NaN` → step rejection) and surfaces bad scales in
the per-subject diagnostics rather than silently producing a mis-scaled
fit.

## Comparison with NONMEM and nlmixr2

| Need | NONMEM | nlmixr2 | ferx |
|---|---|---|---|
| Scalar unit conversion | `S1 = 1000` | (multiplier in `cmt`/`f`) | `[scaling] obs_scale = 1000` |
| Amount-state ODE with concentration DV | `S2 = V/1000` plus `Y = A(2)/S2` | `cmt(central); f = central/V/1000` | `[scaling] y = central / (V/1000)` |

The ferx form is divisive by convention, so an `obs_scale = V/1000`
reads as "divide raw by V/1000" — matching NONMEM's `S2`.

## Interaction with gradients (AD vs FD)

All `[scaling]` variants on the **analytical PK path** support both
`gradient = ad` and `gradient = fd`:

| Form | AD | FD | Notes |
|---|---|---|---|
| Scalar `obs_scale = K` | ✓ exact | ✓ exact | The constant threads as a `Const` slice (one entry per obs). |
| Expression `obs_scale = <expr>` | ✓ subject-static | ✓ exact | See subject-static caveat below. |
| Per-CMT `obs_scale[CMT=N]` | ✓ subject-static | ✓ exact | One per-obs scale entry per observation, dispatched by `subject.obs_cmts[i]`. |
| Form C `y[…] = <expr>` (ODE only) | ✗ — forces `gradient = fd` | ✓ | Form C only exists on ODE models, where the AD path is not available regardless of scaling. |

**Subject-static caveat (Expression / per-CMT under AD).** The
per-observation scale array passed to the AD entry points is
materialised from a single subject-static `pk_param_fn` evaluation —
the same simplification documented for the FD path's `apply_scaling`.
This means AD treats the scale as constant w.r.t. eta. For the common
eta-independent scale (`WT/70`, `TVV/1000`, or `V` reading the EBE
value) the AD and FD gradients are identical. For the rare case of a
scale expression that explicitly references eta (e.g.
`obs_scale = exp(ETA_V)`), the AD gradient ignores that dependence
while FD captures it numerically — set `gradient = fd` if you need the
eta sensitivity in the gradient.

## Interaction with SDE / `[diffusion]`

In Phase 1, `[scaling]` is **not supported** on SDE models. The EKF /
Kalman update computes both the predicted mean and the prediction
covariance `p_obs` in the observation space, and the per-observation
`r_obs` callback evaluates the residual variance from that predicted
mean. Forms A/B post-multiply only the mean, so the EKF variance would
remain in the unscaled space — producing mis-scaled OFVs.

A correct SDE+scaling integration needs the scale factor threaded into
both the EKF `p_obs` propagation (scales by `1/K²`) and the residual
variance callback. That's a wider change deferred to Phase 1.5. Until
then, the parser rejects any `[scaling]` block on a model with a
`[diffusion]` block (Forms A, B, and C alike).

## Multi-analyte / per-CMT scaling

For models that observe multiple compartments (parent + metabolite,
sum-of-moieties, free vs. total, ...), specify a separate scale per
observed CMT using the `obs_scale[CMT=N]` (Forms A/B) or `y[CMT=N]`
(Form C) syntax. `N` is the 1-based CMT index from the data file's
`CMT` column.

```ferx
[scaling]
  obs_scale[CMT=1] = 1000    # parent in mg/L → mg/mL
  obs_scale[CMT=2] = 1       # metabolite already in target units
```

Form C (ODE) per-CMT:

```ferx
[structural_model]
  ode(states=[depot, parent, metab])

[scaling]
  y[CMT=1] = parent / V_parent
  y[CMT=2] = metab  / V_metab
```

**Coverage rule** — every CMT that has at least one observation in the
data must have a matching `[CMT=N]` entry. The parser only checks
syntax; the fit-time validation (run automatically at the top of
`fit()`) errors with a list of the missing CMTs:

```
[scaling]: per-CMT scaling is missing entries for observed CMTs [2, 3].
Every observed CMT must have an `obs_scale[CMT=N]` (or `y[CMT=N]` for ODE) entry.
```

**Mixing rule** — the uniform form (`obs_scale = K`) and the per-CMT
form (`obs_scale[CMT=N] = K`) are mutually exclusive within the same
group. The parser rejects mixing them so the user is explicit about
intent. The same rule applies to `y` and `y[CMT=N]`.

**Gradients** — per-CMT `obs_scale[CMT=N]` works with both
`gradient = ad` and `gradient = fd`. The AD path materialises a
per-observation scale array (one entry per obs in the subject) from a
subject-static `pk_param_fn` evaluation and passes it to the AD entry
points as a `Const` slice. See [AD interaction](#interaction-with-gradients-ad-vs-fd)
below for the subject-static caveat that applies to all expression-form
scales.

Form C per-CMT (`y[CMT=N] = <expr>`) still requires `gradient = fd` —
not because of the per-CMT dispatch but because Form C only exists on
ODE models, and the AD path requires the analytical PK path
(`tv_fn.is_some()`). The parser rejects `Form C + gradient = ad` with a
clear hint.
