# Lagtime

> **Maturity: stable** — see [Feature Maturity](../maturity.md) for what this means.

A `lagtime` PK parameter shifts the effective start of every dose record by
a fixed amount. This is the standard way to model delayed-onset oral
absorption: even after the dose is administered at clock time `t_dose`,
no drug enters the system until `t_dose + lagtime`.

`lagtime` is the ferx-core name for what NONMEM calls `ALAG`. The DSL
accepts both `lagtime=` (preferred) and `alag=` (alias) on the
`[structural_model]` line; both route to the same internal slot.

## DSL example

```
[parameters]
  theta TVCL(5.0,  0.1,  50.0)
  theta TVV(50.0,  5.0, 500.0)
  theta TVKA(1.5, 0.05,  20.0)
  theta TVLAGTIME(0.5, 0.001, 5.0)

  omega ETA_CL      ~ 0.09
  omega ETA_LAGTIME ~ 0.10

  sigma PROP_ERR ~ 0.10

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  LAGTIME = TVLAGTIME * exp(ETA_LAGTIME)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)

[error_model]
  DV ~ proportional(PROP_ERR)
```

This is exactly analogous to declaring `F` (bioavailability): a typical
value (optionally with a random effect) flows through
`[individual_parameters]` and lands in a dedicated PK slot.

## Semantics

For every dose record (bolus, infusion, or oral depot), the effective
arrival time is `dose.time + lagtime`. For infusions the duration is
unchanged — only the start (and therefore the end) shifts. An observation
before the lagged arrival reads `0`.

If `lagtime` is not declared, the slot defaults to `0.0` — existing models
behave identically to the pre-feature path.

`lagtime` is supported on:

- the analytical superposition path (predictions),
- the analytic `Dual2` sensitivity gradient used during `fit()`,
- the ODE path.

In the niche case of a time-varying-covariate subject combined with a
lagtime-bearing model, ferx-core silently falls back from the analytic gradient
to finite differences — correctness is preserved at a small performance cost.

## NONMEM equivalence

NONMEM's per-compartment `ALAG1`, `ALAG2`, ... are mapped to a single
`lagtime` slot in ferx-core, applied to all dose records of the model.
For typical PK models (one absorption compartment, doses into the depot)
this is equivalent to setting `ALAG1` on the depot compartment.

If both `lagtime=` and `alag=` are supplied on the same `[structural_model]`
line (or if both keys appear in a programmatically constructed PK
parameter map), `lagtime` wins. This is intentional but unlikely to be
intended; prefer using only one of the two names per model.

## Notes and caveats

- `lagtime` may carry a random effect like any other PK parameter
  (`LAGTIME = TVLAGTIME * exp(ETA_LAGTIME)`). The optimiser handles
  derivatives through the same machinery used for `CL`, `V`, `KA`, etc.
- Negative `lagtime` is not clamped — if a user writes an expression that
  yields a negative value, predictions will treat the dose as effective
  before its record time. Prefer a log-link (`exp(...)`) or any other
  parameterisation that keeps `lagtime ≥ 0`. ferx emits a warning in
  `FitResult.warnings` when the initial typical-value lagtime is negative.
- Steady-state doses (`SS=1`) combined with non-zero `lagtime` are
  fully supported. Under linear superposition the lagged SS curve at
  time `t` equals the un-lagged SS curve evaluated at `t - lagtime`,
  i.e. `C_ss(t, L) = C_ss(t - L, 0)`. See
  [Steady-State Doses](steady-state.md) for the full SS=1 reference.

See `examples/oral_with_lagtime.ferx` for a runnable model.
