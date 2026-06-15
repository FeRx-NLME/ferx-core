# Built-in Absorption Models

ferx provides built-in **absorption input-rate functions** for ODE models — a
dose-driven appearance rate `R_in(tad)` that is added into the dosing
compartment instead of treating the dose as an instantaneous bolus. They let you
write a flexible absorption *shape* (transit-chain delay, etc.) as a single
intrinsic in the `[odes]` block, rather than hand-coding a chain of physical
transit compartments.

Phase 0 ships the **transit-compartment** model; more input-rate models
(inverse-Gaussian, Weibull, zero-order families) are planned — see
`plans/absorption-models.md`.

## Transit-compartment absorption — `transit(n, mtt)`

The Savic et al. (2007) transit-compartment model with a **continuous** number
of compartments `n`:

\\[
R_\text{in}(t_\text{ad}) = F\cdot\text{Dose}\;\cdot\;
\text{KTR}\;\frac{(\text{KTR}\cdot t_\text{ad})^{n}\,e^{-\text{KTR}\cdot t_\text{ad}}}{\Gamma(n+1)},
\qquad \text{KTR}=\frac{n+1}{\text{MTT}}
\\]

where `tad` is time after the dose, `MTT` is the mean transit time, and `KTR` is
the transit rate constant. The input integrates to the full dose
(`∫₀^∞ R_in dt = F·Dose`), and `R_in = 0` for `tad ≤ 0`. With `n = 0` it reduces
to a first-order (Bateman) input with rate `1/MTT`. Because `n` is continuous,
the absorption *shape* itself is an estimable parameter.

### Syntax

Add `transit(...)` to the right-hand side of the depot compartment's ODE:

```text
[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV  * exp(ETA_V)
  KA  = TVKA
  MTT = TVMTT          # mean transit time (h)
  NTR = TVN            # number of transit compartments (continuous)

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = transit(n=NTR, mtt=MTT) - KA*depot
  d/dt(central) = KA*depot/V - CL/V*central
```

- **Arguments are named**: `n=<param>` and `mtt=<param>`. Both must be declared
  `[individual_parameters]` (so they fold in theta/eta/covariates and can carry
  IIV/IOV exactly like any other individual parameter).
- `transit(...)` may appear **once** on a `d/dt(...)` line and must not be
  scaled or combined with other input-rate terms — it is the chain input, not an
  ordinary expression. (It is split out of the RHS at parse time and evaluated
  by the engine with dose context; the rest of the RHS — here `- KA*depot` — is
  the disposition you write normally.)

### Dose routing — the dose feeds the function, not a bolus

A dose into a transit compartment is delivered **entirely** through `R_in` over
time. ferx therefore does **not** also add it as an instantaneous bolus — doing
so would double-count the dose. This mirrors NONMEM, where the transit dose
compartment carries `F1 = 0` for the bolus while the analytical/`$DES` input
delivers the mass.

- **Bioavailability `F`** scales the delivered mass (`Dose = F · AMT`), exactly
  as for ordinary doses — see [Bioavailability](ode-models.md#bioavailability).
  Do **not** multiply `transit(...)` by `F` in the RHS.
- **Lagtime** shifts `tad` (the input starts at `dose time + lagtime`).
- **Multiple doses** superpose: `R_in` is summed per dose. With **IOV** on the
  transit parameters, an absorption tail that is still appearing when the next
  occasion begins is evaluated with the *current* occasion's `n`/`mtt` — exact
  when `II` exceeds the absorption window (the usual case) and for IIV-only
  models, approximate only for overlapping occasions.

### Parameter domains

The domain is `mtt > 0` and `n ≥ 0`. It is enforced in two places:

- **Typical values** (η = 0, per subject, so covariate relationships are
  included) are validated at fit time. A non-finite or out-of-domain typical
  value is rejected with `E_ABSORPTION_DOMAIN` — a clear error, not an opaque
  `NaN` fit failure. Constrain the parameter so it stays in range, e.g.
  `MTT = TVMTT * exp(ETA_MTT)` keeps `MTT > 0`.
- **Transient mid-fit excursions** are clamped. With an additive
  parameterisation (`MTT = TVMTT + ETA_MTT`), the inner EBE search or a
  finite-difference step can momentarily push `mtt ≤ 0` / `n < 0` even though the
  typical value is in range. There `R_in` is evaluated at the domain boundary
  (a finite value) rather than producing a `NaN` that would poison the objective.
  Because the converged optimum is interior, this clamp never affects reported
  estimates — it only keeps the optimiser numerically stable. (A log-normal
  parameterisation avoids the excursion entirely and is recommended.)

### Not yet supported (Phase 0)

These combinations are **rejected with a clear error** rather than silently
mis-modeled:

- **An infusion (`RATE>0`) into a transit compartment** (`E_ABSORPTION_RATE`) —
  the dose mass is delivered through `R_in`, computed from the dose *amount*, so
  an infusion rate on the same record would double-count it. Use a plain bolus
  record (`RATE=0`) into the absorption compartment; the transit chain provides
  the input-rate shape.
- **Steady-state dosing (`SS=1`) into a transit compartment** (`E_ABSORPTION_SS`)
  — periodic steady state with an in-progress absorption tail needs dedicated
  treatment. Expand the run-in with explicit dosing records instead.
- **A `[diffusion]` block (SDE/EKF) together with `transit()`**
  (`E_ABSORPTION_DIFFUSION`) — the EKF propagation does not yet carry the
  input-rate forcing.

> **Note:** these guards run at `fit()` time (and via `ferx check`), which is
> where model–data compatibility is validated. The lower-level `predict()` and
> `simulate()` entry points assume an already-checked model and do not re-run
> them, so run `ferx check` (or `fit()`) on a new model before relying on
> `predict`/`simulate` output.

## Worked example

[`examples/transit_savic.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/transit_savic.ferx)
is a complete one-compartment oral model with built-in transit absorption. Run
it on simulated data:

```bash
cargo run --release -- examples/transit_savic.ferx --simulate
```

For comparison,
[`examples/transit_2cpt.ferx`](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/transit_2cpt.ferx)
codes the same idea as three **explicit** transit ODE states (fixed integer
`n = 3`); the built-in `transit()` collapses that to one line with a single
continuous `n`. See [Transit Absorption](../examples/transit-absorption.md) for
the worked two-compartment example and diagnostics guidance.

## Numerical note

The transit input is integrated numerically through the ODE solver (the same
RHS-wrapper mechanism that injects `+rate` for infusions). An analytical
closed form for continuous-`n` transit (via the regularized incomplete gamma
function) is planned so 1-/2-compartment transit models can stay in the
analytical engine — see `plans/absorption-models.md`.
