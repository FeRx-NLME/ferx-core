# ODE Models

For pharmacokinetic models without analytical solutions (e.g., saturable elimination, target-mediated drug disposition), FeRx provides an ODE solver.

## Structural Model Declaration

```
[structural_model]
  ode(obs_cmt=OBSERVABLE_COMPARTMENT, states=[state1, state2, ...])
```

- **obs_cmt**: The compartment whose concentration is observed (matched to DV)
- **states**: List of state variable names (compartments)

## ODE Equations

The `[odes]` block defines the right-hand side of the ODE system:

```
[odes]
  d/dt(state_name) = expression
```

Expressions can reference:
- State variables by name
- Individual parameters defined in `[individual_parameters]`
- Arithmetic operators and functions (`exp`, `log`, `sqrt`, etc.)
- Conditional logic with the same `if (cond) { ... } else { ... }` and inline
  `if (cond) expr else expr` syntax described in
  [Individual Parameters](individual-parameters.md). For example, you can
  switch between linear and saturable elimination based on the central
  amount:

  ```
  [odes]
    d/dt(depot)   = -KA * depot
    if (central > KM_THRESHOLD) {
      d/dt(central) = KA * depot - VMAX * central / (KM + central)
    } else {
      d/dt(central) = KA * depot - CL_LIN * central
    }
  ```

  Each `d/dt(state)` reachable from any branch counts as defined; states
  that aren't assigned in the firing branch this step receive a derivative
  of `0`.

## Example: Michaelis-Menten Elimination

A one-compartment oral model with saturable (Michaelis-Menten) elimination:

```
[parameters]
  theta TVVMAX(10.0, 0.1, 1000.0)
  theta TVKM(2.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_VMAX ~ 0.09
  omega ETA_V    ~ 0.04

  sigma PROP_ERR ~ 0.1

[individual_parameters]
  VMAX = TVVMAX * exp(ETA_VMAX)
  KM   = TVKM
  V    = TVV * exp(ETA_V)
  KA   = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) = KA * depot / V - VMAX * central / (KM + central)

[error_model]
  DV ~ proportional(PROP_ERR)
```

## Solver Details

FeRx uses a Dormand-Prince RK45 adaptive solver:

| Setting | Value |
|---------|-------|
| Method | Explicit Runge-Kutta 4(5) |
| Absolute tolerance | 1e-6 |
| Relative tolerance | 1e-4 |
| Max steps | 10,000 |
| Initial step size | 0.1 |
| Minimum step size | 1e-12 |

The solver automatically adapts step sizes based on local error estimates.

## Dose Handling

- **Bolus doses**: Applied as instantaneous state changes at dose times. The dose amount is added to the target compartment: `state[cmt] += amt`
- **Infusion doses** (`RATE > 0`): Treated as a continuous zero-order input. The integrator's timeline is broken at the infusion's end (`time + amt/rate`), and `+rate` is added to the target compartment's derivative for every segment fully spanned by the infusion. Overlapping infusions on the same compartment sum their rates
- **Compartment indexing**: Compartments are 1-indexed in the data file (`CMT=1` corresponds to the first state in the `states` list)
- **Multiple doses**: The ODE is integrated in segments between dose events, with state discontinuities at each bolus

## Stochastic ODE Models (SDE)

To model within-subject system noise that accumulates between observations, add
a `[diffusion]` block to your ODE model.  See [Stochastic Differential
Equations](diffusion.md) for a full description, worked example, and comparison
with sigma and omega.

## Limitations

- The observable compartment contains the amount (not concentration). Divide by volume in the ODE equations if needed
- Steady-state (`SS=1`) is not directly supported for ODE models
- SDE (`[diffusion]`) is not compatible with SAEM or the autodiff gradient path
