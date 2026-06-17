# ODE Models

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

For pharmacokinetic models without analytical solutions (e.g., saturable elimination, target-mediated drug disposition), ferx-core provides an ODE solver.

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
- The reserved builtins `TIME`/`TAFD`/`TAD` (solver time axes) and `MACHEPS`
  (machine epsilon, `f64::EPSILON`)
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

Every name in an ODE expression must resolve to a declared state, an individual
parameter, an intermediate variable assigned earlier in the block, or one of the
reserved builtins `TIME`/`TAFD`/`TAD`/`MACHEPS`. A name that matches none of
these — a typo, an omitted parameter, or a covariate — is **rejected at parse time**
rather than silently read as `0.0`, the same structurally-broken-fit guard the
analytical `pk(...)` mappings apply. Covariates cannot be referenced directly in
an ODE RHS: pre-compute the covariate-dependent term in `[individual_parameters]`
and reference that variable here instead.

## Initial Compartment Amounts

By default every compartment starts at zero, and drug enters only through dose
records. To start a compartment at a non-zero amount — e.g. a pre-dose baseline
for an indirect-response / turnover model — declare an initial condition in the
`[odes]` block:

```
[odes]
  init(state_name) = expression
  d/dt(state_name) = expression
```

- The right-hand side is evaluated **once per subject** at the start of the
  record and may reference individual parameters (and therefore folds in
  `theta`, `eta`, and covariates through the `[individual_parameters]` layer).
  State names referenced in an `init` expression are treated as `0` (no drug
  is present yet).
- A name in an `init` expression that is not a declared state or individual
  parameter is rejected at parse time (it would otherwise be read as `0.0`).
- Compartments without an `init(...)` line start at zero, as before.
- This is the analogue of NONMEM's `A_0(n)`.

**Time-varying covariates.** Because the initial condition is a pre-record
baseline, it is evaluated a single time using the covariate values from the
subject's **first record**. If a covariate that feeds the `init` expression
changes later in the record, the initial amount is *not* re-evaluated — the
later covariate values affect `d/dt(...)` going forward (the system evolves
from the baseline), but the t=0 starting point is fixed by the first record's
covariates. For most models this is exactly what you want, since the baseline
represents the pre-dose steady state. If you need the starting amount to track
a covariate value observed mid-record, model it as a state driven by `d/dt`
rather than as an `init`.

A turnover model whose response variable sits at its baseline `KIN/KOUT`
before any perturbation:

```
[odes]
  init(response) = KIN / KOUT
  d/dt(response) = KIN - KOUT * response
```

**Interaction with system resets (EVID=3/4):** a reset re-applies the `init`
expression to initialized compartments (returning them to baseline) and zeros
all other compartments — so a reset behaves like the start of a fresh episode.
See [Data Format](../data-format.md) for reset rows.

Note one deliberate asymmetry with the start-of-record seeding described above:
the re-applied baseline at a reset is evaluated with the covariate values in
effect **at the reset time**, not the first record's. With time-varying
covariates this means the post-reset baseline reflects the most recent
covariate values — appropriate for a "fresh episode" that starts under current
conditions — whereas the very first baseline uses the first record's
covariates. For time-constant covariates the two are identical.

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

ferx-core uses a Dormand-Prince RK45 adaptive solver:

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

- **Bolus doses**: Applied as instantaneous state changes at dose times. The dose amount, scaled by bioavailability (`F · AMT`), is added to the target compartment (the state at `CMT − 1`, since `CMT` is 1-based — see indexing below)
- **Infusion doses** (`RATE > 0`): Treated as a continuous zero-order input. The integrator's timeline is broken at the infusion's end (`time + amt/rate`), and `F · RATE` is added to the target compartment's derivative for every segment fully spanned by the infusion. Overlapping infusions on the same compartment sum their rates
- **Compartment indexing**: Compartments are 1-indexed in the data file (`CMT=1` corresponds to the first state in the `states` list)
- **Multiple doses**: The ODE is integrated in segments between dose events, with state discontinuities at each bolus
- **Built-in absorption input rates**: A dose can instead be delivered as a dose-driven appearance rate `R_in(tad)` (e.g. transit-compartment absorption) added into the depot over time — see [Built-in Absorption Models](absorption.md)

### Bioavailability

If your `[individual_parameters]` block declares an `F` parameter, the ODE engine
applies it **when the dose enters the compartment** — the dosing compartment is
loaded with `F · AMT` (and an infusion rate with `F · RATE`) — exactly like
NONMEM's `F1` and like ferx's analytical PK functions. Write the depot's
elimination as the plain `KA · depot` and **do not** multiply by `F` anywhere in
the right-hand side, or bioavailability is applied twice. `F` defaults to `1.0`
when not declared, so IV and non-bioavailability models are unaffected.

```text
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = inv_logit(logit(THETA_F) + ETA_F)   # F is applied at dose entry

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot / V - CL/V * central   # no F here
```

> ⚠️ **Migration note.** Earlier versions of ferx added the *full* dose to the
> compartment and required `F` to be folded into the absorption flux (e.g.
> `d/dt(central) = F * KA * depot / V - …`). That `F` must now be removed from
> the right-hand side — otherwise it is applied both at dose entry **and** in the
> flux, giving an effective bioavailability of `F²`.

The name `F` (any case) is what flags a parameter as bioavailability and routes
it to the dosing compartment. If you need a fraction-like quantity inside the
RHS that is *not* bioavailability, give it a different name.

See `examples/bioavailability_ode.ferx` for a complete worked model.

### Compartment-indexed bioavailability and lag (`Fn` / `ALAGn`)

When a model is dosed into **more than one compartment**, bioavailability and
absorption lag can differ by route. Mirroring NONMEM's `F1`/`F2` and
`ALAG1`/`ALAG2`, name an individual parameter `F{n}` or `ALAG{n}` (equivalently
`LAGTIME{n}`), where `n` is the 1-based dose compartment:

```text
[individual_parameters]
  CL    = TVCL * exp(ETA_CL)
  V     = TVV
  F1    = inv_logit(THETA_F1)   # bioavailability for doses into compartment 1
  F2    = inv_logit(THETA_F2)   # ... and into compartment 2
  ALAG2 = TVLAG2                # absorption lag for compartment-2 doses only
```

- A dose into compartment `n` uses `F{n}` / `ALAG{n}` if declared.
- A **bare** `F` / `lagtime` (no index) remains the all-compartment default, so
  existing single-route models are unchanged. An indexed value overrides the
  bare default for its compartment only; compartments without an indexed entry
  fall back to the bare value (or to `F = 1`, `lag = 0`).
- The index must refer to a compartment the model actually has — `F3` on a
  two-state model is a parse error, not a silently-ignored parameter.
- Each declared `Fn`/`ALAGn` occupies one of the seven spare slots in the
  fixed 16-slot PK parameter layout (shared with other ODE structural
  parameters). Declaring the full set for many compartments can exhaust them;
  if so, `ode_param_slots` reports a clear "too many individual parameters"
  error rather than failing silently.

> ⚠️ **`F{n}` / `ALAG{n}` / `LAGTIME{n}` are reserved names** (just like the
> bare `F` / `lagtime` above, and exactly as in NONMEM). On an ODE model,
> declaring an individual parameter with one of these names binds it as
> compartment `n`'s bioavailability / lag and applies it to **every** dose into
> compartment `n` — even if you also reference the parameter in the `[odes]`
> RHS. So don't reuse `F2`, `ALAG2`, … for an unrelated fraction or rate term;
> give such a quantity a different (un-indexed-looking) name.

This is an **ODE-engine** feature: the analytical PK functions have a single
fixed dose route, so they take only the bare `f=`/`lagtime=` mapping. (The
EKF/`[diffusion]` path applies per-compartment `F` but, as elsewhere, does not
apply absorption lag.)

> Per-compartment **observation scaling** (NONMEM's `Sn`, e.g. `S2 = V`) is a
> separate, readout-side concept — it divides a compartment's amount to give the
> observed concentration. It is configured in the [`[scaling]`](scaling.md)
> block (`obs_scale[CMT=n] = …` or `y[CMT=n] = …`), not via a reserved `Sn`
> individual parameter.

### Modeled infusion duration (`Dn`, `RATE=-2`)

NONMEM's `RATE = -2` makes a zero-order infusion's **duration** a model parameter
rather than a data value. Mirror it by naming an individual parameter `D{n}` for
the dose compartment `n`, and coding `RATE = -2` on the dose row (`AMT` is still
the amount). ferx then infuses `AMT` over the modeled duration `D{n}` — i.e. at
rate `AMT / D{n}` — resolved **per iteration and occasion** from the parameter,
so the duration can carry covariate effects and between-occasion variability:

```text
[parameters]
  theta TVD1(2.0, 0.1, 24.0)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  D1 = TVD1 * exp(ETA_D1)   # modeled duration for infusions into compartment 1
```

```text
# dataset: a RATE=-2 dose of 100 units into compartment 1
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV
1,0,.,1,100,1,-2,1
```

- A `RATE=-2` dose into compartment `n` **requires** a `D{n}` parameter; without
  one it is a loud error at the model+data join (`ferx check` / `fit`), never a
  silent bolus.
- `D{n}` composes with the dose attributes above: bioavailability `F{n}` scales
  the delivered amount **once** (`F·AMT` over `D{n}`, matching NONMEM's `F·RATE`),
  and absorption lag `ALAG{n}` shifts the infusion window's start while `D{n}`
  sets its length.
- A transient `D{n} ≤ 0` during estimation is clamped to a tiny positive floor
  (so `AMT / D{n}` stays finite); the converged optimum is interior, so reported
  estimates are unaffected — the same guard the built-in absorption models use.

> ⚠️ Like `F{n}` / `ALAG{n}`, `D{n}` is a **reserved name** when a `RATE=-2` dose
> targets compartment `n` (as in NONMEM). It then denotes that compartment's
> infusion duration even if you also reference it in the `[odes]` RHS — so don't
> reuse `D1`, `D2`, … for an unrelated decay constant or rate term.

This is an **ODE-engine** feature. On an **analytical** model a `RATE=-2` dose is
rejected with a pointer to the follow-up that will add analytical support;
`RATE=-1` (modeled *rate*, `R{n}`) is not yet supported on either engine.

## Stochastic ODE Models (SDE)

To model within-subject system noise that accumulates between observations, add
a `[diffusion]` block to your ODE model.  See [Stochastic Differential
Equations](diffusion.md) for a full description, worked example, and comparison
with sigma and omega.

## Limitations

- The observable compartment contains the amount (not concentration). Divide by volume in the ODE equations if needed
- SDE (`[diffusion]`) is not compatible with SAEM or the autodiff gradient path

Steady-state (`SS=1`) is supported for ODE models via numerical
pulse-expansion equilibration — see [Steady-State Doses](steady-state.md)
for the mechanism and how it differs from the analytical closed forms.
