# Frequently Asked Questions

## Do I need to use MU-referencing in my model definitions, like in NONMEM / nlmixr2?

**No.** You do not need to write an explicit `MU_i` intermediate variable — ferx detects the same structure automatically from the right-hand side of each `[individual_parameters]` line.

In NONMEM and nlmixr2, MU-referencing is a convention in which each random effect `ETA(i)` is linearly associated with a single `MU_i` term (typically `MU_i = LOG(THETA(i))`), so individual parameters look like:

```
MU_1 = LOG(THETA(1))
CL   = EXP(MU_1 + ETA(1))
```

This structure is required by NONMEM's SAEM implementation, whose conjugate-Gibbs E-step is only valid when the `MU_i → ETA(i)` relationship is strictly linear (typically on a log scale). Deviating from it — for example writing `CL = THETA(1) * EXP(ETA(1))` without going through an intermediate `MU_1` — causes NONMEM SAEM to reject the model or silently produce biased estimates.

### How ferx handles it

In ferx, you simply write the individual-parameter line in the form that makes sense for your model. The parser inspects each line and records which `THETA` acts as the "anchor" for each `ETA`, then uses that information to re-centre the estimation search at every outer iteration. The following patterns are detected automatically:

```
CL = TVCL * exp(ETA_CL)              # multiplicative exp — most common
CL = exp(log(TVCL) + ETA_CL)         # canonical MU form
CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)   # with covariate adjustment
CL = TVCL + ETA_CL                   # additive eta (linear shift)
```

The parser records `ETA_CL → (TVCL, log_transformed)` and at each outer step computes the shift

```
mu[i] = log(theta[i])   # for multiplicative/exp patterns
mu[i] = theta[i]        # for additive patterns
```

The inner optimizer (and the SAEM exploration-phase MH proposal) is then centred on this `mu` shift rather than on `eta = 0`. This matches the convergence behaviour that NONMEM / nlmixr2 get from an explicit `MU_i = LOG(THETA(i))` block, without requiring you to write it.

Patterns that ferx **does not** auto-detect (and therefore falls back to the plain `eta = 0` start) include:

- Parameters with two or more free thetas in the product (`CL = TVCL * TVSCALE * exp(ETA_CL)`) — ambiguous anchor.
- Compound eta expressions inside the `exp` (`CL = TVCL * exp(ETA_CL + ETA_OCC)`).
- Non-standard transforms the parser doesn't recognise.

Mu-referencing being absent is not an error — it just means that eta for that parameter is initialised to zero at each outer step, same as the pre-automatic behaviour. When a fit runs, ferx records the list of auto-detected mu-referenced ETAs in `FitResult.warnings` (as a line starting with `mu-ref: …`) so you can confirm what the parser picked up.

### Why it matters

Mu-referencing mostly affects convergence speed and robustness, not the final MLE:

- **FOCE / FOCEI**: each outer step re-optimises ETA by BFGS. With a mu-shift, the warm-started BFGS sees a much better starting point when `THETA` moves away from the initial estimate, so fewer inner iterations are needed and pathological steps are less likely.
- **SAEM**: during the exploration phase, Metropolis–Hastings proposals are centred on `mu_k` instead of on the current chain state. This helps the chain escape the (incorrect) `eta = 0` basin when `TVCL` is still far from the true value. During the convergence phase the proposal reverts to a symmetric random walk so that detailed balance holds.

Because the estimator is MLE in both cases, models that converge without mu-referencing will converge to the same estimates with it — just (usually) in fewer iterations.

### Disabling it

If you want to benchmark against the pre-2026 behaviour, set:

```
[fit_options]
  mu_referencing = false
```

or, from the Rust API, `FitOptions { mu_referencing: false, .. }`. The default is `true`.

### Porting from NONMEM

If you have a NONMEM model that uses an explicit `MU_1 = LOG(THETA(1))` line, just drop the `MU_i` intermediate and write the individual parameter directly — ferx will detect the equivalent mu-reference automatically and the fit will be equivalent.

## Can I use `if` / `else` statements like in NONMEM or nlmixr2?

**Yes.** ferx supports two forms of conditional logic in both
`[individual_parameters]` and `[odes]` blocks:

```
# Block form
if (WT > 70) {
  CL = TVCL * (WT / 70)^0.75 * exp(ETA_CL)
} else if (SEX == 1) {
  CL = TVCL * 1.2 * exp(ETA_CL)
} else {
  CL = TVCL * exp(ETA_CL)
}

# Inline (ternary) form
CL = if (SEX == 1) TVCL * 1.5 else TVCL
```

Conditions support comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`) and
logical operators (`&&`, `||`, `!`). Either form may be combined with
arbitrary arithmetic, including covariate references and other ETAs.

**Compared to NONMEM:** NONMEM uses Fortran-style `IF (cond) THEN ... ELSE
... ENDIF` and Fortran comparison aliases (`.GT.`, `.LE.`, etc.). ferx uses
C-style braces and operator symbols. The semantics are otherwise the same —
exactly one branch fires per evaluation, the rest are ignored.

**Compared to nlmixr2:** nlmixr2's `if (cond) { ... } else { ... }` syntax
is a near-verbatim match for ferx's block form. Drop the `<-` assignments
in favour of `=` and the model translates directly.

**Caveat:** when a parameter is assigned inside an `if` block, ferx skips
mu-reference detection for that parameter (the `(ETA → THETA)` link is no
longer unconditional). See
[Individual Parameters](model-file/individual-parameters.md#interaction-with-mu-referencing)
for the workaround if you need both conditional logic and mu-referencing
on the same parameter.

**A note on `==` and `!=`:** these operators do an exact bitwise float
comparison. They work as you'd expect for integer-coded covariates
(`SEX == 1`, `STUDY != 3`) but are unreliable on continuous values, where
floating-point round-off can flip the result. For continuous thresholds
prefer a bracketed comparison (`WT >= 70 && WT < 80`) over `WT == 75`.

## Which outer optimizer should I pick?

`slsqp` (the default) is the right choice for most models — it is fast, handles
box constraints cleanly, and behaves well on the log-transformed parameter
scale that ferx uses internally.

Reach for a different optimizer when SLSQP misbehaves:

- **`bobyqa`** — derivative-free, good when FOCE's FD gradients are noisy and
  SLSQP stalls or oscillates. Slower per iteration on smooth problems, but
  often converges when gradient-based methods give up.
- **`trust_region`** — second-order Newton trust-region. Can be faster near
  convergence because it uses curvature information; tune the CG budget with
  `steihaug_max_iters` (default 50) if you have more than ~50 packed
  parameters.
- **`lbfgs` / `bfgs`** — fall back to these only when NLopt is unavailable.

See [Fit Options](model-file/fit-options.md#optimizer-choices) for the full
list.

## How do I scale predictions to match my data's units, like NONMEM's `S1`/`S2`?

Use the `[scaling]` block. The convention is divisive
(`pred_scaled = pred_raw / scale`), matching the natural reading of
`obs_scale = V/1000`.

```
[scaling]
  obs_scale = 1000          # mg/L → mg/mL
```

The block also supports expression-form scales (`obs_scale = WT / 70`)
and an ODE-only Form C that replaces the state readout entirely:

```
[structural_model]
  ode(states=[depot, central])

[scaling]
  y = central / V           # central holds amount; observe as concentration
```

See [Scaling](model-file/scaling.md) for the full reference, including
how this compares to NONMEM's `S1`/`S2` and nlmixr2's `cmt(central); f = ...`.
