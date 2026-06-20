# Outer Optimizers

> **Maturity: beta** — see [Feature Maturity](../maturity.md) for what this means.

The **outer optimizer** is the algorithm that minimises the population objective function (OFV) over the transformed parameter vector `[log θ, chol(Ω), log σ]`.

It only applies to methods that *have* a single outer-loop minimisation over that vector:

| method | does `optimizer` apply? |
|---|---|
| `foce` / `focei` | yes — picks the algorithm for the population OFV minimisation |
| `gn` (pure Gauss-Newton) | no — GN runs its own Levenberg-Marquardt loop |
| `gn_hybrid` | yes for the FOCEI polish phase only; the preceding GN phase ignores it |
| `saem` | **no** — SAEM has no single outer minimisation; the M-step uses a hardcoded NLopt algorithm and is not user-selectable. Setting `optimizer` under `method = saem` triggers a warning and is otherwise ignored. |
| `[saem, focei]` (chain) | applies to the FOCEI polish stage only |
| `imp` | no — IS is a sampling pass, not an optimisation |

Set via `[fit_options]`:

```
[fit_options]
  method    = focei
  optimizer = bobyqa
```

---

## Available optimizers

### NLopt algorithms

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `bobyqa` | Bounded Optimization BY Quadratic Approximation | **Default.** Derivative-free quadratic trust-region. Avoids the fixed-EBE gradient bias that stalls gradient methods on ill-conditioned fits; consistently reaches a lower OFV on ODE/PD models, sparse data, and Hill-ridge problems without reconverging the inner loop. |
| `slsqp` | Sequential Least Squares Programming | Gradient-based, handles bounds well. Fast on well-conditioned analytical PK models; can stall above the true minimum on ODE/PD / sparse-data fits unless paired with `reconverge_gradient_interval = 1` (5–6× cost). |
| `nlopt_lbfgs` | L-BFGS via NLopt | Limited-memory BFGS. Fast on analytical 1-/2-/3-cpt FOCE/FOCEI fits (exact analytic gradient); also useful for high-parameter-count models. |
| `mma` | Method of Moving Asymptotes | Alternative constrained gradient optimizer. Rarely needed. |

### Built-in algorithms

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `bfgs` | BFGS with backtracking line search | Quasi-Newton, no NLopt dependency. |
| `lbfgs` | L-BFGS | Memory-efficient BFGS variant. |

### Newton trust-region

| Key | Algorithm | Notes |
|-----|-----------|-------|
| `trust_region` | Newton trust-region with Steihaug CG | Uses the AD-based outer gradient and a BHHH approximate Hessian `H ≈ 4 Σ gᵢgᵢᵀ` (always positive semi-definite). The CG budget defaults to `ceil(sqrt(n_params)).clamp(5, n_params)`; pin it with `steihaug_max_iters`. Best combined with `inits_from_nca` since it benefits from good starting values. |

---

## When to use which

**`bobyqa` (default)** — start here. Derivative-free quadratic trust-region; re-evaluates EBEs at every trial point so it avoids the fixed-EBE gradient bias that stalls gradient-based optimizers on ill-conditioned fits. On the cefepime 2-cpt benchmark below it reaches a lower OFV than `slsqp` (even reconverged at full cost) in less wall time. Works equally well on smooth analytical PK models — converges more slowly than gradient methods per iteration but each iteration is cheap (no FD gradient sweep).

**`slsqp`** — switch to this when `bobyqa` is too slow on a smooth, well-conditioned model with many parameters (it can need many quadratic-interpolation samples to triangulate a high-dimensional surface). Gradient-based, handles box constraints cleanly; pair with `reconverge_gradient_interval = 1` if it stalls above an expected OFV.

**`trust_region`** — for models with many parameters (large θ + Ω) or when combined with `inits_from_nca`. The second-order curvature information helps when starting values are already in the basin.

**`gn` / `gn_hybrid`** — these are [Gauss-Newton estimation methods](gauss-newton.md), not outer optimizers in the same sense. They replace the FOCE outer loop entirely rather than selecting an algorithm within it. (`gn_hybrid` polishes via FOCEI and inherits the `optimizer` setting for that stage — so the polish runs with `bobyqa` unless overridden.)

**`lbfgs` / `nlopt_lbfgs`** — strong choices on **analytical 1-/2-/3-cpt FOCE/FOCEI** fits, where they get an exact bias-free gradient (see [Analytic FOCE / FOCEI gradient](#analytic-foce--focei-gradient-analytical-pk-models)) that reaches the true optimum at wall-time comparable to `bobyqa` (and several × faster than the built-in `bfgs`). On ODE/PD or sparse-data models they fall back to the finite-difference gradient and inherit its fixed-EBE bias — prefer `bobyqa` there. **`mma`** is rarely needed.

> **Why is BOBYQA the default?** Previously the default was `slsqp`. The Emax PKPD benchmark in [`saem.md`](saem.md) and the cefepime validation below both showed that the fixed-EBE FD gradient drives `slsqp` to local minima hundreds of OFV units above the true optimum on ODE/PD models and sparse data — exactly the workloads that aren't covered by the analytical-PK comfort zone. `bobyqa` doesn't use the gradient, doesn't see the bias, and reaches the same (or lower) OFV in the same or less wall time. The previous behaviour is one line away: `optimizer = slsqp` in `[fit_options]`.

---

## Fixed-EBE gradient bias and `reconverge_gradient_interval`

By default, gradient-based optimizers (`slsqp`, `lbfgs`, etc.) hold each subject's EBEs fixed while computing the population gradient. This is cheap but omits the response of the inner solution to θ and Ω. On ill-conditioned models the omitted term causes `slsqp` to stall well above the `bobyqa` optimum.

Set `reconverge_gradient_interval = 1` to re-solve the inner loop at every gradient evaluation, recovering the full gradient surface at roughly **5–6× cost**. A value of `N` reconverges every Nth evaluation and uses the cheap gradient in between — often enough to close most of the gap:

| Configuration | OFV | Wall time |
|---|---:|---:|
| `slsqp` (fixed-EBE) | 68,252 | 390 s |
| `slsqp` + `interval = 10` | 66,118 | 633 s |
| `slsqp` + `interval = 1` | **65,485** | 1,871 s |
| `bobyqa` (derivative-free, **default**) | 65,598 | 315 s |

On this cefepime 2-compartment dataset `bobyqa` reaches a near-optimal solution faster than any reconverged `slsqp` setting. The reconverged gradient is most useful when derivative-free search is too slow (high parameter count) or when a gradient optimizer is required for other reasons.

IOV models (`kappa`/`block_kappa`) always reconverge regardless of this setting. Even so, a *pure* `slsqp` cold-start on an IOV model can terminate a few OFV units above the minimum, and the exact stopping point is platform-dependent (the re-converged FD gradient's summation order differs across architectures — issue #160). Prefer the default `bobyqa` or a `methods = [saem, focei]` chain for IOV fits; both reach the minimum platform-independently. See [Inter-Occasion Variability](../model-file/iov.md#limitations).

---

## Analytic FOCE / FOCEI gradient (analytical PK models)

For **analytical 1-, 2-, and 3-compartment models** (IV bolus/infusion, oral, and steady state) under **FOCE or FOCEI**, the gradient-based optimizers (`bfgs`, `lbfgs`, `nlopt_lbfgs`, `slsqp`) automatically switch from the finite-difference gradient to an **exact closed-form marginal gradient** (Almquist et al. 2015). FOCEI differentiates the Laplace marginal; FOCE differentiates ferx's Sheiner–Beal linearized marginal. Both are computed analytically through second-order dual numbers and *include the full EBE response* (the term `reconverge_gradient_interval` recovers by brute force) — so they do **not** carry the fixed-EBE bias described above, at no extra cost. There is nothing to configure: the `method` and `optimizer` choices are the switch, and any model outside the analytical scope falls back to the finite-difference gradient transparently.

Because the gradient is both exact and bias-free here, `lbfgs` / `nlopt_lbfgs` are strong choices on these models: each gradient evaluation is a closed form rather than an `O(n)` finite-difference sweep, and — unlike the FD gradient — it carries no fixed-EBE bias, so the optimizer reaches the true optimum. Wall-time versus the derivative-free `bobyqa` default is model-dependent and roughly comparable (e.g. warfarin FOCEI: `lbfgs` ≈ 0.29 s vs `bobyqa` ≈ 0.43 s; on a 2-cpt fit `bobyqa` is faster but can stop short of the optimum), and both are several times faster than the built-in `bfgs`. The closed form also threads through an EBE warm-start predictor (Almquist Eq. 48) that starts each subject's inner solve closer to its optimum as the population parameters move.

The exact gradient lands on the same optimum as the underlying objective, validated against NONMEM on the warfarin 1-cpt oral model: FOCE reaches OFV −280.36 (NONMEM −280.36; TVCL 0.1330, TVKA 0.7252, ω²: 0.0286 / 0.00958 / 0.349) and FOCEI reaches −286.00 (NONMEM −286.00) — both agreeing to ~4–5 significant figures across θ, Ω, and σ.

The exact gradient also covers log-transform-both-sides (`log(DV) ~ additive(...)`) and constant output scaling (`[scaling] obs_scale = k`): the analytic provider applies the `g = ln(f)` jet transform (value, gradient, and Hessian) and the constant divisor in closed form. Validated against NONMEM on the warfarin LTBS model — the gradient-based `lbfgs` path reaches OFV −675.302 and recovers NONMEM's MLE (TVCL 0.1327, TVV 7.738, TVKA 0.811) to ~4 significant figures.

The fallback (finite-difference) gradient is used when the model has: an ODE system, inter-occasion variability (`kappa`), a dose lagtime, time-varying covariates, system resets, expression/per-compartment output scaling, or an overlapping steady-state infusion (`T_inf > II`, [#379](https://github.com/FeRx-NLME/ferx-core/issues/379)). For those, the guidance above (prefer `bobyqa`, or reconverge `slsqp`) still applies.

---

## Global pre-search

Enable `global_search = true` to run a derivative-free CRS2-LM pre-search before the local optimizer. This helps escape local minima on complex models:

```
[fit_options]
  method        = focei
  global_search = true
  global_maxeval = 2000   # auto-scaled if omitted
```

The pre-search uses NLopt's Controlled Random Search and requires lower/upper bounds on all theta parameters. It hands off to the local optimizer (`optimizer` key) once the budget is exhausted.

See also [`fit-options.md`](../model-file/fit-options.md) for the full list of optimizer-related options.
