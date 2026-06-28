# Plan: custom / time-varying residual-error magnitude (#484)

## Goal

Let the residual-error *magnitude* depend on `TIME`, covariates, and thetas —
not just on the prediction `f`. The motivating NONMEM idiom:

```nonmem
PROP = THETA(2)
IF(TIME.GT.24) PROP = THETA(2) * THETA(3)
W = SQRT(THETA(3)**2 + PROP**2 * IPRED**2)
```

Today ferx residual variance is a pure function of `(error_model, f, sigma)`:
`V = Σ (loading_k · σ_k)²`, loading ∈ {1, f}. Nothing time- or theta-dependent
can enter. The `busulfan_shukla` model shows the same class of limitation for a
*structural* parameter — its within-subject `DAY1` (TIME>24) CL effect had to be
smuggled into `[odes]` because `[individual_parameters]` are subject-static.
There is **no equivalent escape hatch for the error model**, so time-varying RUV
is currently inexpressible.

## Target syntax

Each sigma argument in `[error_model]` may be an **expression** over thetas,
covariates, `TIME`/`TAD`/`TAFD`, and individual parameters — not only a bare
sigma name. A bare name keeps today's exact behaviour.

```ferx
[parameters]
  sigma PROP_ERR ~ 0.1  (sd)
  sigma ADD_ERR  ~ 22.0 (sd)
  theta RUV_LATE(1.5, 0.0, 10.0)   # late-phase RUV inflation

[error_model]
  # proportional magnitude inflates after 24 h; additive unchanged
  DV ~ combined(PROP_ERR * (if (TIME > 24.0) RUV_LATE else 1.0), ADD_ERR)
```

Semantics: the expression is a per-observation **multiplier on that sigma's
loading**. Effective variance

```
V_i = Σ_k (loading_k,i · m_k(TIME_i, cov_i, θ) · σ_k)²   (+ block_sigma cross terms)
```

For a bare sigma name `m_k ≡ 1`, so existing models are bit-identical.

## Non-goals (deferred)

- Magnitude depending on `eta`/EBE or on `f` beyond the built-in proportional
  `f` loading. Multipliers are functions of `(TIME, cov, θ, indiv-param)` only —
  this keeps the multiplier independent of the inner loop, so it is constant
  across the EBE optimisation for a fixed θ.
- Closed-form θ-gradient of variance for the analytic-Laplace / SAEM M-step
  paths (`sens_outer_gradient.rs`, `saem.rs`). Phase 1 supports the OFV on the
  default derivative-free outer optimiser (BOBYQA) and any FD-gradient outer
  path; methods that consume the closed-form `dvar_dlogsigma` / `dvar_df`
  θ-derivatives with a *custom* magnitude error out up front (Phase 4) until
  the AD gradient is wired (Phase 5, follow-up).
- Per-CMT custom magnitude. Phase 1 targets `ErrorSpec::Single`. `PerCmt` keeps
  bare-name behaviour; a custom expression on a per-CMT endpoint errors.

## Current state

- Parse: `parse_error_model()` `model_parser.rs:7401`; resolves sigma *names* →
  flat indices in `build_error_spec()` `:7571`.
- Store: `ErrorModel` / `ErrorSpec::{Single,PerCmt}` / `EndpointError`
  `types.rs:1327-1692`.
- Compute: `residual_variance()` `residual_error.rs:8`;
  `ErrorSpec::sigma_loadings/variance_at/variance_at_with_correlations`
  `types.rs:1391-1490`; θ/f closed-form derivs `dvar_df` `:1531`,
  `d2var_df2` `:1561`, `dvar_dlogsigma` `:1588`.
- Reusable evaluator: the `[scaling]` AD-expression path
  (`eval_scale_dual` `model_parser.rs:10251`, `eval_bytecode_g<T: PkNum>` `:9144`)
  already runs **inside** the likelihood with analytic θ/η gradients and carries
  `TIME/TAD/TAFD`, covariates, thetas, indiv-params. This is the evaluator to
  reuse — *not* the post-fit `[derived]` path.
- Variance is consumed at ~170 call sites across 11 files; the multiplier must
  reach all of them that feed the likelihood. Minimal-churn strategy below.

## Design: precomputed per-observation loading multipliers

Threading expression context through ~170 `variance_at(cmt, f, sigma)` call
sites is untenable. Instead **precompute** the multipliers once per
`(θ, subject)` and carry them as an obs-parallel buffer, mirroring how
`obs_cmts`/`ipreds` are already threaded.

1. New field `CompiledModel.ruv_magnitude: Option<RuvMagnitude>` where

   ```rust
   pub struct RuvMagnitude {
       /// One optional multiplier program per flat sigma slot. `None` = bare
       /// sigma (multiplier ≡ 1). Programs are the same bytecode the [scaling]
       /// block compiles to, evaluable with f64 and Dual2.
       pub per_sigma: Vec<Option<ExprProgram>>,
   }
   ```

2. `RuvMagnitude::eval(theta, cov, indiv, time/tad/tafd) -> Vec<f64>` returns a
   per-sigma multiplier vector for one observation. For a subject we materialise
   a `Vec<Vec<f64>>` (obs × sigma) once per outer evaluation.

3. Variance core stays a pure function but gains an **effective-loading** form.
   Add `sigma_loadings_scaled(cmt, f, n_sigma, mult: &[f64])` that multiplies
   loading_k by `mult[k]`; existing `sigma_loadings` delegates with `mult = 1`.
   `variance_at` / `variance_at_with_correlations` gain `*_scaled` siblings
   taking the per-obs `mult` slice.

4. Threading: the obs-parallel multiplier slice rides alongside `ipreds`. The
   hot likelihood functions (`obs_nll_subject_from_preds`,
   `dense_residual_data_term`, `compute_r_diag*`) take an extra
   `ruv_mult: Option<&[Vec<f64>]>` (obs × sigma). When `None`, every helper
   takes the *exact* current path (no multiplier, bit-identical). This confines
   churn to the residual/likelihood core; far-flung consumers (npde, simulate,
   gauss_newton, saem) pass `None` until individually upgraded.

This keeps the default-no-custom-magnitude path provably unchanged (the
`Option` is `None`), and localises Phase 1 to the FOCE/FOCEI OFV path.

## Phase 1 — parser + types (no behaviour change yet)

- Extend `parse_error_model()` so each sigma argument is parsed as an
  `Expression` (reuse the `[scaling]` expression parser) instead of a bare name.
  A lone identifier that resolves to a declared sigma → bare slot (multiplier
  `None`); anything else → compile a multiplier program. The program references
  exactly one sigma (the slot it multiplies) plus thetas/covariates/TIME; reject
  programs that reference a *different* sigma or that reference no sigma.
- `build_error_spec()` produces `RuvMagnitude` alongside the existing
  `ErrorSpec`. Store on `CompiledModel`.
- Validation: custom magnitude only with `ErrorSpec::Single`; reject covariate
  names not in `[covariates]`; reject `IPRED/PRED/DV` references (magnitude must
  not depend on the prediction beyond the built-in `f` loading — Phase 1).
- Tier-1 unit tests: parse `combined(PROP*(if TIME>24 then T else 1), ADD)`,
  assert `RuvMagnitude.per_sigma[0].is_some()`, `[1].is_none()`; assert a
  bare-name model yields `ruv_magnitude == None`.

## Phase 2 — variance core (`residual_error.rs`, `types.rs`)

- Add `sigma_loadings_scaled`, `variance_at_scaled`,
  `variance_at_with_correlations_scaled`, `compute_r_diag_scaled`.
- Default (`mult` all-ones / `None`) delegates to existing fns — assert
  bit-identical in a unit test.
- Tier-1 tests: combined model, `mult = [2.0, 1.0]` → variance equals
  `(2·f·σ_p)² + σ_a²`.

## Phase 3 — FOCE/FOCEI OFV wiring (`likelihood.rs`)

- Materialise the obs×sigma multiplier matrix per subject when
  `model.ruv_magnitude.is_some()` (once per outer eval, before the inner loop —
  it does not depend on η).
- Thread `Option<&[Vec<f64>]>` into `obs_nll_subject_from_preds`,
  `dense_residual_data_term`, and the FOCE marginal R assembly; `None` keeps the
  current path.
- Inner loop: multipliers are η-independent, so EBE optimisation just sees the
  scaled R; no Jacobian change w.r.t. η.
- Slow (Tier-3, `slow-tests`) convergence test on a time-varying-RUV model.

## Phase 4 — guardrails for unsupported paths

- When `ruv_magnitude.is_some()`, error out clearly at `fit()` entry if the
  chosen method routes through a closed-form variance θ-gradient that Phase 1
  does not yet feed (analytic-Laplace outer gradient, SAEM M-step). Message
  points at BOBYQA / FD-gradient outer optimisers as the supported path.
- CWRES/IWRES/NPDE post-processing: use scaled variance so diagnostics match the
  fitted model (`residual_error.rs` IWRES at `:195`, `npde.rs`). If not wired in
  this PR, document the limitation and gate.

## Phase 5 — analytic θ-gradient (follow-up PR, not this one)

The multiplier programs are already AD-ready (`eval_bytecode_g<Dual2>`), so
`∂V/∂θ` via the chain rule through `m_k(θ)` is mechanical. Wire it into
`sens_outer_gradient.rs` and the SAEM M-step, then drop the Phase 4 guard for
those methods. Tracked separately to keep this PR reviewable.

## Validation vs NONMEM (required by CLAUDE.md)

Build a small time-varying-RUV `.mod` (proportional error inflated for
`TIME>24` via a theta), fit in NONMEM via the licensed `pmx` container, and
reproduce in ferx. Compare OFV and final estimates; record in
`docs/estimation/*.qmd` (or `docs/faq.qmd`) and the PR description. The
`busulfan_shukla` directory is the staging ground for the NONMEM reference.

## Docs & changelog

- `docs/model-file/individual-parameters.qmd` or a new
  `docs/model-file/error-model.qmd`: document expression-valued sigma arguments,
  the `(TIME, cov, θ)`-only restriction, and the supported-method matrix.
- `CHANGELOG.md` under `## [Unreleased] / Added`: "Residual-error magnitude may
  now be an expression of TIME/covariates/thetas (#484)."

## Files touched (Phase 1–4)

- `src/parser/model_parser.rs` — expression-valued sigma parse + `RuvMagnitude` build.
- `src/types.rs` — `RuvMagnitude`, `*_scaled` variance fns, `CompiledModel` field.
- `src/stats/residual_error.rs` — scaled variance / R-diag.
- `src/stats/likelihood.rs` — materialise + thread multiplier through FOCE/FOCEI OFV.
- `src/api.rs` — Phase-4 method guard.
- tests: Tier-1 in the modules above; Tier-3 convergence test.
