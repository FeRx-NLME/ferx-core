# Plan: `omega_dist = vine` — vine-copula OMEGA as an opt-in SAEM variant

**Branch:** `feat/saem-copula` (per-phase PRs cut from it)
**Scope:** ferx-core only. ferx-r picks up the new key automatically through the
shared `apply_fit_option` dispatch — no R-side code change, only R-facing tests.

---

## Background

SAEM models the random effects η as multivariate normal with covariance Ω. That
forces Gaussian marginals and Gaussian (linear) dependence. Some PK/PD random
effects are skewed or tail-dependent, which an MVN Ω cannot represent. The goal
is an **opt-in** SAEM variant that replaces the MVN η-distribution with a **vine
copula** (flexible marginals + non-Gaussian pair dependence), without touching
the existing Gaussian path.

The single fitted copula object serves three uses that must stay mutually
consistent because they read the same object:

1. **Reporting** — marginal variances/correlations, tail-dependence coefficients,
   vine structure and pair-copula families, bootstrap uncertainty.
2. **Simulation** — draw joint η via the h-function recursion + inverse marginals.
3. **Individual estimation** — the vine density is the η prior for MAP/posterior.

### What this is *not*

- Not a fix for discrete subpopulations (missing covariate, metabolizer split) —
  that needs a finite mixture, not a continuous copula. Document for users.
- **SAEM-only.** FOCEI, Gauss-Newton, and importance sampling stay Gaussian. The
  `saem → focei` chain is incompatible with `omega_dist = vine` and is refused.
- Not Enzyme-differentiable in v1 (HMC E-step deferred — see Phase 5).

---

## Style decisions

- New SAEM fit-option fields use the `saem_` prefix in `FitOptions`
  (`saem_omega_dist`), mirroring `saem_omega_burnin` / `saem_n_leapfrog`.
- The DSL key is `omega_dist` (values `gaussian` | `vine`), parsed in the shared
  `apply_fit_option` dispatch so both `.ferx` and the R `settings` path get it.
- The `OmegaDist` enum lives in `types.rs` next to `EstimationMethod`, deriving
  `Default` (`Gaussian`).
- The FOCEI-chain incompatibility is enforced in `check_model_options`
  (`api.rs`) — the shared source of truth for `ferx check` and `fit()`, where the
  IMP-chain and SDE/trust-region guards already live — not in the parser. Code
  `E_OMEGA_DIST_CHAIN`.
- **Never modify the Gaussian arm of `run_saem`.** The branch is structured so
  the Gaussian path is byte-for-byte the pre-existing code; the existing SAEM
  tests are the regression net.

---

## The branch point in `run_saem`

`run_saem` branches on `options.saem_omega_dist` at the two places the Gaussian
assumption is load-bearing:

- **E-step acceptance** — replace `individual_nll_into` (which folds in the
  Gaussian prior) with the data-only `obs_nll_subject_into` (already
  `pub(crate)` in `stats/likelihood.rs`) minus the vine log-prior. The MH
  proposal is symmetric and cancels in the ratio, so the working proposal scale
  is unchanged.
- **M-step Ω update** (gated by `omega_burnin`) — replace the
  `eta_outer += ev·evᵀ` sample-covariance update with `vine.mstep_update(&etas)`.

Until the copula implementation lands, the `VineCopula` arm returns
`Err("omega_dist = vine (vine-copula SAEM) is not yet implemented")` — this is
the Rung-0 regression anchor (Phase 1).

---

## Copula M-step: IFM in two stages

`VineCopulaOmega::mstep_update` runs inference-functions-for-margins:

1. **Marginals** — fit each η_i marginal to the pooled E-step samples; PIT to
   pseudo-observations on [0,1].
2. **Vine** — hold the tree structure and family assignments fixed; update only
   the pair-copula parameters by bivariate MLE on the sequential
   pseudo-observations from the h-function recursion.

Joint marginal+copula fitting is unstable and must not run inside the SAEM loop.
**Fit the E-step conditional samples, never the EBEs** — EBEs are shrunken and
destroy spread, correlation, and tail structure.

---

## Phased delivery (one PR per phase, merged in order)

**Phase 1 — Option routing + regression anchor** (`feat/saem-copula`, this PR)
- `OmegaDist` enum + `saem_omega_dist` field/default in `types.rs`.
- `omega_dist` parser branch in `model_parser.rs`.
- `E_OMEGA_DIST_CHAIN` FOCEI-chain guard in `check_model_options`.
- `run_saem` branch with a "not yet implemented" `VineCopula` arm.
- Tests: parser accept/reject, guard, run_saem rejection; **Rung 0** = all
  existing SAEM tests pass unchanged.

**Phase 2 — `RandomEffectDistribution` trait + Gaussian parity** (**Rung 1**)
- Trait (`log_prior`, `mstep_update`, `sample`, `proposal_chol`).
- `GaussianOmega` wrapping `OmegaMatrix`; wire it into the copula arm and show it
  reproduces the Gaussian arm bit-for-bit on the existing SAEM cases.

**Phase 3 — Bivariate copula families**
- Gaussian, Student-t, Clayton, Gumbel, Frank: density, h-function, h-inverse,
  scalar MLE. Unit tests vs known values + finite-difference gradient checks.

**Phase 4 — Vine density + structure + integration** (**Rung 2/3/4**)
- R-vine type (tree sequence, edge families), density via h-recursion,
  structure-selection pre-fit (FFI to vinecopulib or via ferx-r; never in-loop).
- `VineCopulaOmega: RandomEffectDistribution`, IFM M-step wired into the arm.
- MH-fallback warning when `n_leapfrog > 0` with vine active.
- Bootstrap uncertainty (SIR/IS do not extend — Gaussian-only).
- Individual estimation: detect multimodal posteriors (Hessian PD check).
- Simulate-and-refit diagnostic shipped as a first-class function.

**Phase 5 (v2) — HMC E-step**
- Analytical `log_prior_grad` per family, Enzyme-differentiable; integrate with
  `compute_nll_gradient_ad`; remove the MH-fallback warning.

---

## Acceptance criteria

- **Rung 0:** existing SAEM tests pass unchanged after Phase 1.
- **Rung 1:** `GaussianOmega` in the copula arm == Gaussian arm on all SAEM cases.
- **Rung 2:** skewed-marginal simulation recovered within 10%.
- **Rung 3:** tail-dependent simulation recovered; correct family selected;
  params within 10%.
- **Rung 4:** simulate-and-refit preserves marginals + tail dependence; bootstrap
  intervals cover truth.
- FOCEI guard: `omega_dist = vine` + `saem → focei` errors, never silently wrong.
- No regression on any existing ferx-core test at any phase.

---

## Validation against NONMEM (required by CLAUDE.md)

Phases 2–4 each add a numerical comparison. The Gaussian-parity Rung 1 (Phase 2)
reproduces an existing NONMEM-validated SAEM fit exactly. Phases 3–4 validate the
copula machinery via simulate-and-refit (truth known) and, where a comparable
reference exists, against an rvinecopulib fit on the same pseudo-observations.
Record each comparison in the PR and on `docs/src/estimation/saem.md`.

---

## Risks / open questions

- **Re-selecting vine structure inside the SAEM loop** — forbidden; freeze after
  the pre-fit.
- **Fitting to EBEs instead of E-step samples** — forbidden; collapses spread.
- **HMC with vine active** — `log_prior_grad` is undefined in v1; warn + fall
  back to MH rather than run a broken gradient.
- **Accidental edits to the Gaussian arm** — the Rung-0 net guards this; run the
  SAEM tests after every change to `saem.rs`.
- **Identifiability** — initialise every vine fit from a Gaussian-copula solution
  and AIC/BIC-select per edge so the vine must *earn* its departure from
  normality; a vine silently absorbs a missing categorical covariate as apparent
  tail dependence. Exhaust covariate search first.
- **Regulatory** — a vine-copula Ω is novel; position for research/simulation use.
