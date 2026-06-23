# Changelog

All notable changes to **ferx-core** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(see the [Releases](https://ferx-nlme.github.io/ferx-core/development/sdlc.html#9-releases)
section of the SDLC for the versioning policy).

<!--
  HOW TO MAINTAIN THIS FILE
  - Add an entry under [Unreleased] in the same PR as your change, in the right
    category (Added / Changed / Deprecated / Removed / Fixed / Security /
    Performance). One bullet, user-facing language, reference the issue/PR (#NN).
  - At release time: rename [Unreleased] to the new version + date, then start a
    fresh empty [Unreleased] and update the compare links at the bottom.
  - The R wrapper (ferx-r) tracks its own user-facing changes in NEWS.md.
-->

## [Unreleased]

### Performance
- **Exact analytic gradients for `transit()` absorption ODE models** (#430). The
  built-in transit input-rate forcing's `ln Γ(n+1)` constant now has a `Dual2` rule
  (analytic digamma/trigamma derivatives of the shared Lanczos `ln_gamma`), so a
  `transit()` model is evaluated over `Dual2` by the ODE sensitivity provider and
  drives exact analytic FOCE/FOCEI/Bayes gradients instead of finite differences —
  joining `igd()` on the analytic path. Estimates are unchanged; gradients are exact
  and drop the `(n_params+1)×` FD multiplier on transit fits.
- **Faster analytic time-varying-covariate inner η-gradient + ODE-sensitivity path
  consolidation** (#451). The per-subject event schedule is now reused across inner
  BFGS steps instead of rebuilt each step, identical per-event covariate snapshots are
  seeded once, and the time-after-dose anchor advances incrementally — cutting
  redundant work in the inner EBE loop for TV-covariate analytical fits. Internally,
  the production `f64` and dual ODE-sensitivity paths now share single generic helpers
  for the built-in absorption input-rate forcing and the LTBS log transform, so the
  predictor and the analytic gradient can't silently drift; no change to results.
- **Analytic inner η-gradient for time-varying covariates / oral infusion on
  analytical PK models** (#447). The light `Dual1` inner EBE gradient previously
  declined these subjects and reverted to finite differences even though the
  **outer** gradient already served them; it now uses a first-order event-driven
  walk (`subject_eta_grad_tvcov`, the light mirror of `subject_sensitivities_tvcov`),
  so the inner EBE loop is exact and replaces FD's `~2·n_eta+1` predictions per step
  with one. Validated against the FD-validated outer `df_deta` (1-/2-/3-cpt, IV/oral,
  steady state).

### Added
- **`[event_model]` hazard expressions can reference `[individual_parameters]`** names —
  e.g. a hazard driven by an individual `CL` — resolved per subject at evaluation time, in
  addition to the existing theta/eta/covariate namespace. Intermediate variables and names
  defined with a NONMEM-style `if (...) { ... } else { ... }` block are supported; only the
  individual parameters the hazard actually references are computed. A hazard reference to an
  individual parameter that depends on an inter-occasion (IOV/kappa) random effect — or on a
  `[covariate_nn]` output — is rejected with a clear error, since the per-subject hazard
  cannot evaluate either. Behind the `survival` feature (#440).
- **Analytic FOCE/FOCEI gradients for time-varying covariates on ODE models** (#439).
  An ODE model whose covariates change over time (per-event `WT`, `CRCL`, …) with
  **bolus** dosing now gets the exact analytic outer gradient and the light `Dual1`
  inner η-gradient instead of falling back to finite differences. The dual is seeded
  on `(θ,η)` (`M = n_theta + n_eta`) and walked over a per-event event-driven
  integration, mirroring the analytical TV-cov path and matching production's
  `ode_predictions_event_driven` predictor bit-for-bit (validated against it + FD).
  Combined with infusion / steady-state / reset / `init(...)`, TV-cov still falls
  back to FD.
- **Analytic gradients for per-CMT (multi-endpoint) ODE readouts** (#439). The
  `[scaling] y[CMT=N] = <expr>` Form-C readout is now differentiated by the ODE
  sensitivity provider — each endpoint's compiled output program is evaluated over
  `Dual2` (outer) and `Dual1` (inner), dispatched per observation by its CMT — so
  multi-analyte / PK-PD models (e.g. parent + metabolite, or PK + effect) get the
  exact analytic FOCE/FOCEI gradient instead of falling back to finite differences;
  `gradient = fd` is no longer required for these models. Validated against finite
  differences of the production predictor.
- **Analytic FOCE/FOCEI gradients for user-`[odes]` models** (#410). The ODE
  sensitivity engine — an augmented `Dual2` RK45 that propagates `∂state/∂(θ,η)`
  alongside the state — is now armed, so in-scope ODE models drive the exact
  analytic outer gradient (and the Eq. 48 EBE predictor) instead of the prior
  gradient-free path. The inner EBE loop likewise gets an exact η-gradient from a
  lighter `Dual1` (gradient-only) walk — one integration per inner step in place of
  finite differences' `2·n_eta+1`, so the EBE search is exact and faster. Scope: RHS-program models with an `ObsCmt` or simple Form-C
  (`y = central/V1`) readout, bolus + finite infusion, bioavailability `F`, EVID
  3/4 resets, `init(...)`, static covariates, a constant `obs_scale` divisor, and
  LTBS (`log(DV) ~ …`) output transforms. Out-of-scope features (steady state,
  estimated lagtime, IOV, `input_rate`, SDE, time-varying covariates, expression
  `obs_scale`, modeled-`RATE` doses, `F` on a rate-defined infusion) fall back to
  the existing path unchanged. Validated against finite differences of the
  production predictor, reconverged FD of the FOCEI marginal, and a full-convergence
  cross-check that an ODE fit reproduces the analytical (NONMEM-validated) twin's
  estimates and standard errors.
- **Analytic sensitivities for oral infusion** on the analytical 1-/2-/3-cpt
  models: a depot-bypass infusion into the central compartment (RATE>0 into cmt 2,
  #350) and a zero-order input into the oral depot (RATE>0 into cmt 1, #400) are
  now carried through the second-order-dual event-driven walk (`rate_central`/
  `rate_depot` forced responses), so these subjects drive the exact analytic
  FOCE/FOCEI gradient instead of falling back to finite differences. Validated
  against finite differences of the production predictor across 1-/2-/3-cpt and
  both infusion compartments (#367).
- **Analytic sensitivities for expression output scaling** (`[scaling] obs_scale =
  <expr>`) on analytical PK models. An `obs_scale` expression that references
  individual parameters, θ, or covariates (e.g. `1000 / V`, `WT / 70`) is now
  compiled to a `Dual2`-differentiable program, so the analytic FOCE/FOCEI outer
  gradient differentiates the scaled prediction `f / scale` exactly (quotient
  rule) instead of falling back to finite differences. Validated against finite
  differences of the production predictor and against a NONMEM reference (#367).
- **Analytic sensitivities for inverse-Gaussian (`igd()`) absorption** on ODE
  models: the built-in input-rate forcing is now evaluated over `Dual2` by the
  analytic ODE sensitivity provider, so an `igd()` model drives exact FOCE/FOCEI/
  Bayes gradients instead of falling back to finite differences (estimates
  unchanged; gradients exact and cheaper). The forcing was lifted to a
  `PkNum`-generic form; transit (`transit()`) still uses FD pending its own
  `ln_gamma` `Dual2` rule. Validated by an analytic≡central-FD gradient parity
  test in the default build (#430).

### Changed
- **Documentation now builds as a Quarto website** using the shared ferx site
  branding and styling instead of mdBook. Source pages now live under
  `docs/**/*.qmd`, with navigation in `docs/_quarto.yml` (#443).
- **FOCE/FOCEI and SAEM/Bayes HMC gradients now come from hand-rolled analytic
  `Dual2` sensitivities** rather than Enzyme automatic differentiation. The inner
  EBE gradient, the outer θ/Ω/Σ gradient, and the SAEM/Bayes HMC η-sampler all use
  the same exact closed-form sensitivity provider; models outside its scope (ODE,
  LTBS, expression scaling, time-varying covariates, SDE) fall back to finite
  differences. The HMC sampler (`saem_n_leapfrog > 0`) no longer requires an
  autodiff build — it matches the FOCEI point estimate on warfarin with R̂ ≈ 1.00
  (#367).

### Removed
- **The Enzyme automatic-differentiation path is retired** — the `ad/` module, the
  `autodiff` Cargo feature, and the custom `enzyme` toolchain pin are removed.
  ferx-core now builds on a stock nightly toolchain with `cargo build` (no
  from-source compiler, no `RUSTFLAGS="-Z autodiff=Enable"`). `gradient_method = ad`
  now returns an `E_AD_RETIRED` error; use `gradient = auto` (the exact analytic
  gradient where it is in scope, finite differences otherwise) or `gradient = fd`
  (#367).

### Fixed
- **Analytic sensitivities and predictions for time-varying covariates with
  intermediate `[individual_parameters]` assignments** (#455, #456). A model whose
  individual-parameter block computes intermediate quantities (e.g.
  `WTREL = WT / 70`) before the structural PK outputs now gets the exact analytic
  `Dual2` gradient on every path — the TV-cov gate plus the previously-overlooked
  non-TV (`subject_sensitivities` / `subject_eta_grad`) and IOV gates all key on the
  required structural PK slots instead of the assignment count, so these models no
  longer silently fall back to a fallback that mis-seeded `∂f/∂η`. Additionally,
  the public `predict()` and the sdtab `PRED` column now both route through the
  TV-covariate-aware predictor, so they honour per-event covariate breakpoints
  (and EVID=3/4 resets) and agree with each other. Cross-checked against NONMEM
  7.5.1 (ADVAN3 TRANS4, EVID=2 covariate update).
- **FOCE/FOCEI analytic outer gradients stay enabled for populations that include
  dosing-only subjects**. Such subjects contribute zero to the marginal objective,
  so they now return a zero analytic gradient instead of forcing SLSQP/L-BFGS onto
  the slower fixed-EBE fallback path (#455).
- **Gradient-based optimizers no longer stall when a few subjects are declined by
  the analytic outer gradient** (#455). The exact analytic outer gradient was
  assembled all-or-nothing: a single declined subject — whether structurally out
  of scope (steady-state + reset, modeled-duration dose, oral infusion under F≠1)
  or numerically declined (an indefinite per-subject inner Hessian that fails the
  Cholesky factor in the gradient assembly) — forced the whole population onto the
  θ-only fixed-EBE fallback, whose biased Ω/σ block left the variance components
  pinned at their start and stalled `slsqp` / `nlopt_lbfgs` / `mma` / `lbfgs` well
  above the derivative-free (`bobyqa`) optimum. The non-IOV outer gradient is now
  assembled per subject — exact analytic for in-scope subjects, a reconverged
  per-subject finite-difference (carrying the full η̂/Ω/σ EBE response, no PD
  Hessian required) for the declined ones — so one declined subject no longer
  disables the exact gradient for the other thousands. On the 5937-subject
  pediatric Jasmine fit (one subject with an indefinite inner Hessian), default-
  start FOCEI `slsqp` improves from the previous stalled best OFV 73468 to 66593,
  while `mma` reaches 66560.68 best-seen — about 21 OFV above the NONMEM reference
  (66539.38) and below both `bobyqa` (68456 best-seen) and SAEM 500/500 (67377).
- **Documentation no longer references the retired Enzyme/autodiff installation or
  usage path**, and now describes `gradient = auto` / `gradient = fd` with the
  analytic `Dual2` sensitivity provider (#381).
- **SAEM/Bayes HMC step-size adaptation** targeted the random-walk acceptance rate
  (≈0.234) for the gradient-guided HMC η-kernel, which over-inflated the leapfrog
  step until trajectories diverged — over-dispersing η and biasing the residual
  error (a warfarin Bayes-HMC run gave `PROP_ERR` ≈ 0.05 / R̂ > 2 vs the correct
  ≈ 0.011). The HMC kernel now adapts toward ≈0.7, matching the SAEM split (#367).
- **Overlapping steady-state infusions (`T_inf > II`)** are now solved exactly for
  the analytical 1-/2-/3-compartment models instead of being skipped. Previously
  the closed form returned 0 and the dose was applied as a single (non-SS)
  infusion (with a `W_STEADY_STATE_INFUSION` warning); the steady-state
  concentration now superposes the infinite past pulse train (several pulses
  simultaneously active), validated against explicit superposition. The analytic
  FOCE/FOCEI sensitivity provider carries the same closed form, so these subjects
  no longer fall back to finite differences. The warning now fires only for model
  paths that still skip SS pre-equilibration (ODE models, or EVID=3/4 resets)
  (#379).

### Performance
- **Faster outer-gradient sensitivities for user-`[odes]` models with IIV-free
  parameters** (#445). The augmented-`Dual2` RK45 now carries a second-order
  Hessian only over the individual parameters that bear IIV (η), dropping the
  block among the IIV-free (θ-only) parameters — which the FOCEI gradient never
  reads, since it uses no `∂²f/∂θ²`. On a 2-compartment ODE with 2 of 4
  individual parameters fixed, the per-subject sensitivity cost falls ≈2.2×; the
  retained dual entries and the first-order chain (`df_deta`, `df_dtheta`) are
  bit-for-bit, and the chained second-order outputs (`d2f_deta2`,
  `d2f_deta_dtheta`) agree to ~1e-9 (the terms are identical but summed in a
  different order). Models whose individual parameters all carry IIV are unaffected.

### Added
- **Analytic sensitivities for dose lagtime (ALAG)** on analytical PK models: a
  declared `LAGTIME`/`alag` parameter is now differentiated exactly by the
  sensitivity provider — it enters every dose through the elapsed-time argument
  (`∂elapsed/∂lagtime = −1`, seeded as its own dual axis), including the
  steady-state pre-arrival tail. Lagtime models therefore drive the analytic
  FOCE/FOCEI outer gradient and the analytic inner EBE gradient instead of
  falling back to finite differences. Validated against finite differences of the
  production predictor (value, ∂/∂η, ∂²/∂η², ∂/∂θ, ∂²/∂η∂θ) and as a full packed
  outer gradient (#367).
- **Analytic M3 (BLOQ) outer gradient for both FOCE and FOCEI** on analytical PK
  models: the exact closed-form marginal gradient now covers M3-censored subjects.
  Under FOCEI a censored row enters the Almquist Laplace assembly as a data term
  `−logΦ((LLOQ−f)/√V)` plus its true-inner-Hessian curvature, excluded from
  `H̃`/`log|H̃|`. Under FOCE it leaves the Sheiner–Beal marginal (`R̃` and the
  quadratic form are built over the quantified rows only) and re-enters as
  `−logΦ((LLOQ−f̂)/√R⁰)` with the population variance. Both match ferx's M3
  objective and are validated against reconverged finite differences (~1e-6 on
  every θ/Ω/σ packed parameter) and against NONMEM (`METHOD=1 LAPLACE` with and
  without INTER) to <1% on the structural parameters (#367).
- **Analytic M3 (BLOQ) inner EBE gradient** for analytical PK models: the
  per-subject EBE optimiser now has an exact closed-form η-gradient for the M3
  censored term `−logΦ((LLOQ−f)/√V)` (inverse-Mills-ratio coefficient), replacing
  the finite-difference inner gradient on `bloq_method = m3` fits (#367).
- **Analytic FOCE and FOCEI outer gradient** for analytical 1-/2-/3-compartment
  models (IV bolus/infusion, oral, and steady state): the gradient-based outer
  optimizers (`bfgs`, `lbfgs`, `nlopt_lbfgs`, `slsqp`) now drive both FOCEI and
  FOCE with an exact closed-form marginal gradient (Almquist et al. 2015), evaluated
  through hand-rolled second-order dual numbers — no finite differences and no
  Enzyme. FOCEI differentiates the Laplace marginal (Eq. 23); FOCE differentiates
  ferx's Sheiner–Beal linearized marginal — both carry the exact EBE response
  (Eq. 46) on every θ/Ω/σ block, share an exact inner-loop Jacobian, and use an
  EBE warm-start predictor (Eq. 48). Estimates and OFV are unchanged, but the
  gradient is exact: it carries the EBE response in closed form, so `lbfgs`/
  `nlopt_lbfgs` reach the true optimum where the previous fixed-EBE FD gradient
  stalls short (warfarin FOCEI: −286.00 vs −281.83) — and do so ~13× faster than
  the only FD setting that also converges (`reconverge_gradient_interval = 1`:
  0.30 s vs 4.11 s). Validated against NONMEM on warfarin (FOCE OFV −280.36,
  FOCEI −286.00 — both matching to ~4–5 significant figures).
  Models outside the analytical scope (ODE models, steady-state edges) transparently
  fall back to the existing finite-difference gradient (#367).
- **Analytic FOCE/FOCEI outer gradient for time-varying covariates** on the
  analytical 1-/2-/3-compartment models. A covariate that changes within a subject
  (e.g. an allometric `(WT/70)^θ` on CL with a time-varying weight) makes the PK
  parameters switch mid-decay, which dose superposition cannot express; these
  subjects now route through the second-order-dual event-driven walk, with each
  event's PK-parameter derivatives evaluated at that event's covariate snapshot.
  The walk handles covariate breakpoints carried by EVID=2 records between
  observations, combined with EVID 3/4 resets, with **steady-state dosing** (each
  occasion's SS state is equilibrated at the dose's covariate snapshot), with a
  **constant `obs_scale` divisor**, and with **inter-occasion variability (IOV)**
  (the covariate and κ both switch the individual parameters across occasions).
  The result is the standard `(η, θ)` jet, so the exact θ/Ω/σ packed gradient
  (incl. the covariate coefficients and the EBE response) is assembled unchanged.
  Validated against reconverged finite differences (~1e-6 on every packed
  parameter, FOCEI and FOCE), against finite differences of the production
  predictor across 1-/2-/3-cpt (incl. SS, the constant scale, and the IOV+covariate
  merge with an EVID=2 breakpoint), and end-to-end on a simulated WT-on-CL dataset.
  Requires a gradient-based outer optimizer (`lbfgs`/`bfgs`/`slsqp`); the analytic
  *inner* EBE gradient still uses finite differences for these subjects. Time-varying
  covariates combined with **dose lagtime** or with **expression-based output
  scaling** (`obs_scale = <expr>` referencing parameters/covariates) still fall back
  to the finite-difference gradient (#367).
- **Analytic FOCE/FOCEI outer gradient for inter-occasion variability (IOV)** on
  the analytical 1-/2-/3-compartment models. The exact closed-form marginal
  gradient now covers κ (kappa) random effects: the EBE response, inner Jacobian,
  and θ/Ω/σ packed blocks are assembled over the stacked random-effects vector
  `[η_bsv, κ_occasion₁, …, κ_occasion_K]` with the block-diagonal prior
  `Ω_bsv ⊕ K·Ω_iov` (the shared per-occasion κ-variance). Cross-occasion carryover
  is differentiated exactly through a second-order-dual event-driven walk (no
  superposition approximation, no finite differences). **EVID 3/4 resets /
  washout occasions** are supported on the IOV path as well: the walk zeros the
  state at each reset and rebuilds the following occasion. Validated against
  reconverged finite differences (~1e-6 on every packed parameter, FOCEI and
  FOCE) and against NONMEM on the warfarin IOV model (FOCEI OFV 307.8 vs 308.8,
  structural parameters within ~1%). Requires a gradient-based outer optimizer
  (`lbfgs`/`bfgs`/`slsqp`); IOV fits with steady-state doses still fall back to
  finite differences (#367).
- **Analytic gradient now covers log-transform-both-sides (LTBS) and constant
  output scaling** for the analytical PK models: the sensitivity provider applies
  the `g = ln(f)` jet transform (value, gradient, and Hessian via
  `∂²g/∂x∂y = f_xy/f − f_x·f_y/f²`) and the constant `obs_scale` divisor in closed
  form, so `log(DV) ~ additive(...)` and `[scaling] obs_scale = k` fits run on the
  exact analytic FOCE/FOCEI gradient instead of falling back to finite
  differences. Validated against NONMEM on the warfarin LTBS model: the
  gradient-based L-BFGS path reaches OFV −675.302 and recovers NONMEM's MLE to
  ~4 significant figures (#367).
- **`inner_optimizer` fit option** (`auto` | `bfgs` | `lbfgs` | `nelder_mead`)
  to pin the inner EBE optimizer explicitly. `auto` (default) preserves the prior
  behaviour (dense BFGS, switching to L-BFGS above 32 random effects); the other
  values force a single algorithm with no automatic switching (#367).
- **Analytic FOCE/FOCEI gradient for user-specified `[odes]` models** (issue #367,
  Option A): the same exact closed-form marginal gradient now covers hand-written
  ODE models, not just the analytical PK solutions. The compiled `[odes]` RHS is
  evaluated over hand-rolled second-order dual numbers through a generic bytecode
  VM, and a dual-state RK45 (value-based step control) propagates the exact
  PK-parameter sensitivities through the integration — no Enzyme, no finite
  differences of the integrator. Supported scope: IV **bolus and infusion** doses,
  **bioavailability F** (including estimated, any parameterization — log-normal,
  logit-normal, additive), `obs_cmt` or simple Form C (`y = central/V1`) readouts,
  static covariates, **EVID 3/4 resets / multi-occasion**, **non-zero `init(...)`
  initial conditions**, and up to 12 individual parameters. Models outside this
  scope (steady-state dosing, lagtime, built-in input-rate absorption, IOV, SDE,
  `obs_scale`/LTBS transforms, time-varying covariates) transparently fall back
  to the finite-difference gradient (#367).
- **Modeled infusion rate (`RATE=-1` → `R{cmt}`)** — NONMEM's coded `RATE=-1`
  now makes the infusion *rate* a `$PK`-style individual parameter `R{cmt}`
  (duration = `AMT/R{cmt}`), the mirror of the modeled-duration `RATE=-2`/`D{cmt}`
  support. Works on both the analytical `pk(...)` engine and `ode(...)` models;
  resolves per iteration/occasion and composes with `F`/lag/SS. A `RATE=-1` dose
  with no matching `R{cmt}` is a loud `E_MODELED_RATE_NO_PARAM` error (never a
  silent bolus), and a non-positive `R{cmt}` at the initial estimate warns
  (`W_MODELED_RATE_NONPOSITIVE`). This completes NONMEM coded-`RATE` support
  (#324). Under bioavailability `F ≠ 1` it holds the rate and scales the duration
  to `F·AMT/R{cmt}`, matching NONMEM for rate-defined infusions (#419, see
  **Changed**).
- M3 likelihood now supports above-LOQ/right-censored observations via `CENS=-1`,
  with `DV` carrying the ULOQ value (#297). A `CENS` value other than `-1`, `0`,
  or `1` now raises a `W_CENS_UNEXPECTED` data warning instead of being silently
  scored as censored.
- `imp_auto` / `impmap_auto` fit options (NONMEM `AUTO`), **on by default**:
  adaptive importance-sample count. `imp_samples` / `impmap_samples` is the
  *starting* count and is ramped up (×2 per iteration, capped at 10000) whenever
  the objective's Monte-Carlo standard deviation exceeds 1.0 (NONMEM `STDOBJ`),
  so high-dimensional / FREM fits reach a low-noise objective automatically
  instead of carrying a sample-count-dependent M-step bias. On the FREM workshop
  model (13 ETAs) this ramps 300→10000 and brings the absorption typical value
  from ~4.6 (fixed K=300) to ~3.0, matching NONMEM. Low-dimensional, well-sampled
  fits never trip the threshold, so there is no cost there; set `false` to pin
  the sample count (#411).
- IMP/IMPMAP now warn when the importance-sample count is low for the model
  dimension (`K < 100·n_eta`) or when a subject's proposal fully collapses
  (ESS ≈ 0). The self-normalized M-step moments carry a finite-sample bias that
  grows with dimension, so high-dimensional / FREM fits at the default sample
  count can converge to biased typical-value and Ω estimates; the warning
  recommends raising `impmap_samples` / `imp_samples` (#411).
- `frem_rao_blackwell` fit option (default `true`): toggle the Rao-Blackwellised
  FREM covariate-ETA integration in IMP/IMPMAP. Set `false` only to diagnose the
  RB path against the full-dimensional importance sampler (#406).
- **IIV on residual error (`iiv_on_ruv`)** — a random effect can now scale the
  residual error per subject (NONMEM `Y = IPRED + EPS*EXP(ETA)`). Declare an
  `omega` and reference it from `[error_model]` with `iiv_on_ruv = NAME`; the
  residual variance of every observation is multiplied by `exp(2*ETA_i)`.
  Supported under FOCEI, IMP, IMPMAP, and SAEM (non-interaction FOCE is rejected
  with a clear error). Previously such a random effect was silently dropped on
  import (#409).
- **Covariance step progress reporting** — under `verbose`, the covariance step
  now prints throttled per-loop progress (Hessian finite-difference points and
  the score cross-product) with a wall-clock ETA, e.g.
  `[covariance] Hessian 12/40 (~8s left)`, so long covariance computations are
  no longer silent.
- **Cancellable covariance step** — a `CancelFlag` tripped *during* the
  covariance step (not just before it) now cooperatively aborts the
  finite-difference Hessian and score-matrix loops and finishes the fit without
  standard errors (recording a warning), instead of running the cancelled work
  to completion.
- `impmap_mceta` fit option: multi-start MAP for IMPMAP (NONMEM `MCETA` equivalent),
  improving IS efficiency in high-dimensional models (e.g. FREM with ≥5 ETAs).
- Analytical Jacobian for FREM pseudo-observations: covariate rows in the FD
  Jacobian are overwritten with exact ∂Y/∂η values (0 or 1), eliminating noise
  that corrupted the IS proposal in high-dimensional FREM models.
- `iscale_min` / `iscale_max` fit options: adaptive IS proposal scaling (NONMEM
  `ISCALE_MIN`/`ISCALE_MAX` equivalent). Per-subject pilot search over log-spaced
  scale factors selects the proposal width that maximises ESS. Defaults: 0.1–10.0.
- `impmap_sobol` fit option: use Sobol quasi-random sequences (with Cranley-Patterson
  randomization) for IMPMAP IS draws instead of pseudo-random, giving more uniform
  coverage of the posterior. MVN proposals only; Student-t falls back to pseudo-random.
- Full off-diagonal omega standard errors for block omega via multivariate delta
  method on the Cholesky parameterization. `se_omega` is now the full lower
  triangle (length n_eta*(n_eta+1)/2) instead of diagonal-only. Added
  `omega_se_at()` helper for indexed lookup.
- Per-iteration IMPMAP parameter trace (`FitResult.impmap_trace`), analogous to
  NONMEM `.ext` file output. Opt-in via `impmap_trace = true` in `[fit_options]`.
- FREM (Full Random Effects Model) covariate analysis: `prepare_frem()` API
  transforms a base model + dataset into a FREM model with extended block omega,
  covariate pseudo-observations, and FREMTYPE dispatch in the likelihood. The
  covariates (and their continuous/categorical kind) are taken from the model's
  `[covariates]` block; the `covariates` argument is an optional subset filter
  over them (#194).
- **Zero-order absorption into the oral depot on analytical models** — a `RATE=-2`
  modeled duration `D1` (or an explicit positive-`RATE` infusion) into compartment 1
  of an analytical oral model (`one_cpt_oral` / `two_cpt_oral` / `three_cpt_oral`)
  now models zero-order release into the depot followed by first-order `KA`
  absorption into central, all on the closed-form engine — no `ode(...)` block
  needed (previously rejected at parse time). Validated against NONMEM 7.5.1
  `ADVAN2` (`$PK D1`) and against the ODE transcription across 1-/2-/3-cpt oral
  models. Per-compartment amounts in
  `sdtab`/`[derived]` are not available for those subjects (predictions are exact;
  a `W_DERIVED_CMT_ORAL_DEPOT_INFUSION_ANALYTICAL` warning flags it) (#400).
- `RATE=-2` (modeled infusion duration via a `D{cmt}` parameter) is now supported
  on **analytical** PK models, not just ODE models — declare a `D{cmt}` individual
  parameter and the closed-form infusion uses `rate = AMT / D{cmt}`, matching
  NONMEM's `$PK D{n}` (#394, follow-up to #324).
- **Full MCMC Bayesian estimation** (`method = bayes`, Gibbs-within-HMC, NONMEM
  `METHOD=BAYES` parity). Draws from the joint posterior `p(θ, Ω, Σ, {ηᵢ} | y)`:
  per-subject η block (block-MH, or gradient HMC on the analytic `Dual2` gradient
  with `n_leapfrog > 0`), conjugate inverse-Wishart Ω block, exact Gaussian
  full-conditional draw for mu-referenced θ, and a random-walk block for the
  remaining θ/σ. Reports posterior summaries (mean/sd/2.5%/median/97.5%) with
  split-R̂, ESS, and MCSE per parameter on `FitResult.bayes` and in the
  `.fit.yaml` `bayes:` section. Options: `bayes_warmup`, `bayes_iters`,
  `bayes_chains`, `bayes_thin`, `bayes_seed`. Supports BSV and zero-mean IOV
  (per-occasion `kappa`, with a conjugate inverse-Wishart `Omega_iov` draw).
  Validated against FOCEI and NONMEM `METHOD=BAYES` on warfarin (#380).
- **Modeled infusion duration (`RATE=-2` → `Dn`) for ODE models** — NONMEM's
  `RATE=-2` makes a zero-order infusion's *duration* a modeled parameter: name an
  individual parameter `D{n}` for the dose compartment `n` and ferx infuses `AMT`
  over that duration (rate `AMT/Dn`), resolved per iteration and occasion (so it
  can carry covariate effects and IOV). Composes with `F{n}` (applied exactly
  once — `F·AMT` over `Dn`) and `ALAG{n}` (shifts the window; `Dn` sets its
  length), and works with steady state, multi-dose, and system resets. A
  `RATE=-2` dose with no matching `D{n}` parameter — or on an analytical model —
  is now a loud error rather than a silent bolus (the original #324 bug), both at
  the model+data join (`fit`/`ferx check`) and at the `predict()`/`simulate()`
  entrypoints (which skip the full data-check). A modeled `D{n}` that is
  non-positive at the initial estimate is flagged with a `W_MODELED_DURATION_NONPOSITIVE`
  warning (use a positive link such as `exp`). `RATE=-1` (modeled *rate*, `Rn`)
  and analytical-engine support remain tracked #324 follow-ups (#324).
- **Simulation-based NPDE / NPD diagnostics** in the `sdtab` output. Set
  `[fit_options] npde_nsim = 1000` (and optionally `npde_seed`) to add `NPDE`
  (Normalized Prediction Distribution Errors, decorrelated within subject) and
  `NPD` (Normalized Prediction Discrepancies) columns, computed post-fit by
  Monte-Carlo simulation under the fitted model (Brendel et al. 2006; Comets et
  al. 2008). Unlike CWRES, these are robust to model nonlinearity and non-Gaussian
  random effects, and follow N(0,1) under a correctly specified model. Off by
  default (`npde_nsim = 0`). The effective simulation seed (including the default
  when `npde_seed` is unset) is recorded as `npde_seed` in `{model}-fit.yaml` and
  the `.fitrx` archive, so the diagnostics are reproducible from the saved fit.
  Validated against a NONMEM `$SIMULATION` + `npde` R-package reference on the
  warfarin example. M3/BLQ censoring and IOV-kappa resampling are out of scope
  (#260).
- **Compartment-indexed bioavailability and lag for ODE models** — name an
  individual parameter `F{n}` or `ALAG{n}`/`LAGTIME{n}` (e.g. `F2`, `ALAG2`) to
  apply a per-route bioavailability/lag to doses into compartment `n`, mirroring
  NONMEM's `F1`/`F2`/`ALAG1`/`ALAG2`. A bare `F`/`lagtime` stays the
  all-compartment default (existing single-route models are unchanged); an
  indexed value overrides only its compartment. Resolved uniformly across every
  ODE dose-application path (event-driven, steady-state, and the EKF/diffusion
  path — the latter applies `F` but not lag). An index past the model's
  compartment count is a parse error rather than a silently-ignored parameter.
  Foundation for the modeled-duration/rate (`Dn`/`Rn`) work in #324 (#369).
- **`ode_template NAME(...)`** in `[structural_model]` generates the standard
  disposition ODE for a named model (`one/two/three_cpt_iv|oral`) from the same
  closed-form↔ODE transcription the analytical `pk NAME(...)` uses — so you get
  the explicit, runnable ODE form without hand-writing the states, RHS, and
  `obs_scale`. It takes the same parameters as `pk NAME(...)` (including `ka` for
  oral routes). Re-declaring a `d/dt(X)` in `[odes]` **overrides** the generated
  equation for compartment `X` (e.g. to add a `transit(...)` absorption input);
  undeclared compartments keep their generated equations. Combining the ODE-only
  `transit(...)` absorption with an analytical `pk NAME(...)` is now a clear error
  pointing at `ode_template`, never a silent analytical→ODE conversion. (Future
  ODE-only absorption functions join that error rule as each is implemented.)
  (#322).
- Built-in **transit-compartment absorption** for ODE models via a `transit(n, mtt)`
  input-rate function in the `[odes]` block (Savic et al. 2007, continuous `n`):
  `R_in(tad) = F·Dose·KTR·(KTR·tad)^n·e^(−KTR·tad)/Γ(n+1)`, `KTR=(n+1)/mtt`. The
  dose is delivered as this appearance rate into the depot (∫R_in dt = F·Dose) —
  not also as a bolus — so a flexible, continuously-estimable absorption shape
  takes one line instead of a hand-coded transit chain. Honors `F`/lagtime and
  superposes over doses; works with IIV/IOV, resets, and time-varying covariates.
  Unsupported combinations are rejected with a clear error rather than silently
  mis-modeled: steady-state dosing into a transit compartment (`E_ABSORPTION_SS`),
  an infusion (`RATE>0`) into a transit compartment (`E_ABSORPTION_RATE`, which
  would double-count the dose), a `[diffusion]` block together with `transit()`
  (`E_ABSORPTION_DIFFUSION`), and an out-of-domain `mtt`/`n` at typical values
  (`E_ABSORPTION_DOMAIN`). New example `examples/transit_savic.ferx` and docs
  page *Built-in Absorption Models* (#322).
- Built-in **inverse-Gaussian (Freijer & Post) absorption** for ODE models via an
  `igd(mat, cv2)` input-rate function in the `[odes]` block:
  `R_in(tad) = F·Dose·√(MAT/(2π·CV2·tad³))·exp(−(tad−MAT)²/(2·CV2·MAT·tad))`, the
  inverse-Gaussian density with mean absorption time `MAT` and relative dispersion
  `CV2` (shape `λ = MAT/CV2`). Models the entire absorption delay and feeds the
  central compartment directly (no first-order `ka`); `∫R_in dt = F·Dose`. Reuses
  the same dose routing, `F`/lagtime, superposition, IOV, domain validation
  (`mat>0`, `cv2>0`), and unsupported-combination guards as `transit()`; the
  essential singularity at `tad→0` is handled (`R_in→0`). NONMEM-anchored against a
  `$DES` IG run (`nonmem_anchor/freijer_ig.ctl`). New example
  `examples/igd_inverse_gaussian.ferx`. The biphasic Freijer sum-of-two is a
  planned follow-up (#347, #388).
- Example `dose_rate.ferx` (+ `data/dose_rate.csv`) demonstrating the supported
  NONMEM `RATE` dosing forms — a bolus (`RATE=0`) and a constant-rate infusion
  (`RATE>0`) mixed in one dataset (#324).
- Configurable RK45 ODE solver tolerances via `[fit_options]` (and call-time
  settings): `ode_reltol` (default `1e-4`), `ode_abstol` (default `1e-6`), and
  `ode_max_steps` (default `10000`). Defaults are unchanged, so existing fits
  are unaffected. Previously the tolerance was hardcoded, which made the OFV of
  an ODE-form model differ from its analytical equivalent by several units
  (the FOCE objective amplifies the ~`1e-4` solver error); a tighter
  `ode_reltol` now lets the two forms agree. Carried on `OdeSpec::solver_opts`
  and applied via `CompiledModel::sync_ode_solver_opts` (#127).
- `parameter_scaling` fit option (`none` / `abs` / `rescale2`): parameter
  scaling for the outer optimizer. `rescale2` is the nlmixr2-style
  bound-half-width normalisation (maps each packed parameter toward `(−1, 1)`)
  and substantially improves cold-start convergence for gradient-based
  optimizers on ill-conditioned multi-parameter surfaces — e.g. `bfgs` reaches
  OFV −1198.97 on `two_cpt_oral_cov` (≈ nlmixr2's −1199.24) where the unscaled
  optimizer stalls near −1192. The default `auto` applies `rescale2` to the
  gradient-based optimizers (`bfgs`/`lbfgs`/`nlopt_lbfgs`/`slsqp`) and leaves the
  derivative-free `bobyqa` unscaled (where `rescale2` distorts its trust region)
  (#341).
- `covariance_ofv_hessian` fit option: build the covariance R-matrix from second
  differences of the reconverged marginal OFV instead of a central difference of
  the analytical population gradient. The analytical stencil holds the H-matrix
  `a = ∂f/∂η` fixed in the `log|H̃|` θ-gradient, biasing the SE of
  weakly-identified structural parameters (e.g. warfarin TVKA reads ~9% high
  versus a Richardson FD-of-OFV ground truth); the OFV-Hessian stencil recomputes
  `a` at every perturbed point and matches the ground truth to <1%, at ≈ the same
  wall-clock cost (both stencils parallelise over perturbation points). Default
  `true`; set `false` to force the faster analytical-gradient stencil (#335).
- Propensity-score-matched simulation: `simulate_with_options()` with a new
  `SimulateOptions { seed, match_method }`. When `match_method` is `Some(..)`,
  each replicate's drawn etas are reassigned to subjects by Mahalanobis matching
  (under the model Ω) against the subjects' fitted (posthoc) etas, so a subject's
  observed dosing/sampling design is paired with a similar drawn eta. This
  corrects VPC bias from treatment adaptation in real-world data (longer
  intervals for high-clearance patients, etc.). Three methods are offered via
  `MatchMethod`: `Optimal` (global linear-assignment minimum; best on average in
  simulation, recommended default), `Nearest` (greedy nearest-neighbour,
  `MatchIt(method="nearest", distance="mahalanobis")`), and `Rank` (pair by the
  rank of the Mahalanobis norm). Operates on observed data; returns the usual
  simulation rows for the caller to build the VPC (#288, #396).
- New `importance_sampling_map` (alias `impmap`) estimation method: a Monte-Carlo
  EM estimator equivalent to NONMEM `METHOD=IMPMAP`. Each iteration re-centers a
  per-subject importance-sampling proposal on the conditional mode (MAP) and
  updates θ/Ω/σ from the importance-weighted posterior moments. Runs standalone
  or chained (`methods = [focei, impmap]`); multivariate-normal proposal by
  default (`impmap_proposal_df = normal`), Student-t optional. Validated against
  FOCEI on warfarin. IOV and SDE models are not yet supported (#270).
- Importance sampling can now run **standalone** (`method = imp`), evaluating the
  IS log-likelihood at the initial parameters — IMP derives the EBEs/Jacobian it
  needs via a FOCE inner loop at those parameters instead of requiring a
  preceding estimator. Useful for scoring imported/fixed parameter sets. IMP
  still may appear at most once and must be the terminal stage of a chain.
- SAEM conditional-distribution pass: set `conddist = true` in `[fit_options]`
  to estimate each subject's conditional distribution of the random effects
  `p(η_i | y_i)` by MCMC after the fit — reporting per-subject conditional mean,
  SD, distribution-based η-shrinkage, and (with `conddist_keep_samples = true`)
  the raw draws. Surfaced on `FitResult.cond_dist` and written to
  `{model}-conddist.csv` (+ `-conddist-samples.csv`). This is the SAEM analogue
  of saemix `conddist.saemix` / Monolix's "Conditional Distribution" task and is
  the shrinkage-unbiased basis for η diagnostics; validated against saemix on
  warfarin (#257).
- Feature maturity labels (`stable` / `beta` / `experimental`) documented for
  every major feature: a new *Feature Maturity* docs page with definitions and a
  per-feature table, plus a maturity banner on each feature reference page.
  Experimental features (`[diffusion]` / SDE, `[covariate_nn]` / neural networks)
  now emit a runtime warning at fit time (`W_EXPERIMENTAL_SDE`,
  `W_EXPERIMENTAL_NN`), also surfaced by `ferx check` (#175).
- `covariance_method` fit option: choose the covariance estimator, mirroring
  NONMEM `$COV MATRIX=` — `r` (inverse Hessian `R⁻¹`, default), `s` (inverse
  score cross-product `S⁻¹`), or `rsr` (the Huber–White sandwich `R⁻¹SR⁻¹`,
  robust to model mis-specification). Supported for FOCEI, FOCE, and IOV fits;
  anchored against NONMEM `$COV MATRIX=S`/`RSR` within ~10% for both FOCEI (#266)
  and FOCE (#250) (#223).
- `covariance_fallback = sir` fit option: when the FD Hessian is non-positive-definite,
  run SIR with an `|eigenvalue|`-rectified proposal (4× inflated) instead of leaving
  the covariance step as failed; `covariance_status` reports `sir_fallback` (#223).
- `covariance_matrix:` block in `*-fit.yaml`: the full optimizer-space parameter
  covariance matrix (log-theta, Cholesky-omega, log-sigma; kappa appended for IOV
  models), parameter-labelled, emitted when the covariance step succeeds or is
  regularised. Omega/kappa diagonal entries are keyed `log_chol_<eta>` (packed
  value is `log(L_ii)`); off-diagonal entries are keyed `chol_<row>_<col>`
  (`L_ij`, not log-transformed) (#236).
- Time-to-event / survival modelling (Phase 1): `[event_model]` block, TTE
  datareader, likelihood, and API wiring, behind the `survival` feature
  (#191, #192).
- `[data_selection]` block with NONMEM-style `IGNORE`/`ACCEPT` record filtering,
  plus an `ExclusionSummary` on `FitResult` surfaced in the CLI and YAML output.
- Combined ferx-core + ferx-r development documentation: a Development Lifecycle
  (SDLC) page and a Contributing page in the book.
- `[structural_model]` now warns when a `pk(...)` line maps a parameter the
  chosen model does not use (e.g. `ka` or `f` on an IV model, or `q`/`v2` on a
  one-compartment model); the mapping is accepted but has no effect (#309).
- `[individual_parameters]` now warns when a declared parameter is computed but
  never used — neither mapped into the `pk(...)` model nor referenced in any
  other block (e.g. declaring `F` but forgetting `f=F`); it silently has no
  effect (#309).
- `MACHEPS` (machine epsilon) is now available in `[odes]` RHS and `init(...)`
  expressions, matching its existing availability in `[derived]` (#314).
- The "computed but never used" warning above now also covers **ODE models**: an
  `[individual_parameters]` entry never referenced in the `[odes]` right-hand
  side (nor in `[scaling]`/`[derived]`/`[output]`) is flagged the same way. The
  engine-applied `F` (bioavailability) and `lagtime` (alias `alag`), which act on
  the dose without appearing in the RHS, are exempt (#315).

### Changed
- **Bioavailability `F` now reshapes a rate-defined infusion the NONMEM way**
  (`RATE>0` data and `RATE=-1` → `R{cmt}`): `F` holds the rate and scales the
  **duration** to `F·AMT/RATE`, instead of scaling the rate over a fixed duration.
  A duration-defined infusion (`RATE=-2` → `D{cmt}`) is unchanged — `F` still
  scales its rate. Total exposure (`F·AMT`) is unchanged in both cases; only the
  infusion **shape** changes, and only for an existing `RATE>0`/`RATE=-1` infusion
  with `F ≠ 1`. Predictions, simulations, and fits for such models will differ;
  models with `F = 1`, bolus, oral-depot, or `RATE=-2` dosing are unaffected. This
  aligns all engines (analytical superposition, event-driven, ODE, analytic
  sensitivities) with NONMEM's `RATE`/`F` convention (#419, follow-up to #327/#324).
- **`method = foce` with M3 BLOQ no longer promotes censored subjects to FOCEI.**
  Previously a subject with any `CENS=1` row was silently evaluated with
  η-interaction (mixing a Sheiner–Beal FOCE objective with a FOCEI censored term).
  Plain FOCE now keeps a consistent Sheiner–Beal objective for the whole subject,
  with censored rows entering as `−logΦ((LLOQ−f̂)/√R⁰)` (population variance,
  excluded from `R̃`). FOCE-M3 and FOCEI-M3 are genuinely different optima — on
  warfarin BLOQ, FOCE TVKA ≈ 0.71 vs FOCEI ≈ 0.81, each matching the corresponding
  NONMEM `METHOD=1 LAPLACE` (with/without INTER) fit. M3 fits that relied on the
  old auto-promotion should set `method = focei` explicitly (#367).
- Bumped `nalgebra` to 0.35 (from 0.34). The `argmin-math` dependency now uses
  its `vec` feature instead of `nalgebra_latest`, since the argmin trust-region
  path operates on `Vec` params and never on `nalgebra` types — this avoids
  pulling a second, conflicting `nalgebra` version into the graph. Downstream
  Rust consumers (e.g. `ferx-r`) must move to `nalgebra` 0.35 in lockstep.
- IMP fit options now use the `imp_*` prefix (`imp_samples`,
  `imp_eval_only`, `imp_auto`, etc.) instead of the older `is_*` names. The
  old names are not retained as aliases because IMP support is still new.
- SAEM no longer automatically runs a FOCEI polish when a combined-error
  additive sigma collapses; it now leaves the SAEM estimate unchanged and records
  a warning that the additive component hit its lower bound (#420).
- **IMPMAP default proposal is now a Student-t** (`impmap_proposal_df = 4`)
  instead of a multivariate normal. A Gaussian proposal's tails are lighter than
  the posterior of weakly-identified parameters, so importance weights blow up in
  the tail and bias the M-step moments — drifting typical-value estimates (e.g.
  the absorption `MAT`/`KA` on modeled-duration models). The heavier-tailed
  default removes that bias and matches FOCEI/NONMEM. Set `impmap_proposal_df =
  normal` for the previous behaviour (#411).
- **IMP/IMPMAP now warn about estimated parameters with no random effect**: any
  non-fixed `theta` that has no associated `ETA` is estimated only through the
  importance-weighted M-step, which is biased for weakly-identified parameters and
  can converge to the wrong value (e.g. a FREM absorption fraction drifting to ~0.9
  vs a FOCEI/NONMEM value of ~0.4). The estimator now emits a strong warning naming
  such parameters and recommending an `ETA` be added (ferx mu-references
  automatically), the parameter be held `FIX`, or FOCEI be used. `prepare_frem`
  (`ferx_to_frem`) also surfaces this advisory at conversion time via a new
  `FremPrepareResult.warnings` field, so it shows up before fitting. (#406)
- **IMP/IMPMAP now Rao-Blackwellise FREM covariate ETAs**: the Gaussian covariate
  pseudo-observation ETAs are integrated analytically (conditional PK prior from
  the Ω precision blocks) and only the PK ETAs are importance-sampled. This turns
  the high-dimensional, multi-scale IS (≈1–2% effective sample size, unstable
  M-step) into a well-conditioned low-dimensional one: on the workshop 12-ETA FREM
  the share of low-ESS subjects dropped from ~80% to ~23%, the −2logL trajectory
  is smooth (no spikes), and estimates land near NONMEM (TVCL 6.7 vs 6.97, TVMAT
  2.8 vs 2.75). Automatic for FREM models; falls back to full-dimensional IS if
  the PK/covariate partition is degenerate. (#406)
- **`imp` is now a Monte-Carlo EM estimator by default** (NONMEM `METHOD=IMP`
  parity): `method = imp` updates θ/Ω/σ instead of only evaluating the marginal
  `−2 log L`. **Breaking:** model files that used `imp` (e.g. `[focei, imp]`)
  purely to *score* a fit now re-estimate. Add `imp_eval_only = true` (NONMEM
  `EONLY=1`) to recover the previous evaluation-at-fixed-parameters behaviour.
  New options `imp_iterations` (default 200) and `imp_averaging` (default 50)
  control the MCEM loop; `imp_proposal_df` now also accepts `normal`/`mvn`. The
  estimating `imp` may lead or sit mid-chain; the evaluation-only `imp` must
  still be terminal. Plain `imp` re-centers its proposal from the previous
  iteration's sample moments and so is fragile on rich data (warm-start with
  `[focei, imp]`, or use `impmap`); validated against NONMEM 7.5.1 `METHOD=IMP`
  on warfarin (#402).
- The analytical `pk NAME(...)` parameter list is now parsed strictly: a malformed
  `role=VAR` pair (no `=`, an empty side, or a stray extra `=`) or a duplicate role
  is a clear parse error instead of being silently dropped or last-winning. The
  `pk` and `ode_template NAME(...)` directives share one strict parser, so they
  can't drift in strictness. Well-formed model files (including a tolerated
  trailing comma) are unaffected (#363).
- FOCEI gradient-based optimizers (SLSQP, L-BFGS, built-in BFGS, Gauss-Newton)
  now add the `log|H̃|` EBE-response term (the #274/#289 Δ) to the population
  gradient, so they reach the true marginal minimum instead of stalling above it
  on the fixed-EBE gradient (e.g. warfarin FOCEI −282.8 → −286.0, matching the
  derivative-free BOBYQA default). The term reuses the Laplace intermediates the
  gradient already forms (one extra `n_eta×n_eta` solve per subject) and is zero
  for additive error; the BOBYQA default is unaffected (it uses no gradient). The
  ω-block of the correction remains deferred (#335) (#330).
- The default inner (per-subject EBE) convergence tolerance `inner_tol` is now
  `1e-5` (was `1e-4`). A looser inner tolerance left residual noise in each
  subject's EBE solution that propagated into the marginal objective, causing the
  derivative-free BOBYQA outer optimizer to false-converge above the true
  minimum on noisy-marginal models (notably log-transform-both-sides FOCE). The
  tighter default matches NONMEM's minimum at roughly 1.5× the per-fit cost;
  loosen it via `inner_tol` in `[fit_options]` to recover the old speed on
  well-conditioned fits (#330).
- FOCE (non-interaction) now evaluates the residual variance at the population
  prediction `f(η=0)` — NONMEM's `METHOD=1` (no `INTER`) semantics — instead of
  the linearized `f0 = f(η̂) − H·η̂`. On nonlinear models (e.g. oral absorption)
  with proportional/combined error, `f0` could extrapolate to near-zero or
  negative concentrations, collapsing `R(f0) = (f0·σ)²` and making the marginal
  multimodal with an indefinite covariance Hessian (garbage SEs reported as
  "likely reliable"). FOCE+proportional fits now converge deterministically,
  reproduce NONMEM FOCE estimates/SEs (within ~3% on a 1-cpt oral benchmark),
  and yield a positive-definite covariance. Additive-error FOCE is unchanged
  (its variance is `f`-independent). The FOCE covariance for `f`-dependent error
  uses the reconverged-OFV second-difference Hessian (the true objective
  curvature) rather than the envelope-approximation analytical gradient (#319).
- IMP (importance sampling) now jointly samples (η, κ) for IOV models,
  integrating over inter-occasion variability so the reported `−2 log L` is
  directly comparable to FOCE/FOCEI and NONMEM `METHOD=IMP`. Previously κ was
  held fixed at its EBE mode, giving a partial marginal; `kappa_treatment` in
  the fit YAML is now `marginalized` rather than `fixed_at_mode` (#186).
- A `[structural_model]` `pk(...)` line that omits a required parameter for the
  chosen model (e.g. `ka` for `one_cpt_oral`) is now a parse error naming the
  missing parameter, instead of silently defaulting that slot to `0.0` and
  fitting to a structurally broken optimum (#309).

### Fixed
- **M3 BLOQ fits with a gradient-based optimizer no longer stall above the true
  minimum.** Previously the analytic outer gradient declined on censored subjects
  and the fixed-EBE finite-difference fallback was biased there, so on warfarin
  BLOQ a gradient optimizer settled at TVKA ≈ 1.10 / OFV ≈ −213.8 while the
  derivative-free BOBYQA reached the true TVKA ≈ 0.81 / OFV ≈ −217.2. FOCEI now
  has an **exact closed-form M3 censored gradient** (see Added), and plain FOCE
  with M3 forces the EBE-reconverging gradient automatically (as IOV already
  does), so every optimizer reaches the minimum and matches a NONMEM 7.5.1 LAPLACE
  M3 reference (TVCL 0.1328, TVV 7.731, TVKA 0.810, to ~4 significant figures). The
  `docs/src/examples/bloq.md` expected results, which showed the stalled point,
  are corrected (#367).
- **IMPMAP warns instead of silently ignoring `impmap_sobol` under a Student-t
  proposal.** Sobol draws apply only to the multivariate-normal proposal; with
  the Student-t default `impmap_sobol = true` was a no-op. It now emits a warning
  pointing to `impmap_proposal_df = normal` (#406).
- **FREM Rao-Blackwell sampler falls back to full-dimensional IS for covariates
  with more than one pseudo-obs row.** A time-varying or duplicated covariate
  row broke the closed-form covariate-likelihood cancellation in the RB marginal;
  such subjects now use the full-dimensional sampler, which scores every row
  consistently (#406).
- **Adaptive-sampling (`imp_auto`/`impmap_auto`) trigger is now per-subject.** It
  used the total-objective Monte-Carlo SE, which grows as √N, so a large but
  well-sampled dataset could ramp the sample count to the cap purely from subject
  count. The trigger now normalizes by √N (per-subject objective SE), making it
  N-independent (#411).
- **IMP/IMPMAP no longer freeze the typical value of a mu-referenced parameter
  with negligible IIV**: a log-mu-referenced θ (e.g. `KA = TVKA*exp(ETA_KA)`)
  whose random effect has a tiny, often `FIX`ed ω was updated only through the
  closed-form `log θ += mean(η)` shift — which is ≈ 0 when the η carries no
  variance, leaving the typical value stuck at its initial value. Such
  parameters are now routed to the weighted-likelihood M-step (the channel that
  estimates σ and non-mu-ref θ), so the data can move them; a warning names any
  parameter routed this way. Makes the estimate init-independent (#411).
- **FREM IMP/IMPMAP marginal −2 log L over-counted by a 2π constant**: the
  Rao-Blackwellised covariate-data marginal included the covariate pseudo-obs
  `nc·ln(2π)` normalizer, which the rest of the objective (and NONMEM's
  "OBJECTIVE FUNCTION WITHOUT CONSTANT") drops. This inflated the reported FREM
  marginal by `Σ nc·ln(2π)` (≈ n_covariate_obs · ln2π) and made the
  Rao-Blackwell and full-dimensional importance samplers disagree on the same
  point. The constant is now dropped in both; the value is otherwise unchanged
  (it lies outside the importance weights, so estimates were never affected) (#406).
- **IMP/IMPMAP now report the NONMEM-comparable objective**: estimating `imp` and
  `impmap` runs surface the importance-sampling Monte-Carlo *marginal* −2 log L —
  the number NONMEM `METHOD=IMP`/`IMPMAP` reports as its `#OBJV` — evaluated at the
  final estimates on `FitResult.importance_sampling.minus2_log_likelihood` (± MC
  SE). Previously this was populated only by the evaluation-only path, so the only
  available number was the FOCE-Laplace `ofv`, which matches NONMEM's *COND/FOCE*
  OBJ rather than the IMP marginal and diverges from it on sparse / strongly
  nonlinear data. `ofv` is unchanged (still a Laplace pass, for cross-method
  AIC/BIC comparability) (#406).
- **IMP/IMPMAP no longer diverge on FREM models with missing covariates**: the
  Rao-Blackwellised E-step previously bailed to the unstable full-dimensional
  importance sampler for any subject missing a covariate pseudo-observation row
  (the FREM data omits rows for missing covariate values — ~28% of subjects on
  the workshop model). Those subjects then blew the −2logL up to ~1e14 within a
  few iterations under `method = imp`. Missing-covariate etas (which have no data)
  are now sampled together with the PK etas, conditioning only on the *observed*
  covariates; both IMP and IMPMAP now converge with near-zero low-ESS subjects and
  agree on the estimates. (#406)
- **FREM covariate pseudo-observations are no longer clamped to a positive
  prediction**: the observation likelihood clamped every prediction to `≥1e-12`,
  but a FREM covariate pseudo-obs predicts a covariate *value* (centered,
  standardized, or log-scale covariates are routinely `≤0`). Clamping a
  non-positive covariate prediction fabricated a huge residual, which corrupted
  the Rao-Blackwellised IS marginal/weights for affected subjects. Covariate rows
  now keep their (possibly negative) prediction; ordinary PK rows keep the
  positivity clamp. (#406)
- **FREM model generation dropped the `[scaling]` / `[odes]` blocks**: `prepare_frem`
  now carries the base model's `[scaling]` (e.g. `obs_scale`) and `[odes]` blocks
  into the generated FREM model. Previously they were silently omitted, so a base
  model with `obs_scale` (NONMEM `CP = A*1000/V`) produced a FREM model whose
  predictions were mis-scaled; the estimator then compensated by collapsing a PK
  typical value (TVCL → ~1e-2 instead of ~7 on the workshop FREM model, now ~6.6
  vs NONMEM 6.97). (#406)
- **IMP/IMPMAP on high-dimensional FREM**: the inner EBE/MAP solver no longer
  returns a nonsensical joint mode on multi-scale FREM posteriors (3 PK + many
  covariate ETAs). The inner BFGS is now FREM-preconditioned (per-dimension
  initial inverse-Hessian ≈ posterior variance) and the covariate ETAs are
  cold-started at their data-implied mode `cov_obs − TV`; the IS proposal jitter
  is now per-dimension instead of a single global value. Previously the mode
  collapsed (obs-NLL ~1e8) and standalone IMP/IMPMAP diverged (−2logL ~1e13) on
  ≥8-covariate FREM models; the typical-value estimates for volume and absorption
  now recover. (Full NONMEM parity still pending the mu-referencing θ M-step and
  high-dimensional IS effective-sample-size work — see #406.) (#406)
- **Bayesian estimation** (`method = bayes`) now samples the per-occasion IOV
  `kappa` block when `OMEGA_IOV` is FIX-ed. Previously an all-FIX `OMEGA_IOV`
  disabled kappa sampling entirely, so the kappas stayed pinned at their initial
  values (IOV effectively ignored); a fixed `OMEGA_IOV` still defines the kappa
  prior variance, so the block is now sampled while its conjugate covariance
  draw remains correctly skipped (#415).
- **Bayesian estimation** (`method = bayes`) now responds to a cooperative
  cancellation (e.g. an R-session interrupt): the Gibbs sampler polls the cancel
  flag at each sweep boundary and aborts within one sweep, returning
  `cancelled by user` instead of running every chain to completion. Previously a
  Bayes run could not be stopped once started (#393).
- **IMPMAP** now responds to a cooperative cancellation (e.g. an R-session
  interrupt) during an iteration's E-step, instead of only at iteration
  boundaries. The importance-sampling pass — the dominant per-iteration cost on
  large datasets — previously ran to completion before the cancel flag was
  checked, so a kill request could appear to hang for minutes; the E-step now
  polls per subject and the run aborts promptly (#273).
- An individual parameter assigned only inside symmetric `if`/`else` branches in
  `[individual_parameters]` (the NONMEM-style `IF (cond) CL = ...` /
  `IF (!cond) CL = ...` construction) on an **ODE model** is no longer rejected
  by the `[odes]` RHS validator as an undefined name. A name written on every
  branch is now promoted to a real individual parameter — getting a PK slot,
  being written back, and resolving in the ODE RHS — provided a downstream block
  (`[odes]`, `[structural_model]`, `[scaling]`, `[derived]`) actually references
  it. Purely internal branch helpers stay branch-local and never consume a PK
  slot (#357).
- The covariance-family fit options `covariance_method`, `covariance_fallback`,
  and `covariance_ofv_hessian` no longer emit a spurious "is not used by method
  `<method>` and will be ignored" warning. They are framework-wide covariance-step
  options (honoured for every estimator) but were missing from the warning's
  allowlist; the options were always applied — only the warning was wrong.
- A missing `DV` (`.`/`NA`/blank) on an `EVID=0` observation row without `MDV=1`
  is no longer silently scored as `DV=0`. Such rows are now treated as `MDV=1`
  (skipped) and a single `W_MISSING_DV` warning reports how many rows were
  skipped, surfaced in fit warnings and `ferx check` (#258).
- Bioavailability `F` is now applied to **IV bolus and infusion** doses on the
  analytical path, not just oral depot doses. The analytical superposition path
  (used for subjects with no time-varying covariates) previously dropped `F` for
  IV/infusion dosing, so the same model gave `F`×-different predictions for a
  no-TV subject versus a time-varying/IOV subject (the event-driven path applied
  `F` correctly) — a silent inconsistency that biased fits and made an estimated
  `F` a no-op on all-IV/infusion datasets. `F` now scales the bioavailable
  amount/rate on every route, matching NONMEM's `F1`, the ODE engine, and the
  event-driven path. Mapping `f=` on an IV model is no longer warned as unused
  (#327).
- Infusion (zero-order, `RATE>0`) doses into the central compartment of an
  **oral** model are no longer silently dropped on the event-driven analytical
  path. The oral propagators ignored the infusion input rate, so a depot-bypass
  infusion produced ~0 concentration for any subject routed through the
  event-driven path (time-varying covariates, EVID=3/4 resets, or IOV) — while
  no-covariate subjects (superposition path) got the correct curve. The oral
  propagators now carry the central zero-order input by linear superposition,
  matching the superposition path and NONMEM. (Infusion into an oral *depot*
  compartment, `cmt=1`, remains an explicit error rather than silently bypassing
  the depot.)
- NONMEM coded `RATE` values (`-1` = modeled rate, `-2` = modeled duration) — and
  any other negative or non-finite `RATE` on a dose row — are now rejected with an
  informative error naming the subject and time, instead of being silently treated
  as an IV bolus (which produced wrong predictions with no warning). Modeled
  rate/duration support is not yet implemented; convert such rows to an explicit
  positive `RATE` (= `AMT`/duration) before importing (#324).
- Cold-start FOCEI/SLSQP on IOV models now reaches the marginal minimum instead
  of stalling: under the default `parameter_scaling = auto`, `slsqp` now gets the
  `rescale2` bound-half-width scaling, so pure FOCEI/SLSQP on `warfarin_iov`
  converges to OFV 307.84 (ω_iov ≈ 0.046) from the cold default start rather than
  stalling at 343.5 with ω_iov pinned at its init (#335).
- FOCEI covariance score cross-product (`covariance_method = s` / `rsr`) now
  carries the `log|H̃|` EBE-response term (`½·∂log|H̃|/∂η̂·dη̂/dθ`, the #274 `tᵢ`):
  the per-subject score is differenced with the conditional estimate η̂ responding
  to the parameters, matching how NONMEM forms its S matrix. Previously the score
  held η̂ fixed (the R-matrix already captured this term via reconvergence, but S
  did not), so the RSR sandwich SEs were biased on weakly-identified structural
  parameters — warfarin SE(TVKA) ~5% out. With the term, FOCEI RSR matches NONMEM
  7.5.1 to <1.8% on every parameter (#335).
- A `[structural_model]` PK parameter that references a name not defined in
  `[individual_parameters]` (e.g. `pk one_cpt_oral(cl=CL, ...)` with no `CL`)
  is now a parse error instead of being silently dropped and defaulting the
  slot to 0.0 — which previously produced a "converged" but structurally broken
  fit (all predictions floored, 100% shrinkage). An unrecognized PK-parameter
  key (e.g. the typo `clx=`) is likewise rejected, and a numeric-literal value
  (e.g. `ka=1.0`) is now honored as a constant rather than dropped to 0.0 (#261).
- A name in an `[odes]` RHS or `init(...)` expression that is not a declared
  state, individual parameter, ODE-block intermediate, or reserved time variable
  (`TIME`/`TAFD`/`TAD`) is now a parse error instead of silently resolving to
  `0.0` — the ODE counterpart of the analytical guard above, which otherwise
  produced a "converged" but structurally broken fit (#314).
- Datasets without an `EVID` column no longer silently fit a dose-free model.
  ferx now infers a dose from a nonzero `AMT` when `EVID` is absent (matching
  NONMEM), so legacy datasets that mark doses only by `AMT`/`MDV=1` administer
  correctly. As a safety net, the reader also warns when `AMT != 0` rows are not
  treated as doses (`W_AMT_NOT_DOSED`) or when a population with observations
  parses zero dose events (`W_NO_DOSES`) (#262).
- Autodiff builds now fall back to finite differences for analytical models the
  single-snapshot AD kernel cannot represent faithfully: non-log-normal ETAs
  (additive / logit), conditional (`if`-branch) individual-parameter
  expressions, log-transform-both-sides (`log_additive`) error, eta-dependent
  `[scaling] obs_scale` expressions (e.g. `obs_scale = V`), and time-to-event
  (`[event_model]`) hazard likelihoods. The kernel hardcodes the log-normal map
  `param = tv*exp(eta)` (plus a log-wrap for LTBS, a subject-static eta-frozen
  `obs_scale`, and the PK NLL rather than the hazard term for TTE), so these
  previously
  produced inner gradients inconsistent with the objective - a small bias on
  well-conditioned data, but on ill-conditioned FOCEI-INTER fits a spurious
  variance-collapsed optimum with an OFV far below NONMEM's. FD-only CI never
  exercised the AD path, so the divergence went undetected (surfaced by an
  external NONMEM/OpenPMX/ferx benchmark, FeRx-NLME/ferx-r#154). The default
  non-autodiff build was never affected (#278).
- FOCEI covariance standard errors (non-IOV) now include the `log|H̃|` EBE-response
  curvature for mu-referenced structural parameters, bringing the non-IOV stencil
  in line with the IOV stencil and matching NONMEM `$COV MATRIX=R` more closely on
  models with η-dependent (proportional/combined) residual error. The fixed-η̂
  analytic gradient previously dropped this term — the envelope theorem zeros the
  inner objective but not `log|H̃|` — and the resulting SE gap grew with the
  proportional error magnitude. Additive-error SEs are unchanged (the correction is
  identically zero when `∂R/∂f = 0`) (#274).
- IOV models: `[derived]` columns, `[output]` individual parameters, and the
  TAD column in `sdtab` now use each observation's **occasion** kappa instead of
  silently treating every kappa as zero. Post-fit diagnostic columns that depend
  on a κ-varying parameter (e.g. `CL`, `V`, `KA`) were wrong for IOV subjects;
  the fitted estimates, OFV, and IPRED/IWRES were unaffected (#238).
- The `sdtab` TAD column now shifts each dose by **its own** absorption lag —
  evaluated with that dose's occasion kappa and that dose's covariate snapshot —
  rather than applying the observation's lag to every dose. This changes TAD only
  when the absorption lag varies across doses, i.e. when it carries IOV (kappa) or
  depends on a time-varying covariate, *and* dosing spans the differing values
  (e.g. BID across two occasions); models with a constant lag are unaffected
  (follow-up to #238).
- FOCE (non-interaction) omega standard errors now match NONMEM `$EST METHOD=1`
  `$COVARIANCE MATRIX=R` (to ~3–6% on warfarin, previously ~31% low). The
  covariance step had added the Ω prior (`η̂ᵀΩ⁻¹η̂ + log|Ω|`) on top of the
  Sheiner–Beal marginal, which already carries Ω through `R̃ = HΩHᵀ + R` —
  double-counting Ω and flattening the omega-block curvature. FOCE estimates were
  already correct; only the SEs were affected (#243).
- The covariance step now succeeds on models with a mixed block + diagonal Ω: the
  structural-zero cross-block off-diagonals (`free_mask == false`) are excluded
  from the parameter set like FIX parameters, so their flat Hessian diagonal no
  longer aborts the step. This affected both FOCE and FOCEI (#243).
- Covariance standard errors now match NONMEM `$COVARIANCE MATRIX=R` (within ~2%
  on warfarin). The covariance step reconverges the inner EBE loop at every
  finite-difference point — holding the EBEs fixed gave an indefinite Hessian
  that was clipped and inflated theta/sigma SEs 30–94× — and applies the correct
  factor of two for the `−2·logL` objective (every SE was previously `1/√2` too
  small) (#209, #196, #129).
- Covariance step: `fd_hessian_step` is now an *initial* step; ferx automatically
  halves it up to 8× if any diagonal FD stencil is non-finite (#223).
- IOV FOCEI marginal likelihood now matches NONMEM after the Almquist Laplace
  correction (#109, #203).
- SAEM no longer collapses a block Ω to a rank-1 (near-unit-correlation)
  solution (#191).
- Stacked `EVID=4` reset occasions are segmented onto a monotonic timeline
  (#195, #197).
- `sdtab` no longer emits stray ETA columns (regression from #185).
- `warfarin --simulate` works again, and the docs `verify-build` step is fixed
  (#199, #200).
- FREM with `log_additive` error model: covariate pseudo-observation predictions
  are no longer log-transformed. The FREM override (θ + η) now runs after the
  LTBS log-transform, producing raw covariate predictions as NONMEM does. Without
  this fix the OFV was inflated by ~10 orders of magnitude.
- FREM with IMPMAP/IMP: the IS posterior Hessian now applies the FREM R-diagonal
  override (EPSCOV² variance) for covariate pseudo-observations, matching the
  FOCEI and SAEM code paths.
- `frem_predictions` and `frem_sigma` fit options are now registered as framework
  keys, suppressing spurious "not used by method" warnings on non-FOCEI chains.
- FREM data generation: missing covariate values (default -99) are now excluded
  from mean/variance computation and their pseudo-observation rows are omitted,
  matching PsN/NONMEM behavior.
- FREM data generation: records within each subject are now sorted by (time,
  event priority) to prevent backwards-in-time sequences that NONMEM rejects.

### Performance
- The inner EBE optimizer now selects between dense BFGS and L-BFGS by the inner
  problem dimension: dense BFGS (full inverse-Hessian, Newton-fast and cheap at
  low dimension) for the usual `n_eta ≲ 8` PK case, and L-BFGS (two-loop
  recursion, `O(m·n)` per step) once the inner dimension is large enough that the
  dense `O(n²)` update dominates — high-dimensional IOV (`n_eta + K·n_kappa`).
  Converges to the same EBEs (estimates and OFV unchanged); the crossover keeps
  small problems on the faster dense solver while making large random-effect
  inner problems scale (benchmarked: L-BFGS ~2× faster at dim 64, ~17× at 256)
  (#367).
- The covariance step is now built as a single parallel work-list over the
  finite-difference points (subjects iterated serially within each point) instead
  of firing a per-subject parallel reduction at every perturbed point. This removes
  the fork/join overhead of up to `4·n_free` rayon barriers in series — the
  bottleneck was scheduling, not core utilisation — making the covariance step
  ~9–11× faster across error models and structures, with bit-identical results.
  Both stencils are flattened: the non-IOV analytic-gradient difference and the
  IOV `OFV`-second-difference (the latter has `~2·n_free²` points, so it benefits
  even more) (#256).
- The covariance Hessian is built from a central difference of the analytical
  population gradient — reusing H-matrix columns for mu-referenced parameters
  instead of finite-differencing predictions — making the covariance step ~9×
  faster than scalar finite differencing on warfarin, scaling with the number of
  free parameters (#209, #196).
- Autodiff inner gradients now flow through `EVID=3/4` resets and lag time,
  removing a large finite-difference fallback slowdown (#198).

## [0.1.5] - 2026-06-01

Released before this changelog was started. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.5)
and `git log v0.1.0..v0.1.5` for details.

## [0.1.0] - 2026-05-29

Initial tagged release. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0).

[Unreleased]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0
