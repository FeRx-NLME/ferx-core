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

### Added
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
  `SimulateOptions { seed, propensity_match }`. When `propensity_match` is set,
  each replicate's drawn etas are reassigned to subjects by optimal Mahalanobis
  matching (under the model Ω) against the subjects' fitted (posthoc) etas, so a
  subject's observed dosing/sampling design is paired with a similar drawn eta.
  This corrects VPC bias from treatment adaptation in real-world data (longer
  intervals for high-clearance patients, etc.). Operates on observed data;
  returns the usual simulation rows for the caller to build the VPC (#288).
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

### Performance
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
