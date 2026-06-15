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
- `impmap_mceta` fit option: multi-start MAP for IMPMAP (NONMEM `MCETA` equivalent),
  improving IS efficiency in high-dimensional models (e.g. FREM with ≥5 ETAs).
- Analytical Jacobian for FREM pseudo-observations: covariate rows in the FD
  Jacobian are overwritten with exact ∂Y/∂η values (0 or 1), eliminating noise
  that corrupted the IS proposal in high-dimensional FREM models.
- `iscale_min` / `iscale_max` fit options: adaptive IS proposal scaling (NONMEM
  `ISCALE_MIN`/`ISCALE_MAX` equivalent). Per-subject pilot search over log-spaced
  scale factors selects the proposal width that maximises ESS. Defaults: 0.1–10.0.
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
  robust to model mis-specification). Supported for FOCEI/IOV fits (#223).
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

### Changed
- IMP (importance sampling) now jointly samples (η, κ) for IOV models,
  integrating over inter-occasion variability so the reported `−2 log L` is
  directly comparable to FOCE/FOCEI and NONMEM `METHOD=IMP`. Previously κ was
  held fixed at its EBE mode, giving a partial marginal; `kappa_treatment` in
  the fit YAML is now `marginalized` rather than `fixed_at_mode` (#186).

### Fixed
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
