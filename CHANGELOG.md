# Changelog

All notable changes to **ferx-core** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(see the [Releases](https://ferx-nlme.github.io/ferx-core/development/sdlc.html#releases)
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

### Changed
- **`block_sigma` now estimates its off-diagonal correlation (#847).** A plain
  `block_sigma (...) = [...]` is NONMEM `$SIGMA BLOCK(n)`: its diagonal SDs *and* its
  off-diagonal correlation are estimated. Previously the correlation was always frozen at the
  declared value, so a model with a `$SIGMA BLOCK` counterpart in NONMEM optimized a different
  objective — on the fluconazole RadboudUMC model NONMEM moved the residual correlation to
  ~0.93 while ferx held the ~0.2 init. Append `FIX` to hold the whole block — SDs and
  correlation alike, matching `$SIGMA BLOCK(n) FIX`. Note that the old behaviour (SDs
  estimated, correlation frozen) is no longer expressible: NONMEM has no such form either, and
  it was the mismatch this issue is about. The shipped
  `examples/correlated_residual_combined.ferx` already uses `FIX` and is unaffected. The
  correlation is optimized as its Fisher-z transform `atanh(rho)`, so it stays strictly inside
  `(-1, 1)` and the residual covariance can never go singular from the correlation alone, and
  it rides the exact analytic FOCE/FOCEI outer gradient rather than falling back to finite
  differences. Estimators that do not estimate it (SAEM, IMP/IMPMAP, Bayes, AGQ, VI) continue
  to hold it at the declaration. Because a method chain starts each stage from the previous
  stage's estimates, a chain that runs `foce`/`focei` before a stage that cannot read the
  estimated correlation is now rejected with `E_BLOCK_SIGMA_CHAIN_UNSUPPORTED` rather than
  silently scoring the declared value — `[saem, focei]` is fine, `[focei, imp]` needs `FIX`.

### Performance
- **A joint PK-TTE model with a linear PK block now takes the exact steady-state solve
  (#1210).** The hazard accumulator's one-cycle map is the identity, so it made `I - M`
  singular and the exact `(I - M)^-1 b` fixed point (#914) declined for *every* joint model,
  whatever its PK block looked like — the equilibration ran the full 50-cycle pulse train, and
  could emit a spurious `Steady-state (SS=1) equilibration ... did not converge` warning for a
  PK block that had settled long before, since the accumulator never stops growing. The
  accumulator rows are now projected out of that solve, so a linear PK block gets its handful
  of one-cycle integrations back.

### Added
- **Fitted `block_sigma` correlations are reported with their fixedness and standard error
  (#847).** `FitResult` gains `residual_correlation_fixed` and `se_residual_correlations` (the
  SE on the natural `rho` scale, by the delta method on the packed Fisher-z coordinate), the
  fit YAML's `block_sigma:` section gains `correlation_se` and now reports
  `correlation_fixed` from the model rather than always `true`, and `.fitrx` bundles
  round-trip all three. A bundle written before this change loads with every correlation
  marked fixed, which is what it meant at the time.

### Fixed
- **`predict_survival` returned `NaN` for a time grid with no point past the first event
  (#1218).** Asking for the curve at `[0.0]` alone — or any grid whose largest time does not
  pass the subject's first dose or observation — returned `NaN` for `cum_hazard` and `hazard`
  with no warning, while the same `t = 0` on a longer grid was fine. It now returns the state
  at that instant, post-dose, exactly what the longer grid reads. The same one-break timeline
  reached two other readers of the dense state: `[derived]` output columns on the event-driven
  path (time-varying covariates, resets, or a model-time read) were `NaN` for a subject whose
  only observation coincides with its dose, and a joint PK-TTE subject outside the shared
  single-solve (same routing) whose only event or censor sits on its first dose scored the
  `1e20` sentinel instead of its finite likelihood. Both fixed by the same change.
- **ODE solver settings passed to `fit()` now reach the solver (#1212).** `ode_reltol`,
  `ode_abstol`, `ode_max_steps`, `ode_method`, `ode_stiff_abort_after` and `ode_auto_switch`
  were stamped onto the compiled model at parse time and read from there by every integration
  path, so the same keys set on a `FitOptions` handed to `fit()` were silently ignored: the
  fit ran at the model file's (or the default) accuracy, on the default stepper, and reported
  success. Tightening a tolerance to test an integration-noise hypothesis returned a
  bit-identical objective across six orders of magnitude, and a caller selecting
  `rosenbrock23` / `rodas4` / `rodas5p` for a stiff system stayed on the explicit stepper with
  nothing to say so. A fit now carries the caller's ODE settings to the integrator for the
  duration of that fit. Precedence is per key and one-directional: a key the caller moved off
  its default wins, a key left at its default yields to the model file, so passing a
  hand-built `FitOptions` cannot loosen a model that pinned `ode_reltol = 1e-10`. The same
  now applies to the standalone `run_covariance()`, `run_sir()` and `run_sir_core()` entry
  points, which previously ignored a caller's ODE settings — so a covariance step run beside
  a tight fit no longer differences a coarser surface than the estimates came from. The
  model-file route, `predict()` and `simulate()` are unchanged; note that the override lasts
  one call, so `predict()` after a tight `fit()` on the same model still uses the model
  file's accuracy unless you `sync_ode_solver_opts` an owned model. Concurrent fits are
  isolated from each other: a call's settings travel on the thread that made it and on a
  thread pool keyed to those settings, so a fit that asked for nothing keeps the model
  file's accuracy even while another fit runs at `1e-10` beside it (and vice versa) — which
  matters for `ferx-tools`' parallel replicate fits and for any caller sharing the fit pool.
- **An `SS=1` dose no longer carries the steady-state run-in into a joint PK-TTE model's
  cumulative hazard (#1210).** The appended `d/dt(__chz_<cmt>)` accumulator was cycled through
  the equilibration along with the PK compartments, but it is a pure integrator with no steady
  state — so `H(0)` came back holding the run-in's own hazard (`H0 x 50 cycles x II`, e.g.
  `12.0` for a constant `H0 = 0.02` at `II = 12`) and every survival quantity downstream of it
  was displaced by that amount. `S(t) = exp(-H(t))` made this fatal rather than cosmetic: on a
  drug-driven hazard `S(0)` underflowed to `0` and each subject's objective was inflated by
  ~2500, and a simulated `SS=1` subject drew its event at `t = 0` in every draw. `SS=1` now
  equilibrates the PK compartments only and the accumulator keeps its value at the dose record
  — `0` for a subject's first dose, and, for a **later** `SS=1` dose, the hazard accrued so far
  (an SS dose re-loads the compartments; it is not a reset — `EVID=3`/`4` still are). A lagged
  SS dose no longer banks its phase advance either, which also removes a non-monotone `H`. A
  hazard reading `TAD`/`TAFD` under `SS=1` returned `NaN` for the whole subject and now works.
- **An estimated `block_sigma` correlation is bounded at `|rho| <= 0.995` (#847).** Merely
  keeping rho inside `(-1, 1)` is not enough: a paired residual block's determinant carries a
  factor `1 - rho^2`, so a rho of 0.9999 leaves `R` numerically singular and the likelihood
  will chase `log|R| -> -inf`. On the 12-observation `examples/correlated_residual_combined`
  fixture an unbounded rho ran to -0.99993 and reported convergence at a degenerate optimum. A
  rho sitting on this rail is now a legible diagnostic: the two endpoints are carrying the same
  noise.
- **A zero `block_sigma` off-diagonal is now estimated rather than dropped (#847).**
  `block_sigma (A, B) = [0.04, 0.0, 1.0]` — the natural translation of `$SIGMA BLOCK(2)` with a
  zero covariance init — previously built no correlation at all, so under the new
  estimate-by-default semantics it would have silently fitted a diagonal residual with no
  coordinate to move. A `FIX`ed zero is still dropped: a fixed zero correlation is the same
  object as no correlation.
- **`method = laplace` / `n_agq > 1` with a free `block_sigma` now uses the reconverged-FD
  outer gradient (#847).** AGQ's analytic score assembles theta / omega / sigma / omega_iov and
  never writes the rho slot, while its objective does depend on rho — so the analytic gradient
  is declined for a free correlation rather than handing the optimizer a hard zero there. A
  `FIX`ed block keeps the analytic score. The chain guard was widened to match: `laplace` can
  move rho, so `[laplace, imp]` is rejected exactly like `[focei, imp]`.
- **The `block_sigma` correlation coordinate is no longer magnitude-scaled below 1 (#847).**
  The outer optimizer's `abs` preconditioner divides each packed coordinate by its own
  `|value|`, which is right for a log-space coordinate but meaningless for the Fisher-z
  `atanh(rho)` — that is a *position* in a bounded range which passes through zero. At the
  common `rho = 0.2` init it handed the optimizer a coordinate with roughly thirty times the
  scaled room of every other one, and NLopt L-BFGS failed on its first step. The scale is now
  `max(|atanh rho|, 1)`. On the fluconazole RadboudUMC model a cold-start FOCEI fit goes from
  failing at OFV 1111.12 to converging at **736.89 with rho = 0.9319**, against NONMEM's
  734.644 / 0.9312. Models without a `block_sigma` are unaffected by construction.
- **`simulate()` drew correlated residuals at the declared `block_sigma` correlation (#847).**
  The dense residual `R` it samples from used the live sigmas but the model's declared
  correlation, so a VPC or posterior-predictive check of a fit with an estimated off-diagonal
  would not reproduce the correlation the fit reported. It now uses the parameter vector's.
- **`block_sigma` residual derivatives were built at the declared correlation, not the live
  one (#847).** The outer-gradient assembly (`corr_residual_diag` /
  `corr_residual_rd_at_sigma`) and the inner eta-gradient read the correlations off the frozen
  `CompiledModel`, so once the off-diagonal became estimable the whole `(R, dR/df, d2R/df2)`
  chain would have been evaluated at the initial value while the objective moved. Both now
  take the live parameter vector's correlations.
- **`ode_method = auto` no longer keeps a stiff solve whose analytic derivatives have
  overflowed (#1204).** The escalation guard checked that every saved state was finite, but
  read *values* only. A dual number's derivative jets carry higher powers of what its value
  carries linearly — integrating `u' = p·u` gives `∂u/∂p = t·u` and `∂²u/∂p² = t²·u` — so a
  trajectory near the top of double precision overflows its Hessian first, its gradient next,
  and its predicted value not at all. Such a segment reported success, clamped nothing,
  finished cleanly, returned finite predictions, and handed FOCE/FOCEI a `NaN` gradient with
  every counter reading zero. The guard now also rejects on non-finite jets and re-solves the
  segment explicitly, and the same check scores the explicit fallback, so a fallback that did
  not repair the gradient is reported instead of being presented as a successful retry. The
  new `auto_stiff_rejected_jets` counter is reported in the `ode_solver` warning with its own
  advice — the stiff method worked and the sensitivities did not, so naming another method
  will not help; check the model's units and scaling. A **named** `ode_method` stays
  unguarded, as before. To make the counter observable at all, the post-fit diagnostic sweep
  now also runs one analytic sensitivity solve per subject for models on the analytic ODE
  sensitivity path — the second-order provider where the model has an analytic outer gradient,
  since the Hessian is what overflows first, and nothing at all for an FD fit. The `f64`
  prediction pass carries no derivatives and could never see this decision, so that sweep is
  collected separately and only the new counter is reported from it.
### Changed
- **VI early stopping is now judged with robust statistics.** The settling test compared
  the *mean* of the last window of the objective trace against the mean of the one before
  it, and sized its tolerance from the trace's own sample variance. Both estimators have a
  breakdown point of zero, so on a heavy-tailed trace — what an unhealthy VI run emits — a
  handful of outliers inflated the spread until the tolerance swallowed the drift that was
  still there, and the run reported `converged` at the earliest iteration arithmetically
  allowed. The criterion now uses the median and the MAD (scaled by 1.4826) in the same
  `SETTLE_Z · spread + rel_tol · (1 + |location|)` form, so a tail can no longer buy a
  premature stop. Estimates on healthy fits are unchanged: `propofol_schnider` and
  `vancomycin_uvm`, which stop on the parameter-stability criterion, are identical down to
  the last reported digit, and `warfarin`, `two_cpt_oral_cov` and `warfarin_iov`, which
  stop on the trace, agree to four or five significant figures. The trace criterion is
  slightly more conservative, so those three run longer for the same answer (5125 → 8375,
  6625 → 7250 and 3500 → 3625 iterations). A noiseless trace still settles on the relative
  floor alone (#1119).
### Added

- **`ferx_core::edit` — a typed model-transformation API, so a program can now *write* a
  `.ferx` model as well as read one (#1176).** `ModelText::parse` / `render` round-trips a
  model file byte for byte (comments, alignment, blank lines and line endings included), and
  `ModelText::apply` applies one typed `ModelEdit`: swap the structural model, add or drop a
  `[covariate_model]` relation, add or drop an η, block two ηs together, change the residual
  error model, carry a parent fit's estimates into the child (`SeedInits`, keyed on parameter
  *names*, not positions), or set a `[fit_options]` key. A structural swap performs the coupled
  θ/η/expression edits across all three blocks — `two_cpt_oral` → `one_cpt_oral` drops `Q`,
  `V2`, `TVQ`, `TVV2`, `ETA_Q` and `ETA_V2` in one call. η surgery requires the canonical
  `P = TVP * exp(ETA_P)` form and is a hard error naming the parameter on anything else, never
  a silent wrong edit. `ModelText::canonical_hash` gives a candidate a stable identity — equal
  across comment and whitespace changes, different for every semantic one — for use as a fit
  cache key. This is the prerequisite for the model-search tooling in #1175.
- **A cancellable bootstrap (#1161).** `BootstrapOptions::cancel` takes a `CancelFlag`; setting
  it from another thread stops a long `ferx_tools::bootstrap` run at the next replicate boundary
  and returns the new `BootstrapError::Cancelled`, so a caller reports an abort as an abort
  rather than as "every remaining replicate failed". Replicates the cancel unwound are dropped
  rather than journaled as failures, so `--resume` refits them; everything already finished stays
  on disk and resumes into exactly the run that was cancelled.
### Fixed
- **A fit whose estimate ran to an internal safety rail no longer reports `converged: true`
  (#1118).** ferx caps a few packed coordinates internally (an implicit THETA cap, the OMEGA /
  SIGMA runaway rails); unlike a THETA bound you declared, one of those cannot be a valid
  constrained optimum. When a free estimate ends pinned to a rail the fit is now reported as not
  converged and the `parameter_at_runaway_guard` warning is raised to `Critical`, so a script or
  agent keying off the boolean stops accepting a point that is by construction not an interior
  optimum. A *collapse* hit — a variance falling to its floor at zero — is unchanged: it stays a
  `Warning` and leaves `converged` alone, because that is usually an unsupported component to
  remove rather than a numerical runaway. Each listed hit now says which of the two it is
  (`verdict: runaway` / `collapse` in the warning's `details`), because the side does not decide
  it: an OMEGA off-diagonal is bounded symmetrically at ±10, so a correlation driven to the
  *lower* rail is a runaway too. A SIGMA at its ceiling additionally suggests rescaling DV or
  using a proportional / log-transformed error model, since that rail is the one an otherwise
  sound model can reach on unscaled data.
- **`ferx bootstrap` reflects the same rule.** A replicate that ends at a runaway rail is now a
  non-converged replicate, so with the default `skip_minimization_terminated` it is excluded
  from the confidence intervals and counts against the reported
  `minimization_successful` fraction. Re-running a bootstrap of an unchanged model can therefore
  report slightly different CIs than before; `--summarize` over a stored `raw_results.csv`
  re-applies the criteria without refitting.

## [0.3.1] - 2026-09-02

### Added
- **A progress bar for `ferx bootstrap`.** A 200-sample bootstrap is minutes to hours of
  fitting behind one command, with nothing on the terminal to say whether it was working
  or wedged. The run now reports each fit as it completes: the base model gets a spinner,
  the replicates a bar with an ETA, `--dofv` its own bar for its second pass, and a
  `--resume` run counts only what is left to fit. The bar is drawn only when stderr is a
  terminal; `--no-progress` suppresses it. `ferx_tools::bootstrap::run_bootstrap_with_progress`
  is the same run with a `BootstrapEvent` sink, which is what ferx-r renders onto a `cli`
  bar; `run_bootstrap` is unchanged.
- **`ferx gam` — GAM covariate pre-screening from the command line (#1114).** Screens every
  declared covariate against every ETA with independent GAM regressions and prints a table
  ranked by delta-AIC (`AIC_null − AIC_best`), the Rust equivalent of Xpose4's `xpose.gam()`.
  Run it on a model and dataset, on an existing fit with `--from-fit <run.fitrx>`, or without
  estimating at all with `--no-fit` (the NONMEM `MAXEVAL=0` equivalent). Candidate forms are
  tuned with `--spline-df N` (repeatable) and `--no-linear`, the shrinkage warning threshold
  with `--shrink FRAC`. Results are written to `{model}-gam.csv` by default; `--csv PATH`
  redirects the file and `--no-csv` suppresses it. The same screen runs as part of an ordinary
  fit with the `--gam` flag. See [GAM covariate screening](https://ferx-nlme.github.io/ferx-core/tools/gam-screening.html).

### Changed
- **The published tarball no longer ships repo infrastructure (#1170).** `tests/`,
  `nonmem_anchor/`, `docs/`, `plans/`, `tools/`, the workflow and hook directories and the
  repo-process markdown are excluded from the crate: 1289 files / 5.47 MB compressed becomes
  366 files / 3.22 MB. None of it is needed to build or document the crate. Running the test
  suite needs a git checkout, as it needed the NONMEM anchors anyway. No source or API change.
- **The three published crates now carry `keywords` and `categories` (#1170).** Registry metadata
  only -- no code change. It is what covers discoverability on crates.io, which is why claiming
  the `ferx-nlme` name was judged unnecessary in #1170.
- **`FitOptions::threads` now also caps a multi-start fit (#1115).** A pinned positive `threads`
  was honored only when `n_starts <= 1`; with several starts the fan-out ran on the full-width
  shared pool regardless, so a tool that pinned one thread per fit from a `PoolPlan` and asked
  for multiple starts got its replicate-level pool *and* a full-width pool underneath every
  replicate. The pin is now an upper bound on the whole `fit()` call. A multi-start fit that
  pins `threads` below the core count is correspondingly slower than before, and unpinned
  multi-start fits are unchanged.
- **The four documentation sections the callout blind spot had been hiding are split into
  addressable subsections (#1190).** ODE models' *Which regime am I in?* gains **Where the step
  counts show up**, **Bounding what a stalled segment costs**, **Measured: an accuracy-limited
  fit** and **Every feature works with every method**; the analytic transit closed form gains
  **Scope (first version)** and **The flip-flop regime**; VI's *How `σ` is updated* gains **Why
  the default is 32 draws** and **If a VI fit lands short**; adaptive dosing's *Keys* gains
  **Writing an `observe` expression**. Nothing was deleted and every existing anchor still
  resolves — the R1 baseline is one entry shorter than before this PR, not longer.
- **`tools/render-docs.sh` renders the docs site; prefer it to a bare `quarto render docs`
  (#1190).** Quarto's project input discovery skips any path containing a hidden (dot-prefixed)
  directory component, so from a worktree under `.claude/worktrees/<name>/` a plain
  `quarto render docs` discovers zero inputs: it writes `robots.txt` and `sitemap.xml`, renders no
  page, warns about nothing and exits 0. The script stages `docs/` outside the dot directory when
  it has to, and fails loudly on a zero-page render.
- **The repo is now a cargo workspace, and the `ferx` binary moved into a `ferx-cli` package
  (#1114).** The installed binary is unchanged — still `ferx`, same arguments, same output — but
  building it from a source checkout now needs the workspace or the package named:
  `cargo build --release --workspace` (or `-p ferx-cli`), and `cargo run --release -p ferx-cli --
  model.ferx --data data.csv`. A bare `cargo build --release` at the root now builds the
  `ferx-core` *library* only. Feature flags belong to `ferx-core`, so workspace-wide commands
  write them package-qualified (`--features ferx-core/ci`). Consumers of the `ferx-core` crate —
  including the ferx-r wrapper, which patches the repo root — are unaffected: the root package is
  still `ferx-core`.
- **A covariate column with no value for a subject now reads as `NaN`, not as a dropped key
  (#1111).** Previously the key was simply absent from that subject's covariate map, and every
  evaluation site resolves an absent covariate to `0.0` — so a subject whose `WT` column was `.`
  on every row was fitted at `WT = 0`, indistinguishable from a genuine zero, and `present(WT)`
  reported it as present. The reader now stores `NaN` and warns, naming the covariate and the
  affected subjects, so the gap is either guarded with `present(...)`, imputed, or fails loudly.
- **The five largest documentation sections are split into addressable subsections (#1162).**
  `[fit_options]`'s 36 shared keys are grouped into seven tables (run control, outer optimizer,
  inner loop, covariance, ODE tolerances, ODE stepper, data handling); FOCE's gradient-route
  section leads with the decision and moves the scope history into named sections plus an FD
  fallback table; SAEM's theta/sigma M-step, `ode_method = auto`, and the check-report code table
  are likewise subdivided. The three heaviest `[fit_options]` key descriptions are cut to what
  choosing the key needs, with the detail deep-linked into the pages that already carry it —
  `ODE stepper selection` 909 → 367 words, `Inner loop (EBE)` 769 → 365, `VI-Specific Options`
  747 → 378. Every previous anchor still resolves, eight already-broken in-page links are
  repaired, and the site's `toc-depth` is raised to 4 so the new subsections appear in the
  on-page table of contents.
- **The docs content column is wider, and wide tables no longer run under the "On this page"
  TOC (#1162).** `grid.body-width` goes from Quarto's 800px default to 1000px (a 749px → 895px
  content column at 1512px wide), and the desktop table rule now actually contains an oversized
  table: it set `display: table` with `overflow-x: auto`, which is inert, so a table wider than
  the column drew over the margin TOC instead of scrolling inside itself.

### Added
- **`ferx_tools::gam::gam_screen()`: GAM-based covariate pre-screening (#1114).** For each
  ETA × covariate pair, fits `η_i ~ f(cov_i)` (linear, natural cubic spline, or one-hot
  categorical) and ranks covariates by AIC improvement over the null model. High-ΔAIC
  covariates are then prioritised in an SCM. This is the Rust equivalent of Xpose4's
  `xpose.gam()`. Requires `ferx-tools`. See
  [GAM covariate screening](https://ferx-nlme.github.io/ferx-core/tools/gam-screening.html).

  A covariate that cannot be screened is now reported in `GamResult::warnings` rather than
  dropped in silence — being skipped and being screened-but-unimportant are indistinguishable
  in a ranking. Skips cover: a constant covariate, a single-level categorical, a categorical
  or spline form spending more than half the subjects as parameters, a singular design, a
  column whose length does not match the subject count, and fewer than three usable subjects.
  An ETA whose EBEs are constant, or not all finite, is refused outright. Shrinkage above
  30% warns, and so does an ETA whose shrinkage the fit did not report at all.
  `gam_screen_raw()` panics on a length mismatch instead of truncating to the shortest input.
- **`PoolPlan` and `FitOptions::quiet()`: the two knobs a tool needs to run many fits (#1115).**
  `PoolPlan::from_budget(total_threads, n_units)` splits a thread budget between the replicate
  level and `fit()`'s own per-subject level (outer level first, so a 200-replicate bootstrap on
  8 threads is 8 single-threaded fits at a time), `apply_to()` pins the inner fit, and
  `install()` builds the outer Rayon pool with ferx's 32 MiB worker stack
  (`FIT_RAYON_STACK_SIZE`) — a pool built by hand inherits the platform-default 2 MiB stack and
  overflows on wide ODE+IOV analytic-gradient models. `FitOptions::default().quiet()` turns off
  the per-iteration console output; every non-fatal message stays reachable on
  `FitResult::warnings`, the optimizer-trace filename now included (it embeds a pid and a
  timestamp, so a quiet caller could not otherwise learn where the trace went).
- **A `present(COV)` condition for covariates that may be missing (#1111).** A missing covariate
  value is `NaN`, and division by it underflows to `0.0` here rather than erroring, so an
  unguarded `(CRCL/100)^THETA` silently zeroes the parameter on a row with no `CRCL`.
  `if (present(CRCL)) ... else 1.0` says the guard plainly; the equivalent `CRCL == CRCL` idiom
  still works. Usable anywhere a condition is (`!present(X)`, `present(X) && X > 100`), and it is
  what `[covariate_model]` wraps every generated factor in.
- **A `[covariate_model]` block: covariate–parameter relationships declared as data (#1111).**
  `CL ~ WT power(center = median)` desugars into exactly the classical
  `[individual_parameters]` expression — with the theta declaration, the centering constant and a
  missing-value guard filled in — so the OFV and estimates are identical to writing it by hand.
  All five PsN `scm` states are expressible (`none`, `linear`, `hockey`, `exponential`, `power`,
  plus `categorical` and an `expr("…")` escape hatch) with PsN's default inits and bounds;
  `center`/`breakpoint`/`ref` accept a literal or a data-derived statistic (`median`, `mean`,
  `min`, `max`, `mode`), and `[covariates]` gains `categorical(levels = [0, 1])` /
  `levels = auto`. The relations are echoed on `FitResult.covariate_relations` and in the fit
  YAML — each θ naming the categorical `level` it contrasts, each `expr(…)` relation carrying
  the expression that ran, and a relation with no θ emitting `thetas: []` rather than a bare key.
  Statistics summarise every event record (observations, doses, `EVID=2` markers, `EVID=3/4`
  resets), so the default bounds cover the range the fit evaluates over. Centres that would break
  their own form are refused up front: outside the observed range for the linear family (whose
  default bounds would come back reversed), non-positive for `power`, zero for `linear_relative`;
  so is a categorical value the block never declared, which would otherwise be modelled silently
  as the reference level (`E_COV_LEVEL_UNKNOWN`). `ferx check` prints the desugared block. Each
  relation owns its own θ (as in `scm`);
  a θ shared across relations is still written classically. Relations are line-oriented and
  independent,
  so a covariate search rewrites this one block instead of doing surgery on an expression. See
  `docs/model-file/covariate-model.qmd`.
- **A CI-enforced public-API baseline for `ferx-core` (#1114).** `api/ferx-core-public-api.txt` is
  a committed snapshot of the crate's public surface; a CI job regenerates and diffs it, so any
  widening of the API fails until the baseline is updated in the same PR. Regenerate with
  `tools/update-public-api.sh`. This doubles as semver protection for the crates.io release and
  for ferx-r. `#[doc(hidden)] pub` is banned, because `cargo public-api` omits such items and the
  attribute would otherwise be a silent bypass of the gate.
- **A `ferx-tools` crate (#1114)** — the home of multi-fit tooling (bootstrap, stepwise
  covariate modelling, model search, cross-validation). It can only reach `ferx-core` through the
  same public API the R wrapper uses.
- **`ferx bootstrap` — the non-parametric case bootstrap (#1140).** Resamples subjects with
  replacement, refits the model to each replicate, and reports bias, standard errors and
  confidence intervals from the spread of the estimates, at `PsN::bootstrap` feature parity:
  `--samples`, `--seed`, `--threads`, `--sample-size` (including PsN's per-stratum
  `"1001=>12,1002=>24"` form), `--stratify-on`, `--update-inits`, `--run-base-model`,
  `--keep-covariance`, the four `--skip-*` exclusion filters, `--dofv`, and `--summarize` to
  recompute results under different filters without refitting. Writes PsN-named CSV artefacts
  (`raw_results.csv`, `bootstrap_results.csv`, `included_individuals1.csv`, `sample_keys1.csv`, …).
  Unlike PsN, the drawn datasets depend only on the seed and the design — never on the thread
  count, the completion order, or whether the base model was fitted first. Bootstrapped parameters
  are every estimated theta, Omega (free lower triangle), sigma, and IOV Omega, so a `kappa` model
  gets bootstrap SEs and CIs for its inter-occasion variance and `--update-inits` / `--dofv` carry
  it too. A model that reads `ID` as a covariate is refused rather than silently mis-fitted, and so
  is a `[mixture]` model — its classes are identified only up to relabelling, so averaging across
  replicates would mix them; use SIR for uncertainty on a mixture model instead (#1145).
  See `docs/tools/bootstrap.qmd`.
- **`ferx bootstrap --resume` — pick an interrupted run back up (#1143).** The per-replicate files
  (`raw_results.csv`, `included_individuals1.csv`, `included_keys1.csv`, `sample_keys1.csv`) are now
  written as each replicate finishes rather than once at the end, so a run killed part-way leaves
  everything it had already completed. `--resume` refits only the samples the directory does not
  hold, and reuses the base fit rather than repeating it. Because each draw comes from the seed and
  the replicate's own index, a resumed run reproduces the uninterrupted one exactly — not merely
  statistically, which is what PsN's shared, order-dependent RNG can offer. A replicate whose fit
  *errored* is carried forward by default, as in PsN; `--retry-failed` refits those instead, for a
  failure that was a transient resource problem. Every run now writes `bootstrap_run.json`
  recording its seed, options — including what the replicates were started from, so a resume
  cannot mix replicates begun at the base fit with replicates begun at the model file's
  estimates — and the hashes of the model and data files; a `--resume` whose inputs or options
  disagree is refused, naming the field. A trailing row cut off by a hard kill is
  dropped and that sample refitted. `raw_results.csv` now carries full round-trip precision rather
  than ten fixed decimals, because a resumed run's replicates start from the base-fit estimates read
  back out of it; the statistics files are unchanged. See `docs/tools/bootstrap.qmd`.
- **`prepare_run` / `prepare_run_with_inits` (#1140)** — public "load a model and its dataset, but
  do not fit" entry points returning a `PreparedRun`. `run_model_with_data` is now implemented on
  top of them, so a tool and the CLI cannot diverge in how a model is loaded.

### Fixed
- **A joint PK-TTE model with a time-dependent hazard no longer integrates twice per subject
  (#1166).** The `[event_model] hazard = …` expression is appended to the ODE system as
  `d/dt(__chz_<cmt>)`, and a Weibull or Gompertz baseline hazard reads `TIME` by definition —
  which made ferx treat the whole model as having non-autonomous PK dynamics, declining the
  single shared solve (#570) and leaving the dense driver, for a property the PK block does
  not have. The "does this model read model time?" question is now asked of the PK equations
  alone. Measured on a 40-subject fixture, one variable: **2.1×** faster on one objective
  evaluation and **2.6×** over eight outer iterations. Results are unaffected beyond solver
  tolerance — the same fixture's objective moves 4.4e-3 at ferx's default tolerances and
  7.7e-7 at `ode_reltol = 1e-9`, i.e. the difference decays with the integrator rather than
  being a fixed offset — so earlier fits do not need re-running. Steady-state gates deliberately keep the wider question, since SS
  equilibration integrates the augmented system including the hazard state. Validated against
  a new NONMEM anchor (`nonmem_anchor/pktte_tdep.ctl`): `H(t)` matches `A(3)` to 3.6e-9
  relative and the per-subject objective matches the `.phi` `OBJ` to 6.0e-8.
- **`__chz_*` is now rejected as a read (#1166).** The cumulative-hazard accumulator the
  parser appends for a joint PK-TTE hazard is write-only: referencing it from an `[odes]`
  equation, an `if` condition or the hazard expression, or seeding it with
  `init(__chz_<cmt>)`, is a parse error naming the state instead of silently producing a
  coupled system the shared solve was never designed for. A `[scaling]` readout may still
  read it.
- **The packaged crate no longer ships two files the CC BY-NC exclusion was meant to cover (#1170).**
  `exclude`'s `data/mbma_naproxen.*` glob required a literal `.` after the stem, so
  `data/mbma_naproxen_source.py` slipped through, and `examples/` was never listed, so
  `examples/mbma_naproxen.ferx` shipped as an example whose data is excluded and so cannot run.
  Neither file was itself CC BY-NC (both are repo-authored MIT, and the restricted csv was
  correctly excluded throughout), but the licence note in `data/mbma_naproxen.LICENSE` described
  a mechanism the globs did not implement. Both are now excluded.
- **An infusion into a built-in absorption compartment is no longer delivered twice on the
  dense/states engines (#1187).** With a `RATE>0` or `RATE=-2` dose into a `first_order()`,
  `transit()`, `igd()` or `weibull()` kernel (#719 gap 2), the mass was released *both* through
  the convolved `R_in_inf` and again as a plain constant rate injected straight into the
  compartment — bypassing absorption entirely. `predict()` was correct throughout; what was
  wrong is everything routed through the dense engines: sdtab `IPRED` and compartment states,
  `[derived]` grid integrals, `simulate()` event times, and — as a wrong *objective*, not just a
  wrong diagnostic — the joint PK-TTE cumulative hazard and the Markov/CTMM likelihood. Measured
  against NONMEM 7.6.0: the cumulative hazard at the event time read **31× high**
  (`3.796844` vs `1.2217e-1`), and a concentration readout up to **214×** during the infusion
  window, settling near 1.9× afterwards. Models without an infusion into an absorption
  compartment are unaffected. Models with one should be re-fitted if they carry a joint PK-TTE
  or a Markov/CTMM endpoint: those objectives were wrong, so the estimates they produced were
  too. A pure-PK fit keeps its estimates — its objective always ran through the guarded
  prediction path — but its sdtab `IPRED`, compartment states and `[derived]` columns were
  wrong and should be regenerated.
- **The docs linter no longer reads a Quarto callout title as a section boundary (#1190).**
  Pandoc lifts the heading that opens a `::: {.callout-*}` block into the callout header, so the
  rendered page gets neither a section nor an anchor from it — but `docs-lint` counted all 37 of
  them across 15 pages as real headings. That chopped each callout's body into chunks R1 never
  measured whole (three oversized sections were hidden, and one baselined section was reported at
  5,756 characters when it is 7,331 — all four are now split, see below), and registered ids R3
  and R4 treated as addressable. A
  heading *later* inside a callout is still a real section. Slug generation is fixed alongside it:
  inside a code span `<op>` is literal text, not an HTML tag, so
  `` ## Rules — `when signal <op> <value>` `` addresses as `#rules-when-signal-op-value` and no
  longer truncates at the first placeholder. Checked against a full `quarto render` of all 86
  pages, in both directions — every heading the parser computes is a section on the site, and
  every section on the site is one the parser computes.
- **A lagged `zero_order` absorption route is no longer dropped by the ODE engines that serve
  diagnostics, the joint PK-TTE hazard and `simulate()` (#1171).** `zero_order(dur=D, lag=L)`
  is delivered as a per-segment constant rate, admitted only when the segment sits fully inside
  the window `[t_dose + L, t_dose + L + D]`, so the integrator has to break its timeline at the
  window's onset as well as its end. The shared helper pushed only the *end*, leaving the start
  unbracketed on the two builders that did not separately bracket the route onset — the rate was
  then dropped for the whole window, and a model whose only input is a lagged `zero_order` read
  **exactly zero everywhere** on those paths. The helper now emits both edges, so the timeline
  and the containment test can no longer disagree in any builder. Affected:
  the sdtab `IPRED` and compartment-state columns (and the `IWRES`/`CWRES`/`[derived]` values
  computed from them), `[derived]` grid integrals, the **joint PK-TTE** cumulative hazard and
  its contribution to the objective — which collapsed to the drug-free baseline `H0·t`,
  removing the entire drug effect — the Markov/CTMM endpoint likelihood, the adaptive-dosing
  window AUC signal, and `simulate()` event times, where a subject who should have had an event
  was administratively censored. A **pure-Gaussian** fit's OFV and `predict()` are unaffected:
  their predictor already carried the onset break, which is why no existing test caught this.
  Anchored against a new NONMEM `ADVAN13` run (`nonmem_anchor/lagged_zo.ctl`, `D1`+`ALAG1` with
  `RATE=-2`), which the fixed paths now match to the `$TABLE` print precision. The onset break is
  also restored for a lagged **`first_order`**/`transit`/`weibull` route on those engines, whose
  kernels step at the onset rather than losing the window; those move within solver tolerance
  rather than from zero, so no fit result changes materially.
- **An `[odes]` right-hand side that is nonlinear in the compartment amounts is no longer
  served by the closed-form fast path (#1149).** Linearity was established by evaluating the
  right-hand side one compartment at a time with every other amount held at zero, so a term
  that only switches on away from those points was invisible to it: a product of two amounts
  (`- KON*central*periph`, the target-binding/TMDD shape) is exactly zero whenever either is
  zero, and a branch testing an amount (`if (central > 10)`) was never entered, because the
  largest amount probed was 2. Such models were admitted to the closed form, which does not
  evaluate the right-hand side at all, so the term was **dropped from the predictions** —
  the same silent failure as #1124, with no time variable involved. Identification now also
  checks the right-hand side against the recovered linear system away from the axes, across
  six decades of amount. Models that are linear in the amounts are unaffected and keep the
  fast path, including those whose coefficients are arbitrary functions of `θ`, `η`, and
  covariates.
- **An `EVID=2` record no longer aborts a fit on a parameter-static subject (#1124 review).**
  Any subject routed to the event-driven walker while its PK parameters are constant — an
  `EVID=3/4` reset, or (since the change above) an `[odes]` right-hand side that reads model
  time — had its per-event parameter snapshots built with the `EVID=2` rows omitted, and the
  walker asserts one snapshot per record. The run failed on an assertion rather than
  returning a diagnostic, in release builds as well as debug (the assertion is
  `assert_eq!`, not `debug_assert_eq!`; it unwinds, so from R it surfaces as an opaque
  error rather than a message naming the subject). Reaching it needed only a dataset
  carrying `EVID=2` rows alongside a varying column the model does not reference, because the
  irrelevant-covariate pruning clears those rows' covariate snapshots while keeping their
  times. Both the ODE and the analytical event-driven engines were affected.
- **A pre-arrival observation on a model-time-reading `[odes]` RHS no longer puts `NaN`
  into the FOCEI objective (#1124 review).** A subject observed before its first dose — a
  baseline sample — took the static superposition walk for its analytic sensitivities, and
  that walk resolves `TAD`'s anchor by a fold over doses at or before the segment start,
  leaving it at `-inf` and evaluating the right-hand side with `TAD = NaN`. The provider
  still reported success, so the `NaN` gradient reached the optimiser instead of falling
  back to finite differences. Such subjects now take the event-driven walk, which seeds the
  anchor at the first arrival. Affected fits change their reported `gradient_method` and
  objective.
- **Steady-state (`SS=1`) dosing on a model-time-reading `[odes]` RHS now takes finite
  differences on both loops (#1124).** The three gates that decline the analytic jet for this
  combination asked a predicate that could not see the bare `TIME` spelling, so a
  `TIME`-reading right-hand side reached the analytic steady-state equilibration while the
  identical model written with `T` correctly declined. Affected fits report a different
  `gradient_method` and a correspondingly different objective. This changes only which
  gradient is used; the steady-state *value* for such a model is wrong on every engine and is
  tracked separately in #1139.
- **An `[odes]` RHS that reads `TAD` is no longer silently dropped from the predictions
  (#1124).** A model whose right-hand side read the time-after-dose built-in — for example
  `- CL/V*central*(1 + 0.3*TAD)` on a mixed `first_order` + `zero_order` absorption model —
  was served by the closed-form modified-release fast path, which never evaluates that
  right-hand side. The `TAD` term was **absent** from the predictions, not approximated:
  output was bit-identical to the same model with the term deleted, reaching 8× the correct
  concentration by 12 h, with no error, no warning, and `converged: true`. Because the value
  is wrong, the error reached `IPRED`, the OFV, residuals, and every downstream diagnostic.
  Such models now integrate on the event-driven ODE path, which evaluates the right-hand side
  as written. The same routing covers `TAFD`, `T`/`t`, and `TIME`, including when read only
  from inside an `if` condition (measured 40% error, all four spellings) — the closed form's
  time-invariance probe samples two times and could not see any of them. Models that do not
  read model time are unaffected and keep the fast path.
- **An `[odes]` RHS that reads `TAD` together with a lagtime no longer returns `NaN`
  predictions (#1110).** The dense predictor anchored `TAD` on doses that had already arrived,
  and under a lagtime none has at the first segment's start, so every prediction came back
  `NaN`. These models now take the event-driven predictor, whose timeline starts at the lagged
  arrival. (#1073, released alongside this, independently gives the dense predictor a
  pre-arrival anchor, so the two engines now agree to 6e-12 on such a model rather than one of
  them being unusable.) One part of #1110 stays open and is **not** fixed here: the variance
  path used by `[diffusion]` / EKF models still reaches the dense predictor with empty
  parameter slices, returning `p_obs = NaN` and `ipred = 0` for any `TAD`-reading right-hand
  side, with or without a lagtime (#1131).
- **41 dead links and 11 duplicate anchors in the documentation (#1163).** Every internal
  documentation link now resolves, anchor included: numbered headings (`## 3. Communication` is
  addressed as `#communication`, not `#3-communication`), anchors hand-written with a double
  hyphen where the generated id has one, and links pointing at sections that never existed.
  Repeated headings on a page — four sections called "Syntax" on the absorption page, addressed
  as `#syntax`, `#syntax-1`, `#syntax-2`, `#syntax-3` — were given distinct titles, so some
  anchors on the published site changed. A new `docs-lint` check keeps all four properties true
  on every PR; see the [Docs linter](https://ferx-nlme.github.io/ferx-core/development/docs-lint.html)
  page.
- **An EVID=3/4 reset now re-seeds `[odes] init(...)` from the reset row's own covariates
  (#1133).** A reset row is a NONMEM data record — `$PK` runs at it — but ferx restarted the
  episode using the *previous* record's covariate snapshot, so a covariate-driven
  `init(state) = <expr>` began the new episode on a stale value. With `init(central) = 10*WT`
  and `WT` stepping 70 → 140 at the reset, every post-reset prediction was a factor of two out
  against NONMEM 7.6.0. Fixed in all four affected engines (the dense ODE predictor, the
  analytic-sensitivity twin that supplies the FOCE/FOCEI gradient, the adaptive-dosing driver,
  and its frozen-schedule replay verifier), so fitted parameters, `predict()`, and
  `simulate_adaptive()` all move. Datasets with time-constant covariates, and models without an
  `init(...)` seed, are unaffected. The same rule applies to the reset row's **occasion**: with an
  `iov_column`, `$PK` at the reset runs under that row's own `OCC`, so a reset that opens a new
  occasion re-seeds under the new occasion's κ rather than the previous record's — measured against
  NONMEM in `nonmem_anchor/reset_init_snapshot_J.ctl` (42.0 under the reset row's occasion against
  14.0 under the preceding record's). Note the remaining gaps: an **analytical** model using
  `[initial_conditions]` still drops its baseline at the first reset instead of re-depositing it
  (#1135), a reset row whose occasion carries no kappa group falls back to kappa = 0 inconsistently across engines (#1153), and an adaptive controller's decision-time covariate LOCF still skips reset rows (#1148).
- **A steady-state dose with a lagtime is now seeded at the dose record, not equilibrated at the
  arrival (#1121).** Under time-varying covariates an `SS=1` dose carrying an `ALAG` had its
  periodic trough computed *at the lagged arrival*, entirely under the dose row's covariate
  values. NONMEM loads the trough at the dose **record** (phase `II − ALAG`) and advances to the
  arrival under the record governing that interval, so ferx applied too little elimination across
  the pre-arrival window and ran high for the rest of the cycle. On an eight-subject anchor
  (`nonmem_anchor/dose_form_lag_ss`) this was worth **7.67 objective units**; predictions are now
  within 1e-4 of NONMEM 7.6.0 point by point. Flat-covariate fits are unaffected — the two
  constructions agree exactly when nothing changes inside the window, which is why this went
  unnoticed. The same fix gives the IOV predictor a pre-arrival state, where it previously read
  zero for any observation between an `SS` dose's record and its lagged arrival.
- **A steady-state infusion whose previous cycle is still running at the dose record keeps
  delivering (#1121).** When `ALAG > II − T_inf` the pulse before the record is recent enough that
  its infusion has not finished, so the rate flows on past the record and stops at
  `record + (T_inf − phase)`. That window belongs to no dose row, and every engine dropped it —
  reading about **4 %** low across the whole pre-arrival window and, since the two production
  engines and the dense predictor disagreed about it, giving the same model different answers
  depending only on whether the subject happened to carry a time-varying covariate. Now carried
  explicitly and matched against NONMEM 7.6.0 (`nonmem_anchor/ss_lag_infusion`).
- **A lagtime of a full dosing interval or longer on a steady-state dose is now supported
  (#1121).** `ALAG >= II` has no phase `II − ALAG`; ferx now clamps it to zero, matching NONMEM —
  the pulse lands on the dose record, so the record carries the steady-state peak and decays from
  there. Previously the event-driven walks returned exactly `0` for the whole pre-arrival window
  (under a proportional error model, a very loud failure), while the static predictors wrapped the
  phase into `[0, II)` and then re-equilibrated at an arrival that is no longer the periodic
  trough, running about 4 % high after it. Clamping is also continuous in `ALAG`, so an
  *estimated* lagtime no longer steps the objective as the optimiser walks it across the interval.
  One corner is a deliberate divergence: for an `ALAG >= II` **infusion** NONMEM delivers
  `T_inf + (ALAG − II)` hours of drug rather than the dose's own `T_inf`, which is not
  mass-balanced; ferx does not reproduce it. See `docs/model-file/lagtime.qmd`.
- **VI no longer freezes from ordinary starting values (#1097).** Adam's gradients are now
  clipped to a global L2 norm, controlled by the new `vi_grad_clip` fit option (default `1e4`;
  `0` restores the old behaviour). Under a proportional error model a prediction near zero
  makes the ELBO gradient enormous, and Adam's second moment — an average of *squares* with a
  ~1000-iteration memory — absorbed it and left every subsequent step numerically zero for tens
  of thousands of iterations. The frozen trace then read as settled, so the run stopped early and
  reported `σ` welded to its internal `exp(5)` guard. On warfarin from the `ferx-testdata`
  initial estimates this cost ~1594 objective units against FOCEI; the same fit now reproduces
  the NONMEM FOCEI estimates (`TVCL 0.13269`, `TVV 7.7379`, `σ 0.0116`) with no warm start.
  Fits that already converged are unaffected — clipping is asymptotically a no-op because Adam
  is scale-invariant in steady state.
- **VI's bad-basin detector now reports `converged: false` (#1098).** An implausibly loose
  final ELBO already identified a fit trapped far from a usable variational approximation, but
  the result still exposed `converged: true` to programmatic consumers. Such fits are now demoted,
  omit the misleading "increase `vi_iters`" warning, and carry the critical structured warning
  code `vi_bad_basin`.
- **The analytic gradient of an `[odes]` RHS that reads `TAD` under an estimated lagtime is now
  correct (#1070).** `TAD` was handed to the ODE right-hand side as a plain constant, but its
  anchor *is* the dose's lagged arrival (`t_dose + ALAG`), so the `∂TAD/∂ALAG = −1` term was
  silently missing from the analytic η/θ gradient — everywhere the trajectory is integrated, not
  just at a dose event. Predictions were unaffected, so nothing failed visibly; but FOCEI builds
  its `h` matrix from that same gradient, so the error reached the reported OFV. Measured
  against finite differences of the production predictor, the worst axis was 66% wrong
  (`∂f/∂η_ALAG`) and 68% wrong (`∂f/∂θ_ALAG`); against NONMEM 7.6.0 the objective was 0.176 off
  on a single-dose anchor and 0.107 on a two-dose one. `TAD` is now threaded through the
  sensitivity walks as a dual, so those become 0.0004 and 0.032 — the same values the finite-
  difference route gives (`nonmem_anchor/tad_lag_*`). Affected models keep the fast analytic
  route on both loops and under IOV, including when the lagtime carries inter-occasion
  variability. Predictions are bit-identical to before. Models reading `TAD` **without** a
  lagtime, or reading `TAFD` **with** one, were always exact and are untouched. `TAD` combined
  with a **steady-state** dose still routes to finite differences, under the separate
  non-autonomous-RHS rule.

  Two consequences worth knowing before you re-run an affected model. First, `gradient_method =
  auto` resolves this model class to **L-BFGS** where it previously chose BOBYQA, because the
  choice keys on whether an analytic outer gradient is available — so the optimizer changes as
  well as the gradient. Set `optimizer = bobyqa` to keep the previous behaviour. Second, the
  second-order block `∂²f/∂η_ALAG²` is improved but **not** exact (6.4e-1 wrong before, 2.2e-1
  after); it does not affect the objective value, but it does feed the analytic outer gradient
  and the covariance step, so standard errors and the optimizer's search direction on these
  models carry that residual until #1075 lands. The `TAD` value in the window **before the first
  dose arrives** is differentiated consistently with whatever convention the predictor uses, but
  that convention itself is still open (#1110) and is not anchored against NONMEM — NONMEM has
  no `TAD` built-in in `$DES`, so there is nothing external to anchor it to.
- **A lagged dose's covariate snapshot no longer stretches to its arrival (#1073).** With a
  lagtime, ferx broke the integration timeline only at the arrival `t + ALAG`, never at the dose
  row's own time, so the dose row's covariate / IOV snapshot governed everything up to the
  arrival. NONMEM evaluates `$PK` at every data record and then advances *to* that record, so the
  interval containing a lagged arrival belongs to the record that **terminates** it. On the
  committed multi-dose anchor this was worth **14.89 OFV** (ferx `−474.660106` vs NONMEM
  `−459.772592`); it is now `~1e-5`. Applying the same rule to the other non-record boundaries —
  an infusion end, a zero-order absorption cutoff, a per-route onset falling strictly between two
  records — fixed a second divergence of the same family in the ODE engine, measured at **4.2 %**
  on the predictions after the boundary and **23.5 OFV** on a dedicated NONMEM probe. The
  closed-form engine already had that one right, so the two production engines now agree where
  they used to differ. **This moves converged fits** for any model combining a lagtime with
  time-varying covariates or IOV, and for any ODE model whose infusion or zero-order window ends
  between records under a changing covariate; re-run affected analyses rather than comparing
  estimates across the change. Fixed in all four engines — both production predictors and both
  analytic-sensitivity twins — so FOCE/FOCEI, SAEM, VI and the HMC sampler move together,
  and in the reactive adaptive-dosing walk (#391), where a base infusion whose window ended
  between two records under a changing covariate was **11 %** off the static engine.
- **`TAD` is finite before a lagged first dose arrives (#1073).** An `[odes]` RHS reading the
  `TAD` builtin previously saw `NaN` for any step between the first dose row and that dose
  landing, which multiplies into the compartment state and turns an otherwise finite fit into
  the `1e20` objective sentinel. `TAD` there is now measured from the subject's first arrival —
  negative in that window, reaching `0` when the dose lands — and depends only on the dosing,
  so adding an observation inside the window cannot move a later prediction. Both ODE
  predictors agree, so two subjects of the same model no longer behave differently depending on
  whether they carry an `EVID=3/4` row. A subject with no doses at all still reads `NaN`. What
  `TAD` *should* mean before a dose has arrived remains open (#1110).
- **`vi_mc_samples` now defaults to 32, not 8 (#1017).** This draw count sets the noise floor
  VI's settling test measures against, so it decides *where a fit stops* — and therefore what is
  certified — rather than merely how noisy the trace is. At 8, on `data/warfarin.csv` (~1%
  proportional residual), `σ` landed 31% above the `0.010565` both AGQ (`n_agq = 9`) and FOCEI
  give, with the OFV 9.6 units short of AGQ's `−285.977`, and it was not reliably flagged: the
  drift check only demotes a run still descending when it stops, and at 8 draws the run can
  instead settle at a biased fixed point where the trace tail has genuinely plateaued and
  `elbo_tightness_ratio` reads a healthy `0.989` — `vi_seed = 99` with otherwise default options
  returned `converged: true` at `σ = 0.013761`. At 32, `σ` is +10% with the OFV within 1.4 units
  and no seed tried certified a materially wrong fit (a bad-basin seed was correctly reported
  `converged: false`). Cost is sublinear, because a lower noise floor also settles sooner: 4× the
  draws cost ~2.3× the wall time (3.4 s → 7.7 s on this fit). Set `vi_mc_samples = 8` explicitly
  to restore the old behaviour, and check `σ` against a `laplace` / `focei` fit if you do.
- **`methods = [vi, laplace]` with `n_agq > 1` is no longer rejected (#1017).** `n_agq` is a
  chain-wide option, so the documented VI readout — `methods = [vi, laplace]`,
  `agq_eval_only = true`, which turns VI's ELBO lower bound into a real `−2 log L` — carries it
  legitimately: the grid belongs to the Laplace stage and VI ignores it. The check rejected it
  for the whole chain, making the recommended path impossible to ask for. It now fires only
  when no stage consumes the option (a VI-only chain).
- **A VI run that stops while its objective is still falling no longer reports `converged: true`
  (#1017).** The settling test asks whether the ELBO's remaining drift is distinguishable from
  Monte-Carlo noise; at a low `vi_mc_samples` that can become true while the objective is still
  descending, so the fit stopped short and certified itself. A sign test over the trace tail now
  separates a real trend from noise — below the amplitude of any single window comparison, which
  is what makes such drift invisible to the settling test — and a run stopped that way reports
  `converged: false` with a warning naming `vi_mc_samples`.

  **The sign test alone did not make the old `vi_mc_samples = 8` safe on a small residual
  error, which is why the default moved to 32 (see above).** On
  `data/warfarin.csv` (~1% proportional residual) the default draw count lands `σ ≈ 0.0138`
  against `0.010565` from both AGQ (`n_agq = 9`) and FOCEI — 31% high, OFV 9.4–9.6 units short of
  AGQ's `−285.977`, and every variational covariance ~1.7× too wide. The sign test catches this
  only in the regime where the run is still descending at the `vi_iters` ceiling. It can also
  settle at a *biased fixed point* of the sampled objective, where the trace tail has genuinely
  plateaued and no trend test can see anything: with `vi_seed = 99` and otherwise default options
  the fit stops at 17 125 iterations reporting `converged: true` at `σ = 0.013761`. Raise
  `vi_mc_samples` (32 → 1.4 OFV, 128 → 0.37) or check `σ` against a `laplace`/`focei` fit when the
  residual error is under ~3%. See the [VI page](docs/estimation/vi.qmd).
- **VI's early stopping now requires the per-subject posteriors to have settled, not just the
  population parameters (#1017).** Either convergence criterion is sufficient, and the
  parameter-stability one measured only the packed `(θ, Ω, σ)` vector. A chain in which the
  population coordinates settle first — or in which most are FIXed — could therefore stop and
  return a still-moving variational posterior as converged, which is the object a VI fit exists
  to produce. Both halves are now judged on the same window and tolerance.
- **`block_sigma` correlated residual errors work with `method = vi` (#1017).** The option docs
  said this configuration falls back to Adam on `σ`, and the code implements that, but the
  check-time allow-list omitted VI, so every `block_sigma` + VI fit was refused before the
  fallback could run. VI's data term routes a dense `R` through the same full-FD path FOCE and
  SAEM use, so it is supported; only the closed-form `σ` maximizer is not, and that declines
  with a recorded reason as documented.
- **A VI block overtaken by a later estimating stage now says so (#1017).** On
  `methods = [vi, focei]` the reported θ/Ω/σ and subject diagnostics come from FOCEI while
  `vi.eta_means`, `eta_covs`, both ELBO halves and `elbo_tightness_ratio` still describe VI's
  parameter point. The block is still reported — a trailing `agq_eval_only` / `imp_eval_only`
  readout does not move the estimates, and the variational covariance is the only per-subject
  one in the result — but it now carries `superseded_by` (in the API and the fit YAML) naming
  the stage that moved on, plus a warning.
- **Subject IDs in the VI posterior YAML are quoted (#1017).** `id: 001` parsed back as the
  integer `1`, and an ID containing `:` or `#` could change the document's shape or truncate the
  value — defeating the point of keying the posterior by the original subject ID.

### Added
- **Correlated-residual fits now retain their fixed `block_sigma` correlations (#1100).**
  `FitResult.residual_correlations`, JSON, `.fitrx`, and fit YAML now carry the declared
  correlations, so consumers can reconstruct the sigma-scale residual covariance without re-reading
  the model source; diagonal-sigma output remains unchanged. The fit YAML's `block_sigma` section
  reports `correlation_fixed` (always true) separately from `covariance_fixed`, which is true only
  when both sigma SDs were declared `FIX`.
- **Theta level blocks — hundreds of fixed effects from one declaration (#1064).**
  `theta PLACEBO[800](0.0, -10.0, 10.0)` declares 800 thetas sharing one init/bounds triple, read back
  by a *gather* — `PL = PLACEBO[PLA_IDX]`, where `PLA_IDX` is a 1-based data column. The data-driven
  form `theta PLACEBO[STUDY, TIME](0.0, -10.0, 10.0)` — data columns in the brackets rather than a
  count — instead declares one theta per **observed**
  combination of the named columns, discovered when the data is bound, and reports them as
  `PLACEBO[STUDY=7,TIME=4]`. This is what makes an unstructured placebo effect writable in a
  model-based meta-analysis — and fittable: the whole block occupies a single parameter slot, so it is
  not bounded by the fixed individual-parameter layout the way 800 separate parameters would be.
  A level block is rank-deficient against a fixed intercept *and* against a random effect at the
  same or a coarser grouping, so a sum-to-zero convention is always applied and its grouping is chosen
  from both the model and the data; `contrast = sum_to_zero | sum_to_zero_within | ref | none` overrides
  it, and a choice that leaves a nested group's mean free is refused. An out-of-range or non-integral
  gather index is reported before the fit starts, naming the column and value. See
  `docs/model-file/parameters.qmd`.
- **Three-argument `clamp(x, lo, hi)` in the model DSL (#1092).** Bounding a readout into an
  interval no longer has to be written as the nested `min(max(x, lo), hi)`, whose argument order
  flips between the inner and outer call. Available in every expression the DSL parses
  (`[individual_parameters]`, `[scaling]`, `[odes]`, `[derived]`); it desugars to the inline
  conditional, so it differentiates and compiles exactly like the nested form. `clamp` with any
  arity other than three is a parse error — a one-argument `clamp(x)` would otherwise have been
  read as the silent identity — and literal bounds given the wrong way round (`clamp(x, 0.9, 0.1)`)
  are rejected rather than quietly returning a bound for every `x`. One input is deliberately not
  the nested form: `NaN` fails both bound tests, so `clamp` propagates it into the objective
  function, where `min(max(x, lo), hi)` would have pinned it to `lo` and let a wrong number fit.
- **`[simulation]` can now state the covariates of the arms it invents (#1083).** `covariate NAME = <value>`
  — a scalar for every subject, or `= [v1, v2, ...]` with one value per subject — gives the synthetic
  subjects a covariate value, which a `[simulation]` design previously had no way to express at all.
  This is what makes trial simulation work for a model-based meta-analysis: `kappa K ~ γ² weight = NARM`
  and `DV ~ additive(S) weight = WPSE` both read a data column, and the arm a simulation is proposing
  has no row to read it from. A model that references a covariate the design does not supply is now
  refused by name, with the fix in the message, rather than reported as a column "not found in data" on
  a path that has no data file. See `docs/model-file/simulation.qmd`.

- **`ode_method = auto` picks the ODE stepper for you, and is now the default (#978).** An a-priori stiffness probe builds
  the Jacobian `J = ∂f/∂u` of the `[odes]` right-hand side at the state each integration segment
  starts from and reads its fastest mode, `max |Re λ(J)|`; a system above `30` per time unit starts
  on a stiff Rosenbrock method (`rodas4`, or `rodas5p` at `ode_reltol ≤ 1e-8`), everything else stays
  on the explicit default. Probing **per segment** rather than once per model is what makes it work
  on binding/TMDD models, whose fast eigenvalue is carried by a term like `KON · central` and so is
  identically zero at the declared initial condition — the one state at which such a model looks
  non-stiff — and it re-decides as the optimizer moves θ. An escalation that returns a non-finite
  trajectory, or that clamps at the minimum step (the freeze-pad failure that produces finite,
  silently wrong output), is discarded and re-solved explicitly; if that fallback also fails, the
  solver reports that both attempts were unusable rather than presenting the retry as a successful
  repair. The guard applies to `auto` only, and a
  named `ode_method` is honoured exactly as before. Measured on ferx-testdata: the stiff cyclophosphamide model
  converges in 0.7 s at OFV 3241.708 (NONMEM FOCEI: 3241.721) where `rk45` spends 827 s
  without finishing one optimizer iteration, and a TMDD fit runs 372.4 s → 2.3 s at a
  fixed iteration budget; nothing changes where nothing is stiff. Two new counters on the solver-statistics struct record what it did — segments escalated, escalations rejected — and the fit reports them through the `ode_solver` warning (#1080). The
  threshold is a rate and therefore carries the model's time unit (calibrated on the hour-based PK
  convention), so name a method explicitly on an unusual time scale. See
  [ODE models → Letting ferx pick the stepper](https://ferx-nlme.github.io/ferx-core/model-file/ode-models.html#letting-ferx-pick-the-stepper-ode_method-auto).
- **Fits now report what the ODE solver did (#1080).** A new `ode_solver` warning summarises one
  post-fit prediction pass over every subject: steps that clamped at the minimum step size (a
  stability-limited segment whose un-integrated tail is freeze-padded with the last state), segments
  `ode_method = auto` escalated to a stiff method, and — the actionable one — escalations the guard
  had to discard and re-solve explicitly, which means the stiffness probe was right that the segment
  is stiff and wrong about which stiff method could integrate it. Until now no production path
  reported solver statistics at all, so both the escalation and its rejection were invisible outside
  the test suite. An escalation that simply worked is reported at `Info` severity; anything that did
  not integrate cleanly is a `Warning`. The counters ride along in the warning's `details` payload.
- **`ode_method = auto` can now change stepper *inside* a segment (#1080).** The stiffness probe reads
  the Jacobian at each segment's entry state, which is the state a binding model looks least stiff at —
  both factors of a `KON · C · R` term are zero when a dose lands. A depot-absorption binding model
  reads `|Re λ|max = 10` on entry and `7.6e3` an hour later; integrated explicitly, that segment
  exhausts `ode_max_steps` and comes back with a **median relative error of 143 %**, and it does so
  with **zero** minimum-step clamps and a lower step-rejection rate than the accurate cases, so no
  step-history counter can see it. `auto` now re-runs the probe every 25 accepted steps and swaps the
  stepper in place when the verdict changes, keeping the state integrated so far: 143 % error in
  10 000 steps becomes `7e-8` in 90, matching the method a user would have had to know to name in
  advance. A mid-segment escalation is guarded exactly like a start-of-segment one (discarded and
  re-solved explicitly if it comes back non-finite or clamps on the stiff stepper), the event-time
  path switches on the same rule, and a benign segment stays bit-identical — it is never re-probed
  at all below 25 accepted steps, and pays about one probe per 25 steps above that. New
  `ode_auto_switch = false` restores one method per segment, chosen at its start; a named
  `ode_method` is pinned as before. See
  [ODE models → When a segment turns stiff halfway through](https://ferx-nlme.github.io/ferx-core/model-file/ode-models.html#mid-segment-switching).
- **`ode_stiff_abort_after` bounds what a stalled ODE segment costs (#708, #1080).** A segment that is
  stability-limited keeps stepping at the minimum step size until it exhausts `ode_max_steps`;
  setting this key gives up after that many clamped steps instead. Off by default and deliberately
  so — aborting freeze-pads the segment's remaining output times, and it does so on every
  likelihood evaluation, which makes the objective discontinuous in θ; a fit whose `ode_solver`
  warning reports aborts is a diagnosis ("these segments are stability-limited"), not an estimate.
  It is a way to make a grinding fit say so quickly, not a substitute for choosing a stiff method.
  On the time-to-event path an abort is reported as a failed segment rather than padded. See
  [ODE models → Which regime am I in?](https://ferx-nlme.github.io/ferx-core/model-file/ode-models.html#which-regime-am-i-in).
- **The `init(...)` scope rule is now published by the engine (#994).** `ODE_INIT_SCOPE_BUILTINS` and
  `ODE_INIT_REJECTED_BUILTINS` name the built-ins an `[odes] init(...)` expression may and may not
  reference, so a code generator can source the rule from the same binary it will parse with instead
  of mirroring it by guesswork — the arrangement `known_block_names()` (#1040) already provides for
  block names. The `[odes] init(...)` diagnostics are rendered from the same lists, so message and
  guard cannot drift. They are named for the `[odes]` surface because the two surfaces genuinely
  differ: outside `[odes]`, `MACHEPS` / `TAFD` / `TAD` are ordinary covariates rather than
  solver-injected built-ins (the same deliberate rule `[scaling]` follows), so only `TIME` is
  rejected in `[initial_conditions]`. Both surfaces are documented side by side under
  [ODE models → What an `init(...)` expression may reference](https://ferx-nlme.github.io/ferx-core/model-file/ode-models.html#init-scope).
- **`center` / `scale` on `[covariate_nn]`** — per-input normalization, so the network
  sees `(x - center) / scale`. Both default to the identity, leaving existing models
  unchanged. Raw covariates are badly scaled for a neural net: `WT ≈ 70` saturates a
  `tanh` layer at initialization, where its derivative is ~0 and the layer is nearly
  blind to its input. On a two-covariate DCM, unnormalized inputs pushed weights to ~1e11
  and left the residual error 15× too high; the same model with standardized inputs kept
  every weight inside `[-1.3, 1.6]`. The constants are declared rather than estimated
  from the data so that `predict()` applies the same transform the fit used — recomputing
  statistics on new data would silently change the model — and so the model file remains
  a complete description of the transform, as `(WT/70)^0.75` already is. The fit output
  records the transform alongside the weights: `FitResult.neural_networks[k]` carries
  `input_center` / `input_scale`, and the `neural_networks:` YAML block prints `center:` /
  `scale:` when the model declares them — the reported weights were fitted against
  `(x - center) / scale`, so a consumer that reconstructs the network needs both.

- **`init` on `[covariate_nn]`** — one starting value per output, on the parameter's own
  scale (`init = [1.0, 10.0]` for a CL of 1 L/h and a V of 10 L). **Set this on every
  DCM.** Without it every output-layer bias starts at 0, so a `softplus` head starts
  *every* PK parameter at `softplus(0) = 0.693` — a 0.69 L volume regardless of what the
  parameter means — and the fit begins orders of magnitude away from the data. That is not
  a slow start, it changes the answer: on a 60-subject busulfan-shaped deep compartment
  model with identical data and seed, the bare head converged to `−2 log L` 2033.6 with the
  clearance decline understated 43%, and **reported `converged: true`**; declaring
  `init = [1.0, 10.0]` reached 1061.0 and recovered the decline. The value is realised
  exactly rather than approximately — the output-layer weight block is zeroed alongside the
  biases, so the network emits exactly `init` for every subject at iteration 0 whatever its
  covariates, and those weights take gradient from the first step. Values must be reachable
  by the output activation, or the model file is rejected rather than producing a `NaN`
  weight.

### Performance
- **The `ode_method = auto` probe is ~10× cheaper on models that are not stiff (#1080).** Before
  running the eigensolve, the probe now checks Gershgorin's bound on the Jacobian's spectrum: a bound
  below the stiffness threshold *proves* no eigenvalue reaches it, so the segment can keep the
  explicit stepper without the `O(n³)` decomposition. Ordinary absorption/elimination segments — the
  overwhelming majority of what a fit integrates — exit there. Measured on a 3-state binding model:
  the probe drops from 0.56 µs to 0.055 µs per segment, taking `auto`'s overhead over a pinned `rk45`
  from 33 % to 3 % on a bare segment and from 19 % to 1.3 % on a realistic observation grid. No
  decision changes — the bound only short-circuits the direction it can prove.

### Changed
- **The ferx website and GitHub landing page are easier to discover and share (#1089).**
  Documentation pages now publish descriptive search and social metadata, canonical URLs, and
  clearer NLME and population PK/PD summaries while retaining the generated sitemap and crawler
  instructions.
- **`optimizer = auto` no longer picks BOBYQA on high-dimensional problems (#1064).** BOBYQA interpolates
  a quadratic over the whole parameter space, so its model grows quadratically in the parameter count;
  above 64 free coordinates `auto` now resolves to `nlopt_lbfgs`, whose finite-difference cost is linear.
  An explicit `optimizer = bobyqa` is still honoured, now with a warning about the size.
- **The covariance step routes away from `MATRIX=R` on high-dimensional problems (#1064).** The default
  `covariance_method = r` re-converges every subject's EBEs at each of `n(n+1)/2` stencil points — around
  320,000 population objectives at 800 parameters. Above 100 free coordinates a *defaulted* covariance
  method now uses the score cross-product (one pass) and says so; setting `covariance_method = r`
  explicitly still forces it, with a warning naming the cost.
- **Large theta level blocks are reported compactly (#1064).** A block of more than 20 free coefficients
  leaves the main estimate table for a one-line summary (count, min, median, max) on the console and in
  the text report. Every independently estimated coefficient is written to the fit YAML under
  `theta_blocks:`; constrained dependent levels are derived values rather than estimates.
- **ODE models now choose their own stepper by default (#978).** `ode_method` defaults to `auto`
  instead of `rk45`, so a model that names no stepper is probed per integration segment and runs
  a stiff method on the segments that need one. A model that *does* name a method is unaffected —
  naming a stepper still pins it exactly, probe and all. Two consequences worth knowing:
  predictions on a **stiff** model will change, because they are now produced by a different (and
  better-conditioned) integrator — on the cyclophosphamide model this is the difference between a
  fit that converges in 0.7 s and one that spends 827 s without finishing an optimizer iteration;
  and a model that reports its resolved options will show `auto` where it used to show `rk45`.
  Most non-stiff models keep the explicit stepper and their existing numbers, paying one Jacobian
  and one eigensolve per segment — the work of a single Rosenbrock step attempt — for the check.
  Two kinds do not, because the probe reads how *fast* a system's fastest mode is and not how
  *separated* its modes are: a model on a minute clock (the threshold is a rate, calibrated for
  hours) and a model whose modes are all equally fast (a transit chain written out in `[odes]`
  with a large `ktr`). Those escalate without being stiff, which costs time rather than accuracy.
  Pin `ode_method = rk45` to restore the previous behaviour exactly.
- **The Rosenbrock steppers are promoted from experimental to beta (#978).** Making `auto` the
  default puts them on the default path, so leaving them marked experimental would have meant
  shipping experimental components to every ODE user. The promotion rests on this release's
  validation: a NONMEM FOCEI anchor on the cyclophosphamide model (ferx 3241.708 vs NONMEM
  3241.721) and prediction-equivalence tests across all five methods.
- **The naproxen MBMA fixture is re-sourced to its primary publication, and is licensed CC BY-NC 4.0
  rather than MIT (#1085).** `data/mbma_naproxen.csv` was derived from the supplementary material of
  Bracis et al. (*CPT:PSP* 2026;15:e70158), which is CC BY-NC-**ND** — a licence that forbids
  distributing an adapted copy at all. The dataset is not theirs: it originates with Boucher &
  Bennetts, *Many Flavors of Model-Based Meta-Analysis: Part II*, *CPT:PSP* 2018;7:288–297
  (CC BY-NC, no ND clause), whom that tutorial replicates. The fixture is now derived from the
  primary source instead, which removes the no-derivatives problem and makes the attribution
  correct. Everything in the repository stays MIT **except** `data/mbma_naproxen.csv`, which carries
  CC BY-NC 4.0 — so commercial use of that file is not granted. The exception is recorded in the
  root `LICENSE`, in `data/mbma_naproxen.LICENSE`, and in the data README, and `Cargo.toml` now
  excludes the files — together with the slow test that reads them, which would otherwise panic on a
  missing fixture — so the crate published to crates.io is uniformly MIT. The data itself is
  unchanged but for one digit: the tutorial's precomputed `WP / WPSE` column stores 7 decimals, and
  for one of 122 rows that intermediate rounds the sixth decimal down where the primary source's
  `WP` and `WPSE` round it up. No estimate or OFV moves.
- **Every line in `[structural_model]` is now checked (#811).** The block was scanned for the first
  `pk NAME(...)` match with an unanchored pattern and every other line was discarded, so
  `zpk one_cpt_iv(cl=CL, v=V)` parsed as a valid one-compartment IV model and a mistyped or stray
  line vanished without a word. Each line must now be one of the four accepted forms —
  `pk NAME(...)`, `ode(states=[...])`, `ode_template NAME(...)`, or an equation line — and anything
  else, a second disposition line, or a mix of a compartment model with equation lines is a parse
  error naming the offending line. A model file that relied on the old leniency was already being
  read as something other than what it said.
- **A single-endpoint `[error_model]` must now name its sigmas in declaration order (#1001).**
  A one-line `DV ~ ...` error model consumes its sigmas **positionally** from the `[parameters]`
  declaration order. The names written in the arguments were checked for existence and then
  discarded, so `DV ~ proportional(S_SMALL)` bound `S_BIG` whenever `S_BIG` happened to be declared
  first — and `combined(A, B)` ignored a transposition of its two arguments entirely. The fit
  converged and every reported SE and diagnostic was internally consistent, so there was nothing to
  notice; on a 24-observation dataset two sigmas 40× apart moved the objective by 110.7 units
  (confirmed on both engines against NONMEM 7.6.0, `nonmem_anchor/sigma_order_{small,big}.ctl`).
  Mismatches are now rejected at parse time with `E_SIGMA_ORDER_MISMATCH`, naming the argument, the
  sigma that actually occupies the slot, and the fix. This generalises a check that previously fired
  only when `block_sigma` was present. Per-CMT and covariate-selected error models bind by name and
  are unaffected, as is a trailing sigma consumed elsewhere (e.g. FREM's `frem_sigma`). **Breaking**
  for any model that named its sigmas out of order — such a model was already getting a different fit
  from the one it appeared to describe. Fix it by reordering the `sigma` **declarations** (and, for a
  `block_sigma`, permuting its lower triangle to match); reordering the *arguments* instead also
  parses but swaps which sigma is the proportional component, which changes the model rather than the
  spelling.
- **The dose-attribute double-use error now covers analytical (`pk ...`) models (#1004).** #993
  rejected this on ODE models only, reasoning that an analytical model's explicit
  `pk(..., f=F)` mapping made a second use "stated rather than silent". It is not: nothing in the
  model says the value is applied twice, and a `[scaling]` or `[adaptive_dosing] observe`
  expression that reads a mapped `f=`/`lagtime=` parameter applied it
  once at the dose and once where it was read — on the **default** engine, with no diagnostic.
  Measured at exactly `F` on the prediction. Now rejected at parse time with the same
  `E_DOSE_ATTR_DOUBLE_USE` code. The remediation differs from the ODE engine's: there the *name*
  routes the parameter, so renaming fixes it; here the **mapping** binds it, so the fix is to drop
  the `f=`/`lagtime=` argument (or the read), and the message says so. A parameter merely *named*
  `F` that no `pk(...)` argument maps stays an ordinary parameter — unchanged, as does an
  `[initial_conditions]` read: an initial condition is not an absorbed dose, so the engine seeds
  the amount with `F = 1` and no lag and `init(depot) = F * 500` applies `F` once. **Breaking** for a
  model that maps a dose attribute and also reads it, e.g. the apparent-volume idiom
  `pk(..., f=F)` + `obs_scale = V / F`; a model using `CL/F`, `V/F` apparent parameters *without*
  mapping `f=` is unaffected, which is the ordinary NONMEM convention. Note this is again
  **stricter than NONMEM**: `$PK` defining `F1` **and** `S2 = V/F1` runs clean under `ADVAN2` and
  returns predictions scaled by exactly `F1` — anchored on NONMEM 7.6.0, two streams differing in
  one `$PK` line (`nonmem_anchor/analytical_dose_attr_double_use_{A,B}.ctl`).
- **`method = imp`, `impmap` and `bayes` are now rejected by `ferx check` on a model with no
  random effects (#1007).** All three already refused `n_eta = 0` at run time, so a
  `methods = [focei, imp]` chain ran its whole FOCEI stage before failing and `ferx check`
  reported the model as valid. The new `E_METHOD_NO_RANDOM_EFFECTS` diagnostic fires up front,
  anywhere in a method chain, matching the `saem` guard added in #1002. The run-time errors stay
  as the backstop for direct `fit()` callers. One consequence to note when upgrading: a chain with
  `imp_eval_only = true` on a fixed-effects-only model previously returned a fit result with the
  IMP failure downgraded to a warning, and now returns this error instead.
- **`method = gn` on a fixed-effects-only model now warns (#1006).** Pure Gauss-Newton is
  start-sensitive at `n_eta = 0`: with no inner EBE loop to absorb a poor `sigma` start, the BHHH
  step can collapse far from the optimum and return a badly wrong answer whose only signal was
  `Converged: NO`. `ferx check` now emits `W_GN_NO_RANDOM_EFFECTS` pointing at `gn_hybrid` /
  `focei`, and an unconverged pure-GN run at `n_eta = 0` adds a matching post-fit warning. A
  warning rather than an error, since `gn` does reach the optimum from a good start. Both are
  suppressed when a later stage re-optimises the GN result — `gn_hybrid`, and equally a
  hand-written `methods = [gn, focei]`.
- **An unrecognised `[block]` name is now an error (#1040).** Blocks were read by name lookup, so a
  header the parser did not know was never read and never reported: a misspelled `[fit_option]` left
  `ferx check` saying `valid: true` while the fit ran with the default method, the default iteration
  cap and **no covariance step** — returning without standard errors and no indication why. The same
  went for `[scalings]`, `[outputs]`, `[derived]`, `[covariates]` and friends. Block names are now
  closed-world, like the keys inside a block already were: an unknown header is `E_UNKNOWN_BLOCK`,
  listing every offender with its line, the full valid set, and a did-you-mean for a near match. Two
  neighbouring silent drops go with it — an instance name where none is taken (`[fit_options DOSE]`)
  or missing where one is required (`[covariate_nn]`) is `E_BLOCK_INSTANCE_NAME`, and a block whose
  cargo feature this binary lacks (`[event_model]` without `--features survival`, `[markov_model]`
  without `--features markov`) is `E_BLOCK_FEATURE_DISABLED` instead of being parsed away, and
  `[initial_values]` — ferx's own former spelling for initial estimates, unread since they moved
  inline into `[parameters]` — is `E_DEPRECATED_BLOCK`, naming the replacement rather than offering
  a did-you-mean that does not exist. The recognised block names are exported as
  `known_block_names()` so wrappers can read the list from the engine rather than keeping their own
  copy; it reflects the features the binary was built with, so it never advertises a name the same
  binary would refuse.
- **A dose attribute that is also read by the model is now an error (#993).** `F`,
  `LAGTIME`/`ALAG` and the compartment-indexed `F{n}`/`ALAG{n}`/`LAGTIME{n}` are applied by the
  engine **at the dose event**. A model that declares one and *also* references it in the `[odes]`
  **RHS**, the `[scaling]` readout, or the `[adaptive_dosing] observe`
  controller signal was applying it twice — silently, and by exactly that factor: an
  ODE model reading its own `F` produced every prediction scaled by `F`, and renaming the parameter
  changed the fit by that amount with no diagnostic either way. This is now rejected at parse time
  with `E_DOSE_ATTR_DOUBLE_USE`, naming both readings and the fix. `D{n}`/`R{n}` carry the same
  reservation but are consulted only for a coded `RATE=-2`/`-1` dose, so that collision is reported
  against the dataset (same code) and a model whose data never codes `RATE` is untouched. Reads from
  `[derived]`/`[output]` are post-solve reporting and remain silent, and so is an **initial
  condition** on either engine (`[odes] init(state) = …`, `[initial_conditions]`): the engine seeds
  the state with the raw expression value and consults dose attributes only at dose events, so
  `init(central) = F * 100` — the bioavailable residue of a pre-study dose — applies `F` exactly
  once and is a legitimate model ([#1046](https://github.com/FeRx-NLME/ferx-core/issues/1046)).
  NONMEM agrees: `A_0(1) = F1*100` with `F1 = 0.5` seeds 50, not 25, and `A_0(1) = ALAG1*100` is
  deposited unshifted, each run byte-identical to the twin seeding from an ordinary parameter of
  equal value (`nonmem_anchor/odes_init_dose_attr_{f,lag}_{A,B}.ctl`). (Analytical models were left out
  of this first pass; they are covered by #1004 above.) **Breaking** for a model that folds `F` into the absorption flux
  — the pre-dose-entry convention the ODE docs' migration note describes, which until now computed
  `F²` without complaint; the fix is to drop `F` from the right-hand side, or rename the parameter
  if it was never bioavailability. Note this makes ferx **stricter than NONMEM**, which allows a
  `$PK` `F1` to be referenced in `$DES` and quietly computes `F²` — so a mechanically translated
  control stream can newly fail to parse even though it ran in NONMEM. Anchored on NONMEM 7.6.0:
  two `ADVAN13` streams differing in one `$DES` line give predictions differing by exactly `F1`,
  with no diagnostic from NONMEM (`nonmem_anchor/dose_attr_double_use_{A,B}.ctl`).

### Fixed
- **A failed `ode_method = auto` fallback is no longer reported as a successful repair
  (#1080).** When the stiff attempt is rejected and its pinned-`rk45` re-solve also stops before
  the segment end or returns non-finite output, the `ode_solver` warning now says that both
  attempts failed. The escalation guard also rejects a stiff attempt that stopped before the end
  of its segment — a method that exhausts `ode_max_steps` without ever reaching the minimum step
  size used to return its freeze-padded trajectory with every failure counter at zero. Solver
  statistics distinguish unfinished attempts discarded by the guard from unfinished trajectories
  actually returned to the caller, and the warning's clauses no longer double-count a segment
  that `ode_stiff_abort_after` abandoned.
- **Hidden parameter guards now produce a fit warning (#1099).** A free Theta at
  the implicit `1e-10` / `1e9` cap, or an Omega, Omega-IOV, Sigma, or mixture
  override pinned to an internal packed-space safety limit, emits the typed
  `parameter_at_runaway_guard` warning. FIX'd coordinates remain excluded;
  user-declared Theta bounds continue to use the separate `boundary_estimate`
  warning. Lower hits are identified as collapse toward zero, while upper hits
  are identified as runaway estimates.
- **A closed-form model with IOV and a `[scaling] y = <expr>` readout evaluated the readout's
  individual parameters at `kappa = 0` — predictions and the objective were silently wrong
  (#1079).** The readout is the analogue of NONMEM's `$ERROR` and is evaluated per record, so an
  individual parameter carrying a `kappa` — an additive baseline `BASE = TVBASE * exp(KAPPA_B)`, a
  second analyte, a bounded transform — must take that record's occasion value. On the analytical
  (`pk ...`) engine it took the parameter's *typical* value in every occasion instead: the
  concentration handed to the readout carried the occasion κ, but every parameter the readout
  itself read did not. On the reproduction in the issue the prediction is off by 12–20 %.
  **Anyone who fitted such a model should re-run it** — the reported estimates, OFV, and
  diagnostics were computed from the wrong predictions. The ODE engine was always correct, as was
  the compartment-free (`[structural_model]`-less) path, and a `y = central / V` readout was
  unaffected on every engine (the `conc × V` amount reconstruction and the readout's own `/ V`
  cancel, with or without a κ on `V`) — which is why the natural readout to test could not show
  it. The analytic sensitivities are re-seeded to match, so FOCE/FOCEI/SAEM gradients now
  differentiate the corrected prediction.
- **A third argument to a `[derived]` row aggregate was dropped in silence (#1092).**
  `CMAX = max(IPRED, TIME > 0, 99)` computed `max(IPRED, TIME > 0)` — the aggregate form reads only
  the value and the optional row filter, and never looked at what followed. It is now a parse error
  that names `clamp(x, lo, hi)`, which is what the extra argument usually means. The same mistake
  written inside a larger expression was already rejected.
- **`simulate()` silently removed an arm's residual noise when its `weight = <expr>` column was blank
  or zero (#1083).** `fit()` has rejected a non-positive residual magnitude since #1029, but no
  `simulate()` entry point ran that check — each ran its own subset of the model-vs-data checks and
  this one was in none of them. Because the modifier multiplies the *additive* loading, a zero weight
  did not blow up: it removed that arm's residual variability entirely, and the simulated observation
  came back equal to its own IPRED — finite, plottable, and wrong. The weighted-kappa check had the
  same uneven coverage: present on `simulate_with_options*` but absent from `simulate` /
  `simulate_with_seed` and from the adaptive-dosing path, where `κ/√W` with a zero `W` evaluates to
  zero rather than to infinity, so the arm quietly lost its between-arm variability with no `NaN` to
  trip over. Every simulate entry point now runs one shared list of checks; the two that return a
  bare `Vec` and cannot signal enforce it as a panic, matching the existing IOV precondition.

- **An estimated lagtime whose lagged dose arrival crossed a time-varying covariate change got a
  silently wrong analytic gradient on ODE models (#1060).** The dose lands at a moving boundary
  `t + ALAG`, and the walk injects the resulting jump as a saltation. The segment ending at that
  arrival belongs to the dose record's covariate snapshot and the segment it opens to the next
  record's, but the injection evaluated *both* sides on the dose record's — so whenever those two
  records carried different covariates, the post-arrival velocity, its Jacobian and the cross term
  all used the wrong parameters. Individual predictions were unaffected (the correction is
  derivative-only) — but the FOCEI objective builds its `h` matrix from this same analytic
  `∂f/∂η`, so the **reported OFV was wrong too**: on a crossing dataset the objective missed
  NONMEM 7.6.0 by 7.52 units before the fix and by 5e-6 after it, with every EBE reproduced to
  the printed digits (`nonmem_anchor/tvcov_lag_saltation.ctl`). Against finite differences of the
  production predictor the gradient error reached 300× and the Hessian 23×. Fixed with it: the
  matching defect at a finite-duration **infusion**'s lagged rate-on; concurrently active
  forcings missing from the boundary velocities, which cost a further ~2% of the second order
  once the two sides read different snapshots; and a value-only snapshot comparison that let an
  IOV occasion boundary at the rate-on drop a κ jet, giving `∂²f/∂η_LAG∂κ` the wrong sign.
  Constant-covariate fits, and any fit whose arrivals do not cross a covariate change, are
  bit-identical to before.
- **`TIME` in an `init(...)` expression is now rejected instead of silently reading zero (#994).**
  Both init surfaces — `[odes] init(state) = ...` and the analytical `[initial_conditions]
  init(cmt) = ...` — accepted a bare `TIME` (and `time`), while rejecting `Time`, `T`, `TAFD` and
  `TAD`. An initial condition is evaluated at the time origin, so `TIME` there always read exactly
  `0`: `init(central) = TIME + 50` was bit-identical to `init(central) = 0 + 50`, and
  `init(central) = TIME * SOMETHING` initialised the compartment to zero, validated clean, and
  fitted. The leak was representational, not a scope decision — a bare `TIME` parses to its own AST
  node rather than to a variable, so the undefined-name check could not see it. It now errors,
  pointing at `d/dt(...)` (or `[scaling]` Form C) for a genuinely time-dependent expression.
  `MIXNUM`, the other built-in the check cannot see, stays in scope: it resolves to the subject's
  mixture class, so a class-switched baseline does real work — it is now named in the diagnostic,
  which previously listed neither.
- **Every spelling of every clock in an `[odes] init(...)` now gets the same explanation (#994).**
  `TIME`, `time`, `Time`, `T`, `TAFD` and `TAD` used to split three ways on representation alone —
  accepted, reported as a plain undefined name, or reported as an undefined name — so the message a
  user got depended on the casing they happened to type. All of them now report why a clock cannot
  appear in an initial condition. An expression carrying both a clock and an undefined name reports
  both problems from one parse instead of one per parse.
- **A `[derived]` row aggregate followed by more expression is now an error instead of silently
  dropping the rest (#1030 review).** `CMAX = max(IPRED) * 2` parsed as the aggregate and discarded
  the `* 2`, reporting the unscaled maximum. There is no aggregate form that continues into a larger
  expression, so it now says so rather than returning a wrong number.
- **A zero, negative, or non-finite residual-error magnitude is now rejected (#484 / #1029).** The
  magnitude multiplies a sigma loading, so a zero one — an MBMA arm with no reported standard error,
  say — collapses that observation's variance onto the internal `1e-12` floor and lets a single row
  own the entire objective. Fits like that used to run and report a converged answer; they now stop
  at `E_RUV_MAGNITUDE_NONPOSITIVE`, naming the subject and TIME.
- **A covariate used only by the residual-error magnitude is no longer frozen at the subject's first
  value (#484, found while implementing #1029).** A covariate that a model referenced *only* from an
  `[error_model]` magnitude expression was not counted as model-referenced, so the pre-fit pass that
  drops unused per-record covariate snapshots discarded them for any subject whose only time-varying
  column was that one. Every observation was then scored with the subject's first value of it —
  silently, with no diagnostic, and with the whole point of a time- or record-varying magnitude lost.
  Such covariates now register as referenced, so their per-observation snapshots survive.
- **A sigma declared past the ones a `combined` `[error_model]` names no longer perturbs the FOCE
  score or the `s` / `rsr` standard errors (#1001 review).** `dr_diag_d_log_sigma`'s `Combined` arm
  answered every `sigma_k >= 1` with the *additive* sigma's derivative, where its `additive` and
  `proportional` siblings correctly returned zero. Since the loop runs over the whole flat sigma
  vector, a third, unreferenced sigma — legal, and documented as inert — picked up a non-zero
  `∂R/∂log σ` it has no business having, putting a spurious row and column into
  `subject_nll_pop_grad` and hence into the score cross-product behind `covariance = s` / `rsr`.
  Affected `method = foce` / `gn` / `gn_hybrid` fits of a `combined` error model with an extra
  declared sigma; the reported standard errors were wrong for *every* parameter, silently.
- **`examples/per_route_lag_absorption.ferx` had its two residual components the wrong way round
  (#1001 review).** The file declared `sigma ADD_ERR` before `sigma PROP_ERR` and wrote
  `combined(ADD_ERR, PROP_ERR)`; a single-endpoint `combined` model consumes those positionally, so
  0.02 was the *proportional* coefficient and 0.10 the *additive* SD — the inverse of the file's own
  header, both inline comments, and the `--simulate` parameters it advertises. Both lists were
  transposed together, so the new #1001 check could not see it. Declarations and arguments are now
  in role order. Users who copied this example's error model should swap it back the same way.
- **An adaptive-dosing run now reads a `TIME`-dependent `[scaling]` Form C readout on the same
  clock as `fit()` / `predict()` (#1028 follow-up).** #1028 moved the readout's `TIME` to the raw
  data-file clock (the `$ERROR` convention shared by sdtab, `predict()`/`simulate()` and `[derived]`
  windows) on the static predictors, but the reactive driver and the frozen-schedule replay
  verifier kept feeding it the integrator break the observation was keyed to. For a subject with
  stacked reset occasions — whose data `TIME` restarts while the internal timeline stays monotonic —
  those are different numbers, so the *same record* got one `TIME` under `simulate_adaptive` and
  another under `fit()`, and the replay verifier's bit-equality against the static engine no longer
  held. Both now use the raw clock. Also in the same area: a `T` / `t` declared in `[covariates]` is
  now honoured **case-insensitively** (declaring `T` protects a `t` reference and vice versa —
  previously the case-mismatched pair silently folded to the model clock in `y`, and raised the
  time-in-`obs_scale` error against a legitimately declared column), and the declaration now reaches
  `[adaptive_dosing] observe`, which compiles through the same readout compiler — so a declared `T`
  can no longer be the data column in `[scaling]` and the clock in `observe` for one model. The
  `T`-fold warning for `observe` is emitted at parse time, since the block is compiled at simulate
  time where there is no warnings channel.
- **`TIME` now works in a `[scaling]` Form C readout, and an undefined name in `[scaling]` is no
  longer a silent zero (#1028).** A `y = <expr>` / `y[CMT=N] = <expr>` readout referencing the
  `TIME` built-in parsed fine but was never bound to the observation — the integrator's model-time
  guard is dropped before the readout runs — so `TIME` read `0` at every row and the whole
  time-dependent term vanished. A response-versus-time readout such as
  `y[CMT=1] = EMAX * TIME / (TIME + T50)` therefore fit, converged, and reported plausible
  parameters for a structural model nobody wrote. `TIME` (and the `T` alias `[odes]` also accepts)
  now resolves to each observation's own time on both the ODE and analytical Form C paths, on the
  production predictor and the analytic sensitivity walks alike — and to each decision's time in an
  `[adaptive_dosing] observe` expression, which compiles through the same readout compiler — so the
  dummy `d/dt(clock) = 1`
  workaround is no longer needed (and is better dropped: `clock` starts at the subject's first
  record, not at `t = 0`). Separately, `obs_scale` expressions never registered their covariate
  references as required data columns, and `predict()` ran no covariate check at all, so an
  unresolvable identifier anywhere in `[scaling]` reached the predictor as the covariate map's
  `0.0` default. Both halves of the block now register their references, and `predict()` reports
  `E_MISSING_COVARIATE` for a missing column just as `fit()` and `simulate()` already did.
  **Breaking** in two narrow places: `obs_scale = TIME` (or `= T`) is now a parse error naming Form
  C as the place for a time-dependent readout — the divisor is subject-static, evaluated once at
  `t = 0`, so it could only ever have read `0`; and a `[scaling]` expression referencing an
  *undeclared* data column named `T` now reads the model-time built-in instead, matching `[odes]`,
  where that name has always been reserved. Declaring `T` in `[covariates]` keeps it a data column,
  and whenever the fold does happen ferx warns and names both escapes, so the substitution is never
  silent. `TAFD` / `TAD` are unaffected and remain ordinary covariate references in `[scaling]`;
  when such a column is missing, `E_MISSING_COVARIATE` now explains that the name is an `[odes]`-only
  built-in rather than reading as a plain typo report. The readout's `TIME` is the raw data-file
  clock — the same one sdtab, `predict()`/`simulate()` and `[derived]` windows report, and NONMEM's
  `$ERROR` uses — which differs from the integrator timeline only for datasets with stacked reset
  occasions. A modified-release model whose closed-form fast path applies now declines to the ODE
  path when its readout reads `TIME`, instead of dropping the time term via a state-space linearity
  probe. `predict()`'s new covariate check accepts a name every subject's covariate map carries even
  when the population's `covariate_names` list is empty, so a programmatically built in-memory
  `Population` keeps working.
- **An adaptive-dosing `dv` monitor no longer floors a negative Form C `[scaling]` readout at zero
  (#1039).** The assay floor on the `ObserveMode::Dv` path ("an assay cannot read below zero") was
  written when every monitored readout was a compartment amount or concentration, and was applied
  unconditionally after the residual draw. A Form C `y = <expr>` readout is an arbitrary
  expression — a change from baseline, a difference from a comparator, a z-score, the
  `sqrt(N) * logit(p)` transform — so the *same model* read correctly under `mode = ipred` and
  came back as exactly `0` under `mode = dv` for every negative sample, silently: a controller
  thresholding a change-from-baseline signal saw `0` over precisely the region it was written to
  react to, and dosed accordingly. The floor is now gated on the same predicate as the prediction
  path (#1020), so it applies only to the bare-state readout and to Forms A/B, which keep it. With
  `sigma → 0` a `dv` monitor again reproduces the `ipred` monitor sample for sample, negative
  samples included.
- **SAEM no longer lets a fixed-effect-only theta drift away from the marginal optimum (#1011).** The
  numerical θ/σ M-step assigned NLopt's maximiser outright, re-maximising against a *single* MCMC η
  draw each iteration — `argmax` of one draw rather than the stochastic-approximation average of
  `E[argmax]`, a Monte-Carlo bias that does not decay with iteration count. A log-mu-referenced theta
  was unaffected (its closed-form `log θ += γ·mean(η)` update is already an exact Robbins-Monro
  average), and Ω was already protected by a per-iteration SA cap; the θ channel was the one left
  exposed. The M-step result is now blended in as `θ ← θ + γ_θ·(θ* − θ)` with `γ_θ` capped at 0.03
  during exploration and following the full decaying `γ = 1/(k−k1)` in convergence — the θ-side
  counterpart of the existing Ω cap. On the FREM `iiv_on_ruv` model of #1011, whose absorption
  fraction `TVFRD1` carries no ETA, SAEM moves from **0.039 to 0.290** against a marginal −2logL
  optimum of ≈ 0.29 (NONMEM IMP 0.394, ferx IMP 0.311, IMPMAP 0.318); `TVMAT` 3.020 → 2.686 (NONMEM
  2.680) and σ 0.213 → 0.170 (NONMEM 0.177). Damping applies **only when NLopt is left estimating a theta
  that is not mu-referenced** — the shape the bias was measured on, and the same condition the
  advisory below warns about. A theta that is `FIX`, that is pinned out by the mu-reference shift,
  or that is mu-referenceable at all (so the exact closed-form `log θ += γ·mean(η)` update is
  available to it) stays on the undamped update, and a fit whose every estimated theta is one of
  those is bit-identical to before — `warfarin_saem`, all three of whose thetas are
  log-mu-referenced, is unchanged. Mu-referencing here means the pairing ferx *detects*: a parameter
  whose eta is attached in a form ferx cannot pair with a single theta (an additive
  `X = TVX + ETA_X`, a covariate model that is not log-linear in one theta) counts as un-referenced
  and is damped, which is the intended side to err on — that theta has no closed-form shift either. Mixture models are excluded: a `MIXNUM`-switched
  typical value uses the same M-step but must *separate* from a common start before the class
  assignments settle, and damping that excursion stalls it (a 0.03 cap left `TVCL1 = 1.145` against
  NONMEM's 1.002 on `tests/nonmem/mixture_iv_saem`), so mixtures keep the undamped update. The bias
  is reduced, not removed, so the #1011 advisory still fires — attaching an ETA or holding the
  parameter `FIX` remains the better fix. The cap is exposed as the `mstep_damping` `[fit_options]`
  key (default `0.03`, must be in `(0, 1]`); smaller damps harder, and **`mstep_damping = 1.0`
  disables the damping entirely**, restoring the previous behaviour exactly if a model fitted better
  without it. Setting it on a model it cannot affect warns rather than being silently ignored, and a value
  outside `(0, 1]` reaching `run_saem` from a programmatic caller that bypassed the parser is
  clamped with a warning rather than applied (a negative damping would step theta and sigma away
  from the M-step optimum every iteration; `+∞`, which reads as "no damping", now clamps to the
  `1.0` off value rather than to the tightest damping). `mstep_damping = 1.0` is a *sentinel*, not
  a cap value — the option is discontinuous there, since `0.999` still buys the full convergence
  schedule — and the no-effect warning now names which of the three reasons applies, including the
  `mu_referencing = false` case it previously mis-described as "every theta is mu-referenced".
  Sigma is blended with the same `γ_θ` on the same gate: theta and sigma come out of one joint
  NLopt solve, so damping only theta would leave sigma absorbing the misfit the damping just
  stopped theta from fixing.
- **`[covariate_nn]` inputs were frozen at each subject's baseline value when the
  covariate varied over time.** The network reads its inputs from the same per-event
  covariate map every other consumer uses, but its input names are declared in the block
  rather than in an expression, so nothing registered them as *referenced* covariates.
  The fit pipeline then pruned their trajectories as irrelevant and the network saw one
  constant value per subject for the entire record — silently, with no error or warning,
  so a time-varying covariate simply had no effect on the fit. NN input names are now
  registered alongside `[scaling]` / error-selector / `[initial_conditions]` covariates.
  The same registration also closes a second silent failure: a `[covariate_nn]` input the
  data does not carry — a typo, or `inputs = [TIME]` (a reserved column, not a covariate)
  — used to be zero-filled, degenerating the network to a constant and producing a
  plausible-looking fit that had learned nothing. Such an input is now rejected at fit
  time with `E_MISSING_COVARIATE`, like any other missing covariate.

### Added
- **`[structural_model]` accepts a compartment-free model — the `$PRED` equivalent (#811).** A
  block with no `pk ...` / `ode(...)` line and at least one `NAME = <expr>` line declares its
  prediction directly, with no compartments underneath: write named intermediates above a final
  `y = <expr>`, exactly as in `[scaling]`, and reference thetas, etas, individual parameters,
  `TIME`, and any data column. This is the shape of every model-based meta-analysis structural
  model (a dose-response or time-course regression, not a PK system), which until now had to be
  written as a dummy compartment driven by `d/dt(clock) = 1` with the real equation hidden in a
  `[scaling]` readout. Such a model carries no doses and lays its individual parameters out like an
  ODE model, so it is not limited to the handful of spare parameter slots an analytical readout
  draws from. Blocks that presuppose compartments (`[odes]`, `[initial_conditions]`,
  `[diffusion]`, `[scaling]`) are rejected by name rather than silently ignored. Such a model
  takes the **analytic** `Dual2` gradient on both the inner and outer loops — with no state to
  integrate the sensitivity is a chain rule over the individual-parameter program — including
  under inter-occasion variability, where the equation is evaluated per occasion and the gradient
  seeded on that occasion's `kappa` axes (what makes a between-treatment-arm variance component
  estimable rather than frozen). Finite differences remain only past the axis caps, and are
  reported as such. Anchored against the equivalent NONMEM `$PRED` fit: agreement to ~1e-4
  relative on every estimate and standard error, and to 5 decimal places on the objective
  (`nonmem_anchor/algebraic_emax.*`). See `examples/emax_timecourse.ferx`, and
  `examples/mbma_naproxen.ferx` for the published model-based meta-analysis case study
  (Boucher & Bennetts, *CPT:PSP* 2018;7:288–297) it reproduces to three significant figures.
- **Sample-size-weighted IOV: `weight = <expr>` on a `kappa` declaration (#1031).** The arm-level
  random effect of every longitudinal MBMA — between-treatment-arm variability — is distributed
  `κ_ik ~ N(0, γ²/N_ik)`: a 400-subject arm's mean wanders a quarter as far as a 25-subject arm's.
  ferx could express that only by hand, inside a structural equation
  (`... + ETA_EMAX + KAPPA_EMAX / sqrt(NARM)`), which put a variance-structure decision where
  `/ NARM` instead of `/ sqrt(NARM)` produces a plausible wrong answer rather than an error, and
  where nothing marks the term as weighted. Declare it where the rest of the variance structure is
  declared instead: `kappa KAPPA_EMAX ~ 2.0 (sd) weight = NARM`, and write the structural expression
  as `... + ETA_EMAX + KAPPA_EMAX`. The engine applies the scaling by rewriting the kappa as
  `KAPPA / sqrt(W)`, which is exactly the hand-written form — so the objective, the analytic
  sensitivities, the EBEs and every estimator that supports IOV (FOCE, FOCEI, Laplace/AGQ, SAEM)
  are unchanged, and the reported Ω_IOV stays the **unweighted** γ² a published analysis quotes.
  The fit printout adds the number a reader actually needs next to it — the effective SD at the
  median arm, `γ/√N` — and `FitResult` carries `kappa_weights` / `kappa_weight_typical`. The weight
  may reference covariates, `FIX`ed thetas and `TIME` but neither a random effect nor an estimated
  theta (both would make the divisor move underneath the up-front positivity check); a weighted
  kappa may be referenced only in `[individual_parameters]`, and must be referenced somewhere in it
  (elsewhere it would read as the unweighted κ, and nowhere would report a scaling that was never
  applied); `block_kappa` cannot carry one. A weight that is zero, negative or non-finite at any
  record — a blank arm-size cell — is rejected up front by both `fit()` and `simulate()` with
  `E_KAPPA_WEIGHT_NONPOSITIVE` instead of dividing an individual parameter by zero mid-run, and one
  that moves *within* an occasion warns.
- **`[scaling]` accepts named intermediates, expressions can be split across lines, and `min`/`max`
  take two arguments (#1030).** Three restrictions that individually looked defensible combined to
  make a standard bounded-endpoint readout unmaintainable: a model-based meta-analysis logit-Emax
  readout with the published `[0.01, 0.99]` clamp came out as one ~200-character line with the same
  sub-expression written four times, because there was nowhere to name it. Now: (1) any `[scaling]`
  line whose key is not `obs_scale` / `y` declares a **named intermediate** usable by the entries
  below it, following the same define-above-use-below rule as `[individual_parameters]` —
  intermediates are inlined, so covariates reached only through one are still required data columns
  and the dose-attribute double-use rejection still sees them; (2) a long expression may be
  **continued across lines** by starting the continued line with an operator or ending the previous
  one with it, in `[individual_parameters]`, `[odes]`, `[scaling]`, `[derived]`, and
  `[initial_conditions]`; (3) **`min(a, b)`
  and `max(a, b)`** are available everywhere the DSL parses expressions, desugaring to the inline
  conditional so they differentiate and compile exactly like the hand-written `if (a >= b) a else b`
  — each argument appears twice in the desugared tree, so clamp a named intermediate rather than a
  long expression. In `[derived]`, where `min`/`max` also name the row aggregate, the two are told
  apart by the second argument: a comparison is a row filter, anything else is the numeric clamp.
  The three-line readout now reads the way the published Mlxtran does. Two smaller diagnostics come
  with it: `max(a, b)` used to report `Missing closing parenthesis for function max`, which sent the
  reader after a bracket bug that did not exist — arity errors now say so — and a `[scaling]` key
  that is neither `obs_scale`/`y` nor read by any entry is rejected, so a misspelt `obs_scal = V`
  still fails loudly instead of silently disabling scaling.
- **First-class residual weighting: `weight = <expr>` on the error model (#1029).** Meta-analysis
  rows are trial-arm summaries that differ in precision, and inverse-variance weighting is what makes
  an MBMA an MBMA. Until now it had to be hand-built in three places that must agree — divide `DV` in
  R, divide the prediction again in `[scaling]`, and pin `sigma` to 1 in `[parameters]` — with the
  side effect that every diagnostic ferx ships (`sdtab`, VPC, obs-vs-pred, residual-vs-time) came out
  in weighted units nobody can read. Declare it once instead:
  `DV ~ additive(ADD_ERR) weight = WPSE`. `DV` stays on the natural scale in the data, and `PRED`,
  `IPRED`, `CWRES`, `sdtab` and the VPC are all reported on the natural scale with no
  back-transformation. The modifier means "score as if `DV` and the prediction were both divided by
  the weight", so the additive loading picks up a factor `W` and the proportional loading is
  untouched (a common scale factor cancels out of a constant-CV error) — `weight =` on a purely
  proportional model is therefore rejected rather than silently doing nothing. Whether `sigma` is
  fixed stays your call, because both conventions are real: fix it to 1 for a continuous endpoint
  with a known SE, estimate it for a weighted logit of a responder proportion (σ absorbs the
  `p(1-p)` factor). One caveat when reconciling with a published run: ferx scores on the natural
  scale, so its OFV differs from the hand-built one by the change-of-variable constant
  `Σ 2·ln(weight)` — data-only, identical across models on the same rows, and cancelling out of every
  ΔOFV, LRT and AIC comparison. Not yet supported with per-CMT error models, inside a
  covariate-selected `if/else`.
- **Residual-error magnitudes now work under every estimator (#484 / #1029).** A `weight = …`
  modifier — and any `[error_model]` sigma written as an expression of TIME / covariates / thetas —
  used to be rejected up front for anything but `foce` and `focei`, because SAEM's M-step, the
  Gauss-Newton BHHH gradients, the importance-sampling likelihood and proposal, and the IOV
  individual likelihood (SAEM's E-step, the Bayes MH target) all read the residual variance through
  call sites that never applied the per-observation multiplier. They all thread it now, so FOCE,
  FOCEI, Laplace/AGQ, SAEM, IMP, IMPMAP, GN, GN-hybrid and Bayes score the same weighted likelihood
  and `method =` chooses the algorithm rather than the model. Gradients keep up: SAEM's M-step
  carries the magnitude's *direct* θ channel (a θ that appears only in the error model used to read
  a gradient of exactly zero there), and the Gauss-Newton closed forms — which have no such term —
  route a θ-dependent magnitude to their magnitude-aware finite-difference fallback. A weight built
  from covariates and TIME alone, which is every #1029 model, has no direct-θ channel and keeps the
  fast analytic path everywhere.
- **SAEM now warns when an estimated theta carries no ETA at all.** A fixed-effect-only theta is not
  mu-referenced, so it never gets the γ-damped closed-form `log θ += γ·mean(η)` update and is moved
  only by the η-frozen numerical M-step — which re-maximises against a *single* MCMC η draw with no
  stochastic-approximation damping and can drift far from the marginal optimum, dragging correlated
  typical values with it. SAEM previously said nothing (its existing advisory only covers a theta
  whose ETA could not be mu-referenced); IMP/IMPMAP have warned about this since #406. On a FREM
  `iiv_on_ruv` model whose absorption fraction `TVFRD1` has no ETA, SAEM drove it to **0.039** while
  IMP (0.311), IMPMAP (0.318) and NONMEM IMP (0.394) agree, with `TVV` +6% and `TVMAT` +9% carried
  along; restarting SAEM *at* the IMPMAP solution still walked it down to 0.065. Adding
  `FRD1 = TVFRD1*exp(ETA_FRD1)` (ω² = 0.01) recovers 0.313, and holding `TVFRD1` `FIX` puts every
  other theta within 3% of NONMEM. The advisory names the remedy that fits the parameter — put a
  typical value in a mu-referenceable form, or hold a covariate coefficient / allometric exponent /
  structural constant `FIX` and cross-check against FOCEI/IMPMAP — and stays quiet about thetas for
  which "has no ETA" would be false: a mixture's mixing coefficients (never moved by the numerical
  M-step, and forbidden from depending on an eta at all), and `MIXNUM`-switched class typical values
  or identity-scale-dropped log-mu-references, which carry an eta and already get their own message.
  See
  [SAEM: non-mu-referenced parameters](https://ferx-nlme.github.io/ferx-core/estimation/saem.html).
- **Class-aware mu-referencing for mixture models (#996).** A `MIXNUM`-switched typical value written
  as `CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)` is now recognised at parse
  time and resolved to one anchor theta per class (any number of classes; a trailing `else` covers
  every class the chain does not name). Estimating **IMP / IMPMAP** uses it to apply the EM
  responsibility-weighted shift `log θ_k += (Σ_i PMIX_ic · η̄_ic) / (Σ_i PMIX_ic)` and pins those θ out
  of the importance-weighted M-step. This closes the accuracy gap that made IMP the noisiest mixture
  estimator: on the two-class anchor IMPMAP now recovers `TVCL1 = 0.952`, `TVCL2 = 2.821` against
  `0.949` / `2.819` from NONMEM IMPMAP with `MU_1` assigned inside the `MIXNUM` branch
  (`tests/nonmem/mixture_iv_impmap_mu.ctl`, NM 7.6.0) — under 0.5 % on every estimated parameter,
  versus `2.60` for `TVCL2` before. **SAEM** gains the closed-form shift for class-*shared* typical
  values (`V = TVV * exp(ETA_V)` inside a mixture), which previously fell back to the numerical
  M-step; its class-*switched* θ deliberately stay numerical, because SAEM's hard per-subject class
  draw makes the per-class η mean a biased classification-EM statistic (measured: `TVCL2` 2.75 → 3.02,
  OFV 302.2 → 305.0). `mu_referencing = false` restores the pre-#996 behaviour for both estimators.
  A `MIXNUM`-switched expression that cannot be resolved to a class-aware anchor now warns instead of
  silently dropping to the numerical M-step.
- **Models with no random effects — fixed-effects-only / naive-pooled fits (#989).** A continuous
  (residual-error) model may now omit every `omega` declaration; previously this was rejected at
  parse time with `No omega parameters defined`, and the 0×0 Ω path was reachable only from an
  `[event_model]` / `[binary_model]` endpoint. With `n_eta = 0` there is no inner empirical-Bayes
  problem and no `log|Ω|` term, so FOCE/FOCEI collapse to the plain maximum-likelihood objective.
  An `[error_model]` and its `sigma` are still required for a continuous endpoint — only Ω becomes
  optional. Anchored against a NONMEM `$OMEGA 0 FIX` population fit on `data/one_cpt_iv.csv`
  (OFV −269.637010 vs −269.63700440; θ and σ to 4+ significant figures; SEs match under
  `covariance_method = rsr`, which is NONMEM's `$COVARIANCE` default). `method = saem` is
  rejected at `n_eta = 0` (`E_SAEM_NO_RANDOM_EFFECTS`); `gn` runs but is not recommended, since
  without an inner loop it is unusably sensitive to the `sigma` start — use `gn_hybrid` or
  `focei`. An unresolved `ETA…` / `KAPPA…` identifier is now reported as an undeclared random
  effect (`E_ETA_NOT_DECLARED`, or `W_ETA_NOT_DECLARED` without `--data`) rather than a missing
  covariate, so deleting an `omega` line and leaving its `exp(ETA_…)` term behind fails loudly.
- **Fixed-effects-only fits no longer print an empty `OMEGA Estimates` section (#989).** At
  `n_eta = 0` the header used to be written above an empty body, which reads as an estimation
  that failed rather than one that was never requested.
- **IMP / IMPMAP estimation and objective evaluation for mixture models (#985).** Importance sampling
  now both **estimates** a `[mixture]` model (`method = imp` / `impmap`) and **evaluates** its
  class-marginal likelihood `−2 Σ log Σ_k p_ik L_ik` (`imp_eval_only`, NONMEM `METHOD=IMP EONLY=1`).
  Estimation runs a class-partitioned MCEM: per subject and class it importance-samples η under the
  `MIXNUM` guard, forms the responsibilities `PMIX_ik ∝ p_ik L_ik`, and runs responsibility-weighted
  M-steps for the mixing coefficients, the class-shared Ω, and the class-switched typical values.
  Objective evaluation combines the per-class IS marginals by log-sum-exp and matches NONMEM to well
  under a Monte-Carlo SE (`−2 log L = 300.82 ± 0.06` vs NONMEM `300.87` on the two-class anchor).
  Ω/σ are class-shared (per-class overrides rejected); IOV/FREM/SDE are not supported under the
  mixture IMP paths, and a theta shared between the mixing expression and a structural typical value
  is rejected (as under SAEM). The per-class draws use the same ISCALE pilot search and `impmap_mceta`
  multi-start MAP as the single-population MCEM; objective evaluation reports a real per-subject ESS
  (the worst ESS among the classes a subject loads on), so `imp_low_ess_threshold` and the
  proposal-collapse warning apply to mixtures. `impmap_trace` and `impmap_auto` / `imp_auto` are not
  wired for mixtures and now warn instead of being silently ignored.
- **Bayesian estimation for mixture models (#985).** `[mixture]` models can now be fit with
  `method = bayes`. The latent class is Rao-Blackwellised — marginalised out of every Gibbs block
  rather than sampled — so each subject's likelihood contribution is the K-class marginal
  `−log Σ_k p_ik·exp(−nll_ik)` and the sampler keeps a smooth continuous target. The η block samples
  against the class-marginal posterior (HMC is disabled for mixtures; its analytic gradient is
  single-class), the (θ,σ) block samples the mixing thetas (constant or covariate logit) with no extra
  machinery, and Ω is drawn from its class-shared conjugate conditional. Reported OFV, per-subject
  `MIXEST`/`PMIX`, and EBEs are the K-fold marginal values at the posterior mean. Per-class Ω/σ
  overrides are rejected (Ω/σ are class-shared). Validated by recovering the mixture MLE under diffuse
  priors (a direct NONMEM `METHOD=BAYES` reference is impractical — its sampler aborts on `$MIX` with
  FIXed Ω/Σ). Inter-occasion variability is supported: the κ block samples against the same
  class-marginal target. A run that asks for HMC (`saem_n_leapfrog > 0`) on a mixture now warns that
  the Metropolis-Hastings η kernel was used instead, and a high R̂ on a mixture warns about label
  switching, which would make the reported posterior mean an average across class labels.
- **SAEM estimation for mixture models (#985).** `[mixture]` models can now be fit with
  `method = saem`, not only FOCE/FOCEI. The E-step samples the latent class per subject (from the
  current posterior `PMIX_i`) and runs the η-MCMC within the drawn class; the M-step estimates the
  class-switched typical values (each from its own class members), the per-class Ω overrides, and the
  mixing coefficients (constant *or* covariate-dependent logit mixing) from the sampled class
  frequencies. Reported OFV, per-subject `MIXEST`/`PMIX`, and standard errors come from the K-fold
  mixture marginal, matching FOCEI. Cross-checked against NONMEM `METHOD=SAEM` (estimates agree to
  ≤ 3 %). A mixing theta marked `FIX` is honoured; a theta shared between the mixing expression and a
  structural typical value is rejected with a clear error. Per-class σ overrides (`sigma(k)`) are held
  at their initial values under SAEM (with a warning, and reported with SE 0 like other fixed
  parameters) — route those to FOCEI.
- **Inter-occasion variability under a mixture (#985).** A `[mixture]` model may now also carry a
  `kappa` (inter-occasion variability) term — the two features compose, where before `fit()` rejected
  the combination. Each class's per-subject inner solve estimates the per-occasion κ̂ under that
  class's typical values, and the FOCE/FOCEI marginal `L_ik` is the κ-augmented occasion likelihood,
  so the usual `L_i = Σ_k p_ik · L_ik` mixture is formed over IOV-aware class likelihoods. The IOV Ω
  is shared across classes (matching NONMEM `$OMEGA BLOCK(1) … SAME`). The analytic outer gradient
  does not yet emit κ-slot derivatives for a mixture, so an IOV mixture optimises against a
  finite-difference outer gradient; the covariance step runs as usual on the K-fold objective.
  Per-subject diagnostics are IOV-aware: the winning (`MIXEST`) class's per-occasion κ̂ flow into the
  sdtab IPRED/IWRES/CWRES and per-subject OFV, into κ shrinkage, and into the `.fitrx` `ebe_kappas`
  export. Cross-checked against NONMEM 7.5.1 (`tests/nonmem/mixture_iv_iov.ctl`): OFV, class
  clearances, `V`, mixing fraction, and all 30 `MIXEST` classifications agree.
- **Mixture models — `$MIXTURE`-style discrete latent subpopulations (#977).** Model files can
  declare a `[mixture]` block giving the number of classes (`nsub`), the per-class mixing rule
  (`logit(k) = …` softmax, or `p(k) = …` direct probability, over theta + covariates), and optional per-class Ω/Σ overrides (`omega(k)` /
  `sigma(k)`); the reserved read-only `MIXNUM` index (1..=K) selects class-specific typical values
  inside `[individual_parameters]`. `fit()` estimates such models by **FOCE / FOCEI** — each
  subject's marginal is the covariate-weighted mixture `L_i = Σ_k p_ik · L_ik` and the objective is
  the numerically stable log-sum-exp `−2 Σ_i log Σ_k p_ik exp(−nll_ik)`, with a separate empirical
  Bayes solve per (subject × class). The mixing-logit coefficients are ordinary thetas and the
  per-class Ω/Σ are estimated jointly. Estimation defaults to the derivative-free (BOBYQA) outer
  optimizer, but an analytic posterior-weighted outer gradient is available, so a user-selected NLopt
  gradient optimizer (SLSQP / L-BFGS / MMA) is honoured — with an automatic finite-difference fallback
  for models outside analytic scope (e.g. `MIXNUM`-branched typical values). Other estimators
  (SAEM/IMP/Bayes) are not yet supported and error clearly (inter-occasion variability is supported
  since #985). The parser rejects `nsub < 2`, `MIXNUM`
  assignment, `MIXNUM` outside a mixture model, eta-dependent mixing expressions, missing class
  coverage, `omega(k)` on a block base, and overrides of the base class. The `sdtab` output gains
  per-subject `MIXEST` (most-probable class, 1-based like NONMEM) and `PMIX_1..PMIX_K` (posterior
  class-membership probabilities `PMIX_ik ∝ p_ik·exp(−nll_ik)`) columns for a mixture fit.
- **Standard errors / covariance for mixture fits (#983).** The covariance step now runs for
  mixture models: its finite-difference Hessian is built on the K-fold mixture objective
  (`−2 Σ_i log Σ_k p_ik exp(−nll_ik)`), not the single-population marginal, so a mixture fit reports
  SEs, RSEs, and a covariance matrix like any other fit. The mixing-fraction SE is reported on the
  scale the mixing form is parameterized in — the coefficients of a `logit(k) = …` form on the logit
  scale, a `p(k) = …` probability directly — since those coefficients are ordinary thetas. The
  covariance-matrix labels now name the per-class Ω/Σ override coordinates (`omega[<eta>_MIX{k}]` /
  `sigma[<sigma>_MIX{k}]`) instead of a generic `packed[N]`.
- **A missing `DV` no longer empties a simulation.** Simulating from a design — dosing plus
  sampling times, with `DV = .` because the values are what the run is about to produce — used
  to return zero rows: every `EVID=0` row with a missing `DV` was skipped as a forgotten
  `MDV=1` (#258), which is the right reading only when the `DV` is an input. The new
  `read_population_for_simulation()` reads such a row as a **design point** instead, so the
  natural template simulates as written and no placeholder number is needed in the column
  about to be overwritten. Fitting is unchanged, and `MDV=1` still excludes a record on both
  paths (#957). Kept design rows are reported as `W_DESIGN_DV`, the simulation-side counterpart
  of `W_MISSING_DV`, so a simulation run off an *observed* dataset makes its extra rows visible
  rather than silently carrying more rows than a fit of the same data.

### Changed
- **API (breaking for struct-literal construction): `MixtureSpec` gained a `mu_refs` field and is
  now `#[non_exhaustive]` (#996).** The new field carries the class-aware mu-references described
  above. Downstream Rust code that built a `MixtureSpec` with a struct literal will no longer
  compile — read its fields, or obtain one from the parser, instead. `#[non_exhaustive]` makes the
  next field addition non-breaking. No effect on `.ferx` models, the CLI, or the R wrapper, neither
  of which constructs the struct.

### Performance
- **The ODE inter-occasion-variability (IOV) analytic sensitivity path now compiles from a
  bucketed set of dual widths instead of one specialisation per stacked axis count**, cutting
  the crate's generated LLVM IR by 43 % (17.4 M → 9.8 M lines) and the lib's `-Ztime-passes`
  total by ~2.95× locally (376.7 s → 127.5 s) — the direct attack on the compile-bound CI wall clock tracked in
  #969. Gradients are unchanged: a stacked width that is not a bucket boundary is padded with
  zero lanes and returns bit-identical `∂f/∂η` / `∂f/∂θ` to the exact-width walk. Widths up to
  24 axes — the ordinary IOV model — are still specialised exactly, so they pay no runtime
  cost at all; wider subjects run up to ~1.5× more work in the outer walk for the padded
  lanes, and the 96-axis cap (past which a subject falls back to finite differences) is
  unchanged (#971).
- **Deep compartment models with IOV no longer fall back to finite-difference η-gradients.**
  `[covariate_nn]` weights are auto-generated thetas, so `model.n_theta` (declared +
  weights) can never equal the compiled `[individual_parameters]` program's θ-axis count
  (declared only — the program can only reference θ by name). The analytic IOV predicate
  required them equal, so **every** subject of every DCM+IOV fit took the slow path: 60 of
  60 on a busulfan-shaped model, correct but roughly twice the necessary runtime. That
  clause is load-bearing for the outer gradient, which seeds θ axes, but not for the inner
  η-gradient, whose walk documents that it uses no θ axes and reads only the η block. The
  two predicates are now separate, and the inner loop's own gates — the ones that decide the
  route and the ones that report it — read the η-only one, so FOCE/FOCEI, AGQ and the
  reported `gradient_method_inner` all agree on the route taken. Measured on that fit:
  336 s → **162 s** (2.07×), with `n_fd_subjects` 60 → 0 and the objective unchanged at
  1060.99 after an identical 14 625 iterations. Models that already had the analytic path
  keep it byte-for-byte — the η-only route is taken only where the full one was unavailable.

- **Analytic weight gradients for `[covariate_nn]` models** — the fixed-η θ gradient that
  SAEM's M-step, IMP and VI share used one perturbed model solve per θ, and on a deep
  compartment model the network's weights *are* θ. It now takes the network's weights
  analytically: the NN reaches the likelihood only through its output layer, so
  `∂NLL/∂w = Σₖ (∂NLL/∂zₖ)·(∂zₖ/∂w)`, where the second factor is exact backpropagation and
  only the `n_outputs` values of the first need the model solved. Those are obtained from
  the output-layer biases, which move one output's pre-activation and nothing else. On the
  reference 141-weight DCM (2 → 8 → 8 → 5) that is 10 solves per subject per draw instead
  of 141. Because the few remaining differences are now shared by every weight, they are
  taken *centrally* rather than forward, and agreement with a central finite difference of
  the objective improves from ~2e-5 to ~1e-9 relative — the weight gradients stop being the
  least accurate part of a DCM fit. Wall-clock on that model is ~1.4× (42 s → 30 s at a
  fixed 1500 VI iterations, estimates unchanged); the finite-difference loop was roughly
  30% of VI's runtime there, so the 14× cut in solves does not carry through to the total.
  Models whose NN inputs vary within a subject keep the per-θ loop — a single output vector
  no longer mediates every observation — and models with no `[covariate_nn]` block are
  untouched.

### Added
- **`vi_sigma_update`** — `σ` can now be replaced each iteration by the **exact** ELBO maximizer
  rather than stepped by Adam, the same treatment `Ω` already gets from
  `vi_omega_update = closed_form`. For a single proportional or additive `σ`, stationarity of the
  data term gives `σ*² = (1/n_obs) · Σ E_q[(y − f(η))²/f(η)²]`, and the same identity expresses
  that sum in terms of the `σ` gradient the ELBO already computes, so the update costs nothing.
  `closed_form` is the default; `adam` restores the previous behaviour. Error structures with no
  scalar stationary point (combined error, several `σ`, per-endpoint or covariate-selected error,
  correlated residuals, M3 BLOQ, IIV-on-RUV, FREM, or a FIXed `σ`) fall back to Adam with the
  reason recorded in `FitResult$warnings`.

  **What this does and does not buy.** It makes `σ` exact given `q` and removes it from the
  stochastic trajectory, so `σ` no longer carries its own `vi_lr` sensitivity. It is *not* what
  closes the warfarin gap below — at `vi_mc_samples = 128` the two routes agree to 0.3 OFV, and
  Adam is marginally ahead. Adopted for exactness and consistency with `Ω`, not for a measured
  improvement at converged settings.

### Changed
- **The documented size of VI's posterior-variance understatement is now measured rather than
  cited.** `docs/estimation/vi.qmd` presented "on the order of 20–25%" as a general figure; it is
  the number for deep compartment models. Measured against per-subject NUTS at a fixed population
  estimate, ferx's variational posterior matches the exact posterior to **0.2%** in variance on
  warfarin (means to `2×10⁻⁵`), and to 4% when thinned to two observations a subject. The Laplace
  covariance matches NUTS to 0.1% on the same fits, so the true posterior is Gaussian there and a
  Gaussian `q` has nothing to get wrong. The page now reports the measurements and scopes the
  citation to the regime it came from.

### Fixed
- **`method = vi` no longer reports `converged: true` after ~500 iterations when every population
  parameter is FIXed.** Fitting `q` alone at a pinned `(θ, Ω, σ)` — how you read per-subject
  posteriors at a known estimate — is a legitimate request, but VI's parameter-stability
  convergence test was comparing a vector that cannot move against itself, reporting "settled" at
  its first opportunity and stopping the run. Since either convergence criterion is sufficient,
  that overrode the objective test, which had correctly reported "still moving". On warfarin with
  everything FIXed at a known-good estimate the fit stopped after 500 iterations with
  `elbo_tightness_ratio: 78` (implausible above 25) and `−2·ELBO = +2026`, on a model that reaches
  `−283` once `φ` converges; it now runs 6250 iterations to `−282.6` with a ratio of `1.4`. When no
  population coordinate is free, convergence is judged on the objective alone and a warning says
  so.

### Changed
- **`method = vi` needs more Monte-Carlo draws than the default provides.** On
  `data/warfarin.csv` (~1% proportional residual), `vi_mc_samples = 8` (the default) returns
  `σ ≈ 0.0138` against `0.010565` from both AGQ (`n_agq = 9`) and FOCEI — 31% high — with the OFV
  9.4–9.6 units short of AGQ's `−285.977`. Because per-subject posterior width scales with `σ²`,
  every variational covariance reported at that point is ~1.7× too wide. The cause is the
  convergence rule meeting its own noise floor: it stops when the ELBO's drift is no longer
  distinguishable from Monte-Carlo noise, and at 8 draws that happens while real drift remains —
  so the fit stops short and `σ`, the slowest-moving coordinate, is left furthest from its
  optimum. Raising the draw count resolves it (`32` → −284.55, `128` → −285.61 against AGQ's
  `−285.977`), and lowering `vi_lr` does too. Starting from a fitted FOCEI point does **not**
  help. Documented in `docs/estimation/vi.qmd`. The default has since moved from 8 to 32 for this
  reason (see the entry above). **If a VI fit lands well short of a FOCEI or AGQ fit of the same
  data, raise `vi_mc_samples` first.**

### Added
- **The per-subject variational posterior is now written to the fit YAML.** A `method = vi` fit
  emits an `eta_posterior` block under `vi:`, keyed by subject ID, carrying each subject's
  variational mean and its **full** covariance (plus per-occasion `kappa` means under IOV).
  Previously `FitResult$vi$eta_means` / `eta_covs` were reachable only from the Rust and R APIs,
  so the CLI could not support any per-subject comparison. Note the covariance is the
  *variational* one, which understates the true posterior variance — see
  `docs/estimation/vi.qmd`.
- **`agq_eval_only`** — a `laplace` stage can now *evaluate* the adaptive-Gauss-Hermite
  marginal likelihood at the parameters it is handed instead of estimating, reporting it as
  `ofv` and leaving `θ`/`Ω`/`σ` untouched. It must be the final stage. This is the
  deterministic counterpart to `imp_eval_only`: `method = vi, laplace` with
  `agq_eval_only = true` turns a VI fit's ELBO — which is only a lower bound — into a real
  `−2 log L` that carries no Monte-Carlo error, so two runs agree bit for bit. See
  `docs/model-file/fit-options.qmd`.


### Fixed
- **`optimizer = trust_region` no longer reports `Converged: YES` when it merely ran out of
  iterations (#1000).** The underlying solver has no convergence criterion of its own, so
  exhausting `maxiter` was its only way to stop — and every such run was labelled converged, with
  standard errors computed at a non-stationary point (on the `one_cpt_iv_pooled` zero-Ω anchor,
  8 000–11 000 OFV units short of the optimum). The trust region now stops when it can no longer
  make progress — no objective improvement beyond `outer_ftol` and no parameter movement beyond
  `outer_xtol` for 20 consecutive iterations — and a run that instead hits `maxiter` reports
  `Converged: NO` with a warning naming the budget and the gradient norm it stopped at. Fits that
  do settle now also stop as soon as they settle instead of grinding out the remaining budget (the
  warfarin fit returns at iteration 59), and the trust-region path now reports `final_gradient` and
  the number of outer iterations it actually ran (it reported `Iterations: 0` before). `outer_ftol`
  and `outer_xtol`, previously `bobyqa`-only, now also govern `trust_region`; `outer_ftol` resolves
  the same way for both, so a pure-TTE fit gets the #469 `1e-8` tightening rather than a looser
  hardcoded value. Two verdicts are newly explicit: a `maxiter` below 21 cannot demonstrate settling
  at all (the warning says so instead of blaming the fit), and a run that rejects every step from
  the first iteration is reported against its *starting values* and bails out immediately rather
  than spending the whole budget frozen. Note for multi-start users: `n_starts` prefers a converged
  candidate over a non-converged one, and that key was inert while every `trust_region` run claimed
  convergence — a settled start can now be selected over a better-OFV start that ran out of budget.
- **`optimizer = trust_region` under `method = focei` no longer descends on a truncated gradient.**
  The trust region used the fixed-η̂ score `2·Σ gᵢ`, which drops the `log|H̃|` EBE-response term
  `tᵢ` (#274/#289) that the Gauss-Newton path already adds — the omission a gradient optimizer
  stalls above the minimum on. It now uses the same `2·Σ (gᵢ + tᵢ)` marginal gradient, pinned
  against central finite differences of the objective in a unit test. FOCE and additive-error fits
  are unchanged bit-for-bit (`tᵢ` is identically zero there); on warfarin the FOCEI fit now
  reaches OFV −286.0042, the same value `nlopt_lbfgs` converges to, instead of settling 4.1 units
  above it at −281.88. `final_gradient` is now the marginal gradient rather than a quantity that
  stays large at the optimum.
- **`optimizer = trust_region` is now rejected on a quadrature stage (`E_OPTIMIZER_AGQ`) instead
  of silently fitting the wrong objective.** `method = laplace` (any `n_agq`) and `method = focei`
  with `n_agq > 1` minimise the adaptive-quadrature marginal, and every optimizer scores it — but
  only the NLopt and BFGS paths route the matching gradient (`agq_population_gradient`). The
  trust region has its own gradient with no quadrature branch, so it descended on the
  FOCE/Laplace closed form while reporting quadrature OFVs: a fit that converged, smoothly, to
  the FOCE optimum with no warning. `ferx check` and `fit()` now both reject the combination and
  name the optimizers that do support it. Full quadrature support in the trust region needs a
  BHHH Hessian built from per-subject quadrature scores and is tracked in #1047.
- **A transit / inverse-Gaussian absorption model no longer panics mid-fit when its ODE fallback
  cannot be built (#1008).** The analytic `one_cpt_transit` / `two_cpt_transit` / `one_cpt_ig` /
  `two_cpt_ig` closed forms carry a synthesized ODE twin that serves the subjects the closed form
  cannot (time-varying covariates, a `TIME`-dependent parameter, IOV, steady-state or infusion
  doses, the flip-flop regime). The twin is an ODE model while the model you wrote is analytical,
  so parse checks that apply only to ODE models run on the twin and not on your model — and a
  model ferx accepted could produce a twin ferx rejected. That surfaced as an internal panic
  during `fit()` / `predict()`, the first time a subject actually needed the fallback. The twin is
  now built at parse time: if it cannot be built the model simply stays closed-form and a
  `W_ABSORPTION_TWIN_DECLINED` warning (`absorption_twin_declined`) reports the reason, and any
  subject that needs the fallback is rejected up front with an explicit message instead of
  crashing. That rejection quotes the reason too — previously the twin-less messages described
  only the *desugar's* scope limits ("an unrecognised closed form", "outside the automatic
  ODE-equivalent rewrite"), which is the wrong cause for a twin that was built and rejected, and
  the actionable reason never reached `fit()`'s `Err` or `predict()`'s panic. A live example: an
  individual parameter named `CENTRAL` (or `PERIPH`), which collides with the twin's own state
  names.
- **A negative Form C `[scaling]` prediction is no longer silently clamped to zero on ODE models
  (#1020).** The ODE predictor applied its negative-prediction guard to the *final* prediction
  vector — after the `y = <expr>` / `y[CMT=N] = <expr>` readout had been evaluated. That guard is a
  statement about a compartment amount (which cannot go below zero, so a negative value is solver
  overshoot), but a Form C readout is an arbitrary user expression that is often legitimately
  signed: a change from baseline, a difference from a comparator, a z-score, or the
  `sqrt(N) * logit(p)` transform used in model-based meta-analysis of a bounded endpoint, which is
  negative for every arm below 50%. Every such prediction came back as exactly `0`, with no warning
  and nothing in the fit output to show it — the fit converged with the residual σ inflated to
  absorb the mismatch and the between-subject variance collapsed. The clamp now applies only to the
  default bare-state readout (`obs_cmt`, and the analytical PK concentration), matching the
  analytical Form C path, which never clamped its readout. `NaN` is still never clamped on either
  path, so a bad scale or a missing per-CMT entry keeps surfacing as a `NaN` objective. Applies to
  every ODE driver (dense, dense-with-states, event-driven, adaptive-dosing replay) and to the
  analytic `Dual2`/`Dual1` sensitivity walks, whose clamp is gated identically so the analytic
  gradient still matches finite differences of the predictor.
- **SIR no longer fails with "All SIR samples had invalid weights" on a rank-deficient covariance
  (#1021).** A parameter direction the data do not identify comes back from the covariance step with
  a variance around `1 / eigenvalue floor` — thousands of standard deviations in packed log-space —
  so every proposal draw landed outside the parameter bounds and was rejected. Each proposal
  direction is now capped so ±2 standard deviations stay inside the room between the estimate and
  its nearer packed bound, near-null
  directions (a likelihood ridge left over after `FIX`ed parameters are excluded) are floored rather
  than fatal, and both cases are reported as `SIR:` warnings naming the parameters involved. When
  every sample *is* still rejected, the error now reports the rejection tally, the coordinates whose
  bounds were hit, and the proposal's rank deficiency instead of the bare message. This is the
  common model-based meta-analysis case, where fixing the residual variance is the weighting scheme
  and cannot be dropped.
- **Simulating an IOV (`kappa`) model with parameters that carry no IOV covariance now reports an
  error instead of panicking (#1019).** `simulate()` draws one κ per occasion from `omega_iov`; a
  caller that rebuilds `ModelParameters` from a fit and drops that block (the R
  `ferx_simulate(..., fit = f)` bridge did) hit an `expect()` deep in the row emitter, which crossed
  the FFI boundary as a process panic. `simulate_with_options`/`_diag` now return a clean `Err`
  naming the missing `omega_iov` and the fix; the `Vec`-returning `simulate`/`simulate_with_seed`
  fail loud with the same message rather than emitting rows with no inter-occasion variability.
- **Log-mu-referenced θ with a negative lower bound no longer takes the wrong closed-form update
  (#996).** Such a θ is packed on the identity scale, so the SAEM/IMP `log θ += mean(η)` shift was not
  its EM optimum — it applied `θ += mean(η)` where the closed form means `θ *= exp(mean(η))`. It is
  now excluded from the closed-form channel and estimated by the numerical / weighted M-step instead,
  with a warning naming it, on the mixture *and* the single-population IMP/IMPMAP path.
- **A collapsed mixture class no longer freezes its typical values (#996).** When no subject carried
  any responsibility for a class, its class-aware mu-ref shift was undefined but the θ stayed pinned
  out of the weighted M-step, so it could never move again for the rest of the fit. Those θ are now
  left free for that iteration and the fit warns that the class received no responsibility mass.
- **The "no associated ETA" advisory (#406) is no longer suppressed for mixture class thetas whose
  class-aware shift is switched off (#996)** — under `mu_referencing = false`, for identity-packed θ,
  or for a random effect with negligible variance those θ really are estimated by the
  importance-weighted M-step alone, and the fit now says so.
- **Fits no longer stall on their initial estimates under the default optimizer (#751).** A fit whose
  opening line search fails could previously quit a hair off its starting point — the user-ODE
  warfarin twin stopped 35 OFV units short of the optimum — and report the initial estimates, with
  their standard errors, as the result. Such a fit is now automatically re-run once with the
  identity-Hessian overshoot guard held on until it escapes its starting point; the second attempt is
  reported when it both left the initial estimates and reached a lower objective, and otherwise the
  original result stands. Affected models reach the same optimum, and the same NONMEM-validated
  standard errors, as their analytical twin. Fits that were already leaving their initial estimates
  are unaffected — they never trigger the retry and their trajectory is unchanged.
- **A fit pinned at its initial estimates is no longer reported as converged (#751).** The
  plateau rule that reclassifies a bare NLopt `Failure` at a flat optimum as convergence now also
  requires the fit to have left its starting point. A stalled fit that booked one small objective
  improvement and then went flat previously satisfied the rule and was flagged `converged`, which
  also handed the covariance step a non-stationary point.
- **Inner EBE no longer certifies a runaway mode, so the FOCE/FOCEI objective is
  gradient-path-independent (#958).** On a model where a prediction is driven toward zero under
  proportional error, that observation's variance is clamped to the floor and its residual term
  amplifies the ODE solver's local error by ~1e8 — enough for the *finite-difference* inner
  gradient to come back with the wrong sign. The inner search then walked tens of prior SDs away
  from η = 0 and accepted the point it landed on, because a noise-driven search satisfies the
  objective-stall convergence test exactly like a converged one. `gradient = fd` and
  `gradient = auto` therefore reported different objectives for the same model at the same
  estimates, which made ΔOFV/ΔAIC model selection depend on the gradient route. The inner loop now
  re-solves derivative-free from the prior mean whenever the returned EBE's Mahalanobis distance
  `ηᵀΩ⁻¹η` exceeds 10 prior SDs per random effect, and keeps whichever point has the lower
  objective. On the reported reproducer the first-evaluation objective goes from `fd` 6082.24 /
  `auto` −5.478 to −5.479 / −5.478, and both routes now recover the simulating parameters. The
  analytic (`Dual2`) sensitivities were correct throughout — verified against finite differences of
  the predictor to 2.6e-7 — so `gradient = auto`, the default, was never affected.
- **`gradient` option error no longer advertises the retired `ad` route (#958).** The parse-time
  message for an unknown `gradient` value listed `'auto'`, `'ad'`, or `'fd'`, but no fit accepts
  `ad` — the Enzyme automatic-differentiation path was retired in #428 and the engine rejects it —
  so a mistyped option pointed users at a setting that cannot work. The message now lists only
  `'auto'` and `'fd'`. Writing `gradient = ad` still parses, so it continues to reach the engine's
  specific "no longer supported" error rather than a generic one.
- **Per-subject diagnostics now honour `MIXEST` in a mixture fit (#985).** `IPRED`, `PRED`,
  `IWRES`, `CWRES`, `EBE_OFV`, and `[derived]`/`[output]` columns were computed with `MIXNUM`
  pinned to class 1 for every subject, even for subjects the fit assigned to another class — so a
  class-2 subject's sdtab row paired its class-2 EBEs with class-1 typical values. They are now
  evaluated in each subject's fitted class. Affects FOCE/FOCEI, SAEM, Bayes, and estimating IMP/IMPMAP mixture fits.
- **Mixture covariance-step and diagnostics correctness (#984, follow-up to #983).** Five fixes to
  the mixture SE/covariance and checkpoint paths: (1) the covariance step now reconverges its per-class EBEs at
  `cov_inner_tol`, not the fit's `inner_tol`, so a loose fit followed by a tight `cov_inner_tol` for
  trustworthy SEs is honoured instead of silently building the Hessian on the loose EBEs; (2) when a
  per-class `omega(k)`/`sigma(k)` override collapses and makes the covariance base OFV non-finite,
  the diagnostic now inspects the worst-conditioned class Omega and reports the collapse instead of
  misattributing it to a model-evaluation overflow; (3) an explicitly chosen optimizer that cannot
  drive a mixture (built-in BFGS/L-BFGS, trust-region, Gauss-Newton) is still run under BOBYQA but
  now emits a warning rather than dropping the choice silently; (4) EBE non-convergence / fallback /
  hard-reject are now counted over every class that contributes to a subject's marginal, not just its
  winning class, so a non-converged non-winning class is no longer hidden from the convergence guard;
  (5) `.fitrx` restore now orders the `PMIX_*` columns by class number rather than raw header position,
  so a bundle whose columns were reordered (e.g. alphabetically, `PMIX_10` before `PMIX_2`) no longer
  silently swaps class probabilities.
- **Mixture posteriors survive a `.fitrx` checkpoint (#983).** A saved-then-restored mixture fit
  now re-emits its per-subject `MIXEST` / `PMIX_1..K` columns: they round-trip through optional
  trailing columns on `ebes.csv` (a non-mixture bundle is byte-identical to before). Previously the
  checkpoint dropped them, so a restored mixture fit's `sdtab` silently lost the mixture columns.
- **Mixture-model correctness fixes (#980, follow-up to #977).** The analytic outer gradient for a
  `MIXNUM`-branched typical value (e.g. class-specific clearance) now resolves the correct class on
  every rayon worker, so a gradient optimizer (SLSQP/L-BFGS/MMA) is no longer misled by a
  class-swapped gradient; a covariate used only in a mixing expression is now registered as a
  required data column (it was silently read as 0, degrading covariate mixing to intercept-only);
  `MIXNUM` outside a `[mixture]` model is rejected everywhere, not just in `[individual_parameters]`;
  and an evaluation-only mixture run (`outer_maxiter = 0`) now emits the `PMIX_*`/`MIXEST` columns
  like a converged fit.
- **A design population is now rejected by `fit()` instead of being fitted to its placeholders.**
  The population returned by `read_population_for_simulation()` carries a `NaN` placeholder for
  each not-yet-generated observation, and nothing checked observations for finiteness: with
  log-transformed-both-sides on natural-scale data the placeholder was floored to a finite,
  extreme value and the fit ran to completion on fabricated data with no warning. Such a
  population now fails up front with `E_NONFINITE_DV`, and propensity-score-matched simulation
  reports the real cause instead of a misleading "EBE did not converge" (#957).
- **The default (`Auto` → analytic-gradient NLopt L-BFGS) optimizer no longer stalls at the
  initial estimates on its first step.** From the identity initial Hessian the opening search
  direction is `−∇`, and on models where the scaled gradient is large at the start (e.g.
  warfarin FOCEI) that step overshot, the line search failed on evaluation 1, and the fit never
  left its initial θ — reporting standard errors for the initial point instead of the optimum.
  The identity-Hessian overshoot cap (previously SLSQP-only) now also tames the **first** L-BFGS
  gradient evaluation; later evaluations are left untouched so every *later* `(s, y)` curvature
  pair L-BFGS builds stays intact (only the first pair reflects the capped opening gradient).
  Combined with the analytic covariance Hessian (default-on), this
  re-enables the warfarin FOCEI covariance SE cross-checks against NONMEM and a steady-state
  oral fit smoke test (#960).
- **`sir = true` is no longer silently skipped when the covariance step fails**
  ([#972](https://github.com/FeRx-NLME/ferx-core/issues/972)). A non-positive-definite
  FD Hessian used to leave a requested SIR run with no intervals and a warning pointing
  at `covariance = true`, even though the `|eigenvalue|`-rectified fallback proposal for
  exactly that case had already been built — reaching it required the separate
  `covariance_fallback = sir` option. `sir = true` now arms that fallback itself and the
  fit reports `covariance_status: sir_fallback`. When no proposal can be built at all,
  the warning now points at the covariance-step message that carries the actual cause,
  and a Bayesian fit is told that posterior credible intervals replace the
  Hessian-based covariance — instead of both being sent to an option that is already on.
- **`FitResult$vi$elbo_tightness_ratio` says whether the reported bound is usable**, which
  `converged` cannot: a stuck optimizer produces a flat objective and stable parameters
  just like a successful one. The data term's excess over its value at the variational
  means is `≈ d/2` per subject when `q` has the posterior's curvature, so the ratio of
  measured to expected is ~1 for a healthy fit; above 25 the fit warns and names the likely
  fix. A deep compartment model started badly scored 306.6 while reporting
  `converged: true`; the same model with `init` declared scored 0.54.

  **`FitResult$vi$elbo_tightness_ratio` says whether the reported bound is usable**, which
  `converged` cannot: a stuck optimizer produces a flat objective and stable parameters
  just like a successful one. The data term's excess over its value at the variational
  means is `≈ d/2` per subject when `q` has the posterior's curvature, so the ratio of
  measured to expected is ~1 for a healthy fit; above 25 the fit warns and names the likely
  fix. A deep compartment model started badly scored 306.6 while reporting
  `converged: true`; the same model with `init` declared scored 0.54.


## [0.3.0] - 2026-08-07

### Added
- **Variational inference (`[fit_options] method = vi`)** — a third way to marginalize the
  random effects, alongside profiling them out at their mode (FOCE/Laplace) and sampling them
  (SAEM/IMP). VI fits a tractable posterior `q(η)` per subject and optimizes its parameters
  jointly with θ/Ω/Σ by maximizing the evidence lower bound, with no inner loop. It suits
  models whose fixed-effects part is highly flexible (deep compartment models), cases where
  FOCE's inner optimization is unstable, and anyone who wants a per-subject posterior
  *covariance* without paying for a Hessian — VI produces one as a by-product, reported on
  `FitResult$vi`. Following Janssen et al. (2024), with two departures: the KL term is taken
  in closed form rather than by Monte Carlo, and Ω is then updated by its exact maximizer
  rather than by gradient descent, which removes the Ω instability that paper reports.
  VI stops as soon as the objective has settled rather than always burning `vi_iters`
  iterations: `vi_iters` is a **ceiling** (default 25000), and settling is judged by testing
  whether the remaining drift is distinguishable from Monte-Carlo noise. Defaults were set
  against the NONMEM FOCEI warfarin reference (`tests/nonmem/warfarin_imp.lst`), which the
  previous defaults missed badly — θ and Ω are now within ~1% of NONMEM out of the box.
  `vi_lr` defaults to 0.02 (was 0.05): at 0.05 the σ trajectory is non-monotone in iteration
  count. Note that the residual-error estimate is the one parameter limited by Monte-Carlo
  noise rather than by iterations. `vi_mc_samples` defaults to 8 (was 3, the value Janssen
  et al. use): at 3 draws warfarin's σ² lands ~220% high, and an IOV recovery fit returns
  `TVCL` 0.85 against a true 1.0 while reporting `converged: true` — both failing as a
  quiet stop at a worse point rather than as an obvious error. At 8 draws the IOV fit is
  indistinguishable from SAEM and FOCEI on every parameter. Raise it further when the
  residual error itself matters: σ keeps improving with more draws (~72% high at 8, ~39%
  at 16, ~20% at 32 on warfarin) long after θ and Ω have stopped moving.
  Tunable via `vi_iters`, `vi_mc_samples`, `vi_lr`, `vi_family`, `vi_omega_update`,
  `vi_avg_last`, `vi_eta_grad`, `vi_kl` and `vi_seed`. `vi_kl = mc` selects the
  Monte-Carlo KL — unbiased but noisier, what a variational family with no closed-form KL
  requires, and a cross-check of the analytic one. Declared parameter bounds
  (`theta TVCL(0.13, 0.001, 10.0)`), `block_omega` structure and `FIX` all behave as they do for
  the other estimators, and standard errors come from the ordinary covariance step run at the VI
  estimate, so `covariance_status` / `se_theta` are populated as usual. **Note that the ELBO is
  a lower bound, not a
  likelihood:** `ofv` is `NaN` by default rather than being filled with a number that is not
  comparable to a FOCE/SAEM OFV. Set `vi_final_ofv = laplace`, or chain `methods = vi, imp`
  with `imp_eval_only = true`, to evaluate a genuine marginal likelihood at the VI estimate.
  **IOV is supported**: the variational posterior covers the stacked `[η, κ₁ … κ_K]`
  jointly against a block-diagonal prior `Ω ⊕ Ω_iov^{⊗K}`, `Ω_iov` gets its own
  closed-form maximizer pooled over every occasion of every subject, and per-occasion `κ`
  means are reported on `FitResult$vi$kappa_means`. IOV composes with a **time-varying
  clearance** (the busulfan shape) from either the `TIME` built-in or a time-varying
  covariate — a pairing worth stating explicitly because the two effects are confusable:
  both make clearance differ between early and late records. Read `Ω_iov` from a VI fit
  with one caveat: variational posteriors understate posterior variance and `Ω_iov` is a
  mean of `S + μμᵀ` over occasions, so it leans low (about 24% low on a simulated
  60-subject, 4-occasion recovery where `θ` landed within 2%); confirm it with SAEM or
  FOCEI if it is the parameter you care about. Non-Gaussian endpoints
  (TTE / categorical) are still unsupported and are refused with an actionable message.
  **`FitResult$vi$elbo_tightness_ratio` says whether the reported bound is usable**, which
  `converged` cannot: a stuck optimizer produces a flat objective and stable parameters
  just like a successful one. The data term's excess over its value at the variational
  means is `≈ d/2` per subject when `q` has the posterior's curvature, so the ratio of
  measured to expected is ~1 for a healthy fit; above 25 the fit warns and names the likely
  fix. A deep compartment model started badly scored 306.6 while reporting
  `converged: true`; the same model with `init` declared scored 0.54.
- **New ODE steppers via `[fit_options] ode_method`,** on two independent axes. For
  **stability**: the linearly implicit Rosenbrock methods `rosenbrock23` (order 2, aliases
  `ros23`/`ode23s`), `rodas4` (order 4) and `rodas5p` (order 5). These are stable at the step size the tolerance needs on **stiff**
  systems (fast reversible binding / TMDD, Michaelis-Menten with `KM` far below observed
  concentrations, long transit chains, QSP cascades), where the explicit method is
  stability-limited and either crawls or exhausts `ode_max_steps`. Analytic sensitivities run
  through the identical stepper, so switching does not move a model off the analytic-gradient
  path. `rk45` remains the default, and on the `f64` prediction path existing fits are
  bit-identical. The one deliberate exception is the generic (`Dual2`) analytic-sensitivity
  path: RK45's stage combinations previously existed as two separate transcriptions that were
  *not* bit-identical to each other — the `f64` driver associated them as `u + h·(b₁k₁ + b₂k₂)`
  while the generic one accumulated `((u + k₁·(h·b₁)) + k₂·(h·b₂))`. Unifying them onto the
  `f64` association shifts the sensitivity path's trajectory in the last bits, which the outer
  line search can amplify; it also means the gradient is now taken along exactly the trajectory
  the predictor reports, which it was not before. The stiff methods are full peers — each carries its own continuous extension,
  so every feature that reads ODE state between solver steps works with every method:
  non-Gaussian endpoints (TTE / categorical / CTMM), time-to-event simulation, adaptive /
  feedback dosing, `[output]` state columns and the analytic-sensitivity path. Internally the
  three integration drivers (dense saves, soft sampling, event-time root-finding) are now
  written once against a `Stepper` abstraction rather than per method.

  For **order**: `vern7` (Verner 7(6), 10 stages, explicit), for fits that are *accuracy*-limited
  rather than stability-limited — where step count scales as `tol^(−1/p)` and a stiff method
  buys nothing. On the Savic transit NONMEM anchor at `TOL=9`-equivalent accuracy it takes 2.8×
  fewer steps than `rk45` and is ~2.3× faster; at default tolerances it is ~1.4× slower, so it
  is a tight-tolerance tool rather than a blanket upgrade. Its in-step readouts interpolate with
  a cubic Hermite (3rd-order) rather than a matching continuous extension — documented under
  [ODE Models](https://ferx-nlme.github.io/ferx-core/model-file/ode-models.html#stiff-systems).

  `docs/model-file/ode-models.qmd` now carries a measured "which regime am I in?" table so the
  choice is made from solver statistics rather than guesswork.
- **Pre-scheduled base regimen (loading dose) in adaptive-dosing simulation** (#702).
  `simulate_adaptive()` / `simulate_adaptive_from_spec()` now accept a base subject that
  already carries pre-scheduled doses — a loading / maintenance regimen, including a
  steady-state (`SS=1`) dose — instead of requiring a dose-free subject: the driver
  integrates the base regimen and the controller augments it at the decision schedule (the
  real TDM / MIPD workflow of starting on a fixed regimen and titrating on measured levels).
  The pre-scheduled doses reuse the same dose-resolution, break-timeline, and steady-state
  machinery as `predict()` / `simulate()`, appear in the controller's dose `history`, and
  are rebuilt alongside the realized ledger by the default-on frozen-replay verifier. A base
  dose sharing a time with a decision is observed **pre-dose** (the trough), symmetric with
  the controller's own doses (#933); the TAFD anchor is the true global earliest dose even
  when a controller dose precedes the earliest base dose (#934); a base dose past a
  controller `Stop` still lands (the base regimen is the patient's standing prescription);
  and base doses into lagged / built-in input-rate (transit / zero-order absorption)
  compartments are supported (#935). Initially supported on constant-covariate, non-reset
  models only; time-varying-covariate (#930), IOV (#931), and system-reset (#932) base
  regimens are separate follow-ups (see below), never a silent mis-integration. Previously any
  base regimen was rejected outright.
- **Pre-scheduled base regimen under a time-varying covariate in adaptive-dosing
  simulation** (#930). `simulate_adaptive()` / `simulate_adaptive_from_spec()` now accept a
  base subject carrying a plain bolus / infusion loading (or maintenance) regimen together
  with a **time-varying covariate** (declining renal function, a `TIME`-driven parameter,
  etc.): each base dose's bioavailability `F` is resolved from its own covariate snapshot
  (the covariate active at the dose's administration time — symmetric with a controller dose,
  whose `F` is fixed at injection) and the dose is integrated under the per-segment PK, with
  the base-aware frozen-replay verifier carrying that `F` so the run is checked bit-for-bit.
  Validated dose-for-dose against an independent mrgsolve renal-decline + loading-dose run
  (`tests/reference/vanco_renal_loading_mrgsolve/`). Lifts the #702 `base × time-varying`
  restriction. Still a typed error under a time-varying covariate (a #930 follow-up): a
  steady-state, lagged, built-in input-rate, or modeled-`RATE` base dose; and a base regimen
  combined with system resets (#932) remains rejected (base × IOV is lifted by #931, below).
- **Pre-scheduled base regimen under inter-occasion variability (IOV) in adaptive-dosing
  simulation** (#931). `simulate_adaptive()` / `simulate_adaptive_from_spec()` now accept a
  base subject carrying a plain bolus / infusion loading (or maintenance) regimen together
  with **inter-occasion variability** (`kappa`): each base dose's bioavailability `F` (and any
  other IOV-affected individual parameter) is resolved under the κ of the occasion — the
  decision window (#701) active at the dose's administration time — symmetric with a
  controller dose, whose `F` is fixed from the per-decision snapshot at injection, and the dose
  is integrated under the per-segment occasion PK with the base-aware frozen-replay verifier
  carrying that `F` so the run is checked bit-for-bit. Validated against the independent
  `predict_iov` engine on a reconstructed per-occasion κ and dose-for-dose against an mrgsolve
  loading-dose IOV run (`tests/reference/vanco_iov_loading_mrgsolve/`). Lifts the #702/#930
  `base × IOV` restriction. Still a typed error under IOV (a #931 follow-up): a steady-state,
  lagged, built-in input-rate, or modeled-`RATE` base dose; and a base regimen combined with
  system resets (#932) remains rejected.
- **System resets (EVID=3) in adaptive-dosing simulation** (#716). `simulate_adaptive()`
  / `simulate_adaptive_from_spec()` now honor an EVID=3 reset carried by the base
  subject: the reactive driver zeros the compartments at the reset time (re-seeding any
  `init(state)=expr`) and turns off controller-issued infusions opened before it, exactly
  as `predict()` / `simulate()` do, and the default-on frozen-replay verifier is
  reset-aware so the reset is validated each run. Previously a reset subject was rejected
  with a typed error. (Initially only a pure EVID=3 reset on a **dose-free** base subject;
  #932 below lifts that to a reset combined with a base regimen, including an EVID=4
  reset+dose row.)
- **Base regimen combined with a system reset (EVID=3 / EVID=4) in adaptive-dosing
  simulation** (#932). `simulate_adaptive()` / `simulate_adaptive_from_spec()` now accept a
  base subject that carries BOTH a pre-scheduled loading / maintenance regimen (#702) AND a
  system reset — composing #716's reset machinery with #702's base-dose seeding on the
  constant-covariate path. The reset zeros the compartments and turns off a base infusion
  opened before it (the reset floor, previously applied only to controller-issued infusions),
  and an EVID=4 reset+dose row's dose now reaches the adaptive path — landing after its own
  reset (Reset < Dose) — instead of tripping the base-regimen guard. Validated by a degenerate
  oracle (base regimen + mid-horizon EVID=3 + controller reproduces `predict()` on the realized
  regimen carrying the same reset), a positive control (a base infusion spanning the reset is
  turned off, so the oracle is not vacuous), an EVID=4 oracle, and a steady-state base × reset
  oracle; the default-on frozen-replay verifier (reset- and base-aware) checks every run. Lifts
  the #702/#716 `base × reset` restriction. Base × reset UNDER a time-varying covariate (#930)
  or IOV (#931) remains a typed error — a #932 follow-up — never a silent mis-integration.
- **Warning when a `combined` error model's additive initial estimate is negligibly
  small** (#847). A pre-fit check (`W_ADDITIVE_INIT_SCALE`) flags an additive SD start
  below 1% of the observation scale (median `|DV|`) on that endpoint. A near-zero
  additive start can trap the fit in a local minimum where the additive term collapses
  and the proportional term inflates — the worse basin on multimodal / over-parameterised
  problems (e.g. the cyclophosphamide parent→metabolite fit, where additive variance
  seeded at 0.5 traps at ≈0.94 instead of the global optimum ≈1878). The initial estimate
  is never changed — the warning advises a larger start.
- **Simulation and prediction for binary (`[binary_model]`) endpoints** (#760). `simulate()` now
  draws a 0/1 outcome per binary observation record (previously it emitted **no rows at all** for
  a binary endpoint, silently and without an error), and the new `predict_categorical()` returns
  the category probabilities `P(Y = 0)` / `P(Y = 1)` per record — a probability vector, which the
  scalar `predict()` cannot represent. `predict()` remains the Gaussian predictor and returns no
  rows for a binary endpoint (see `Changed` below for the one behavioural change it did gain). A
  `[simulation]` block driving a binary endpoint needs `times` (binary outcomes are observed on
  the fixed grid), and `--simulate` stamps the drawn states onto the simulated population it then
  fits.
  Validated by a simulate → fit recovery (SSE) round trip on top of the existing R `glm` / NONMEM
  fit anchors.
- **sdtab diagnostics for binary endpoints** (#760). `{model}-sdtab.csv` now carries one row per
  binary observation record — `DV` (observed 0/1), `PRED` (`P(Y=1)` at η=0), `IPRED` (at the EBE η)
  and `IWRES` (standardized Pearson residual). `CWRES` is left blank, since the conditional
  weighted residual is defined through the Gaussian residual-variance model that a Bernoulli
  outcome does not have; columns undefined for a discrete record are likewise blank rather than
  sentinel values. Previously a binary endpoint produced no diagnostic rows at all.
- **Analytic sensitivities for steady-state dosing into a built-in absorption compartment**
  (#835). Fitting a model with an `SS=1` dose into a `first_order` / `transit` / `igd` /
  `weibull` absorption compartment now uses exact analytic FOCEI gradients — the closed-form
  steady-state trough `u_ss = (I − M)⁻¹·b` carried over dual numbers — instead of finite
  differences, so these fits run at full analytic speed. Steady-state into a `zero_order`
  window, and steady-state combined with an absorption lagtime, remain out of scope (rejected
  with a clear error).

### Changed
- **`ObsRecord::DiscreteState` and `ObsRecord::Count` gained a `raw_time` field** (#760). Discrete
  observations now carry the user's TIME alongside the engine's internal (occasion-shifted) clock,
  so reported rows join back to the input CSV the way Gaussian rows always have. Source-breaking
  for any external code that constructs or exhaustively destructures these variants.

### Fixed
- **LTBS models scored a different objective under SAEM / IMP / VI than under FOCE.** The
  fixed-η observation likelihood applied a `max(1e-12)` positivity floor to the
  prediction — correct for a concentration, wrong for `log(concentration)`, which is
  legitimately negative for any concentration below one unit (ng/mL data, late samples, a
  high-clearance subject). Affected observations had their residual computed against ~0
  instead of the true negative log-prediction, inflating it silently. FOCE/Laplace/AGQ
  never applied the floor and were unaffected, so this showed up as SAEM/IMP/VI
  disagreeing with FOCE on the same model. The floor now stands down under LTBS, where
  positivity is already enforced on the natural scale before the log is taken.

- **Gradient-path-dependent FOCE/FOCEI objective on proportional-error models with a
  near-zero prediction** (#958). The residual-variance floor (`MIN_VARIANCE`, which clamps
  `(f·σ)²` so a vanishing prediction cannot produce a zero variance) was applied to the
  variance but **not** to its analytic derivatives `∂R/∂f` / `∂²R/∂f²`. On a row whose
  prediction is driven to ~0 (e.g. drug fully eliminated at a late sample) the variance is
  clamped and locally constant in `f`, so its true derivatives are 0, but the accessors
  returned the raw `2·f·σ²` / `2·σ²`. That made the analytic inner-EBE gradient disagree with
  the finite-difference objective it minimises, so `gradient = auto` (analytic) and
  `gradient = fd` could converge to different empirical-Bayes modes and report different
  objective values (hence different ΔOFV/ΔAIC) for the same model at the same estimates. The
  variance-derivative accessors are now floor-aware, restoring analytic ↔ FD agreement.
- **Standard errors for closed-form LTBS models under IOV** (#486). The tighter inner-EBE
  tolerances that log-transform-both-sides models take for the fit and covariance steps
  (`LTBS_FIT_INNER_TOL` / `LTBS_COV_INNER_TOL`, #665) were previously skipped whenever the
  model also carried `[iov]`, on the grounds that LTBS × IOV ran its inner loop on finite
  differences. Now that this combination takes the analytic `ln(f)` inner gradient, it carries
  the same tolerance sensitivity as any other closed-form LTBS model, so it takes the same
  tightened tolerances. Without this a fit would return quietly inflated standard errors —
  the same mechanism measured at roughly 65% on warfarin theta SEs — with point estimates
  unchanged, so nothing looked wrong. Set `cov_inner_tol` / `inner_tol` explicitly to override.
- **Parser-internal readout parameters no longer surface in warnings or diagnostics** (#486).
  A `[scaling] y = ...` readout that names a theta or eta directly is desugared into a hidden
  individual parameter. That hidden parameter could reach the "individual parameter(s) not
  mu-referenced" warning (advising the user to rewrite a parameter absent from their model
  file) and the eta/parameter metadata carried in `FitResult`, where it appeared under its
  internal `__ferx_ro_*` name with a `Custom` parameterisation. Both now filter it out, as the
  other consumers of that list already did. Present on `[odes]` models since #631.
- **FD-fallback warning for an oversized readout quotes the right slot budget** (#486). When a
  direct theta/eta readout cannot be given PK slots, the resulting parse warning reported the
  128-slot ODE layout even for analytical models, whose spare region holds at most about 11
  slots and typically 4–7 — overstating the user's headroom by an order of magnitude and
  making the suggested remedy unactionable. It now quotes the pool the model actually draws from.
- **Adaptive-dosing `auc_target` exposure metric now integrates the pre-scheduled base regimen**
  (#940). The signal-AUC pass behind `auc_target_attainment` (#391 S2.5b) previously scored each
  decision window on the controller's realized doses only, dropping any pre-scheduled base regimen
  (#702 — a loading / maintenance dose), so on a base-regimen run the reported exposure — and hence
  `auc_target_attainment` — was biased low. Each window now integrates the base regimen (a loading
  dose before the window and a maintenance dose landing inside it) alongside the realized ledger,
  matching `predict()` / `simulate()` on the combined regimen. Constant-covariate subjects only, as
  before (time-varying-covariate / IOV / reset `auc_target` runs remain rejected).
- **FOCE inner EBE recovery no longer discards a near-optimal partial for a worse fallback**
  (#378). When a subject's individual objective is multimodal (e.g. a 3-cpt IV proportional
  model with six IIV etas conditioned on ten points), the closed-form inner BFGS can reach the
  conditional mode yet stall at a gradient norm just above `inner_tol`; the exact-path recovery
  then restarted Nelder–Mead from η=0 and kept that (worse) basin, discarding the good BFGS
  partial. The recovery now keeps the lower-objective of {BFGS partial, NM} on non-FREM models —
  the guard the ODE inner path already used (#555) — so the closed-form and ODE forms converge to
  the same EBE. This removes the ODE↔analytical FOCE **marginal** OFV divergence (up to ~18 OFV
  units, platform-sensitive) that surfaced after the `inner_tol` tightening in #330. FREM keeps
  its cold-restart re-centering unchanged.
- **The reported inner-loop gradient method for `[odes]` models is no longer mislabeled "finite
  differences"** (#926, follow-up to #378). `fit()` already runs the exact analytic inner
  η-gradient for an in-scope ODE model (as it does for closed-form and IOV models), but the
  reported `gradient_method_inner` — shown in the fit banner and `{model}-fit.yaml` — was derived
  from a closed-form-only predicate and always read "finite differences" for ODE, even while the
  outer-loop report correctly read "analytic". It now reports the analytic method for an in-scope
  ODE model, matching the outer report and the route actually run. The report and the
  FD-fallback warning now share one predicate, so an in-scope ODE model whose every subject is
  genuinely finite-differenced (oral infusion into a built-in absorption compartment, or a
  rate-defined infusion under `F ≠ 1`) is still surfaced by the route banner and the warning
  rather than silently labeled analytic. Reporting only — no estimate, OFV, or diagnostic
  changes.
- **Steady-state (`SS=1`) dosing on `[odes]` models is now exact for a linear disposition, and
  warns instead of silently truncating on a nonlinear one** (#914). An ordinary ODE bolus or
  infusion SS dose previously equilibrated by expanding a pulse train capped at 50 cycles, which
  under-reported the steady state by tens of percent for a slow disposition (the same truncation
  #908 removed from the analytical engine) — silently. A linear disposition now solves the exact
  periodic fixed point `(I − M)⁻¹·b` directly (value and analytic FOCE/FOCEI gradient), so slow PK
  is exact rather than low; a genuinely nonlinear RHS (e.g. Michaelis–Menten) still falls back to
  the capped iteration but now surfaces a non-convergence warning when the cap is reached without
  settling. Also faster: the linear case replaces ~50 RK45 cycles with a handful.
- **`predict()` on a CTMM (`[markov_model]`) model now fails loud instead of silently returning
  no rows** (#759). The equivalent `simulate()` guard already existed and its message claimed to
  cover `predict()`, but it only ran on the simulate path. State-occupancy prediction `π(t)` is
  still to come (#820).
- **A pure-TTE / pure-discrete population no longer panics on a `CMT ≥ 2` dose** (#905). Such a
  population is exempt from dose-compartment validation — its `pk` line is a placeholder and the
  TTE endpoint's `CMT` is routinely ≥ 2 — but the predictor still rerouted the dose onto the
  event-driven walk, which aborted the process on the very dose the validator had declined to
  check. The analytical predictor now declines in lockstep with the exemption: a subject with no
  Gaussian observation returns `NaN` without entering the walk (value and FOCE/FOCEI gradient
  paths alike), so `predict()`, `simulate()` and `fit()` stay panic-free on such a population
  through both the endpoint-routed and model-blind loaders.

### Performance
- **`method = laplace` gradients are far more accurate, and its objective is faster.** The
  posterior Hessian that scales the quadrature grid is now taken analytically — it is the exact
  conditional `∂²nll/∂b²` the shared sensitivity sweep already assembles — instead of being
  rebuilt by a `2d²+1` per-subject finite-difference sweep. The grid-response term of the
  gradient likewise stopped re-sweeping the whole quadrature grid once per parameter: each
  node's analytic `∂nll/∂b` is computed once and contracted against a node displacement
  differenced from a cheap exact linear-algebra map. Between them these removed a
  finite-difference *of* a finite-difference, and the gradient's agreement with a reconverged
  finite difference of the objective improved by two to three orders of magnitude (warfarin:
  `2.0e-5` → `6.5e-8` relative at `n_agq = 1`, `2.1e-7` → `1.8e-9` at `n_agq = 7`). The Hessian
  change also speeds up every objective evaluation, and therefore the covariance step, which
  evaluates it `~2·n_free²` times. Laplace OFVs and standard errors shift very slightly, since
  the exact Hessian replaces an approximated one. `method = focei` is unaffected: the two
  estimators still use **different** Hessians — that is what distinguishes them — and only the
  computation is now shared. Models outside the analytic sensitivity scope (TTE, categorical)
  keep the finite-difference path.

- **Exact analytic covariance R-matrix for FOCE/FOCEI** (#436). Standard errors on in-scope
  models are now the exact second derivative of the marginal the outer loop minimises,
  assembled from third-order sensitivities, instead of a second difference of the reconverged
  objective. The finite-difference stencil evaluated the objective `~2·n_free²` times, each
  re-solving every subject's inner loop, and amplified error as `1/h²`; the analytic route
  costs `2N+1` sensitivity evaluations per subject (`N = n_theta + n_eta`) with **no inner
  re-solve** beyond the single reconvergence at the converged point, and has no
  `fd_hessian_step` to tune. Measured on warfarin: agreement with the
  reconverged finite difference to `8.2e-5`, at 3.0× the speed per subject. Out-of-scope models
  (ODE, LTBS, IOV, M3/BLOQ censoring, expression scaling, Form-C readouts, `iiv_on_ruv`,
  correlated or custom-magnitude residuals, time-varying covariates, FREM, covariate-selected
  error models, non-Gaussian endpoints, `method = laplace`/`agq`, and `gradient = fd`) keep the
  finite-difference covariance unchanged — it is correct for all of them — and a single
  out-of-scope subject drops the whole population back to it rather than mixing two
  approximations in one matrix. Set `analytic_cov_hessian = false` in `[fit_options]` to force
  finite differences.

- **Exact analytic inner (EBE) gradients for log-transform-both-sides under inter-occasion
  variability** (#486). Fitting a closed-form model that combines `log(DV) ~ additive(...)`
  with `[iov]` now uses exact analytic sensitivities for the per-subject empirical-Bayes
  gradient, not finite differences. The population (outer) gradient has been analytic for this
  combination since #677; the inner loop had stayed on FD because the first-order IOV walk
  carried no `ln` jet, which made LTBS × IOV the last combination whose two loops
  differentiated by different means. Both now apply the same `g = ln(f)` transform last, after
  the per-occasion output-scale quotient, so they differentiate the same `ln(f/s)` the
  objective scores — including combined with an expression `obs_scale`. Estimates are
  unchanged; the EBE search reaches the same modes with one provider evaluation per inner step
  instead of `~2·n_eta` predictions. LTBS × IOV on `[odes]` models keeps the finite-difference
  fallback on both loops, as before.
- **Exact analytic gradients for a closed-form Form C readout that references a θ or η
  directly** (#486). An analytical (1-/2-/3-cpt) model whose `[scaling] y = <expr>` readout
  names a theta or eta directly — e.g. `y = central/V * TVSCALE + ETA_BASE`, a baseline or
  scale factor that is not an `[individual_parameters]` entry — now takes the analytic
  sensitivity path on both loops instead of falling back to finite differences. The parser
  already desugared such a reference into a hidden individual parameter on the ODE path
  (#631); that pass now runs for the closed-form engine too, where the hidden parameter draws
  a free differentiable PK slot exactly like any other non-structural readout parameter
  (`BMAX`/`KD`, #650). Predictions are unchanged — only the gradient moves off finite
  differences. A model whose readout parameters overflow the slots its PK model leaves spare
  keeps the FD fallback, with the existing parse warning, rather than failing to parse.
- **Honest `gradient_method` reporting when a readout outgrows the closed-form dispatch
  tables** (#486). A closed-form model whose differentiated PK-slot count exceeds the width
  the sensitivity providers instantiate now reports `fd` instead of `analytic`. Previously the
  scope check verified that every slot was differentiable but never how many there were, so
  such a model was labelled analytic and then fell back to finite differences on every
  subject — the persisted label disagreed with the route actually taken. No estimates or
  standard errors change (those fits were already running on FD); only the reported and
  persisted gradient method does.
- **Closed-form modified-release absorption** (#860). A static multi-route absorption model
  (parallel / mixed pathways #505, per-route lag #856 — one `[odes]` central compartment fed by a
  fraction-weighted superposition of `first_order` / `transit` / `igd` input-rate forcings into a
  linear 1-/2-compartment disposition) now evaluates `predict()` / `simulate()` and
  finite-difference fits as a **closed-form superposition of shifted single-route solutions, with
  no ODE integration**, when the subject has no time-varying covariates, IOV, resets, or
  steady-state / infusion doses. The disposition is recognised from the compiled model by its
  behaviour (a probed constant, canonical 1-/2-cpt Jacobian), not by matching how it is written,
  and a non-linear disposition is declined and integrated as before. Predictions are unchanged to
  within the ODE solver tolerance — the closed form reduces to the same integrated twin — and any
  model outside this scope (including `weibull` pathways) continues to integrate.
- **Closed-form modified-release absorption now covers `zero_order` routes** (#860 Phase B). A
  `zero_order(dur=...)` pathway in a static multi-route model no longer falls back to ODE
  integration for `predict()` / `simulate()` / finite-difference fits — the box-car input rate has
  its own closed-form convolution against the linear 1-/2-cpt disposition, superposed exactly like
  the other pathway kinds. Predictions are unchanged to within solver tolerance.
- **Closed-form modified-release absorption now accelerates the analytic FOCE/FOCEI gradient
  too** (#860 Phase A6). Fitting a static multi-route model with the default analytic-sensitivity
  method now skips the ODE integration for both the value AND the gradient — previously only the
  value took the closed-form fast path, and the gradient still integrated. The disposition-recovery
  and superposition formulas are evaluated once, generically, over dual numbers instead of plain
  doubles, so there is no separate hand-derived gradient to drift out of sync. Gradients are
  unchanged to within solver tolerance — verified directly against the ODE-integrated analytic
  provider, not only against finite differences.
- The closed-form steady-state equilibration for dosing into a built-in absorption compartment
  (#834) now actually takes effect. Its self-verification tolerance sat just below the solver's
  own noise floor, so it silently fell back to the 50-cycle pulse-train iteration at every
  realistic `ode_reltol`; `predict()` / `simulate()` on these models are correspondingly faster.
  Predictions are unchanged to within the ODE solver tolerance — the closed-form fixed point and
  the iteration converge to the same periodic trough (#835).
- **Exact analytic FOCE/FOCEI gradients for per-route absorption lag** (`fn(..., lag=L)`, #859).
  Fitting a model with a per-route absorption lag on a `first_order`, `zero_order`, `transit`, or
  `igd` input-rate forcing now uses exact analytic sensitivities instead of finite differences —
  each route's onset is a moving boundary carried by a rate-on saltation (and, for `zero_order`,
  a matching rate-off at the window end), so these fits run at full analytic speed. A per-route
  lag on a `weibull` forcing keeps the finite-difference fallback (its onset diverges for shape
  β < 1, so no closed-form saltation exists). The predicted values are unchanged — only the
  gradient moves off finite differences.
- **Exact analytic FOCE/FOCEI gradients for per-route absorption lag under IOV** (#877). A
  per-route absorption lag (`first_order` / `zero_order` / `transit` / `igd`) combined with
  inter-occasion variability now uses exact analytic sensitivities too — the per-route onset
  saltation carries each occasion's `κ` through the same event-driven walk as `η`/`θ`, so a
  route-lag model with IOV fits at full analytic speed instead of finite differences. A
  `weibull` per-route lag keeps the finite-difference fallback (as on the non-IOV path).
  Predictions are unchanged.

### Changed
- **SAEM prints its final OFV before the covariance step** (#893). In a verbose run
  (the CLI default), the `SAEM completed. Final OFV = …` line is now emitted *before*
  the covariance matrix is computed rather than after, so you can judge the fit and
  interrupt (Ctrl-C) before paying for the — often expensive — covariance step when the
  OFV already rules the run out.
- **`method = agq` removed; adaptive quadrature is now an *argument*, not a method**
  (#251). Adaptive Gauss–Hermite quadrature is not a separate estimator — it is the
  single-point method (Laplace / FOCEI) evaluated on more nodes. So the **method name now
  selects the Hessian anchor** and **`n_agq` (default 1) is the node count**:
  - `method = laplace` — the exact-Hessian anchor. `n_agq = 1` is the Laplace approximation
    (NONMEM `LAPLACIAN`); `n_agq > 1` is adaptive Gauss–Hermite quadrature (what `method =
    agq` used to be).
  - `method = focei` — the Gauss-Newton anchor. `n_agq = 1` is plain FOCEI (unchanged,
    bit-identical); `n_agq > 1` is a **new** Gauss-Newton-anchored quadrature that refines
    FOCEI toward the exact marginal (requires the analytic sensitivity scope).

  Both anchors converge to the same marginal likelihood as `n_agq → ∞`; they differ only in
  node placement and, at one node, in whether `½log|H|` carries the exact curvature or the
  Gauss-Newton approximation. The old `method = agq` (and its `gauss_hermite` /
  `adaptive_gaussian_quadrature` aliases) is rejected by the parser with a message pointing
  to `method = laplace` + `n_agq`. **This is a breaking change to the model file's
  `[fit_options]`; `method = agq` was unreleased, so no released version — and no persisted
  `.fitrx` bundle — is affected.**
- **Laplace / adaptive-GH quadrature uses a tighter default `inner_tol` (`1e-8`, was the
  shared `1e-5`)** (#251). Its analytic gradient assumes the EBE is exactly the posterior
  mode; a loose inner tolerance left `b̂` off-mode from a poor start and could stall the
  outer optimizer. The tighter default converges robustly from realistic starts at
  negligible cost near the optimum. FOCE/FOCEI are unchanged (their Gauss-Newton `log|H̃|`
  is forgiving of a loose mode).

### Added
- **Infusion into an oral model's peripheral compartment on the analytical engine** (#375).
  `two_cpt_oral` (`CMT=3`) and `three_cpt_oral` (`CMT=3`/`CMT=4`) previously rejected a
  positive `RATE` — and, before that, crashed on one — because the oral closed-form
  propagators had no peripheral forcing term where the IV ones did. They need no new closed
  form: nothing flows back into the depot, so a rate into a peripheral drives exactly the
  central/peripheral sub-system the IV model has, and the oral propagator superposes that
  same forced response onto its own homogeneous evolution. Combined with the bolus change
  below, **every compartment of the six analytical disposition models now accepts both a
  bolus and an infusion**, alone or together. (The transit and inverse-Gaussian absorption
  models are unchanged — they are dosed through the depot, `CMT=1`, only.) Validated
  against NONMEM 7.6.0 `ADVAN4`/`ADVAN12` and —
  more tightly than NONMEM can express, since its own forced response carries ~2e-6 here —
  against a `1e-12` integration of the same system written out as explicit `[odes]`, which
  agrees to ~1e-11 (`tests/oral_peripheral_infusion.rs`), including a peripheral infusion
  overlapping an oral depot dose.

### Fixed
- **Analytic-gradient fits no longer report `converged = false` at a plateaued optimum**
  (#751). The default analytic-gradient NLopt L-BFGS drives the OFV flat to ~8 significant
  figures and then returns a bare `NLOPT_FAILURE` — its line search can no longer beat an
  objective already at the noise floor. That terminal status was taken at face value, so a
  finished fit was mislabelled non-converged, the "Outer optimization did not converge"
  warning fired spuriously, and the covariance step ran flagged as off-stationary. A bare
  `Failure`/`ForcedStop` is now reclassified as converged only when the fit actually
  descended past its initial estimates, the OFV trace has plateaued (a flat tail of evals
  with no meaningful improvement), **and** the restored best point is self-consistent (a
  cold inner-loop restart reproduces the best-seen OFV). A genuine early stall — a first
  step that overshoots and leaves the fit pinned at its initial estimates, one still
  descending when it stopped, or an unreproducible warm-start "optimum" — keeps
  `converged = false`, so real non-convergence is never masked.
- **A dose into a compartment an `[odes]` model does not declare is now an error, not a
  silent drop** (#899). On the ODE engine every dose-application site was an unguarded
  `if cmt_idx < n { … }` with no `else`, and nothing upstream rejected an out-of-range
  `CMT`: the data reader has no model, and the analytical dose-compartment check returned
  early for ODE models. A typo'd `CMT` therefore produced a fit that **converged and
  reported a finite OFV having ignored the dose entirely** — no error, no warning. Such
  doses are now rejected up front, naming the subject, time, and the states the `[odes]`
  block declares; `fit()` returns an error and `predict()`/`simulate()` fail with the same
  message, matching every other dose precondition. This is the ODE half of the analytical
  fix in #375. `predict_survival()` — the one member of the `predict`/`simulate` family
  that was missing the guard — now enforces it too, so an unroutable dose can no longer
  silently change the exposure a joint PK-TTE hazard reads.
- **`CMT=0` means the same thing on both engines** (#899). `CMT=0` is NONMEM's *default
  dose compartment* and resolves to compartment 1. The analytical engine has done this
  consistently since #375; the ODE engine did **four** different things with it depending
  on which driver a subject happened to take. The plain dataset path computed `0 − 1` on an
  unsigned index and underflowed — a debug build panicked with "attempt to subtract with
  overflow", a release build wrapped to `usize::MAX`, failed the bounds check, and dropped
  the dose in silence. The event-driven driver (taken when a subject has a time-varying
  covariate, an `EVID=3/4` reset, or IOV) applied it to compartment 1. The steady-state
  equilibration bailed out and returned the *single-dose* curve. The remaining sites — the
  infusion channel list, the `_with_states` driver, the segment-boundary walk, and the
  `sens/` gradient twins — skipped the dose outright. So the same dataset could get three
  different answers, and a fit could differentiate a different dosing history than it
  predicted. Every site now resolves `CMT=0` to compartment 1 — including **compartment-indexed
  dose attributes**: a dose written `CMT=0` now reads `F1` / `ALAG1` where before it missed the
  indexed lookup and silently fell back to the bare `F` / `ALAG` slot, which on a model declaring
  only `F1` means bioavailability defaulted to `1.0` and the dose was delivered at full amount.
  Predictions change for ODE datasets written with `CMT=0`, which previously got nothing (or
  crashed). This unification reaches every dose-compartment comparison, not just the state-vector
  index: a `CMT=0` dose into a built-in `zero_order` absorption compartment now opens its release
  window on the event-driven driver (a reset / time-varying covariate / IOV) — it previously
  matched neither the bolus nor the window there and delivered no mass at all; the steady-state
  gradient of a `CMT=0` dose into a built-in absorption compartment now equilibrates like the value
  path (it previously returned an un-accumulated trough, a silent value≠gradient FOCEI error); and
  the "unsupported steady-state combination" rejections (`E_ABSORPTION_SS_ZERO_ORDER` /
  `E_ABSORPTION_SS_LAG`) now fire for `CMT=0` instead of being bypassed. As on the analytical
  engine, an **infusion** with `CMT=0` is rejected rather than remapped: the default dose compartment
  is defined for a bolus but not for a zero-order input. This closes the cross-engine disagreement on
  `SS` + `CMT=0` noted under #375 below. The `cmt → state index` (and its 1-based complement) is now a
  single named accessor (`DoseEvent::cmt_idx` / `cmt_1based`, #912) so the convention lives in one
  place rather than a dozen open-coded `cmt - 1` / `cmt >= 1` sites that drifted apart. The same
  unification reaches the analytical **closed-form absorption** guard: a `CMT=0` dose on a
  `one_cpt_transit` / `two_cpt_transit` / inverse-Gaussian model is now accepted as the depot
  (compartment 1) and predicts identically to `CMT=1`, where it was previously rejected as a
  "non-depot compartment" — the closed form folds every dose through absorption regardless of
  `cmt`, and its ODE twin resolves `CMT=0` and `CMT=1` to the same forcing, so the two paths agree.
  A genuine non-depot dose (`CMT>=2`) is still rejected.
- **Steady-state doses on the analytical event-driven path are now exact, not truncated**
  (#908). A subject that cannot use dose superposition — because it has a time-varying
  covariate, an `EVID=3/4` reset, IOV, or a dose into a non-default compartment — is served by
  the event-driven walk, which equilibrated an `SS=1` dose by iterating a pulse train capped
  at 50 cycles. The leftover was `≈ exp(−50 · λ_slow · II)`, negligible for typical PK but
  **not** for slow drugs, and the early stop never fired there (it needs the per-cycle
  increment to be negligible, which is exactly what a slow mode prevents). The walk now solves
  the periodic steady state in closed form as `u_ss = (I − M)⁻¹·b`, the fixed point of the
  affine one-cycle map, using the same propagators it already runs. It agrees with the
  superposition closed forms to f64 precision (`≤ 1e-12` relative, bit-identical on several
  models) rather than to a tolerance, so the two representations of one dataset agree by
  construction. Measured error that this removes:

  | model | `II` | was |
  |---|---|---|
  | `one_cpt_oral` (CL 0.1, V 50 — t½ ≈ 350 h) | 12 | **3.0e-1** |
  | `two_cpt_iv` (Q 0.5, V2 500) — central | 12 | 1.0e-1 |
  | `two_cpt_iv` (Q 0.5, V2 500) — peripheral amount | 12 | **5.8e-1** |
  | `three_cpt_iv` (Q3 0.5, V3 400) — central | 12 | 8.2e-2 |
  | `three_cpt_iv` (Q3 0.5, V3 400) — third-compartment amount | 12 | **5.2e-1** |
  | `three_cpt_iv` (CL 5, V1 50, Q 3, V2 80, Q3 1, V3 120) | 24 | 2.3e-3 |

  Compartment *amounts* — what `[derived]` and per-compartment sdtab columns report — were
  affected considerably more than concentrations. **If you have results involving `SS=1` on
  this path from an earlier version, regenerate them.** The gradient walk was converted in the
  same change, so FOCE/FOCEI differentiates the steady state it actually predicts; over dual
  numbers the same solve yields the exact implicit-function derivative. The truncated pulse
  train remains only where no periodic steady state exists (a zero disposition rate constant,
  e.g. `CL = 0`), and that case now raises the existing non-convergence warning instead of
  returning a silently truncated state. The ODE path's ordinary bolus/infusion steady state
  still expands a pulse train (#914) — see `docs/model-file/steady-state.qmd`. Validated against
  NONMEM 7.6.0 in the slow regime the fix targets — new `ss_slow_advan1`/`ss_slow_advan2` anchors
  (`CL = 0.1`, t½ ≈ 347 h) that the pre-fix walk missed by 29 %.
- **SAEM FREM / `iiv_on_ruv` mixing diagnostics and safeguards** (#895). The optimizer-trace
  `mh_accept_rate` (and the verbose banner) now reports the **combined** block + componentwise
  Metropolis-Hastings acceptance rate. Previously it showed only the block kernel, which reads a
  misleading 0% for FREM-scale Ω — the near-deterministic covariate ETAs reject every joint
  move — even when the componentwise sweep is mixing the chain fine. The block kernel now damps
  each FREM covariate coordinate by `min(1, √EPSCOV/√Ω_jj)` so its joint acceptance recovers from
  0% (non-FREM models are unaffected — the multiplier is exactly 1). SAEM also now warns when the
  combined post-burn-in acceptance stays below 1% (the sampler is not mixing, so Ω/σ are
  unreliable).
- **SAEM `iiv_on_ruv` σ × ω_RUV runaway fixed** (#895, #904). Free-σ `iiv_on_ruv` models — where
  the residual is `Y = f + EPS·exp(η_RUV)` — could diverge under SAEM, with ω_RUV inflating toward
  ~49 and σ toward its ceiling (worst on FREM models with an extreme Ω-diagonal scale range). Root
  cause: η_RUV is a residual-scale random effect with no typical-value θ, so its mean was never
  absorbed and drifted along the σ × η_RUV degenerate direction, injecting a spurious mean² into
  `ω_RUV = mean(η_RUV²)`. SAEM now re-centres η_RUV to zero mean each iteration, absorbing the
  shift into σ (which leaves every subject's residual variance exactly unchanged) — the same
  device mu-referenced structural etas already use with their θ. Re-centring runs only when every
  RUV-scaled σ component is free; if any is FIXed (e.g. a fixed additive term), it is skipped and
  the growth caps below act as the backstop. On the 475-subject FREM reprex
  this converges to ω_RUV ≈ 0.28 / σ ≈ 0.20 from both a too-small and a too-large σ start (NONMEM:
  0.28 / 0.18). Belt-and-braces σ and ω_RUV growth caps (each ≈ 20× the reference scale; the Ω cap
  is a correlation-preserving rescale that leaves FIXed off-diagonals untouched) remain as no-op
  backstops, warning if they ever bind. The reference scale is re-anchored to the data-informed
  value reached by the end of exploration and defers to a tighter user upper bound, so a fit
  started from a σ/ω_RUV guess far below the truth is not spuriously clamped or warned.
- **Guarded multi-start inner EBE now covers weakly-identified random effects** (#891). The
  per-subject EBE search (`inner_restarts`, default `1`) previously re-seeded only subjects with
  system resets or time-varying covariates. It now also detects a **weakly-identified coordinate**
  — a random effect whose individual objective is flat (the data adds less curvature than the
  prior, i.e. high per-subject shrinkage) — and re-seeds just that coordinate on the cold start,
  so a distant lower posterior mode is no longer silently missed (e.g. a poorly-identified `V1` in
  a saturable-clearance fluconazole model). The flatness check is a two-point finite difference per
  coordinate; well-identified subjects are unchanged and pay only that probe. The guarded
  multi-start now also runs on an **evaluation-only** fit (`maxiter = 0`, NONMEM `MAXEVAL=0`),
  which previously seeded the inner EBE from `η = 0` but was not recognised as a cold start, so
  the reported per-subject EBEs and objective now reflect the recovered modes.
- **An analytic `[scaling]` readout that references the oral `depot` is rejected when the
  data dose a non-default compartment, instead of silently corrupting the objective**
  (#375). The depot amount behind a Form C readout is reconstructed by dose superposition,
  which never reads the dose's `CMT` — so a bolus written `CMT=2` on a `one_cpt_oral` model
  was reconstructed as if it had been absorbed through the depot, adding a phantom depot
  amount to `PRED` and therefore to the OFV. Measured on `y = (central + depot)/V`: OFV
  188.37 where an explicit `[odes]` twin of the same model gives 1761.47 (the same model
  with the dose at `CMT=1` agrees with the twin exactly). This joins the existing
  reset-based rejection in `check_analytic_readout_support`, with the same remedy —
  reference only `central`, or use an `ode(...)` model.
- **A `[derived]` integral over `compartments[i]` no longer returns a wrong finite value
  for a subject dosing a non-default compartment** (#375). The per-observation compartment
  columns correctly degrade to `NaN` for those subjects, and the emitted warning says so —
  but the separate dense-grid reconstruction used by `integral(...)` still used the older,
  narrower predicate and fell through to the compartment-blind superposition helper. In the
  same sdtab row, `compartments[1]` read `NaN` while `integral(compartments[1], 0→24)` read
  19.17 against a true 31.13. Both paths now use the same predicate.
- **A zero-amount dose with an out-of-range `CMT` is rejected instead of aborting the
  process** (#375). The dose-compartment check skipped `AMT=0` rows entirely, on the
  reasoning that a zero bolus is a no-op — but both prediction walks bound-check the
  compartment *before* the amount is read, so such a row still panicked. It is reachable
  from ordinary NONMEM data: an `EVID=4` reset row written with `AMT=0` and a stale `CMT`.
  `ferx check` reported the dataset clean and `fit()` then aborted. The range rule now
  applies to every dose regardless of amount; the zero-amount exemption is kept only for the
  infusion routing rule, where `duration = AMT/RATE = 0` genuinely means nothing is
  delivered.
- **A dose into a non-default compartment is now computed in that compartment on the
  analytical engine** (#375). The closed-form dose-superposition path never read the dose's
  `CMT`: it chose the formula from the *model*, so it placed every bolus in compartment 1
  (the depot of an oral model, central of an IV one) and every infusion into central,
  whatever the data said. A bolus into an IV model's peripheral, or into an oral model's
  central compartment (an IV loading dose against an oral maintenance model), was therefore
  computed in the wrong compartment — silently, with a finite OFV and no warning, and
  disagreeing with NONMEM by up to two orders of magnitude on a 3-compartment model. Which
  answer you got depended only on whether the subject happened to carry a time-varying
  covariate, an `EVID=3/4` reset, or IOV, since those route to the event-driven walk, which
  places doses correctly. Such doses now route to that walk on every dataset, so both paths
  agree and both match NONMEM. Validated against NONMEM 7.6.0 `ADVAN1/2/3/4/11/12`
  (`tests/nonmem_dose_compartment_anchor.rs`). Per-compartment amounts in
  sdtab / `[derived]` are reported as `NaN` for these subjects rather than wrong, with the
  existing warning extended to explain why; the rerouted doses themselves are computed
  exactly (that is what the NONMEM anchors pin).
- **A steady-state dose with `CMT=0` no longer loses its accumulation on the analytical
  event-driven path** (#375). `CMT=0` is NONMEM's "default dose compartment", and every dose
  site resolves it to the model's first compartment — except the event-driven walk's
  steady-state equilibration, which bailed out early on `CMT=0` and returned an unequilibrated
  (all-zero) starting state. An `SS=1` dose written with `CMT=0` therefore produced the
  *single-dose* curve instead of the accumulated steady state whenever the subject took that
  path (a time-varying covariate, an `EVID=3/4` reset, or IOV), while the same dataset without
  those features returned the correct steady state from the superposition path — a silent
  ~30 % under-prediction on a one-compartment example, with no warning. Both the value walk and
  the gradient walk now equilibrate the default compartment like any other, matching the
  closed form `(D/V)·e^{−kt}/(1−e^{−k·II})`. Predictions change only for `SS` doses written
  with `CMT=0`. (The ODE engine bailed on `SS` with `CMT=0` for a while longer, so an
  analytical model and its explicit `[odes]` twin disagreed on that combination; #899 above
  brought the ODE engine into line and closed that gap.)
- **An infusion into a compartment the analytical model cannot deliver into is now an error,
  not a crash** (#375). A positive `RATE` into a compartment outside the model's infusable set
  — an oral model's peripheral (`CMT=3` on `two_cpt_oral`), which the `Added` entry above now
  makes work, or a `CMT` the model does not have at all — used to abort the
  process from deep inside the event-driven prediction walk whenever the subject also had a
  time-varying covariate, an `EVID=3/4` reset, or IOV. Nothing validated a *fixed* `RATE`
  against the model's topology: the data reader has no model, and the parse-time check only
  fires for a declared `D{cmt}`/`R{cmt}`. Such doses are now rejected up front, naming the
  subject, time, and the compartments the model *can* infuse — `fit()` returns an error, and
  `predict()`/`simulate()` fail with the same message, matching every other dose precondition.
  An out-of-range dose compartment (`CMT` past the end of the model's compartment list) is
  rejected the same way. This also removes a silent disagreement between the three analytical
  paths on the same dataset: the dose-superposition path used to route the infusion into the
  central compartment regardless of `CMT`, and the gradient (sensitivity) walk used to drop it
  entirely — so a fit could have differentiated a different dosing history than it predicted.
  One behaviour change worth calling out: an infusion with `CMT=0` is now **rejected**. On an
  IV model that previously fitted, since superposition delivers into central, which is what
  `CMT=0` means — so this is a deliberate tightening, not a bug fix: `CMT=0` is NONMEM's
  *default dose compartment*, well defined for a bolus but not for a zero-order input, and
  leaving it implicit hid which compartment was being infused. Write the compartment
  explicitly. A **bolus** with `CMT=0` is unchanged (every path agrees it means compartment 1),
  as is `SS` with `CMT=0` after the fix above.
- **Analytic FOCEI sensitivities for IIV on an absorption lag feeding a `first_order` forcing**
  (#880). Fixes to the rate-on onset of a built-in `first_order` (Bateman) input-rate forcing
  whose arrival is a moving boundary — a compartment lagtime (`ALAG1`/`LAGTIME`) **or** a
  per-route `lag=` (#859): (1) the exact second-order sensitivity block (`∂²f/∂η²`) was wrong —
  disagreeing in sign and magnitude with finite differences — because the onset saltation's
  curvature term dropped the forcing's own time-variation at the onset (`∂R_in/∂tad`), non-zero
  only for such decaying kernels (constant infusion and zero-order windows were unaffected);
  (2) under a time-varying covariate crossing the onset, the onset jump read its absorption-rate
  constant and pathway fraction from the dose record's covariate snapshot instead of the segment
  where the forcing actually turns on (NONMEM end-of-interval), giving a several-percent gradient
  error; and (3) an `n = 1` (Erlang-2) `transit` kernel's continuous-but-kinked onset dropped its
  curvature term. Both the shared-dose onset and the per-route onset are covered. Ordinary
  predictions and — outside the TV-covariate case — the FOCEI gradient were already correct;
  standard errors (the objective curvature) and the TV-covariate gradient now match finite
  differences.
- **Pre-flight flat-theta freeze no longer freezes an identifiable parameter with a
  coincidentally-tiny initial gradient** (#826 follow-up). The #826 guard freezes a theta
  whose outer gradient is ~0 at the initial estimate, on the premise it is unmapped. But a
  near-zero *initial* gradient is not sufficient: e.g. a joint PK-TTE fit's event-model
  hazard baseline `H0`, evaluated at `BETA = 0` where `hazard = H0` is momentarily flat in
  the coupling term (and whose ODE-path outer gradient is finite-differenced), tripped the
  guard and was frozen at its wrong initial value — biasing every other estimate
  (`joint_pktte` CL/V drifted out of tolerance). The guard now **confirms** each candidate
  with a perturbation probe: it only freezes a theta that leaves the reconverged objective
  exactly unchanged when moved (genuinely unmapped). Identifiable-but-flat-at-init thetas are
  left free, so the fit recovers them.
- **Steady-state dosing into a built-in absorption compartment with a nonlinear disposition is
  now solved accurately** (#867). For an `SS=1` dose into a `first_order` / `transit` / `igd` /
  `weibull` absorption compartment on a **nonlinear** (e.g. Michaelis–Menten) disposition that
  accumulates heavily — elimination half-life far exceeding the dosing interval `II` — the old
  50-cycle pulse-train equilibration stopped well short of the true periodic steady state and
  silently returned a trough that was too low (38–79% low in pathological cases). The periodic
  steady state is now found by an Anderson-accelerated solve of the exact one-cycle fixed point
  `u = P(u)` — a bounded handful of cycles across the clinical accumulation range — so
  `predict()` / `simulate()` / `fit()` return the correct trough (and, for fits, analytic
  sensitivities via a dual Newton derivative correction). A non-convergence warning is raised
  (through `simulate()` / `fit()`) when **no** periodic steady state exists — mean input rate ≥
  maximum elimination rate, a saturable drug dosed above its capacity — or, for an extreme model
  the bounded solve cannot converge, in place of a silently-biased trough. The **linear** case is
  exact via the closed form `u_ss = (I − M)⁻¹·b` (#835) and unchanged.
- **FREM: the analytic gradients differentiated the wrong likelihood on covariate
  pseudo-observation rows** (#251). `individual_nll` scores a `FREMTYPE > 0` row against
  the prediction `theta[i] + eta[j]` with the dedicated covariate error `EPSCOV` — but the
  sensitivity provider returned the ordinary **PK** jet for those rows, and the gradient
  assemblies read the ordinary residual variance rather than the `EPSCOV` override that
  SAEM, importance sampling and the CWRES path all already applied. Both loops were
  affected:
  - the **outer** (population) gradient, so FOCE/FOCEI were minimising one objective while
    differentiating another; and
  - the **inner** (EBE) gradient, so the empirical Bayes estimates themselves converged to
    the mode of the wrong likelihood.

  The provider now rewrites both jets for pseudo-observation rows (`f = theta[i] +
  eta[j]`, unit first derivatives, zero second derivatives — the same `{0, 1}` Jacobian
  the FOCE H-matrix already stamped in), and both gradient assemblies use the `EPSCOV`
  variance — consistently at every consumer, not only the two that motivated the fix:
  the outer jet override now also covers the ODE sensitivity provider (previously only
  the closed-form/TV-cov routes got it, so an ODE FREM subject still combined the raw
  PK jet with the `EPSCOV` variance); `method = foce`'s Sheiner–Beal `R⁰` and its σ-FD
  now take the `EPSCOV` override (previously only FOCEI's `score_core` did); the FOCEI
  `sigma_block` and `subject_eta_dx` σ-FD loops now use it too (previously they FD'd the
  *PK* variance's — zero — dependence on `EPSCOV`, so `grad[EPSCOV]` was identically zero
  under `focei` and a spurious term leaked into the other residual-error σ instead); and
  the `iiv_on_ruv` residual-eta block and the custom-magnitude direct-θ channel now both
  skip FREM rows (a pseudo-observation's likelihood has no η_ruv or magnitude dependence
  at all).

  On the warfarin FREM example the effect is large. A converged FOCEI fit goes from
  **OFV 4900.6 to 211.0**, and the importance-sampling marginal (`method = imp`) from
  **19781.1 to 211.6** — `imp` scores FREM rows correctly but centres its proposal on the
  inner-loop EBEs, so it inherited the wrong mode, the weights collapsed, and its estimate
  was meaningless.

  This is primarily a gradient fix, and most of the OFV gap is simply the fit landing
  somewhere else because the gradient that drove it there was wrong. One piece is not
  gradient-only, though: `find_ebe`'s non-IOV `h_matrix` — the Jacobian `foce_subject_nll`
  uses to build the `log|H̃|` Laplace curvature term, which **is** part of the reported
  OFV — reuses the same provider choke point this fix corrects, and that Jacobian never
  received the FREM `{0, 1}` override before (only the IOV path and the FD-Jacobian
  fallback already had it). So a non-IOV FREM subject's own curvature term was also wrong
  pre-fix, independently of the outer-gradient bug above. That the corrected FOCEI Laplace
  OFV (211.0) and the corrected 6000-sample IS marginal (211.6) — two independent
  approximations, and IS's data term does not go through `h_matrix` at all — now agree to
  under one unit is nonetheless a strong check that the new values are the right ones.

  Note the recovered covariate omegas barely move (118.71 → 118.67 for WT), because they
  are pinned by the pseudo-observations themselves. **A FREM fit could therefore look
  entirely plausible on the one diagnostic a user would naturally check, and still be badly
  wrong.**

  Latent because FREM models are conventionally fit with `method = saem`, which uses
  neither gradient — and the covariate-omega regression test runs SAEM. Fits under `focei`,
  `imp` (or now `agq` / `laplace`) were affected. **SAEM fits are unchanged.**

- **AGQ / `laplace`: the analytic outer gradient now covers the same models as
  FOCE/FOCEI** (#251). Its score previously carried a Gaussian-only residual chain,
  so five endpoint families that FOCE/FOCEI already handled analytically — **M3
  censoring, IIV-on-RUV, a custom or time-varying residual magnitude, LTBS, and
  correlated residuals (`block_sigma`)** — silently fell back to a finite-differenced
  score. They now share the same per-observation chain as FOCE/FOCEI and take the
  analytic route, so scope parity holds by construction rather than by a list that
  can drift. Under a custom residual magnitude this also **corrects** the θ gradient:
  `mult(θ)` makes the residual variance depend on θ directly, and that channel was
  not approximated before — it was missing entirely. FREM is included too — its
  pseudo-observation rows now ride the same analytic score as FOCE/FOCEI via the FREM
  fix above. TTE and categorical endpoints still take the finite-differenced score
  (neither re-solves the inner loop, so they remain fast) — they have no analytic
  chain in the `Dual2` provider at all, not merely an AGQ-side gate. `block_sigma` is
  now accepted for `method = laplace` (previously rejected at `fit()` even
  though the analytic score already carried a `corr_diag` branch for it). Under IOV,
  the scope check now excludes non-Gaussian endpoints and bounds the custom-magnitude
  axis count the same way the non-IOV gate does — previously an IOV + TTE/categorical
  subject could pass the gate and silently score only the Ω prior, dropping the hazard
  term. A per-subject runtime decline inside the analytic score (an off-diagonal
  `block_sigma` subject, or magnitude × M3-censored) now falls back to the fixed-η FD
  score for just that subject, rather than dropping the whole population onto the
  `2·n_free`-inner-resolve `reconverged_fd_gradient` fallback.

### Added
- **Per-route absorption lag — an optional `lag=` argument on every input-rate
  function** (#856) (`first_order(ka=KA, lag=L)`, `zero_order(dur=DUR, lag=L)`, …). Each
  parallel / mixed pathway can now switch on at its own delay — the immediate-release +
  delayed-release picture — instead of sharing one per-dose lagtime. The per-route lag is
  additive on top of any compartment `lagtime`/`ALAG` (a route's onset is
  `dose + lag_cmt + lag_route`); `lag=0` (or no `lag`) is bit-identical to an unlagged
  route. A model carrying a per-route lag is fit over finite differences (the analytic
  per-route onset saltation is a planned follow-up), like `weibull()` + lagtime; a
  negative lag warns (`W_NEGATIVE_LAGTIME`), a non-finite one is rejected. Steady-state
  (`SS=1`) dosing into a per-route-lagged absorption compartment is rejected
  (`E_ABSORPTION_SS_LAG`), consistent with a compartment `lagtime`/`ALAG` (#719). New example
  `examples/per_route_lag_absorption.ferx`; validated by reduction to the NONMEM-anchored
  compartment lag (`tests/per_route_lag.rs`) and by a direct NONMEM `ADVAN13 $DES` anchor
  — ferx's objective at NONMEM's optimum matches `#OBJV = −882.357` to ~1e-6
  (`tests/per_route_lag_nonmem_anchor.rs`).
- **Infusion (`RATE>0`) into a built-in absorption compartment** (#719): an
  infusion into a `transit()` / `igd()` / `weibull()` / `first_order()` absorption
  input-rate compartment is now supported on the ODE path — previously rejected
  with `E_ABSORPTION_RATE`. The dose is treated as a **zero-order source feeding
  the kernel**: its mass is released at a constant rate over the infusion window
  `T`, so `R_in` becomes the convolution `(F·amt/T)·[G(t) − G(t − T)]` of the
  kernel with the rectangle (`G` = the kernel's absorbed-fraction CDF), and the
  dose's plain `+rate` injection is suppressed. Predictions match NONMEM's native
  `ADVAN2` zero-order-into-depot behaviour and an explicit sub-dose train.
  Closed-form `pk *_transit` / `*_ig` models with an infusion reroute to their ODE
  twin automatically. Sensitivities use a finite-difference fallback (at normal
  FOCEI speed — an infusion prediction needs no equilibration). Still rejected,
  with clear codes: an infusion into a `zero_order()` window
  (`E_ABSORPTION_RATE_ZERO_ORDER`) and a steady-state infusion
  (`E_ABSORPTION_SS_INFUSION`).
- **Steady-state (`SS=1`) dosing into a built-in absorption compartment** (#719):
  an `SS=1` dose into a `transit()` / `igd()` / `weibull()` / `first_order()`
  absorption input-rate compartment is now supported on the ODE path — previously
  rejected with `E_ABSORPTION_SS`. The dose is equilibrated *through* the
  absorption kernel (the periodic pulse train is superposed as `R_in`, and the
  disposition trough is the periodic steady state, a closed form
  `(I − M)⁻¹·b` for a linear disposition), so predictions match an explicit long
  run-in of the same schedule and NONMEM's exact analytic `ADVAN2` steady state.
  Closed-form `pk *_transit` / `*_ig` models with an `SS` dose reroute to their ODE
  twin automatically. `fit()` on an SS-absorption model converges; its sensitivities
  currently use a finite-difference fallback of the (exact) prediction, so large-dataset
  fits are slower than an analytic ODE model pending an analytic dual SS-equilibration.
  Still rejected, with clearer codes: `SS` into a `zero_order()` window
  (`E_ABSORPTION_SS_ZERO_ORDER`) and `SS` combined with an absorption lagtime
  (`E_ABSORPTION_SS_LAG`).
- **Flat (zero-gradient) thetas are now frozen at start instead of killing the fit** (#826).
  A pre-flight check computes the outer gradient at the initial estimate; any non-fixed
  theta whose gradient is ≈ 0 (a parameter that never reaches the objective — typically
  unmapped, or dropped from the structural / scaling model) is held fixed at its initial
  value and reported with a `flat_parameter` warning naming it, rather than leaving a zero
  search direction that made gradient-based NLopt return `Failure` on the first evaluation
  and pin *every* parameter at its initial value. The remaining parameters now estimate
  normally.
- **Exact (analytic) inner EBE gradient for CTMM (`[markov_model]`) fits** (#759). The
  transition likelihood `−Σ log P(Δt)[s,s']`, `P = expm(Q·Δt)`, was finite-differenced
  end-to-end: every EBE step perturbed η, rebuilt `Q`, and redid a matrix exponential per
  observation gap. The intensities are now replayed over dual numbers with θ and η seeded,
  giving an exact `∂Q/∂η`, which is chained to `∂P/∂Q` through the Van Loan (1978) Fréchet
  derivative of the matrix exponential. The generator's row-sum-zero constraint is inherited
  for free (differentiation is linear, so the derivative of a valid generator is a valid
  direction). Because only one *entry* of `P` is read per gap, the adjoint form
  `⟨C, L(A,E)⟩ = ⟨L(Aᵀ,C), E⟩` lets a **single** Fréchet solve per gap serve every parameter
  at once — so the exact gradient is *cheaper* than the finite differences it replaces, not
  just more accurate. A drug-driven (time-inhomogeneous) `Q(t)` keeps the FD path: its
  likelihood is an occupancy ODE, not an `expm`, so the identity does not apply. The FOCEI
  Laplace `½log|H̃|` term and the outer θ-gradient are still FD.
- **AGQ and `laplace` now support inter-occasion variability (`[iov]`)** (#251).
  Under IOV the integral runs over the **stacked** random-effect vector
  `b = [η, κ₁ … κ_K]`, whose prior is the block-diagonal `Ω ⊕ Ω_iov^⊕K` — which is
  exactly what ferx's IOV likelihood already scores, so every AGQ formula carries
  over with `d` the stacked dimension. IOV is a change of *dimension*, not of
  method. The tensor grid is therefore `n_agq^(n_eta + K·n_kappa)` and grows with
  the occasion count `K`, so the 100 000-node cap is now enforced against the
  stacked dimension once the data is read (the error names `K`). **`method =
  laplace` is always tractable under IOV** — its grid is a single point regardless
  of `d`. Previously both methods rejected `[iov]` models outright.
- **`method = laplace` — the Laplace approximation as a first-class estimator** (#251).
  Alias `laplacian`. This is NONMEM's `$EST METHOD=1 LAPLACIAN`: the Laplace
  approximation built from the **exact** Hessian of the conditional likelihood.
  It is *not* the same estimator as `focei`, which builds its Gaussian from the
  Gauss-Newton Hessian `CᵀC + Ω⁻¹` (dropping `∂²f/∂η²`) and therefore reports a
  different OFV; `laplace` carries the curvature of the η-dependent residual
  variance that the Gauss-Newton form discards, which is why it reproduces NONMEM's
  LAPLACIAN to six significant figures on warfarin. At the default `n_agq = 1` it is a
  single node, no grid — the cheapest configuration, and on warfarin it converges
  *faster than FOCEI* (0.23 s vs 0.60 s); `n_agq > 1` turns it into adaptive
  Gauss–Hermite quadrature over the same objective. See the
  [AGQ docs page](https://ferx-nlme.github.io/ferx-core/estimation/agq.html).
- **Exact (analytic) FOCE/FOCEI gradients for lagtime models with IOV, time-varying
  covariates, or `TIME`** (#486): a closed-form model carrying an `ALAG`/`LAGTIME` used to
  fall back to finite differences the moment the subject also had IOV, a time-varying
  covariate, or a `TIME`-dependent parameter — which covers a large share of everyday oral
  popPK models. Those fits now use the exact analytic gradient on both loops: they are
  faster (FD costs one extra objective evaluation per parameter) and no longer inherit the
  finite-difference step's accuracy loss. Estimates are unchanged within convergence
  tolerance. Steady-state doses combined with a lagtime still use finite differences.
- **Exact (analytic) gradients for steady-state dosing combined with an EVID 3/4 reset**
  (#486) on the closed-form engine — previously finite differences (the ODE engine already
  had it).
- **Analytic gradients for `[scaling] y = <expr>` readouts that reference many parameters**
  (#486): a Form-C readout whose individual parameters spilled past the eight structural PK
  slots (e.g. a sigmoid-Emax readout on a 3-compartment oral model) fell back to finite
  differences on the time-varying-covariate path; it is now analytic.
- **Wider models keep the analytic gradient** (#486): the monomorphisation caps that decide
  when a model is too wide for the exact gradient were raised from 16 to 24 `θ + η` (the ODE
  and output-scaling paths). A mid-sized covariate model (5 structural θ + 6 covariate-effect
  θ + 5 η) sat at exactly the old limit, so adding one more covariate silently dropped the
  whole fit to finite differences. The closed-form event walk's cap is now coupled to the ODE
  one, closing a pre-existing gap where a 17–24-axis model took an analytic *outer* gradient
  against a finite-difference *inner* one.
- **Guarded multi-start inner EBE, on by default** (#830, `inner_restarts`,
  default `1`): escapes a multimodal individual objective, where a single
  warm-started inner optimizer can settle in the wrong basin and inflate a
  subject's objective. The classic case is saturable protein binding (a
  high-volume/low-concentration and a low-volume/high-concentration fit both
  explain a total-concentration profile). Subjects on the event-driven path
  (system resets or time-varying covariates) now re-solve the EBE on a cold start
  from Ω-scaled seeds per random effect and keep the lowest-objective mode; the
  outer warm start carries it forward, so the scan runs once per subject per fit
  (≈0 % overhead). A seed that reconverges to the same mode is not accepted, so
  unimodal subjects are unchanged; only a genuinely trapped subject moves — to
  the deeper, correct basin. On the fluconazole free/total binding model this
  recovers the NONMEM fit (OFV 734.67 vs NONMEM 734.64, versus 749.3 before). Set
  `inner_restarts = 0` to restore the previous single-start behaviour. See
  [Fit options](https://ferx-nlme.github.io/ferx-core/model-file/fit-options.html).
- **`L2` data column for correlated observation units** (#827): the reader now
  recognizes NONMEM's level-2 grouping item. Observation rows sharing an `L2`
  value within a subject are paired into one correlated unit for a `block_sigma`
  residual (e.g. the total + unbound rows of one blood draw), giving the user
  explicit control over which records the cross covariance couples instead of
  relying on co-temporal row order. See
  [Data format](https://ferx-nlme.github.io/ferx-core/data-format.html).
- **Two `block_sigma` / `L2` data diagnostics** (#830), reported by `fit()` and
  `ferx check`: `W_BLOCK_SIGMA_L2_ORDER` when a correlated residual has a
  co-temporal group that can pair more than one way and no `L2` column is present
  (the fallback pairs in CSV row order, so reordering rows changes the fit — add
  an `L2` column); and `W_L2_UNUSED` when the data has an `L2` column but the
  model declares no `block_sigma` correlation (the reserved column is inert and,
  if it was meant as a covariate, was silently dropped).
- **Continuous-time Markov model (CTMM) endpoint** (#759): a new
  `[markov_model]` block fits a discrete-state Markov process observed at
  irregular times. Declare states bound to their integer DV code
  (`states = [awake=0, asleep=1]`) and one `transition A -> B = <intensity>`
  line per allowed transition; ferx fills the generator's row-sum-zero diagonal
  and scores each consecutive observation pair with the exact transition
  matrix `P(Δt) = expm(Q·Δt)`. Time-homogeneous generators fit with FOCEI
  (default), SAEM, or IMP; intensities may carry covariates and between-subject
  random effects. An intensity may also depend on a **model state** — a drug
  concentration (`central / V`) or a PD response — making the generator
  time-inhomogeneous `Q(t) = f(state(t))` (#817); ferx then integrates the
  occupancy ODE `dP/dτ = P·Q(state(t))` (forward Kolmogorov) over each
  observation gap instead of the closed-form matrix exponential (requires an ODE
  model). Requires the `markov`
  cargo feature. See the
  [Markov models](https://ferx-nlme.github.io/ferx-core/model-file/markov-model.html)
  and [CTMM estimation](https://ferx-nlme.github.io/ferx-core/estimation/ctmm.html)
  pages. (mCTMM/DTMM and CTMM simulation are planned follow-ups.)
- **Adaptive Gaussian quadrature (`method = laplace` with `n_agq > 1`)** (#251):
  generalises Laplace. Instead of approximating each subject's
  marginal likelihood with a single Gaussian at the empirical-Bayes mode, it
  evaluates the *exact* conditional likelihood on a Gauss-Hermite grid laid
  around that mode (`[fit_options] n_agq`, default 1 node per random effect).
  `n_agq = 1` reproduces the Laplace approximation identically — it matches NONMEM
  `$EST METHOD=1 LAPLACIAN` to five significant figures on warfarin. Because it
  makes no Gaussian-residual assumption it handles non-Gaussian endpoints (TTE,
  categorical) that FOCE/FOCEI structurally cannot, and unlike SAEM/IMP its
  objective is deterministic — the OFV is bit-identical run to run. AGQ carries an
  **analytic outer gradient** (the posterior-weighted score over the quadrature
  nodes), so a converged `n_agq = 3` warfarin fit takes 0.39 s against FOCEI's
  0.29 s and NONMEM LAPLACIAN's 1.21 s, and reproduces NONMEM's estimates to 4–5
  significant figures on every parameter. Cost is `n_agq ^ n_eta` per subject per
  iteration, so it suits models with few random effects; grids over 100 000 nodes,
  out-of-range node counts, and IOV models are rejected at check time. See the
  [AGQ docs page](https://ferx-nlme.github.io/ferx-core/estimation/agq.html).
- **Restart of an interrupted run from a checkpoint** (#755): a fit now
  periodically saves a small `{model}.tmp` resume point (throttled to
  `[fit_options] checkpoint_interval_secs`, default 300 s, so short runs write
  nothing). If the process is stopped, the next run of the same model + data
  resumes from the last saved estimates instead of starting over, and the file
  is deleted on successful completion. A model/data hash check invalidates a
  stale checkpoint (the run then starts fresh). Pass the CLI flag `--clean` to
  force a fresh start, or set `checkpoint = false` to disable saving. Works
  across all estimation methods (resume is a coarse warm-restart from the saved
  population estimates, not a bit-exact optimizer-state resume).
- **Binary / logistic endpoint** (`[binary_model]`, #760): mixed-effects logistic
  regression as a first-class non-Gaussian endpoint (Phase 4, Track C). Declare a
  binary observation compartment with `cmt` and a `logit` linear predictor over
  θ/η/covariates (and the per-record `TIME` builtin); `DV ∈ {0,1}` on that CMT is
  scored with the Bernoulli likelihood `−Σ[y·log p + (1−y)·log(1−p)]`,
  `p = logit⁻¹(lp)`. Works with FOCEI (FD-Laplace), SAEM, and IMP, and supports the
  fixed-effects (`n_eta = 0`) special case — ordinary logistic regression. Validated
  exactly against base-R `glm(DV ~ X + TIME, family = binomial)` and NONMEM `F_FLAG=1`:
  ferx reproduces the glm/NONMEM coefficients and its OFV equals the glm deviance /
  NONMEM −2 log L to 5 decimals (see `examples/binary_logistic.ferx`,
  `docs/estimation/categorical.qmd`, `tests/reference/binary_logistic/`). A non-Bernoulli
  code (`DV ≥ 2`) on a binary CMT is rejected fail-loud. Ordinal / Poisson / negative
  binomial are planned follow-up slices.
- **IOV (inter-occasion variability) for the analytic absorption models** (#719):
  `pk one_cpt_transit` / `two_cpt_transit` / `one_cpt_ig` / `two_cpt_ig` now accept IOV
  (`kappa` parameters with an `iov_column` or `iov_occasion` rule). A subject carrying IOV
  is transparently rerouted to the model's exact `transit()` / `igd()` ODE twin, which
  integrates the cross-occasion dose carryover the closed-form superposition cannot express
  (#104) — so fits and predictions are correct on both the analytic and ODE engines, for IOV
  specified either via a dataset `OCC` column or a model-code `iov_occasion` rule. Previously
  rejected with a clear error; twin-less forms (a user `[odes]`/`[scaling]`/`[initial_conditions]`
  block) still are. Validated by exact analytic≡ODE-`transit()`/`igd()`-forcing equivalence at
  non-zero per-occasion κ, including multiple-dose regimens
  (`tests/transit_analytic_equivalence.rs`, `tests/ig_analytic_equivalence.rs`).
- **Left truncation (delayed entry) for clock-reset RTTE** (#740): repeated
  time-to-event models with `clock = reset` now accept a `TENTRY > 0` entry time
  instead of rejecting it. The first inter-event sojourn is conditioned on survival
  to entry in absolute time (`H(t₁) − H(TENTRY)`), then the renewal clock takes over
  for later gaps — the same delayed-entry convention already used for single-event
  and clock-forward RTTE, so `TENTRY` means one thing across every survival endpoint
  (condition on survival past entry, never restart the clock at entry). Assumes the
  time origin `t = 0` is the renewal origin of the first sojourn; coincides with the
  pure renewal form (and with clock-forward) for a memoryless exponential hazard.
- **Left truncation for RTTE _simulation_** (#740): `simulate()` now accepts
  `TENTRY > 0` for repeated events on both clocks, drawing the stream on the time
  origin conditioned on survival to entry (clock-forward seeds its conditioning clock
  at `TENTRY`; clock-reset draws its first sojourn conditional on entry, then renews
  from 0) — the simulate dual of the fit-side conditioning, so a simulated
  left-truncated stream refits under the same convention. `TENTRY = 0` stays
  byte-identical to the non-truncated draw.
- **Analytic inverse-Gaussian (IG) absorption closed form** (#790): the Freijer &
  Post inverse-Gaussian absorption model is now available as the analytic
  structural models `pk one_cpt_ig(cl, v, mat, cv2)` and
  `pk two_cpt_ig(cl, v1, q, v2, mat, cv2)` — the exponential-tilting closed form of
  the same `igd(mat, cv2)` density the ODE path uses, giving exact `Dual2`
  FOCE/FOCEI gradients that are independent of ODE-solver tolerance, and a uniform
  `pk`-line interface consistent with the analytic transit models. Supports
  absorption `lagtime`, bioavailability `f`, IIV, and time-varying covariates
  (auto-rerouted to the ODE `igd()` twin per subject); IOV / steady-state /
  infusion / non-depot doses are rejected with a clear error, as for the transit
  closed form. Outside the tilting convergence domain (`ke ≥ 1/(2·MAT·CV²)`) a plain
  model transparently falls back to its ODE twin. **Note on performance:** unlike
  the analytic *transit* models (whose stiff-ish ODE makes the closed form ~28–31×
  faster), IG's `igd()` ODE is non-stiff and cheap, so the closed form is not a
  speed win — it is ~2× slower per objective evaluation than the `igd()` forcing;
  use the ODE `igd()` forcing if raw speed matters, and this closed form when you
  want exact tolerance-free gradients or the uniform interface. Validated against
  the numerical `igd()` ODE (`tests/ig_analytic_equivalence.rs`, 1-/2-cpt) and
  directly against NONMEM `$DES` on an in-domain IG-truth dataset
  (`tests/ig_analytic_nonmem_anchor.rs`: ferx −1303.528 vs NONMEM −1303.639). See
  [examples/one_cpt_ig.ferx](https://github.com/FeRx-NLME/ferx-core/blob/main/examples/one_cpt_ig.ferx).
- **High parameter-correlation warning** (#781): a fit now emits a
  `high_correlation` warning when a THETA (fixed-effect) pair's estimate
  correlation (from the covariance matrix) has |r| ≥ 0.95 — a sign of
  over-parameterization / non-identifiability that names the specific culprits
  (complementing the aggregate `condition_number`). Emitted typed at source with
  `details` listing each `{parameter_a, parameter_b, correlation}`. See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **Inflated-RSE warning** (#781): a fit now emits an `inflated_rse` warning
  when a free THETA's relative standard error (`100 · se / |estimate|`) exceeds
  ~50% — an imprecisely estimated parameter, often a sign of
  over-parameterization. Emitted typed at source with `details` listing each
  `{parameter, estimate, se, rse_pct}`. Requires a successful covariance step
  (no SEs → no warning). See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **`simulate_with_options_diag` surfaces per-subject simulation diagnostics** (#762,
  #763): a new entry point returning `SimulationOutput { results, warnings }` — the
  simulation analogue of `FitResult.warnings`. It reports subjects handled specially
  during a run (a degenerate hazard draw, an over-large recurrent stream) instead of
  letting them look like ordinary censoring. `simulate_with_options` is unchanged (a
  thin wrapper returning just the rows), and `ferx <model> --simulate` now echoes these
  warnings alongside the fit warnings — including in the structured `warnings_structured`
  / JSON output, under a new typed `simulation` [`WarningCode`](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **Boundary-estimate warning** (#781): a fit now emits a `boundary_estimate`
  warning when a free THETA estimate is pinned to an optimizer bound (evaluated
  in the optimizer's packed/log space) — a sign of non-identifiability or a
  too-tight bound. The structured warning carries `details` listing each
  `{parameter, estimate, bound, side}`, and is emitted **typed at its source**
  (the first warning to use the native at-source path rather than string
  classification). See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **Finer covariance-step warning codes** (#781): the overloaded
  `covariance_step` warning code is split by severity into `covariance_failed`
  (Critical — no standard errors), `covariance_regularized` (Warning — SEs
  degraded but present), and `covariance_step` (Info — cost notes), so an agent
  can branch on the outcome. The failure/regularized codes carry `details` with
  `condition_number`, `min_eigenvalue`, and `n_negative_eigenvalues` when those
  were computed. See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **High ETA-shrinkage warning** (#781): a fit now emits an `eta_shrinkage`
  warning when any random-effect (ETA) shrinkage exceeds ~30% (the Savic &
  Karlsson rule of thumb) — the data poorly inform that IIV, so EBE-based
  diagnostics for it are unreliable and removing the IIV is often warranted.
  The structured warning carries `details` listing the affected ETAs and their
  shrinkage percent. See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **Warning `details` payloads for numeric diagnostics** (#781): structured
  warnings for `dw_autocorrelation`, `eps_shrinkage`, and `condition_number` now
  carry a `details` object with the value behind the message (e.g.
  `{"durbin_watson": 1.20, "iwres_lag1_autocorr": 0.40}`), sourced from the
  fit's typed fields so an agent reads the number directly instead of parsing
  prose. First increment of the at-source warning work; other codes omit
  `details` until converted. See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html#details).
- **Typed warning taxonomy** (#778): structured warnings
  (`FitResult.warnings_structured`, surfaced in the JSON output) now carry a
  typed `WarningCode` instead of a free-text category string — a stable,
  exhaustive vocabulary an agent or the R wrapper can branch on. Each code
  serializes as a fixed snake_case token (unchanged from the previous category
  strings, so JSON consumers are unaffected), and each entry gains an optional
  `details` payload for machine-readable numbers behind the message. See the
  [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **Machine-readable JSON fit output** (#777): `ferx <model> --data <csv>
  --output-format json` (or `both`) writes `{model}-fit.json` — the *complete*
  fit result (every estimate, standard error, diagnostic, per-subject record,
  and provenance field), not the curated human YAML. It carries a top-level
  `schema_version` so programmatic/agent consumers can pin, matrices serialize
  as `{rows, cols, data}` (row-major) and vectors as flat arrays, and non-finite
  floats become JSON `null`. `--output-format yaml` (the default) is unchanged.
  Library callers can get the same payload in-process via
  `FitResult::to_json_value()`. See the
  [output documentation](https://ferx-nlme.github.io/ferx-core/output.html).
- **Experimental `markov` feature — CTMM matrix-exponential foundation**
  (#759): a new default-off `markov` cargo feature adds the numerical core for
  continuous-time Markov models — transition probabilities
  `P(Δt) = expm(Q·Δt)` (nalgebra's scaling-and-squaring Padé) with exact
  Van Loan (1978) parameter gradients, plus a guarded individual CTMM
  likelihood term. This is a library-internal primitive with **no model-file
  syntax yet**; wiring it into estimation is a later phase (see
  `plans/tte-survival-markov.md`).
- **Declare IOV occasions in the model** (#756): a new `iov_occasion` key in
  `[fit_options]` derives the occasion partition from each subject's timeline
  instead of requiring a precomputed dataset column. `iov_occasion = dose`
  starts a new occasion at each administration; `iov_occasion = time(24, 48)`
  splits by time-window breakpoints. When both `iov_occasion` and `iov_column`
  are set, the model-side rule wins (with a warning). Both rules bucket
  observations and doses on the same internal event clock (correct for
  reset-stacked crossover subjects), co-timed doses share one occasion, a
  degenerate single-occasion partition errors instead of silently
  under-identifying kappa, and the derived `OCC` column is written to `sdtab`
  (#757). A `time(...)` breakpoint list must contain only interior boundaries
  (the first occasion starts at `-∞`, the last runs to `+∞`); a leading `0`
  (the `c(0, 24, 48)` habit) is rejected with a message pointing at the correct
  `time(24, 48)` form. See the
  [IOV documentation](https://ferx-nlme.github.io/ferx-core/model-file/iov.html).
- **`ferx summary` compares multiple runs** (#749): pass two or more `.fitrx`
  bundles (`ferx summary run1.fitrx run2.fitrx run3.fitrx`) to print a Markdown
  table comparing them side by side — method, convergence, OFV/AIC/BIC, ΔOFV,
  runtime, subject/observation/parameter counts, and THETA/OMEGA/SIGMA
  estimates. A single bundle still prints the detailed `psn::sumo`-style report.
- **Standalone covariance step** (#738): `run_covariance()` runs the FD-Hessian
  covariance step against an existing fit without re-fitting, mirroring
  `run_sir()`. It re-reads the model/data from the fit's recorded paths (with
  SHA-256 integrity checks, refusing stale inputs) or accepts caller-supplied
  `model`/`population`, then returns a fit with `covariance_matrix`, standard
  errors, `covariance_status`, and condition-number diagnostics refreshed. It
  reuses `fit()`'s inline covariance step at the converged point, so the result
  matches an inline `covariance = true` fit up to finite-difference noise. A
  covariance step that runs but fails (non-PD / unusable FD Hessian) is non-fatal
  — the returned fit reports `covariance_status = Failed` with a diagnostic
  warning.
- **Per-parameter estimates + gradients in the optimizer trace** (#640): when
  `optimizer_trace = true`, each CSV row now also records the full parameter
  vector (`val:<name>` columns, natural/reporting scale) for every method and
  the full gradient vector (`grad:<name>` columns, optimizer-scaled space) for
  the gradient methods (FOCE/FOCEI/GN; `NA` for SAEM). Columns are named after
  the declared parameters (`TVCL`, `ETA_CL`, `ETA_V~ETA_CL`, `PROP_ERR`) with
  `THETA1` / `OMEGA(2,1)` / `SIGMA(1)` fallbacks, and reconstruct `grad_norm`
  as `sqrt(sum(grad:<name>^2))`. This powers a per-parameter convergence view
  in the ferx-r trace UI.
- **`[data]` block column renaming** (#730, #742): rename any dataset header to
  any new name with `new-name = actual` entries (e.g. `TIME = TAFD`,
  `DV = CONC`), the ferx equivalent of NONMEM's `$INPUT TIME=TAFD`. Targets are
  not limited to canonical roles — arbitrary columns and covariates can be
  renamed too, and a column can be renamed *aside* to free its name for another
  (e.g. `ODV = dv` then `DV = lndv`). Header matching is case-insensitive,
  renamed headers are excluded from covariate auto-detection under their old
  name, and typos (absent header, duplicate target, or a target colliding with a
  surviving column) fail loudly. See
  [Data → Column mapping](https://ferx-nlme.github.io/ferx-core/model-file/data.html).

- **Adaptive (feedback) dosing now supports time-varying covariates** (#700): the
  reactive driver recomputes each subject's PK per event/segment from the covariate
  active in that segment (the same NONMEM end-of-interval convention `predict()` /
  `simulate()` use), instead of freezing it at the `t=0` snapshot. A covariate that
  changes over the horizon — e.g. declining renal function driving clearance under
  TDM titration — now correctly drives the predictions, the monitored signal, and
  every dose decision, and the frozen-replay verifier validates the per-event
  bookkeeping. Models whose PK reads the `TIME` built-in are covered too (previously
  also silently frozen at `TIME=0`). The `auc_target_attainment` metric is not yet
  available for time-varying-covariate subjects and is rejected with a typed error
  rather than reported from a frozen snapshot.
- **Adaptive (feedback) dosing now supports inter-occasion variability (IOV)** (#701):
  the reactive driver draws a fresh occasion `kappa` per **decision window** (occasion =
  decision index) and threads it through the per-event PK — instead of silently holding
  every kappa at zero — so occasion-to-occasion shifts in CL/V correctly drive the
  predictions, the monitored signal, and every reactive dose decision, and the
  frozen-replay verifier validates the per-occasion bookkeeping. Composes with the #700
  time-varying-covariate path (a model with both is per-event correct in each). `kappa`
  is drawn on a dedicated per-(subject, replicate) substream, so a non-IOV run is
  byte-identical to before and enabling a `Dv` monitor never shifts the draws. The
  `auc_target_attainment` metric is not yet available for IOV subjects and is rejected
  with a typed error rather than reported from a κ-frozen snapshot. A non-ascending or
  duplicated adaptive `decision_times` schedule (programmatic `simulate_adaptive`) is now
  rejected with a typed error, matching the declarative `[adaptive_dosing]` path — an
  out-of-order schedule would otherwise mis-map a record to the wrong occasion.
- **`estimation:` block in `{model}-fit.yaml` now splits wall time by stage** (#713):
  a `{method}_wall_time_secs` entry (e.g. `focei_wall_time_secs`, `imp_wall_time_secs`)
  is reported for each stage of `method`/`methods`, plus a `covariance_wall_time_secs`
  for the post-estimation FD-Hessian / SIR-fallback step, alongside the existing
  `wall_time_secs` total. Also carried on `FitResult.method_wall_times_secs` /
  `FitResult.covariance_wall_time_secs` and round-trips through `.fitrx` bundles.
- **Repeated time-to-event (RTTE) models** (Phase 3): `[event_model]` accepts
  `type = rtte` for endpoints with multiple events per subject, with
  `clock = forward` (Andersen–Gill total time, the default) or `clock = reset`
  (gap time / renewal). Clock-forward integrates the cumulative hazard once across
  each subject's records (`Σ_k log h(t_k) − H(T)`); clock-reset restarts the hazard
  clock at each event (`Σ_k log h(Δ_k) − Σ_k H(Δ_k)` over inter-event gaps). Both
  use the analytic hazard families. Because Laplace/FOCEI can severely
  underestimate the frailty variance ω² for RTTE at low event rates (Karlsson et
  al. 2009), fitting RTTE under a Laplace-based method with a frailty now emits a
  warning recommending `method = saem` or `method = imp` (fired only for a frailty
  model — `n_eta > 0` — whose chain's final estimating stage is Laplace-based, so a
  warm-start like `[focei, saem]` does not false-warn). **`simulate()` draws a
  recurrent event stream** per subject up to the administrative `[simulation]
  horizon` — clock-forward via conditional inverse-CDF draws (each event conditioned
  on survival past the previous), clock-reset via fresh gap draws — for the analytic
  hazard families. Unsupported configurations (interval-censored, out-of-order,
  non-finite, or — for simulation — left-truncated, multiple RTTE causes, an RTTE
  cause mixed with a competing single-event cause, or EVID=3/4 resets) are rejected
  with a clear error rather than silently producing a wrong answer;
  `predict_survival()` reports first-event survival for RTTE (use its `cum_hazard`
  field for the recurrent `E[N(t)]`).
- **`two_cpt_transit` now supports time-varying covariates and `TIME`-dependent parameters** (#724):
  a 2-cpt transit model whose disposition switches mid-profile is transparently routed to an exact
  ODE `transit()` twin (`central` + `periph`), exactly as `one_cpt_transit` already was — instead
  of being rejected. This removes the 1-cpt/2-cpt asymmetry. IOV, steady-state, infusion, and reset
  doses on a transit closed form remain rejected (use an explicit ODE `transit()` model for those).
  A non-depot (`CMT≠1`) dose on either transit closed form is now also rejected with a clear error
  rather than silently mis-predicted — the closed form transits every dose into central regardless
  of compartment, so it would disagree with the ODE twin (which honours the dose compartment).
- **Optional `[data]` model-file block** (#690): a model can now declare
  `path = ...` to point at its own dataset (`$DATA` equivalent), so `ferx
  model.ferx`, `ferx check model.ferx`, and the public `fit_from_files()`
  (now `data_path: Option<&str>`) work without an explicit data path. An
  explicit CLI `--data`/R `data =`/`fit_from_files()` path still overrides
  the model's `[data]` block, with a warning when the two differ (path
  equivalence, not textual equality, so a dir-joined model path and a raw
  external path to the same file don't false-positive).
- **`-h`/`--help` flag for the `ferx` CLI** (#688): `ferx --help`, `ferx check --help`,
  and `ferx summary --help` now print usage to stdout and exit 0, matching standard
  CLI convention (previously only printed on no-args/bad-args, to stderr, exit 1).
- **`ferx check` warns when `[scaling] obs_scale` references the same individual
  parameter bound to a built-in `pk <model>(...)` block's `v`/`v1` role** (#712):
  the closed-form kernel already divides by that volume internally to produce
  concentration, so an `obs_scale` referencing it divides by it a second
  time — a common mistake when translating an `ode(...)` model (where that
  division is required) to an equivalent closed-form `pk` block (where it
  already happens). Not rejected — `obs_scale` referencing an individual
  parameter is also a supported feature for an intentional additional
  transform — just flagged, since ferx can't tell intent from mistake.
  `docs/model-file/scaling.qmd` now documents the raw-output convention per
  structural-model type.

### Changed
- **Flip-flop transit / inverse-Gaussian models with `lagtime` / `f` now auto-route
  to their ODE twin** (#735) instead of being rejected. The analytic
  `pk one_cpt_transit` / `two_cpt_transit` / `one_cpt_ig` / `two_cpt_ig` closed forms
  clamp to an identically-zero profile outside their tilting-convergence domain (the
  flip-flop regime) and are transparently rerouted to the equivalent ODE `transit()` /
  `igd()` model — but previously only when the model carried no `lagtime=` / `f=`
  mapping. Those mappings now carry into the generated twin (via reserved-name
  individual parameters), so a flip-flop model with absorption lag or bioavailability —
  including a `TIME` / time-varying-covariate model that *requires* the twin — now fits
  and predicts correctly instead of erroring or returning zero. Validated by
  closed-form↔ODE equivalence tests (`tests/transit_analytic_equivalence.rs`,
  `tests/ig_analytic_equivalence.rs`) for lag, f, and both. A guard declines the twin
  (keeping the model closed-form, and rejected up front if flip-flop) whenever building it
  would misbehave: a parameter name that *shadows* a reserved F/lagtime slot it was not
  mapped to (which would silently apply an extra F/lag), or an `f=`/`lagtime=` parameter
  whose name *collides* with a disposition slot (e.g. a bioavailability parameter named
  `V1`, which would otherwise make the twin fail to build). User `[odes]` / `[scaling]` /
  `[initial_conditions]` forms remain twin-less (no unique desugar) and are still rejected
  up front in the flip-flop regime.
- **Default thread count capped at 8** (#707): when `threads` is unset (or
  `0`/`auto`) — via `[fit_options] threads`, the CLI `--threads` flag, or the R
  binding — the engine now defaults to `available cores - 1` (floored at 1),
  capped at 8, instead of one worker per logical core. Most fits gain little from
  scaling past a handful of cores, and not all cores are equal on
  asymmetric platforms (e.g. Apple Silicon's E-cores). An explicit
  `threads = N` / `--threads N` still pins the exact count requested.
- **CLI default output no longer writes a separate `{model}-timing.txt` file** (#704):
  the estimation step's wall-clock time and thread count now live under a new
  `estimation:` section in `{model}-fit.yaml` (narrower in scope than the old
  file, which also covered model parsing and data loading), alongside a new
  `environment:` section (OS, CPU architecture, whether running in Docker, OS
  username, ferx version) for troubleshooting and reproducibility. Both are also carried on
  `FitResult.environment` and round-trip through `.fitrx` bundles.
- **`simulate()` now samples inter-occasion variability (kappa)** (#723): simulating
  an IOV model draws an independent `kappa ~ N(0, Omega_IOV)` for each occasion
  (matching NONMEM `$SIM`), instead of holding every kappa at zero. Simulated /
  VPC datasets from IOV models now carry the between-occasion spread the model
  parameterizes; previously they silently under-dispersed relative to the fitted
  model. Non-IOV models are unaffected (bit-identical output).
- **The adaptive frozen-replay verifier now independently validates its snapshots** (#748):
  the default-on safety net for `simulate_adaptive` / `simulate_adaptive_from_spec` reused the
  same precomputed per-occasion / per-event PK snapshots the reactive driver did, so it checked
  that the two *consumed* them identically but never that they were *correct* — a mis-built
  snapshot (a wrong covariate, occasion, or `kappa` fed to a parameter) was applied by both sides
  and passed the replay bit-exact. The verifier now re-derives each snapshot from the run's
  primitives via the canonical helpers and bit-asserts it against what the driver was handed, so a
  build that mis-resolves a covariate or occasion (the class fixed in #732 / #739) fails loudly for
  every model instead of slipping past the replay. No effect on correct models.

### Fixed
- **AGQ / `laplace` fits no longer report "did not converge" while sitting on a settled
  OFV** (#251). The gradient-based outer optimizer set NLopt's stopping tolerances to
  `1e-12`, which FOCE/FOCEI can reach (their analytic gradient is exact to ~1e-11) but
  AGQ cannot: AGQ's gradient is exact yet *finite-difference-limited* (the grid-response
  term and the posterior Hessian are central differences), so it carries a noise floor.
  L-BFGS ground on past the point where the objective had settled, until its line search
  failed on a noise-dominated direction and NLopt returned a bare failure — reporting
  "not converged" for a result that had been flat to eight significant figures. AGQ now
  stops on the reachable objective-change criterion (`outer_ftol` / `outer_xtol`), which
  both fixes the flag and makes the fits **faster** (AGQ+IOV on warfarin: 14.5 s → 8.2 s;
  Laplace+IOV: 2.2 s → 1.3 s), with identical estimates. FOCE/FOCEI are unchanged.
- **Wrong analytic gradient for an observation sampled exactly on a moving dose boundary**
  (#486) — a modeled infusion end, or a lagged dose arrival. With a modeled `RATE=-1`/`-2` dose the infusion window end moves with the
  estimated `D{cmt}`/`R{cmt}`. An observation whose time coincided with that end had its
  gradient taken *along the moving boundary* rather than at the sample's own fixed clock
  time, adding a spurious term: on a 1-cpt fixture it returned `−1.9`, which is not even a
  valid subgradient in general. The closed-form walk now steps the read-out state back across
  the zero-length window `end(D) − t_obs` with the infusion still running, recovering the
  derivative at the sample's own time. Sampling at the end of an infusion is a normal design,
  so this is worth knowing about even though the exact coincidence is transient (`D` is
  estimated, so it only sweeps past a fixed sample time momentarily).

  The prediction is genuinely **kinked** in `D` at that point — just above the coincidence the
  infusion is still running at the sample, just below it the dose has finished and a decay term
  appears — so no two-sided derivative exists there (the one-sided slopes are `−8.6` and
  `+2.3`, and a central finite difference returns their average, `−3.2`). ferx returns the
  **one-sided** derivative — specifically, the derivative of the branch ferx's own event
  ordering already uses to define the *value* there (an infusion at its end is still
  contributing; a dose at its arrival has landed). That is the same convention its ODE engine's
  jump/saltation sensitivities already used, and the two engines now return the same number.

  The same correction applies to an observation landing on a **lagged dose arrival**, where it
  matters more than it looks: on an oral model the prediction's *value* at that instant is zero
  (the depot bolus has only just landed, so the central compartment is still empty) while its
  derivative is not — the closed form previously reported a derivative of zero, which looks
  innocuous and is wrong.

  Note the deliberate consequence: at *exactly* such a coincidence an analytic gradient and a
  finite-difference gradient legitimately disagree — FD averages across a branch switch the
  model does not make there. That is a property of a kinked model, not a defect.
- **Multi-start (`n_starts`) no longer returns a diverged run as the "best" fit**
  (#830): the start selection preferred any `converged` run over an unconverged
  one *before* comparing OFVs, so a start that diverged — driving the residual
  covariance indefinite and reporting a ~1e20 sentinel OFV while still flagged
  `converged` — outranked a valid but unconverged start (including the exact-inits
  start 0). Multi-start could therefore return a worthless fit with an enormous
  OFV. Validity (a finite OFV below a divergence-large threshold) is now the
  primary ranking key, so a valid run always wins; converged-vs-OFV ordering is
  unchanged within a validity class.
- **`block_sigma` correlated residuals no longer collapse the objective when a
  subject has two samples at the same time** (#827): with a `block_sigma` +
  covariate-selected / per-CMT error model (the free-vs-total assay pattern),
  replicate assays at one `TIME` were cross-correlated all-to-all, making the
  dense residual `R` indefinite so the FOCEI objective returned the invalid
  sentinel and the optimizer was repelled from the correct (correlated) optimum.
  Rows are now paired into disjoint correlated units — by the new `L2` column
  when present, otherwise one-to-one in co-temporal row order — keeping `R`
  positive-definite. On the fluconazole
  2-cpt binding model this recovers the NONMEM fit (OFV 742 vs NONMEM 734.6,
  previously stuck ~140 higher with a collapsed peripheral Q).
- **`block_sigma` cross covariances within one `L2` group are now correlated
  all-to-all** (#830): an explicit `L2` group is the user's declared correlated
  unit, so a genuine block of 3+ distinct endpoints (e.g. parent + two
  metabolites, each pair correlated) keeps its full cross-covariance structure
  instead of only one greedy pair. The disjoint one-to-one pairing that keeps
  co-temporal replicates positive-definite still applies to the implicit
  `(time, occasion)` fallback, where replicate rows cannot be told apart.
- **Float-formatted `L2` ids are no longer silently ungrouped** (#830):
  pandas/R exports float-format the whole `L2` column (`"10.0"`) when any row is
  blank; the reader now accepts integer and float-formatted ids, so the user's
  `block_sigma` grouping is honoured instead of every record falling back to
  ungrouped `(time, occasion)` pairing.
- **`block_sigma` cross derivative is no longer dropped when a prediction hits
  zero** (#830): the observation pairing is now decided from the loadings'
  sigma-slot structure rather than the value covariance, so a pure proportional
  paired endpoint whose prediction is momentarily `f = 0` keeps its (nonzero)
  slope cross term in `∂R/∂f`. This also stops the pairing from flickering as a
  prediction crosses zero between iterations.
- **Standalone covariance step (`run_covariance`) now reproduces the inline
  covariance bit-for-bit for FOCE/FOCEI fits**: running the covariance step after a
  fit (`covariance = false` then `run_covariance`, e.g. the R wrapper's standalone
  SE step) previously rebuilt the parameters by re-decomposing the reported `omega`
  (`chol(L·Lᵀ) ≠ L` to machine precision). The resulting `Ω⁻¹` differed slightly
  from the one the inline step used, which the finite-difference Hessian amplified —
  up to ~10% on an ill-conditioned variance (e.g. a warfarin ω²(KA) with ~115%
  RSE), giving standard errors that disagreed with an equivalent
  `covariance = true` fit. The step now reuses the optimizer's exact packed vector
  when the fit carries one, so the covariance matrix and SEs match exactly — for
  every in-memory packed-Cholesky-space optimizer (NLopt, BFGS, trust region,
  Gauss-Newton). Fits reloaded from `.fitrx`, or from SAEM/importance-sampling/Bayes
  (which rebuild `omega` from the reported matrix on both paths), are unaffected —
  they already agreed. Surfaced during the #816 review.
- **Closed-form transit/IG absorption under IOV / time-varying covariates now honors
  call-time ODE tolerances and converges its EBEs correctly** (#814, #719 follow-up): a
  `one_cpt_transit` / `two_cpt_transit` / `one_cpt_ig` / `two_cpt_ig` model routes its IOV,
  time-varying-covariate, and `TIME`-switch subjects to an internally generated ODE "twin".
  Three fixes to that reroute: (1) a call-time `ode_reltol` / `ode_abstol` / `ode_max_steps`
  (e.g. from `ferx_fit(settings = …)`) now reaches the twin's solver — previously the twin
  silently integrated at its parse-default tolerance, ignoring the requested accuracy for the
  whole fit/predict; (2) the inner EBE loop's ODE gradient-noise convergence stop is now
  enabled for those rerouted subjects, so an individual estimate that dropped to the
  finite-difference inner gradient no longer risks running to the iteration cap and returning
  an under-converged EBE; (3) when such a subject falls back to the FD inner gradient, the
  emitted diagnostic now names the precise twin-ODE reason (e.g. "steady-state dose + built-in
  absorption forcing") instead of a generic "outside IOV analytic scope". The converged
  population objective is unchanged (still NONMEM-anchored by the #719 transit+IOV cross-check).
  Regression tests in `types.rs` / `estimation/inner_optimizer.rs`.
- **Inner EBE line search no longer aborts on a non-finite objective** (#719
  follow-up): the FOCEI inner-loop backtracking line search
  (`estimation/inner_optimizer.rs`) could panic with `clamp(NaN, NaN)` (a process
  `SIGABRT`) when a trial η drove the objective non-finite — e.g. an absorption
  closed form (transit/IG) evaluated at a step outside its convergence region, or a
  blown-up BFGS direction giving `dg = ±inf`. The quadratic step-length safeguard
  then produced `inf/inf = NaN`, which poisoned the step length and crashed the
  next `clamp`. The search now rejects non-finite trials (falling back to plain
  halving) and non-finite directions up front, degrading gracefully to "no step"
  instead of crashing — surfaced while building the transit multi-dose + covariate
  NONMEM anchor. Regression tests in `estimation/inner_optimizer.rs`.
- **Declining-hazard (`γ < 0`) Gompertz median/mean survival diagnostics** (#805):
  `median_survival` returned `NaN` for *every* `γ < 0` Gompertz, even when a finite median
  genuinely exists — the closed form generalizes to `γ < 0`, so the reported TTE median is
  now finite whenever the cure fraction `S(∞) = e^{−α·e^{loghr}/|γ|} < 0.5` (and stays
  `NaN` when `S(∞) ≥ 0.5`, where more than half never event and the median is undefined).
  `mean_survival` now returns `NaN` for any `γ < 0` Gompertz via an explicit guard: its
  mean is genuinely infinite (a positive cure fraction leaves a non-decaying survival
  tail), which the previous code reported correctly only by coincidence through the `NaN`
  median. Diagnostic/reporting only (`predict_survival` summaries); does not touch
  estimation or simulation numerics. Same `γ < 0` boundary fixed for the samplers in
  #803 / #804.
- **Declining-hazard (`γ < 0`) Gompertz simulation no longer censors every event**
  (#803): the analytic inverse-CDF event-time samplers guarded the Gompertz draw with
  `inner ≤ 1`, which fires for *every* `u ∈ (0,1)` when the shape `γ < 0`, so a
  declining-hazard Gompertz simulated an empty / all-censored stream even though `fit()`
  scored the same `γ < 0` with finite density — simulate was not the inverse of fit for
  this family, breaking simulate→fit round-trips (VPC/SBC). A `γ < 0` Gompertz has a
  finite limiting cumulative hazard `H(∞) = −α·e^{loghr}/γ`, i.e. a genuine cure fraction
  `S(∞) = e^{−H(∞)}`; the guard now rejects only that true cure fraction (`inner ≤ 0`), so
  a draw above it produces a valid finite event. Affects single-event TTE, clock-forward
  RTTE, and the clock-reset first sojourn for any `γ < 0`.
- **A dose landing exactly on a subject's last observation is now applied before that
  observation is read (post-dose) on the constant-parameter ODE engine, fixing a
  false rejection by the adaptive frozen-replay verifier** (#731). The engine applied
  doses only at each integration segment's left boundary and treated the timeline's
  final break as an endpoint only, so a dose coinciding with the last observation was
  dropped and that observation read the pre-dose state — disagreeing with the
  analytical engine, the reactive adaptive driver, and NONMEM (all post-dose). This
  surfaced as the default-on adaptive frozen-replay verifier rejecting a valid
  constant-covariate run whose final dosing decision coincided with the last sample.
  Interior breaks were already handled post-dose. The same terminal-break fix is
  applied to the two sibling break-walking paths so a terminal dose is read post-dose
  consistently everywhere: the dedicated dense state solve (`ode_dense_solve_states`),
  keeping the joint PK-TTE hazard (#570) consistent between its shared one-solve and
  two-solve paths when an event time coincides with a dose at the subject's last time
  point; and the per-compartment states path (`ode_predictions_with_states`), so the
  post-fit sdtab IPRED and compartment states agree with the fitted IPRED in that case.
- **Adaptive/feedback dosing now runs the same dose-precondition guards as the
  static paths, instead of silently mis-delivering a dose** (#721). The reactive
  entry points (`simulate_adaptive()` / `simulate_adaptive_from_spec()`) skipped the
  modeled-`RATE` (#324), analytic-absorption closed-form, and built-in-absorption
  (#588) checks that `simulate()` / `predict()` / `fit()` all run before integrating,
  so a feedback-dosed model with malformed built-in-absorption pathway fractions or an
  out-of-domain absorption parameter simulated with a silently wrong absorbed dose.
  The guards now run at the adaptive chokepoint and fail with the same typed error
  before any decision is taken.
- **A twin-less transit / inverse-Gaussian absorption fit no longer silently
  degenerates a subject whose fitted random effects reach the flip-flop regime**
  (#785). An analytic `one_cpt_transit`/`two_cpt_transit` (or `one_cpt_ig`/
  `two_cpt_ig`) model carrying a `lagtime` / `f` / user-`[odes]` mapping declines the
  ODE-twin desugar, and was only checked for the flip-flop regime at typical (η = 0)
  values at fit start. A subject whose empirical-Bayes estimate drove `ke = CL/V`
  past the tilting abscissa still hit the closed form's identically-zero profile,
  silently collapsing that subject's likelihood contribution. The fit now emits a
  typed `flip_flop` warning naming the affected subject(s) and pointing at the ODE
  `transit()`/`igd()` forcing form (which reroutes per subject at the actual η). See
  the [warnings documentation](https://ferx-nlme.github.io/ferx-core/warnings.html).
- **`simulate_with_uncertainty` no longer panics when a parameter draw enters the
  flip-flop regime** (#786). For a twin-less transit / IG closed form whose point
  estimate is in-domain, a sampled uncertainty draw that crossed the flip-flop
  boundary previously aborted the *entire* simulation via a panic. Such draws are
  now skipped so the remaining draws still yield results (the run no longer panics;
  this aggregated-uncertainty entry point returns only the rows, so a skipped draw is
  not surfaced as a warning — use `simulate_with_options` when a skip must be
  visible). The single-shot `predict()` / `simulate()` panic paths are unchanged.
- **A time-to-event hazard that references an inter-occasion (IOV) `kappa` by name
  is now rejected at parse instead of silently using zero** (#770). A hazard is
  evaluated once per subject with no occasion context, so an IOV `kappa` has no
  well-defined value there. Referencing one *through* an `[individual_parameters]`
  value was already rejected (#442); a hazard expression that names a `kappa`
  **directly** (e.g. `scale = TVLAMBDA * exp(KAPPA_CL)`) previously fell back to a
  leniently-read `0.0` covariate, silently dropping the IOV term. It now fails
  loud, naming the offending random effect — write the hazard in terms of θ/η, or
  reference an IOV-free parameter.
- **A degenerate hazard draw in simulation no longer vanishes silently, and a pathological
  RTTE hazard no longer aborts the whole run** (#762, #763). When an analytic hazard's
  effective rate degenerates (non-positive / non-finite), the affected subject is censored
  with no event — previously indistinguishable from ordinary administrative censoring; the
  `simulate_with_options_diag` path now names it in a `W_TTE_DEGENERATE_HAZARD` warning. An
  RTTE hazard so extreme it would fire more than a million times over the window is now
  skipped (censored) with a `W_RTTE_DEGENERATE` warning and the run continues for the rest
  of the population, instead of panicking the entire `simulate()` call — and without first
  materialising ~1e6 rows. The `simulate()` / `simulate_with_seed()` entry points apply the
  same per-subject handling (no panic) but return only the rows.
- **A time-varying covariate on a survival hazard is now a hard error instead of a
  silently frozen baseline value** (#741). A `[event_model]` hazard that references a
  covariate whose value changes within a subject was evaluated at the covariate's
  baseline — the analytic hazard families take no time argument, and the joint PK-TTE
  ODE hazard integrates with the PK parameters frozen at `t=0` — so the fit or
  simulation silently used the wrong hazard across every TTE / RTTE / competing-risks /
  joint-PK-TTE endpoint. `fit()` now rejects it (and `predict()` / `simulate()` panic),
  naming the covariate and the subject. A time-varying covariate the hazard does *not*
  reference — e.g. one used only by a shared PK model in a frailty-only joint fit — is
  unaffected. Hold the covariate constant within each subject for now.
- **An `[initial_conditions]` covariate that matches no data column now fails the
  fit loudly instead of silently dropping the baseline** (#765). A covariate named
  only inside an init expression (e.g. `init(central) = CONC0 * V`) was never
  registered as a required data column, so with a `[covariates]` block it was never
  read, and under auto-detect a case mismatch (`CONC0` vs a `conc0` header) resolved
  to `0` — zeroing the initial amount with no diagnostic (identical OFV with and
  without the block). Init-expression covariates are now registered like every other
  model covariate, so a missing or miscased name raises `E_MISSING_COVARIATE` listing
  the available columns. Rename the column in `[data]` (`CONC0 = conc0`) or match the
  header's case in the expression.
- **A dataset with dose rows but no `AMT` column is now a hard error instead of a
  silent bad fit** (#753). When the amount column is named something other than
  `AMT` (e.g. a NONMEM export using `DOSE`), every dose parsed with amount 0, so no
  drug entered the system, the objective was flat, and the fit "converged" with
  every parameter pinned at its initial estimate. The reader now rejects such data
  with `E_DOSE_NO_AMT`, naming the fix (rename the column to `AMT`). A companion
  warning `W_ALL_DOSES_ZERO` fires when an `AMT` column *is* present but every dose
  amount is 0 (e.g. a mis-scaled column).
- **Loading a `.fitrx` bundle whose data has two subjects sharing an ID no longer
  fails with a spurious `corrupt or missing entry` error.** `load_fit` (and thus
  `ferx summary`) matched `predictions.csv` rows to subjects by ID, so when a
  dataset reuses an ID across subjects (e.g. an ID repeated across studies or a
  reset-split subject), every duplicate's rows were routed to one subject and the
  other was left with zero rows — tripping the `n_obs` consistency check. Rows are
  now assigned positionally in `ebes.csv` subject order (which the writer already
  guarantees), with the row ID kept as an ordering cross-check.
- **A forward reference in `[individual_parameters]` is now a parse error instead of a
  silent zero** (#710). A statement that referenced a name declared *later* in the same
  block (e.g. `CL = ... * exp(IMAX*...)` with `IMAX` defined below it) previously resolved
  the not-yet-defined name to `0.0` — collapsing the formula (`exp(0)=1`) with no diagnostic
  from `ferx check` or at fit time. Such an out-of-order reference is now rejected loudly,
  naming the offending variable; reorder the block so each name is declared before it is
  used. This mirrors the existing `[odes]` undefined-reference guard (#314).
- **`ADDL` on a coded `RATE=-1`/`-2` dose no longer collapses to boluses** (#722):
  additional doses expanded from a modeled-rate (`RATE=-1`/`R{cmt}`) or
  modeled-duration (`RATE=-2`/`D{cmt}`) record now stay modeled infusions like
  the first dose, instead of silently becoming instantaneous boluses. Previously
  a regimen such as `AMT=100, RATE=-2, D1=2, ADDL=5, II=24` fitted (and predicted/
  simulated) as one modeled infusion followed by five boluses, with no warning.
- **`SS=2` steady-state dose records are now rejected instead of silently run as
  `SS=1`** (#729): NONMEM `SS=2` (superimpose the steady state of a regimen on top
  of the compartment's existing amounts, without resetting) was collapsed to the
  same internal `ss = true` flag as `SS=1` (reset then equilibrate) — every
  `SS >= 0.5` cell became `SS=1` — so an `SS=2` dataset fitted, predicted, and
  simulated with the wrong (reset) initial conditions and no warning. The data
  reader now accepts only `SS=0`/`SS=1` and rejects `SS=2` (and any other code)
  with a clear message. Full `SS=2` support is tracked in #694.
- **An *unmapped* per-compartment `F{cmt}`/`ALAG{cmt}` on an analytical `pk` model is now a clear
  error** (#725): these are ODE-only dose attributes. Naming an analytical individual parameter
  `F1`/`ALAG1` (or the `LAGTIME1` alias) *without* binding it to the model's single dose route used
  to drop its value into an unused slot — so effective bioavailability stayed 1 / lag stayed 0 with
  no effect (a footgun when porting a NONMEM `$PK` that sets `F1`/`ALAG1`). The parser now rejects
  that silent no-op, pointing at the bare `f=`/`lagtime=` mapping (e.g. `f=F1`) or an `ode(...)`
  model. A parameter that *is* correctly mapped (`pk(..., f=F1)`) is unaffected — its value was, and
  remains, applied as bioavailability/lag.
- **Flip-flop transit models now evaluate correctly instead of returning a zero
  profile** (#733): when a `pk one_cpt_transit(...)` / `two_cpt_transit(...)` model's
  individual parameters put the disposition rate at or above the transit rate
  (`ke ≥ KTR`, or `α ≥ KTR` for 2-cpt — the flip-flop regime of a slow-absorption
  depot), the exponential-tilting closed form is outside its convergence domain and
  clamped the prediction *and its gradient* to `0`, silently degenerating a
  proportional-error objective. `predict()`, `simulate()`, `fit()` and the
  diagnostics now route such a model — per evaluation — to its exact ODE `transit()`
  twin, which is valid in that regime (matched to a NONMEM ADVAN13 transit
  simulation to ~1e-4); a twin-carrying flip-flop model gets an informational
  `W_TRANSIT_FLIP_FLOP` heads-up. A flip-flop model that carries a `lagtime`,
  bioavailability `f`, or a user `[odes]` / `[scaling]` / `[initial_conditions]` block
  has **no** ODE twin to route to, so
  rather than silently returning a zero profile that degenerates the objective it is
  now **rejected with a hard error** (`fit()` returns `Err`, `predict()`/`simulate()`
  panic, `ferx check` reports `E_TRANSIT_FLIP_FLOP`) — consistent with the other
  unsupported-transit rejects. Rewrite such a model as an explicit ODE `transit()`
  model, or adjust the MTT / CL starting estimates.
- **Fits are now reproducible regardless of the worker-thread count** (#703). The FOCE/FOCEI,
  SAEM, and importance-sampling objectives summed the per-subject log-likelihood with a parallel
  reduction whose grouping depended on the number of rayon threads; because floating-point
  addition is not associative, the objective (and, in non-converged runs, the final OFV and
  estimates) differed between e.g. 4 and 15 threads. The per-subject contributions are now summed
  in a fixed subject order, so a given fit returns bit-identical results at any thread count.
- **CLI flags in `--flag=value` form are no longer silently ignored** (#693):
  `--data`, `--output`, `--threads` (and any other value-taking flag) now accept
  `=` the same as a space, e.g. `ferx model.ferx --data=d.csv --threads=4`.
- **TTE non-monotone-hazard guard now tracks the ODE solver tolerance** (#618). For a
  drug-driven `[odes]` `hazard =` expression (no `h >= 0` constraint), the cumulative-
  hazard monotonicity check rejected a negative increment `H(b) < H(a)` only past a fixed
  `1e-3*|H|` round-off floor - 10x looser than the solver's default `reltol` (1e-4) and
  growing without bound as `H` accumulates, so a genuinely negative step up to ~0.1% of a
  large accumulated `H` slipped through as round-off (admitting `S = exp(-ΔH) > 1` and
  biasing the optimizer toward the negative-hazard region). The floor is now tied to the
  model's *configured* `ode_reltol`/`ode_abstol` (`abstol + reltol*|H|`, mirroring the
  integrator's own per-step monotonicity tolerance), and the analytic closed-form path
  uses a tight fixed floor. Such a step now correctly folds into the `1e20` sentinel,
  while legitimate solver round-off on a flat/slow `H` stays finite.
- **Built-in absorption pathway-fraction validation now covers `simulate()` and
  `predict()`** (#588): a multi-pathway model with malformed fractions — a bare term
  alongside a fractioned one, a lone `FR*fn(...)`, a fraction outside `(0, 1]`, or
  fractions not summing to 1 — was rejected by `fit()` / `ferx check` but could be
  simulated or predicted with silently wrong dose delivery. The data-independent
  *structural* rules now fire at parse time (so every entry point rejects them), and
  the typical-value *value* checks are enforced on the `predict`/`simulate` paths too.
- **Adaptive dosing now rejects models/data it cannot faithfully simulate**, instead
  of silently returning wrong results (#391): a model with inter-occasion variability
  (`kappa` / IOV) or a stochastic (`[diffusion]` / SDE) term, or a subject with a
  time-varying covariate or a system reset (EVID=3/4), now raises a typed error rather
  than being run with kappas held at zero, covariates frozen at their baseline value,
  process noise dropped, or the reset ignored.

## [0.2.0] - 2026-07-03

### Added
- **New `ferx summary <run.fitrx>` CLI subcommand** (#684): prints a concise,
  `psn::sumo`-style summary (parameter estimates with SE / %RSE, OMEGA / SIGMA with
  CV%, condition number, shrinkage, and run info) from a saved `.fitrx` bundle to
  stdout — no re-fitting or data required.
- **Log-transform-both-sides (LTBS) combined with IOV now gets an analytic *outer* (θ/Ω/σ)
  gradient** (#486): the closed-form IOV sensitivity walk applies the `ln(f)` jet after its
  in-walk scale quotient, reproducing production's scale-then-log order `ln(f/s)`, so LTBS × IOV
  models (including with an `ExpressionScale obs_scale`) no longer fall back to finite
  differences on the population gradient. Validated against reconverged finite differences of
  the FOCEI-IOV objective. The inner EBE gradient still uses finite differences for LTBS × IOV.
- **Custom / time-varying residual-error magnitude combined with `iiv_on_ruv` is now analytic
  under IOV too** (#486): #673 covered the non-IOV case; the stacked `[η_bsv, κ]` residual-eta
  assembly is dimension-generic, so occasion (κ) random effects compose with the magnitude
  direct-θ terms with no extra work. Validated against reconverged finite differences.
- **Log-transform-both-sides (LTBS) analytic *inner* EBE gradient now covers the remaining
  closed-form combinations** (#486): plain LTBS landed in #665; this extends the same
  `g = ln(f)` inner jet to LTBS combined with an η-dependent `ExpressionScale obs_scale`, with
  **time-varying covariates** (the event-driven inner walk), and with a `TIME`-built-in
  structural parameter. The inner η-gradient matches the outer to ~1e-10, and gradient-based
  HMC now engages for these models. LTBS × IOV still uses the finite-difference inner gradient.
- **Custom / time-varying residual-error magnitude combined with `iiv_on_ruv`** now gets an
  exact analytic FOCEI outer (θ/Ω/σ) gradient instead of finite differences (#486): the
  residual-eta `c̃`-column coupling `d/R` gains its magnitude direct-θ terms, mirroring the
  σ-parameter block. Validated against reconverged finite differences of the FOCEI objective.
- **`prepare_frem()` accepts a prior fit to seed FREM init values** (#239). The new
  optional `fit_init: Option<&FremFitInit>` parameter carries a completed fit's theta
  and omega estimates; when supplied, the generated FREM model's PK theta inits and
  PK-PK omega block are seeded from those converged values instead of the base
  model file's declared inits, so a subsequent fit of the FREM model warm-starts
  closer to convergence. Names are matched case-insensitively against the base
  model; unmatched names fall back to the declared inits. `None` preserves the
  prior behaviour unchanged.
- **Covariate-selected residual error models (`if/else` in `[error_model]`)** (#658).
  The `[error_model]` block can now select a residual error model per observation
  by an arbitrary covariate condition — e.g. a free-vs-total assay switched by a
  `FREE` flag: `if (FREE == 0) { DV ~ proportional(PROP_TOTAL) } else { DV ~
  proportional(PROP_UNBOUND) }`, with `else if` chains and a required final
  `else`. This mirrors the Form C `[scaling] y = <expr>` selector (#650), so a
  model can express both the readout **and** its residual error against the same
  per-row flag without recoding it into a synthetic `CMT` column. Works on
  analytical **and** ODE models, across FOCE/FOCEI, Gauss-Newton, SAEM, and
  importance sampling. The selector covariate becomes a required data column
  (`E_MISSING_COVARIATE`). `block_sigma` correlated residuals are supported
  together with a selected error model (#669): co-temporal rows resolving to
  different branches (e.g. a total/unbound assay pair) pick up the cross-branch
  covariance `ρ·σ_i·σ_j` in the dense residual `R`, exactly as for per-CMT
  endpoints. See
  [Error model → Covariate-selected error models](https://ferx-nlme.github.io/ferx-core/model-file/error-model.html).
- **Full `[scaling] y = <expr>` output readouts (Form C) on analytical PK models** (#650).
  A closed-form (`pk one_cpt_iv(...)`, …) model can now replace the built-in
  concentration output with an arbitrary readout expression — enabling flexible
  multi-DV residual errors such as a free-vs-total protein-binding correction
  (`y = if (FREE == 0) central/V + BMAX*(central/V)/(KD + central/V) else central/V`),
  previously expressible only on ODE models. The readout may reference the central
  compartment **amount** (`central`, and the oral `depot`), individual parameters —
  **including non-structural ones** like a binding `BMAX`/`KD` — thetas, etas, covariates
  (read per-observation, so a per-row flag switches the readout), and `if/else`. FOCEI/FOCE
  gradients flow through it **analytically** (outer and inner) on both the static
  dose-superposition path and the time-varying-covariate / oral-infusion event-walk path —
  so a free-vs-total readout gated on a per-row `FREE` flag stays analytic — including on
  **IOV** subjects (`kappa` declarations) since #655, where the readout parameters are
  BSV-only (a `kappa` reference is rejected at parse) so only the concentration carries the
  occasion κ. A readout referencing the oral depot amount, per-CMT readouts, and direct θ/η
  references fall back to finite-difference gradients (the prediction stays exact, and the
  parser emits a warning). Peripheral compartment amounts are rejected (use an ODE model). See
  [Scaling → Form C](https://ferx-nlme.github.io/ferx-core/model-file/scaling.html).

### Changed
- **Bumped `MAX_PK_PARAMS` from 16 to 128**, raising the ceiling on ODE structural
  parameters (rate constants, Emax/EC50, baselines, …) that an ODE model may declare
  in `[individual_parameters]` from 7 to 119 (slots 0–8 remain reserved for the named
  PK params CL, V, Q, V2, KA, F, Q3, V3, LAGTIME). Complex multi-analyte models —
  e.g. simultaneous parent/metabolite systems with ~20+ structural parameters — that
  previously failed the parser's "too many individual parameters" check now compile.
  The ceiling is a compile-time constant because the Enzyme autodiff backend requires
  stack-allocated arrays of statically-known size; the cost of the higher ceiling is
  purely stack (`MAX_PK_PARAMS * 8` bytes per `PkParams`, ~1 KB at 128).
- **M3 BLOQ censored rows now enter the FOCEI Laplace determinant `log|H̃|`** for a
  consistent likelihood (#486). Previously censored rows contributed to the data term and
  the true inner Hessian but were dropped from the outer `log|H̃|` — an internal
  inconsistency with quantified rows. They now enter `H̃` at FOCEI (Gauss-Newton) order
  (structural `g2·a·aᵀ`, plus the `iiv_on_ruv` residual-eta cross terms), with the exact
  analytic gradient matching reconverged finite differences to ~1e-6 across non-IOV/IOV and
  closed-form/ODE, including the `M3 + IOV + iiv_on_ruv` triple. **M3 FOCEI OFV values shift
  accordingly** (estimates/SEs are essentially unchanged), and the OFV now matches NONMEM
  `METHOD=1 LAPLACE` M3 up to the residual FOCEI-vs-LAPLACE second-order term. FOCE
  (Sheiner–Beal) is a distinct objective, updated separately (see the next entry).
- **FOCE (Sheiner–Beal) M3 BLOQ now uses the linearized-marginal moments** for the
  censored tail probability — `−logΦ((LLOQ − f0)/√R̃ⱼⱼ)` with the marginal mean
  `f0 = f(η̂) − Hη̂` and marginal variance `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰`, the same moments the
  quantified rows use — instead of the conditional prediction and residual variance
  (#646). This makes plain FOCE a self-consistent Sheiner–Beal objective (matching
  Monolix's linearization likelihood and first-order/Tobit theory); the analytic FOCE
  gradient is updated to match, including a new direct Ω-gradient channel for the censored
  variance, on both the non-IOV and IOV paths. **FOCE M3 OFV and estimates shift** (most
  when between-subject variance is large, where `HΩHᵀ` dominates `R⁰`). FOCEI M3 keeps the
  conditional censored term — the treatment NONMEM's `METHOD=1 LAPLACE` M3 uses (NONMEM
  runs M3 only under LAPLACE), which ferx's first-order FOCEI matches up to the
  FOCEI-vs-Laplace `∂²f/∂η²` second-order term.
- **Removed the automatic SLSQP fallback after a non-converged outer optimization** (#657).
  When the primary optimizer stopped without clean convergence, ferx used to silently re-run
  a full second outer optimization (with inner EBE loops) with SLSQP from the same point —
  roughly doubling wall-time on already-slow non-converged runs while rarely rescuing the
  fit. Non-convergence is now reported directly (`converged = false` plus the "Outer
  optimization did not converge" warning) with no automatic retry. Users who want SLSQP can
  still set `optimizer = slsqp`.
- **IMPMAP/IMP's FREM Rao-Blackwell E-step now runs a per-subject adaptive ISCALE
  pilot search** instead of a fixed `iscale = 1.0` (#406 follow-up). The RB
  conditional PK proposal is usually well matched, but for subjects where the
  inner-loop Hessian is a poor estimate of the true PK conditional curvature
  (sparse PK data), a fixed proposal width could leave ESS low even after RB.
  Mirrors the ISCALE rescue the full-dimensional sampler already had. Applies to
  the iterative MCEM E-step only; the eval-only / final-marginal IS report keeps
  a fixed proposal for run-to-run reproducibility.

### Fixed
- **`.fitrx` bundles now carry SAEM conditional-distribution results** (#675). A fit run with
  `conddist = true` writes a `conddist.csv` entry (`ID, ETA, COND_MEAN, COND_SD, COND_MODE`)
  into the bundle, and loading it back now populates `FitResult.cond_dist` instead of always
  reporting `None` — enabling the FeRx GUI's "Cond. Dist." Evaluation section to read this data
  from a saved fit.

### Added
- **Log-transform-both-sides (LTBS) combined with time-varying covariates** now gets an exact
  analytic FOCE/FOCEI **outer** (θ/Ω/σ) gradient on the closed-form (analytical 1-/2-/3-cpt)
  models instead of finite differences (#486). The event-driven TV-cov walk applies the same
  post-walk `g = ln(f)` jet transform the dose-superposition path already used — last, after
  any `ScalarScale`/`ExpressionScale` quotient, reproducing production's scale-then-log order
  `ln(f/s)` — so LTBS composes with a time-varying-covariate and an `ExpressionScale`
  `obs_scale`. Validated against FD of the log-scale production predictor.
- **Plain closed-form LTBS now also gets an exact analytic *inner* EBE gradient** (#486), not
  just the outer gradient — the light inner provider applies the same `g = ln(f)` jet.
  Previously all LTBS models used a finite-difference inner gradient. Because the analytic inner
  gradient makes the marginal surface slightly noisier (the `ln` wrap amplifies the ~1e-9
  provider-vs-predictor gap), a **closed-form, non-IOV** LTBS fit converges the inner EBE loop to
  at least `1e-6` (up from the `1e-5` default, unless you set `inner_tol` explicitly) so the fit
  lands reproducibly on flat Ω directions and the covariance SEs of weakly-identified variances
  are stable; the covariance step then reconverges tighter still (see the next entry). LTBS
  combined with time-varying covariates, IOV, ODE, or an η-dependent `ExpressionScale` still uses
  the FD inner gradient (those inner kernels do not yet carry the transform, or already agree
  with the objective as ODE-LTBS does). Validated: the analytic inner η-gradient matches the
  outer, and warfarin LTBS covariance SEs match NONMEM `$COV MATRIX=R`.
- **New `[fit_options] cov_inner_tol`** — the inner EBE-reconvergence tolerance used **only by
  the covariance step**, decoupled from the fit's `inner_tol`. The covariance R-matrix is a
  second-difference of the reconverged OFV and is more sensitive to EBE precision than the fit
  itself, so a sensitive/flat covariance can be reconverged tighter without slowing every outer
  iteration (e.g. the heavily-censored M3 + IOV case in #654 — set `cov_inner_tol = 1e-11`).
  Unset (default) uses `inner_tol` for ordinary models — SEs are byte-identical to before — and
  `min(inner_tol, 1e-8)` for **closed-form, non-IOV LTBS** models, whose `g = ln(f)` covariance
  Hessian needs the tighter reconvergence. (The covariance step is *not* tightened blanket-wide:
  over-converging some ill-conditioned inner Hessians, e.g. IOV block-Ω, drives the covariance
  indefinite.)
- **A `TIME`-built-in structural parameter combined with a built-in absorption input-rate
  forcing or a non-zero ODE `init(...)` baseline** now gets exact analytic FOCE/FOCEI
  sensitivities instead of finite differences (#486). The event-driven walk that threads the
  per-event `TIME` already carries the absorption `R_in` forcing (since #643) and seeds the
  `init(...)` state (since #662), so the model-level decline for those combinations was stale;
  it has been removed. Validated against finite differences of the production predictor.
- **Several inter-occasion-variability (IOV) analytic-gradient cells that were arbitrarily
  narrower than their non-IOV counterparts are now analytic** (#486, "IOV-scope parity"),
  closing gates that were more restrictive than the walk actually required:
  - **All built-in absorption input-rate kinds under IOV** — the smooth densities
    `igd`/`transit`/`weibull` now get exact analytic FOCE/FOCEI sensitivities under IOV, not
    just `zero_order`/`first_order`/`mixed`/`parallel`. The IOV gate now mirrors the non-IOV
    kind-agnostic rule exactly; only `weibull` + estimated lagtime (β<1 onset divergence) and
    any forcing combined with a steady-state dose remain on finite differences.
  - **Compartment-indexed bioavailability `F{cmt}` and lagtime `ALAG{cmt}` under IOV** — the
    event-driven walk already resolves each dose's own compartment slot, so these no longer
    fall back to finite differences.
  - **A constant `ScalarScale` `obs_scale` divisor under IOV on both engines** — the trivial
    covariate-independent case of the `ExpressionScale` quotient the IOV walk already applies:
    on the closed-form models the final jet is divided uniformly, and on ODE models the
    in-walk readout already divides `p/k` over the stacked dual.

  All validated against finite differences of the production `predict_iov` (value, gradient,
  and Hessian over the stacked `[η, κ]` vector).
- **Built-in absorption forcings (`zero_order(dur)`, `first_order`, and `mixed`) combined
  with inter-occasion variability (IOV)** now get exact analytic FOCE/FOCEI sensitivities on
  the ODE path instead of finite differences (#486), closing the last zero-order gap. The IOV
  analytic walk is the same event-driven walk as the non-IOV time-varying-covariate path, so
  each forcing's rate and moving-boundary window are rebuilt from that dose's own
  per-occasion PK jet — the κ (occasion) sensitivity rides through exactly as η/θ do.
  Validated against finite differences of the production `predict_iov` (value, gradient, and
  Hessian over the stacked `[η, κ]` vector), including a κ-coupled `DUR` axis-placement check,
  a `parallel` two-`first_order` pathway, and the `first_order` + estimated-lagtime and
  `+ EVID 3/4 reset` combinations. The smooth-density input-rate kinds (igd / transit /
  weibull) under IOV are now analytic as well (see the IOV-scope-parity entry above); only a
  built-in forcing combined with a steady-state dose under IOV remains on finite differences.
- **Modeled-duration/rate doses (`RATE=-1`/`-2`) combined with steady-state dosing on the
  closed-form (analytical 1-/2-/3-cpt) models** now get exact analytic FOCE/FOCEI
  sensitivities instead of finite differences (#486), the last modeled-dose gap after #652
  (the ODE path had it via #642). The closed-form dual steady-state equilibration threads the
  modeled infusion-window jet `(rate, dur)` into each cycle's active/quiet split, so the
  moving infusion-end flows through the steady-state trough exactly as it does through the
  current pulse. Validated against finite differences of the production predictor and against
  the independently NONMEM-anchored ODE steady-state modeled-dose twin.
- **Initial conditions (`init(...)` / `[initial_conditions]`) are now fully analytic on the
  FOCE/FOCEI sensitivity gradient** — every remaining combination that previously fell back to
  finite differences is closed (#486). A parameter-dependent baseline (e.g. an indirect-response
  or disease-progression PD baseline, `init(central) = BASE/V`) now gets exact analytic
  gradients when combined with: a finite infusion, a built-in input-rate forcing
  (igd/transit/weibull/`first_order`/`zero_order`), an estimated lagtime, steady-state dosing, a
  modeled-duration/rate dose, and an EVID 3/4 reset — on the ODE event-driven walk; the
  closed-form (1-/2-/3-cpt) `init` baseline on the time-varying-covariate walk; and `init`
  **under IOV** on both engines (the amount stays BSV-only while the decay kernel follows each
  occasion's clearance, matching `predict_iov`). Previously `init` was analytic only on the
  closed-form dose-superposition path (#527) and the ODE plain-bolus TV-cov subset (#649);
  everything else finite-differenced. Fits are unchanged — only the gradient path is now exact
  (and faster) for these models.
- **Zero-order absorption (`zero_order(dur)`, and the `zero_order` leg of a `mixed` model)
  combined with time-varying covariates or an estimated lagtime** now gets exact analytic
  FOCE/FOCEI sensitivities on the ODE event-driven walk instead of finite differences
  (#486). The constant `F·amt·frac/dur` window is delivered per integration segment, with
  its moving end `d.time + lag + dur` (and, under lagtime, its moving start) carried by
  rate-off / rate-on saltations; the rate-off uses the general `g⁻ − g⁺` form so a
  covariate that varies across the window end stays exact. Only `zero_order` under IOV
  remains on finite differences.
- **Modeled-duration/rate doses (`RATE=-1`/`-2`, `D{cmt}`/`R{cmt}`) on the analytical
  (closed-form 1-/2-/3-cpt) models** now get exact analytic FOCE/FOCEI sensitivities
  instead of finite differences (#486), closing the largest closed-form-vs-ODE gap (the
  ODE path already had this via #630/#635). The modeled infusion window resolves from the
  PK parameters, so the infusion *end* is a moving boundary in `D`/`R`; the closed-form
  event-driven walk now carries it exactly (to second order) via the dual window length,
  the sign-mirror of the existing lagtime dose-*start* handling. Covers non-IOV and IOV
  (per-occasion windows, including κ-coupled slots), on both the outer θ/Ω/σ gradient and
  the inner EBE η-gradient. Validated against finite differences of the production
  predictor and against the independently NONMEM-anchored ODE twin. Modeled-dose ×
  steady-state and rate-defined (`RATE=-1`) infusion under `F ≠ 1` remain on finite
  differences (as on the ODE path).
- **An `ExpressionScale` `obs_scale` divisor (e.g. `obs_scale = V`) combined with IOV on a
  closed-form (analytical 1-/2-/3-cpt) model** now gets exact analytic FOCE/FOCEI
  sensitivities on both the outer and inner loops instead of finite differences (#486). The
  scale divisor is applied as a per-occasion-group post-walk quotient over the stacked
  `(θ, η, κ)` axes — each occasion's divisor rides its own κ through the PK parameters —
  porting the pattern already used on the ODE IOV path. Time-varying covariates compose
  (the divisor stays subject-static, matching NONMEM's per-occasion `S1` scaling). LTBS and
  constant `ScalarScale` under IOV continue to use finite differences.
- **Custom / time-varying residual-error magnitude (`[error_model]` σ-scaling expression)
  now gets an exact analytic gradient** on both loops instead of finite differences
  (#484/#576/#486). The magnitude is η-independent, so the inner EBE gradient just
  threads the per-observation multiplier into the residual variance and its
  `f`-derivative; the FOCEI outer θ/σ population gradient additionally
  dual-differentiates the compiled magnitude program w.r.t. θ, adding a new
  *direct*-θ term to `∂R/∂θ` for any theta the magnitude expression references
  (e.g. a late-phase RUV inflation `PROP_ERR * (1 + RUV_LATE * TIME/48)`).
  Validated against a live NONMEM FOCEI fit (OFV and every estimate, including
  `RUV_LATE`, match to ~4-5 significant figures — see `examples/warfarin_ruv_magnitude.ferx`).
  Plain `method = foce` (non-interaction) now gets the analytic gradient too, on both
  the non-IOV and IOV paths: the Sheiner–Beal marginal threads the magnitude into its
  typical-value residual variance `R⁰` (value and direct-θ derivative), so `auto`
  resolves a FOCE magnitude model to a gradient-based optimizer instead of BOBYQA. The
  supported theta count is also raised from 16 to 32. `block_sigma` correlated residual
  error, `iiv_on_ruv`, an M3-BLOQ censored row, and more than 32 thetas still fall back
  to the (magnitude-aware) finite-difference gradient.
- **`init(...)` initial conditions with time-varying covariates** now get exact analytic
  FOCE/FOCEI sensitivities on the ODE path instead of finite differences (#486). The
  event-driven walk seeds the dual initial state from the subject's first-record covariate
  snapshot (matching the production predictor's `init_pk`), so a covariate- or η-dependent
  baseline (e.g. `init(central) = BASE / V`) carries `∂/∂(θ,η)`. Analytic for the plain-bolus
  subset; `init(...)` combined with an EVID 3/4 reset, an estimated lagtime, a finite
  infusion, a built-in input-rate forcing, steady-state, or a modeled-duration/rate dose
  stays on the finite-difference fallback.
- **Modeled-duration/rate doses (`RATE=-1`/`-2`, `D{cmt}`/`R{cmt}`) under IOV** now get
  exact analytic FOCE/FOCEI sensitivities on the ODE path instead of finite differences
  (#486). Each occasion resolves its own modeled infusion window from the per-occasion PK
  jet, and the moving infusion-end boundary carries `∂/∂{θ,η,κ}` — including when the
  modeled slot is itself κ-coupled (`D1 = TVD1·exp(η + κ)`).
- Three more **steady-state (`SS=1`) ODE dosing** combinations now get exact analytic
  FOCE/FOCEI sensitivities instead of finite differences (#486): a modeled-duration/rate
  dose (`RATE=-1`/`-2`), a rate-defined infusion under bioavailability `F ≠ 1`, and an
  estimated lagtime. The SS dual equilibration now threads the
  same mode-aware rate/window jet the non-SS event-driven walk uses into its per-cycle
  active/quiet split, and a lagged SS dose's pre-arrival window `[t_dose, t_dose+lag)` is
  seeded from the previous interval's steady-state tail (mirroring the production
  predictor's own pre-arrival seed). Only SS combined with a non-autonomous RHS (one that
  reads `TIME`/`TAFD`/`TAD`) stays on the FD fallback — a time-invariant pulse train has no
  well-defined steady state under a time-dependent RHS. See
  [Steady-state dosing](model-file/steady-state.qmd).
- A structural parameter that reads the event-time built-in `TIME`/`time` (a
  NONMEM-style `$PK IF (TIME.GE.45) CL=…` time-dependent switch) now gets exact
  analytic FOCE/FOCEI sensitivities instead of falling back to finite differences
  (#486 / #610). The per-event time is threaded into the same event-driven `Dual2`
  (outer) / `Dual1` (inner EBE) walk used for time-varying covariates, so the
  gradient is exact and faster. Covers closed-form (1-/2-/3-cpt) and ODE models,
  with and without inter-occasion variability, including together with an
  η-dependent `obs_scale` expression (the event-driven walk now applies the scale
  quotient — which also makes time-varying-covariate + expression-scale models
  analytic). The direct `pk(...=TIME)` structural mapping is covered too: the parser
  desugars the mapped slot into a hidden individual parameter (`__ferx_pktime_*`), so
  it rides the same per-event analytic walk as an `[individual_parameters]` switch.
- A Form-C ODE readout (`[scaling] y = <expr>`) that references a θ or η
  **directly** (e.g. `y = central/V1 * (1 + ETA_CL) + TVBASE`) now gets exact
  analytic FOCE/FOCEI sensitivities instead of falling back to finite differences
  (#486). The parser desugars each bare `THETA(i)`/`ETA(k)` in the readout into a
  hidden individual parameter, so its `∂y/∂θ`/`∂y/∂η` (and the 2nd-order blocks)
  ride the same validated individual-parameter sensitivity chain as `central/V1`;
  the prediction value is unchanged and the synthetic parameters never appear in
  EBE / sdtab output. (A readout referencing a neural-network output stays on the
  FD fallback.)
- Two more **non-IOV ODE** model combinations now get exact analytic FOCE/FOCEI
  sensitivities instead of finite differences (#486): a time-varying-covariate ODE
  model with **(a)** an `EVID=2` covariate-only breakpoint, or **(b)** an
  η-dependent `obs_scale = expr(θ,η)` divisor. The `obs_scale` divisor is applied
  as a single subject-static post-walk quotient (production evaluates it at the
  subject covariate snapshot), and the EVID=2 breakpoint rides the event-driven
  walk that already carried it — closing the matching cells the IOV path gained in
  #590/#591. LTBS-combined `obs_scale` stays on the FD fallback.
- **Built-in absorption input-rate forcing** (`igd`/`transit`/`weibull`/`first_order`/
  `zero_order`, incl. `mixed`) combined with an **EVID 3/4 reset** now gets exact
  analytic FOCE/FOCEI sensitivities instead of finite differences (#486): the fix
  threads the already-tracked `reset_floor` into the shared forcing helper (turning
  off a dose's pre-reset tail, matching the infusion rule) kind-agnostically, so
  `zero_order`'s own separate per-segment window mechanism (#530) inherits it too.
  `igd`/`transit`/`weibull`/`first_order` (but not `zero_order`/`mixed`) also now get
  exact analytic sensitivities combined with **time-varying covariates**, via the
  same helper wired into the event-driven walk, hoisting the forcing's
  dose-invariant constants fresh per segment as the PK snapshot changes. Combined
  with an **estimated lagtime**, `igd`/`transit`/`first_order` are also now
  analytic: the continuous `∂R_in/∂lag` flows through the walk's dual
  time-after-dose, and the forcing's onset at the dose's lagged arrival is
  injected as an exact rate-on saltation. `weibull` stays on the FD fallback when
  combined with lagtime (its onset can diverge for shape `β < 1`), as does
  `zero_order`'s own moving-boundary cutoff combined with TV-cov or lagtime (a
  separate per-segment mechanism not yet ported to the event-driven walk).
- **Analytic transit-compartment absorption** (#386). A new `pk one_cpt_transit(cl, v, n, mtt)`
  structural model evaluates Savic (2007) transit absorption into a one-compartment disposition as
  an exponential-tilting closed form (the incomplete-gamma `convolve_1cpt`), with exact `Dual2`
  FOCE/FOCEI sensitivities `∂C/∂{CL,V,N,MTT,F,η}` and no ODE solve — the fast analytic counterpart
  to the `transit()` ODE forcing, with continuous (estimable) `N`. Supports single/multiple bolus
  doses, bioavailability, and lag time; with `N = 0` it reduces exactly to first-order (Bateman)
  oral absorption. Steady-state doses, IOV, time-varying covariates, infusions, and a `depot`
  initial amount are rejected with an actionable message (use an ODE transit model for those). See
  `examples/one_cpt_transit.ferx`. System resets (EVID=3/4) are also rejected, since the
  superposition closed form cannot express mid-profile compartment zeroing (#634). A typical-value
  warning (`W_TRANSIT_FLIP_FLOP`) now fires when the disposition rate exceeds the transit rate
  `KTR = (n+1)/mtt` (the flip-flop regime, where the closed form returns an identically-zero profile
  that would silently degenerate the objective) (#634).
- **Analytic transit absorption into a two-compartment disposition** (#386). A new
  `pk two_cpt_transit(cl, v1, q, v2, n, mtt)` structural model extends the closed form to a 2-cpt
  disposition: the Gamma(N+1, KTR) absorption time is convolved with the bi-exponential disposition
  (`convolve_2cpt` — two `convolve_1cpt` terms at the macro-rates α, β), again with exact `Dual2`
  FOCE/FOCEI sensitivities `∂C/∂{CL,V1,Q,V2,N,MTT,F,η}` and no ODE solve. With `N = 0` it reduces
  exactly to 2-cpt first-order oral absorption. Same scope/limits as `one_cpt_transit` (bolus doses,
  bioavailability, lag time; SS/IOV/TV-covariate/infusion/`depot`-init/resets rejected, flip-flop
  warning). NCA initial-estimate seeding now peels Q/V2 and seeds the lag time for the transit
  models too (#634). See `examples/two_cpt_transit.ferx`.
- `block_sigma` correlated residual errors are now supported under
  `method = focei` and `method = imp`, not just `foce` and `saem` (#616). FOCEI
  carries the off-diagonal residual covariance through the Almquist interaction
  Hessian (`H̃ = HᵀR⁻¹H + ½·tr(R⁻¹∂R/∂η R⁻¹∂R/∂η) + Ω⁻¹`), and IMP builds its
  Student-t proposal precision from the dense `R⁻¹`. On the committed
  `correlated_residual_combined` anchor, ferx FOCEI OFV 18.722087 matches NONMEM
  `METHOD=1 INTER` (18.722087) to better than 1e-5. The Gauss-Newton
  (`gn` / `gn_hybrid`) paths remain diagonal-only and are still rejected.
- `block_sigma` correlated-residual `foce` / `focei` fits now run **exact analytic
  gradients on both loops** instead of finite differences (#627). The within-observation
  `combined(...)` cross term is carried through the same dense-`R` builders the marginal
  uses (`compute_dr_df_matrices`, `compute_d2r_df2_matrices`), so the inner EBE η-gradient
  and the outer θ/Ω/σ gradient are noise-free and the `auto` optimizer resolves to a
  gradient-based method. The OFV is unchanged (Eval 1 on the anchor is still 18.722087);
  a rare cross-endpoint off-diagonal-`R` subject falls back to per-subject finite
  differences. (The `gn` / `gn_hybrid` paths stay diagonal-only.)
- **AUC-target attainment metric + vancomycin AUC-TDM example/anchor** (#391, S2.5b). A new
  optional `[adaptive_dosing] auc_target = [low, high]` key adds `auc_target_attainment` to
  `AdaptiveSubjectMetrics` — the fraction of inter-decision windows whose area under the monitored
  signal (e.g. vancomycin AUC₂₄) falls in the band (`high` may be `inf`). Like `target_window` it
  reports a metric only and never influences dosing; declaring it turns on a signal-AUC pass that
  re-integrates the realized doses on a dense grid (trapezoid), leaving the reactive run untouched.
  A new bundled model `examples/adaptive_vanco_auc.ferx` titrates a once-daily infusion on the
  pre-dose trough and reports AUC₂₄ attainment, cross-validated against an external **mrgsolve** run
  (reference kit in `tests/reference/vanco_mrgsolve/`, slow-gated `tests/adaptive_vanco_anchor.rs`).
  See [Adaptive dosing](model-file/adaptive-dosing.qmd).
- `TIME`/`time` are now built-in event-time values in `[individual_parameters]`
  expressions and direct analytical `pk(...=TIME)` mappings, enabling
  NONMEM-style time-dependent PK parameter switches without declaring `TIME` as
  a covariate (#607). The event time is threaded through every prediction and
  diagnostic path — analytical and ODE predictions, the `[odes]` right-hand side,
  sdtab individual-parameter columns, `[derived]` columns, the survival/TTE
  hazard, and the SDE EKF — so `TIME` resolves to each event's time everywhere
  rather than only on the main prediction path; for models that use it, analytic
  FOCE/FOCEI sensitivities fall back to finite differences (#610).
- **Platelet-ladder adaptive-dosing example + mrgsolve external anchor** (#391, S2.5a). A new
  bundled model `examples/adaptive_platelet_ladder.ferx` exercises the reactive
  `[adaptive_dosing]` `levels` ladder on an oncology dose-modification scenario — a Friberg
  myelosuppression model whose simulated platelet count titrates the dose down a discrete ladder
  (100 → 75 → 50 → 25 mg). It is cross-validated against an external **mrgsolve** run, the
  apples-to-apples comparator for feedback dosing (NONMEM has none), which ferx reproduces
  dose-for-dose (reference kit in `tests/reference/platelet_mrgsolve/`, slow-gated
  `tests/adaptive_platelet_anchor.rs`). With this external anchor, adaptive (feedback) dosing
  graduates from **experimental** to **beta**. See [Adaptive dosing](model-file/adaptive-dosing.qmd).
- **Per-subject outcome metrics for adaptive dosing** (#391, S2.4). `simulate_adaptive()` and
  `simulate_adaptive_from_spec()` now return a `metrics` field on `AdaptiveSimulationResult` —
  one `AdaptiveSubjectMetrics` row per realized `(subject, draw, sim)` run: cumulative dose,
  dose-increase / -decrease / hold / discontinuation counts, time-to-discontinuation, and the
  observed-signal summary (min / max / mean). A new optional `[adaptive_dosing] target_window =
  [low, high]` key adds `pct_time_in_window` (the fraction of the signal-bearing decisions whose
  observed signal fell in the band; `high` may be `inf` for a one-sided target) — it reports a metric only and never
  influences dosing. Every metric is derived from the realized dose ledger and decision log alone.
  See [Adaptive dosing](model-file/adaptive-dosing.qmd).
- **Drug-driven event-time simulation for joint PK-TTE** (#564). `simulate()` /
  `simulate_with_options()` now sample event times for an ODE-accumulated hazard
  (`hazard =` in `[event_model]`), not just analytic families: the augmented ODE is
  integrated until the cumulative hazard reaches `−log u`, with the crossing located by a
  root-finder. A finite `[simulation] horizon` (or `SimulateOptions.horizon`) is **required**
  for these models — a drug-driven hazard can vanish and never fire, so there is no implicit
  observation window; EVID-3/4 resets and left truncation on an ODE-TTE subject are not yet
  supported and are rejected with a clear error.
- **Exact analytic gradients for M3 BLOQ + IOV models — full FOCEI/FOCE matrix**
  (closed-form 1/2/3-cpt and user-ODE, #580/#591/#486). An inter-occasion-variability model with M3
  below-limit handling now runs on exact analytic sensitivities instead of finite
  differences across the whole estimator matrix: **FOCEI**, **non-interaction FOCE**, and
  the triple **M3 + IOV + `iiv_on_ruv`**. The censored data term `−logΦ((LLOQ−f)/√v)` and
  its `f`-derivatives ride the stacked `[η_bsv, κ]` layout — censored rows enter the data
  gradient and the true inner Hessian but stay excluded from the Laplace `H̃`/`log|H̃|`
  (matching `foce_subject_nll_iov`); for `iiv_on_ruv` the censored residual-eta cross
  coefficients `(C·z, C·m)` enter the true inner Hessian and the `h·z` residual-eta column
  enters the inner gradient. The FOCE path differentiates the augmented Sheiner–Beal
  marginal with censored rows re-entering as `−logΦ` at the population (η=0, κ=0) variance.
  Inner stacked-η gradients match central FD of the IOV inner objective and outer packed
  gradients match Richardson reconverged FD of the corresponding marginal, all to ~1e-3;
  estimate-level tests confirm the analytic fits land on the FD (NONMEM-anchored) optima.
  This applies equally to **user-ODE** models (#486), including the triple **M3 + IOV +
  `iiv_on_ruv`**: the event-driven ODE sensitivity walk emits the standard per-observation
  shape (with a structural zero `∂f/∂η_ruv` column for the residual-error η), and the
  censoring and `exp(2·η_ruv)` variance scaling are applied downstream keyed on the `CENS`
  flag and `residual_error_eta`, so the ODE path rides the exact same analytic assembly as
  the closed-form path (inner and outer FD-comparison tests on censored ODE-IOV fixtures
  confirm both tails for plain IOV, M3 + IOV, IOV + `iiv_on_ruv`, and the full triple). The
  **non-IOV** ODE M3 + `iiv_on_ruv` combination is analytic too — the last `iiv_on_ruv`
  holdout: the ODE and closed-form packed gradients are bit-identical and both match
  reconverged FD to ~1e-7 on each censoring tail (inner and outer), completing the entire
  `iiv_on_ruv` × {plain, IOV, M3} × {closed-form, ODE} matrix.
- **Parallel / mixed dual-pathway absorption — `first_order(ka)` composition** (#505). A new
  built-in `first_order(ka)` input-rate function exposes the classic first-order (Bateman)
  absorption for composition in `[odes]`, so two absorption pathways can be split by a dose
  fraction: `parallel` (dual first-order, `FR1*first_order(ka=KA1) + FR2*first_order(ka=KA2)`) and
  `mixed` (zero-order + first-order, `FZO1*first_order(ka=KA) + FZO*zero_order(dur=DUR)`). A pathway
  fraction on `zero_order(...)` (`FR*zero_order(...)`) is now accepted (previously rejected), so the
  per-segment zero-order channel carries the fraction; the fractions must partition the dose
  (each `0 < FR ≤ 1`, `Σ FR ≈ 1`). `parallel` keeps exact analytic FOCEI gradients (including
  ∂/∂fraction); `mixed` differentiates the zero-order duration/fraction by finite differences (the
  moving-boundary case, #530). Standalone first-order absorption still uses the analytical
  `pk *_oral` path. See [Absorption models](model-file/absorption.qmd).
- **Joint PK-TTE — drug-driven hazard via `[event_model] hazard = <expr>`** (#564). On an ODE
  model, a `hazard` expression that references the PK state (e.g. `H0 * exp(BETA * (central / V))`)
  is accumulated as a cumulative-hazard ODE compartment and estimated jointly with the PK by
  FOCEI/SAEM, with shared random effects. Mutually exclusive with the analytic `family` hazard;
  requires an ODE model (no IOV yet). Simulation of the ODE-accumulated hazard follows in a later
  slice. See [Time-to-event](estimation/tte.qmd).
- **Custom / time-varying residual-error magnitude** (#484). An `[error_model]`
  sigma argument may now be an expression of `TIME`, covariates, and thetas
  rather than a bare parameter — e.g.
  `DV ~ combined(PROP_ERR * (if (TIME > 24) RUV_LATE else 1.0), ADD_ERR)` —
  reproducing the NONMEM `$ERROR` idiom of a time- or covariate-dependent error
  coefficient. The expression scales that sigma's loading per observation;
  magnitudes may depend only on `TIME`/covariates/thetas (not η or the
  prediction) and are supported for `method = foce`/`focei` (the analytic
  gradient falls back to finite differences when active).
- **`[fit_options] outer_xtol` / `outer_ftol`** (#469) — expose the derivative-free
  `bobyqa` outer optimizer's step (`xtol_rel`) and objective (`ftol_rel`) stop
  tolerances, previously hardcoded. Lets a fit tighten or loosen BOBYQA's
  convergence on flat/noisy objective ridges. See
  [fit options](model-file/fit-options.qmd).
- **Defensive-mixture importance sampling for IMP/IMPMAP — new `imp_defensive_alpha`
  fit option** (#528). Each subject can draw an `imp_defensive_alpha` fraction of its
  importance samples from the prior `N(0, Ω)`, bounding the importance weights so a
  weakly-identified subject — e.g. an analytical `[initial_conditions]` baseline whose
  `V` cancels in the amplitude — can no longer hijack the weighted M-step and walk θ to
  the bounds. The option is **opt-in** (default `0.0`, the legacy single-proposal sampler
  that stays bit-comparable with NONMEM); set a small positive value such as `0.1` to
  enable the rescue. Applies to `imp` and `impmap`, including the FREM Rao-Blackwell
  path; for an `impmap` stage it may also be written `impmap_defensive_alpha`. See
  [Importance sampling](estimation/importance-sampling.qmd#defensive-mixture).
- IMP/IMPMAP and SAEM now flag a finite-but-enormous runaway objective (≥ `1e15`) as
  **not converged**, so a collapsed-weight blow-up can no longer report `converged` or
  win multi-start selection (#528).
- **Experimental `simulate_adaptive()` — state-reactive ("feedback") dosing simulation**
  (#553, epic #391). A programmatic entry point that simulates regimens where each dose is
  chosen at run time by a controller reading the simulated state (TDM target attainment,
  oncology dose reduction, biomarker titration). ODE models only; the controller is supplied
  as a per-subject factory; every realized dose and every decision (including holds) is
  returned alongside the trajectories, and a frozen-schedule replay verifier checks the dose
  bookkeeping by default. See [Adaptive dosing](model-file/adaptive-dosing.qmd).
- **Assay-noised (`Dv`) monitors for `simulate_adaptive()`** (#566, epic #391). A controller can
  titrate on the realized, assay-noised measurement — `IPRED + ε·√(residual variance)`, clamped
  at 0, drawn from the endpoint's `[error_model]` — instead of (or, per-monitor, alongside) the
  latent `Ipred`. This is the realistic TDM / titration signal. The assay draws come from a
  per-purpose RNG substream keyed by `(subject, replicate, decision, analyte)`, so they are
  deterministic under a fixed seed, invariant to subject ordering, and never perturb another
  monitor's (or η's) draws.
- **Declarative `[adaptive_dosing]` model-file block — `simulate_adaptive_from_spec()`**
  (#584, epic #391). A reactive dosing *policy* can now be written in the model file — an
  `observe` signal expression, a decision schedule (`at`), `start_dose` / `route` /
  `dose_bounds`, an optional `confirm` debounce and discrete `levels` ladder, and a
  first-match-wins ladder of `when signal <op> value : increase/decrease/hold/stop` rules —
  and run with `simulate_adaptive_from_spec()`, no controller code required. It compiles to
  the same reactive engine, dose ledger, decision log, RNG substreams, and frozen-replay
  verifier as the programmatic `simulate_adaptive()`; titrating on the assay-noised
  measurement (`with_assay_error`) reuses the `Dv` substream. Example
  `examples/adaptive_tdm_titration.ferx`. See [Adaptive dosing](model-file/adaptive-dosing.qmd).
- Warn when no estimation method is set in the model file's `[fit_options]` or by
  the caller, making the implicit fallback to FOCEI visible instead of silent (#558).
- Support NONMEM-style `block_sigma` residual covariance under SAEM for ordinary
  Gaussian paired-endpoint models (#548).
- **Built-in zero-order absorption — `zero_order(dur)`** (#504). A new `[odes]`
  input-rate function delivering the dose at a constant rate `F·Dose/dur` over the
  window `(0, dur]` (a zero-order infusion whose duration is an estimated
  parameter, reusing the `RATE=−2`/`D1` modeled-duration machinery). Compose it
  with a hand-written `- KA*depot` for *sequential* (zero-then-first-order)
  absorption. Like the other absorption inputs it routes the dose through the
  forcing (bolus suppressed), supports `F`/lagtime/superposition, and requires an
  explicit ODE disposition (a `pk ... + zero_order(...)` model errors, pointing at
  `ode_template`). The hard cutoff at `tad = dur` is delivered exactly as a
  per-segment constant; `dur`'s gradient is finite-difference for now (the analytic
  boundary impulse is follow-up #530). Examples
  `examples/zero_order_absorption.ferx` and `examples/sequential_absorption.ferx`.
- **Biphasic / parallel absorption via a pathway-fraction multiplier** (#388). An
  `[odes]` input-rate function may now be scaled by a declared individual parameter
  (`FR*igd(...)`), and more than one input-rate term may feed a compartment — so the
  Freijer & Post biphasic inverse-Gaussian model is written as
  `d/dt(central) = FR1*igd(...) + FR2*igd(...)`, splitting the dose across two
  pathways. The multiplier must be a single declared parameter (not an expression
  like `(1-FR)`), so a two-pathway split declares a complementary fraction
  (`FR2 = 1 - FR1`); the fit-time check enforces `0 < FR ≤ 1` and that the fractions
  on a compartment sum to 1. The fraction's gradient is exact (analytic `Dual2`).
  Example `examples/biphasic_igd_absorption.ferx`. (A fraction on `zero_order(...)`,
  i.e. the `mixed`/`parallel` zero-order family, is not yet supported — follow-up
  #505.)
- Support NONMEM-style `block_sigma` residual covariance across paired same-time
  multi-endpoint observations under FOCE (#546).
- Support fixed residual-error correlations via `block_sigma` for FOCE combined-error
  models, with a NONMEM `$SIGMA BLOCK(2) FIX` validation example (#537).
- **Analytic FOCE/FOCEI gradients for Form C readouts that reference covariates** (#540).
  An ODE Form C readout (`[scaling] y = <expr>`) that branches on or scales by a covariate
  — e.g. a free→total protein-binding readout gated on a `FREE` assay flag — now gets the
  exact analytic `Dual2`/`Dual1` gradient instead of falling back to finite differences.
  Covariates carry no parameter derivative in the individual-parameter dual basis the ODE
  sensitivity provider seeds, so they thread into the dual readout as constants from the
  per-observation covariate snapshot (consistent with #535/#538), for both the static and
  time-varying-covariate walks. θ or η referenced *directly* in a Form C readout (rather
  than via an `[individual_parameters]` entry) still falls back to FD. Validated on the
  `fluconazole_radboudumc` readout shape (free/total fluconazole with saturable
  albumin-dependent protein binding): the analytic `∂f/∂η`/`∂f/∂θ` match the production
  predictor and its central finite differences to ~1e-6 for both subject-static and
  per-observation `FREE` snapshots (`ode_provider_form_c_*` tests).
- **`[data_selection]` string equality on label columns, mirroring NONMEM `IGNORE(C.EQ.C)`**
  (#536). A `==`/`!=` condition may now compare a covariate column against an unquoted
  label, matched against the raw cell value — so a non-numeric comment-flag column (the
  NONMEM convention of a `C` column holding the literal `C`) is dropped correctly:
  `ignore = C == C`. The bare shorthand `ignore = C` expands to `C == C`. A non-numeric
  value against a *standard* numeric column (e.g. `DV == 0.O01` with a letter O) is now a
  parse error rather than a silent never-matching no-op, and a clause referencing a column
  absent from the data emits a `W_FILTER_COLUMN_ABSENT` warning instead of fitting
  unfiltered data silently.
- **Exact analytic FOCE/FOCEI gradients for η-dependent `ExpressionScale` `obs_scale`** (#486),
  on **both** the analytical 1-/2-/3-cpt path (inner EBE gradient) and the user-`[odes]` path
  (outer θ/Ω/σ gradient **and** inner EBE gradient). A divisor scale such as `obs_scale =
  1000 / V` (with `V` carrying IIV) previously routed parts of the gradient to finite
  differences: on the analytical path the per-subject inner EBE loop reverted to FD (the outer
  was already analytic), and on the ODE path *both* loops did. The provider now applies the
  scale's quotient rule `∂(f/s)/∂x = (∂f/∂x)·s⁻¹ − f·(∂s/∂x)·s⁻²` (`x ∈ {η, θ}`) over the
  differentiable scale program — the η-block for the inner loop, the full `(θ, η)` jet
  (including second-order blocks) for the outer — applied once per subject on the final
  prediction jet. The same `apply_expression_scale_*` routines now serve the closed-form and
  ODE providers. Result-neutral (estimates and SEs unchanged; this removes FD steps, so the
  affected fits are faster and report the gradient method as "analytic"). On the ODE path the
  scale is served on the static walk only — combined with **LTBS** or **time-varying
  covariates** it still routes to FD, as does IOV + `ExpressionScale`. As a consequence the
  SAEM/Bayes HMC sampler now takes its gradient-based path (rather than the gradient-free
  Metropolis fallback) for closed-form `ExpressionScale` models. Validated analytic ≡
  production + finite differences (ODE outer), and light ≡ full provider (both inner loops).
- **Exact analytic FOCE/FOCEI gradients for steady-state (SS=1) ODE dosing** (#439). User-
  `[odes]` models with a steady-state dose now get exact analytic gradients instead of
  finite differences. NONMEM SS=1 loads the compartments with an infinite-past pulse
  train's trough; there is no closed form for a general ODE, so production expands it as a
  *finite* `(apply dose; integrate II)` loop — running that same loop over the dual type
  propagates `∂(steady state)/∂(θ,η)` directly (no implicit fixed-point differentiation).
  Both SS **boluses** and SS **infusions** are supported (an SS infusion equilibrates with
  an active-rate window + quiet window per cycle), and SS composes with time-varying
  covariates, IOV, and EVID 3/4 resets. Routes to FD: a rate-defined SS infusion under
  `F ≠ 1` (its equilibration cycles would each need the `F`-scaled active window), and SS
  combined with an **estimated lagtime** (observations in the pre-arrival window
  `[t_dose, t_dose+lag]` must read the previous interval's steady-state tail, which the
  dual walk does not yet seed — production handles it via `ss_state_at_phase`). Result-
  neutral. **NONMEM comparison:** the SS=1 semantics this differentiates (the infinite-past
  pulse-train trough) are the production predictor's, NONMEM-validated for SS dosing in
  `docs/model-file` / `tests/`; the analytic gradient is the exact derivative of that
  NONMEM-matching prediction (FD-confirmed via `check_vs_production` / `predict_iov`).
- **Analytic gradients for rate-defined infusion under bioavailability `F ≠ 1`** in
  `[odes]` models (#419). NONMEM holds a rate-defined infusion's rate and scales its
  *duration* to `F·amt/rate`, so `F`'s sensitivity is a moving window boundary rather than
  a rate-magnitude scale — previously this routed to finite differences. The event-driven
  walk now carries it: the bioavailable window length `F·amt/rate` is the rate-off
  saltation boundary (combined with any lagtime shift), with the rate held. Such subjects
  route to the event-driven walk automatically. (A *steady-state* rate-defined infusion
  under `F ≠ 1` still uses FD.) Result-neutral.
- **Exact analytic FOCE/FOCEI gradients for IOV `[odes]` models** (#439). User-ODE
  models with inter-occasion variability (`iov_column`, `kappa`) now get the exact
  analytic outer (θ/Ω/σ) gradient over the stacked `[η_bsv, κ₁..κ_K]` random effects,
  via the event-driven `Dual2` walk seeded with per-occasion κ axes (the same walk the
  time-varying-covariate path uses, fed per-occasion parameters). Previously these fell
  back to finite differences. First cut covers bolus dosing, **with or without
  time-varying covariates** (each event is seeded at its own occasion × covariate
  snapshot); out-of-scope subjects (infusion, steady state, resets, lagtime, scaling/LTBS,
  IIV-on-residual-error, survival/TTE, or `n_θ + n_η + K·n_κ > 16`) route to FD as before.
  The inner EBE loop also uses an exact analytic stacked-η gradient (a light first-order
  walk), under the same model-level exclusions as the outer (it shares the
  `gradient = fd` / escape-hatch / `iiv_on_ruv` / FREM / TTE bails); the IOV outer is
  assembled per subject (exact analytic where in scope, per-subject reconverged-FD
  elsewhere), so one out-of-scope subject no longer forces the whole fit onto FD.
  **NONMEM comparison:** this is a gradient swap on the IOV FOCEI objective that is itself
  NONMEM-validated — `tests/warfarin_iov_nonmem.rs` (`iov_objective_matches_nonmem`,
  `iov_individual_cl_matches_nonmem`; OFV within ~0.6 units, all (ID,OCC) CL within 6.6%)
  and `docs/model-file/iov.qmd`. The analytic gradient is result-neutral against finite
  differences of that same objective / the production predictor and `predict_iov`.
- **Exact analytic inner EBE gradient for closed-form IOV models** (#439). The inner
  EBE optimisation for analytical 1-/2-/3-cpt IOV models now uses an exact analytic
  stacked-`[η_bsv, κ₁..κ_K]` gradient (a light first-order event-driven walk) instead of
  finite differences, matching the ODE IOV inner. Both IOV paths — closed-form and ODE
  — now have analytic gradients on the inner and outer loops. Result-neutral (validated
  against the second-order outer walk and finite differences of the inner objective).
- **Exact analytic FOCE/FOCEI gradients for ODE models with an estimated lagtime**
  (#439). User-`[odes]` models with an estimated lagtime — bare `LAGTIME`/`ALAG` **or**
  compartment-indexed `ALAG{n}` — now get the exact analytic outer (θ/Ω/σ) gradient and
  inner EBE η-gradient instead of finite differences. Lagtime is an *event-time*
  sensitivity (the dose arrives at `t_dose + lagtime`); it is handled on the event-driven
  walk via a per-dose event-time saltation injected at each dose and propagated through
  the per-event parameters, so it is **exact across occasion / covariate boundaries and
  for per-compartment (non-uniform) lags** — and **fully analytic, with no finite
  differences** (the one non-parameter-dual piece, the trajectory curvature `J·ẋ`, comes
  from a directional RHS evaluation). Composes with **time-varying covariates, IOV, EVID
  3/4 resets, and finite-duration infusions** (for an infusion the window `[t+lag, t+lag+
  dur]` shifts, so the saltation is applied at both rate boundaries). Lagtime + steady-
  state dosing routes to FD (pending the separate SS feature). Result-neutral — validated
  against the closed-form analytical twin (full Hessian), the production predictor (incl.
  TV-cov, `ALAG1`, reset, infusion), and finite differences of `predict_iov` / the
  population objective. **NONMEM comparison:** the lagtime semantics this differentiates
  (dose/absorption shifted to `t_dose + ALAG`) are the production predictor's, validated
  against NONMEM in `docs/model-file/lagtime.qmd` (NONMEM equivalence); the analytic
  gradient is the exact derivative of that NONMEM-matching prediction (FD-confirmed).
- **Event-driven analytic ODE sensitivities now cover EVID 3/4 resets and finite-duration
  infusions** (#439). The TV-covariate / IOV event-driven sensitivity walk previously
  declined subjects with a reset or an infusion (→ finite differences); it now zeros the
  dual state at each reset (EVID=4 = reset + dose) and applies the per-event `F·rate`
  forcing over each infusion window, so TV-cov and IOV models with resets or infusions get
  exact analytic gradients. Result-neutral.
- **`[initial_conditions]` block for analytical PK models** (#521). Declare a
  non-zero starting compartment amount with `init(central) = <expr>` (or
  `init(depot) = ...`) on a closed-form 1-/2-/3-cpt model — the analytical
  equivalent of NONMEM's `A_0(cmt)` and of the ODE-path `init(...)` in `[odes]`.
  A pre-dose baseline (e.g. `init(central) = CONC0 * V`) no longer forces the
  numerical ODE solver: on the 6-thioguanine `run14` model this cuts FOCEI wall
  time ~13× (27 s → ~2 s) at matching estimates. Non-IOV init models use exact
  analytic FOCE/FOCEI gradients under `gradient = auto` (#524); IOV init models
  use `gradient = fd` for now. Edge cases are handled explicitly: the baseline is wiped by a
  system reset (`EVID = 3/4`), its decay uses each occasion's PK parameters under
  IOV, a `KAPPA_*` reference in the init expression is rejected, and the
  combination with a steady-state dose (`W_STEADY_STATE_INIT`) or a compartment
  `[derived]` reference (`W_DERIVED_INIT_ANALYTICAL`) warns rather than silently
  mispredicting. See [Initial Conditions](model-file/initial-conditions.qmd).
- **Datasets whose TIME column does not start at zero** (#573). ODE models now
  begin integration at each subject's first record (matching NONMEM) instead of
  at a fixed `t = 0`, so a subject whose first TIME is off-zero is no longer
  integrated over a phantom `[0, first_record]` window. TIME stays on the raw
  data clock everywhere — the model `TIME`/`T` builtin, `[derived]` columns,
  sdtab/predict/simulate output, and the survival left-truncation `TENTRY` all
  report the value in the data file; no per-subject time shift is applied.

### Fixed
- **A `one_cpt_transit` model with a `TIME`-dependent structural parameter or time-varying
  covariates now works** (#486). The transit closed form assumes constant parameters over
  each absorption window, so it cannot serve a subject whose parameters switch mid-profile;
  previously such a model was rejected (`TIME` / TV covariates) or, on one internal path,
  produced a silently wrong all-zero gradient. For a plain `cl/v/n/mtt` transit model the
  parser now builds its exact ODE `transit()` equivalent — `d/dt(central) = transit(n, mtt)
  − (CL/V)·central`, `obs_scale = V`, validated to predict identically to the hand-written
  ODE twin — and the prediction / gradient dispatch routes only the subjects the closed form
  cannot serve (a `TIME` switch, or time-varying covariates) to it, keeping the fast, exact
  closed form for every constant-parameter subject. Transit forms outside the equivalent's
  scope (a `lagtime=`/`f=` mapping, a custom `[scaling]`, or an `[initial_conditions]` block)
  carry no equivalent and are still rejected up front (`fit()` errors; `predict()`/
  `simulate()` panic) rather than mis-predict — write the ODE `transit()` model directly for
  those. Follow-up: the sdtab compartment/state (`[derived]`) columns for such a subject now
  come from the ODE equivalent too (previously they were `NaN` because the states path did
  not route to the equivalent, even though IPRED did).
- **Finite / modeled-duration infusions combined with a time-varying covariate that
  changes across the infusion's end** now get an exact analytic second-order gradient
  (#486). The rate-off boundary sits between records, so the RHS Jacobian jumps there;
  the closed-form rate-off saltation assumed a single parameter set and dropped the
  `(J⁺ − J⁻)·x` curvature term, biasing the FOCEI Hessian / covariance-step SEs by a few
  percent (first-order gradient and OFV were unaffected). The infusion end now uses the
  same general `g⁻ − g⁺` saltation as the zero-order window end. Cases without a covariate
  varying across the infusion end are unchanged.

### Changed
- For `block_sigma` correlated residual models, the SAEM reported OFV (the
  FOCE-approximation used for AIC/BIC) now follows the `interaction` flag like
  FOCE/FOCEI instead of always using the non-interaction marginal: with
  interaction on (the default) it reports the dense interaction marginal
  (e.g. 18.7221 on the `correlated_residual_combined` anchor, matching ferx
  FOCEI) rather than the previous non-interaction value (18.7274) (#616). The
  off-diagonal correlation is carried in both cases; only the marginal's
  curvature term changed.
- **SAEM now warns on non-mu-referenced individual parameters instead of listing detected
  mu-referencing** (#621). The broad `mu-ref: ...` info notice is replaced by a SAEM-only
  warning that names any individual parameter whose random effect is not mu-referenced
  (e.g. `CL = TVCL + ETA_CL` rather than `CL = TVCL * exp(ETA_CL)`), since such forms can
  strongly slow SAEM convergence. The warning fires whenever the estimation chain runs SAEM,
  independent of the `mu_referencing` fit option.
- **IOV occasions with doses but no observations now contribute their own κ random-effect
  axis** (#590). Occasion grouping (`iov_occasion_groups`) now includes every occasion in
  the dose record, not only those carrying sampled observations, so a dose-only occasion
  (e.g. a loading dose with no PK samples) adds an IOV κ axis and a `k_occasions·log|Ω_iov|`
  prior term. This shifts converged OFV / estimates / SEs for datasets with dose-only
  occasions versus prior versions; it is intended (carryover means such an occasion's κ is
  still informed by later observations).
- **FOCE + M3 BLOQ + IOV no longer silently promotes censored subjects to interaction**
  (#591). Under `method = foce` (non-interaction), an IOV subject with `CENS != 0` rows is
  now scored with a consistent Sheiner–Beal objective for the whole subject — the censored
  rows leave the linearized marginal and re-enter as `−logΦ((LLOQ−f)/√R⁰)` at the
  population (η=0, κ=0) variance — instead of being evaluated with η-interaction. This
  mirrors the non-IOV FOCE-M3 change (#367) and matches NONMEM `METHOD=1 LAPLACE` with vs
  without `INTER`: FOCE-IOV-M3 and FOCEI-IOV-M3 are genuinely different optima. **Fits that
  relied on the old auto-promotion should set `method = focei` explicitly.** The FOCE-M3
  notice — which described the now-removed promotion ("evaluated with η-interaction") — is
  reworded to state the non-interaction (Sheiner–Beal) semantics accurately (#599).

### Removed
- The `covariance_ofv_hessian` fit option and the analytical-gradient covariance
  R-matrix stencil it selected (`covariance_ofv_hessian = false`) have been removed.
  The covariance R-matrix is now **always** built from second differences of the
  reconverged marginal OFV — the accurate, envelope-free stencil that recomputes the
  full marginal curvature (`a = ∂f/∂η` and the `log|H̃|` EBE-response) at every
  perturbed point. The old analytical stencil held `a` fixed and biased the SE of
  weakly-identified structural parameters; its exact form requires third-order
  sensitivities (tracked separately). Models that set `covariance_ofv_hessian` should
  drop the key (it is now an unknown option) (#639).

### Fixed
- **Optimizer trace is now flushed to disk after every row** so live consumers
  (e.g. the ferx-r trace UI) see iterations as they happen. The `TraceWriter`
  wrapped the file in a `BufWriter` and only flushed at `finish()`; high-volume
  methods (SAEM) filled the buffer and streamed incidentally, but gradient
  methods (FOCE/FOCEI/GN) emit few rows (smaller than the buffer) so the trace
  file did not appear until the fit completed.
- **Gradient optimizers no longer fail on a first-step overshoot into the EBE guard**
  (#486). When the outer optimizer's inner EBE loop rejected a trial step (too many
  unconverged subjects, or a non-finite OFV), the objective was clamped to a flat `1e20`
  while the gradient was set to a non-zero "push back toward the bound centre" vector — an
  objective/gradient pair NLopt's L-BFGS / SLSQP line search cannot reconcile (the slope of
  a constant is zero). For most fits this was harmless because the guard only triggers deep
  in the run; but a model whose **first** optimizer step overshoots straight into the guard
  (notably ODE models with `iiv_on_ruv`, where a large step diverges the inner EBEs and
  overflows the `exp(2·η_ruv)` marginal) failed on iteration one and never moved off the
  initial estimates. The guard now returns a quadratic penalty whose gradient *is* the
  push-back vector, so the line search backtracks to a feasible step and the fit proceeds.
  This makes the analytic **M3 + IOV + `iiv_on_ruv`** triple on ODE models (#486) converge
  under the default gradient optimizer, matching the closed-form fit to estimator precision.
- **Estimation-method chains now run the covariance step only once, at the end of
  the chain** (#615). When a chain ended in a *default (estimating)* IMP stage
  (e.g. `methods = [saem, imp]`), both the preceding estimator and the IMP stage
  computed the (expensive) finite-difference covariance matrix — the trailing-IMP
  heuristic incorrectly treated every trailing IMP as an evaluation-only stage.
  The covariance / SIR step now runs only on the last *estimating* stage;
  evaluation-only IMP (`imp_eval_only`) still cedes the step to the preceding
  estimator as before. Plain chains without IMP were already correct.
- **M3 BLOQ above-ULOQ (right-censored, `CENS = -1`) handling under FOCE and the analytic
  gradients** (#591). The non-interaction FOCE marginal (`foce_subject_nll_standard`) and
  the analytic FOCE/FOCEI censored-row sensitivities (inner EBE gradient, outer
  θ/Ω/σ gradient, and the `iiv_on_ruv` cross-terms) hardcoded the lower (below-LLOQ) tail,
  so an above-ULOQ observation was scored and differentiated with the wrong normal tail —
  giving a wrong FOCE objective and a wrong-signed EBE/parameter gradient for any dataset
  with `CENS = -1` rows. The censored kernels and the FOCE marginal are now tail-aware
  (selecting `z = (f − ULOQ)/√v` for `CENS < 0`, matching `m3_logcdf`). This also repairs
  the pre-existing **non-IOV** M3 right-censored gradient/objective (the bug predated the
  IOV work). Left-censored (`CENS = 1`) results are unchanged.
- **A time-dependent individual parameter written with the `TIME` built-in
  inside a conditional-expression RHS now switches** (e.g.
  `MAINT = if (TIME > 45) 1 else 0`). The "uses TIME" flag that routes such a
  model through the per-event evaluation path was computed *after* the
  individual-parameter statements were bytecode-compiled — a step that replaces
  the `TIME` node with an `Op::PushTime` op the flag's AST scan can no longer
  see. The flag therefore read `false`, the analytical path evaluated PK
  parameters once at `t = 0`, and the parameter never changed over time (the
  effect collapsed and its θ became unidentifiable). The flag is now computed on
  the pre-compilation AST. A `TIME` reference inside a full `if { … }` statement
  block was unaffected; only the conditional-expression form regressed
  (introduced with the `TIME` built-in in #610).
- **A trough observation listed before a same-TIME dose is now evaluated
  pre-dose**, matching NONMEM's record-order semantics. The data reader ordered
  events by time and, at an equal TIME, placed the dose first on every path (the
  event-driven sort and the analytical superposition gate alike), so an
  observation sharing a dose's timestamp was scored as a post-dose peak instead
  of the pre-dose trough the data intended. On trough-rich datasets this railed
  fits to their bounds. The reader now honors data record order: an observation
  written before its coincident dose sorts just before that dose, while a
  post-dose observation (dose row first) and steady-state doses are unchanged.
  The raw user-clock TIME reported in sdtab/covtab and by
  `predict()`/`simulate()` is unaffected. With both fixes the infliximab run55
  benchmark — which had railed to its bounds (eval-at-NONMEM-estimates OFV 3751
  vs NONMEM 662; FOCEI and SAEM both converging to nonsense) — reproduces NONMEM:
  FOCEI OFV 664.0 vs 662.2, TVCL 0.198 vs 0.199, maintenance-phase CL multiplier
  1.41 vs 1.40.
- **Joint PK-TTE fit now rejects a non-monotone (negative) cumulative hazard** (#564).
  A drug-driven `hazard =` expression is unconstrained, so a sign-flipped hazard could make
  the cumulative hazard *decrease* — implying a survival `S(t) > 1`. The right-censored and
  exact-event likelihood terms previously accepted this silently (a finite, spuriously low
  objective that could pull the optimizer into the ill-posed region); they now return the
  same `1e20` sentinel as the other ill-defined cases. This matches the simulation path,
  which already hard-errors on a non-monotone cumulative hazard.
- ODE+IOV fits now report their actual analytic-vs-finite-difference inner-gradient route,
  including subject-level fallback reasons, instead of using the non-IOV gradient probe
  for diagnostics (#590).
- ODE+IOV models with an expression `[scaling] obs_scale` and time-varying covariates
  now stay on the analytic inner/outer gradient route instead of falling back to finite
  differences (#590).
- ODE+IOV models with EVID=2 covariate-only breakpoints now keep analytic inner/outer
  gradients when otherwise in scope; the breakpoint updates the ODE segment PK snapshot
  with κ fixed at zero, matching production prediction semantics (#590).
- ODE+IOV models with many occasion blocks or dose-only occasions now keep analytic
  inner/outer gradients when otherwise in scope, covering per-subject stacks up to 96
  axes (#590).
- Wide ODE+IOV analytic gradients now run on larger Rayon worker stacks, avoiding
  native stack-overflow crashes in R/CLI release builds for PNA-scale occasion counts
  (#590).
- ODE+IOV fits no longer launch Nelder-Mead EBE fallback searches for bad outer
  trial points, and RK45 now exits repeated **non-finite** minimum-step clamps early,
  avoiding apparent stalls after rejected LBFGS steps in PNA-scale models while leaving
  finite-but-stiff segments to integrate normally (#590, #603). Subjects rejected at a
  pathological inner start now force the outer trial to be rejected outright — including
  in the SLSQP fallback — so a degenerate EBE can no longer bias an accepted OFV (#603).
- Standard errors for `theta` parameters with a **negative lower bound** (estimated on
  the natural scale — e.g. exposure–hazard slopes, covariate exponents) are no longer
  mis-scaled (#564). The delta-method back-transform `SE(θ) = θ·SE(log θ)` was applied to
  every theta, but it only applies to log-packed (non-negative) parameters; for
  natural-scale thetas the reported SE was multiplied by the estimate (and would flip sign
  for a negative estimate). Such thetas now report `SE = SE(packed)` directly. Surfaced by
  the joint PK-TTE anchor, where `BETA`'s SE matched NONMEM only after the fix.
- **Custom residual-error magnitude (#484) now applies on every path, not just
  the FOCE/FOCEI objective** (#576). The per-observation multiplier was wired
  only into the OFV and silently dropped everywhere else, all without a guard:
  `simulate()`/`--simulate` and NPDE drew residual error with a constant SD; the
  sdtab IWRES/CWRES columns (and downstream VPC/goodness-of-fit) were mis-scaled
  wherever the magnitude departed from 1; an ODE model under `focei` ran its
  inner EBE loop with an analytic gradient that omitted the multiplier
  (mismatched against the magnitude-aware objective → biased η̂ and estimates);
  and a mixed PK+TTE model dropped the multiplier on its PK rows. All four paths
  are now magnitude-aware. The parser also now rejects a magnitude expression
  that references an **undeclared covariate** (including typos) even when the
  model has no `[covariates]` block — previously such a name silently evaluated
  to 0 and collapsed the multiplier to a constant.
- **TTE frailty ω² on a nonlinear hazard parameter now converges onto the
  NONMEM/nlmixr2 consensus** (#469). The derivative-free `bobyqa` outer optimizer
  false-converged on the near-flat ω² ridge — its `ftol_rel` default (`1e-6`)
  stopped it short of ferx's *own* objective minimum, so a Weibull shape-frailty
  read ω² 0.204 against the NONMEM LAPLACIAN 0.175 / nlmixr2 0.173 consensus on
  identical data. The TTE objective is evaluated exactly, so its `ftol_rel` is now
  auto-tightened to `1e-8` (it lands 0.176); non-TTE fits keep `1e-6` to avoid
  grinding on noisy ODE/FD-inner objectives. This is a pure optimizer-convergence
  fix and does not touch the separate FOCEI-Laplace *method* bias (#440).
- A diverged IMP/IMPMAP run is no longer reported as converged (#528). A
  collapsed-weight runaway pins θ to the parameter bounds and the final objective
  blows up to a finite-but-enormous value (~1e35); the convergence check only
  tested `is_finite()`, so such a run could be flagged converged and even win
  multi-start selection. It is now treated as diverged.
- `outer_maxiter = 0` (NONMEM `MAXEVAL=0`) now means *evaluation only* on every
  optimizer (#562). The gradient NLopt path (`nlopt_lbfgs`/`slsqp`/`mma`) passed
  `maxiter = 0` straight to NLopt's `set_maxeval`, where `0` means **no limit** —
  so a `maxiter = 0` request silently ran a *full* fit and reported a converged,
  optimizer- and platform-dependent OFV instead of the objective at the initial
  parameters. All optimizers now route through a single eval-only path that runs
  one inner EBE solve at θ₀ and reports `2·NLL` there (covariance step still
  honoured). This is what surfaced as the `two_cpt_oral_cov_ode` ODE-vs-analytical
  "init OFV" diverging ~534 on x86 Linux in the ferx-r equivalence tests.
- FOCEI now falls back to finite-difference h-matrices when an ODE analytic
  Jacobian is unavailable or non-finite, avoiding sentinel-inflated OFVs on sparse
  subjects such as the pembrolizumab RadboudUMC model (#551).
- Reject `block_sigma` with IOV until the IOV inner objective supports the full
  residual covariance matrix, use shifted times when pairing reset-segment
  residual blocks, and keep FREM CWRES variances unscaled by `iiv_on_ruv`
  (#549).
- **Inner EBE optimizer no longer spuriously fails on ODE objectives, fixing a wrong OFV
  for η-dependent `[scaling] obs_scale` models** (#555). The per-subject empirical-Bayes
  BFGS stopped only on its gradient norm, but an adaptive-ODE-solver objective puts a
  noise floor on the gradient that can sit above the inner tolerance — so a search that
  had already reached the mode spun to `max_iter` and reported failure. The inner loop
  then discarded the correct estimate and restarted Nelder–Mead from η=0, which on a
  multimodal inner objective (e.g. `obs_scale = V1` with `V1 = … · exp(ETA_V1)`) settled
  in a worse local minimum and inflated the FOCEI objective (≈370 OFV on the
  `two_cpt_oral_cov` example; its analytical twin was unaffected). Two changes fix it, both
  scoped to ODE objectives so analytical, event-driven, FREM and finite-difference fits stay
  bit-identical to before: for ODE models the inner fallback (both the BSV and the IOV paths)
  now keeps the **lower-objective** of the BFGS partial and the Nelder–Mead restart instead
  of blindly overwriting with NM, and the inner BFGS gained an objective-stall stop so it
  converges at the mode rather than spinning. Exact objectives have no gradient-noise floor,
  so a BFGS failure there is genuine non-convergence and the historical NM-from-η=0 recovery
  is retained. The ODE form's OFV-at-init now matches its analytical twin (`−1193.59`,
  previously `−823.05`), at the default ODE tolerance, and affected subjects converge in far
  fewer inner iterations. *Note:* with `ebe_warm_start` off (the default), an **ODE** fit that
  hits the inner fallback with a BFGS partial that beats the η=0 restart now returns that
  partial rather than the NM-from-0 result, so a previously fallback-stalled EBE/OFV may shift
  toward the better optimum.
- **Form C (`[scaling] y = <expr>`) ODE readouts now use per-observation covariate
  snapshots** (#535, #538). The explicit-output readout is evaluated against the
  covariate values on each observation's own data row rather than the subject's
  first-row values, so time-varying covariates referenced in a Form C expression
  now drive predictions, diagnostics, and the adaptive-trial decision monitors at
  the correct time. As a consequence, covariates referenced **only** from a Form C
  expression are now treated as required data columns: a model whose readout
  references a covariate absent from the dataset now fails loudly with
  `E_MISSING_COVARIATE` (and undeclared-but-present covariates raise the usual
  warning), where previously the missing value silently read as `0.0`. **NONMEM
  comparison:** validated against the `fluconazole_radboudumc` model (ADVAN3 TRANS4
  with a free/total protein-binding `$ERROR` that selects `CTOT` when `FREE==0` and
  `CU` when `FREE==1` — paired assay rows at the same time). Evaluated at identical
  parameters, ferx's per-record population predictions match NONMEM's `PRED` to
  ~1e-4 relative on **both** the total-assay and free-assay rows (e.g. subject 1 at
  t=1: ferx 21.5105 / 2.9070 vs NONMEM 21.511 / 2.907), confirming the readout reads
  each observation's own `FREE` value rather than the subject's first row. (The two
  rows at a given time differ only by that per-record covariate.) For time-constant
  covariates the readout is byte-identical to the prior behaviour; the
  `ode_event_driven_form_c_uses_observation_covariates` unit test pins the
  per-observation path.
- **Gradient-based outer optimizers now precondition with magnitude scaling
  (`Abs`) instead of bound-half-width (`Rescale2`).** Under the default
  `optimizer = auto` (which resolves to NLopt L-BFGS when an analytic gradient is
  available), `Rescale2` was the wrong preconditioner and made FOCE/FOCEI
  converge to a parameter bound or a local minimum on several models — warfarin
  FOCEI stalled at OFV −243 (TVV 6.08) instead of −286 (TVV 7.74); a
  time-varying-covariate fit landed at a +166 local minimum with TVV pinned at
  its lower bound; SLSQP froze at its start on a 2-cpt covariate model. Switching
  the gradient-based optimizers (`bfgs`/`lbfgs`/`nlopt_lbfgs`/`slsqp`) to `Abs`
  scaling recovers the correct optimum in every case while preserving the SLSQP
  cold-start fix (#335). This fixes the downstream IMP/IMPMAP warm-start collapse
  and the simulation-based NPDE/NPD diagnostic, which inherited the bad fit.
  (Scaling is disabled automatically when an identity-packed covariate θ is
  present, as before.)
- **Exact analytic FOCE/FOCEI gradient for `iiv_on_ruv` (IIV on residual error).**
  Models with a residual-error eta (`Y = IPRED + EPS·EXP(η_ruv)`) now use the
  exact closed-form gradient on both the inner EBE and outer θ/Ω/σ loops, where
  the residual-eta column previously fell back to (and, with the `auto`/L-BFGS
  optimiser, silently mis-computed) a gradient that omitted the `exp(2·η_ruv)`
  variance scaling. The inner η-gradient scales `v`/`dv_df` and adds the
  `Σ(1−ε²/v)` residual-eta column; the outer assembly adds the Almquist `c̃=2`
  interaction column to `H̃`, the true-Hessian `2ε²/R` / `κⱼaⱼ` terms, and their
  `log|H̃|` θ/Ω/σ derivatives. Validated to ~1e-11 against reconverged finite
  differences of ferx's own FOCEI marginal (whose value is NONMEM-validated, #413).
  The assembly is provider-agnostic, so it covers the closed-form (analytical
  1-/2-/3-cpt), **ODE** (`[odes]`), and **LTBS** (`log_additive`) paths — for LTBS
  the outer gradient is analytic while the inner EBE keeps finite differences (the
  existing LTBS choice, #438). IOV and M3-BLOQ `iiv_on_ruv` keep the
  finite-difference gradient. (#474)
- **Spurious "not referenced" warning for the `iiv_on_ruv` eta.** A residual-error
  random effect is referenced from `[error_model]` (not an individual-parameter
  expression), so it was falsely warned as "declared but not referenced … will not
  affect predictions or be meaningfully estimated" even though it scales the
  residual variance and is estimated. The warning is now suppressed for that eta.
  (#474)

### Performance
- **Analytic sensitivity gradients for moving infusion-end boundaries: modeled
  duration / rate doses and `zero_order(dur)` absorption** (#530). Three dosing
  features previously routed both the outer (θ/Ω/σ) and inner (EBE η) FOCE/FOCEI
  gradients to finite differences because the infusion *end* time is a moving
  boundary in an estimated parameter: a `RATE=-2` (`D{cmt}`, modeled duration) or
  `RATE=-1` (`R{cmt}`, modeled rate) dose (end `t_dose + D` resp. `t_dose + amt/R`),
  and a `zero_order(dur)` absorption forcing (end `t_dose + dur`). The dual walk now
  resolves the modeled rate/window from its PK slot as a live jet and carries the
  boundary derivative via the rate-off **event-time saltation** — the exact
  sign-mirror of the estimated-lagtime dose-*start* saltation (#472). Modeled
  duration/rate doses ride the event-driven walk; `zero_order(dur)` is delivered as a
  per-segment constant window (like an infusion) on the static walk, with the
  saltation injected at its cutoff. So these fits take the exact `Dual2`/`Dual1`
  gradient (the estimates are unchanged; the gradient is faster and Hessian-clean).
  Validated against finite differences of the production predictor, with the modeled
  parameter η-coupled so both the θ- and η-blocks of the moving-boundary term are
  checked, plus inner/outer scope parity. Modeled duration/rate doses stay analytic
  when composed with an estimated lagtime (the start and end saltations carry the
  combined `δlag + δdur` shift), an EVID 3/4 reset, time-varying covariates, or
  multiple doses; `zero_order(dur)` stays analytic across multiple doses and a mixed
  (zero- + first-order) pathway. Still FD: a *steady-state* modeled dose or
  `zero_order` window (the SS equilibration reads a fixed per-cycle window), a modeled
  dose under IOV, and a `zero_order(dur)` *forcing* combined with an estimated
  lagtime, an EVID 3/4 reset, or time-varying covariates (which keep it on the static
  walk's FD fallback).
- **Joint PK-TTE fits integrate the augmented PK + cumulative-hazard ODE once per
  inner likelihood evaluation instead of twice** (#570). For a drug-driven hazard
  (`[event_model] hazard = …`), the cumulative hazard at the event/censor times is now
  read off the *same* integration as the Gaussian predictions by in-step cubic Hermite
  interpolation, rather than a second dedicated solve. The predictions are bit-identical
  and the hazard term is unchanged to integrator tolerance, so estimates and OFV are
  unaffected within the solver's accuracy — the only difference is speed. Applies to
  plain ODE PK-TTE subjects (no time-varying covariates, EVID-3/4 resets, SDE, or FREM,
  which keep the previous path).
- **Analytic sensitivity gradients for ODE IOV models with an `ExpressionScale`
  `obs_scale` divisor** (#575). An `[odes]` model combining IOV (occasion `kappa`)
  with an η-dependent `obs_scale = expr` (e.g. `obs_scale = V1`) previously routed
  both the outer (θ/Ω/σ) and inner (EBE η) gradients to finite differences; each
  feature was analytic alone (IOV #466, `ExpressionScale` #534) but not together.
  The divisor's exact quotient rule is now applied as a post-walk per-occasion-group
  jet over the stacked `(θ, η, κ)` axes, so these fits take the exact `Dual2`/`Dual1`
  gradient — faster and Hessian-clean. Validated against finite differences of the
  production predictor and against the equivalent Form-C readout (`y = central/V1`).
  Still FD: the combination with LTBS or time-varying covariates, and the
  closed-form (non-ODE) IOV path.
- **Convergence-based early stop for steady-state equilibration** (#519). The SS=1
  pre-equilibration (both the f64 predictor and the `Dual1`/`Dual2` gradient path,
  and the closed-form/event-driven SS loops) previously always expanded a fixed
  50-cycle `(apply dose; integrate II)` train. It now stops once the trough stops
  moving — a shared mixed `atol`/`rtol` test on the per-cycle increment
  (`|Δ| ≤ tol·|cur| + tol·max`, `SS_EQUILIBRATION_TOL = 1e-12`) applied identically
  across all paths, driven by the value parts so the dual truncates on the same
  cycle as the f64 path (making the gradient the exact derivative of the value the
  optimizer sees). The stop fires only after the value reaches its fixed point to f64
  precision: fast disposition converges in ~14 cycles (~3.5× fewer), slow PK still
  runs the full budget. **SS predictions are unchanged to f64 precision**; gradients
  and covariance SEs match a full-budget run to `< 1e-6` relative (a small derivative
  tail, ~`1e-8` even on a deliberately scale-separated 2-compartment model, contracts
  a constant few cycles behind the value) — 3–4 orders below the `1e-3` gradient
  validation tolerance, the `1e-9` ODE solver `reltol`, and NONMEM's ~`1e-5`
  SE-matching precision, i.e. invisible to every reported number. This was the
  dominant cost of analytic-gradient SS fits.
- **Exact analytic gradients for `[initial_conditions]` models** (#524). A non-IOV
  closed-form model with an `[initial_conditions]` baseline now runs FOCE/FOCEI
  on exact analytic `Dual2`/`Dual1` sensitivities under `gradient = auto` instead
  of falling back to finite differences: the init impulse `A₀ · kernel(t, pk)`
  and its θ/η dependence thread through the analytic provider (outer θ/η jet and
  inner η-gradient). Faster (no per-parameter FD probe) and exact, and it
  re-enables the HMC SAEM E-step (`n_leapfrog > 0`) for baseline models. The
  analytic gradient matches Richardson finite differences of the (NONMEM-validated)
  FOCEI marginal to ~1e-3. IOV init models keep the FD fallback (follow-up).
- **Exact analytic gradients for IOV + `iiv_on_ruv` models** (closed-form
  1/2/3-cpt, #486). An inter-occasion-variability model that also puts IIV on the
  residual error (`iiv_on_ruv`) now runs FOCEI on exact analytic sensitivities
  instead of finite differences: both the stacked-η inner gradient and the outer
  θ/Ω/σ assembly carry the `exp(2·η_ruv)` residual-variance scaling and the
  `η_ruv` variance column (the same treatment the non-IOV `iiv_on_ruv` path
  already used, #474). Faster (no per-parameter FD probe) and exact — the analytic
  inner gradient matches central FD of the IOV inner objective and the outer
  θ-gradient matches Richardson FD of the FOCEI marginal to ~1e-3. ODE IOV +
  `iiv_on_ruv` keeps the FD fallback (follow-up).
- **Exact analytic gradients for closed-form `iiv_on_ruv` + M3 BLOQ models**
  (#486). A model with IIV on the residual error *and* M3 below-quantification-
  limit handling now runs FOCEI on exact analytic sensitivities. The censored
  data term `−logΦ((LLOQ−f)/√v)` (with `v = R·exp(2·η_ruv)`) contributes the
  residual-eta column `h·z` and the cross-curvature `∂²L/∂η_ruv²`, `∂²L/∂η_l∂η_ruv`,
  `∂²L/∂η_ruv∂θ`, `∂²L/∂η_ruv∂σ` to the true inner Hessian and the mixed blocks,
  while censored rows stay excluded from the Laplace `H̃`/`log|H̃|` (matching the
  objective). Inner η-gradient vs central FD and the outer packed gradient vs
  Richardson reconverged FD of the censored FOCEI marginal both match to ~1e-3.
  **ODE** M3 + `iiv_on_ruv` keeps the FD fallback (not yet regression-tested).
- **Ω-preconditioned inner EBE loop for all FOCE/FOCEI fits.** The inner BFGS
  now initialises its inverse-Hessian (the search `H0`) to the prior conditional
  scale `diag(1/Ω⁻¹ᵢᵢ)` for every model, not just FREM. A correlated or
  multi-scale Ω (e.g. a block-Ω where one η has several× the variance of another)
  otherwise mis-scales the identity-`H0` search, costing extra inner iterations.
  The convergence *test* stays the raw L2 gradient norm for general fits (only
  FREM needs the preconditioned norm, issue #406), so `H0` changes only the path
  to the mode — the converged EBE and the estimates are unchanged. On the
  two-compartment UVM FOCEI/MMA benchmark this cuts inner BFGS steps per EBE
  solve ~25→16 and total predictions ~17M→6.2M for a **~1.23× faster fit**
  (single- and 8-thread) at the same optimum (OFV within 4e-5 of the prior
  result; matches NONMEM `run18`).
- **Interpolating inner-loop line search** (#462). The EBE BFGS line search now
  picks each trial step by safeguarded quadratic interpolation instead of fixed
  halving, and reuses the objective value the optimiser already tracks instead of
  recomputing it. On the two-compartment UVM FOCEI/MMA benchmark this cuts the
  average backtracks per line search from ~22 to ~3 (cap-exhaustion 20% → 0.1%),
  roughly halving the prediction-walk count for a ~2.5× faster single-threaded
  fit at the same optimum.
- Reuse per-thread scratch buffers when evaluating individual PK parameters,
  reducing allocator traffic in FOCE/FOCEI inner loops with time-varying
  covariates (#462).
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
- **Constant-fold covariate-only individual-parameter sub-expressions in the
  analytic sensitivity walks** (#485). The `[individual_parameters]` block is
  re-evaluated on every inner-EBE and outer-gradient step; for covariate-heavy
  models its covariate-only prefix (e.g. CKD-EPI / Schwartz / FFM / maturation —
  often the bulk of the `pow`/`exp`/`log` work) does not depend on θ or η, yet was
  carried through `Dual2`/`Dual1` arithmetic (gradient + Hessian per operation)
  every call. The parser now classifies those slots once at compile time and the
  `Dual2`/`Dual1` providers evaluate them once in plain `f64` and seed them as
  dual constants, skipping the redundant dual re-derivation. Numerically identical
  (bit-for-bit gradients and Hessians); only θ/η-free slots are folded, so all
  dual axes — including `∂/∂θ_fixed` — are preserved. On a jasmine-style
  covariate kernel (8/10 slots foldable) this is ~1.7× faster per `Dual2`
  individual-parameter evaluation. Found while profiling the jasmine
  vancomycin-pediatrics FOCEI fit.
- **Light `Dual1` inner η-gradient for analytical PK models** (#491). The inner
  EBE loop's `∂p/∂η` for analytical 1-/2-/3-cpt models was computed over the full
  `Dual2<n_theta + n_eta>` (carrying the θ-axes gradient and the second-order
  Hessian) and then all but the η-block discarded. It now uses the light
  `Dual1<n_eta>` walk the ODE inner loop already used (#410), seeding η only — so
  e.g. a 10-θ / 4-η fit drops a `Dual2<14>` (14-vector grad + 14×14 Hessian per
  op) to a `Dual1<4>`. Converged EBEs and OFV are unchanged (the inner gradient
  method only affects the path to the mode); validated by the existing
  analytic-vs-FD inner-gradient tests. Also serves models whose combined
  `n_theta + n_eta` exceeds the dual dispatch ceiling but whose `n_eta` does not
  (previously an FD fall-back).

### Added
- **Built-in Weibull absorption — the `weibull(td, beta)` input-rate function** (#322, Phase 2).
  Use it inside an `[odes]` RHS, with `td` (scale) and `beta` (shape) bound to
  `[individual_parameters]` (so they carry IIV / covariates for free):
  `d/dt(central) = weibull(td=TD, beta=BETA) - CL/V*central`. The dose is delivered as the
  Weibull density over time (`∫R_in dt = F·Dose`) and its bolus is suppressed — the same
  dose-into-the-input-rate-compartment convention as `transit()` / `igd()`. Shape `beta`
  selects the profile: `>1` a delayed interior peak, `=1` first-order absorption with
  `ka = 1/Td`, `<1` fast early uptake (an integrable spike at the dose). Weibull has no
  elementary closed form, so it always runs on the numerical ODE path and **requires an
  explicit ODE disposition** — combining it with an analytical `pk ...` is a clear error
  pointing at `ode_template`. Because the forcing is evaluated over `Dual2`, a `weibull()`
  model drives **exact analytic** FOCE/FOCEI/Bayes gradients (no finite-difference fallback),
  validated against NONMEM. See `examples/weibull_absorption.ferx` and
  `docs/model-file/absorption.qmd`.
- **Analytic FOCE/FOCEI gradients for compartment-indexed bioavailability
  (`F1`/`F2`, …) on ODE models** (#486). An ODE model that sets a per-compartment
  bioavailability now drives the exact analytic outer gradient and light `Dual1`
  inner η-gradient instead of finite differences: both the static and
  time-varying-covariate dual walks resolve `F` per dose compartment (the indexed
  `F{cmt}` slot, else the bare `F`), matching production's `DoseAttrMap::f_bio` and
  carrying `∂/∂F{cmt}` exactly. Estimates are unchanged; the gradient is exact and
  cheaper. Validated by an analytic≡production+central-FD parity test (single
  indexed `F1` with IIV, and distinct `F1`≠`F2` dosed into two compartments).
  Per-compartment *lag* (`ALAG{cmt}`) stays on FD for now (→ #472).
- **`ebe_warm_start` fit option** (default `false`, opt-in). When a per-subject
  inner BFGS solve fails and falls back to Nelder–Mead, seed the simplex from the
  BFGS partial η̂ instead of cold-starting from the prior mode η=0. On
  fallback-heavy fits (e.g. an unidentifiable peripheral volume that drives BFGS
  far onto the steep prior slope) NM then converges in a fraction of the
  iterations — ≈1.7× faster on a 2-cpt unidentifiable-V2 benchmark. Off by
  default because warm-starting moves the fallback subjects' EBEs, which perturbs
  the outer optimiser's trajectory: harmless for the BOBYQA default but can derail
  a gradient-based outer optimiser (e.g. `mma`) into a worse basin on some models.
  Validate OFV/estimates on your model + `optimizer` before enabling.
- **Competing-risks TTE (cause-specific hazards)** (#440). Multiple `[event_model NAME]`
  blocks on distinct compartments now model mutually-exclusive event types that share the
  model's random effects (a common frailty). `simulate()` draws the competing causes
  correctly — the earliest latent event is observed and the others are right-censored at
  that time — and `predict_survival()` gains a cause-specific cumulative incidence `cif`
  plus the all-cause survival `survival_all` (with `Σ_k cif_k(t) + survival_all(t) = 1`),
  the correct competing-risks quantities. Example `examples/tte_competing_risks.ferx`.
  Behind the `survival` feature.
- **`[simulation] horizon` for TTE / competing-risks VPC** (#522). A new
  `horizon = <t>` key sets an administrative censoring time that is *decoupled
  from the observed event times*: when present it overrides each TTE record's
  per-record observation window, so re-simulating event-bearing data (a VPC)
  censors every cause at the planned study end `t` instead of drawing unbounded.
  It is also honoured by the `[simulation]`-block `--simulate` path, which now
  generates one right-censored TTE row per cause compartment per synthetic subject
  (a TTE model under `[simulation]` therefore requires `horizon`); previously that
  path emitted zero TTE rows. Exposed on the library `SimulateOptions { horizon }`.
  Behind the `survival` feature.
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
- **`optimizer` now defaults to `auto`** (#490). The new `auto` choice picks the
  outer optimizer per model: `nlopt_lbfgs` when the exact analytic FOCE/FOCEI
  gradient is available, and `bobyqa` when only finite differences are (ODE/PD
  models, LTBS/SDE, or `gradient = fd`). Limited benchmarking across ~10 real
  FOCEI datasets found `nlopt_lbfgs` fastest-to-optimum on every analytic-gradient
  problem and `bobyqa` fastest and most reliable on the finite-difference ones, so
  `auto` gives most users a good default without tuning. The fit output reports
  the resolved pick as `auto (<resolved>)`; set `optimizer` explicitly (e.g.
  `optimizer = bobyqa`) to keep the previous fixed default.
- **The SLSQP fallback no longer triggers on `MaxEvalReached`** (#499). After the
  primary NLopt run (`nlopt_lbfgs`/`slsqp`/`mma`), ferx retried from the current
  point with a fresh, full-budget SLSQP optimization whenever the primary didn't
  report a clean convergence code — including when it simply hit the evaluation
  budget. A spent budget is not a failure a second optimizer can fix (it just
  doubles the cost); ferx now emits an "increase `maxiter`" warning and returns
  the best-seen point instead. The genuine-failure fallback (`Failure` /
  `RoundoffLimited`) is unchanged. Found during the jasmine FOCEI slowness
  investigation.
- **`optimizer = lbfgs` and `optimizer = bfgs` now select the NLopt L-BFGS**
  (`nlopt_lbfgs`) instead of the hand-rolled built-in BFGS / limited-memory L-BFGS
  (#483). Across analytic-gradient FOCEI benchmarks (jasmine, infliximab, uvm) the
  NLopt path reaches the best OFV and is 3–5× faster than the built-in, which on
  harder fits diverged (infliximab) or hung with no outer progress (busulfan
  ODE+IOV). The two keys are now deprecated aliases; the built-in implementation is
  slated for removal. The NLopt path's accuracy is validated against NONMEM/nlmixr2
  reference fits on the [Outer Optimizers](docs/estimation/optimizers.qmd) page
  (e.g. warfarin LTBS OFV −675.302, recovering NONMEM's MLE; `two_cpt_oral_cov`
  OFV −1197.23 ≈ nlmixr2's −1199.24).
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
- The `auto` optimizer now selects the derivative-free Bobyqa for time-to-event
  (`[event_model]`) objectives, which are finite-difference-only. The shared
  analytic-outer-gradient predicate previously reported a gradient for TTE (and
  mixed PK+TTE) models that the sensitivity provider cannot supply, so `auto`
  resolved to a gradient-based optimizer that stalled at the initial estimates;
  TTE fits with the default optimizer now converge (#490).
- **`[simulation]` block now honours the documented `n_subjects` / `dose_amt` /
  `dose_cmt` keys.** The parser previously only recognised the short
  `subjects` / `dose` / `cmt` spellings and **silently ignored** every other key,
  so all `examples/*.ferx` (which use the long forms) fell back to the defaults
  (10 subjects, dose 100, compartment 1) — e.g. `n_subjects = 12` simulated 10.
  Both spellings are now accepted (long forms canonical, short forms as aliases),
  and an **unknown or malformed key in `[simulation]` is now a hard parse error**
  instead of a silent default, matching `[fit_options]`.
- The ODE-solver fit options `ode_reltol`, `ode_abstol`, and `ode_max_steps` no
  longer emit a spurious "is not used by method … and will be ignored" warning
  (#516). They configure the RK45 integrator and *are* applied to any ODE model
  under every estimation method; they were simply missing from the warning's
  framework-key allowlist. Behaviour is unchanged — only the misleading warning
  is removed.
- Simulation, NPDE/NPD diagnostics, and the NCA-init grid sweep now honour
  time-varying covariate snapshots on dose, observation, and EVID=2 rows instead
  of using only each subject's baseline covariates (#506). FREM covariate
  pseudo-observations keep their additive `EPSCOV` error in simulation/NPDE
  rather than being fed through the PK residual-error model.
- **TTE simulation now applies administrative right-censoring** (#440). `simulate()`
  for a `[event_model]` (TTE) endpoint previously emitted *every* drawn event time as
  an uncensored event, so simulated data could not reproduce a study's censoring
  pattern (which broke simulation-estimation validation). A subject's administrative
  observation horizon is now honoured: a draw that reaches it is recorded as
  right-censored at the horizon (`observed = false`). The horizon is the
  `ObsRecord::Event` time of a *right-censored* record; an exact-event (or
  interval-censored) record carries no horizon — its `time` is the event time, not a
  censoring window — so it draws uncensored rather than being truncated at the
  realized event time (which would bias re-simulation / VPC). Left-truncated
  (delayed-entry) subjects draw conditional on survival past entry. Behind the
  `survival` feature.
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

### Fixed
- **`simulate()` now reproduces a fitted or fixed `block_sigma` cross-endpoint
  correlation** (#672): paired rows sharing a subject time and occasion (e.g.
  total/unbound assays, per-CMT or covariate-selected) are drawn from the dense
  residual covariance `R` instead of independent per-row normals, so a VPC or
  posterior-predictive check now recovers the specified residual covariance
  instead of understating it.

## [0.1.5] - 2026-06-01

Released before this changelog was started. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.5)
and `git log v0.1.0..v0.1.5` for details.

## [0.1.0] - 2026-05-29

Initial tagged release. See the
[GitHub release](https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0).

[Unreleased]: https://github.com/FeRx-NLME/ferx-core/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/FeRx-NLME/ferx-core/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/FeRx-NLME/ferx-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0
