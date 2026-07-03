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

[Unreleased]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/FeRx-NLME/ferx-core/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/FeRx-NLME/ferx-core/releases/tag/v0.1.0
