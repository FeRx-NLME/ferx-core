//! Tier-2 smoke tests for Phase 1 TTE (time-to-event) support.
//!
//! These exercise the public parse / `fit()` boundary.  They are NOT gated
//! with `slow-tests` — they must finish in a handful of outer iterations.
//! Full convergence tests live in `tests/tte_convergence.rs` (Tier 3).
//!
//! All TTE-specific items are behind `#[cfg(feature = "survival")]` so the
//! file compiles on every PR without the feature enabled (it just contributes
//! no test functions).

mod common;

#[cfg(feature = "survival")]
mod survival_smoke {
    use crate::common;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{DoseEvent, EventType, ObsRecord, Population};
    use ferx_core::{fit, EndpointLikelihood, FitOptions};

    // ── Model strings ────────────────────────────────────────────────────────

    /// Standalone exponential TTE model.  Kept with its legacy dummy 1-cpt structural
    /// block for historical reference; the block is never invoked (no CMT-1 observations).
    /// See `EXP_TTE_ONLY` below for the equivalent model using the compact TTE-only syntax.
    const EXP_TTE_MODEL: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)

  omega ETA_LAMBDA ~ 0.09

  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)

[fit_options]
  method  = focei
  maxiter = 3
";

    /// Fixed-effects (n_eta = 0) exponential TTE — validates the empty-Omega path.
    const EXP_TTE_FIXED: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)

  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)

  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[fit_options]
  method  = focei
  maxiter = 3
";

    // ── Population helpers ───────────────────────────────────────────────────

    // The one-record-per-subject `(time, dv)` TTE builder lives in the shared
    // `tests/common/mod.rs` as `common::tte_pop_from_pairs` (also used by
    // `tte_convergence.rs`) — call that instead of duplicating it here.

    // Synthetic data: 20 subjects, ~75% events, ~25% censored at t=30.
    const TTE_DATA: &[(f64, u8)] = &[
        (7.23, 1),
        (30.0, 0),
        (3.61, 1),
        (14.47, 1),
        (30.0, 0),
        (22.31, 1),
        (1.83, 1),
        (30.0, 0),
        (9.12, 1),
        (30.0, 0),
        (4.55, 1),
        (18.79, 1),
        (30.0, 0),
        (11.34, 1),
        (2.67, 1),
        (30.0, 0),
        (25.88, 1),
        (6.04, 1),
        (30.0, 0),
        (13.52, 1),
    ];

    // ── Tests ────────────────────────────────────────────────────────────────

    /// Parser must recognise [event_model] and populate model.endpoints.
    #[test]
    fn tte_exponential_model_parses() {
        let model = parse_model_string(EXP_TTE_MODEL).expect("EXP_TTE_MODEL must parse");

        // CMT 2 must be registered as a TTE endpoint.
        assert!(
            model.endpoints.contains_key(&2),
            "endpoints must contain CMT=2; got: {:?}",
            model.endpoints.keys().collect::<Vec<_>>()
        );
        match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte { hazard: _ }) => {}
            other => panic!("expected Tte endpoint for CMT=2, got: {other:?}"),
        }

        // n_theta = TVLAMBDA + DUMMY_CL + DUMMY_V = 3
        assert_eq!(model.n_theta, 3, "n_theta should be 3");
        // n_eta = ETA_LAMBDA = 1
        assert_eq!(model.n_eta, 1, "n_eta should be 1");
    }

    /// Parser must recognise [event_model] with family=weibull (scale + shape).
    #[test]
    fn tte_weibull_model_parses() {
        let src = r"
[parameters]
  theta TVSCALE(10.0, 0.1, 1000.0)
  theta TVSHAPE(1.5,  0.1, 10.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  SCALE = TVSCALE
  SHAPE = TVSHAPE
  CL    = DUMMY_CL
  V     = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE
  shape  = TVSHAPE
";
        let model = parse_model_string(src).expect("Weibull TTE model must parse");
        assert!(
            model.endpoints.contains_key(&2),
            "endpoints must contain CMT=2 for Weibull model"
        );
        match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte { hazard: _ }) => {}
            other => panic!("expected Tte endpoint for CMT=2 (Weibull), got: {other:?}"),
        }
        assert_eq!(
            model.n_theta, 4,
            "n_theta should be 4 (TVSCALE, TVSHAPE, CL, V)"
        );
    }

    /// Parser must recognise [event_model] with family=gompertz (alpha + gamma).
    #[test]
    fn tte_gompertz_model_parses() {
        let src = r"
[parameters]
  theta TVALPHA(0.05, 0.001, 10.0)
  theta TVGAMMA(0.05, 0.001, 5.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  ALPHA = TVALPHA
  GAMMA = TVGAMMA
  CL    = DUMMY_CL
  V     = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA
";
        let model = parse_model_string(src).expect("Gompertz TTE model must parse");
        assert!(
            model.endpoints.contains_key(&2),
            "endpoints must contain CMT=2 for Gompertz model"
        );
        match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte { hazard: _ }) => {}
            other => panic!("expected Tte endpoint for CMT=2 (Gompertz), got: {other:?}"),
        }
        assert_eq!(
            model.n_theta, 4,
            "n_theta should be 4 (TVALPHA, TVGAMMA, CL, V)"
        );
    }

    /// Fixed-effects (no omega) model with CMT 2 TTE endpoint must parse.
    #[test]
    fn tte_fixed_effects_model_parses() {
        let model = parse_model_string(EXP_TTE_FIXED).expect("EXP_TTE_FIXED must parse");
        assert!(model.endpoints.contains_key(&2));
        // n_eta = 0 (no omega declarations)
        assert_eq!(model.n_eta, 0, "n_eta should be 0 for fixed-effects model");
    }

    /// `fit()` with 3 outer iterations on TTE data must return Ok.
    ///
    /// The result must carry finite OFV; we do NOT assert convergence here.
    #[test]
    fn tte_fit_exponential_3iter() {
        let model = parse_model_string(EXP_TTE_MODEL).expect("model must parse");
        let pop = common::tte_pop_from_pairs(TTE_DATA);

        let mut opts = FitOptions::default();
        opts.verbose = false;

        let result = fit(&model, &pop, &model.default_params, &opts);
        match result {
            Ok(r) => {
                assert!(
                    r.ofv.is_finite(),
                    "OFV must be finite after 3 iterations; got {}",
                    r.ofv
                );
            }
            Err(e) => panic!("fit() must not error within 3 iterations: {e}"),
        }
    }

    /// `fit()` on a fixed-effects TTE model (n_eta=0, no inner loop) must
    /// return Ok immediately (single outer-loop evaluation per iteration).
    #[test]
    fn tte_fit_fixed_effects_n_eta_0() {
        let model = parse_model_string(EXP_TTE_FIXED).expect("model must parse");
        let pop = common::tte_pop_from_pairs(TTE_DATA);

        let mut opts = FitOptions::default();
        opts.verbose = false;

        let result = fit(&model, &pop, &model.default_params, &opts);
        match result {
            Ok(r) => {
                assert!(r.ofv.is_finite(), "OFV must be finite; got {}", r.ofv);
            }
            Err(e) => panic!("fixed-effects TTE fit must not error: {e}"),
        }
    }

    /// A nonzero `loghr` must actually change the OFV — i.e. the parser must wire it
    /// into the param_fn so it reaches the likelihood computation.
    ///
    /// IMPORTANT: `TVLAMBDA` is **FIXed** in both models. A constant `loghr` offset is
    /// otherwise non-identifiable against a free exponential rate — the optimizer simply
    /// rescales `TVLAMBDA` by `exp(-loghr)` and both fits converge to the *same* OFV
    /// (verified: diff ≈ 2.6e-5 when `TVLAMBDA` is free). Fixing the rate makes the
    /// `exp(0.5)` hazard multiplier identifiable, so a non-wired `loghr` (the bug this
    /// test guards against) is the only way the two OFVs can coincide.
    #[test]
    fn tte_loghr_nonzero_changes_ofv() {
        // Baseline: FIXed rate, no loghr.
        let src_no_lhr = r"
[parameters]
  theta TVLAMBDA(0.05, FIX)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  omega ETA_LAMBDA ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)

[fit_options]
  method  = focei
  maxiter = 3
";
        // Model B: identical, but with a hard-coded loghr = 0.5.
        let src_with_lhr = r"
[parameters]
  theta TVLAMBDA(0.05, FIX)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  omega ETA_LAMBDA ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
  loghr  = 0.5

[fit_options]
  method  = focei
  maxiter = 3
";
        let model_no_lhr = parse_model_string(src_no_lhr).expect("baseline model must parse");
        let model_with_lhr = parse_model_string(src_with_lhr).expect("model with loghr must parse");

        let pop = common::tte_pop_from_pairs(TTE_DATA);
        let mut opts = FitOptions::default();
        opts.verbose = false;

        let r0 = fit(&model_no_lhr, &pop, &model_no_lhr.default_params, &opts)
            .expect("baseline fit must succeed");
        let r1 = fit(&model_with_lhr, &pop, &model_with_lhr.default_params, &opts)
            .expect("loghr fit must succeed");

        assert!(
            r0.ofv.is_finite() && r1.ofv.is_finite(),
            "both OFVs must be finite; got {} and {}",
            r0.ofv,
            r1.ofv
        );
        // With the rate FIXed, loghr=0.5 multiplies the hazard by exp(0.5) ≈ 1.65 for
        // every subject and the offset cannot be absorbed by the rate. The OFV gap is
        // several units; a threshold of 1.0 rules out the silent-zero bug where loghr
        // is not wired through and both models return identical OFVs.
        assert!(
            (r0.ofv - r1.ofv).abs() > 1.0,
            "loghr=0.5 must change the OFV by > 1.0 — no_loghr_OFV={} loghr_OFV={}; diff={:.6}",
            r0.ofv,
            r1.ofv,
            (r0.ofv - r1.ofv).abs()
        );
    }

    /// `family=exponential` with a `shape` key must be rejected at parse time.
    #[test]
    fn tte_incompatible_key_exponential_shape_errors() {
        let src = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
  shape  = 2.0
";
        let err = parse_model_string(src)
            .err()
            .expect("shape with exponential must be rejected");
        assert!(
            err.contains("shape") || err.contains("exponential"),
            "error must mention the incompatible key: {err}"
        );
    }

    /// `family=gompertz` with a `scale` key must be rejected at parse time.
    #[test]
    fn tte_incompatible_key_gompertz_scale_errors() {
        let src = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta TVGAMMA(0.005, 0.0001, 1.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  ALPHA = TVLAMBDA
  GAMMA = TVGAMMA
  CL    = DUMMY_CL
  V     = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = gompertz
  scale  = TVLAMBDA
  gamma  = GAMMA
";
        let err = parse_model_string(src)
            .err()
            .expect("scale with gompertz must be rejected");
        assert!(
            err.contains("scale") || err.contains("gompertz"),
            "error must mention the incompatible key: {err}"
        );
    }

    /// Duplicate CMT in two [event_model] blocks must be rejected at parse time.
    #[test]
    fn tte_duplicate_cmt_parse_error() {
        let src = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  theta DUMMY_CL(1.0, FIX)
  theta DUMMY_V(1.0, FIX)
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  LAMBDA = TVLAMBDA
  CL     = DUMMY_CL
  V      = DUMMY_V

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model CMT2_A]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[event_model CMT2_B]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA
";
        let err = parse_model_string(src)
            .err()
            .expect("duplicate CMT must be rejected");
        assert!(
            err.contains("CMT=2") || err.contains("more than once"),
            "error must mention duplicate CMT: {err}"
        );
    }

    // ── Phase 1 follow-up: TTE-only model syntax (no dummy PK blocks) ─────────

    /// Minimal TTE-only model: no [structural_model], [error_model], or
    /// [individual_parameters] — all three blocks are now optional when an
    /// [event_model] block is present.
    const EXP_TTE_ONLY: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)

[fit_options]
  method  = focei
  maxiter = 3
";

    /// TTE-only with a covariate term — tests that covariate names from
    /// [event_model] expressions are injected into model.referenced_covariates.
    const EXP_TTE_WITH_COVARIATE: &str = r"
[parameters]
  theta TVLAMBDA(0.05, FIX)
  theta BETA_WT(0.1, -5.0, 5.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
  loghr  = BETA_WT * WT

[fit_options]
  method  = focei
  maxiter = 1
";

    #[test]
    fn tte_only_model_parses_without_pk_blocks() {
        let model =
            parse_model_string(EXP_TTE_ONLY).expect("TTE-only model without PK blocks must parse");
        // Should still have the TTE endpoint registered.
        assert!(
            model.endpoints.contains_key(&2),
            "endpoints must contain CMT=2 for TTE-only model"
        );
        assert_eq!(model.n_theta, 1, "n_theta should be 1 (TVLAMBDA only)");
        assert_eq!(model.n_eta, 1, "n_eta should be 1 (ETA_LAMBDA)");
    }

    #[test]
    fn tte_only_fit_completes_without_pk_blocks() {
        let model = parse_model_string(EXP_TTE_ONLY).expect("must parse");
        let pop = common::tte_pop_from_pairs(TTE_DATA);
        let mut opts = ferx_core::FitOptions::default();
        opts.verbose = false;
        let result = ferx_core::fit(&model, &pop, &model.default_params, &opts);
        match result {
            Ok(r) => assert!(r.ofv.is_finite(), "OFV must be finite; got {}", r.ofv),
            Err(e) => panic!("TTE-only fit must not error: {e}"),
        }
    }

    #[test]
    fn event_model_covariate_names_tracked() {
        let model = parse_model_string(EXP_TTE_WITH_COVARIATE)
            .expect("model with covariate loghr must parse");
        assert!(
            model.referenced_covariates.contains(&"WT".to_string()),
            "referenced_covariates must include WT from [event_model] loghr expression; \
             got: {:?}",
            model.referenced_covariates
        );
    }

    /// `[event_model]` expressions may reference names defined in
    /// `[individual_parameters]`; the hazard `param_fn` resolves them per subject at
    /// eval time. Regression: before this was wired, such references silently
    /// evaluated to 0.0. Here `scale = SCALE_I`, where `SCALE_I = LAMBDA0 * TVEFF`
    /// and `LAMBDA0 = TVBASE * exp(ETA_BASE)` — a two-level individual reference that
    /// also threads an η through to the hazard.
    #[test]
    fn event_model_references_individual_parameters() {
        // `[individual_parameters]` present ⇒ structural/error blocks are required
        // (the realistic joint PK + TTE shape). The hazard references SCALE_I, which is
        // not a PK parameter — it exists only to drive the hazard.
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVBASE(0.05, 0.001, 10.0)
  theta TVEFF(2.0, 0.1, 10.0)
  omega ETA_BASE ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  CL      = TVCL
  V       = TVV
  LAMBDA0 = TVBASE * exp(ETA_BASE)
  SCALE_I = LAMBDA0 * TVEFF

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = SCALE_I
";
        let model =
            parse_model_string(src).expect("model referencing individual params must parse");
        let ep = model
            .endpoints
            .get(&2)
            .expect("CMT=2 must be a TTE endpoint");
        let EndpointLikelihood::Tte { hazard } = ep else {
            panic!("expected Tte endpoint");
        };
        let param_fn = match hazard {
            ferx_core::HazardSpec::Analytic { param_fn, .. } => param_fn,
        };

        let covariates = std::collections::HashMap::new();
        // theta = [TVCL=1, TVV=10, TVBASE=0.05, TVEFF=2.0]; eta = [0.0]
        //   LAMBDA0 = 0.05·e^0 = 0.05 ; SCALE_I = 0.05·2.0 = 0.10  (lambda).
        let theta = [1.0, 10.0, 0.05, 2.0];
        let p0 = param_fn(&theta, &[0.0], &covariates);
        assert!(
            (p0[0] - 0.10).abs() < 1e-9,
            "hazard lambda must resolve the individual parameter to 0.10; got {} \
             (0.0 would mean the [individual_parameters] reference was not threaded)",
            p0[0]
        );
        // eta = [0.5] → LAMBDA0 = 0.05·e^0.5 ; SCALE_I = that · 2.0 — η flows through.
        let expected = 0.05 * 0.5_f64.exp() * 2.0;
        let p1 = param_fn(&theta, &[0.5], &covariates);
        assert!(
            (p1[0] - expected).abs() < 1e-9,
            "hazard lambda must track eta via the individual parameter; got {}, expected {expected}",
            p1[0]
        );
    }

    /// Issue #442 (review #1): a hazard that references an `[individual_parameters]`
    /// value whose definition uses an IOV **kappa** must be rejected at parse time,
    /// not crash the fit. The hazard `param_fn` is handed the BSV-only η, but a kappa
    /// compiles to an η-index *past* that slice (`Eta(n_eta + k)`), so evaluating the
    /// kept statement would index out of bounds and abort. Here `scale = CL` with
    /// `CL = TVCL * exp(ETA_CL + KAPPA_CL)`.
    #[test]
    fn event_model_referencing_kappa_indiv_param_is_rejected() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma SIGMA_ADD ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_ADD)

[event_model]
  cmt    = 2
  family = exponential
  scale  = CL
";
        let err = parse_model_string(src).expect_err(
            "a hazard referencing a kappa-bearing individual parameter must be rejected, \
             not OOB-panic",
        );
        assert!(
            err.contains("inter-occasion") && err.contains("KAPPA_CL"),
            "the error should name the offending IOV random effect; got: {err}"
        );
    }

    /// Issue #442 (review #2): a hazard may reference an `[individual_parameters]`
    /// value defined by a NONMEM-style `if (...) { ... } else { ... }` block. Before
    /// the fix, such a name was classified as a covariate and silently resolved to
    /// 0.0 (a degenerate hazard). `HAZ` is assigned on both branches; the `param_fn`
    /// must select the subject's branch and thread η through.
    #[test]
    fn event_model_references_conditional_individual_parameter() {
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta TVBASE(0.05, 0.001, 10.0)
  omega ETA_BASE ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[individual_parameters]
  CL = TVCL
  V  = TVV
  if (WT > 70) {
    HAZ = TVBASE * 2.0 * exp(ETA_BASE)
  } else {
    HAZ = TVBASE * exp(ETA_BASE)
  }

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = HAZ
";
        let model = parse_model_string(src)
            .expect("hazard referencing a conditionally-defined individual parameter must parse");
        let ep = model
            .endpoints
            .get(&2)
            .expect("CMT=2 must be a TTE endpoint");
        let EndpointLikelihood::Tte { hazard } = ep else {
            panic!("expected Tte endpoint");
        };
        let param_fn = match hazard {
            ferx_core::HazardSpec::Analytic { param_fn, .. } => param_fn,
        };
        let theta = [1.0, 10.0, 0.05]; // TVCL, TVV, TVBASE

        // WT = 80 (> 70) takes the *2.0 branch: HAZ = 0.05·2·e^0 = 0.10.
        let mut hi = std::collections::HashMap::new();
        hi.insert("WT".to_string(), 80.0);
        let p_hi = param_fn(&theta, &[0.0], &hi);
        assert!(
            (p_hi[0] - 0.10).abs() < 1e-9,
            "WT>70 branch must resolve HAZ to 0.10 (0.0 = unresolved conditional param); got {}",
            p_hi[0]
        );

        // WT = 60 takes the else branch: HAZ = 0.05·e^0.5, so η also flows through.
        let mut lo = std::collections::HashMap::new();
        lo.insert("WT".to_string(), 60.0);
        let p_lo = param_fn(&theta, &[0.5], &lo);
        let expected = 0.05 * 0.5_f64.exp();
        assert!(
            (p_lo[0] - expected).abs() < 1e-9,
            "else branch must resolve HAZ to {expected} and track η; got {}",
            p_lo[0]
        );
    }

    /// Issue #442 (review #3): a hazard that references an `[individual_parameters]`
    /// value driven by a `[covariate_nn]` output must be rejected — the hazard
    /// `param_fn` runs without the network forward pass, so the reference would
    /// silently resolve to 0.0. Gated on `nn` (NnOutput nodes only exist there).
    #[cfg(feature = "nn")]
    #[test]
    fn event_model_referencing_nn_driven_indiv_param_is_rejected() {
        let src = r"
[parameters]
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.09
  sigma SIGMA_DV ~ 0.01 FIX

[covariate_nn TYPICAL_PK]
  inputs = [WT]
  outputs = [CL]
  layers = [3]
  activation = tanh
  output = softplus

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_DV)

[event_model]
  cmt    = 2
  family = exponential
  scale  = CL
";
        let err = parse_model_string(src)
            .expect_err("a hazard referencing an NN-driven individual parameter must be rejected");
        assert!(
            err.contains("covariate_nn") || err.contains("network"),
            "the error should explain the NN-output limitation; got: {err}"
        );
    }

    // ── Phase 1 follow-up: median/mean survival in predict_survival ───────────

    #[test]
    fn predict_survival_has_median_and_mean() {
        use ferx_core::predict_survival;

        let model = parse_model_string(EXP_TTE_MODEL).expect("must parse");
        let pop = common::tte_pop_from_pairs(&TTE_DATA[..3]);
        let grid = vec![1.0, 5.0, 10.0, 20.0];
        let rows = predict_survival(&model, &pop, &model.default_params, &grid);
        assert!(
            !rows.is_empty(),
            "predict_survival must return rows for TTE model"
        );
        for row in &rows {
            assert!(
                row.median_survival.is_finite() && row.median_survival > 0.0,
                "median_survival must be finite and positive; got {}",
                row.median_survival
            );
            assert!(
                row.mean_survival.is_finite() && row.mean_survival > 0.0,
                "mean_survival must be finite and positive; got {}",
                row.mean_survival
            );
            // For Exponential: mean = 1/lambda, median = ln(2)/lambda; mean > median.
            assert!(
                row.mean_survival > row.median_survival,
                "Exponential: mean_survival {} must exceed median_survival {}",
                row.mean_survival,
                row.median_survival
            );
            // median_survival and mean_survival are constant across the time grid
            // for the same subject (they are distributional properties, not time-varying).
        }
        // All rows for the same subject should have identical median/mean.
        let first_median = rows[0].median_survival;
        let first_mean = rows[0].mean_survival;
        for row in rows.iter().filter(|r| r.id == rows[0].id) {
            assert_eq!(
                row.median_survival, first_median,
                "median should be constant per subject"
            );
            assert_eq!(
                row.mean_survival, first_mean,
                "mean should be constant per subject"
            );
        }
    }

    // ── Phase 1 follow-up: example file parse tests ───────────────────────────

    /// `examples/tte_weibull.ferx` must parse and expose a CMT-2 Weibull endpoint.
    /// Guards against syntax drift in the example file — CI catches it here.
    #[test]
    fn tte_weibull_example_file_parses() {
        let src = include_str!("../examples/tte_weibull.ferx");
        let model = parse_model_string(src).expect("tte_weibull.ferx must parse");
        assert!(
            model.endpoints.contains_key(&2),
            "CMT=2 must be registered as a TTE endpoint"
        );
        match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte { hazard: _ }) => {}
            other => panic!("expected Tte endpoint for CMT=2 (Weibull), got: {other:?}"),
        }
        assert_eq!(model.n_theta, 2, "n_theta should be 2 (TVSCALE, TVSHAPE)");
        assert_eq!(model.n_eta, 1, "n_eta should be 1 (ETA_SCALE)");
    }

    /// `examples/tte_gompertz.ferx` must parse and expose a CMT-2 Gompertz endpoint.
    #[test]
    fn tte_gompertz_example_file_parses() {
        let src = include_str!("../examples/tte_gompertz.ferx");
        let model = parse_model_string(src).expect("tte_gompertz.ferx must parse");
        assert!(
            model.endpoints.contains_key(&2),
            "CMT=2 must be registered as a TTE endpoint"
        );
        match model.endpoints.get(&2) {
            Some(EndpointLikelihood::Tte { hazard: _ }) => {}
            other => panic!("expected Tte endpoint for CMT=2 (Gompertz), got: {other:?}"),
        }
        assert_eq!(model.n_theta, 2, "n_theta should be 2 (TVALPHA, TVGAMMA)");
        assert_eq!(model.n_eta, 1, "n_eta should be 1 (ETA_GAMMA)");
    }

    // ── Phase 1 follow-up: Weibull / Gompertz fit smoke tests ─────────────────

    /// Simulated Weibull TTE data (30 subjects, seed=42).
    /// TVSCALE=20 h, TVSHAPE=1.5, omega(ETA_SCALE)=0.04, censor=60 h.
    /// Mirrors data/tte_weibull.csv.
    const WEIBULL_DATA: &[(f64, u8)] = &[
        (23.04, 1),
        (25.31, 1),
        (4.59, 1),
        (26.89, 1),
        (25.32, 1),
        (15.87, 1),
        (13.01, 1),
        (14.66, 1),
        (7.46, 1),
        (60.0, 0),
        (23.39, 1),
        (22.63, 1),
        (42.43, 1),
        (33.56, 1),
        (8.37, 1),
        (7.41, 1),
        (11.62, 1),
        (12.52, 1),
        (6.42, 1),
        (10.51, 1),
        (25.52, 1),
        (21.77, 1),
        (39.51, 1),
        (25.29, 1),
        (17.57, 1),
        (23.34, 1),
        (10.9, 1),
        (19.99, 1),
        (34.66, 1),
        (26.03, 1),
    ];

    /// Simulated Gompertz TTE data (50 subjects, seed=42).
    /// TVALPHA=0.002 h⁻¹, TVGAMMA=0.05 h⁻¹, omega(ETA_GAMMA)=0.04, censor=80 h.
    /// Mirrors data/tte_gompertz.csv (BSV on gamma, censoring at 80 h, 42/50 events).
    const GOMPERTZ_DATA: &[(f64, u8)] = &[
        (61.16, 1),
        (48.39, 1),
        (58.89, 1),
        (53.94, 1),
        (44.24, 1),
        (51.71, 1),
        (34.54, 1),
        (80.0, 0),
        (80.0, 0),
        (44.35, 1),
        (56.79, 1),
        (56.51, 1),
        (32.43, 1),
        (80.0, 0),
        (80.0, 0),
        (57.19, 1),
        (71.02, 1),
        (19.65, 1),
        (80.0, 0),
        (60.92, 1),
        (55.66, 1),
        (37.74, 1),
        (53.19, 1),
        (17.59, 1),
        (50.21, 1),
        (51.33, 1),
        (54.48, 1),
        (29.41, 1),
        (1.19, 1),
        (74.71, 1),
        (44.94, 1),
        (54.26, 1),
        (11.05, 1),
        (41.52, 1),
        (79.74, 1),
        (55.77, 1),
        (25.96, 1),
        (80.0, 0),
        (65.97, 1),
        (80.0, 0),
        (42.91, 1),
        (57.34, 1),
        (22.3, 1),
        (80.0, 0),
        (76.81, 1),
        (36.22, 1),
        (55.52, 1),
        (29.98, 1),
        (53.71, 1),
        (65.81, 1),
    ];

    /// TTE-only Weibull model for smoke-fit tests (maxiter=3 for speed).
    const WEIBULL_TTE_ONLY: &str = r"
[parameters]
  theta TVSCALE(20.0, 0.1, 500.0)
  theta TVSHAPE(1.5,  0.1, 10.0)
  omega ETA_SCALE ~ 0.04

[event_model]
  cmt    = 2
  family = weibull
  scale  = TVSCALE * exp(ETA_SCALE)
  shape  = TVSHAPE

[fit_options]
  method  = focei
  maxiter = 3
";

    /// TTE-only Gompertz model for smoke-fit tests (maxiter=3 for speed).
    const GOMPERTZ_TTE_ONLY: &str = r"
[parameters]
  theta TVALPHA(0.002, 1e-5, 1.0)
  theta TVGAMMA(0.05,  1e-4, 5.0)
  omega ETA_GAMMA ~ 0.04

[event_model]
  cmt    = 2
  family = gompertz
  alpha  = TVALPHA
  gamma  = TVGAMMA * exp(ETA_GAMMA)

[fit_options]
  method  = focei
  maxiter = 3
";

    /// SAEM model for the M-step TTE smoke test.  Uses the compact TTE-only syntax
    /// and SAEM with minimal iterations — verifies that the SAEM M-step includes the
    /// TTE data term (obs_nll_subject_into fix, item 2 of Phase 1 follow-up).
    const EXP_TTE_SAEM: &str = r"
[parameters]
  theta TVLAMBDA(0.05, 0.001, 10.0)
  omega ETA_LAMBDA ~ 0.09

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)

[fit_options]
  method        = saem
  n_exploration = 2
  n_convergence = 2
  maxiter       = 3
";

    /// Weibull TTE fit must return a finite OFV after 3 outer iterations.
    #[test]
    fn tte_weibull_fit_completes() {
        let model = parse_model_string(WEIBULL_TTE_ONLY).expect("WEIBULL_TTE_ONLY must parse");
        let pop = common::tte_pop_from_pairs(WEIBULL_DATA);
        let mut opts = FitOptions::default();
        opts.verbose = false;
        match fit(&model, &pop, &model.default_params, &opts) {
            Ok(r) => assert!(
                r.ofv.is_finite(),
                "Weibull OFV must be finite; got {}",
                r.ofv
            ),
            Err(e) => panic!("Weibull TTE fit must not error: {e}"),
        }
    }

    /// Gompertz TTE fit must return a finite OFV after 3 outer iterations.
    #[test]
    fn tte_gompertz_fit_completes() {
        let model = parse_model_string(GOMPERTZ_TTE_ONLY).expect("GOMPERTZ_TTE_ONLY must parse");
        let pop = common::tte_pop_from_pairs(GOMPERTZ_DATA);
        let mut opts = FitOptions::default();
        opts.verbose = false;
        match fit(&model, &pop, &model.default_params, &opts) {
            Ok(r) => assert!(
                r.ofv.is_finite(),
                "Gompertz OFV must be finite; got {}",
                r.ofv
            ),
            Err(e) => panic!("Gompertz TTE fit must not error: {e}"),
        }
    }

    /// SAEM on a TTE-only exponential model must return a finite OFV.
    /// Specifically exercises the obs_nll_subject_into TTE data term (SAEM M-step fix).
    #[test]
    fn tte_saem_fit_completes() {
        let model = parse_model_string(EXP_TTE_SAEM).expect("EXP_TTE_SAEM must parse");
        let pop = common::tte_pop_from_pairs(TTE_DATA);
        let mut opts = FitOptions::default();
        opts.verbose = false;
        match fit(&model, &pop, &model.default_params, &opts) {
            Ok(r) => assert!(
                r.ofv.is_finite(),
                "SAEM TTE OFV must be finite; got {}",
                r.ofv
            ),
            Err(e) => panic!("SAEM TTE fit must not error: {e}"),
        }
    }

    // ── Phase 1 follow-up: IOV + TTE subjects ────────────────────────────────

    /// Mixed IOV+TTE model: one-cpt IV PK with a per-occasion kappa on CL,
    /// plus an exponential TTE endpoint on CMT=2.  `maxiter=3` keeps it Tier-2.
    const IOV_TTE_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVLAMBDA(0.05, 0.001, 5.0)

  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04

  sigma SIGMA_ADD ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(SIGMA_ADD)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_CL)

[fit_options]
  method  = focei
  maxiter = 3
";

    /// Build a population of `n` subjects each having:
    ///   - 2 IV doses (occasions 0 and 1)
    ///   - 1 PK observation per occasion (CMT=1)
    ///   - 1 TTE event (CMT=2)
    ///
    /// This exercises the code path in `foce_subject_nll_iov` that was
    /// previously bypassing the TTE Laplace correction when kappas are
    /// non-empty (fix in commit 9d954f1).
    fn iov_tte_population(n: usize, event_times: &[f64]) -> Population {
        // For TVCL=1.0, TVV=10.0, dose=100 at t=0:
        //   conc(t=4) = 100/10 * exp(-0.1*4) ≈ 6.7
        let pk_conc = 6.7_f64;

        let subjects = (0..n)
            .map(|i| {
                // Dose 100 at t=0 (occ 0) and dose 100 at t=24 (occ 1).
                // One PK obs per occasion at t=4 and t=28.
                let mut s = common::subject(
                    &format!("{}", i + 1),
                    vec![
                        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                        DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
                    ],
                    vec![4.0, 28.0],
                    vec![pk_conc, pk_conc],
                    vec![1, 1],
                );
                s.obs_raw_times = vec![4.0, 28.0];
                s.occasions = vec![0, 1];
                s.dose_occasions = vec![0, 1];
                s.obs_records = vec![ObsRecord::Event {
                    time: event_times[i % event_times.len()],
                    event_type: EventType::Exact,
                    entry_time: 0.0,
                    cmt: 2,
                }];
                s
            })
            .collect();

        Population {
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects,
        }
    }

    /// IOV subjects with TTE obs_records must produce a finite FOCEI OFV.
    ///
    /// This is the Tier-2 regression guard for `foce_subject_nll_iov`:
    /// when kappas are non-empty AND the subject carries TTE obs_records,
    /// the function must route through `foce_subject_nll_interaction_with_tte`
    /// rather than the plain interaction/standard paths that ignore TTE.
    #[test]
    fn iov_tte_focei_returns_finite_ofv() {
        let model = parse_model_string(IOV_TTE_MODEL).expect("IOV+TTE model must parse");
        let event_times = [16.0_f64, 10.0, 22.0, 8.0, 30.0, 18.0];
        let pop = iov_tte_population(6, &event_times);
        let mut opts = FitOptions::default();
        opts.verbose = false;
        match fit(&model, &pop, &model.default_params, &opts) {
            Ok(r) => assert!(
                r.ofv.is_finite(),
                "IOV+TTE FOCEI OFV must be finite; got {}",
                r.ofv
            ),
            Err(e) => panic!("IOV+TTE FOCEI fit must not error: {e}"),
        }
    }
}
