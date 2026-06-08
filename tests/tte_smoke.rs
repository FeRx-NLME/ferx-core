//! Tier-2 smoke tests for Phase 1 TTE (time-to-event) support.
//!
//! These exercise the public parse / `fit()` boundary.  They are NOT gated
//! with `slow-tests` — they must finish in a handful of outer iterations.
//! Full convergence tests live in `tests/tte_convergence.rs` (Tier 3).
//!
//! All TTE-specific items are behind `#[cfg(feature = "survival")]` so the
//! file compiles on every PR without the feature enabled (it just contributes
//! no test functions).

#[cfg(feature = "survival")]
mod survival_smoke {
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::{DoseEvent, EventType, ObsRecord, Population, Subject};
    use ferx_core::{fit, EndpointLikelihood, FitOptions};
    use std::collections::HashMap;

    // ── Model strings ────────────────────────────────────────────────────────

    /// Standalone exponential TTE model.  A dummy 1-cpt structural block is
    /// required syntactically; it is never invoked (no CMT-1 observations).
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

    /// Build a TTE-only population from (time, dv) pairs.
    /// `dv`: 0 = right-censored, 1 = exact event.
    fn tte_population(data: &[(f64, u8)]) -> Population {
        let subjects = data
            .iter()
            .enumerate()
            .map(|(i, &(t, dv))| {
                let event_type = if dv == 1 {
                    EventType::Exact
                } else {
                    EventType::RightCensored
                };
                Subject {
                    id: format!("{}", i + 1),
                    doses: vec![],
                    obs_times: vec![],
                    obs_raw_times: vec![],
                    observations: vec![],
                    obs_cmts: vec![],
                    covariates: HashMap::new(),
                    dose_covariates: vec![],
                    obs_covariates: vec![],
                    pk_only_times: vec![],
                    pk_only_covariates: vec![],
                    reset_times: vec![],
                    cens: vec![],
                    occasions: vec![],
                    dose_occasions: vec![],
                    obs_records: vec![ObsRecord::Event {
                        time: t,
                        event_type,
                        entry_time: 0.0,
                        cmt: 2,
                    }],
                }
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
        let pop = tte_population(TTE_DATA);

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
        let pop = tte_population(TTE_DATA);

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
    #[test]
    fn tte_loghr_nonzero_changes_ofv() {
        // Model B: hard-coded loghr = 0.5 (fixed offset on the log-hazard scale).
        let src_with_lhr = r"
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
  loghr  = 0.5

[fit_options]
  method  = focei
  maxiter = 3
";
        let model_no_lhr = parse_model_string(EXP_TTE_MODEL).expect("EXP_TTE_MODEL must parse");
        let model_with_lhr = parse_model_string(src_with_lhr).expect("model with loghr must parse");

        let pop = tte_population(TTE_DATA);
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
        // loghr=0.5 multiplies the hazard by exp(0.5) ≈ 1.65 for all subjects.
        // Analytically, the OFV shift at the initial theta is ~6 units for this
        // 20-subject dataset.  After 3 outer iterations the models diverge further.
        // A threshold of 1.0 is conservative but rules out the silent-zero bug where
        // loghr is not wired through and both models return identical OFVs.
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

    /// DoseEvent helper — not used in TTE-only tests but checks Subject
    /// constructors compile correctly with the obs_records field.
    #[allow(dead_code)]
    fn _dummy_subject_with_dose() -> Subject {
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![],
            obs_raw_times: vec![],
            observations: vec![],
            obs_cmts: vec![],
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![],
            occasions: vec![],
            dose_occasions: vec![],
            obs_records: vec![],
        }
    }
}
