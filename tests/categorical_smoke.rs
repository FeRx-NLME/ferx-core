//! Tier-2 smoke tests for the Phase 4 Binary / logistic endpoint (Track C, #760).
//!
//! These exercise the public parse / `fit()` boundary. They are NOT gated with
//! `slow-tests` — every fit runs `outer_maxiter = 3` and returns immediately.
//! Full-convergence anchors (vs base-R `glm`) live in
//! `tests/categorical_convergence.rs` (Tier 3).
//!
//! All binary-specific items are behind `#[cfg(feature = "survival")]` so the file
//! compiles on every PR without the feature enabled (it contributes no tests then).

mod common;

#[cfg(feature = "survival")]
mod binary_smoke {
    use crate::common;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::{fit, fit_from_files, EndpointLikelihood, FitOptions};

    /// Mixed-effects logistic: a per-subject random intercept `ETA_I` plus a covariate.
    const MIXED_MODEL: &str = r"
[parameters]
  theta TH0(0.0, -10.0, 10.0)
  theta THX(0.0, -10.0, 10.0)
  omega ETA_I ~ 0.25

[binary_model]
  cmt   = 3
  logit = TH0 + THX * X + ETA_I
";

    /// Fixed-effects (n_eta = 0) logistic — ordinary logistic regression.
    const FIXED_MODEL: &str = r"
[parameters]
  theta TH0(0.0, -10.0, 10.0)
  theta THX(0.0, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0 + THX * X
";

    fn smoke_opts() -> FitOptions {
        FitOptions {
            outer_maxiter: 3,
            ..Default::default()
        }
    }

    /// The parser recognises `[binary_model]` and registers a `Binary` endpoint on its CMT.
    #[test]
    fn binary_model_parses() {
        let model = parse_model_string(MIXED_MODEL).expect("MIXED_MODEL must parse");
        assert!(
            model.endpoints.contains_key(&3),
            "endpoints must contain CMT=3; got {:?}",
            model.endpoints.keys().collect::<Vec<_>>()
        );
        match model.endpoints.get(&3) {
            Some(EndpointLikelihood::Binary { .. }) => {}
            other => panic!("expected a Binary endpoint for CMT=3, got {other:?}"),
        }
        assert_eq!(model.n_theta, 2, "n_theta should be TH0 + THX");
        assert_eq!(model.n_eta, 1, "n_eta should be ETA_I");
    }

    /// A mixed-effects fit runs a few iterations and returns a finite OFV (FOCEI FD-Laplace).
    #[test]
    fn binary_mixed_effects_runs() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        // 6 subjects × 3 binary observations each; one covariate value per subject.
        let subjects: Vec<(f64, Vec<(f64, u8)>)> = vec![
            (1.0, vec![(0.0, 1), (1.0, 1), (2.0, 0)]),
            (-0.5, vec![(0.0, 0), (1.0, 0), (2.0, 1)]),
            (0.3, vec![(0.0, 1), (1.0, 0), (2.0, 0)]),
            (-1.2, vec![(0.0, 0), (1.0, 0), (2.0, 0)]),
            (0.8, vec![(0.0, 1), (1.0, 1), (2.0, 1)]),
            (-0.1, vec![(0.0, 0), (1.0, 1), (2.0, 0)]),
        ];
        let pop = common::binary_pop(&subjects, 3);
        let res = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect("mixed-effects binary fit must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// Fixed-effects (n_eta = 0) logistic — the empty-Omega path — read from the CSV through
    /// the real datareader (so the `DiscreteState` routing is exercised) and fit end-to-end.
    #[test]
    fn binary_fixed_effects_reads_and_fits() {
        let res = fit_from_files(
            "examples/binary_logistic.ferx",
            Some("data/binary_logistic.csv"),
            Some(&["X"]),
            Some(smoke_opts()),
        )
        .expect("fixed-effects binary fit_from_files must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// A non-Bernoulli code (DV ≥ 2) on a binary CMT is rejected fail-loud at fit setup,
    /// never silently folded into the Bernoulli term.
    #[test]
    fn binary_rejects_non_bernoulli_state() {
        let model = parse_model_string(FIXED_MODEL).unwrap();
        let pop = common::binary_pop(&[(0.5, vec![(0.0, 0), (1.0, 2)])], 3);
        let err = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect_err("state = 2 on a binary CMT must be rejected");
        assert!(
            err.contains("0 or 1"),
            "expected a fail-loud message, got: {err}"
        );
        assert!(
            err.contains("cmt = 3"),
            "message should name the CMT, got: {err}"
        );
    }
}
