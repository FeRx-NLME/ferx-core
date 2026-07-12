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
    use ferx_core::api::check_model_options;
    use ferx_core::diagnostics::Diagnostic;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::EstimationMethod;
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

    /// Joint PK + binary model declaring inter-occasion variability (κ) — used to check that
    /// binary + IOV is rejected (the outer FOCE-IOV objective omits the binary term).
    const IOV_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TH0(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.05

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[binary_model]
  cmt   = 3
  logit = TH0

[fit_options]
  method     = focei
  iov_column = OCC
";

    fn has_error(diags: &[Diagnostic], code: &str) -> bool {
        diags.iter().any(|d| d.code == code)
    }

    /// SAEM — the preferred estimator for categorical data — runs a short binary fit and
    /// returns a finite OFV. Exercises the SAEM M-step site (`obs_nll_subject_from_preds`),
    /// which the FOCEI smoke tests above do not reach.
    #[test]
    fn binary_saem_runs() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let subjects: Vec<(f64, Vec<(f64, u8)>)> = vec![
            (1.0, vec![(0.0, 1), (1.0, 1), (2.0, 0)]),
            (-0.5, vec![(0.0, 0), (1.0, 0), (2.0, 1)]),
            (0.3, vec![(0.0, 1), (1.0, 0), (2.0, 0)]),
            (-1.2, vec![(0.0, 0), (1.0, 0), (2.0, 0)]),
            (0.8, vec![(0.0, 1), (1.0, 1), (2.0, 1)]),
            (-0.1, vec![(0.0, 0), (1.0, 1), (2.0, 0)]),
        ];
        let pop = common::binary_pop(&subjects, 3);
        let opts = FitOptions {
            method: EstimationMethod::Saem,
            saem_n_exploration: 2,
            saem_n_convergence: 2,
            ..Default::default()
        };
        let res = fit(&model, &pop, &model.default_params, &opts)
            .expect("SAEM binary fit must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// The Gauss-Newton gradient is Gaussian-specific (plan §13), so `method = gn` on a binary
    /// model is rejected fail-loud — the review found GN silently drops the logit term.
    #[test]
    fn binary_gauss_newton_rejected() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let opts = FitOptions {
            method: EstimationMethod::FoceGn,
            ..Default::default()
        };
        assert!(
            has_error(
                &check_model_options(&model, &opts),
                "E_BINARY_GN_UNSUPPORTED"
            ),
            "method = gn + binary must be rejected"
        );
    }

    /// Binary + inter-occasion variability is rejected: the binary term enters the inner EBE
    /// objective but not the outer FOCE-IOV objective (the inner/outer mismatch the review found).
    #[test]
    fn binary_iov_rejected() {
        let model = parse_model_string(IOV_MODEL).expect("joint PK + binary + IOV model parses");
        assert!(model.n_kappa > 0, "fixture must declare a κ");
        assert!(
            has_error(
                &check_model_options(&model, &FitOptions::default()),
                "E_BINARY_IOV_UNSUPPORTED"
            ),
            "binary + IOV must be rejected"
        );
    }

    /// The `[binary_model]` parser rejects its error cases fail-loud (unsupported link, unknown
    /// key, missing required keys) rather than mis-parsing.
    #[test]
    fn binary_parser_rejections() {
        let wrap = |block: &str| {
            format!("[parameters]\n  theta TH0(0.0, -10.0, 10.0)\n{block}\n[fit_options]\n  method = focei\n")
        };
        // Unsupported link.
        assert!(parse_model_string(&wrap(
            "[binary_model]\n  cmt = 3\n  logit = TH0\n  link = probit"
        ))
        .is_err());
        // Unknown key.
        assert!(parse_model_string(&wrap(
            "[binary_model]\n  cmt = 3\n  logit = TH0\n  bogus = 1"
        ))
        .is_err());
        // Missing `logit`.
        assert!(parse_model_string(&wrap("[binary_model]\n  cmt = 3")).is_err());
        // Missing `cmt`.
        assert!(parse_model_string(&wrap("[binary_model]\n  logit = TH0")).is_err());
    }

    /// A logit predictor that references an IOV κ directly is rejected at parse (BSV-only scope).
    #[test]
    fn binary_logit_direct_kappa_rejected() {
        let m = IOV_MODEL.replace("logit = TH0", "logit = TH0 + KAPPA_CL");
        assert!(
            parse_model_string(&m).is_err(),
            "a direct κ reference in the logit must be rejected at parse"
        );
    }
}
