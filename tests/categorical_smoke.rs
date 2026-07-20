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
        // Non-integer cmt.
        assert!(parse_model_string(&wrap("[binary_model]\n  cmt = xx\n  logit = TH0")).is_err());
        // Malformed line (no `=`).
        assert!(
            parse_model_string(&wrap("[binary_model]\n  cmt = 3\n  logit = TH0\n  oops")).is_err()
        );
    }

    /// A binary `logit` that references an `[individual_parameters]` value which itself reads a
    /// covariate — exercises the parser's transitive needed-indiv-statements restriction and the
    /// covariate-union-through-`[individual_parameters]` path (the #741 completeness guard).
    #[test]
    fn binary_logit_references_individual_parameter() {
        const M: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBETA(0.1, -5.0, 5.0)
  theta TH0(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.05

[covariates]
  AGE continuous

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  LP = TH0 + TVBETA * AGE

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[binary_model]
  cmt   = 3
  logit = LP

[fit_options]
  method = focei
";
        let model = parse_model_string(M).expect("binary logit referencing an indiv param parses");
        // AGE is used only by LP (the value the logit references), so it can only appear in the
        // model's covariate set via the binary covariate-union-through-[individual_parameters].
        assert!(
            model.referenced_covariates.iter().any(|c| c == "AGE"),
            "AGE (reached via the indiv-param the logit references) must be collected: {:?}",
            model.referenced_covariates
        );
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

    /// A κ reached *through* an `[individual_parameters]` value is also rejected in the logit
    /// (the predictor is BSV-only) — the transitive analogue of the direct check.
    #[test]
    fn binary_logit_indirect_kappa_rejected() {
        let m = IOV_MODEL
            .replace("  V  = TVV\n", "  V  = TVV\n  LP = TH0 + KAPPA_CL\n")
            .replace("logit = TH0", "logit = LP");
        assert!(
            parse_model_string(&m).is_err(),
            "a κ reached via an [individual_parameters] value must be rejected in the logit"
        );
    }

    // ---- Slice 1b: simulate + predict -------------------------------------------------

    /// Six subjects × three binary records, used by the simulate / predict smokes.
    fn sim_subjects() -> Vec<(f64, Vec<(f64, u8)>)> {
        vec![
            (1.0, vec![(0.0, 1), (1.0, 1), (2.0, 0)]),
            (-0.5, vec![(0.0, 0), (1.0, 0), (2.0, 1)]),
            (0.3, vec![(0.0, 1), (1.0, 0), (2.0, 0)]),
            (-1.2, vec![(0.0, 0), (1.0, 0), (2.0, 0)]),
            (0.8, vec![(0.0, 1), (1.0, 1), (2.0, 1)]),
            (-0.1, vec![(0.0, 0), (1.0, 1), (2.0, 0)]),
        ]
    }

    /// **The regression test for the Slice-1a gap.** A binary endpoint used to contribute
    /// **zero** rows to `simulate()` — silently, with no error — because its records live in
    /// `obs_records` and the Gaussian emitter walks `obs_times`. Assert one `Category` row
    /// per binary record, on the endpoint's CMT, carrying a valid state and `ipred = p`.
    #[test]
    fn binary_simulate_emits_one_category_row_per_record() {
        use ferx_core::types::SimOutcome;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let subjects = sim_subjects();
        let n_records: usize = subjects.iter().map(|(_, obs)| obs.len()).sum();
        let pop = common::binary_pop(&subjects, 3);

        let rows = ferx_core::simulate(&model, &pop, &model.default_params, 1);
        assert_eq!(
            rows.len(),
            n_records,
            "expected one simulated row per binary record, got {}",
            rows.len()
        );
        for r in &rows {
            assert_eq!(r.cmt, 3, "row must carry the binary endpoint's CMT");
            match r.outcome {
                SimOutcome::Category { state } => {
                    assert!(state <= 1, "binary draw must be 0/1, got {state}")
                }
                ref other => panic!("expected a Category outcome, got {other:?}"),
            }
            assert!(
                (0.0..=1.0).contains(&r.ipred),
                "ipred must be the probability p, got {}",
                r.ipred
            );
        }
    }

    /// The draw is seed-reproducible, and a different seed gives a different stream —
    /// i.e. the outcomes are actually random, not a constant the first assert would miss.
    #[test]
    fn binary_simulate_is_seed_reproducible() {
        use ferx_core::types::SimOutcome;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);
        let states = |seed: u64| -> Vec<usize> {
            ferx_core::simulate_with_seed(&model, &pop, &model.default_params, 1, seed)
                .iter()
                .map(|r| match r.outcome {
                    SimOutcome::Category { state } => state,
                    ref o => panic!("expected Category, got {o:?}"),
                })
                .collect()
        };
        assert_eq!(states(42), states(42), "same seed must reproduce the draw");
        // With 18 records the chance two seeds agree everywhere is ~2^-18 under the
        // model, so a mismatch here means the outcomes are genuinely varying.
        assert_ne!(
            states(42),
            states(20260720),
            "different seeds must give a different stream (outcomes are not constant)"
        );
    }

    /// `predict_categorical` returns a probability vector per binary record: normalized,
    /// in `[0,1]`, indexed so `probs[k] == P(Y = k)`, and paired with the observed DV.
    #[test]
    fn binary_predict_categorical_returns_normalized_probabilities() {
        use ferx_core::types::Prediction;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let subjects = sim_subjects();
        let n_records: usize = subjects.iter().map(|(_, obs)| obs.len()).sum();
        let pop = common::binary_pop(&subjects, 3);

        let preds = ferx_core::predict_categorical(&model, &pop, &model.default_params);
        assert_eq!(preds.len(), n_records, "one prediction per binary record");
        for p in &preds {
            assert_eq!(p.cmt, 3);
            let Prediction::CatProbs { ref probs } = p.prediction else {
                panic!("expected CatProbs, got {:?}", p.prediction);
            };
            assert_eq!(probs.len(), 2, "binary prediction has two categories");
            assert!(
                (probs.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                "probabilities must sum to 1, got {probs:?}"
            );
            assert!(probs.iter().all(|q| (0.0..=1.0).contains(q)));
            // The observed DV is carried through so a caller can form a residual
            // without re-joining the input population.
            let y = p.observed.expect("binary records carry an observed DV");
            assert!(y == 0.0 || y == 1.0, "observed DV must be 0/1, got {y}");
            // `prob(k)` indexes by DV code.
            assert_eq!(p.prediction.prob(1), Some(probs[1]));
        }
        // At the default θ = 0 with η = 0 the predictor is 0 ⇒ p = 0.5 exactly.
        let Prediction::CatProbs { ref probs } = preds[0].prediction else {
            unreachable!()
        };
        assert!(
            (probs[1] - 0.5).abs() < 1e-12,
            "θ=0 ⇒ p=0.5, got {}",
            probs[1]
        );
    }

    /// The plain Gaussian `predict()` returns no rows for a binary-only model (its records
    /// are not on the continuous grid) — documented behaviour, matching TTE, with
    /// `predict_categorical` as the endpoint's own entry point.
    #[test]
    fn gaussian_predict_is_empty_for_a_binary_only_model() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);
        assert!(ferx_core::predict(&model, &pop, &model.default_params).is_empty());
    }
}
