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

    /// A fit **restored from a `.fitrx` checkpoint** carries no per-record discrete
    /// diagnostics (the bundle doesn't round-trip them), so writing sdtab would emit a
    /// table silently missing every binary row. `write_sdtab_csv` refuses — keyed on the
    /// `restored_from_checkpoint` flag **and the restored model's declared binary
    /// endpoint** (`model_text`), NOT on discrete records in the reconstructed population.
    /// That distinction is the whole point: `load_fit` re-reads the data with default
    /// (Gaussian) routing, so a genuinely restored population carries *no* `DiscreteState`
    /// records at all — an earlier pop-based guard therefore never fired on a real restore
    /// and wrote a truncated table. A *live* CTMM fit (no binary endpoint) must still
    /// write; see `markov_smoke`.
    #[test]
    fn sdtab_refuses_a_restored_binary_fit_missing_its_discrete_rows() {
        use ferx_core::types::Population;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);
        let mut res = fit(&model, &pop, &model.default_params, &smoke_opts()).expect("fit");

        // A live fit writes fine and carries real discrete rows.
        assert!(!res.restored_from_checkpoint);
        assert!(res.subjects.iter().any(|s| !s.discrete_rows.is_empty()));
        let dir = tempfile::tempdir().expect("temp dir");
        let live = dir.path().join("live.csv");
        ferx_core::io::output::write_sdtab_csv(&res, &pop, live.to_str().unwrap())
            .expect("a live binary fit must write sdtab");

        // Simulate the restore faithfully: the flag flips, the diagnostics are gone, the
        // model source is carried (as `load_fit` sets it) — and, critically, the
        // reconstructed population carries NO discrete records (Gaussian routing). The
        // guard must still refuse, so it cannot be relying on the population.
        res.restored_from_checkpoint = true;
        res.model_text = Some(MIXED_MODEL.to_string());
        for sr in &mut res.subjects {
            sr.discrete_rows.clear();
        }
        let no_discrete_pop = Population {
            subjects: vec![],
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let restored = dir.path().join("restored.csv");
        let err = ferx_core::io::output::write_sdtab_csv(
            &res,
            &no_discrete_pop,
            restored.to_str().unwrap(),
        )
        .expect_err("a restored fit of a binary model must be refused even with no discrete rows in the pop");
        assert!(
            err.contains("restored") && err.contains("fitrx"),
            "message: {err}"
        );
        assert!(!restored.exists(), "no partial file may be left behind");

        // No false positive: the same restored flag on a model with **no** binary endpoint
        // (a plain Gaussian / CTMM fit, whose occupancy diagnostics are #820) must NOT be
        // refused — `binary_cmts()` is binary-only, so the guard stays silent.
        const GAUSSIAN_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  sigma PROP ~ 0.05

[individual_parameters]
  CL = TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
";
        res.model_text = Some(GAUSSIAN_MODEL.to_string());
        // Use the real (aligned) population here: the guard does not fire, so control
        // reaches `sdtab`, which indexes `population.subjects` by `result.subjects` order.
        let out = ferx_core::io::output::write_sdtab_csv(
            &res,
            &pop,
            dir.path().join("gauss.csv").to_str().unwrap(),
        );
        // It may still Err for an unrelated reason (a binary fit has no Gaussian rows to
        // write) — but it must never be the discrete-diagnostics refusal.
        if let Err(e) = &out {
            assert!(
                !e.contains("restored from"),
                "a restored non-binary fit must not hit the discrete-diagnostics refusal: {e}"
            );
        }
    }

    // ---- Slice 1b: sdtab diagnostics ---------------------------------------------------

    /// A binary fit emits one sdtab row per binary record, with `IPRED` = `p`, `DV` the
    /// observed 0/1, and `IWRES` the standardized residual. `CWRES` is blank: the
    /// conditional weighted residual is defined through the Gaussian residual-variance
    /// model, which a Bernoulli outcome has none of.
    #[test]
    fn binary_sdtab_emits_one_row_per_record() {
        use ferx_core::io::output::sdtab;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let subjects = sim_subjects();
        let n_records: usize = subjects.iter().map(|(_, obs)| obs.len()).sum();
        let pop = common::binary_pop(&subjects, 3);
        let res = fit(&model, &pop, &model.default_params, &smoke_opts()).expect("fit");

        let cols = sdtab(&res, &pop);
        let get = |name: &str| -> &Vec<f64> {
            &cols
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| {
                    panic!(
                        "sdtab must have a {name} column; got {:?}",
                        cols.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()
                    )
                })
                .1
        };

        // A binary-only model has no Gaussian rows, so every sdtab row is a discrete one.
        assert_eq!(
            get("DV").len(),
            n_records,
            "one sdtab row per binary record"
        );
        // The endpoint's CMT must survive even with no Gaussian rows to trigger the
        // usual multi-CMT heuristic.
        assert!(
            get("CMT").iter().all(|&c| c == 3.0),
            "CMT column must carry the binary CMT"
        );

        for (i, (&dv, &ipred)) in get("DV").iter().zip(get("IPRED").iter()).enumerate() {
            assert!(dv == 0.0 || dv == 1.0, "row {i}: DV must be 0/1, got {dv}");
            assert!(
                (0.0..=1.0).contains(&ipred),
                "row {i}: IPRED must be a probability, got {ipred}"
            );
        }
        assert!(
            get("IWRES").iter().all(|r| r.is_finite()),
            "IWRES must be finite"
        );
        assert!(get("PRED").iter().all(|p| (0.0..=1.0).contains(p)));
        // CWRES is undefined for a Bernoulli outcome → NaN → blank cell on write.
        assert!(
            get("CWRES").iter().all(|c| c.is_nan()),
            "CWRES must be blank for discrete rows"
        );
    }

    /// **Table-integrity invariant.** Every sdtab column is built by its own pass, and the
    /// CSV writer indexes each column by row — so a column that fails to extend for the
    /// discrete rows panics on write. Assert all columns are equal length, which is what
    /// keeps a *newly added* column from silently truncating the table later.
    #[test]
    fn binary_sdtab_columns_are_all_equal_length() {
        use ferx_core::io::output::sdtab;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);
        let res = fit(&model, &pop, &model.default_params, &smoke_opts()).expect("fit");

        let cols = sdtab(&res, &pop);
        let n = cols[0].1.len();
        for (name, values) in &cols {
            assert_eq!(
                values.len(),
                n,
                "column {name} has {} rows, expected {n} — write_sdtab_csv would panic",
                values.len()
            );
        }
        // And the writer itself must survive a round trip to disk.
        // A unique temp dir, not a fixed filename: this repo routinely has several
        // worktrees building at once, and two concurrent `cargo test` runs sharing one
        // path would delete each other's file mid-read.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("binary_sdtab.csv");
        ferx_core::io::output::write_sdtab_csv(&res, &pop, path.to_str().unwrap())
            .expect("write_sdtab_csv must succeed for a binary-only fit");
        let written = std::fs::read_to_string(&path).expect("sdtab file must exist");
        let mut lines = written.lines();
        let header = lines.next().expect("header");
        assert!(header.contains("CMT") && header.contains("IPRED"));
        assert_eq!(lines.count(), n, "one CSV data row per sdtab row");
    }

    /// **Determinism across endpoints.** `model.endpoints` is a `HashMap`, so iterating it
    /// while consuming the RNG made seeded draws depend on per-process hash ordering.
    ///
    /// An in-process test cannot reproduce cross-process hash randomization — re-running
    /// with the same seed in the same process is a tautology, since the map instance is
    /// unchanged. What *is* testable is that emission order equals the sorted CMT order
    /// the fix guarantees. With five endpoints a random hash order matches sorted with
    /// probability 1/120, so a regression is caught ~99% of the time rather than ~50%.
    #[test]
    fn binary_endpoints_are_visited_in_ascending_cmt_order() {
        use ferx_core::types::{ObsRecord, SimOutcome};
        let cmts = [3usize, 4, 5, 6, 7];
        let mut src = String::from("[parameters]\n  theta TH0(0.3, -10.0, 10.0)\n");
        for (i, c) in cmts.iter().enumerate() {
            let name = if i == 0 {
                String::new()
            } else {
                format!(" B{i}")
            };
            src.push_str(&format!(
                "\n[binary_model{name}]\n  cmt   = {c}\n  logit = TH0\n"
            ));
        }
        let model = parse_model_string(&src).expect("multiple [binary_model] blocks must parse");
        assert_eq!(
            model.binary_cmts(),
            cmts.to_vec(),
            "fixture must have all five binary endpoints"
        );

        let mut pop = common::binary_pop(&[(0.0, vec![(0.0, 0)])], cmts[0]);
        for subj in &mut pop.subjects {
            for &c in &cmts[1..] {
                subj.obs_records.push(ObsRecord::DiscreteState {
                    time: 0.0,
                    raw_time: 0.0,
                    state: 0,
                    cmt: c,
                });
            }
        }

        let emitted: Vec<usize> =
            ferx_core::simulate_with_seed(&model, &pop, &model.default_params, 1, 20260721)
                .iter()
                .map(|r| match r.outcome {
                    SimOutcome::Category { .. } => r.cmt,
                    ref o => panic!("expected Category, got {o:?}"),
                })
                .collect();
        assert_eq!(
            emitted,
            cmts.to_vec(),
            "rows must be emitted in ascending CMT order (the RNG is consumed in this order)"
        );
    }

    /// Binary subjects are sparse, so #891's guarded multi-start inner EBE flags them as
    /// weakly identified and runs its extra probes on them. The posterior is log-concave
    /// (Bernoulli likelihood × Gaussian prior), so multistart cannot relocate the mode —
    /// this pins that the merged interaction stays finite and agrees with the single-start
    /// fit, since nothing else covers binary x `inner_restarts`.
    #[test]
    fn binary_fit_is_stable_under_inner_multistart() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);

        let base = fit(&model, &pop, &model.default_params, &smoke_opts()).expect("single-start");
        let multi = fit(
            &model,
            &pop,
            &model.default_params,
            &FitOptions {
                inner_restarts: 3,
                ..smoke_opts()
            },
        )
        .expect("multi-start binary fit must return Ok");

        assert!(
            multi.ofv.is_finite(),
            "OFV must be finite, got {}",
            multi.ofv
        );
        // Log-concave posterior ⇒ every start reaches the same inner mode ⇒ same objective.
        assert!(
            (multi.ofv - base.ofv).abs() < 1e-6,
            "multistart OFV {} vs single-start {}",
            multi.ofv,
            base.ofv
        );
    }

    /// **The `--simulate` round trip.** `run_model_simulate` builds a synthetic design from
    /// `[simulation]`, draws, writes the outcomes back, and fits. This path had no test, and
    /// that gap hid a real bug: the synthetic subject materialised `observations` /
    /// `obs_cmts` on CMT 1 unconditionally, so a binary-only design (which *must* supply
    /// `times`) fabricated phantom Gaussian rows and died in `check_per_cmt_error_model`
    /// with "[error_model] has no entry for observed compartment(s) 1".
    #[test]
    fn binary_only_simulate_round_trips_through_run_model_simulate() {
        use ferx_core::types::ObsRecord;
        let src = r"
[parameters]
  theta TH0(0.2, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0

[simulation]
  n_subjects = 4
  times      = [0, 1, 2]

[fit_options]
  method  = focei
  maxiter = 2
";
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("binonly.ferx");
        std::fs::write(&path, src).expect("write model");

        let (res, pop) = ferx_core::run_model_simulate(path.to_str().unwrap())
            .expect("binary-only --simulate must round-trip");

        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
        // No phantom Gaussian rows: a binary-only design has an empty continuous grid.
        assert!(
            pop.subjects.iter().all(|s| s.observations.is_empty()),
            "binary-only design must not fabricate Gaussian observations"
        );
        // 4 subjects x 3 times of realised binary rows.
        let states: Vec<usize> = pop
            .subjects
            .iter()
            .flat_map(|s| s.obs_records.iter())
            .filter_map(|r| match r {
                ObsRecord::DiscreteState { state, cmt: 3, .. } => Some(*state),
                _ => None,
            })
            .collect();
        assert_eq!(states.len(), 12, "one binary row per (subject x time)");
        assert!(states.iter().all(|&s| s <= 1), "states must be 0/1");
        // At TH0 = 0.2 (p ~ 0.55) across 12 draws, an all-identical column means the
        // write-back stamped placeholders rather than the realised outcomes.
        assert!(
            states.iter().any(|&s| s == 1) && states.iter().any(|&s| s == 0),
            "expected a mix of realised outcomes, got {states:?}"
        );
    }

    /// Duplicated `[simulation] times` are legal (the parser does not dedup). Each duplicate
    /// record must get its **own** draw — an identity-keyed write-back that stored one state
    /// per (cmt, time) would collapse them and force the two rows to agree.
    #[test]
    fn duplicate_simulation_times_each_get_their_own_draw() {
        use ferx_core::types::ObsRecord;
        let src = r"
[parameters]
  theta TH0(0.0, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0

[simulation]
  n_subjects = 40
  times      = [0.5, 0.5]

[fit_options]
  method  = focei
  maxiter = 2
";
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("dup.ferx");
        std::fs::write(&path, src).expect("write model");
        let (_res, pop) =
            ferx_core::run_model_simulate(path.to_str().unwrap()).expect("simulate must succeed");

        let mut disagreements = 0usize;
        for s in &pop.subjects {
            let st: Vec<usize> = s
                .obs_records
                .iter()
                .filter_map(|r| match r {
                    ObsRecord::DiscreteState { state, .. } => Some(*state),
                    _ => None,
                })
                .collect();
            assert_eq!(st.len(), 2, "both duplicated-time records must survive");
            if st[0] != st[1] {
                disagreements += 1;
            }
        }
        // Two independent p = 0.5 draws disagree half the time; over 40 subjects the
        // chance of zero disagreements by luck is 2^-40. A collapsing write-back gives
        // exactly zero.
        assert!(
            disagreements > 0,
            "duplicated times collapsed onto one shared draw across all 40 subjects"
        );
    }

    /// A predictor that evaluates to NaN yields no simulated row for that record and a
    /// warning naming the CMT and subject — rather than an arbitrary 0/1 drawn from a
    /// meaningless probability. Covers the skip + warning branch of `simulate_binary`.
    ///
    /// The NaN comes from a **NaN covariate**, not a pathological expression: the DSL's
    /// evaluator guards `log(0)`, `0/0` and friends, so an arithmetic route never reaches
    /// this branch. A non-numeric covariate value in the dataset does.
    #[test]
    fn nan_predictor_records_are_skipped_with_a_warning() {
        use ferx_core::SimulateOptions;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        // Same fixture as the other sim tests, but this subject's covariate X is NaN, so
        // `lp = TH0 + THX*X + ETA_I` is NaN at every one of its records.
        let pop = common::binary_pop(&[(f64::NAN, vec![(0.0, 0), (1.0, 1), (2.0, 0)])], 3);
        let out = ferx_core::simulate_with_options_diag(
            &model,
            &pop,
            &model.default_params,
            1,
            &SimulateOptions::default(),
        )
        .expect("simulate must not fail on a NaN predictor");

        assert!(
            out.results.is_empty(),
            "a NaN predictor must emit no rows, got {:?}",
            out.results.len()
        );
        let w = out
            .warnings
            .iter()
            .find(|w| w.contains("[binary_model]"))
            .unwrap_or_else(|| panic!("expected a NaN-predictor warning, got {:?}", out.warnings));
        assert!(w.contains("cmt = 3"), "warning must name the CMT: {w}");
        assert!(
            w.contains("subject 1"),
            "warning must name the subject: {w}"
        );
        assert!(
            w.contains("3 record(s)"),
            "warning must count the records: {w}"
        );
    }

    // ---- Slice 1b: gaps closed after the #900 review -----------------------------------

    /// **Regression (#900 review).** A binary-only `--simulate` whose linear predictor is
    /// NaN for the whole subject produces zero simulated rows, so `sims_by_id.get(id)`
    /// missed and the per-subject loop `continue`d *past* the placeholder cleanup — every
    /// template `state = 0` then reached `fit()` as a fabricated observed failure (the fit
    /// even converged on it). The cleanup now runs on the zero-draw subject too, so the
    /// placeholders are dropped rather than fitted.
    ///
    /// The NaN comes from `TH0 ^ 0.5` at a *negative* `TH0`: `powf` of a negative base with
    /// a fractional exponent is NaN and the DSL evaluator does not guard `Op::Pow`
    /// (`ln`/`sqrt`/`/` are guarded; `pow` is not). It needs no covariate, so a synthetic
    /// `--simulate` subject (empty covariates) reaches it. Note `exp(TH0) - exp(TH0)` does
    /// *not* work here — `x - x` folds to a finite `0`, giving real p = 0.5 draws, not NaN.
    #[test]
    fn binary_only_all_nan_simulate_drops_placeholders_not_fits_them() {
        use ferx_core::types::ObsRecord;
        let src = r"
[parameters]
  theta TH0(-1.0, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0 ^ 0.5

[simulation]
  n_subjects = 4
  times      = [0, 1, 2]

[fit_options]
  method  = focei
  maxiter = 2
";
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("nanbin.ferx");
        std::fs::write(&path, src).expect("write model");
        // With every outcome un-drawable the design legitimately has nothing to fit; the
        // bug produced a *successful* fit on fabricated zeros, so the only wrong outcome
        // is a surviving `DiscreteState` placeholder.
        if let Ok((_res, pop)) = ferx_core::run_model_simulate(path.to_str().unwrap()) {
            let surviving = pop
                .subjects
                .iter()
                .flat_map(|s| s.obs_records.iter())
                .filter(|r| matches!(r, ObsRecord::DiscreteState { .. }))
                .count();
            assert_eq!(
                surviving, 0,
                "every NaN-skipped binary placeholder must be dropped, not fitted as an observed 0"
            );
        }
    }

    /// **The production draw direction, on the fast job.** The sampler draws `state = 1`
    /// iff `u < p`; inverting it (`u > p`, i.e. drawing from `1 − p`) is the exact bug the
    /// SSE exists to catch — but the SSE is `slow-tests`-gated and does not run per PR, and
    /// the Tier-1 `bernoulli_draw_*` test asserts on a *local* re-implementation, not the
    /// production `simulate_binary`. This drives the real `simulate` path at a known
    /// `p = 0.8`, so an inverted comparison (frequency ≈ 0.2) fails here on every PR.
    #[test]
    fn production_binary_draw_frequency_tracks_p() {
        use ferx_core::types::SimOutcome;
        // logit = TH0 with TH0 = ln(0.8 / 0.2) ⇒ p = 0.8 at every record.
        let src = r"
[parameters]
  theta TH0(1.3862943611198906, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = TH0
";
        let model = parse_model_string(src).unwrap();
        let pop = common::binary_pop(&[(0.0, vec![(0.0, 0)])], 3);
        let n = 4000;
        let sims = ferx_core::simulate_with_seed(&model, &pop, &model.default_params, n, 20260722);
        assert_eq!(sims.len(), n, "one row per replicate draw");
        let ones = sims
            .iter()
            .filter(|r| matches!(r.outcome, SimOutcome::Category { state: 1 }))
            .count();
        let freq = ones as f64 / sims.len() as f64;
        // SE ≈ √(0.8·0.2/4000) ≈ 0.0063; 0.8 vs an inverted 0.2 is ~95 SE apart.
        assert!(
            (freq - 0.8).abs() < 0.03,
            "p = 0.8 but realised frequency {freq}; an inverted `u < p` draws ~0.2"
        );
        assert!(
            sims.iter().all(|r| (r.ipred - 0.8).abs() < 1e-9),
            "ipred must carry p = 0.8"
        );
    }

    /// **`raw_time` ≠ `time`.** The predictor runs on the engine's internal
    /// (occasion-shifted) clock while every reported row carries the user's `raw_time` —
    /// the reset-occasion case the field exists for. Every other test sets the two equal
    /// and so cannot tell them apart; a mutation swapping `time` ↔ `raw_time` in
    /// `for_each_binary_lp` passes the whole suite. Here `logit = THT · TIME` with internal
    /// `time = 4.0` and user `raw_time = 1.0` pins both directions at once.
    #[test]
    fn binary_predict_uses_internal_time_and_reports_raw_time() {
        use ferx_core::types::{ObsRecord, Population, Prediction};
        let src = r"
[parameters]
  theta THT(1.0, -10.0, 10.0)

[binary_model]
  cmt   = 3
  logit = THT * TIME
";
        let model = parse_model_string(src).unwrap();
        let mut s = common::subject("1", vec![], vec![], vec![], vec![]);
        s.obs_records = vec![ObsRecord::DiscreteState {
            time: 4.0,     // internal (occasion-shifted) clock the predictor must use
            raw_time: 1.0, // user clock every reported row must carry
            state: 1,
            cmt: 3,
        }];
        let pop = Population {
            subjects: vec![s],
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let preds = ferx_core::predict_categorical(&model, &pop, &model.default_params);
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0].time, 1.0,
            "reported time must be the user raw_time, not the internal clock"
        );
        let Prediction::CatProbs { ref probs } = preds[0].prediction else {
            panic!("expected CatProbs");
        };
        let inv_logit = |x: f64| 1.0 / (1.0 + (-x).exp());
        assert!(
            (probs[1] - inv_logit(4.0)).abs() < 1e-9,
            "predictor must evaluate at internal time 4.0 (p≈{:.3}), got {}",
            inv_logit(4.0),
            probs[1]
        );
        assert!(
            (probs[1] - inv_logit(1.0)).abs() > 0.2,
            "predictor must not have used raw_time 1.0 (p≈{:.3})",
            inv_logit(1.0)
        );
    }

    /// PRED is the population prediction at η = 0; IPRED conditions on the EBE η. The robust
    /// pin that PRED is not accidentally computed at the EBE (a mutation the `∈ [0,1]` range
    /// check in `binary_sdtab_emits_one_row_per_record` misses): sdtab's PRED must equal
    /// `predict_categorical`, which is η = 0 *by contract*. This holds regardless of how far
    /// the EBEs actually move — at `maxiter = 3` they barely do, so a magnitude test on
    /// PRED − IPRED would be far too tight; matching the independent η = 0 predictor is
    /// exact. IPRED is separately checked to *not* be bit-identical to PRED (it conditions
    /// on a non-zero EBE), so the two are genuinely computed at different η.
    #[test]
    fn binary_sdtab_pred_is_population_typical_not_ebe() {
        use ferx_core::io::output::sdtab;
        use ferx_core::types::Prediction;
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let pop = common::binary_pop(&sim_subjects(), 3);
        let res = fit(&model, &pop, &model.default_params, &smoke_opts()).expect("fit");

        // `predict_categorical` is η = 0 by construction; reconstruct the fitted params so
        // it uses the same θ sdtab's PRED does.
        let mut fitted = model.default_params.clone();
        fitted.theta = res.theta.clone();
        let preds = ferx_core::predict_categorical(&model, &pop, &fitted);

        let cols = sdtab(&res, &pop);
        let col = |name: &str| &cols.iter().find(|(n, _)| n == name).expect("column").1;
        let pred = col("PRED");
        let ipred = col("IPRED");
        assert_eq!(pred.len(), preds.len(), "one PRED per binary record");

        // PRED == the η = 0 predictor, row for row. A PRED computed at the EBE diverges here.
        for (k, (&sdtab_pred, ep)) in pred.iter().zip(preds.iter()).enumerate() {
            let Prediction::CatProbs { ref probs } = ep.prediction else {
                panic!("expected CatProbs");
            };
            assert!(
                (sdtab_pred - probs[1]).abs() < 1e-12,
                "row {k}: sdtab PRED {sdtab_pred} must equal the η=0 predictor P(Y=1) {}",
                probs[1]
            );
        }

        // IPRED conditions on a non-zero EBE, so it is not bit-identical to PRED — the two
        // are computed at different η (a mutation collapsing them makes the diff exactly 0).
        assert!(
            pred.iter().zip(ipred.iter()).any(|(p, i)| p != i),
            "IPRED (EBE) must differ from PRED (η=0) for at least one record"
        );
    }

    /// **Mixed Gaussian + binary coexistence.** A joint PK + `[binary_model]` model must
    /// emit *both* row families from a single `simulate()` — the Gaussian emitter and
    /// `simulate_binary` write into separate storage (`observations` vs `obs_records`), and
    /// no test previously exercised them together. Covers the dual-emit seam in
    /// `emit_subject_rows`.
    #[test]
    fn mixed_gaussian_and_binary_simulate_emits_both() {
        use ferx_core::types::{DoseEvent, ObsRecord, Population, SimOutcome};
        let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TH0(0.5, -10.0, 10.0)
  sigma PROP ~ 0.05

[individual_parameters]
  CL = TVCL
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[binary_model]
  cmt   = 3
  logit = TH0
";
        let model = parse_model_string(src).unwrap();
        let mut s = common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![0.5, 1.0, 2.0], // Gaussian obs times on CMT 1
            vec![0.0, 0.0, 0.0],
            vec![1, 1, 1],
        );
        s.obs_records = vec![
            ObsRecord::DiscreteState {
                time: 0.5,
                raw_time: 0.5,
                state: 0,
                cmt: 3,
            },
            ObsRecord::DiscreteState {
                time: 2.0,
                raw_time: 2.0,
                state: 1,
                cmt: 3,
            },
        ];
        let pop = Population {
            subjects: vec![s],
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let rows = ferx_core::simulate(&model, &pop, &model.default_params, 1);
        let n_cont = rows
            .iter()
            .filter(|r| matches!(r.outcome, SimOutcome::Continuous { .. }))
            .count();
        let n_cat = rows
            .iter()
            .filter(|r| matches!(r.outcome, SimOutcome::Category { .. }))
            .count();
        assert_eq!(n_cont, 3, "the 3 Gaussian rows must still emit");
        assert_eq!(n_cat, 2, "the 2 binary rows must coexist, not be dropped");
        assert!(
            rows.iter()
                .any(|r| matches!(r.outcome, SimOutcome::Continuous { .. }) && r.cmt == 1),
            "Gaussian rows on CMT 1"
        );
        assert!(
            rows.iter()
                .any(|r| matches!(r.outcome, SimOutcome::Category { .. }) && r.cmt == 3),
            "binary rows on CMT 3"
        );
    }
}
