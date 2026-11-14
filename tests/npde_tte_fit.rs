//! Tier-2 tests of the time-to-event npde at fit time (#1889): `fit()` with
//! `npde_nsim > 0` stores a score on every single-event TTE row and writes it to the
//! sdtab `NPDE` and `NPD` columns, and a TTE input the score is not defined for is a
//! warning, never a failed fit. A few outer iterations, no convergence.

mod common;

#[cfg(feature = "survival")]
mod npde_tte_fit {
    use crate::common;
    use ferx_core::estimation::uncertainty_samples::fitted_params_from_result;
    use ferx_core::io::output::write_sdtab_csv;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::stats::npde::compute_tte_npde;
    use ferx_core::types::{
        CompiledModel, DoseEvent, EventType, ModelParameters, ObsRecord, Population,
    };
    use ferx_core::{fit, FitOptions, FitResult};

    /// Joint model: a one-compartment concentration on cmt 1 and an exponential hazard
    /// on cmt 2, each with its own random effect.
    const JOINT: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVLAMBDA(0.05, 0.001, 10.0)
  block_omega (ETA_CL, ETA_LAMBDA) = [0.09, 0.03, 0.25]
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA_LAMBDA)
";

    const EXP: &str = "\n[parameters]\n  theta TVLAMBDA(0.05, 0.001, 10.0)\n  omega ETA_LAMBDA ~ 0.09\n\n[event_model]\n  cmt    = 2\n  family = exponential\n  scale  = TVLAMBDA * exp(ETA_LAMBDA)\n";

    const RTTE: &str = "\n[parameters]\n  theta TVLAMBDA(0.15, 0.001, 10.0)\n\n[event_model]\n  cmt    = 2\n  type   = rtte\n  family = exponential\n  scale  = TVLAMBDA\n";

    const COMPETING: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B
";

    const GOMPERTZ_CURE: &str = "\n[parameters]\n  theta TVA(0.2, 0.001, 10.0)\n  theta TVG(-0.5, -5.0, 5.0)\n\n[event_model]\n  cmt    = 2\n  family = gompertz\n  alpha  = TVA\n  gamma  = TVG\n";

    fn options(nsim: usize) -> FitOptions {
        FitOptions {
            outer_maxiter: 3,
            npde_nsim: nsim,
            npde_seed: Some(2024),
            ..FitOptions::default()
        }
    }

    /// A deterministic uniform on (0, 1) from an integer (a Weyl sequence).
    fn unif(i: usize) -> f64 {
        ((i as f64 + 1.0) * 0.618_033_988_749_895).fract()
    }

    /// 30 joint subjects: a dose, two concentrations and, except for five of them, one
    /// TTE record, about half of those censored.
    fn joint_population() -> Population {
        let subjects = (0..30)
            .map(|i| {
                let conc =
                    |t: f64| 10.0 * (-0.1 * t).exp() * (0.8 + 0.4 * unif(3 * i + t as usize));
                let mut s = common::subject(
                    &format!("{}", i + 1),
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    vec![1.0, 6.0],
                    vec![conc(1.0), conc(6.0)],
                    vec![1, 1],
                );
                if i % 6 != 4 {
                    let event = i % 2 == 0;
                    s.obs_records = vec![ObsRecord::Event {
                        time: if event {
                            -(1.0 - unif(i)).ln() / 0.08
                        } else {
                            4.0 + 20.0 * unif(i + 100)
                        },
                        event_type: if event {
                            EventType::Exact
                        } else {
                            EventType::RightCensored
                        },
                        entry_time: 0.0,
                        cmt: 2,
                    }];
                }
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

    /// The fitted parameters as a post-fit caller gets them: the public accessor on
    /// the `FitResult`. Ω is a correlated block here, so its Cholesky factor is
    /// recomputed from the reported matrix and need not equal the factor the fit held
    /// to the last bit. The scores are integer counts, so they agree all the same.
    fn fitted_params(model: &CompiledModel, result: &FitResult) -> ModelParameters {
        fitted_params_from_result(result, model).expect("fitted parameters")
    }

    fn skipped(result: &FitResult) -> Vec<&String> {
        result
            .warnings
            .iter()
            .filter(|w| w.starts_with("W_NPDE_TTE_SKIPPED"))
            .collect()
    }

    fn tte_scores(result: &FitResult) -> Vec<Vec<f64>> {
        result
            .subjects
            .iter()
            .map(|s| s.tte_rows.iter().map(|r| r.npde).collect())
            .collect()
    }

    /// The sentence every `W_NPDE_TTE_SKIPPED` opens with; the reason follows it.
    const SKIPPED: &str = "W_NPDE_TTE_SKIPPED: the time-to-event records were not scored, so \
        the NPDE / NPD cells of their rows are blank. Every other row is scored as usual. \
        Reason: ";

    /// T14. `fit()` scores every TTE row, bit-equal to `compute_tte_npde` called after
    /// the fit with the fitted parameters and the recorded seed; the sdtab file has the
    /// score in both `NPDE` and `NPD` of the TTE rows, and the concentration rows keep
    /// their own scores. Five subjects have no TTE record.
    ///
    /// Dies on: the initial parameters passed instead of the fitted ones; another seed;
    /// scores assigned in another subject order, or flattened across the subjects with
    /// no record; the score written to one column only.
    #[test]
    fn fit_stores_the_scores_the_post_fit_call_gives() {
        let model = parse_model_string(JOINT).expect("model parses");
        let pop = joint_population();
        let result = fit(&model, &pop, &model.default_params, &options(200)).expect("fit");
        assert_ne!(
            result.theta, model.default_params.theta,
            "premise: the fit moved off its start, so fitted and initial parameters differ"
        );
        assert!(skipped(&result).is_empty(), "{:?}", result.warnings);

        let (direct, _) = compute_tte_npde(
            &model,
            &pop,
            &fitted_params(&model, &result),
            200,
            result.npde_seed,
            None,
        )
        .expect("post-fit scores");
        let stored = tte_scores(&result);
        assert_eq!(stored.len(), direct.len());
        let mut n_scored = 0;
        for (i, (got, want)) in stored.iter().zip(&direct).enumerate() {
            let has_record = i % 6 != 4;
            assert_eq!(got.len(), usize::from(has_record), "subject index {i}");
            assert_eq!(want.len(), got.len(), "subject index {i}");
            for (g, w) in got.iter().zip(want) {
                assert!(g.is_finite(), "subject index {i}: {g}");
                assert_eq!(g.to_bits(), w.npde.to_bits(), "subject index {i}");
                n_scored += 1;
            }
        }
        assert_eq!(n_scored, 25);
        // Not one value repeated: an order mix-up would be visible.
        assert!(stored.iter().flatten().any(|v| *v != stored[0][0]));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sdtab.csv");
        write_sdtab_csv(&result, &pop, path.to_str().unwrap()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        let at = |name: &str| {
            header
                .iter()
                .position(|h| *h == name)
                .unwrap_or_else(|| panic!("{name} in {header:?}"))
        };
        let (id, cmt, npde, npd, epred) = (at("ID"), at("CMT"), at("NPDE"), at("NPD"), at("EPRED"));
        let (mut n_tte, mut n_gaussian) = (0, 0);
        for line in lines {
            let cells: Vec<&str> = line.split(',').collect();
            let subject: usize = cells[id].parse::<f64>().unwrap() as usize - 1;
            if cells[cmt] == "2.000000" {
                let want = format!("{:.6}", stored[subject][0]);
                assert_eq!(cells[npde], want, "NPDE of subject index {subject}");
                assert_eq!(cells[npd], want, "NPD of subject index {subject}");
                assert_eq!(cells[epred], "", "EPRED on a TTE row");
                n_tte += 1;
            } else {
                assert!(!cells[npde].is_empty() && !cells[epred].is_empty());
                n_gaussian += 1;
            }
        }
        assert_eq!((n_tte, n_gaussian), (25, 60));
    }

    /// A fit with `npde_nsim = 0` scores nothing and says nothing.
    #[test]
    fn no_npde_requested_no_score_no_warning() {
        let model = parse_model_string(JOINT).expect("model parses");
        let result = fit(
            &model,
            &joint_population(),
            &model.default_params,
            &options(0),
        )
        .expect("fit");
        assert!(tte_scores(&result).iter().flatten().all(|v| v.is_nan()));
        assert!(skipped(&result).is_empty());
    }

    /// A model with no TTE endpoint makes no call: Gaussian scores, no warning.
    ///
    /// Dies on: the TTE step run for every fit with `npde_nsim > 0`, which would tell a
    /// concentration-only fit that its (absent) TTE records were skipped.
    #[test]
    fn a_fit_without_a_tte_endpoint_is_not_warned() {
        let src = JOINT.split("[event_model]").next().unwrap();
        let model = parse_model_string(src).expect("model parses");
        let mut pop = joint_population();
        for s in &mut pop.subjects {
            s.obs_records.clear();
        }
        let result = fit(&model, &pop, &model.default_params, &options(200)).expect("fit");
        assert!(skipped(&result).is_empty(), "{:?}", result.warnings);
        assert!(result.subjects.iter().all(|s| s.npde.len() == 2));
    }

    /// T15. An input the score is not defined for leaves the fit `Ok`, the TTE cells
    /// NaN, and exactly one warning quoting the reason. One case per reason, each
    /// asserted in full, so no reason can be reported as another.
    ///
    /// Dies on: the refusal propagated with `?` (the fit fails); the warning pushed per
    /// subject; two reasons' messages swapped; any sentence of the warning removed.
    #[test]
    fn unscorable_tte_input_is_one_warning_not_a_failed_fit() {
        let interval = {
            let mut p = common::tte_pop_from_pairs(&[(5.0, 1), (9.0, 0), (3.0, 1), (12.0, 0)]);
            p.subjects[2].obs_records = vec![ObsRecord::Event {
                time: 6.0,
                event_type: EventType::IntervalCensored {
                    left: 2.0,
                    right: 6.0,
                },
                entry_time: 0.0,
                cmt: 2,
            }];
            p
        };
        let truncated = {
            let mut p = common::tte_pop_from_pairs(&[(5.0, 1), (9.0, 0), (3.0, 1), (12.0, 0)]);
            p.subjects[1].obs_records = vec![ObsRecord::Event {
                time: 9.0,
                event_type: EventType::RightCensored,
                entry_time: 1.5,
                cmt: 2,
            }];
            p
        };
        let rtte = {
            let mut p = common::tte_pop_from_pairs(&[(30.0, 0), (30.0, 0), (30.0, 0)]);
            for (s, events) in p.subjects.iter_mut().zip([&[4.0, 11.0][..], &[7.0], &[]]) {
                let mut records: Vec<ObsRecord> = events
                    .iter()
                    .map(|&t| ObsRecord::Event {
                        time: t,
                        event_type: EventType::Exact,
                        entry_time: 0.0,
                        cmt: 2,
                    })
                    .collect();
                records.push(ObsRecord::Event {
                    time: 30.0,
                    event_type: EventType::RightCensored,
                    entry_time: 0.0,
                    cmt: 2,
                });
                s.obs_records = records;
            }
            p
        };
        let competing = common::tte_competing_pop(&[(5.0, 2), (9.0, 3), (3.0, 0), (12.0, 2)]);

        let cases: [(&str, &str, Population, &str); 4] = [
            (
                "rtte",
                RTTE,
                rtte,
                "compute_tte_npde: cmt 2 is a repeated events (RTTE) endpoint. npde for \
                 repeated events is not supported yet.",
            ),
            (
                "competing",
                COMPETING,
                competing,
                "compute_tte_npde: the model has TTE endpoints on cmt 2, 3 (competing risks). \
                 npde under competing risks is not supported yet.",
            ),
            (
                "left truncation",
                EXP,
                truncated,
                "compute_tte_npde: subject 2 has entry time 1.5 on cmt 2: left truncation \
                 (delayed entry) is not supported yet.",
            ),
            (
                "interval",
                EXP,
                interval,
                "compute_tte_npde: subject 3 has an interval-censored record on cmt 2; npde \
                 needs an exact or a right-censored time.",
            ),
        ];
        for (label, src, pop, reason) in cases {
            let model = parse_model_string(src).expect("model parses");
            let result = fit(&model, &pop, &model.default_params, &options(200))
                .unwrap_or_else(|e| panic!("{label}: the fit must not fail: {e}"));
            let rows: Vec<f64> = tte_scores(&result).into_iter().flatten().collect();
            assert!(!rows.is_empty(), "{label}: premise, the fit has TTE rows");
            assert!(rows.iter().all(|v| v.is_nan()), "{label}: {rows:?}");
            let warnings = skipped(&result);
            assert_eq!(warnings.len(), 1, "{label}: {:?}", result.warnings);
            assert_eq!(*warnings[0], format!("{SKIPPED}{reason}"), "{label}");
        }
    }

    /// A reference simulation the simulator refuses is the same single warning, quoting
    /// the simulator's own message, with no claim that the model kind is unsupported
    /// (review r1 #5). Every record here is at time 0, so the latest scored time, the
    /// simulation horizon, is 0, which the simulator refuses.
    ///
    /// Dies on: a simulator `Err` propagated out of the fit; its message replaced by a
    /// refusal's.
    #[test]
    fn a_refused_reference_simulation_is_the_same_warning() {
        let model = parse_model_string(EXP).expect("model parses");
        let pop = common::tte_pop_from_pairs(&[(0.0, 0), (0.0, 0), (0.0, 0)]);
        let result = fit(&model, &pop, &model.default_params, &options(200)).expect("fit");
        assert!(tte_scores(&result).iter().flatten().all(|v| v.is_nan()));
        let warnings = skipped(&result);
        assert_eq!(warnings.len(), 1, "{:?}", result.warnings);
        assert_eq!(
            *warnings[0],
            format!("{SKIPPED}SimulateOptions.horizon must be finite and > 0 (got 0)")
        );
        assert!(!warnings[0].contains("not supported"));
    }

    /// A correct fit of a cure-fraction model carries no hazard warning (review r1 #1).
    /// A Gompertz hazard with `gamma < 0` has replicates that never have an event; the
    /// simulator reports each as `W_TTE_DEGENERATE_HAZARD`, which `compute_tte_npde`
    /// returns once with a count, and which the fit does not repeat as its own warning.
    ///
    /// Dies on: the degenerate-hazard warning forwarded to the fit; the fold dropped
    /// from the direct call.
    #[test]
    fn a_cure_fraction_fit_is_not_warned_about_its_reference_draws() {
        let model = parse_model_string(GOMPERTZ_CURE).expect("model parses");
        let pop = common::tte_pop_from_pairs(&[(1.0, 1), (30.0, 0), (4.0, 1), (2.0, 1)]);
        let result = fit(&model, &pop, &model.default_params, &options(200)).expect("fit");
        assert!(skipped(&result).is_empty(), "{:?}", result.warnings);
        assert!(tte_scores(&result).iter().flatten().all(|v| v.is_finite()));
        assert!(
            !result.warnings.iter().any(|w| w.contains("W_TTE_")),
            "{:?}",
            result.warnings
        );
        let (_, direct) = compute_tte_npde(
            &model,
            &pop,
            &fitted_params(&model, &result),
            200,
            result.npde_seed,
            None,
        )
        .expect("post-fit scores");
        assert_eq!(direct.len(), 1, "{direct:?}");
        assert!(
            direct[0].starts_with("W_TTE_DEGENERATE_HAZARD: "),
            "{}",
            direct[0]
        );
        assert!(
            direct[0].contains("warnings were raised over the 200 npde reference replicates"),
            "{}",
            direct[0]
        );
    }
}
