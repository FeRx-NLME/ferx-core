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

    /// Two-cause competing-risks TTE (cause-specific hazards on CMT 2 and CMT 3),
    /// linked by a shared frailty `ETA_F`. TTE-only (no PK blocks).
    const COMPETING_RISKS_MODEL: &str = r"
[parameters]
  theta TVLAMBDA_A(0.05, 0.001, 10.0)
  theta TVLAMBDA_B(0.03, 0.001, 10.0)
  omega ETA_F ~ 0.09

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
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

    /// `simulate()` must administratively right-censor TTE draws at each subject's
    /// observation window (the `ObsRecord::Event.time`) rather than emit every draw
    /// as an uncensored event. Drives the full `simulate_tte` wiring; the draw/censor
    /// logic itself is unit-tested in `survival/mod.rs` (`draw_tte_*`).
    #[test]
    fn simulate_tte_censors_at_observation_window() {
        use ferx_core::{simulate_with_seed, SimOutcome};
        // 300 subjects sharing a τ=20 window. λ≈0.05 ⇒ S(20)=exp(−1)≈0.37 censored,
        // so the run must contain both observed events and administrative censors.
        const TAU: f64 = 20.0;
        let model = parse_model_string(EXP_TTE_MODEL).expect("model must parse");
        let template = common::tte_pop_from_pairs(&vec![(TAU, 0); 300]);

        let sims = simulate_with_seed(&model, &template, &model.default_params, 1, 4242);
        assert_eq!(sims.len(), 300, "one TTE outcome per template subject");

        let (mut events, mut censored) = (0usize, 0usize);
        for r in &sims {
            match r.outcome {
                SimOutcome::Event { time, observed } => {
                    assert!(time <= TAU, "no outcome may exceed the window: {time}");
                    if observed {
                        assert!(
                            time < TAU,
                            "an observed event must precede the window: {time}"
                        );
                        events += 1;
                    } else {
                        assert_eq!(time, TAU, "a censored outcome sits exactly at the window");
                        censored += 1;
                    }
                }
                ref other => panic!("expected an Event outcome, got {other:?}"),
            }
        }
        assert!(events > 0, "expected some observed events (got {events})");
        assert!(
            censored > 0,
            "expected some administratively censored subjects (got {censored})"
        );
    }

    /// An **exact-event** template (`dv = 1`) carries no administrative horizon —
    /// its record `time` is the realized event time, not a censoring window — so
    /// `simulate()` must draw every outcome uncensored from the model's full
    /// predictive distribution rather than truncate at that event time. Guards
    /// against re-introducing the re-simulation / VPC truncation bias.
    #[test]
    fn simulate_tte_exact_event_template_draws_uncensored() {
        use ferx_core::{simulate_with_seed, SimOutcome};
        // 300 exact-event rows at t=5. The (old, buggy) behaviour would have
        // censored every draw past t=5 at the event time; the correct behaviour
        // ignores it as a horizon and draws fresh, finite event times.
        let model = parse_model_string(EXP_TTE_MODEL).expect("model must parse");
        let template = common::tte_pop_from_pairs(&vec![(5.0, 1); 300]);

        let sims = simulate_with_seed(&model, &template, &model.default_params, 1, 1234);
        assert_eq!(sims.len(), 300, "one TTE outcome per template subject");

        let mut beyond_event_time = 0usize;
        for r in &sims {
            match r.outcome {
                SimOutcome::Event { time, observed } => {
                    assert!(
                        observed,
                        "an exact-event template carries no horizon → every draw is observed"
                    );
                    assert!(
                        time.is_finite() && time > 0.0,
                        "event time must be finite positive: {time}"
                    );
                    if time > 5.0 {
                        beyond_event_time += 1;
                    }
                }
                ref other => panic!("expected an Event outcome, got {other:?}"),
            }
        }
        // With λ≈0.05 the median event time is ~14, so the bulk of draws land
        // past t=5 — impossible under the old truncate-at-event-time behaviour.
        assert!(
            beyond_event_time > 0,
            "expected draws beyond the record's event time (got {beyond_event_time})"
        );
    }

    /// Competing risks (two cause-specific hazards on CMT 2 and CMT 3, shared
    /// frailty): the model fits to a finite OFV and `predict_survival` reports a
    /// cause-specific cumulative incidence with `Σ_k CIF_k(t) + S_all(t) = 1`.
    #[test]
    fn competing_risks_fits_and_predicts_cif() {
        use ferx_core::predict_survival;
        use std::collections::HashMap;
        let model =
            parse_model_string(COMPETING_RISKS_MODEL).expect("competing-risks model must parse");
        assert!(
            model.endpoints.contains_key(&2) && model.endpoints.contains_key(&3),
            "both cause CMTs must be registered as TTE endpoints"
        );

        // 24 subjects: a mix of cause-2 events, cause-3 events, and censored.
        let rows: Vec<(f64, u8)> = (0..24)
            .map(|i| match i % 3 {
                0 => (5.0 + i as f64 * 0.3, 2),
                1 => (7.0 + i as f64 * 0.2, 3),
                _ => (30.0, 0),
            })
            .collect();
        let pop = common::tte_competing_pop(&rows);

        let mut opts = FitOptions::default();
        opts.verbose = false;
        let r = fit(&model, &pop, &model.default_params, &opts)
            .expect("competing-risks fit must not error");
        assert!(r.ofv.is_finite(), "OFV must be finite; got {}", r.ofv);

        // CIF invariant: the two CMT rows at the same (subject, time) must
        // partition 1 with the all-cause survival.
        let grid = [0.0, 2.0, 6.0, 15.0, 40.0];
        let preds = predict_survival(&model, &pop, &model.default_params, &grid);
        assert!(!preds.is_empty(), "predict_survival must return rows");
        let mut by_key: HashMap<(String, u64), (f64, f64)> = HashMap::new();
        for p in &preds {
            assert!(
                (0.0..=1.0).contains(&p.cif),
                "CIF must be in [0,1]: {}",
                p.cif
            );
            let entry = by_key
                .entry((p.id.clone(), p.time.to_bits()))
                .or_insert((0.0, p.survival_all));
            entry.0 += p.cif;
        }
        for (_, (sum_cif, s_all)) in by_key {
            assert!(
                (sum_cif + s_all - 1.0).abs() < 1e-9,
                "Σ CIF + S_all must equal 1: {sum_cif} + {s_all}"
            );
        }
    }

    /// Competing-risks simulation: each subject yields one row per cause CMT with
    /// a shared outcome time; at most one cause is observed (the earliest), the
    /// rest right-censored at that time. Over many subjects all three outcomes
    /// (cause-2 event, cause-3 event, censored) appear.
    #[test]
    fn simulate_competing_risks_earliest_cause_wins() {
        use ferx_core::{simulate_with_seed, SimOutcome};
        use std::collections::HashMap;
        const TAU: f64 = 25.0;
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model must parse");
        // Template: 400 subjects censored at τ on both causes; the draw overwrites.
        let template = common::tte_competing_pop(&vec![(TAU, 0u8); 400]);
        let sims = simulate_with_seed(&model, &template, &model.default_params, 1, 7);

        let mut by_id: HashMap<String, Vec<&ferx_core::SimulationResult>> = HashMap::new();
        for s in &sims {
            by_id.entry(s.id.clone()).or_default().push(s);
        }
        assert_eq!(by_id.len(), 400, "one group per subject");

        let (mut ev2, mut ev3, mut cens) = (0usize, 0usize, 0usize);
        for rowset in by_id.values() {
            assert_eq!(rowset.len(), 2, "one row per cause CMT");
            let t0 = match &rowset[0].outcome {
                SimOutcome::Event { time, .. } => *time,
                _ => panic!("expected an Event outcome"),
            };
            let (mut observed_count, mut observed_cmt) = (0usize, 0usize);
            for s in rowset {
                let (time, observed) = match &s.outcome {
                    SimOutcome::Event { time, observed } => (*time, *observed),
                    _ => panic!("expected an Event outcome"),
                };
                assert!(
                    (time - t0).abs() < 1e-12,
                    "both rows share the outcome time"
                );
                assert!(time <= TAU + 1e-12, "outcome time within the window");
                if observed {
                    observed_count += 1;
                    observed_cmt = s.cmt;
                }
            }
            assert!(
                observed_count <= 1,
                "at most one cause observed per subject"
            );
            match observed_cmt {
                2 => ev2 += 1,
                3 => ev3 += 1,
                _ => cens += 1,
            }
        }
        assert!(
            ev2 > 0 && ev3 > 0 && cens > 0,
            "expected cause-2, cause-3, and censored subjects (got {ev2}/{ev3}/{cens})"
        );
    }

    /// Re-simulating a competing-risks subject that already has an observed event
    /// (an `Exact` record, the sibling cause `RightCensored` at the same time)
    /// must NOT truncate the fresh draw at that original event time: an event
    /// record carries no administrative horizon (+∞ from `observation_window`,
    /// #494), so a simulated outcome can fall after it. Guards against the
    /// per-cause window collapsing to the earliest (`min`) horizon.
    #[test]
    fn simulate_competing_risks_event_record_draws_uncensored() {
        use ferx_core::{simulate_with_seed, SimOutcome};
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model must parse");
        // 300 subjects, each with cause A (CMT 2) observed at t=0.5 and cause B
        // (CMT 3) censored at 0.5 — the cause-specific layout of an early event.
        let template = common::tte_competing_pop(&vec![(0.5_f64, 2u8); 300]);
        let sims = simulate_with_seed(&model, &template, &model.default_params, 1, 99);

        let max_t = sims
            .iter()
            .filter_map(|s| match s.outcome {
                SimOutcome::Event { time, .. } => Some(time),
                _ => None,
            })
            .fold(0.0_f64, f64::max);
        // With the #494-consistent horizon (Exact ⇒ +∞) the redraw is unbounded,
        // so outcomes land well after the original 0.5; the buggy `min` horizon
        // clamped every outcome to 0.5.
        assert!(
            max_t > 0.5 + 1e-6,
            "event-bearing competing subjects must redraw uncensored, not clamp to \
             the original event time (max outcome {max_t})"
        );
    }

    /// Competing-risks VPC (#522): re-simulating event-bearing data **with** an
    /// explicit `[simulation] horizon` must (a) decouple the draw from the data's
    /// own event times — simulated events may land *after* the original event
    /// time, not truncated at it — and (b) administratively censor at the planned
    /// horizon, so no outcome exceeds it and a subject surviving every cause past
    /// the horizon lands exactly on it. The complement of
    /// `simulate_competing_risks_event_record_draws_uncensored` (no horizon ⇒
    /// unbounded redraw): the horizon overrides the per-record `observation_window`.
    #[test]
    fn simulate_competing_risks_horizon_censors_at_planned_end() {
        use ferx_core::{simulate_with_options, SimOutcome, SimulateOptions};
        use std::collections::HashMap;
        const H: f64 = 14.0; // planned study end (≈ above the ~8.7 median total event time)
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model must parse");
        // 300 subjects with cause A (CMT 2) observed at t=0.5 — the data's event
        // time, which must NOT bound the re-simulation under an explicit horizon.
        let template = common::tte_competing_pop(&vec![(0.5_f64, 2u8); 300]);
        let opts = SimulateOptions {
            seed: Some(99),
            match_method: None,
            horizon: Some(H),
        };
        let sims = simulate_with_options(&model, &template, &model.default_params, 1, &opts)
            .expect("simulate with horizon must succeed");

        // Group the two cause rows per subject (they share one outcome time, with
        // at most one cause observed — the earliest-cause-wins layout).
        let mut by_id: HashMap<String, Vec<&ferx_core::SimulationResult>> = HashMap::new();
        for s in &sims {
            by_id.entry(s.id.clone()).or_default().push(s);
        }
        assert_eq!(by_id.len(), 300, "one group per subject");

        let (mut events_after_original, mut censored_at_h) = (0usize, 0usize);
        for rowset in by_id.values() {
            assert_eq!(rowset.len(), 2, "one row per cause CMT");
            let mut t0 = None;
            let mut observed_any = false;
            for s in rowset {
                let (time, observed) = match s.outcome {
                    SimOutcome::Event { time, observed } => (time, observed),
                    _ => panic!("expected an Event outcome"),
                };
                // (b) the horizon is a hard cap: no outcome can exceed the planned end.
                assert!(time <= H + 1e-9, "outcome {time} exceeds horizon {H}");
                let prev = *t0.get_or_insert(time);
                assert!(
                    (time - prev).abs() < 1e-12,
                    "both rows share the outcome time"
                );
                observed_any |= observed;
            }
            let t0 = t0.unwrap();
            if observed_any {
                // (a) decoupled from the data: events may fall after the original 0.5.
                if t0 > 0.5 + 1e-6 {
                    events_after_original += 1;
                }
            } else {
                // A subject with no cause firing is censored administratively at the
                // horizon — exactly, not at the data's 0.5 event time.
                assert!(
                    (t0 - H).abs() < 1e-9,
                    "a fully-censored subject must land on the horizon, got {t0}"
                );
                censored_at_h += 1;
            }
        }
        // A proper VPC mix: events past the original event time, plus survivors
        // censored at the planned horizon.
        assert!(
            events_after_original > 0,
            "horizon must not truncate the redraw at the original 0.5 event time"
        );
        assert!(
            censored_at_h > 0,
            "expected subjects surviving every cause past the horizon (censored at H)"
        );
    }

    /// `[simulation]`-block path (#522): `run_model_simulate` (the `--simulate`
    /// CLI entry) must generate one TTE row per cause CMT per synthetic subject,
    /// censored at the `[simulation] horizon`. Without it the synthetic
    /// `obs_records` are empty and `--simulate` emits zero TTE rows. A
    /// fixed-effects (n_eta=0) competing model keeps the bundled fit fast.
    #[test]
    fn simulate_block_generates_competing_tte_rows() {
        const H: f64 = 8.0;
        let model_src = r"
[parameters]
  theta TVLAMBDA_A(0.10, 0.001, 10.0)
  theta TVLAMBDA_B(0.06, 0.001, 10.0)

[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A

[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B

[fit_options]
  method  = focei
  maxiter = 1

[simulation]
  n_subjects = 40
  horizon    = 8
  seed       = 7
";
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("competing_sim.ferx");
        std::fs::write(&path, model_src).expect("write model");

        let (_fit, pop) =
            ferx_core::run_model_simulate(path.to_str().unwrap()).expect("simulate+fit succeeds");
        assert_eq!(
            pop.subjects.len(),
            40,
            "one synthetic subject per n_subjects"
        );

        let (mut events, mut censored_at_h) = (0usize, 0usize);
        for subj in &pop.subjects {
            // One row per cause CMT (2 and 3), each at entry 0 and within horizon.
            assert_eq!(subj.obs_records.len(), 2, "one TTE row per cause CMT");
            let mut cmts: Vec<usize> = subj
                .obs_records
                .iter()
                .map(|r| {
                    let ObsRecord::Event {
                        cmt,
                        entry_time,
                        time,
                        ..
                    } = r;
                    assert_eq!(
                        *entry_time, 0.0,
                        "synthetic subjects have no left truncation"
                    );
                    assert!(*time <= H + 1e-9, "row time {time} exceeds horizon {H}");
                    *cmt
                })
                .collect();
            cmts.sort_unstable();
            assert_eq!(cmts, vec![2, 3], "rows on both cause CMTs");

            let n_observed = subj
                .obs_records
                .iter()
                .filter(|r| {
                    let ObsRecord::Event { event_type, .. } = r;
                    matches!(event_type, EventType::Exact)
                })
                .count();
            assert!(n_observed <= 1, "at most one cause is the observed event");
            if n_observed == 1 {
                events += 1;
            } else {
                // No cause fired ⇒ both rows right-censored at the horizon.
                for r in &subj.obs_records {
                    let ObsRecord::Event {
                        time, event_type, ..
                    } = r;
                    assert!(matches!(event_type, EventType::RightCensored));
                    assert!(
                        (time - H).abs() < 1e-9,
                        "censored row must land on the horizon, got {time}"
                    );
                }
                censored_at_h += 1;
            }
        }
        assert!(
            events > 0,
            "synthetic competing data must contain observed events"
        );
        assert!(
            censored_at_h > 0,
            "with horizon {H} some subjects survive every cause and censor at it"
        );
    }

    /// `[simulation]`-block path with a **single** `[event_model]` (#522): the
    /// docs promise "a single `[event_model]` yields one row per subject — an event
    /// before the horizon, or right-censoring at the horizon". The competing test
    /// above covers ≥2 causes; this pins the single-cause `run_model_simulate` path
    /// end-to-end: exactly one TTE row per synthetic subject, at entry 0 and within
    /// the horizon, with a mix of observed events and horizon-censored survivors.
    #[test]
    fn simulate_block_generates_single_cause_tte_rows() {
        const H: f64 = 8.0;
        let model_src = r"
[parameters]
  theta TVLAMBDA(0.10, 0.001, 10.0)

[event_model only_cause]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[fit_options]
  method  = focei
  maxiter = 1

[simulation]
  n_subjects = 60
  horizon    = 8
  seed       = 11
";
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("single_cause_sim.ferx");
        std::fs::write(&path, model_src).expect("write model");

        let (_fit, pop) =
            ferx_core::run_model_simulate(path.to_str().unwrap()).expect("simulate+fit succeeds");
        assert_eq!(
            pop.subjects.len(),
            60,
            "one synthetic subject per n_subjects"
        );

        let (mut events, mut censored_at_h) = (0usize, 0usize);
        for subj in &pop.subjects {
            assert_eq!(subj.obs_records.len(), 1, "single cause ⇒ one TTE row");
            let ObsRecord::Event {
                time,
                event_type,
                entry_time,
                cmt,
            } = &subj.obs_records[0];
            assert_eq!(*cmt, 2, "row on the cause CMT");
            assert_eq!(
                *entry_time, 0.0,
                "synthetic subjects have no left truncation"
            );
            assert!(*time <= H + 1e-9, "row time {time} exceeds horizon {H}");
            match event_type {
                EventType::Exact => events += 1,
                EventType::RightCensored => {
                    assert!(
                        (time - H).abs() < 1e-9,
                        "a censored row must land on the horizon, got {time}"
                    );
                    censored_at_h += 1;
                }
                other => panic!("unexpected event_type {other:?}"),
            }
        }
        assert!(
            events > 0,
            "synthetic single-cause data must contain observed events"
        );
        assert!(
            censored_at_h > 0,
            "with horizon {H} some subjects survive past it and censor there"
        );
    }

    /// A TTE model simulated via `[simulation]` *requires* a `horizon` (there are
    /// no continuous `times` to censor against). `times` alone satisfies the
    /// parser, but `run_model_simulate` must then reject the TTE design with a
    /// clear, actionable error (#522).
    #[test]
    fn simulate_block_tte_requires_horizon() {
        let model_src = r"
[parameters]
  theta TVLAMBDA(0.10, 0.001, 10.0)

[event_model only_cause]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[simulation]
  n_subjects = 3
  times      = [1.0]
";
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("tte_no_horizon.ferx");
        std::fs::write(&path, model_src).expect("write model");

        let err = ferx_core::run_model_simulate(path.to_str().unwrap())
            .expect_err("a TTE [simulation] without a horizon must error");
        assert!(
            err.contains("horizon") && err.contains("TTE"),
            "error must name the missing horizon: {err}"
        );
    }

    /// A joint **PK + TTE** `[simulation]` (both `times` and `horizon` set):
    /// continuous PK observations and TTE event rows are generated for the same
    /// subjects, the Gaussian write-back routes the continuous rows into
    /// `observations`, and the TTE write-back routes the event rows into
    /// `obs_records` (its non-`Event` `filter_map` arm sees the continuous rows).
    /// Exercises the mixed path end-to-end (#522 review).
    #[test]
    fn simulate_block_mixed_pk_and_tte() {
        let model_src = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAMBDA(0.10, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[event_model only_cause]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[fit_options]
  method  = focei
  maxiter = 1

[simulation]
  n_subjects = 6
  dose_amt   = 100
  dose_cmt   = 1
  times      = [0.5, 2.0, 8.0]
  horizon    = 14
  seed       = 3
";
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mixed_pk_tte.ferx");
        std::fs::write(&path, model_src).expect("write model");

        let (_fit, pop) =
            ferx_core::run_model_simulate(path.to_str().unwrap()).expect("mixed simulate+fit");
        assert_eq!(pop.subjects.len(), 6);
        for subj in &pop.subjects {
            // Continuous PK observations: one per `times` entry, written by the
            // Gaussian write-back.
            assert_eq!(subj.observations.len(), 3, "3 continuous PK observations");
            // One TTE row on the cause CMT, written by the TTE write-back. Reaching
            // this also drives the non-`Event` `filter_map` arm (the continuous rows).
            assert_eq!(subj.obs_records.len(), 1, "one TTE event row");
            let ObsRecord::Event { time, cmt, .. } = &subj.obs_records[0];
            assert_eq!(*cmt, 2, "TTE row on the cause CMT");
            assert!(*time <= 14.0 + 1e-9, "TTE outcome within the horizon");
        }
    }

    /// A **joint** PK+TTE model given only `horizon` (no `times`) must error, not
    /// silently simulate zero PK observations. The model has a residual-error
    /// (continuous) endpoint, so `times` is required even though it also has a TTE
    /// endpoint — the mixed-model half of the silent-data gap the pure-Gaussian
    /// guard left open (#522 review).
    #[test]
    fn simulate_block_mixed_horizon_only_errors() {
        let model_src = r"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV(50.0, 1.0, 500.0)
  theta TVLAMBDA(0.10, 0.001, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[event_model only_cause]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA

[simulation]
  n_subjects = 4
  dose_amt   = 100
  dose_cmt   = 1
  horizon    = 14
";
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("mixed_horizon_only.ferx");
        std::fs::write(&path, model_src).expect("write model");

        let err = ferx_core::run_model_simulate(path.to_str().unwrap())
            .expect_err("a joint PK+TTE model with horizon but no times must error");
        assert!(
            err.contains("times") && err.contains("continuous"),
            "error must point at the missing continuous `times`: {err}"
        );
    }

    /// The library `simulate_with_options` validates `horizon` the same way the
    /// `.ferx` parser does: a non-finite or non-positive horizon is rejected, so a
    /// NaN window (every `t_event < NaN` is false → silent NaN event times) or a
    /// `<= 0` horizon (censors every subject at/before entry) cannot slip in via
    /// the API (#522 review).
    #[test]
    fn simulate_with_options_rejects_bad_horizon() {
        use ferx_core::{simulate_with_options, SimulateOptions};
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model parses");
        let pop = common::tte_competing_pop(&vec![(0.5_f64, 2u8); 4]);
        for bad in [f64::NAN, f64::INFINITY, 0.0, -3.0] {
            let opts = SimulateOptions {
                seed: Some(1),
                match_method: None,
                horizon: Some(bad),
            };
            let err = simulate_with_options(&model, &pop, &model.default_params, 1, &opts)
                .expect_err("non-finite / non-positive horizon must error");
            assert!(err.contains("horizon"), "got: {err}");
        }
        // A valid horizon still succeeds.
        let ok = SimulateOptions {
            seed: Some(1),
            match_method: None,
            horizon: Some(10.0),
        };
        assert!(simulate_with_options(&model, &pop, &model.default_params, 1, &ok).is_ok());
    }

    /// A horizon below a left-truncated subject's `entry_time` would censor it
    /// before it entered observation (a row with `time < entry_time`); the library
    /// path rejects it. The `[simulation]`-block path always enters at 0, so this
    /// only guards `SimulateOptions { horizon }` on real left-truncated data (#522
    /// review).
    #[test]
    fn simulate_with_options_rejects_horizon_below_entry_time() {
        use ferx_core::{simulate_with_options, SimulateOptions};
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model parses");
        // One left-truncated subject: enters at t = 5 on both causes.
        let mut pop = common::tte_competing_pop(&[(10.0_f64, 0u8)]);
        pop.subjects[0].obs_records = vec![
            ObsRecord::Event {
                time: 10.0,
                event_type: EventType::RightCensored,
                entry_time: 5.0,
                cmt: 2,
            },
            ObsRecord::Event {
                time: 10.0,
                event_type: EventType::RightCensored,
                entry_time: 5.0,
                cmt: 3,
            },
        ];
        let below = SimulateOptions {
            seed: Some(1),
            match_method: None,
            horizon: Some(3.0), // < entry_time 5.0
        };
        let err = simulate_with_options(&model, &pop, &model.default_params, 1, &below)
            .expect_err("horizon below entry_time must error");
        assert!(err.contains("entry_time"), "got: {err}");
        // A horizon at/above entry is fine.
        let above = SimulateOptions {
            seed: Some(1),
            match_method: None,
            horizon: Some(8.0),
        };
        assert!(simulate_with_options(&model, &pop, &model.default_params, 1, &above).is_ok());
    }

    /// `predict_survival` must keep the partition invariant `Σ_k CIF_k + S_all = 1`
    /// even when the caller supplies an out-of-order time grid: the CIF telescopes
    /// the all-cause survival drop, so the grid is sorted internally. Guards the
    /// public-API robustness of the invariant.
    #[test]
    fn predict_survival_cif_invariant_holds_for_unsorted_grid() {
        use ferx_core::predict_survival;
        use std::collections::HashMap;
        let model = parse_model_string(COMPETING_RISKS_MODEL).expect("model must parse");
        let pop = common::tte_competing_pop(&[(5.0, 2u8), (8.0, 3u8), (30.0, 0u8)]);
        // Deliberately unsorted.
        let grid = [15.0, 0.0, 6.0, 2.0, 40.0];
        let preds = predict_survival(&model, &pop, &model.default_params, &grid);
        assert!(!preds.is_empty(), "predict_survival must return rows");
        let mut by_key: HashMap<(String, u64), (f64, f64)> = HashMap::new();
        for p in &preds {
            assert!(
                (0.0..=1.0).contains(&p.cif),
                "CIF must be in [0,1]: {}",
                p.cif
            );
            let entry = by_key
                .entry((p.id.clone(), p.time.to_bits()))
                .or_insert((0.0, p.survival_all));
            entry.0 += p.cif;
        }
        for (_, (sum_cif, s_all)) in by_key {
            assert!(
                (sum_cif + s_all - 1.0).abs() < 1e-9,
                "Σ CIF + S_all must equal 1 on an unsorted grid: {sum_cif} + {s_all}"
            );
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
