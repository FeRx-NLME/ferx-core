//! Tier-2 smoke tests for the Phase 5 CTMM (continuous-time Markov) endpoint
//! (Track D, #759).
//!
//! These exercise the public parse / `fit()` boundary. They are NOT gated with
//! `slow-tests` — every fit runs `outer_maxiter = 3` and returns immediately. A
//! full-convergence anchor (vs. R `msm`, per the plan §14.9) is a Tier-3
//! follow-up; the numerical CTMM kernel is already anchored to closed form in
//! `src/markov`.
//!
//! The whole module is behind `#[cfg(feature = "markov")]` so the file compiles
//! (contributing no tests) on a PR that does not enable the feature.

mod common;

#[cfg(feature = "markov")]
mod ctmm_smoke {
    use crate::common;
    use ferx_core::api::check_model_options;
    use ferx_core::diagnostics::Diagnostic;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::types::EstimationMethod;
    use ferx_core::{fit, fit_from_files, EndpointLikelihood, FitOptions};

    /// Fixed-effects (n_eta = 0) two-state CTMM — the R `msm` analogue.
    const FIXED_MODEL: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01)
  transition asleep -> awake  = exp(LQ10)
";

    /// Mixed-effects: a per-subject random effect on the awake→asleep intensity.
    const MIXED_MODEL: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  omega ETA_Q ~ 0.1

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + ETA_Q)
  transition asleep -> awake  = exp(LQ10)
";

    fn smoke_opts() -> FitOptions {
        FitOptions {
            outer_maxiter: 3,
            ..Default::default()
        }
    }

    fn has_error(diags: &[Diagnostic], code: &str) -> bool {
        diags.iter().any(|d| d.code == code)
    }

    /// The parser recognises `[markov_model]` and registers a `Ctmm` endpoint carrying the
    /// declared DV-code table, on its CMT.
    #[test]
    fn ctmm_model_parses() {
        let model = parse_model_string(FIXED_MODEL).expect("FIXED_MODEL must parse");
        match model.endpoints.get(&5) {
            Some(EndpointLikelihood::Ctmm {
                n_states,
                state_codes,
                ..
            }) => {
                assert_eq!(*n_states, 2);
                assert_eq!(state_codes, &vec![0, 1]);
            }
            other => panic!("expected a Ctmm endpoint for CMT=5, got {other:?}"),
        }
        assert_eq!(model.n_theta, 2, "n_theta should be LQ01 + LQ10");
        assert_eq!(model.n_eta, 0, "fixed-effects model has no η");
    }

    /// Fixed-effects CTMM read from CSV through the real datareader (so `DiscreteState`
    /// routing on CMT 5 is exercised) and fit end-to-end to a finite OFV.
    #[test]
    fn ctmm_fixed_effects_reads_and_fits() {
        let res = fit_from_files(
            "examples/ctmm_2state.ferx",
            Some("data/ctmm_2state.csv"),
            None,
            Some(smoke_opts()),
        )
        .expect("fixed-effects CTMM fit_from_files must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// A mixed-effects fit runs a few FOCEI FD-Laplace iterations and returns a finite OFV —
    /// exercises the generator's η dependence through the FD-Hessian interaction site.
    #[test]
    fn ctmm_mixed_effects_runs() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        // 6 subjects × 3 state observations each (the X covariate is unused by the CTMM).
        let subjects: Vec<(f64, Vec<(f64, u8)>)> = vec![
            (0.0, vec![(0.0, 0), (1.0, 0), (2.0, 1)]),
            (0.0, vec![(0.0, 1), (1.0, 0), (2.0, 0)]),
            (0.0, vec![(0.0, 0), (1.0, 1), (2.0, 1)]),
            (0.0, vec![(0.0, 1), (1.0, 1), (2.0, 0)]),
            (0.0, vec![(0.0, 0), (1.0, 0), (2.0, 0)]),
            (0.0, vec![(0.0, 1), (1.0, 0), (2.0, 1)]),
        ];
        let pop = common::binary_pop(&subjects, 5);
        let res = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect("mixed-effects CTMM fit must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// SAEM runs a short CTMM fit and returns a finite OFV — exercises the SAEM dispatch
    /// site the FOCEI tests do not reach.
    #[test]
    fn ctmm_saem_runs() {
        let model = parse_model_string(MIXED_MODEL).unwrap();
        let subjects: Vec<(f64, Vec<(f64, u8)>)> = vec![
            (0.0, vec![(0.0, 0), (1.0, 0), (2.0, 1)]),
            (0.0, vec![(0.0, 1), (1.0, 0), (2.0, 0)]),
            (0.0, vec![(0.0, 0), (1.0, 1), (2.0, 1)]),
            (0.0, vec![(0.0, 1), (1.0, 1), (2.0, 0)]),
        ];
        let pop = common::binary_pop(&subjects, 5);
        let opts = FitOptions {
            method: EstimationMethod::Saem,
            saem_n_exploration: 2,
            saem_n_convergence: 2,
            ..Default::default()
        };
        let res =
            fit(&model, &pop, &model.default_params, &opts).expect("SAEM CTMM fit must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// An observed DV code that is not one of the declared `states` is rejected fail-loud at
    /// fit setup, never silently folded into the transition sum.
    #[test]
    fn ctmm_rejects_undeclared_state() {
        let model = parse_model_string(FIXED_MODEL).unwrap();
        // DV = 2 is not a declared state (states are coded 0/1).
        let pop = common::binary_pop(&[(0.0, vec![(0.0, 0), (1.0, 2)])], 5);
        let err = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect_err("DV = 2 on a CTMM CMT must be rejected");
        assert!(
            err.contains("cmt = 5"),
            "message should name the CMT, got: {err}"
        );
        assert!(
            err.contains("DV 2"),
            "message should name the code, got: {err}"
        );
    }

    /// The Gauss-Newton gradient is Gaussian-specific, so `method = gn` on a CTMM model is
    /// rejected fail-loud (GN silently drops the transition likelihood).
    #[test]
    fn ctmm_gauss_newton_rejected() {
        let model = parse_model_string(FIXED_MODEL).unwrap();
        let opts = FitOptions {
            method: EstimationMethod::FoceGn,
            ..Default::default()
        };
        assert!(
            has_error(&check_model_options(&model, &opts), "E_CTMM_GN_UNSUPPORTED"),
            "method = gn + CTMM must be rejected"
        );
    }

    /// CTMM + inter-occasion variability is rejected: the generator is built with BSV-only η.
    #[test]
    fn ctmm_iov_rejected() {
        const IOV_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
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

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01)
  transition asleep -> awake  = exp(LQ10)

[fit_options]
  method     = focei
  iov_column = OCC
";
        let model = parse_model_string(IOV_MODEL).expect("joint PK + CTMM + IOV model parses");
        assert!(model.n_kappa > 0, "fixture must declare a κ");
        assert!(
            has_error(
                &check_model_options(&model, &FitOptions::default()),
                "E_CTMM_IOV_UNSUPPORTED"
            ),
            "CTMM + IOV must be rejected"
        );
    }

    /// The `[markov_model]` parser rejects its error cases fail-loud rather than mis-parsing.
    #[test]
    fn ctmm_parser_rejections() {
        let wrap = |block: &str| {
            format!(
                "[parameters]\n  theta LQ01(-0.7, -6.0, 3.0)\n  theta LQ10(-1.2, -6.0, 3.0)\n{block}\n"
            )
        };
        let full = |body: &str| {
            format!("[markov_model]\n  type = ctmm\n  cmt = 5\n  states = [a=0, b=1]\n{body}")
        };

        // A well-formed baseline parses (guards against a broken `wrap`/`full`).
        assert!(parse_model_string(&wrap(&full(
            "  transition a -> b = exp(LQ01)\n  transition b -> a = exp(LQ10)\n"
        )))
        .is_ok());

        // mCTMM / DTMM not yet supported.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = mctmm\n  cmt = 5\n  states = [a=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // `init` not yet supported.
        assert!(parse_model_string(&wrap(&format!(
            "{}  init = [1, 0]\n",
            full("  transition a -> b = exp(LQ01)\n")
        )))
        .is_err());
        // Option-B matrix spelling not yet supported.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  n_states = 2\n  q12 = exp(LQ01)\n"
        ))
        .is_err());
        // Missing `states`.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Missing `cmt`.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  states = [a=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // No transitions.
        assert!(parse_model_string(&wrap(&format!("{}\n", full("")))).is_err());
        // Transition references an unknown state.
        assert!(parse_model_string(&wrap(&full("  transition a -> c = exp(LQ01)\n"))).is_err());
        // Self-transition (a rate is off-diagonal).
        assert!(parse_model_string(&wrap(&full("  transition a -> a = exp(LQ01)\n"))).is_err());
        // Duplicate state code.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  states = [a=0, b=0]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Duplicate transition.
        assert!(parse_model_string(&wrap(&full(
            "  transition a -> b = exp(LQ01)\n  transition a -> b = exp(LQ10)\n"
        )))
        .is_err());
        // Unknown key.
        assert!(parse_model_string(&wrap(&format!(
            "{}  bogus = 1\n",
            full("  transition a -> b = exp(LQ01)\n")
        )))
        .is_err());
        // Unknown `type`.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = xyz\n  cmt = 5\n  states = [a=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Non-integer cmt.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = xx\n  states = [a=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // cmt = 0 (compartment indices are 1-based).
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 0\n  states = [a=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Empty state label (`=0`): an unreferenceable state.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  states = [=0, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // states not bracketed.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  states = a=0, b=1\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Non-integer state code.
        assert!(parse_model_string(&wrap(
            "[markov_model]\n  type = ctmm\n  cmt = 5\n  states = [a=x, b=1]\n  transition a -> b = exp(LQ01)\n"
        ))
        .is_err());
        // Malformed transition: no `->`.
        assert!(parse_model_string(&wrap(&full("  transition a b = exp(LQ01)\n"))).is_err());
        // Malformed transition: no `=` intensity.
        assert!(parse_model_string(&wrap(&full("  transition a -> b\n"))).is_err());
        // Empty state name in a transition.
        assert!(parse_model_string(&wrap(&full("  transition  -> b = exp(LQ01)\n"))).is_err());
        // A bare `key = value` line with an unknown key that is not a transition.
        assert!(parse_model_string(&wrap(&format!(
            "{}  transitionX = 1\n",
            full("  transition a -> b = exp(LQ01)\n")
        )))
        .is_err());
    }

    /// A transition intensity that references `TIME` is rejected at parse: the CTMM is
    /// time-homogeneous (the generator is built outside any model-time scope), so `TIME`
    /// would silently resolve to 0 and drop the time term.
    #[test]
    fn ctmm_intensity_references_time_rejected() {
        // Direct TIME in an intensity.
        const DIRECT: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta BT(0.1, -5.0, 5.0)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [a=0, b=1]
  transition a -> b = exp(LQ01 + BT * TIME)
  transition b -> a = exp(LQ10)
";
        let err = parse_model_string(DIRECT)
            .expect_err("a direct TIME reference in a CTMM intensity must be rejected");
        assert!(err.contains("TIME"), "message should name TIME, got: {err}");

        // TIME reached transitively through an [individual_parameters] value.
        const VIA_INDIV: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta BT(0.1, -5.0, 5.0)

[individual_parameters]
  LQ = LQ01 + BT * TIME

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [a=0, b=1]
  transition a -> b = exp(LQ)
  transition b -> a = exp(LQ10)
";
        assert!(
            parse_model_string(VIA_INDIV).is_err(),
            "TIME reached through an [individual_parameters] value must be rejected"
        );
    }

    /// A CTMM endpoint has no simulation path yet: `simulate()` fails loud (panics at the
    /// single simulate chokepoint) rather than emit meaningless all-zero discrete rows.
    #[test]
    #[should_panic(expected = "CTMM")]
    fn ctmm_simulate_panics() {
        use ferx_core::simulate;
        let model = parse_model_string(FIXED_MODEL).unwrap();
        let pop = common::binary_pop(&[(0.0, vec![(0.0, 0), (1.0, 1)])], 5);
        let _ = simulate(&model, &pop, &model.default_params, 1);
    }

    /// Out-of-order CTMM observation times are rejected at fit setup (the datareader sorts
    /// doses, not observations), rather than silently collapsing the subject to the 1e20
    /// sentinel and biasing the population OFV.
    #[test]
    fn ctmm_out_of_order_times_rejected() {
        let model = parse_model_string(FIXED_MODEL).unwrap();
        // Times 2.0 then 1.0 on CMT 5 — decreasing.
        let pop = common::binary_pop(&[(0.0, vec![(2.0, 0), (1.0, 1)])], 5);
        let err = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect_err("out-of-order CTMM times must be rejected");
        assert!(
            err.contains("cmt = 5") && err.contains("non-decreasing"),
            "message should name the CMT and the constraint, got: {err}"
        );
    }

    /// A transition intensity that references an IOV κ directly is rejected at parse (the
    /// generator is built with BSV-only η).
    #[test]
    fn ctmm_intensity_direct_kappa_rejected() {
        const M: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
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

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [a=0, b=1]
  transition a -> b = exp(LQ01 + KAPPA_CL)
  transition b -> a = exp(LQ10)

[fit_options]
  iov_column = OCC
";
        assert!(
            parse_model_string(M).is_err(),
            "a direct κ reference in a CTMM intensity must be rejected at parse"
        );
    }

    /// An intensity that references an `[individual_parameters]` value which itself reads a
    /// covariate — exercises the transitive needed-indiv-statements restriction and the
    /// covariate-union-through-`[individual_parameters]` path (the #741 completeness guard).
    #[test]
    fn ctmm_intensity_references_individual_parameter() {
        const M: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta BAGE(0.1, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.05

[covariates]
  AGE continuous

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  LQ  = -0.7 + BAGE * AGE

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [a=0, b=1]
  transition a -> b = exp(LQ)
  transition b -> a = exp(LQ10)
";
        let model = parse_model_string(M).expect("intensity referencing an indiv param parses");
        // AGE is used only by LQ (the value the intensity references), so it can only appear
        // in the model's covariate set via the covariate-union-through-[individual_parameters].
        assert!(
            model.referenced_covariates.iter().any(|c| c == "AGE"),
            "AGE (reached via the indiv-param the intensity references) must be collected: {:?}",
            model.referenced_covariates
        );
    }

    /// A one-compartment-oral PK model with a **drug-driven** CTMM intensity that references
    /// the model state `central` (Phase 6, #817). Concentration is written `central / V`,
    /// matching the joint PK-TTE hazard convention.
    const DRUG_DRIVEN_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.1, -5.0, 5.0)
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + SLOPE * (central / V))
  transition asleep -> awake  = exp(LQ10)
";

    /// Slice 2 (#817): an intensity referencing an ODE state parses as a CTMM endpoint that
    /// records the state in `generator_states` (marking it time-inhomogeneous), and the
    /// generator threads the passed state value into the intensity (Q varies with the state).
    #[test]
    fn ctmm_drug_driven_intensity_threads_state() {
        let model = parse_model_string(DRUG_DRIVEN_MODEL).expect("drug-driven CTMM must parse");
        let ep = model.endpoints.get(&5).expect("CTMM endpoint on CMT 5");
        let (generator_fn, generator_states) = match ep {
            EndpointLikelihood::Ctmm {
                generator_fn,
                generator_states,
                ..
            } => (generator_fn, generator_states),
            other => panic!("expected a Ctmm endpoint, got {other:?}"),
        };
        // `central` is ODE state index 1 (states = [depot, central]); it is the only state
        // referenced, and `V` is an individual parameter (not a state), so it is excluded.
        assert_eq!(
            generator_states,
            &vec![("central".to_string(), 1)],
            "the intensity references the `central` state at ODE index 1"
        );

        // Evaluate the generator at two central amounts: Q[awake→asleep] = exp(LQ01 +
        // SLOPE·central/V) must move with the state (V = TVV = 10, LQ01 = -0.7, SLOPE = 0.1).
        let theta = &model.default_params.theta;
        let q_low = generator_fn(theta, &[], &std::collections::HashMap::new(), &[0.0, 0.0]);
        let q_high = generator_fn(theta, &[], &std::collections::HashMap::new(), &[0.0, 20.0]);
        // central = 0 ⇒ conc 0 ⇒ rate exp(-0.7); central = 20 ⇒ conc 2 ⇒ rate exp(-0.5).
        assert!(
            (q_low[(0, 1)] - (-0.7_f64).exp()).abs() < 1e-9,
            "q_low {q_low}"
        );
        assert!(
            (q_high[(0, 1)] - (-0.5_f64).exp()).abs() < 1e-9,
            "q_high {q_high}"
        );
        // The state-independent transition (asleep→awake = exp(LQ10)) is unchanged.
        assert!((q_low[(1, 0)] - q_high[(1, 0)]).abs() < 1e-12);
    }

    /// Degenerate anchor for the Slice-3 integration (#817): with `SLOPE = 0` the generator
    /// is state-**independent** (constant `Q`), so the inhomogeneous occupancy-ODE path must
    /// reduce to the time-homogeneous `expm(Q·Δt)` result. The drug-driven model at `SLOPE = 0`
    /// and the plain homogeneous model share `LQ01 = -0.7`, `LQ10 = -1.2`, so their per-subject
    /// CTMM NLLs must agree — this validates the whole occupancy-integration wiring against the
    /// exact closed form, independently of any dataset.
    #[test]
    fn ctmm_drug_driven_reduces_to_homogeneous_when_flat() {
        use ferx_core::markov::endpoint::ctmm_subject_nll;
        use ferx_core::types::ObsRecord;

        let mut subj = common::subject("1", vec![], vec![], vec![], vec![]);
        subj.obs_records = [(0.0, 0), (0.7, 1), (1.9, 0), (3.1, 1)]
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                state,
                cmt: 5,
            })
            .collect();

        // Drug-driven model, but with SLOPE forced to 0 (index 5: [TVCL,TVV,TVKA,LQ01,LQ10,SLOPE]).
        let drug = parse_model_string(DRUG_DRIVEN_MODEL).unwrap();
        let mut theta_flat = drug.default_params.theta.clone();
        theta_flat[5] = 0.0;
        let nll_inhom = ctmm_subject_nll(&drug, &subj, &theta_flat, &[]);

        // Homogeneous model with the same intensities exp(LQ01), exp(LQ10).
        let homog = parse_model_string(FIXED_MODEL).unwrap();
        let nll_homog = ctmm_subject_nll(&homog, &subj, &homog.default_params.theta, &[]);

        assert!(nll_inhom.is_finite() && nll_homog.is_finite());
        // The inhomogeneous path integrates the occupancy ODE at the model's ODE tolerance
        // (reltol 1e-4 here) while the homogeneous path uses the exact `expm`, so they agree
        // to that tolerance — far tighter than any wiring bug (wrong state indexing / matrix
        // convention would differ by O(1), not ~1e-4).
        assert!(
            (nll_inhom - nll_homog).abs() < 5e-3,
            "SLOPE=0 inhomogeneous NLL {nll_inhom} must match homogeneous {nll_homog}"
        );
    }

    /// The evolving model state genuinely drives the likelihood: with a dose raising the
    /// central concentration and `SLOPE ≠ 0`, the per-subject CTMM NLL differs from the flat
    /// (`SLOPE = 0`) case — the occupancy integration reads the concentration trajectory.
    #[test]
    fn ctmm_drug_driven_state_changes_likelihood() {
        use ferx_core::markov::endpoint::ctmm_subject_nll;
        use ferx_core::types::{DoseEvent, ObsRecord};

        // Bolus of 100 into the depot (cmt 1) at t=0 → central rises then falls.
        let mut subj = common::subject(
            "1",
            vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            vec![],
            vec![],
            vec![],
        );
        subj.obs_records = [(0.5, 0), (1.0, 1), (2.0, 1), (4.0, 0)]
            .iter()
            .map(|&(time, state)| ObsRecord::DiscreteState {
                time,
                state,
                cmt: 5,
            })
            .collect();

        let drug = parse_model_string(DRUG_DRIVEN_MODEL).unwrap();
        let mut theta_flat = drug.default_params.theta.clone();
        theta_flat[5] = 0.0; // SLOPE = 0
        let mut theta_drug = drug.default_params.theta.clone();
        theta_drug[5] = 0.8; // SLOPE = 0.8 (strong concentration effect)

        let nll_flat = ctmm_subject_nll(&drug, &subj, &theta_flat, &[]);
        let nll_drug = ctmm_subject_nll(&drug, &subj, &theta_drug, &[]);

        assert!(nll_flat.is_finite() && nll_drug.is_finite());
        assert!(
            (nll_flat - nll_drug).abs() > 1e-3,
            "a concentration-driven intensity must change the NLL: flat {nll_flat}, drug {nll_drug}"
        );
    }

    /// End-to-end: a **mixed-effects** drug-driven CTMM (a random effect on `CL` flows through
    /// the concentration into `Q`) fits a few FOCEI iterations and returns a finite OFV —
    /// exercising the full inhomogeneous dispatch, including the FD-Hessian interaction site at
    /// perturbed η (which re-solves the ODE and re-integrates the occupancy per perturbation).
    #[test]
    fn ctmm_drug_driven_mixed_effects_fit_runs() {
        use ferx_core::types::{DoseEvent, ObsRecord, Population};

        const M: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.3, -5.0, 5.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01 + SLOPE * (central / V))
  transition asleep -> awake  = exp(LQ10)
";
        let model = parse_model_string(M).expect("mixed-effects drug-driven CTMM must parse");

        // Four subjects: a bolus into depot at t=0 and a handful of state observations on CMT 5.
        let obs_sets = [
            [(0.5, 0), (1.5, 1), (3.0, 1), (5.0, 0)],
            [(0.5, 1), (1.5, 0), (3.0, 0), (5.0, 1)],
            [(0.5, 0), (1.5, 0), (3.0, 1), (5.0, 1)],
            [(0.5, 1), (1.5, 1), (3.0, 0), (5.0, 0)],
        ];
        let subjects = obs_sets
            .iter()
            .enumerate()
            .map(|(i, obs)| {
                let mut s = common::subject(
                    &format!("{}", i + 1),
                    vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                    vec![],
                    vec![],
                    vec![],
                );
                s.obs_records = obs
                    .iter()
                    .map(|&(time, state)| ObsRecord::DiscreteState {
                        time,
                        state,
                        cmt: 5,
                    })
                    .collect();
                s
            })
            .collect();
        let pop = Population {
            covariate_names: vec![],
            dv_column: "DV".to_string(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
            subjects,
        };

        let res = fit(&model, &pop, &model.default_params, &smoke_opts())
            .expect("mixed-effects drug-driven CTMM fit must return Ok");
        assert!(res.ofv.is_finite(), "OFV must be finite, got {}", res.ofv);
    }

    /// A model state reached only *transitively* through an `[individual_parameters]` value is
    /// rejected at parse (that value is evaluated with no state channel, so the state would
    /// silently resolve to 0) — the user must reference the state directly in the transition.
    #[test]
    fn ctmm_state_via_individual_parameter_rejected() {
        const M: &str = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)
  theta SLOPE(0.1, -5.0, 5.0)
  sigma PROP ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL
  V  = TVV
  KA = TVKA
  LQ = LQ01 + SLOPE * central

[structural_model]
  ode(obs_cmt=central, states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[error_model]
  DV ~ proportional(PROP)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [a=0, b=1]
  transition a -> b = exp(LQ)
  transition b -> a = exp(LQ10)
";
        let err = parse_model_string(M).expect_err(
            "a state reached through an [individual_parameters] value must be rejected",
        );
        assert!(
            err.contains("central") && err.contains("directly"),
            "message should name the state and point at the direct-reference form, got: {err}"
        );
    }

    /// `type` is optional and defaults to `ctmm`: a block omitting the `type` line parses
    /// as a CTMM endpoint (the explicit default — locks the behaviour so a future
    /// mCTMM/DTMM addition cannot silently change it).
    #[test]
    fn ctmm_type_defaults_to_ctmm_when_omitted() {
        const M: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)

[markov_model]
  cmt    = 5
  states = [awake=0, asleep=1]
  transition awake  -> asleep = exp(LQ01)
  transition asleep -> awake  = exp(LQ10)
";
        let model =
            parse_model_string(M).expect("a [markov_model] with no `type` defaults to ctmm");
        assert!(
            matches!(
                model.endpoints.get(&5),
                Some(EndpointLikelihood::Ctmm { .. })
            ),
            "omitting `type` must yield a Ctmm endpoint"
        );
    }

    /// Bare numeric state codes (no labels) parse and map to generator indices in order.
    #[test]
    fn ctmm_bare_numeric_states_parse() {
        const M: &str = r"
[parameters]
  theta LQ01(-0.7, -6.0, 3.0)
  theta LQ10(-1.2, -6.0, 3.0)

[markov_model]
  type   = ctmm
  cmt    = 5
  states = [0, 1]
  transition 0 -> 1 = exp(LQ01)
  transition 1 -> 0 = exp(LQ10)
";
        let model = parse_model_string(M).expect("bare-numeric-state CTMM parses");
        match model.endpoints.get(&5) {
            Some(EndpointLikelihood::Ctmm { state_codes, .. }) => {
                assert_eq!(state_codes, &vec![0, 1]);
            }
            other => panic!("expected a Ctmm endpoint, got {other:?}"),
        }
    }
}
