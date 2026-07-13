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
