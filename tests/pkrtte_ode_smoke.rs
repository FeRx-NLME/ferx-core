//! #3152: `type = rtte, clock = forward` with an ODE-accumulated `hazard = ...` against its
//! analytic `family` twin, on the first 20 subjects of the bundled RTTE data. The ODE side
//! integrates the hazard as an appended `[odes]` state and scores each gap as `H(tₖ) − H(tₖ₋₁)`;
//! the analytic side scores the closed form. Equal objectives at the start (`maxiter = 0`) pin
//! the value; equal optima pin the gradient, which runs through the same scorer over dual
//! numbers. Exponential is a constant hazard; Weibull (shape 1.5) varies in time through `TIME`.
//! `clock = reset` with an ODE hazard is still refused, and `simulate()` refuses the model.
//!
//! Tier 2 (the converged pairs: Tier 3).

#[cfg(feature = "survival")]
mod pkrtte_ode {
    use ferx_core::api::read_population_for;
    use ferx_core::parser::model_parser::parse_model_string;
    use ferx_core::{fit, FitOptions, FitResult};

    /// The analytic event model, or its `[odes]` transcription when `ode` is set: one inert
    /// state, and the `[error_model]` an ODE model needs, with no Gaussian row to score.
    fn model(params: &str, indiv: &str, family: &str, hazard: &str, ode: bool) -> String {
        let head = "[event_model]\n  cmt = 2\n  type = rtte\n  clock = forward\n";
        if !ode {
            return format!("{params}{head}{family}");
        }
        let params = params.replace("[parameters]\n", "[parameters]\n  sigma SIG ~ 0.1 FIX\n");
        format!(
            "{params}[individual_parameters]\n{indiv}[structural_model]\n  \
             ode(obs_cmt=x, states=[x])\n[odes]\n  d/dt(x) = 0\n[error_model]\n  \
             DV ~ additive(SIG)\n{head}  hazard = {hazard}\n"
        )
    }

    fn run(src: &str, data: &str, maxiter: usize) -> FitResult {
        let m = parse_model_string(src).unwrap_or_else(|e| panic!("parses: {e}\n{src}"));
        let path = format!("{}/data/{data}", env!("CARGO_MANIFEST_DIR"));
        let (mut pop, _) = read_population_for(&m, &None, &path, None, None, None, &[]).unwrap();
        pop.subjects.truncate(20);
        // The Weibull start gap falls 7e-3 → 1.7e-4 → 3.1e-6 → 4.6e-8 at `ode_reltol` 1e-4 → 1e-6
        // → 1e-8 → 1e-10 (200 subjects, Linux x86_64): the ODE twin's integration error, not
        // the scoring. 1e-8 keeps it under the bound at a fifth of 1e-10's cost.
        let o = FitOptions {
            outer_maxiter: maxiter,
            run_covariance_step: false,
            ode_reltol: 1e-8,
            ode_abstol: 1e-10,
            ..FitOptions::default()
        };
        fit(&m, &pop, &m.default_params, &o).expect("fit runs")
    }

    /// The start per PR; the converged pair (a fit loop) under `slow-tests`. The exact gradient
    /// is pinned per PR by `survival::ode_hazard_exact_tests`.
    const BUDGETS: &[usize] = if cfg!(feature = "slow-tests") { &[0, 200] } else { &[0] };

    /// `|ΔOFV| < tol` and `|Δθ| < tol·|θ|` at each budget in `budgets`.
    fn assert_twins(p: [&str; 4], data: &str, budgets: &[usize], tol: f64) {
        let [params, indiv, family, hazard] = p;
        for &maxiter in budgets {
            let a = run(&model(params, indiv, family, hazard, false), data, maxiter);
            let o = run(&model(params, indiv, family, hazard, true), data, maxiter);
            assert!(a.theta.iter().chain(&o.theta).all(|v| v.is_finite()), "{data}: θ finite");
            let dth =
                a.theta.iter().zip(&o.theta).map(|(x, y)| ((x - y) / x).abs()).fold(0.0, f64::max); // nan-ok: θ asserted finite above; every θ bound is > 0
            eprintln!(
                "{data} maxiter {maxiter}: |ΔOFV| {:.2e}, worst rel θ {dth:.2e}",
                (a.ofv - o.ofv).abs()
            );
            assert_eq!(a.converged, o.converged, "{data} maxiter {maxiter}: converged");
            assert!(
                (a.ofv - o.ofv).abs() < tol,
                "{data} maxiter {maxiter}: {} vs {}",
                a.ofv,
                o.ofv
            );
            assert!(dth < tol, "{data} maxiter {maxiter}: θ {:?} vs {:?}", a.theta, o.theta);
        }
    }

    #[test]
    fn exponential_ode_hazard_matches_its_family() {
        let p = [
            "[parameters]\n  theta TVLAMBDA(0.15, 0.001, 10.0)\n  omega ETA_LAMBDA ~ 0.09\n",
            "  LAMBDA = TVLAMBDA * exp(ETA_LAMBDA)\n",
            "  family = exponential\n  scale = TVLAMBDA * exp(ETA_LAMBDA)\n",
            "LAMBDA",
        ];
        assert_twins(p, "rtte_exponential.csv", BUDGETS, BOUND_EXP);
    }

    #[test]
    fn weibull_ode_hazard_matches_its_family() {
        let p = [
            "[parameters]\n  theta TVSCALE(5.0, 0.1, 100.0)\n  theta TVSHAPE(1.5, 0.1, 10.0)\n  \
             omega ETA_SCALE ~ 0.09\n",
            "  SCALE = TVSCALE * exp(ETA_SCALE)\n  SHAPE = TVSHAPE\n",
            "  family = weibull\n  scale = TVSCALE * exp(ETA_SCALE)\n  shape = TVSHAPE\n",
            "(SHAPE / SCALE) * (TIME / SCALE)^(SHAPE - 1)",
        ];
        assert_twins(p, "rtte_weibull_reset.csv", BUDGETS, BOUND_WEIBULL);
    }

    // Measured (Linux x86_64, `ci-test`), worst of |ΔOFV| and relative |Δθ|: exponential 1.48e-8
    // (θ at the optimum; |ΔOFV| 0 and 1.1e-13), Weibull 8.22e-7 (|ΔOFV| at the optimum; 6.30e-7 at
    // the start). About 13× and 12×.
    const BOUND_EXP: f64 = 2e-7;
    const BOUND_WEIBULL: f64 = 1e-5;

    /// Until #3152 the parser kept this model away from `simulate()`, whose stream draws
    /// analytic hazards only; it panicked on one ("unreachable"). It is refused instead.
    #[test]
    fn simulating_it_is_refused_not_a_panic() {
        let p = "[parameters]\n  theta TVH(0.1, 0.001, 10.0)\n";
        let m = parse_model_string(&model(p, "  H = TVH\n", "", "H", true)).unwrap();
        let path = format!("{}/data/rtte_exponential.csv", env!("CARGO_MANIFEST_DIR"));
        let (pop, _) = read_population_for(&m, &None, &path, None, None, None, &[]).unwrap();
        let o = ferx_core::SimulateOptions { horizon: Some(20.0), ..Default::default() };
        let err = ferx_core::simulate_with_options(&m, &pop, &m.default_params, 1, &o).unwrap_err();
        assert_eq!(err.code(), Some("E_TTE_NOT_SIMULATABLE"), "{err}");
    }

    #[test]
    fn clock_reset_with_an_ode_hazard_is_still_refused() {
        let params = "[parameters]\n  theta TVH(0.1, 0.001, 10.0)\n";
        let src =
            model(params, "  H = TVH\n", "", "H", true).replace("clock = forward", "clock = reset");
        let err = parse_model_string(&src).err().expect("refused");
        assert!(err.contains("clock = reset") && err.contains("clock = forward"), "{err}");
    }
}
