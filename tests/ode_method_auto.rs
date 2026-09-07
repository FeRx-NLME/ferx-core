//! `ode_method = auto` at the public-API boundary (#978).
//!
//! The mechanism — the Jacobian eigenvalue probe, the escalation threshold, the finite/stall
//! guard, the `f64`-vs-dual agreement — is pinned by Tier-1 tests next to the code. What can
//! only be checked out here is that the fit option survives the whole way from a model file to
//! the integrator, on both the prediction and the estimation entry points, and that switching
//! it on does not change the answer on a model where it escalates.
//!
//! That last property is the one worth stating plainly: `auto` is a **performance** feature.
//! Every method it can choose is already validated on its own, so the bar for `auto` is not
//! "produces good numbers" but "produces the *same* numbers as the method a user would have
//! named by hand". A silent change in predictions would be the regression.
//!
//! Fast (a handful of outer iterations, no convergence loop), so this is a Tier-2 test and
//! runs on every PR.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, FitOptions, Population};
use ferx_core::{fit, predict};

mod common;

/// A three-state fast-binding (TMDD-shaped) model. The fast eigenvalue is carried by
/// `KON · central`, so this system is stiff once a dose has landed and *not* stiff at its
/// declared initial condition — the shape that makes the per-segment probe worth having.
const BINDING_MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVVC(3.0, 0.1, 100.0)
  theta TVKON(60.0, 1e-3, 1e4)
  omega ETA_CL ~ 0.04
  sigma PROP_ERR ~ 0.05 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  VC = TVVC
  KON = TVKON
  KOFF = 20.0
  KINT = 0.5
  R0 = 10.0

[structural_model]
  ode(obs_cmt=central, states=[central, target, complex])

[odes]
  init(target) = R0
  d/dt(central) = -(CL/VC) * central - KON * central * target + KOFF * complex
  d/dt(target)  = -KON * central * target + KOFF * complex - 0.05 * target + 0.5
  d/dt(complex) =  KON * central * target - KOFF * complex - KINT * complex

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_method = METHOD
"#;

/// The same model with `ode_method` substituted, parsed through the full file path so the
/// `[fit_options]` value has to travel the route a user's model takes.
fn model_with_method(method: &str) -> ferx_core::CompiledModel {
    let src = BINDING_MODEL.replace("METHOD", method);
    let parsed = parse_full_model(&src).unwrap_or_else(|e| panic!("{method}: parse: {e}"));
    assert_eq!(
        parsed
            .model
            .ode_spec
            .as_ref()
            .expect("ODE model")
            .solver_opts
            .method
            .as_str(),
        method,
        "the `[fit_options] ode_method` value must reach the solver"
    );
    parsed.model
}

/// Three subjects at doses spanning two orders of magnitude. The dose matters: the fast
/// eigenvalue scales with the amount in the central compartment, so the low-dose subject is
/// the non-stiff end of the same model.
fn population(times: &[f64]) -> Population {
    let subjects = [5.0_f64, 100.0, 1000.0]
        .iter()
        .enumerate()
        .map(|(i, &amt)| {
            common::subject(
                &format!("{}", i + 1),
                vec![DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)],
                times.to_vec(),
                vec![1.0; times.len()],
                vec![1; times.len()],
            )
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

#[test]
fn auto_predicts_what_the_named_stiff_method_predicts() {
    let times = [0.02, 0.1, 0.5, 1.0, 2.0, 6.0, 12.0, 24.0];
    let pop = population(&times);

    let auto = model_with_method("auto");
    let auto_pred = predict(&auto, &pop, &auto.default_params);

    // `auto` escalates this model to `rodas4` at the default tolerance, so that is the
    // method whose numbers it must reproduce — not merely "something plausible".
    let named = model_with_method("rodas4");
    let named_pred = predict(&named, &pop, &named.default_params);

    assert_eq!(auto_pred.len(), named_pred.len());
    for (a, b) in auto_pred.iter().zip(&named_pred) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.time, b.time);
        assert!(
            a.pred.is_finite() && b.pred.is_finite(),
            "non-finite prediction at {}/{}: {} vs {}",
            a.id,
            a.time,
            a.pred,
            b.pred
        );
        assert!(
            (a.pred - b.pred).abs() <= 1e-9 * b.pred.abs().max(1e-6),
            "auto diverged from rodas4 at {}/{}: {} vs {}",
            a.id,
            a.time,
            a.pred,
            b.pred
        );
    }

    // Non-vacuity: the explicit default must give *slightly* different numbers, or the check
    // above would pass even if `auto` had quietly stayed on `rk45`.
    let explicit = model_with_method("rk45");
    let explicit_pred = predict(&explicit, &pop, &explicit.default_params);
    let moved = auto_pred
        .iter()
        .zip(&explicit_pred)
        .any(|(a, e)| a.pred != e.pred);
    assert!(
        moved,
        "auto reproduced rk45 bit-for-bit — the escalation never happened, so this test \
         pins nothing about `auto`"
    );
}

#[test]
fn auto_reaches_the_estimation_path_and_returns_a_finite_objective() {
    // A handful of outer iterations only — this is a plumbing check, not a convergence test.
    // The estimation path is worth covering separately from `predict()` because it also runs
    // the analytic-sensitivity solve, which resolves `auto` on its own dual-typed integration.
    let times = [0.1, 0.5, 1.0, 4.0, 12.0, 24.0];
    let pop = population(&times);
    let model = model_with_method("auto");
    let options = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..Default::default()
    };
    let result = fit(&model, &pop, &model.default_params, &options).expect("fit under auto");
    assert!(
        result.ofv.is_finite(),
        "auto produced a non-finite OFV: {}",
        result.ofv
    );
}

#[test]
fn an_unknown_ode_method_is_rejected_with_auto_in_the_expected_values() {
    let src = BINDING_MODEL.replace("METHOD", "lsoda");
    let err = match parse_full_model(&src) {
        Ok(_) => panic!("unknown method must not parse"),
        Err(e) => e,
    };
    assert!(err.contains("ode_method"), "{err}");
    assert!(
        err.contains("auto"),
        "the diagnostic must list `auto` among the accepted values: {err}"
    );
}

/// Tier-3: the same equivalence at **convergence** rather than at a fixed prediction.
///
/// `predict()` compares one trajectory; a fit compares the whole objective surface the
/// optimizer walks, including the analytic gradient. If `auto` resolved differently between
/// the prediction and the sensitivity solve — the one coupling nothing outside the driver
/// enforces — the two would still agree pointwise and disagree on where the optimizer lands,
/// which is exactly what this pins and the fast tests cannot.
///
/// Run at tight tolerances, and that is load-bearing rather than cosmetic. `auto` escalates
/// **per segment**, so on this population it runs a stiff method on the high-dose subjects and
/// the explicit one on the low-dose subject — it is not a relabelling of any single method, and
/// the two therefore agree only as well as both agree with the exact solution. At the default
/// `ode_reltol = 1e-4` that leaves ~0.7 OFV units between them, which says nothing about
/// `auto`; at `1e-10` the solver's own error stops being the dominant term, so a disagreement
/// means a real one. It also puts the fixture in the `ode_reltol <= 1e-8` regime where the
/// probe escalates to `rodas5p` rather than `rodas4`, so the tolerance-dependent choice is on
/// the tested path too.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn auto_converges_to_the_same_optimum_as_the_named_stiff_method() {
    // The fixture at NONMEM-equivalent accuracy, so the two fits differ by their stepper
    // choice and not by their tolerance.
    fn tight(method: &str) -> ferx_core::CompiledModel {
        let src = BINDING_MODEL.replace("METHOD", method)
            + "  ode_reltol = 1e-10\n  ode_abstol = 1e-12\n";
        parse_full_model(&src)
            .unwrap_or_else(|e| panic!("{method}: parse: {e}"))
            .model
    }

    let times = [0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let pop = population(&times);
    let options = FitOptions {
        run_covariance_step: false,
        ..Default::default()
    };

    let auto = tight("auto");
    let auto_fit = fit(&auto, &pop, &auto.default_params, &options).expect("fit under auto");
    let named = tight("rodas5p");
    let named_fit = fit(&named, &pop, &named.default_params, &options).expect("fit under rodas5p");

    assert!(auto_fit.converged, "auto did not converge");
    assert!(
        (auto_fit.ofv - named_fit.ofv).abs() <= 1e-4 * named_fit.ofv.abs().max(1.0),
        "auto converged to a different optimum: OFV {} vs {}",
        auto_fit.ofv,
        named_fit.ofv
    );
    for (i, (a, b)) in auto_fit.theta.iter().zip(&named_fit.theta).enumerate() {
        assert!(
            (a - b).abs() <= 1e-3 * b.abs().max(1e-6),
            "theta[{i}] differs: {a} vs {b}"
        );
    }
}
