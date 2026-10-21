//! End-to-end check that an ODE model wider than `MAX_ODE_AXES` (θ + η = 24)
//! keeps the analytic inner gradient (#1804).
//!
//! The inner η-gradient is seeded on η alone, so its width is `n_eta`, not θ + η.
//! Before #1804 the outer cap was applied to the inner as well, and one FIX θ
//! too many sent every subject's EBE search to finite differences. On this
//! fixture that left one subject at a different EBE and moved the objective by
//! 16.02 (−1017.312 vs −1033.333 at 24 axes).
//!
//! The fixture is `examples/two_cpt_oral_ode.ferx` (5 θ, 5 η) on
//! `data/two_cpt_oral_cov.csv`, with `n_fix` FIX θ at 0 multiplying the
//! individual parameters by covariate factors that are exactly 1 at θ = 0. So
//! the likelihood is unchanged: `n_fix = 14` gives 24 axes (at the cap) and
//! `n_fix = 15` gives 25 (past it). The two fits are evaluated at the start
//! values (`outer_maxiter = 0`) and must agree subject by subject.
//!
//! The outer half (#2048): before the adjoint (#2093) the 25th axis also dropped the
//! outer gradient to finite differences, so `optimizer = auto` resolved to BOBYQA,
//! `gradient_method_outer` read `N/A` and the covariance step ran the FD stencil, with
//! no warning naming the cap. `outer_route_is_the_same_past_the_cap` pins that the
//! outer route no longer depends on the axis count.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{CovarianceStatus, FitOptions};
use ferx_core::{fit, read_nonmem_csv, FitResult};
use std::path::Path;

/// The example model plus `n_fix` zero-effect FIX θ, applied in the order WT
/// power, CRCL power, linear-in-WT, parameter by parameter (CL, V1, Q, V2, KA).
fn make_model(n_fix: usize) -> String {
    let prms = ["CL", "V1", "Q", "V2", "KA"];
    let effs = ["WT", "CRCL", "LIN"];
    let pairs: Vec<(&str, &str)> = prms
        .iter()
        .flat_map(|p| effs.iter().map(move |e| (*p, *e)))
        .take(n_fix)
        .collect();
    assert_eq!(pairs.len(), n_fix, "only 15 zero-effect θ are available");

    let thetas: String = pairs
        .iter()
        .map(|(p, e)| format!("  theta TH_{p}_{e}(0.0, -3.0, 3.0) FIX\n"))
        .collect();
    let factors = |p: &str| -> String {
        pairs
            .iter()
            .filter(|(q, _)| *q == p)
            .map(|(_, e)| match *e {
                "WT" => format!(" * (WT/70)^TH_{p}_WT"),
                "CRCL" => format!(" * (CRCL/100)^TH_{p}_CRCL"),
                _ => format!(" * (1 + TH_{p}_LIN*(WT-70)/70)"),
            })
            .collect()
    };
    let tv = |p: &str| {
        if p == "Q" {
            "TVQ".to_string()
        } else {
            format!("TV{p}")
        }
    };
    let indiv: String = prms
        .iter()
        .map(|p| format!("  {p} = {}{} * exp(ETA_{p})\n", tv(p), factors(p)))
        .collect();

    format!(
        "[parameters]
  theta TVCL(4.0,  0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0,   0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0,  0.01, 10.0)
{thetas}  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_Q  ~ 0.08
  omega ETA_V2 ~ 0.08
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[covariates]
  WT   continuous
  CRCL continuous

[individual_parameters]
{indiv}
[structural_model]
  ode(obs_cmt=central, states=[depot, central, peripheral])

[odes]
  d/dt(depot)      = -KA * depot
  d/dt(central)    =  KA * depot / V1 - (CL/V1) * central - (Q/V1) * central + (Q/V2) * peripheral
  d/dt(peripheral) =  (Q/V1) * central - (Q/V2) * peripheral

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
"
    )
}

/// `ode_tol` overrides `(ode_reltol, ode_abstol)`; `None` keeps the defaults.
fn evaluate(n_fix: usize, ode_tol: Option<(f64, f64)>) -> FitResult {
    let parsed = parse_full_model(&make_model(n_fix)).expect("padded model parses");
    assert_eq!(
        parsed.model.n_theta + parsed.model.n_eta,
        10 + n_fix,
        "axis count of the padded model"
    );
    let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
        .expect("two_cpt_oral_cov.csv loads");
    let mut options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    if let Some((reltol, abstol)) = ode_tol {
        options.ode_reltol = reltol;
        options.ode_abstol = abstol;
    }
    fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("fit evaluates")
}

/// The 24- and 25-axis twins agree bit for bit, and both report an analytic inner.
///
/// Both sides run the same analytic inner on the same data with a FIX θ whose every
/// factor is exactly 1, so nothing can tell them apart. Under the pre-#1804 gate the
/// 25-axis inner is FD and stops one subject at a worse EBE, which these equalities
/// catch. The numbers are checked before the label so that a regression shows the
/// objective it costs, not only the route.
fn assert_twins_agree(case: &str, ode_tol: Option<(f64, f64)>) {
    let at_cap = evaluate(14, ode_tol);
    let past_cap = evaluate(15, ode_tol);

    assert!(
        at_cap.ofv.is_finite() && past_cap.ofv.is_finite(),
        "{case}: OFV {} vs {}",
        at_cap.ofv,
        past_cap.ofv
    );
    assert_eq!(at_cap.subjects.len(), past_cap.subjects.len());
    for (a, b) in at_cap.subjects.iter().zip(&past_cap.subjects) {
        assert_eq!(a.id, b.id);
        assert!(
            a.ofv_contribution.is_finite(),
            "{case}, subject {}: {}",
            a.id,
            a.ofv_contribution
        );
        assert_eq!(
            a.ofv_contribution, b.ofv_contribution,
            "{case}, subject {}: per-subject OFV at 24 vs 25 axes",
            a.id
        );
        assert_eq!(a.eta, b.eta, "{case}, subject {}: η̂ at 24 vs 25 axes", a.id);
    }
    assert_eq!(at_cap.ofv, past_cap.ofv, "{case}: OFV at 24 vs 25 axes");

    for (label, r) in [("24 axes", &at_cap), ("25 axes", &past_cap)] {
        assert_eq!(
            r.gradient_method_inner, "analytic (Dual2)",
            "{case}, {label}: the inner η-gradient is seeded on η only, so it must stay analytic"
        );
        let fd: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("finite-difference inner"))
            .collect();
        assert!(
            fd.is_empty(),
            "{case}, {label}: unexpected FD-inner warning(s): {fd:?}"
        );
    }
}

/// Default ODE tolerances. Measured (macOS arm64): OFV −1033.3327868664 for both
/// twins, every per-subject OFV and η̂ equal. Before #1804 the 25-axis twin read
/// −1017.312, one subject 16.02 above its 24-axis value.
#[test]
fn ode_inner_stays_analytic_one_axis_past_the_cap() {
    assert_twins_agree("default ODE tolerances", None);
}

/// Loose ODE tolerances, where the FD inner was most sensitive: before #1804 the
/// 25-axis twin's objective moved by 26.3 here. The analytic inner
/// differentiates the integrator's own steps, so the twins still agree exactly.
#[test]
fn ode_inner_stays_analytic_past_the_cap_at_loose_ode_tolerances() {
    assert_twins_agree("ode_reltol = 1e-3, ode_abstol = 1e-6", Some((1e-3, 1e-6)));
}

/// #2048: a few outer iterations with the covariance step on, at 24 and 25 axes. Both
/// twins report an analytic outer gradient, resolve `auto` to L-BFGS, carry no
/// finite-difference warning, and compute the covariance step; since the FIX θ is not
/// in the optimizer's vector, the two runs take the same steps and land on the same OFV.
/// Before #2093 the 25-axis twin read `N/A` / `auto (bobyqa)` here. Which R-matrix the
/// step assembles is not visible on `FitResult` — a `Computed` status is reached by the
/// FD stencil too — so that half is pinned by the Tier-1
/// `an_ode_model_past_the_old_axis_cap_assembles_r_analytically_for_every_subject`, which
/// counts the FD objectives the step evaluates.
#[test]
fn outer_route_is_the_same_past_the_cap() {
    let run = |n_fix: usize| {
        let parsed = parse_full_model(&make_model(n_fix)).expect("padded model parses");
        let pop = read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None)
            .expect("two_cpt_oral_cov.csv loads");
        let options = FitOptions {
            outer_maxiter: 3,
            run_covariance_step: true,
            ..parsed.fit_options.clone()
        };
        fit(&parsed.model, &pop, &parsed.model.default_params, &options).expect("fit runs")
    };
    let at_cap = run(14);
    let past_cap = run(15);
    for (label, r) in [("24 axes", &at_cap), ("25 axes", &past_cap)] {
        assert_eq!(
            r.gradient_method_outer, "analytic (Dual2)",
            "{label}: the adjoint outer gradient has no θ + η cap"
        );
        assert_eq!(
            r.optimizer, "auto (nlopt_lbfgs)",
            "{label}: `auto` takes L-BFGS on an analytic outer gradient"
        );
        let fd: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("finite-difference") || w.contains("finite difference"))
            .collect();
        assert!(fd.is_empty(), "{label}: unexpected FD warning(s): {fd:?}");
        assert_eq!(
            r.covariance_status,
            CovarianceStatus::Computed,
            "{label}: covariance step"
        );
    }
    assert!(at_cap.ofv.is_finite(), "OFV {}", at_cap.ofv);
    assert_eq!(at_cap.ofv, past_cap.ofv, "OFV at 24 vs 25 axes");
}
