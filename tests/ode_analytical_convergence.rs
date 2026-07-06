//! ODE-vs-analytical **convergence** equivalence (issue #410).
//!
//! `analytical_ode_equivalence.rs` shows the analytical closed forms and their
//! ODE transcriptions agree *structurally* (`predict()` and fixed-parameter NLL).
//! This goes one step further: with the user-ODE analytic outer gradient now armed
//! (#410), a full FOCEI **fit to convergence** of an ODE model must land on the
//! same estimates *and* standard errors as the analytical twin — which is itself
//! NONMEM-validated (the warfarin 1-cpt-oral covariance cross-check). So this is
//! the transitive NONMEM check for the armed ODE path: analytic ODE gradient
//! drives the optimizer to the NONMEM-validated optimum, and the covariance step
//! reproduces the NONMEM-validated SEs.
//!
//! Tier 3 (full convergence) — gated behind `slow-tests`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// Warfarin 1-cpt oral, proportional error — the NONMEM-validated reference model.
const ANALYTICAL_SRC: &str = r"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// Hand-transcribed ODE twin of the same model. Uses the **Form-C readout**
/// (`y = central / V`), which the ODE sensitivity provider supports — *not* the
/// `obs_scale = V` divisor form (out of scope), so the fit genuinely exercises the
/// armed analytic ODE gradient. Tight ODE tolerances so the integrated solution
/// (and its propagated sensitivities) reproduce the closed form to estimate level.
const ODE_SRC: &str = r"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.2, 0.01, 50.0)
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.10
  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  ode(states=[depot, central])

[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
";

fn fit_warfarin(src: &str) -> ferx_core::FitResult {
    let model = parse_model_string(src).expect("model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin loads");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    // Default optimizer (`Auto` → analytic-gradient NLopt L-BFGS). The built-in
    // `Lbfgs` stops short on the analytical warfarin surface ("did not converge"),
    // leaving a near-saddle where the default OFV-second-difference covariance
    // Hessian picks up a large negative eigenvalue and — after the eigenvalue
    // floor — explodes SE(TVV) to ~50. `Auto` reaches the true stationary point,
    // where the FD-of-OFV Hessian is well-conditioned and SE(TVV) matches NONMEM
    // (0.240). The ODE twin already converged, hence only its SEs were sane.
    opts.outer_maxiter = 300;
    opts.run_covariance_step = true;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("fit runs")
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full FOCEI convergence of an ODE model — opt in with --features slow-tests"
)]
fn ode_fit_converges_to_analytical_estimates_and_se() {
    // Sanity: the ODE model must actually be on the armed analytic path; otherwise
    // this would silently validate the FD fallback instead of #410.
    let ode_model = parse_model_string(ODE_SRC).expect("ODE parses");
    assert!(
        ferx_core::sens::provider::sens_supported(&ode_model),
        "ODE twin must be armed for the analytic outer gradient (#410)"
    );

    let an = fit_warfarin(ANALYTICAL_SRC);
    let ode = fit_warfarin(ODE_SRC);

    let se_an = an.se_theta.as_ref().expect("analytical SEs");
    let se_ode = ode.se_theta.as_ref().expect("ODE SEs");
    let names = ["TVCL", "TVV", "TVKA"];

    // Side-by-side numbers: analytical (NONMEM-validated) vs armed ODE.
    eprintln!("\n=== warfarin 1-cpt-oral FOCEI: analytical vs user-ODE (#410) ===");
    eprintln!(
        "OFV       analytical={:.4}   ODE={:.4}   Δ={:+.4}",
        an.ofv,
        ode.ofv,
        ode.ofv - an.ofv
    );
    eprintln!(
        "{:<10} {:>14} {:>14} {:>10}",
        "param", "analytical", "ODE", "rel"
    );
    for i in 0..3 {
        let rel = (an.theta[i] - ode.theta[i]).abs() / an.theta[i].abs();
        eprintln!(
            "{:<10} {:>14.6} {:>14.6} {:>9.1e}",
            names[i], an.theta[i], ode.theta[i], rel
        );
    }
    for k in 0..3 {
        let rel = (an.omega[(k, k)] - ode.omega[(k, k)]).abs() / an.omega[(k, k)].abs();
        eprintln!(
            "{:<10} {:>14.6} {:>14.6} {:>9.1e}",
            format!("om2[{k}]"),
            an.omega[(k, k)],
            ode.omega[(k, k)],
            rel
        );
    }
    let rel_sig = (an.sigma[0] - ode.sigma[0]).abs() / an.sigma[0].abs();
    eprintln!(
        "{:<10} {:>14.6} {:>14.6} {:>9.1e}",
        "sigma", an.sigma[0], ode.sigma[0], rel_sig
    );
    eprintln!("-- standard errors --");
    for i in 0..3 {
        let rel = (se_an[i] - se_ode[i]).abs() / se_an[i].abs();
        eprintln!(
            "{:<10} {:>14.6} {:>14.6} {:>9.1e}",
            format!("SE({})", names[i]),
            se_an[i],
            se_ode[i],
            rel
        );
    }
    eprintln!();

    // Estimates agree tightly (same objective; only ODE-solver truncation differs).
    for i in 0..3 {
        let rel = (an.theta[i] - ode.theta[i]).abs() / an.theta[i].abs();
        assert!(
            rel < 0.01,
            "θ {} differs: analytical {:.6} vs ODE {:.6} (rel {:.2e})",
            names[i],
            an.theta[i],
            ode.theta[i],
            rel
        );
        // SEs from the covariance step agree to a few percent.
        let rel_se = (se_an[i] - se_ode[i]).abs() / se_an[i].abs();
        assert!(
            rel_se < 0.05,
            "SE({}) differs: analytical {:.6} vs ODE {:.6} (rel {:.2e})",
            names[i],
            se_an[i],
            se_ode[i],
            rel_se
        );
    }

    // Random-effect variances and residual error agree too.
    for k in 0..3 {
        let rel = (an.omega[(k, k)] - ode.omega[(k, k)]).abs() / an.omega[(k, k)].abs();
        assert!(
            rel < 0.05,
            "ω²[{k}] differs: analytical {:.6} vs ODE {:.6} (rel {:.2e})",
            an.omega[(k, k)],
            ode.omega[(k, k)],
            rel
        );
    }
    let rel_sig = (an.sigma[0] - ode.sigma[0]).abs() / an.sigma[0].abs();
    assert!(
        rel_sig < 0.02,
        "σ differs: analytical {:.6} vs ODE {:.6} (rel {:.2e})",
        an.sigma[0],
        ode.sigma[0],
        rel_sig
    );
}
