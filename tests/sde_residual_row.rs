//! #3126 at the population level: the SDE filter's Kalman `R` is the row's scored residual
//! variance, so two spellings of one row variance give one FOCEI objective on a real dataset.
//!
//! Tier 2: each case is one objective evaluation (`outer_maxiter = 0`), no convergence loop.
//! The per-PR guard is `stats::likelihood::ekf_row_variance_tests` (fixed η, ≤ 1e-12); this
//! file is the fit-level confirmation, where the EBEs are re-estimated, so the pairs agree
//! only to the inner loop's tolerance.
//!
//! NONMEM has no EKF, so there is no NONMEM anchor: the oracle is the degenerate pair itself.
//! Before #3126 (measured at `89b666964`, macOS arm64): pair 1 was 3.60 OFV apart, pair 2
//! 1.04; the same pairs without `[diffusion]` agreed to 1.2e-10 and 0 — the inner-loop floor
//! that bounds this file. After #3126 (macOS arm64): 1.6e-13 and 0, control 1.2e-10; with the
//! row dropped from the filter's `R` again, 3.600 and 1.038 (both red).

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, read_nonmem_csv};
use std::path::Path;

const DATA: &str = "data/warfarin.csv";

/// `examples/warfarin_sde.ferx` under FOCEI, with the sigma line and error line substituted.
fn src(sigma: &str, error: &str, diffusion: bool) -> String {
    let diff = if diffusion { "[diffusion]\n  central ~ 0.01\n" } else { "" };
    format!(
        "[parameters]\n  theta TVCL(0.134, 0.001, 10.0)\n  theta TVV(8.1, 0.1, 500.0)\n  \
         theta TVKA(1.0, 0.01, 50.0)\n  omega ETA_CL ~ 0.07\n  omega ETA_V ~ 0.02\n  \
         omega ETA_KA ~ 0.40\n  {sigma}\n[individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  \
         V = TVV * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n[structural_model]\n  \
         ode(obs_cmt=central, states=[depot, central])\n[odes]\n  d/dt(depot) = -KA * depot\n  \
         d/dt(central) = KA * depot / V - (CL / V) * central\n{diff}[error_model]\n  {error}\n\
         [fit_options]\n  method = focei\n  covariance = false\n"
    )
}

/// The FOCEI objective at the initial estimates: an evaluation, not a fit. Every subject
/// carries a constant covariate `W = 2` (the weight pair's; the dataset has no such column).
fn ofv(model_src: &str) -> f64 {
    let parsed = parse_full_model(model_src).expect("the model must parse");
    let mut pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    for s in &mut pop.subjects {
        s.covariates.insert("W".to_string(), 2.0);
        for c in &mut s.obs_covariates {
            c.insert("W".to_string(), 2.0);
        }
    }
    let options = FitOptions { outer_maxiter: 0, ..parsed.fit_options.clone() };
    let result = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(result.ofv.is_finite(), "OFV is not finite");
    result.ofv
}

fn assert_pair(name: &str, a: &str, b: &str) {
    let (oa, ob) = (ofv(a), ofv(b));
    let d = (oa - ob).abs();
    eprintln!("{name}: {oa:.12} vs {ob:.12}, |Δ| {d:.3e}");
    // Inner-loop floor: 1.2e-10 on the no-diffusion control (`89b666964`).
    assert!(d <= 1e-9, "{name}: {oa} vs {ob} (|Δ| {d:.3e})");
}

/// Pair 1: `proportional(PROP_ERR * 2)` at σ = 0.01 is `proportional(PROP_ERR)` at 0.02.
#[test]
fn sde_magnitude_row_pair_gives_one_objective() {
    assert_pair(
        "magnitude",
        &src("sigma PROP_ERR ~ 0.01 (sd)", "DV ~ proportional(PROP_ERR * 2)", true),
        &src("sigma PROP_ERR ~ 0.02 (sd)", "DV ~ proportional(PROP_ERR)", true),
    );
}

/// Pair 2: `additive(ADD_ERR) weight = W` with `W ≡ 2` at 0.5 is `additive(ADD_ERR)` at 1.0.
#[test]
fn sde_weight_pair_gives_one_objective() {
    assert_pair(
        "weight",
        &src(
            "sigma ADD_ERR ~ 0.5 (sd)",
            "DV ~ additive(ADD_ERR) weight = W\n[covariates]\n  W continuous",
            true,
        ),
        &src("sigma ADD_ERR ~ 1.0 (sd)", "DV ~ additive(ADD_ERR)", true),
    );
}

/// Control: without `[diffusion]` there is no filter, and the pair agreed before #3126 too.
#[test]
fn no_diffusion_control_pair_gives_one_objective() {
    assert_pair(
        "control",
        &src("sigma PROP_ERR ~ 0.01 (sd)", "DV ~ proportional(PROP_ERR * 2)", false),
        &src("sigma PROP_ERR ~ 0.02 (sd)", "DV ~ proportional(PROP_ERR)", false),
    );
}
