//! AD-vs-FD inner-gradient consistency guard.
//!
//! ferx's default build uses Enzyme autodiff for the inner-loop gradients; CI
//! builds FD-only (`--features ci`), so the autodiff path is otherwise *entirely
//! untested*. This test fits representative models under both
//! `gradient_method = ad` and `= fd` and asserts the objective agrees.
//!
//! It is the regression guard for the analytical-AD fast-path gaps in issue
//! #278: the single-snapshot Enzyme kernels hardcode the log-normal map
//! `param = tv*exp(eta)` and a log-wrap for LTBS, so additive / logit / custom
//! ETA parameterisations and `log_additive` (LTBS) error get a gradient that
//! disagrees with the objective. `resolve_gradient_method` now routes those
//! AD->FD, so the two runs must match; for genuinely AD-supported models
//! (log-normal ETA + standard error) the AD path is exercised directly and the
//! match verifies AD correctness.
//!
//! Only meaningful with Enzyme, so gated on `feature = "autodiff"`. Run with:
//!   RUSTFLAGS="-Z autodiff=Enable" cargo test --features autodiff \
//!     --test autodiff_fd_consistency
#![cfg(feature = "autodiff")]

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::GradientMethod;
use ferx_core::{fit, FitOptions};

fn fit_ofv(model_src: &str, data: &str, gm: GradientMethod) -> f64 {
    let mut model = parse_model_string(model_src).expect("model parses");
    // `resolve_gradient_method` reads `model.gradient_method`; set it directly so
    // each run is pinned to AD or FD regardless of the default `Auto`.
    model.gradient_method = gm;
    // `read_population_for` (not the bare `read_nonmem_csv`) so `[event_model]`
    // TTE rows are routed to `obs_records` - otherwise the TTE fits error. For
    // Gaussian models it falls through to the same `read_nonmem_csv` path.
    let (pop, _) = ferx_core::api::read_population_for(&model, &None, data, None, None, None)
        .expect("data loads");
    let mut opts = FitOptions::default();
    opts.gradient_method = gm;
    opts.outer_maxiter = 300;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts)
        .expect("fit runs")
        .ofv
}

/// Fit under AD and FD and assert the objective matches. `0.5` absolute tolerance
/// is far below the smallest real divergence we observed (~4 OFV for LTBS on
/// well-conditioned data, tens-to-hundreds otherwise) and well above optimiser
/// path noise (~0.02 OFV for the AD-vs-FD control).
fn assert_ad_matches_fd(label: &str, model_src: &str, data: &str) {
    let ad = fit_ofv(model_src, data, GradientMethod::Ad);
    let fd = fit_ofv(model_src, data, GradientMethod::Fd);
    let diff = (ad - fd).abs();
    assert!(
        diff < 0.5,
        "{label}: AD OFV {ad:.4} vs FD OFV {fd:.4} (|diff| = {diff:.4}) - \
         the AD and FD inner-gradient paths disagree (issue #278)"
    );
}

const WARF_PARAMS: &str = "
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
";

const WARF_STRUCT_PROP: &str = "
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)
";

/// Control: log-normal ETA + proportional error. This is genuinely AD-supported,
/// so the AD path runs for real and must agree with FD (verifies AD correctness,
/// not just the gate).
#[test]
fn proportional_lognormal_ad_matches_fd() {
    let src = format!(
        "{WARF_PARAMS}  sigma PROP_ERR ~ 0.04\n{WARF_STRUCT_PROP}\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = foce\n"
    );
    assert_ad_matches_fd(
        "proportional / log-normal (control)",
        &src,
        "data/warfarin.csv",
    );
}

/// Additive ETA on CL — the analytical kernel's hardcoded `exp()` is wrong;
/// must be gated AD->FD.
#[test]
fn additive_eta_ad_matches_fd() {
    let src = format!(
        "{WARF_PARAMS}  sigma PROP_ERR ~ 0.04\n\
         [individual_parameters]\n  CL = TVCL + ETA_CL\n  V  = TVV * exp(ETA_V)\n  \
         KA = TVKA * exp(ETA_KA)\n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n[fit_options]\n  method = foce\n"
    );
    assert_ad_matches_fd("additive ETA on CL", &src, "data/warfarin.csv");
}

/// LTBS / `log_additive` — the analytical log-wrap Jacobian diverges; gated AD->FD.
#[test]
fn ltbs_ad_matches_fd() {
    let src = format!(
        "{WARF_PARAMS}  sigma ADD_LOG ~ 0.1\n{WARF_STRUCT_PROP}\
         [error_model]\n  log(DV) ~ additive(ADD_LOG)\n[fit_options]\n  method = foce\n"
    );
    assert_ad_matches_fd("LTBS / log_additive", &src, "data/warfarin.csv");
}

/// Logit-normal bioavailability (`EtaParamType::Logit`) — bundled example; gated AD->FD.
#[test]
fn logit_f_ad_matches_fd() {
    let src = std::fs::read_to_string("examples/warfarin_logit_f.ferx")
        .expect("read examples/warfarin_logit_f.ferx");
    assert_ad_matches_fd("logit-normal F", &src, "data/warfarin_logit_f.csv");
}

/// Covariate `if`-expression (`EtaParamType::Custom`) — bundled example; gated AD->FD.
#[test]
fn if_expression_ad_matches_fd() {
    let src = std::fs::read_to_string("examples/warfarin_if.ferx")
        .expect("read examples/warfarin_if.ferx");
    assert_ad_matches_fd("covariate if-expression", &src, "data/warfarin_if.csv");
}

/// Eta-dependent `obs_scale` (`obs_scale = V`, `V = TVV*exp(ETA_V)`) — the AD path
/// freezes the scale subject-static, dropping `d obs_scale / d eta`; gated AD->FD.
/// Bundled example; diverged ~12 OFV before the gate.
#[test]
fn scaling_expression_ad_matches_fd() {
    let src = std::fs::read_to_string("examples/scaling_expression.ferx")
        .expect("read examples/scaling_expression.ferx");
    assert_ad_matches_fd("eta-dependent obs_scale", &src, "data/one_cpt_iv.csv");
}

/// TTE hazard (`[event_model]`) — the analytical AD kernel computes the PK NLL,
/// not the hazard likelihood, so the eta-gradient through the hazard shape is
/// wrong; gated AD->FD. weibull/gompertz diverged ~2-5 OFV before the gate
/// (exponential, no shape param, was borderline). Needs the `survival` feature
/// (TTE parsing) in addition to `autodiff`.
#[cfg(feature = "survival")]
#[test]
fn tte_weibull_ad_matches_fd() {
    let src = std::fs::read_to_string("examples/tte_weibull.ferx")
        .expect("read examples/tte_weibull.ferx");
    assert_ad_matches_fd("TTE weibull hazard", &src, "data/tte_weibull.csv");
}

#[cfg(feature = "survival")]
#[test]
fn tte_gompertz_ad_matches_fd() {
    let src = std::fs::read_to_string("examples/tte_gompertz.ferx")
        .expect("read examples/tte_gompertz.ferx");
    assert_ad_matches_fd("TTE gompertz hazard", &src, "data/tte_gompertz.csv");
}
