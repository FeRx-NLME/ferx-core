//! Integration tests for custom / time-varying residual-error magnitude (#484).
//!
//! Tier-2: each test calls the public `fit()` boundary but returns immediately
//! — eval-only (`outer_maxiter = 0`) OFV evaluations, or a couple of SAEM
//! passes — so they stay fast and are compile-checked on every PR.
//!
//! The headline check is that a sigma magnitude written as an expression of
//! TIME / a theta reaches the FOCEI objective: with the late-phase inflation
//! factor at 1 the OFV must equal the plain-proportional model bit-for-bit
//! (the legacy variance path), and at 2 it must change.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::GradientMethod;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// Warfarin oral 1-cpt FOCEI model whose proportional sigma is inflated by
/// `RUV_LATE` for observations after 24 h. `ruv_late_init` is the initial value
/// of the `RUV_LATE` theta.
fn time_varying_ruv_src(ruv_late_init: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta RUV_LATE({ruv_late_init}, 0.1, 10.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR * (if (TIME > 24.0) RUV_LATE else 1.0))

[fit_options]
  method = focei
"
    )
}

/// Same model with a plain proportional error (no magnitude expression) — the
/// legacy path the inert (factor = 1) magnitude must reproduce.
const PLAIN_PROP_SRC: &str = r"
[parameters]
  theta TVCL(0.13, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta RUV_LATE(1.0, 0.1, 10.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

fn eval_only_ofv(src: &str) -> f64 {
    let model = parse_model_string(src).expect("model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    // `outer_maxiter = 0` evaluates the OFV at the initial parameters. Forcing
    // the FD inner gradient on both models makes the only difference between the
    // inert-magnitude and plain-proportional OFVs the variance path itself (the
    // custom-magnitude model already routes to FD; pinning the plain model too
    // keeps the legacy-equivalence comparison exact, not gradient-route noise).
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.methods = vec![];
    opts.outer_maxiter = 0;
    opts.gradient_method = GradientMethod::Fd;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("eval-only fit must succeed");
    assert!(result.ofv.is_finite(), "OFV must be finite");
    result.ofv
}

#[test]
fn magnitude_parses_into_compiled_model() {
    let model = parse_model_string(&time_varying_ruv_src("2.0")).expect("parses");
    let rm = model
        .ruv_magnitude
        .as_ref()
        .expect("custom magnitude present");
    assert!(rm.is_active());
    assert!(model.has_custom_ruv_magnitude());
    // proportional model → one sigma slot, carrying the multiplier.
    assert_eq!(rm.per_sigma.len(), 1);
    assert!(rm.per_sigma[0].is_some());
}

#[test]
fn inert_magnitude_matches_plain_proportional_ofv() {
    // RUV_LATE = 1 makes the multiplier identically 1, so the OFV must equal
    // the plain-proportional model's OFV bit-for-bit (legacy variance path).
    let ofv_inert = eval_only_ofv(&time_varying_ruv_src("1.0"));
    let ofv_plain = eval_only_ofv(PLAIN_PROP_SRC);
    // Equal to within floating-point associativity noise: the scaled variance
    // forms `f²·σ²` while the legacy path forms `(f·σ)²`, which round
    // differently in the last couple of ULPs — but nothing else changes.
    assert!(
        (ofv_inert - ofv_plain).abs() < 1e-6,
        "an inert (factor = 1) magnitude must reproduce the legacy OFV: \
         inert={ofv_inert}, plain={ofv_plain}"
    );
}

#[test]
fn active_magnitude_changes_ofv() {
    // Inflating the late-phase residual error must move the objective.
    let ofv_inert = eval_only_ofv(&time_varying_ruv_src("1.0"));
    let ofv_active = eval_only_ofv(&time_varying_ruv_src("2.0"));
    assert!(
        (ofv_active - ofv_inert).abs() > 1.0,
        "an active magnitude must change the OFV: inert={ofv_inert}, active={ofv_active}"
    );
}

/// Two SAEM passes on the custom-magnitude model, at the given late-phase
/// inflation factor. SAEM used to be rejected outright for a custom magnitude;
/// since #1029 it applies the per-observation multiplier like every other
/// estimator, so this must run and score.
fn saem_ofv(ruv_late_init: &str) -> f64 {
    let model = parse_model_string(&time_varying_ruv_src(ruv_late_init)).expect("model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.methods = vec![];
    // A couple of cheap passes: this is a wiring check, not a convergence test.
    opts.saem_n_exploration = 2;
    opts.saem_n_convergence = 2;
    opts.saem_seed = Some(1);
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("SAEM must accept a custom residual-error magnitude");
    assert!(result.ofv.is_finite(), "SAEM OFV must be finite");
    result.ofv
}

/// SAEM accepts the custom magnitude *and* applies it: the same two passes with
/// the late-phase inflation at 1 (inert) and at 2 (active) must not score the
/// same objective. A SAEM path that read the residual variance without the
/// per-observation multiplier would return identical OFVs.
#[test]
fn saem_applies_the_custom_magnitude() {
    let ofv_inert = saem_ofv("1.0");
    let ofv_active = saem_ofv("2.0");
    assert!(
        (ofv_active - ofv_inert).abs() > 1e-6,
        "SAEM scored an inert and an active magnitude identically \
         (inert={ofv_inert}, active={ofv_active}) — the per-observation \
         multiplier is not reaching its likelihood"
    );
}

/// Joint model on the #2731 anchor data, a factor `tpf` on the `CMT=1` line (`CMT=2` bare).
fn endpoint_magnitude_src(tpf: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(2.2, 0.01, 100.0)
  theta TVV(18.0, 0.1, 1000.0)
  theta TVKE0(0.6, 0.001, 10.0)
  theta TPF({tpf}, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  sigma S ~ 0.25 (sd)
  sigma U ~ 0.4 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect) = KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[covariates]
  EARLY continuous

[error_model]
  CMT=1: DV ~ additive(S * (1.0 + (TPF - 1.0) * EARLY))
  CMT=2: DV ~ additive(U)
"
    )
}

fn endpoint_magnitude_ofv(tpf: &str, method: EstimationMethod) -> f64 {
    let model = parse_model_string(&endpoint_magnitude_src(tpf)).expect("model must parse");
    let population = read_nonmem_csv(
        Path::new("nonmem_anchor/ruv_endpoint_magnitude.csv"),
        None,
        None,
    )
    .expect("data must load");
    let opts = FitOptions {
        method,
        methods: vec![],
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(2),
        saem_n_exploration: 2,
        saem_n_convergence: 2,
        saem_seed: Some(1),
        ..FitOptions::default()
    };
    let result = fit(&model, &population, &model.default_params, &opts)
        .unwrap_or_else(|e| panic!("{method:?} must accept a CMT-line magnitude: {e}"));
    assert!(result.ofv.is_finite(), "{method:?}: OFV {}", result.ofv);
    result.ofv
}

/// #2731: every estimator family scores a `CMT=N:` magnitude, not only the FOCE/FOCEI the
/// anchor covers. The factor at 1 and at 2 straddle it: a likelihood that dropped the row
/// would score the two alike.
#[test]
fn every_estimator_applies_a_cmt_line_magnitude() {
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::Saem,
    ] {
        let inert = endpoint_magnitude_ofv("1.0", method);
        let active = endpoint_magnitude_ofv("2.0", method);
        eprintln!("{method:?}: inert {inert}, active {active}");
        assert!(
            (active - inert).abs() > 1e-3,
            "{method:?} scored TPF = 1 and TPF = 2 alike ({inert}, {active})"
        );
    }
}
