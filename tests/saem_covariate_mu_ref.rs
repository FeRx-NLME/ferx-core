//! Covariate (multi-theta) mu-referencing under SAEM and IMP (issue #619),
//! anchored to NONMEM.
//!
//! A typical value that reads several thetas — `CL = (TVCL + (CRCL-90)*TH_CRCL)
//! * exp(ETA_CL)`, `CL = TVCL * (WT/70)^TH_WT * exp(ETA_CL)` — has no single
//! mu-reference anchor, so before #619 its thetas sat on the eta-frozen numerical
//! M-step and a covariate slope stalled at its start (measured on these very
//! fixtures: `TH_CRCL` 0.0198 from a 0.02 start, `TH_WT` 0.332 from 0.3). ferx
//! now records a *covariate mu-reference* and re-fits the group's thetas to the
//! population of individual values each iteration — NONMEM's
//! `MU_1 = LOG(THETA(1) + (CRCL-90)*THETA(2))`.
//!
//! ## Fixtures
//!
//! `data/covmuref_additive.csv` and `data/covmuref_power.csv`: 1-cpt IV bolus,
//! 60 subjects × 7 samples, simulated from the respective model by
//! `nonmem_anchor/simulate_covmuref_data.py` (matched, well-specified fits). The
//! covariates are constant within each subject, so the exact Gauss–Newton engine
//! runs; the time-varying (prior + data) engine is exercised on the fluconazole
//! model in the PR write-up, which has no NONMEM SAEM comparator.
//!
//! ## NONMEM anchors
//!
//! `nonmem_anchor/covmuref_{additive,power}_saem.ctl`, NONMEM 7.5.1
//! `METHOD=SAEM` (NBURN=2000 NITER=1000 ISAMPLE=10 SEED=619) followed by an
//! `EONLY` IMP objective; final rows of `nonmem_anchor/results/*.ext`.
//!
//! Run the slow tests with:
//!
//!   cargo test --release --features slow-tests --test saem_covariate_mu_ref

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::MuTransform;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// NONMEM 7.5.1 SAEM finals, `nonmem_anchor/results/covmuref_additive_saem.ext`.
/// Columns: THETA1 (TVCL) THETA2 (TH_CRCL) THETA3 (TVV) SIGMA(1,1) OMEGA(1,1) OMEGA(2,2).
const NM_ADD_TVCL: f64 = 4.916_73;
const NM_ADD_TH_CRCL: f64 = 0.041_777_0;
const NM_ADD_TVV: f64 = 48.085_9;
const NM_ADD_OMEGA_CL: f64 = 0.040_098_0;
const NM_ADD_OMEGA_V: f64 = 0.036_046_6;

/// NONMEM 7.5.1 SAEM finals, `nonmem_anchor/results/covmuref_power_saem.ext`.
const NM_POW_TVCL: f64 = 4.787_63;
const NM_POW_TH_WT: f64 = 0.921_283;
const NM_POW_TVV: f64 = 49.516_1;
const NM_POW_OMEGA_CL: f64 = 0.069_657_2;
const NM_POW_OMEGA_V: f64 = 0.045_890_9;

/// Where the pre-#619 binary landed on the same fixtures and seed (release,
/// `worktree-918-logit-mu-ref`): the covariate theta never left its start.
const BEFORE_TH_CRCL: f64 = 0.019_768;
const BEFORE_TH_WT: f64 = 0.332_157;

fn load(model_file: &str, data_file: &str) -> (ferx_core::CompiledModel, ferx_core::Population) {
    let src = std::fs::read_to_string(Path::new("nonmem_anchor").join(model_file))
        .expect("anchor model file");
    let model = parse_full_model(&src).expect("anchor model parses").model;
    let pop =
        read_nonmem_csv(&Path::new("data").join(data_file), None, None).expect("anchor data loads");
    (model, pop)
}

fn theta(result: &FitResult, name: &str) -> f64 {
    let i = result
        .theta_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("theta {name} missing from {:?}", result.theta_names));
    result.theta[i]
}

fn omega(result: &FitResult, eta: &str) -> f64 {
    let i = result
        .eta_names
        .iter()
        .position(|n| n == eta)
        .unwrap_or_else(|| panic!("eta {eta} missing from {:?}", result.eta_names));
    result.omega[(i, i)]
}

fn saem_opts() -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        run_covariance_step: false,
        verbose: false,
        saem_seed: Some(619),
        ..FitOptions::default()
    }
}

fn assert_finite_close(what: &str, got: f64, want: f64, tol: f64) {
    assert!(got.is_finite(), "{what} is not finite: {got}");
    assert!(
        (got - want).abs() < tol,
        "{what}: ferx {got:.6} vs reference {want:.6} (|Δ| = {:.2e} ≥ {tol:.1e})",
        (got - want).abs()
    );
}

#[test]
fn covariate_mu_refs_are_detected_in_both_anchor_models() {
    let (add, _) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    assert_eq!(add.covariate_mu_refs.len(), 1);
    let g = &add.covariate_mu_refs[0];
    assert_eq!(g.eta_name, "ETA_CL");
    assert_eq!(g.theta_names, vec!["TVCL", "TH_CRCL"]);
    assert_eq!(g.transform, MuTransform::Log);
    assert_eq!(g.covariate_names, vec!["CRCL"]);
    // The additive form never matched a single-anchor pattern: `mu_refs` is V only.
    assert!(!add.mu_refs.contains_key("ETA_CL"));
    assert!(add.mu_refs.contains_key("ETA_V"));

    let (pow, _) = load("covmuref_power_saem_fit.ferx", "covmuref_power.csv");
    assert_eq!(pow.covariate_mu_refs.len(), 1);
    assert_eq!(pow.covariate_mu_refs[0].theta_names, vec!["TVCL", "TH_WT"]);
    // The power form keeps its historical single anchor on TVCL as well.
    assert_eq!(
        pow.mu_refs.get("ETA_CL").map(|m| m.theta_name.as_str()),
        Some("TVCL")
    );
}

/// Tier-2: the group is a closed-form channel. `V` keeps its own single-anchor
/// pair, so the discriminating signal is the `not mu-referenced: CL` warning,
/// which the pre-#619 binary emitted on this exact model.
#[test]
fn saem_covariate_group_removes_the_not_mu_referenced_warning() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let opts = FitOptions {
        saem_n_exploration: 2,
        saem_n_convergence: 1,
        ..saem_opts()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("short SAEM must run");
    assert!(
        result.saem_mu_ref_m_step_evals_saved.unwrap_or(0) > 0,
        "closed-form channel must be active: {:?}",
        result.saem_mu_ref_m_step_evals_saved
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("individual parameter(s) not mu-referenced")),
        "CL is mu-referenced through its covariate group, got {:?}",
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("covariate mu-reference on")),
        "no group may be declined on this model, got {:?}",
        result.warnings
    );
}

/// Tier-3: the additive renal gradient — NONMEM's *nonlinear* MU case — lands
/// where NONMEM SAEM lands. Bounds are measured: realised |Δ| on the anchor run
/// was 4e-3 (TVCL), 8e-6 (TH_CRCL), 3e-4 (TVV), 2e-4 (both ω²); each bound
/// below keeps ≥ 25× headroom on that and still excludes the pre-#619 value.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn saem_recovers_the_additive_renal_gradient() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let result = fit(&model, &pop, &model.default_params, &saem_opts()).expect("SAEM must run");
    let th_crcl = theta(&result, "TH_CRCL");
    assert_finite_close("TH_CRCL", th_crcl, NM_ADD_TH_CRCL, 0.003);
    assert!(
        (th_crcl - BEFORE_TH_CRCL).abs() > 0.015,
        "TH_CRCL = {th_crcl:.5} must be clear of the pre-#619 stall at {BEFORE_TH_CRCL}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_ADD_TVCL, 0.15);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_ADD_TVV, 1.0);
    assert_finite_close(
        "omega^2(ETA_CL)",
        omega(&result, "ETA_CL"),
        NM_ADD_OMEGA_CL,
        0.01,
    );
    assert_finite_close(
        "omega^2(ETA_V)",
        omega(&result, "ETA_V"),
        NM_ADD_OMEGA_V,
        0.01,
    );
}

/// Tier-3: the allometric exponent — NONMEM's linear-in-theta MU, its
/// efficient case — so the group step must not lose ground there either.
/// Realised |Δ| on the anchor run: 7e-4 (TVCL), 8e-4 (TH_WT), 2e-2 (TVV),
/// 4e-4 / 1e-4 (ω²).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn saem_recovers_the_allometric_exponent() {
    let (model, pop) = load("covmuref_power_saem_fit.ferx", "covmuref_power.csv");
    let result = fit(&model, &pop, &model.default_params, &saem_opts()).expect("SAEM must run");
    let th_wt = theta(&result, "TH_WT");
    assert_finite_close("TH_WT", th_wt, NM_POW_TH_WT, 0.05);
    assert!(
        (th_wt - BEFORE_TH_WT).abs() > 0.3,
        "TH_WT = {th_wt:.4} must be clear of the pre-#619 stall at {BEFORE_TH_WT}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_POW_TVCL, 0.15);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_POW_TVV, 1.0);
    assert_finite_close(
        "omega^2(ETA_CL)",
        omega(&result, "ETA_CL"),
        NM_POW_OMEGA_CL,
        0.01,
    );
    assert_finite_close(
        "omega^2(ETA_V)",
        omega(&result, "ETA_V"),
        NM_POW_OMEGA_V,
        0.01,
    );
}

/// Tier-3: IMP shares the group step (exact engine on the importance-weighted
/// posterior means). Same MLE as SAEM, so the same NONMEM finals are the
/// reference; IMP's own Monte-Carlo noise is why the bounds are wider (realised
/// |Δ| on the anchor run: 1e-4 for TH_CRCL, 3e-3 for TVCL).
///
/// This is also the regression pin for the φ-preserving re-centre of the IMP
/// proposal after a group step: without it the first step lands right here
/// (0.041) and the next five run the slope up to 0.10 on a stale proposal, after
/// which it collapses to the bound (measured 1e-5) — the `> 0.015` clearance
/// below is what fails in that case.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored covariate mu-referencing (#619): opt in with --features slow-tests"
)]
fn imp_recovers_the_additive_renal_gradient() {
    let (model, pop) = load("covmuref_additive_saem_fit.ferx", "covmuref_additive.csv");
    let opts = FitOptions {
        method: EstimationMethod::Imp,
        imp_iterations: 150,
        imp_samples: 500,
        imp_auto: false,
        imp_seed: Some(619),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    };
    let result = fit(&model, &pop, &model.default_params, &opts).expect("IMP must run");
    let th_crcl = theta(&result, "TH_CRCL");
    assert_finite_close("TH_CRCL", th_crcl, NM_ADD_TH_CRCL, 0.006);
    assert!(
        (th_crcl - BEFORE_TH_CRCL).abs() > 0.015,
        "TH_CRCL = {th_crcl:.5} must be clear of the pre-#619 stall at {BEFORE_TH_CRCL}"
    );
    assert_finite_close("TVCL", theta(&result, "TVCL"), NM_ADD_TVCL, 0.3);
    assert_finite_close("TVV", theta(&result, "TVV"), NM_ADD_TVV, 2.0);
}
