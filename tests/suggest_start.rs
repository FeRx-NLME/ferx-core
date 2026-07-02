//! Integration tests for `inits_from_nca` (nca / nca_sweep / nca_ebe).

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{inits_from_nca, read_nonmem_csv, NcaInit};
use std::path::Path;

fn warfarin() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn two_cpt_iv() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model = parse_model_file(Path::new("examples/two_cpt_iv.ferx"))
        .expect("two_cpt_iv model must parse");
    let population = read_nonmem_csv(Path::new("data/two_cpt_iv.csv"), None, None)
        .expect("two_cpt_iv data must load");
    (model, population)
}

/// TVCL suggested for warfarin must be closer to the known truth (0.134) than the
/// bad starting value (0.5).  |log(suggested / 0.134)| < |log(0.5 / 0.134)|.
#[test]
fn test_suggest_start_warfarin_direction() {
    let (model, population) = warfarin();
    let result = inits_from_nca(&model, &population, NcaInit::Nca);

    // Find TVCL index.
    let tvcl_idx = result
        .params
        .theta_names
        .iter()
        .position(|n| n == "TVCL")
        .expect("TVCL must be a theta");

    let suggested = result.params.theta[tvcl_idx];
    let truth = 0.134_f64;
    let bad_start = 0.5_f64;

    let err_suggested = (suggested / truth).ln().abs();
    let err_bad = (bad_start / truth).ln().abs();

    assert!(
        err_suggested < err_bad,
        "suggested TVCL ({suggested:.4}) should be closer to truth ({truth}) than bad start ({bad_start}); \
         log-errors: suggested={err_suggested:.3}, bad={err_bad:.3}"
    );
}

/// Suggested params must all be within bounds.
#[test]
fn test_suggest_start_warfarin_within_bounds() {
    let (model, population) = warfarin();
    let result = inits_from_nca(&model, &population, NcaInit::Nca);
    for (i, &theta) in result.params.theta.iter().enumerate() {
        let lo = result.params.theta_lower[i];
        let hi = result.params.theta_upper[i];
        assert!(
            theta >= lo && theta <= hi,
            "theta[{i}] ({name}) = {theta} outside [{lo}, {hi}]",
            name = result.params.theta_names[i]
        );
    }
}

/// For 2-cpt IV, CL and V1 should be within 3× of the model default (sanity,
/// not accuracy — just confirming we get plausible values, not garbage).
#[test]
fn test_suggest_start_two_cpt_iv_sanity() {
    let (model, population) = two_cpt_iv();
    let result = inits_from_nca(&model, &population, NcaInit::Nca);

    let find = |name: &str| -> f64 {
        let idx = result
            .params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or(0);
        result.params.theta[idx]
    };
    let find_default = |name: &str| -> f64 {
        let idx = model
            .default_params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or(0);
        model.default_params.theta[idx]
    };

    for param in &["TVCL", "TVV1"] {
        let suggested = find(param);
        let default = find_default(param);
        if default > 0.0 && suggested > 0.0 {
            let ratio = (suggested / default).ln().abs();
            assert!(
                ratio < (3.0_f64).ln(),
                "{param} suggestion ({suggested:.3}) is more than 3× from default ({default:.3})"
            );
        }
    }
}

/// No subjects → returns model defaults without panicking, with a warning.
#[test]
fn test_suggest_start_empty_population() {
    let (model, _) = warfarin();
    let empty = ferx_core::types::Population {
        subjects: vec![],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let result = inits_from_nca(&model, &empty, NcaInit::Nca);
    assert!(!result.warnings.is_empty(), "must warn on empty population");
    assert_eq!(
        result.params.theta, model.default_params.theta,
        "empty population must return model defaults"
    );
}

/// nca_sweep: all thetas within bounds for 2-cpt IV.
#[test]
fn test_nca_sweep_two_cpt_iv_within_bounds() {
    let (model, population) = two_cpt_iv();
    let result = inits_from_nca(&model, &population, NcaInit::Sweep);
    for (i, &theta) in result.params.theta.iter().enumerate() {
        let lo = result.params.theta_lower[i];
        let hi = result.params.theta_upper[i];
        assert!(
            theta >= lo && theta <= hi,
            "thorough: theta[{i}] ({name}) = {theta} outside [{lo}, {hi}]",
            name = result.params.theta_names[i]
        );
    }
}

/// nca_sweep should move unwritten thetas away from model default.
///
/// For 3-cpt IV, biexponential peeling estimates CL/V1/Q/V2 from the
/// terminal-pooled curve but cannot reach the third distribution phase
/// (Q3/V3), which therefore start at their model defaults. After the
/// sweep, Q3 and/or V3 should have changed (rRMSE found a better value).
///
/// The 2-cpt IV fixture used by the other tests in this file is *not*
/// suitable here: biexponential peeling for 2-cpt writes all four thetas,
/// so the sweep's `remaining` set is empty and it is — correctly — a
/// no-op. The 3-cpt fixture is the smallest case where `nca_sweep` and
/// `nca_only` legitimately produce different theta vectors.
#[test]
fn test_nca_sweep_moves_unwritten_thetas() {
    let model = parse_model_file(Path::new("examples/three_cpt_iv.ferx"))
        .expect("three_cpt_iv model must parse");
    let population = read_nonmem_csv(Path::new("data/three_cpt_iv.csv"), None, None)
        .expect("three_cpt_iv data must load");

    let fast = inits_from_nca(&model, &population, NcaInit::Nca);
    let thorough = inits_from_nca(&model, &population, NcaInit::Sweep);

    // At least one theta should differ between the two results (the sweep did something).
    let any_changed = fast
        .params
        .theta
        .iter()
        .zip(thorough.params.theta.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);

    assert!(
        any_changed,
        "nca_sweep should change at least one theta vs nca-only \
         (3-cpt Q3/V3 left at default by peeling should be filled in by sweep)"
    );
}

/// `two_cpt_transit` must get the same biexponential-peel Q/V2 seeding as the
/// otherwise-equivalent `two_cpt_oral` (#634 review finding 2). Before the fix the
/// peel dispatch skipped `TwoCptTransit`, so Q/V2 silently stayed at their model
/// defaults. Perturb the Q/V2 defaults away from the simulation truth (Q=10, V2=100)
/// and confirm the NCA peel moves them back off those bad defaults.
#[test]
fn test_suggest_start_two_cpt_transit_peels_q_v2() {
    let mut model = parse_model_file(Path::new("examples/two_cpt_transit.ferx"))
        .expect("two_cpt_transit model must parse");
    let population = read_nonmem_csv(Path::new("data/transit_2cpt.csv"), None, None)
        .expect("transit_2cpt data must load");

    let theta_idx = |name: &str| {
        model
            .default_params
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("model must carry a {name} theta"))
    };
    let (q_i, v2_i) = (theta_idx("TVQ"), theta_idx("TVV2"));
    // Perturb well away from truth but inside the declared bounds (Q∈[0.5,200], V2∈[5,1000]).
    let (bad_q, bad_v2) = (150.0, 800.0);
    model.default_params.theta[q_i] = bad_q;
    model.default_params.theta[v2_i] = bad_v2;

    let result = inits_from_nca(&model, &population, NcaInit::Nca);
    assert!(
        (result.params.theta[q_i] - bad_q).abs() > 1e-6
            || (result.params.theta[v2_i] - bad_v2).abs() > 1e-6,
        "two_cpt_transit NCA must peel Q/V2 off the (bad) default, got Q={} V2={}",
        result.params.theta[q_i],
        result.params.theta[v2_i]
    );
}

/// When a model uses block_omega (correlated etas), updating the CL omega from
/// NCA CV² must not discard the off-diagonal entries.
#[test]
fn test_suggest_start_block_omega_preserved() {
    use ferx_core::types::OmegaMatrix;
    use nalgebra::DMatrix;

    // Parse a fresh model so we can mutate default_params.omega.
    let mut model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let n = model.n_eta;
    if n >= 2 {
        // Replace with a 2×2 block omega (ETA_CL correlated with ETA_V) with off-diagonal = 0.02.
        let mut m = DMatrix::<f64>::identity(n, n) * 0.09;
        m[(0, 1)] = 0.02;
        m[(1, 0)] = 0.02;
        let names = model.default_params.omega.eta_names.clone();
        let mut free_mask = DMatrix::from_element(n, n, false);
        for i in 0..n {
            for j in 0..n {
                free_mask[(i, j)] = i == j || (i < 2 && j < 2);
            }
        }
        model.default_params.omega = OmegaMatrix::from_matrix_with_mask(m, names, false, free_mask);

        let result = inits_from_nca(&model, &population, NcaInit::Nca);

        // Off-diagonal (0,1) must survive the omega update.
        let off_diag = result.params.omega.matrix[(0, 1)];
        assert!(
            (off_diag - 0.02).abs() < 1e-10,
            "block omega off-diagonal should be preserved after inits_from_nca, got {off_diag}"
        );
    }
}

/// inits_from_nca = nca_sweep must produce different theta than the model
/// default when NCA or the rRMSE sweep finds a better starting point.
#[test]
fn test_inits_from_nca_sweep_changes_params() {
    // We test the plumbing by calling the nca_sweep strategy directly (the same
    // one inits_from_nca = nca_sweep invokes) and confirming it changes a theta.
    let (model, population) = warfarin();
    let result = inits_from_nca(&model, &population, NcaInit::Sweep);
    let any_changed = result
        .params
        .theta
        .iter()
        .zip(model.default_params.theta.iter())
        .any(|(suggested, &default)| (suggested - default).abs() > 1e-10);
    assert!(
        any_changed,
        "nca_sweep (used by inits_from_nca = nca_sweep) must change at least one theta from model default"
    );
}

/// nca_ebe on an ODE model must fall back to nca_sweep and emit a warning.
#[test]
fn test_nca_ebe_ode_fallback_warning() {
    // mm_iv.ferx uses an ODE-based Michaelis-Menten model.
    let model = parse_model_file(Path::new("examples/mm_iv.ferx")).expect("mm_iv model must parse");
    let population =
        read_nonmem_csv(Path::new("data/mm_iv.csv"), None, None).expect("mm_iv data must load");

    assert!(model.ode_spec.is_some(), "mm_iv model must be an ODE model");

    let result = inits_from_nca(&model, &population, NcaInit::Ebe);
    let has_ode_warning = result
        .warnings
        .iter()
        .any(|w| w.contains("ODE") && w.contains("Falling back"));
    assert!(
        has_ode_warning,
        "nca_ebe on an ODE model must emit the ODE-fallback warning; got: {:?}",
        result.warnings
    );
}

/// Tier 3 — slow: trust_region should converge from the bad TVCL=0.5 start
/// when inits_from_nca = nca_sweep corrects the initial values.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn test_trust_region_converges_with_inits_from_nca() {
    use ferx_core::{fit, EstimationMethod, FitOptions, NcaInit, Optimizer};

    let (mut model, population) = warfarin();

    // Deliberately set TVCL to the bad starting value.
    let tvcl_idx = model
        .default_params
        .theta_names
        .iter()
        .position(|n| n == "TVCL")
        .unwrap();
    model.default_params.theta[tvcl_idx] = 0.5;

    let mut options = FitOptions::default();
    options.method = EstimationMethod::FoceI;
    options.outer_maxiter = 300;
    options.inits_from_nca = Some(NcaInit::Sweep);
    options.optimizer = Optimizer::TrustRegion;
    options.run_covariance_step = false;

    let result =
        fit(&model, &population, &model.default_params, &options).expect("fit must not error");
    let ofv = result.ofv;
    assert!(
        ofv < 1000.0,
        "trust_region + inits_from_nca should converge (OFV < 1000), got {ofv:.2}"
    );
}
