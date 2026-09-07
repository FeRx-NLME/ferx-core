//! Public-API integration tests for the covariate-NN (DCM) regularization
//! penalties: L2 (`nn_l2`) and smoothness/curvature (`nn_smooth`).
//!
//! Scope: this file drives the feature the way ferx-r does — set the two
//! `FitOptions` fields, call [`fit`], read the result. Nothing here touches a
//! crate-private item, so it doubles as the check that the feature is usable
//! from outside the crate at all.
//!
//! The mechanism-level coverage lives in `src/nn/mod.rs`, because it needs
//! `pub(crate)` internals (`NnRegularizer`, `CurvatureGrid`,
//! `MlpMapper::weight_param_indices`) that the workspace boundary rule keeps out
//! of the public API:
//!   - `nn::regularization_tests` — per-kernel L2 / curvature values, their
//!     analytic-vs-FD gradient and Hessian parity, and the grid living in the
//!     network's normalised input space.
//!   - `nn::regularizer_fit_tests` — the whole-`NnRegularizer` λ = 0 no-op, its
//!     FD gradient parity, and the fitted weight-norm / modulator-variance
//!     shrinkage (Tier 3, `slow-tests`).
//!   - `estimation::trust_region` — the assembled packed gradient (likelihood +
//!     penalty) against central FD of the penalized cost at λ > 0.
//!
//! Run via:
//!   cargo test --features nn --test nn_regularization

#![cfg(feature = "nn")]

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{CompiledModel, EstimationMethod, FitOptions, Population};
use ferx_core::{fit, read_nonmem_csv};

/// Two-cpt oral DCM with a small MLP mapping (WT, CRCL) → the five PK params;
/// one file shared with the `src/` tests so the fixtures cannot drift (see the
/// notes in it on why `center` / `scale` are load-bearing).
const MODEL: &str = include_str!("fixtures/two_cpt_dcm_regularized.ferx");

fn load() -> (CompiledModel, FitOptions, Population) {
    let parsed = parse_full_model(MODEL).expect("model parses with --features nn");
    let population = read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads");
    (parsed.model, parsed.fit_options, population)
}

/// A regularization-related warning a regularized FOCE-family fit must *not*
/// carry: the spurious "not used by method" one (the keys were once in no
/// `method_specific_keys` list) and the "not applied" one (reserved for non-FOCE
/// final stages).
fn regularization_warnings(result: &ferx_core::types::FitResult) -> Vec<&String> {
    result
        .warnings
        .iter()
        .filter(|w| w.contains("nn_l2") || w.contains("nn_smooth"))
        .collect()
}

/// A couple of outer iterations with both penalties on — proves the optimizer
/// wiring (penalized objective value + matching analytic gradient) runs end to
/// end from a public-API caller without panicking, returns a finite OFV, and
/// raises none of the regularization warnings.
#[test]
fn fit_with_regularization_runs() {
    let (model, mut options, population) = load();
    options.outer_maxiter = 2;
    options.nn_l2_lambda = 5e-2;
    options.nn_smooth_lambda = 5e-2;
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("regularized fit returns Ok");
    assert!(
        result.ofv.is_finite(),
        "regularized fit OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        regularization_warnings(&result).is_empty(),
        "a regularized FOCEI fit must not warn about nn_l2/nn_smooth: {:?}",
        result.warnings
    );
}

/// Same wiring under Gauss–Newton, which has its own objective / gradient /
/// BHHH assembly: the penalty must reach it (regression — `method = gn` once
/// ignored both keys silently).
#[test]
fn fit_with_regularization_runs_under_gn() {
    let (model, mut options, population) = load();
    options.method = EstimationMethod::FoceGn;
    options.outer_maxiter = 2;
    options.nn_l2_lambda = 5e-2;
    options.nn_smooth_lambda = 5e-2;
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("regularized GN fit returns Ok");
    assert!(
        result.ofv.is_finite(),
        "GN OFV must be finite, got {}",
        result.ofv
    );
    assert!(
        regularization_warnings(&result).is_empty(),
        "a regularized GN fit must not warn about nn_l2/nn_smooth: {:?}",
        result.warnings
    );
}

/// The penalties are applied by the FOCE-family methods only. Asking for them
/// with SAEM as the final stage — which ferx-r can do programmatically, never
/// touching the parser's key check — must say so up front instead of silently
/// returning an unregularized fit.
#[test]
fn non_foce_final_stage_warns_that_regularization_is_not_applied() {
    let (model, mut options, population) = load();
    options.method = EstimationMethod::Saem;
    options.saem_n_exploration = 2;
    options.saem_n_convergence = 1;
    options.nn_l2_lambda = 5e-2;
    // Any outcome is fine — the warning is assembled before the fit starts —
    // but a short SAEM run also has to succeed for the warning to be read back.
    let result = fit(&model, &population, &model.default_params, &options)
        .expect("short SAEM fit returns Ok");
    let hits: Vec<&String> = result
        .warnings
        .iter()
        .filter(|w| w.contains("does not apply covariate-NN regularization"))
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "expected exactly one 'not applied' warning, got {:?}",
        result.warnings
    );
    assert!(hits[0].contains("nn_l2 is set"), "{}", hits[0]);
}
