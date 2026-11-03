//! `W_NN_BARE_OUTPUT_START` (#1561): a `[covariate_nn]` started with every
//! output-layer bias at 0, the parser's start when the block declares no `init`.
//!
//! The shared DCM fixture carries an `init` line, and
//! `crate::nn::dcm_fixture_without_init` is the same model without it, so every
//! test below has its straddle on one model that differs by a single line.

use crate::nn::dcm_fixture_without_init;
use crate::parser::model_parser::parse_full_model;
use crate::types::CompiledModel;

use super::check_nn_bare_output_start;

const DCM: &str = include_str!("../../../tests/fixtures/two_cpt_dcm_regularized.ferx");

fn model(text: &str) -> CompiledModel {
    parse_full_model(text).expect("DCM fixture parses").model
}

fn codes(diags: &[crate::diagnostics::Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.code.as_str()).collect()
}

/// The bare start warns and the declared one does not. Every clause of the
/// message is asserted: the block, the outputs it lists, the activation value
/// it quotes, and the `init` repair.
#[test]
fn bare_start_warns_and_declared_init_does_not() {
    let bare = model(&dcm_fixture_without_init());
    let diags = check_nn_bare_output_start(&bare, &bare.default_params.theta);
    assert_eq!(codes(&diags), ["W_NN_BARE_OUTPUT_START"]);
    let d = &diags[0];
    assert!(!d.is_error(), "a start is a warning, not a refusal");
    let m = &d.message;
    assert!(m.starts_with("W_NN_BARE_OUTPUT_START: "), "{m}");
    assert!(m.contains("[covariate_nn TYPICAL_PK]"), "{m}");
    assert!(m.contains("(CL, V1, Q, V2, KA)"), "{m}");
    assert!(m.contains("softplus(0) = 0.693"), "{m}");
    assert!(m.contains("still reports convergence"), "{m}");
    assert!(m.contains("Declare `init = [...]`"), "{m}");

    let declared = model(DCM);
    assert!(
        check_nn_bare_output_start(&declared, &declared.default_params.theta).is_empty(),
        "a declared `init` must stay quiet"
    );
}

/// The check reads the start it is handed, not the model's defaults: a warm start
/// from a fitted point (non-zero output biases) on the bare model is quiet. The
/// straddle moves one bias only, so a check that looked at `default_params`, or
/// that required *any* bias at 0 instead of all of them, fails here.
#[test]
fn warm_start_with_any_nonzero_output_bias_is_quiet() {
    let bare = model(&dcm_fixture_without_init());
    let nn = &bare.covariate_nns[0];
    let mlp = nn.mapper.mlp();
    let mut theta = bare.default_params.theta.clone();
    assert_eq!(check_nn_bare_output_start(&bare, &theta).len(), 1);
    theta[nn.weights_offset + mlp.output_bias_index(2)] = 1.5;
    assert!(
        check_nn_bare_output_start(&bare, &theta).is_empty(),
        "one non-zero output bias is a chosen start"
    );
}

/// Zero biases over a zeroed output-weight block is what a declared `init` whose
/// values all equal `activation(0)` produces: the start is chosen, so it is quiet.
/// The straddle is one live weight, which brings the warning back.
#[test]
fn zero_biases_over_a_zeroed_output_block_is_quiet() {
    let bare = model(&dcm_fixture_without_init());
    let nn = &bare.covariate_nns[0];
    let mlp = nn.mapper.mlp();
    let mut theta = bare.default_params.theta.clone();
    let range = mlp.output_weight_range();
    for i in range.clone() {
        theta[nn.weights_offset + i] = 0.0;
    }
    assert!(check_nn_bare_output_start(&bare, &theta).is_empty());
    theta[nn.weights_offset + range.start] = 0.1;
    assert_eq!(check_nn_bare_output_start(&bare, &theta).len(), 1);
}

/// `fit()` puts the warning on the result, read from the start it is given, and
/// `ferx check` reports it from the model alone.
#[test]
fn fit_and_ferx_check_both_report_it() {
    let population = crate::read_nonmem_csv(
        std::path::Path::new("data/two_cpt_oral_cov.csv"),
        Some(&["WT", "CRCL"]),
        None,
    )
    .expect("dataset loads");
    let warned = |text: &str| {
        let parsed = parse_full_model(text).expect("DCM fixture parses");
        let mut options = parsed.fit_options.clone();
        options.outer_maxiter = 0;
        let r = crate::fit(
            &parsed.model,
            &population,
            &parsed.model.default_params,
            &options,
        )
        .expect("maxiter = 0 fit runs");
        r.warnings
            .iter()
            .any(|w| w.starts_with("W_NN_BARE_OUTPUT_START"))
    };
    assert!(warned(&dcm_fixture_without_init()), "fit() must carry it");
    assert!(!warned(DCM), "fit() from a declared init must not");

    let checked = |text: &str| {
        use std::io::Write;
        let mut f = tempfile::Builder::new()
            .suffix(".ferx")
            .tempfile()
            .expect("create temp model");
        f.write_all(text.as_bytes()).expect("write temp model");
        f.flush().expect("flush temp model");
        let report = crate::api::validate_model_file(f.path().to_str().expect("utf-8"), None);
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "W_NN_BARE_OUTPUT_START")
    };
    assert!(
        checked(&dcm_fixture_without_init()),
        "ferx check must report it"
    );
    assert!(!checked(DCM), "ferx check on a declared init must not");
}
