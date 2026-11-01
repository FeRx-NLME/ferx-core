//! `W_NN_UNSCALED_INPUT` (#2549): a `[covariate_nn]` input whose data, after the block's
//! `center`/`scale`, saturate a hidden unit at the first layer's initial weight scale.
//! One model and one dataset; the pair differs only by the declared normalisation, so the
//! straddle is the `center`/`scale` lines alone.

use crate::parser::model_parser::parse_full_model;

use super::check_nn_input_scale;

fn prepared(text: &str) -> (crate::types::CompiledModel, crate::types::Population) {
    let model = parse_full_model(text).expect("DCM example parses").model;
    let pop = crate::read_nonmem_csv(std::path::Path::new("data/warfarin_dcm.csv"), None, None)
        .expect("data loads");
    (model, pop)
}

/// The raw example (WT ≈ 70, CRCL ≈ 86, no normalisation) warns and names both inputs; the
/// same block with the docs' `center`/`scale` does not.
#[test]
fn raw_inputs_warn_and_declared_normalisation_does_not() {
    let raw = std::fs::read_to_string("examples/warfarin_dcm.ferx").expect("example");
    let raw = raw.replace("  center     = [70, 86]\n  scale      = [12, 23]\n", "");
    let (m, pop) = prepared(&raw);
    let d = check_nn_input_scale(&m, &pop);
    assert_eq!(d.len(), 1, "{d:?}");
    let msg = &d[0].message;
    assert!(
        msg.starts_with("W_NN_UNSCALED_INPUT: [covariate_nn TYPICAL_PK]"),
        "{msg}"
    );
    assert!(msg.contains("WT (") && msg.contains("CRCL ("), "{msg}");
    assert!(msg.contains("tanh"), "{msg}");

    let normalised = raw.replace(
        "  inputs     = [WT, CRCL]\n",
        "  inputs     = [WT, CRCL]\n  center     = [70, 86]\n  scale      = [12, 23]\n",
    );
    assert_ne!(
        normalised, raw,
        "fixture: the normalisation lines were inserted"
    );
    let (m, pop) = prepared(&normalised);
    assert!(check_nn_input_scale(&m, &pop).is_empty());
}
