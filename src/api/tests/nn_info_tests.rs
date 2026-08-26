//! `FitResult.neural_networks` must describe the network well enough to *use* it.
//!
//! `ferx-r` round-trips this summary through `fit.json` inside the `.fitrx` archive and
//! reconstructs the network from it. Once `[covariate_nn]` gained `center` / `scale`, the
//! reported weights stopped being self-describing: they were fitted against
//! `(x − center) / scale`, and a consumer that evaluates them on raw covariates gets a
//! different network with no error to say so.

use super::build_neural_network_infos;
use crate::parser::model_parser::parse_model_string;

fn model_src(normalization: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVKA(1.0, 0.001, 100.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.04 (sd)

[covariate_nn TYPICAL_PK]
  inputs = [WT, CRCL]
  outputs = [CL, V]
  layers = [3]
  activation = tanh
  output = softplus
{normalization}

[individual_parameters]
  CL = TYPICAL_PK.CL * exp(ETA_CL)
  V  = TYPICAL_PK.V
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP)
"#
    )
}

/// The declared transform reaches the summary verbatim, in `input_names` order.
#[test]
fn neural_network_info_carries_the_declared_center_and_scale() {
    let model = parse_model_string(&model_src("  center = [70, 90]\n  scale  = [15, 30]"))
        .expect("normalised DCM parses");
    let infos = build_neural_network_infos(&model);
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].input_names, vec!["WT", "CRCL"]);
    assert_eq!(infos[0].input_center, vec![70.0, 90.0]);
    assert_eq!(infos[0].input_scale, vec![15.0, 30.0]);
}

/// A model that declares no normalisation reports the identity — not an empty vector, so a
/// consumer can apply `(x − center) / scale` unconditionally without a length check.
#[test]
fn neural_network_info_reports_identity_when_normalization_is_absent() {
    let model = parse_model_string(&model_src("")).expect("plain DCM parses");
    let infos = build_neural_network_infos(&model);
    assert_eq!(infos[0].input_center, vec![0.0, 0.0]);
    assert_eq!(infos[0].input_scale, vec![1.0, 1.0]);
}

/// A `.fitrx` bundle written before `center` / `scale` existed must still load.
///
/// `NeuralNetworkInfo` is `Serialize + Deserialize` and round-trips through `fit.json`
/// inside the archive, so adding a *required* field would have made every existing bundle
/// undeserialisable. Both are `serde(default)`; a legacy bundle predates normalization
/// entirely, so the empty vectors it yields correctly mean "no transform".
#[test]
fn neural_network_info_deserializes_a_bundle_written_before_normalization() {
    let legacy = r#"{
        "name": "TYPICAL_PK",
        "shape": [2, 3, 2],
        "hidden_activation": "tanh",
        "output_activation": "softplus",
        "n_weights": 17,
        "weights_offset": 1,
        "input_names": ["WT", "CRCL"],
        "output_names": ["CL", "V"]
    }"#;
    let info: crate::types::NeuralNetworkInfo =
        serde_json::from_str(legacy).expect("a pre-normalization bundle must still deserialize");
    assert!(info.input_center.is_empty());
    assert!(info.input_scale.is_empty());
}
