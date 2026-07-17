use super::*;
use crate::types::PkParams;

/// #486: individual parameters the parser synthesized for a direct-θ/η Form-C
/// readout (`__ferx_ro_*`) are internal and must never appear in the user-facing
/// per-observation EBE map.
#[test]
fn synthetic_readout_params_hidden_from_indiv_map() {
    let mut pk = PkParams::default();
    pk.values[0] = 1.5; // real CL
    pk.values[2] = 3.0; // synthetic readout slot
    let names = vec!["CL".to_string(), "__ferx_ro_th0".to_string()];
    let pk_indices = vec![0usize, 2usize];
    let map = build_indiv_map(&pk, &names, &pk_indices);
    assert_eq!(map.len(), 1, "only the real parameter is exposed");
    assert_eq!(map.get("CL"), Some(&1.5));
    assert!(
        !map.keys()
            .any(|k| crate::parser::model_parser::is_synthetic_readout_param(k)),
        "synthetic readout params must be hidden from the EBE map"
    );
}
