//! `parse_error_to_diagnostic` mapping for the closed-world `[block]`-name
//! errors (#1040).
//!
//! A parse failure is terminal — `validate_model_file` returns before it has a
//! `ParsedModel` to read `block_lines` from — so the block and line in the
//! check report are recovered from the message text itself. These pin that
//! recovery, and the codes, against the exact strings `check_block_names`
//! emits (the parser side pins the strings themselves).

use super::parse_error_to_diagnostic;

#[test]
fn unknown_block_maps_to_e_unknown_block_with_block_line_and_suggestion() {
    let err = "Unknown block `[fit_option]` (line 16) — did you mean `[fit_options]`. \
               Valid blocks: adaptive_dosing, fit_options.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("fit_option"));
    assert_eq!(d.line, Some(16));
    assert_eq!(
        d.suggestion.as_deref(),
        Some("did you mean `[fit_options]`?")
    );
}

#[test]
fn unknown_block_with_no_near_match_carries_no_suggestion() {
    let err = "Unknown block `[qqqqqqqq]` (line 4). Valid blocks: parameters.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("qqqqqqqq"));
    assert_eq!(d.line, Some(4));
    assert!(d.suggestion.is_none());
}

#[test]
fn wrong_instance_name_form_maps_to_e_block_instance_name() {
    // Instance name on a block that takes none: `block` is the block *type*,
    // not `fit_options DOSE`, so it matches every other block-scoped diagnostic.
    let err = "Block `[fit_options DOSE]` (line 16) does not take an instance name \
               — write `[fit_options]`.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_BLOCK_INSTANCE_NAME");
    assert_eq!(d.block.as_deref(), Some("fit_options"));
    assert_eq!(d.line, Some(16));

    // The mirror case: a named-only block written without one.
    let err = "Block `[covariate_nn]` (line 9) requires an instance name \
               — write `[covariate_nn NAME]`.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_BLOCK_INSTANCE_NAME");
    assert_eq!(d.block.as_deref(), Some("covariate_nn"));
    assert_eq!(d.line, Some(9));
}

#[test]
fn disabled_feature_block_maps_to_e_block_feature_disabled() {
    let err = "Block `[event_model]` (line 16) requires building ferx-core with \
               `--features survival`.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_BLOCK_FEATURE_DISABLED");
    assert_eq!(d.block.as_deref(), Some("event_model"));
    assert_eq!(d.line, Some(16));
}

/// `[covariate_nn]` keeps its own, more specific code — the generic
/// feature-disabled branch must not swallow it.
#[test]
fn covariate_nn_feature_gate_keeps_its_dedicated_code() {
    let err = "[covariate_nn] blocks require building ferx-core with `--features nn`. \
               See plans/dcm-and-low-dim-node.md for the design and roadmap.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_NN_FEATURE_DISABLED");
}

/// Anything else still falls through to the catch-all.
#[test]
fn unrelated_parse_error_stays_e_parse() {
    let d = parse_error_to_diagnostic("Unknown PK model: one_cpt");
    assert_eq!(d.code, "E_PARSE");
    assert!(d.block.is_none());
    assert!(d.line.is_none());
}
