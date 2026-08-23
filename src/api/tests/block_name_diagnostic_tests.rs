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
    // The line moved into the structured field, so it is gone from the message:
    // the renderer already prints `fit_option:16` ahead of it.
    assert_eq!(
        d.message,
        "Unknown block `[fit_option]` — did you mean `[fit_options]`. \
         Valid blocks: adaptive_dosing, fit_options."
    );
}

/// With more than one offending header the structured `line` would have to pick
/// one and mislocate the rest, so none is attached and every parenthetical
/// stays in the message.
#[test]
fn several_unknown_blocks_keep_their_lines_in_the_message() {
    let err = "Unknown block `[scalings]` (line 16) — did you mean `[scaling]`; \
               `[outputs]` (line 20) — did you mean `[output]`. Valid blocks: output, scaling.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("scalings"));
    assert!(d.line.is_none());
    assert_eq!(d.message, err);
}

/// An unknown header written with an instance name is echoed as written, so the
/// text can be grepped for in the user's file — but `block` is still the type.
#[test]
fn unknown_named_block_keeps_its_instance_name_in_the_message() {
    let err = "Unknown block `[foo BAR]` (line 9). Valid blocks: parameters.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("foo"));
    assert_eq!(d.line, Some(9));
    assert!(d.message.contains("`[foo BAR]`"), "{}", d.message);
}

/// A malformed parenthetical is not a line: the block still transfers, the
/// message is left alone, and nothing is mis-parsed into `line`.
#[test]
fn unparseable_line_parenthetical_is_ignored() {
    let err = "Unknown block `[foo]` (line later). Valid blocks: parameters.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_UNKNOWN_BLOCK");
    assert_eq!(d.block.as_deref(), Some("foo"));
    assert!(d.line.is_none());
    assert_eq!(d.message, err);
}

#[test]
fn deprecated_block_maps_to_e_deprecated_block() {
    let err = "Deprecated block `[initial_values]` (line 25) is no longer read: \
               initial estimates are declared inline in `[parameters]` — delete it.";
    let d = parse_error_to_diagnostic(err);
    assert_eq!(d.code, "E_DEPRECATED_BLOCK");
    assert_eq!(d.block.as_deref(), Some("initial_values"));
    assert_eq!(d.line, Some(25));
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

/// `E_BLOCK_FEATURE_DISABLED` is matched on its own sentinel, not taken as the
/// `else` of the instance-name branch: a future ``Block `[…]`` message that is
/// neither shape must fall through to `E_PARSE` rather than be mis-coded.
#[test]
fn other_block_prefixed_errors_fall_through_to_e_parse() {
    let d = parse_error_to_diagnostic("Block `[odes]` (line 3) has something else wrong.");
    assert_eq!(d.code, "E_PARSE");
    assert!(d.block.is_none());
    assert!(d.line.is_none());
}

/// Anything else still falls through to the catch-all.
#[test]
fn unrelated_parse_error_stays_e_parse() {
    let d = parse_error_to_diagnostic("Unknown PK model: one_cpt");
    assert_eq!(d.code, "E_PARSE");
    assert!(d.block.is_none());
    assert!(d.line.is_none());
}

// ── parse-warning → check-report code (`parse_warning_to_code`) ─────────────

/// #1008: a declined absorption ODE twin reaches the check report under its own
/// code, not the generic `W_PARSE`.
///
/// Driven by the message the **parser actually produces** rather than a pasted
/// copy: the mapping keys on the `W_ABSORPTION_TWIN_DECLINED` token the warning
/// text carries, so a reworded warning that dropped the token would silently
/// fall through to `W_PARSE` — and a hand-written string here would not notice.
#[test]
fn declined_absorption_twin_warning_maps_to_its_own_code() {
    // A parameter named after the twin's own `central` state — the twin's parse
    // rejects the collision, so the model keeps no ODE fallback.
    let src = "
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVN(3.0, 1.0, 20.0)
  theta TVMTT(1.5, 0.01, 100.0)
  theta TVX(1.0, 0.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma EPS1 ~ 0.1 (sd)

[individual_parameters]
  CL  = TVCL * exp(ETA_CL)
  V   = TVV
  N   = TVN
  MTT = TVMTT
  CENTRAL = TVX

[structural_model]
  pk one_cpt_transit(cl=CL, v=V, n=N, mtt=MTT)

[error_model]
  DV ~ proportional(EPS1)
";
    let model = crate::parser::model_parser::parse_model_string(src).expect("the primary parses");
    let warning = model
        .parse_warnings
        .iter()
        .find(|w| w.contains("ODE equivalent could not be built"))
        .unwrap_or_else(|| {
            panic!(
                "the declined twin must warn; warnings: {:?}",
                model.parse_warnings
            )
        });
    assert_eq!(
        super::parse_warning_to_code(warning),
        "W_ABSORPTION_TWIN_DECLINED"
    );
}

/// The pre-existing arms and the catch-all, so the chain's ordering is pinned
/// rather than only its newest arm.
#[test]
fn parse_warning_codes_cover_their_arms_and_fall_through() {
    use super::parse_warning_to_code;
    assert_eq!(
        parse_warning_to_code("`TVX` is declared in [parameters] but not referenced anywhere"),
        "W_UNUSED_PARAM"
    );
    assert_eq!(
        parse_warning_to_code(
            "[derived] name `WT` shadows a covariate (W_DERIVED_COVARIATE_SHADOW)"
        ),
        "W_DERIVED_COVARIATE_SHADOW"
    );
    assert_eq!(
        parse_warning_to_code("[derived] `AUC`: step= ignored (W_DERIVED_STEP_IGNORED)"),
        "W_DERIVED_STEP_IGNORED"
    );
    assert_eq!(
        parse_warning_to_code("some unrelated parse-time note"),
        "W_PARSE"
    );
}

// ── #1001: single-endpoint sigma order ──────────────────────────────────────
//
// These two run the *parser* and feed it its own error string, rather than
// asserting a hand-written literal: the mapping keys off a prose substring that
// is split across a `\` line continuation at the emitting site in
// `build_error_spec`, so a copy pasted here could agree with itself while
// disagreeing with what the parser actually emits. Re-wrapping the message
// breaks the mapping, and these are what fail.

/// A mis-ordered single-endpoint `[error_model]` earns its own code, so a
/// consumer can offer the (mechanical) reorder without matching prose.
#[test]
fn sigma_order_mismatch_maps_to_its_own_code() {
    let parsed = crate::parser::model_parser::parse_full_model(
        "
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma S_BIG ~ 2.0 (sd)
  sigma S_SMALL ~ 0.05 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(S_SMALL)
",
    );
    let err = match parsed {
        Ok(_) => panic!("a mis-ordered single-endpoint [error_model] must be rejected"),
        Err(e) => e,
    };
    let d = parse_error_to_diagnostic(&err);
    assert_eq!(d.code, "E_SIGMA_ORDER_MISMATCH", "{err}");
    // The message already opens with `[error_model] `, and the renderer prefixes
    // whatever block it is given — setting both would print it twice.
    assert!(d.block.is_none(), "{:?}", d.block);
}

/// A *repeated* argument is deliberately not the same code: no reordering can
/// fix it, so it must not reach a consumer as the reorder-shaped
/// `E_SIGMA_ORDER_MISMATCH`. Pins the split, which is the whole reason the two
/// messages are worded differently.
#[test]
fn repeated_sigma_argument_is_not_coded_as_an_order_mismatch() {
    let parsed = crate::parser::model_parser::parse_full_model(
        "
[parameters]
  theta TVCL(0.2)
  theta TVV(10.0)
  omega ETA_CL ~ 0.09
  sigma S_A ~ 0.05 (sd)
  sigma S_B ~ 1.0 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ combined(S_A, S_A)
",
    );
    let err = match parsed {
        Ok(_) => panic!("a repeated sigma argument must be rejected"),
        Err(e) => e,
    };
    let d = parse_error_to_diagnostic(&err);
    assert_eq!(d.code, "E_PARSE", "{err}");
}
