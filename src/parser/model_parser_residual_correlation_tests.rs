//! `correlation <kind>(RHO) by COL[, COL…]` in `[error_model]` (#1081): what it
//! compiles to, and every refusal, each pinned on its message.

use crate::parser::model_parser::parse_model_string;
use crate::types::{ResidualSerial, ResidualSerialKind};

/// A one-compartment model; `error` is the whole `[error_model]` body and `params`
/// is appended to `[parameters]`.
fn text(params: &str, indiv_extra: &str, error: &str) -> String {
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 10.0)\n  theta TVV(10.0, 0.1, 100.0)\n{params}  \
         omega ETA_CL ~ 0.09\n  sigma PROP_ERR ~ 0.2 (sd)\n  sigma ADD_ERR ~ 1.0 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV{indiv_extra}\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n{error}\n"
    )
}

const RHO: &str = "  theta RHO(0.3, -0.4, 0.95)\n";

fn err(params: &str, error: &str) -> String {
    match parse_model_string(&text(params, "", error)) {
        Ok(_) => panic!("expected a parse error for:\n{error}"),
        Err(e) => e,
    }
}

#[test]
fn a_correlation_line_compiles_to_its_structure() {
    for (kind, want) in [
        ("cs", ResidualSerialKind::Cs),
        ("AR1", ResidualSerialKind::Ar1),
        ("car1", ResidualSerialKind::Car1),
    ] {
        let m = parse_model_string(&text(
            RHO,
            "",
            &format!("  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation {kind}(RHO) by ARM  # arm"),
        ))
        .expect("parses");
        assert_eq!(
            m.residual_serial,
            Some(ResidualSerial {
                kind: want,
                rho_theta: 2,
                by: vec!["ARM".into()],
            })
        );
        // `by` columns are required data columns, and ρ counts as used.
        assert!(m.referenced_covariates.contains(&"ARM".to_string()));
        assert!(
            m.parse_warnings.iter().all(|w| !w.contains("RHO")),
            "{:?}",
            m.parse_warnings
        );
        assert!(m.residual_is_dense());
    }
}

#[test]
fn several_by_columns_parse_with_or_without_parentheses() {
    for by in ["(STUDY, ARM)", "STUDY, ARM", "STUDY,ARM"] {
        let m = parse_model_string(&text(
            RHO,
            "",
            &format!("  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(RHO) by {by}"),
        ))
        .expect("parses");
        assert_eq!(
            m.residual_serial.unwrap().by,
            vec!["STUDY".to_string(), "ARM".to_string()],
            "{by}"
        );
    }
}

/// `ρ` FIXed at 0 drops the line, as a `FIX` zero `block_sigma` off-diagonal is:
/// the model is the independent one. Any other FIX value keeps it.
#[test]
fn rho_fixed_at_zero_drops_the_line() {
    let line = "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(RHO) by ARM";
    let dropped = parse_model_string(&text("  theta RHO(0.0) FIX\n", "", line)).expect("parses");
    assert_eq!(dropped.residual_serial, None);
    assert!(!dropped.residual_is_dense());
    // still not reported unused: the user wrote it for the line
    assert!(
        dropped.parse_warnings.iter().all(|w| !w.contains("RHO")),
        "{:?}",
        dropped.parse_warnings
    );
    let kept = parse_model_string(&text("  theta RHO(0.2) FIX\n", "", line)).expect("parses");
    assert!(kept.residual_serial.is_some());
    let free_zero =
        parse_model_string(&text("  theta RHO(0.0, -0.4, 0.9)\n", "", line)).expect("parses");
    assert!(free_zero.residual_serial.is_some(), "a free ρ at 0 is kept");
}

#[test]
fn an_unknown_structure_lists_the_supported_ones() {
    let e = err(
        RHO,
        "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation car2(RHO) by ARM",
    );
    assert_eq!(
        e,
        "[error_model] `correlation car2(…)`: unknown structure `car2`. Supported: `cs`, `ar1`, \
         `car1`."
    );
}

#[test]
fn a_malformed_line_shows_the_expected_form() {
    for bad in [
        "correlation cs(RHO)",
        "correlation cs RHO by ARM",
        "correlation cs(RHO) by",
        "correlation cs(RHO, 0.5) by ARM",
    ] {
        let e = err(RHO, &format!("  DV ~ combined(PROP_ERR, ADD_ERR)\n  {bad}"));
        assert_eq!(
            e,
            format!(
                "[error_model]: cannot read `{bad}`. Expected `correlation <structure>(THETA) by \
                 COLUMN`, e.g. `correlation cs(RHO) by ARM`, where <structure> is one of: `cs`, \
                 `ar1`, `car1`."
            )
        );
    }
}

#[test]
fn a_second_correlation_line_is_refused() {
    let e = err(
        RHO,
        "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(RHO) by ARM\n  correlation ar1(RHO) by ARM",
    );
    assert_eq!(
        e,
        "[error_model] has more than one `correlation` line; a model takes one residual \
         correlation structure"
    );
}

#[test]
fn rho_must_be_a_declared_theta() {
    let e = err(
        RHO,
        "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(R0) by ARM",
    );
    assert_eq!(
        e,
        "[error_model] `correlation cs(R0)`: `R0` is not a theta declared in [parameters]. The \
         correlation is an ordinary theta, e.g. `theta R0(0.3, -0.2, 0.95)`."
    );
}

/// The ρ θ read by a model expression too would carry a second gradient channel.
#[test]
fn rho_used_elsewhere_is_refused() {
    let src = text(
        RHO,
        " * (1 + RHO)",
        "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(RHO) by ARM",
    );
    let e = parse_model_string(&src).expect_err("refused");
    assert_eq!(
        e,
        "[error_model] `correlation cs(RHO)`: theta `RHO` is also used by a model expression. \
         The correlation of a `correlation` line is a parameter of its own; declare a separate \
         theta for it."
    );
    // the same model without the second use parses
    assert!(parse_model_string(&src.replace(" * (1 + RHO)", "")).is_ok());
}

#[test]
fn per_cmt_and_selected_error_models_are_refused() {
    let line = "\n  correlation cs(RHO) by ARM";
    let e = err(RHO, &format!("  CMT=1: DV ~ additive(ADD_ERR){line}"));
    assert_eq!(
        e,
        "[error_model] `correlation cs(RHO)` is not supported with per-CMT (multi-endpoint) error \
         models: the correlation applies to one endpoint's rows."
    );
    let e = err(
        RHO,
        &format!(
            "  if (FREE == 0) {{ DV ~ additive(ADD_ERR) }} else {{ DV ~ additive(PROP_ERR) }}{line}"
        ),
    );
    assert_eq!(
        e,
        "[error_model] `correlation cs(RHO)` is not supported with a covariate-selected \
         (`if … else`) error model."
    );
}

#[test]
fn iiv_on_ruv_and_block_sigma_are_refused() {
    let line = "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation ar1(RHO) by ARM";
    let e = err(
        &format!("{RHO}  omega ETA_RUV ~ 0.1\n"),
        &format!("{line}\n  iiv_on_ruv = ETA_RUV"),
    );
    assert_eq!(
        e,
        "[error_model] `correlation ar1(RHO)` is not supported together with `iiv_on_ruv = \
         ETA_RUV`."
    );
    let src = text(
        RHO,
        "",
        "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation ar1(RHO) by ARM",
    )
    .replace(
        "  sigma PROP_ERR ~ 0.2 (sd)\n  sigma ADD_ERR ~ 1.0 (sd)\n",
        "  block_sigma (PROP_ERR, ADD_ERR) = [0.04, 0.01, 1.0]\n",
    );
    assert_eq!(
        parse_model_string(&src).expect_err("refused"),
        "[error_model] `correlation ar1(RHO)` is not supported in a model that also declares \
         `block_sigma`: the two are different residual correlations, and a model takes one."
    );
}

/// #1081 follow-up: a `by` column the model expressions do not read is not reported
/// as an undeclared covariate "used in model expressions" — its type changes
/// nothing. The same column read by an expression too still is. Both sides, and the
/// column is a required data column either way.
#[test]
fn a_by_column_alone_is_not_an_undeclared_expression_covariate() {
    let line = "  DV ~ combined(PROP_ERR, ADD_ERR)\n  correlation cs(RHO) by ARM";
    let with_block = |indiv_extra: &str| {
        parse_model_string(&format!(
            "{}[covariates]\n  WT continuous\n",
            text(RHO, indiv_extra, line)
        ))
        .expect("parses")
    };
    let undeclared = |m: &crate::types::CompiledModel| {
        m.parse_warnings
            .iter()
            .find(|w| w.starts_with("Covariate(s) used in model expressions"))
            .cloned()
    };
    let by_only = with_block("");
    assert_eq!(undeclared(&by_only), None, "{:?}", by_only.parse_warnings);
    assert!(by_only.referenced_covariates.contains(&"ARM".to_string()));
    let also_read = with_block(" * (1 + 0 * ARM)");
    let w = undeclared(&also_read).expect("an expression read is reported");
    assert!(w.contains(": ARM."), "{w}");
}
