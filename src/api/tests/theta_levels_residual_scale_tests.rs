//! The span certificate on a random effect that scales the residual error
//! (`iiv_on_ruv`, #1879). Such a random effect moves no prediction, so its
//! column in the predictions is zero and measures nothing; `spanned` compares it
//! on the predictions stacked with every record's log residual variance, and
//! the levels with it. Binder-only fixtures: FD of the f64 predictor and the
//! engine's residual variance, no fit.

use super::absorption::occ_pop;
use super::readout_share::T6;
use super::*;
use crate::parser::model_parser::LevelContrast;

/// One subject per study, six records each ([`population`]): every level of a
/// `[STUDY]` block lies within one subject.
fn one_study_per_subject() -> Population {
    population(3, 6)
}

/// A one-compartment bolus model with `ETA_RUV` on the residual error and a
/// block `ERRSCALE[cols]` under `contrast` (`""` = auto): `readout` the `y`
/// line, `error` the `DV ~` line.
fn ruv_model(cols: &str, contrast: &str, readout: &str, error: &str) -> String {
    let modifier = if contrast.is_empty() {
        String::new()
    } else {
        format!(", contrast = {contrast}")
    };
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 20.0)
  theta TVV(8.0, FIX)
  theta ERRSCALE[{cols}{modifier}](0.0, -5.0, 5.0)
  omega ETA_V ~ 0.04
  omega ETA_RUV ~ 0.09
  sigma PROP_ERR ~ 0.02 FIX
  sigma ADD_ERR ~ 0.5 FIX
[individual_parameters]
  CL = TVCL
  V = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  {readout}
[error_model]
  {error}
  iiv_on_ruv = ETA_RUV
[fit_options]
  method = focei
"#
    )
}

/// The plain readout.
const Y: &str = "y = central / V";

/// The block scales the whole residual SD, as `ETA_RUV` does: per record,
/// `∂ log V / ∂ERRSCALE = 2` on its level's records and `∂ log V / ∂ETA_RUV =
/// 2` on every record of the subject, and neither moves a prediction (closed
/// form).
const WHOLE: &str = "DV ~ proportional(PROP_ERR * exp(ERRSCALE))";

/// The block scales the additive part alone: `∂ log V / ∂ERRSCALE = 2 a² / (p²
/// f² + a²)` follows the prediction `f`, which falls over the records, so it is
/// not the constant `ETA_RUV` column.
const ADDITIVE_PART: &str = "DV ~ combined(PROP_ERR, ADD_ERR * exp(ERRSCALE))";

/// The measured clause for `ETA_RUV` on a `unit`.
fn residual_clause(unit: &str) -> String {
    format!(
        "the random effect `ETA_RUV` scales the residual error and moves no prediction, at every \
         record of each {unit}, exactly as some combination of this block's levels does"
    )
}

/// The measured clause's caveat.
const MEASURED_AT: &str = "(measured at the initial estimates and at a point near them)";

/// Bind `text` on a copy of `pop`: the resolved contrast of `ERRSCALE`, or the
/// refusal.
fn try_bind(text: &str, pop: &Population) -> Result<LevelContrast, String> {
    let mut p = pop.clone();
    let mut parsed = parse_full_model(text)?;
    crate::api::bind_theta_levels(&mut parsed, text, &mut p).map_err(|e| e.to_string())?;
    Ok(parsed.bindings.levels["ERRSCALE"].contrast)
}

/// R1 (#1879). One study per subject, a `[STUDY]` block scaling the whole
/// residual SD next to `ETA_RUV`: each subject's level and its `ETA_RUV` are
/// the same number, so the fit leaves `ETA_RUV`'s ω to the prior alone. Refused
/// under every contrast and auto, with the residual clause and never the
/// predictions one. The twin scales only the additive part of a combined error,
/// which follows the prediction and so is not `ETA_RUV`'s constant column: it
/// binds — both sides of the gate in one test, the straddle asserted.
///
/// Mutations — the random effect on the residual error measured on the
/// predictions alone, as before #1879 (the `WHOLE` rows bind); its column
/// without the `iiv_on_ruv` variance scale (zero, so it measures nothing: the
/// `WHOLE` rows bind); its own column, or its levels, taken from the
/// predictions channel (the `WHOLE` rows bind); the residual variance evaluated
/// off the prediction (the additive-part twin is refused); the clause's flag
/// ignored (says "moves `y`"); either half of "scales the residual error and
/// moves no prediction" deleted. Every random effect stacked, #1874's M18,
/// binds nothing here: `proportional_branches_are_measured`'s residual row dies.
#[test]
fn a_block_scaling_the_whole_residual_absorbs_an_iiv_on_ruv_eta() {
    let pop = one_study_per_subject();
    let mut wrong = Vec::new();
    for c in ["", "sum_to_zero", "ref", "none"] {
        match try_bind(&ruv_model("STUDY", c, Y, WHOLE), &pop) {
            Err(e) => {
                for must in [residual_clause("subject").as_str(), MEASURED_AT] {
                    if !e.contains(must) {
                        wrong.push(format!("{c:?}: lacks {must:?}: {e}"));
                    }
                }
                if e.contains("`ETA_RUV` moves `y`") {
                    wrong.push(format!("{c:?}: says ETA_RUV moves y: {e}"));
                }
            }
            Ok(b) => wrong.push(format!("{c:?}: bound {b:?}")),
        }
        if let Err(e) = try_bind(&ruv_model("STUDY", c, Y, ADDITIVE_PART), &pop) {
            wrong.push(format!("{c:?} additive part: refused: {e}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// R2 (#1879). A level that scales the prediction under a proportional error
/// scales the residual SD exactly as `ETA_RUV` does — `∂ log V / ∂ERRSCALE = 2`
/// at every record — but it moves the predictions too, which `ETA_RUV` does
/// not: the stacked channel keeps them apart, and the model binds. Each level
/// is a per-subject scale on the mean, and `ETA_RUV` its spread.
///
/// Mutation — the residual channel without the predictions above it: refused.
#[test]
fn a_level_scaling_the_prediction_does_not_absorb_it() {
    let pop = one_study_per_subject();
    for c in ["", "none"] {
        let text = ruv_model(
            "STUDY",
            c,
            "y = central / V * exp(ERRSCALE)",
            "DV ~ proportional(PROP_ERR)",
        );
        assert!(
            try_bind(&text, &pop).is_ok(),
            "{c:?}: {:?}",
            try_bind(&text, &pop)
        );
    }
}

/// R3 (#1879). Nested: `ERRSCALE[STUDY, OCC]` on one study per subject, three
/// occasions each, the block scaling the whole residual SD. `ETA_RUV` scales
/// every record of its subject alike, which is the sum of the subject's three
/// levels — a shift common to the group, not one that sums to zero within it.
/// So `sum_to_zero_within` leaves it identified and auto takes it, while every
/// contrast that lets the group's mean move is refused with the residual
/// clause, without the within-group note.
///
/// Mutation — the within residual taken against the predictions channel's
/// differences: equivalent here (the block moves no prediction, so those are
/// empty and leave `ETA_RUV` outside, as the stacked ones do).
#[test]
fn a_nested_block_binds_within_next_to_an_iiv_on_ruv_eta() {
    let pop = occ_pop(3, 1, &T6, 3, true);
    let text = |c: &str| ruv_model("STUDY, OCC", c, Y, WHOLE);
    assert_eq!(
        try_bind(&text(""), &pop),
        Ok(LevelContrast::SumToZeroWithin)
    );
    assert_eq!(
        try_bind(&text("sum_to_zero_within"), &pop),
        Ok(LevelContrast::SumToZeroWithin)
    );
    for c in ["ref", "none"] {
        let e = try_bind(&text(c), &pop).expect_err(c);
        assert!(e.contains(&residual_clause("subject")), "{c:?}: {e}");
        assert!(!e.contains("sums to zero within each group"), "{c:?}: {e}");
    }
}
