//! Emit a `.ferx` source string from a parsed `MrgModel`.
//!
//! The output is what `ferx translate file.mod` writes. It is also the
//! source the lowerer hands to `parser::model_parser::parse_full_model` to
//! produce a `CompiledModel` — going through `.ferx` text gives us the same
//! battle-tested compilation pipeline that direct `.ferx` users hit.

use super::error_pattern::{classify, ClassifiedError};
use super::intermediate::{MrgModel, OmegaBlock, PkModelSpec};
use super::name_map::NameMap;
use std::fmt::Write;

/// Emit a full `.ferx` source string from `m`. Returns `Err` if any block
/// can't be translated (e.g. unsupported error-model shape).
pub(crate) fn emit(m: &MrgModel) -> Result<String, String> {
    let mut out = String::new();
    write_header(m, &mut out);
    write_parameters(m, &mut out);
    write_individual_parameters(m, &mut out)?;
    write_structural_model(m, &mut out)?;
    write_error_model(m, &mut out)?;
    write_fit_options_defaults(&mut out);
    Ok(out)
}

fn write_header(m: &MrgModel, out: &mut String) {
    if !m.model_name.is_empty() {
        writeln!(out, "# {}", m.model_name).unwrap();
    }
    writeln!(out, "# Translated from mrgsolve by `ferx translate`.").unwrap();
    writeln!(out).unwrap();
}

fn write_parameters(m: &MrgModel, out: &mut String) {
    writeln!(out, "[parameters]").unwrap();
    for t in &m.thetas {
        writeln!(out, "  theta {}({})", t.name, fmt_f64(t.value)).unwrap();
    }
    if !m.thetas.is_empty() && !(m.omegas.is_empty() && m.sigmas.is_empty()) {
        writeln!(out).unwrap();
    }
    for block in &m.omegas {
        write_omega_or_sigma(block, "omega", "block_omega", out);
    }
    if !m.omegas.is_empty() && !m.sigmas.is_empty() {
        writeln!(out).unwrap();
    }
    for block in &m.sigmas {
        write_omega_or_sigma(block, "sigma", "block_sigma", out);
    }
    writeln!(out).unwrap();
}

fn write_omega_or_sigma(b: &OmegaBlock, kw_diag: &str, kw_block: &str, out: &mut String) {
    if b.block_form {
        let names = b.labels.join(", ");
        let vals: Vec<String> = b.values.iter().map(|v| fmt_f64(*v)).collect();
        writeln!(
            out,
            "  {kw_block} ({}) = [{}]",
            names,
            vals.join(", ")
        )
        .unwrap();
    } else {
        for (label, value) in b.labels.iter().zip(b.values.iter()) {
            writeln!(out, "  {kw_diag} {} ~ {}", label, fmt_f64(*value)).unwrap();
        }
    }
}

fn write_individual_parameters(m: &MrgModel, out: &mut String) -> Result<(), String> {
    let Some(body) = m.main_body.as_ref() else {
        return Ok(()); // some models (pure-ODE simulation) skip $MAIN
    };
    writeln!(out, "[individual_parameters]").unwrap();
    for line in body.text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        writeln!(out, "  {}", trimmed).unwrap();
    }
    writeln!(out).unwrap();
    Ok(())
}

fn write_structural_model(m: &MrgModel, out: &mut String) -> Result<(), String> {
    if let Some(pk) = m.pkmodel.as_ref() {
        write_analytical_pk(pk, out)?;
    } else if let Some(ode_body) = m.ode_body.as_ref() {
        // We synthesise an `ode(...)` directive listing every compartment
        // as a state, with the first `$CAPTURE` entry (or `CENT` if it
        // exists) as the observable.
        let states: Vec<&str> = m.compartments.iter().map(|c| c.name.as_str()).collect();
        if states.is_empty() {
            return Err(
                "model has $ODE but no $CMT/$INIT — declare compartments before $ODE".to_string(),
            );
        }
        let obs_cmt = m
            .compartments
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case("CENT"))
            .map(|c| c.name.clone())
            .unwrap_or_else(|| states[0].to_string());

        writeln!(out, "[structural_model]").unwrap();
        writeln!(
            out,
            "  ode(obs_cmt={}, states=[{}])",
            obs_cmt,
            states.join(", ")
        )
        .unwrap();
        writeln!(out).unwrap();

        writeln!(out, "[odes]").unwrap();
        for line in ode_body.text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            writeln!(out, "  {}", trimmed).unwrap();
        }
        writeln!(out).unwrap();
    } else {
        return Err(
            "model has neither $PKMODEL nor $ODE — at least one is required".to_string(),
        );
    }
    Ok(())
}

fn write_analytical_pk(pk: &PkModelSpec, out: &mut String) -> Result<(), String> {
    let model_name = match (pk.ncmt, pk.depot) {
        (1, true) => "one_cpt_oral",
        (1, false) => "one_cpt_iv_bolus",
        (2, true) => "two_cpt_oral",
        (2, false) => "two_cpt_iv_bolus",
        (3, true) => "three_cpt_oral",
        (3, false) => "three_cpt_iv_bolus",
        (n, _) => return Err(format!("$PKMODEL ncmt={n} is out of range (1-3)")),
    };
    let params: Vec<&str> = match (pk.ncmt, pk.depot) {
        (1, true) => vec!["cl=CL", "v=V", "ka=KA"],
        (1, false) => vec!["cl=CL", "v=V"],
        (2, true) => vec!["cl=CL", "v=V", "q=Q", "v2=V2", "ka=KA"],
        (2, false) => vec!["cl=CL", "v=V", "q=Q", "v2=V2"],
        (3, true) => vec!["cl=CL", "v=V", "q=Q", "v2=V2", "q2=Q2", "v3=V3", "ka=KA"],
        (3, false) => vec!["cl=CL", "v=V", "q=Q", "v2=V2", "q2=Q2", "v3=V3"],
        _ => unreachable!(),
    };
    writeln!(out, "[structural_model]").unwrap();
    writeln!(out, "  pk {}({})", model_name, params.join(", ")).unwrap();
    writeln!(out).unwrap();
    Ok(())
}

fn write_error_model(m: &MrgModel, out: &mut String) -> Result<(), String> {
    let Some(err_body) = m.error_body.as_ref() else {
        return Err(
            "model has no $TABLE / $ERROR block — ferx requires a residual error model"
                .to_string(),
        );
    };
    let name_map = NameMap::from_model(m);
    let classified = classify(&err_body.text, &name_map)?;
    writeln!(out, "[error_model]").unwrap();
    match classified {
        ClassifiedError::Proportional { sigma } => {
            writeln!(out, "  DV ~ proportional({sigma})").unwrap();
        }
        ClassifiedError::Additive { sigma } => {
            writeln!(out, "  DV ~ additive({sigma})").unwrap();
        }
        ClassifiedError::Combined {
            sigma_prop,
            sigma_add,
        } => {
            writeln!(out, "  DV ~ combined({sigma_prop}, {sigma_add})").unwrap();
        }
    }
    writeln!(out).unwrap();
    Ok(())
}

fn write_fit_options_defaults(out: &mut String) {
    writeln!(out, "[fit_options]").unwrap();
    writeln!(out, "  method     = focei").unwrap();
    writeln!(out, "  maxiter    = 300").unwrap();
    writeln!(out, "  covariance = true").unwrap();
}

/// Pretty-print an f64 without trailing zeros on integers.
fn fmt_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{}", v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mrgsolve::block_extractor::extract;
    use crate::parser::mrgsolve::blocks::{dispatch_body, dispatch_declarative};

    fn parse(src: &str) -> MrgModel {
        let blocks = extract(src).unwrap();
        let mut m = MrgModel::default();
        for rb in &blocks {
            dispatch_declarative(rb, &mut m).unwrap();
        }
        for rb in &blocks {
            dispatch_body(rb, &mut m).unwrap();
        }
        m
    }

    #[test]
    fn emits_warfarin_one_cpt_oral() {
        let src = "\
$PROB warfarin demo
$PARAM TVCL = 0.2, TVV = 10, TVKA = 1.5
$CMT GUT CENT
$OMEGA @labels ETA_CL ETA_V ETA_KA
0.09 0.04 0.30
$SIGMA @labels PROP_ERR
0.02
$MAIN
double CL = TVCL * exp(ETA(1));
double V  = TVV  * exp(ETA(2));
double KA = TVKA * exp(ETA(3));
$PKMODEL ncmt=1, depot=TRUE
$TABLE
capture IPRED = CENT/V;
capture Y = IPRED * (1 + EPS(1));
";
        let m = parse(src);
        let out = emit(&m).unwrap();
        assert!(out.contains("# warfarin demo"));
        assert!(out.contains("theta TVCL(0.2)"));
        assert!(out.contains("theta TVV(10)"));
        assert!(out.contains("omega ETA_CL ~ 0.09"));
        assert!(out.contains("sigma PROP_ERR ~ 0.02"));
        assert!(out.contains("CL = TVCL * exp(ETA_CL)"));
        assert!(out.contains("pk one_cpt_oral(cl=CL, v=V, ka=KA)"));
        assert!(out.contains("DV ~ proportional(PROP_ERR)"));
        assert!(out.contains("method     = focei"));
    }

    #[test]
    fn emits_block_omega() {
        let src = "\
$PARAM TVCL=1
$OMEGA @block @labels A B
0.1 0.05 0.2
$SIGMA 0.02
$MAIN
double CL = TVCL * exp(A);
$PKMODEL ncmt=1, depot=FALSE
$TABLE
Y = CL * (1 + EPS(1));
";
        let m = parse(src);
        let out = emit(&m).unwrap();
        assert!(out.contains("block_omega (A, B) = [0.1, 0.05, 0.2]"));
        assert!(out.contains("pk one_cpt_iv_bolus(cl=CL, v=V)"));
    }

    #[test]
    fn rejects_missing_error_block() {
        let src = "\
$PARAM TVCL=1
$OMEGA @labels A
0.1
$MAIN
double CL = TVCL * exp(A);
$PKMODEL ncmt=1, depot=FALSE
";
        let m = parse(src);
        let err = emit(&m).unwrap_err();
        assert!(err.contains("no $TABLE / $ERROR"));
    }
}
