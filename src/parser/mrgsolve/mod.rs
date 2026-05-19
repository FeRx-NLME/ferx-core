//! mrgsolve model-file front-end.
//!
//! Parses mrgsolve `.cpp` / `.mod` files into an intermediate `MrgModel`,
//! then either emits a `.ferx` source string (for the `ferx translate`
//! workflow) or lowers to a `CompiledModel` by round-tripping through the
//! existing `.ferx` parser.
//!
//! Supported blocks (MVP): `$PROB`, `$PARAM`, `$INPUT`, `$THETA`, `$CMT`,
//! `$INIT`, `$OMEGA`, `$SIGMA`, `$MAIN`/`$PK`, `$ODE`, `$TABLE`/`$ERROR`,
//! `$CAPTURE`, `$PKMODEL`, `$SET` (ignored).
//!
//! Hard-fail blocks: `$GLOBAL`, `$PLUGIN`, `$NMXML`, `$NMEXT`, `$PRED`,
//! `$EVENT`, `$PREAMBLE`, `$INCLUDE`. Hard-fail C++ constructs: `for`,
//! `while`, `return`, ternary `?:`, C-style casts, `pow(a, b)` calls.

pub(crate) mod block_extractor;
pub(crate) mod blocks;
pub(crate) mod cpp_preprocess;
pub(crate) mod emit_ferx;
pub(crate) mod error_pattern;
pub(crate) mod intermediate;
pub(crate) mod name_map;

use crate::parser::model_parser::parse_full_model;
use crate::types::ParsedModel;
use std::path::Path;

use block_extractor::extract;
use blocks::{dispatch_body, dispatch_declarative};
use intermediate::MrgModel;

/// Parse a mrgsolve `.cpp` / `.mod` source string into the intermediate
/// representation. `source_label` is used in error messages (typically the
/// file path).
fn parse_to_intermediate(content: &str, source_label: &str) -> Result<MrgModel, String> {
    let blocks = extract(content).map_err(|e| format!("{source_label}: {e}"))?;
    let mut m = MrgModel::default();
    if m.model_name.is_empty() {
        m.model_name = source_label.to_string();
    }
    for rb in &blocks {
        dispatch_declarative(rb, &mut m).map_err(|e| format!("{source_label}: {e}"))?;
    }
    for rb in &blocks {
        dispatch_body(rb, &mut m).map_err(|e| format!("{source_label}: {e}"))?;
    }
    Ok(m)
}

/// Read a `.cpp`/`.mod` file from disk, translate it to `.ferx` source, and
/// return the resulting string. This is the engine behind the `ferx translate`
/// subcommand.
pub fn translate_mrgsolve_file(path: &Path) -> Result<String, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {}", path.display(), e))?;
    let label = path.display().to_string();
    let m = parse_to_intermediate(&content, &label)?;
    emit_ferx::emit(&m)
}

/// Translate a mrgsolve source string to `.ferx` source. String-in /
/// string-out helper for callers that already have the contents in memory
/// (tests, the R wrapper).
pub fn translate_mrgsolve_string(content: &str, source_label: &str) -> Result<String, String> {
    let m = parse_to_intermediate(content, source_label)?;
    emit_ferx::emit(&m)
}

/// Read a `.cpp`/`.mod` file and parse it into a `ParsedModel` ready for the
/// rest of ferx (fit, simulate, predict). Internally translates to `.ferx`
/// text and feeds that through `parse_full_model` — same code path direct
/// `.ferx` users hit.
pub fn parse_mrgsolve_file(path: &Path) -> Result<ParsedModel, String> {
    let ferx_text = translate_mrgsolve_file(path)?;
    parse_full_model(&ferx_text)
}

/// String-in version of [`parse_mrgsolve_file`].
pub fn parse_mrgsolve_string(content: &str, source_label: &str) -> Result<ParsedModel, String> {
    let ferx_text = translate_mrgsolve_string(content, source_label)?;
    parse_full_model(&ferx_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WARFARIN_MOD: &str = "\
$PROB warfarin one-cpt oral
$PARAM TVCL = 0.2, TVV = 10.0, TVKA = 1.5
$CMT GUT CENT
$OMEGA @labels ETA_CL ETA_V ETA_KA
0.09 0.04 0.30
$SIGMA @labels PROP_ERR
0.02
$MAIN
double CL = TVCL * exp(ETA_CL);
double V  = TVV  * exp(ETA_V);
double KA = TVKA * exp(ETA_KA);
$PKMODEL ncmt=1, depot=TRUE
$TABLE
capture IPRED = CENT/V;
capture Y = IPRED * (1 + EPS(1));
";

    #[test]
    fn translate_warfarin_round_trip_parses_as_ferx() {
        // The translated .ferx text must parse cleanly via the ferx model parser.
        let ferx = translate_mrgsolve_string(WARFARIN_MOD, "<test>").unwrap();
        // Re-parse it via the existing ferx parser.
        let parsed = parse_full_model(&ferx).expect("translated .ferx parses cleanly");
        assert_eq!(parsed.model.theta_names, vec!["TVCL", "TVV", "TVKA"]);
        assert_eq!(parsed.model.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    }

    #[test]
    fn parse_mrgsolve_string_returns_compiled_model() {
        let parsed = parse_mrgsolve_string(WARFARIN_MOD, "<test>").unwrap();
        // Spot-check the structural model is one_cpt_oral.
        use crate::types::PkModel;
        assert!(matches!(parsed.model.pk_model, PkModel::OneCptOral));
    }
}
