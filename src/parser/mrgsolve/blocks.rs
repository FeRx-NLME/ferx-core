//! Per-block parsers. Consume `RawBlock`s from `block_extractor` and
//! populate fields on a `MrgModel`.
//!
//! Supported blocks (MVP):
//! - `$PROB` → `model_name`
//! - `$PARAM`, `$THETA` → `thetas`
//! - `$INPUT` → `covariates`
//! - `$OMEGA`, `$SIGMA` (diagonal + `@block` + `@labels`)
//! - `$CMT`, `$INIT`
//! - `$MAIN` / `$PK` → `main_body`
//! - `$ODE` → `ode_body`
//! - `$TABLE` / `$ERROR` → `error_body`
//! - `$CAPTURE` → `captures`
//! - `$PKMODEL` → `pkmodel`
//! - `$SET` (parsed and ignored)
//!
//! Hard-fail blocks: `$GLOBAL`, `$PLUGIN`, `$NMXML`, `$NMEXT`, `$PRED`,
//! `$EVENT`, `$PREAMBLE`, `$INCLUDE` (each returns a clear error with the
//! source line).

use super::block_extractor::{BodyLine, RawBlock};
use super::cpp_preprocess;
use super::intermediate::{
    Compartment, MrgModel, NamedValue, OmegaBlock, PkModelSpec, PreprocessedBody,
};
use super::name_map::NameMap;

/// Dispatch a single `RawBlock` to its parser. Mutates `m` in place.
///
/// Some blocks (`$MAIN`, `$ODE`, `$TABLE`) need a `NameMap` to resolve
/// `ETA(n)`/`THETA(n)` references. Pass `None` here on the first declarative
/// pass and re-walk those bodies after the declarative blocks have been
/// processed; see `parser::mrgsolve::parse_to_intermediate` for the two-pass
/// orchestration.
pub(crate) fn dispatch_declarative(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    match rb.name.as_str() {
        "PROB" => parse_prob(rb, m),
        "PARAM" => parse_param(rb, m),
        "THETA" => parse_theta(rb, m),
        "INPUT" => parse_input(rb, m),
        "OMEGA" => parse_omega(rb, m, OmegaKind::Omega),
        "SIGMA" => parse_omega(rb, m, OmegaKind::Sigma),
        "CMT" | "INIT" => parse_cmt(rb, m),
        "CAPTURE" => parse_capture(rb, m),
        "PKMODEL" => parse_pkmodel(rb, m),
        "SET" | "PROB_DOC" => Ok(()), // ignored
        // Bodies — handled in pass 2 once names are resolved.
        "MAIN" | "PK" | "ODE" | "TABLE" | "ERROR" => Ok(()),
        // Hard-fail unsupported blocks.
        "GLOBAL" | "PLUGIN" | "NMXML" | "NMEXT" | "PRED" | "EVENT" | "PREAMBLE" | "INCLUDE" => {
            Err(unsupported_block(rb))
        }
        other => Err(format!(
            "line {}: unknown block `${other}`",
            rb.start_line
        )),
    }
}

/// Second pass — process body blocks (`$MAIN`, `$ODE`, `$TABLE`/`$ERROR`).
/// Requires the declarative pass to have populated `thetas`/`omegas`/`sigmas`
/// so `NameMap::from_model` can resolve `ETA(n)`/`THETA(n)`.
pub(crate) fn dispatch_body(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let name_map = NameMap::from_model(m);
    match rb.name.as_str() {
        "MAIN" | "PK" => parse_body_into(rb, &name_map, &mut m.main_body),
        "ODE" => parse_body_into(rb, &name_map, &mut m.ode_body),
        "TABLE" | "ERROR" => parse_body_into(rb, &name_map, &mut m.error_body),
        _ => Ok(()), // declarative pass already handled it
    }
}

fn unsupported_block(rb: &RawBlock) -> String {
    format!(
        "line {}: `${}` is not supported by ferx",
        rb.start_line, rb.name
    )
}

// ── $PROB ──────────────────────────────────────────────────────────────────

fn parse_prob(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    // The "name" is the rest-of-line arguments; if missing, leave whatever
    // the caller already set (typically the source filename stem).
    if !rb.header_args.is_empty() {
        m.model_name = rb.header_args.clone();
    }
    Ok(())
}

// ── $PARAM ─────────────────────────────────────────────────────────────────

fn parse_param(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let joined = join_with_header(rb);
    for entry in split_decl_list(&joined) {
        let (name, value) = split_eq(&entry).ok_or_else(|| {
            format!(
                "line {}: `$PARAM` entry `{}` must be `NAME = value`",
                rb.start_line, entry
            )
        })?;
        let value: f64 = value
            .parse()
            .map_err(|_| format!("line {}: cannot parse `$PARAM {}` value", rb.start_line, name))?;
        m.thetas.push(NamedValue {
            name: name.to_string(),
            value,
        });
    }
    Ok(())
}

// ── $THETA ─────────────────────────────────────────────────────────────────

fn parse_theta(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    // Two supported forms:
    //   $THETA 1.0 2.0 3.0           (positional; auto-name THETA1, THETA2 ...)
    //   $THETA name=1.0, name=2.0    (named; same shape as $PARAM)
    let joined = join_with_header(rb);
    let tokens: Vec<String> = split_decl_list(&joined);
    if tokens.is_empty() {
        return Ok(());
    }
    let all_have_eq = tokens.iter().all(|t| t.contains('='));
    let none_have_eq = tokens.iter().all(|t| !t.contains('='));
    if all_have_eq {
        for entry in tokens {
            let (name, value) = split_eq(&entry).unwrap();
            let value: f64 = value
                .parse()
                .map_err(|_| format!("line {}: bad value in `$THETA {}`", rb.start_line, name))?;
            m.thetas.push(NamedValue {
                name: name.to_string(),
                value,
            });
        }
    } else if none_have_eq {
        let base = m.thetas.len() + 1;
        for (i, value_str) in tokens.into_iter().enumerate() {
            let value: f64 = value_str.parse().map_err(|_| {
                format!("line {}: bad value in `$THETA`: {}", rb.start_line, value_str)
            })?;
            m.thetas.push(NamedValue {
                name: format!("THETA{}", base + i),
                value,
            });
        }
    } else {
        return Err(format!(
            "line {}: `$THETA` mixes named and positional entries",
            rb.start_line
        ));
    }
    Ok(())
}

// ── $INPUT ─────────────────────────────────────────────────────────────────

fn parse_input(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let joined = join_with_header(rb);
    for entry in split_decl_list(&joined) {
        // `$INPUT WT = 70` declares a covariate with a default — but the
        // default is ignored by ferx (covariates come from data only).
        let name = entry.split('=').next().unwrap().trim();
        if name.is_empty() {
            continue;
        }
        m.covariates.push(name.to_string());
    }
    Ok(())
}

// ── $OMEGA / $SIGMA ────────────────────────────────────────────────────────

enum OmegaKind {
    Omega,
    Sigma,
}

fn parse_omega(rb: &RawBlock, m: &mut MrgModel, kind: OmegaKind) -> Result<(), String> {
    // Header may carry `@block`, `@labels NAME1 NAME2`, or numeric values.
    // Body lines continue the numeric values. Both header and body are
    // whitespace/comma separated.
    let kind_name = match kind {
        OmegaKind::Omega => "OMEGA",
        OmegaKind::Sigma => "SIGMA",
    };
    let mut block_form = false;
    let mut labels: Vec<String> = Vec::new();
    let mut values: Vec<f64> = Vec::new();

    // Walk tokens from the header and then from each body line.
    let mut tokens: Vec<String> = split_decl_list(&rb.header_args);
    for line in &rb.body_lines {
        tokens.extend(split_decl_list(&line.text));
    }

    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if t.eq_ignore_ascii_case("@block") {
            block_form = true;
            i += 1;
        } else if t.eq_ignore_ascii_case("@labels") {
            i += 1;
            // Consume idents until next `@directive` or numeric token.
            while i < tokens.len() {
                let next = &tokens[i];
                if next.starts_with('@') || next.parse::<f64>().is_ok() {
                    break;
                }
                labels.push(next.clone());
                i += 1;
            }
        } else if t.starts_with('@') {
            return Err(format!(
                "line {}: `${kind_name}` directive `{}` is not supported",
                rb.start_line, t
            ));
        } else {
            let v: f64 = t.parse().map_err(|_| {
                format!(
                    "line {}: `${kind_name}` value `{}` is not a number",
                    rb.start_line, t
                )
            })?;
            values.push(v);
            i += 1;
        }
    }

    let dim = if block_form {
        // n(n+1)/2 = values.len() → solve for n.
        let len = values.len();
        let n = (((1.0 + 8.0 * len as f64).sqrt() - 1.0) / 2.0).round() as usize;
        if n * (n + 1) / 2 != len {
            return Err(format!(
                "line {}: `${kind_name} @block` has {} values, which is not n(n+1)/2 for any n",
                rb.start_line, len
            ));
        }
        n
    } else {
        values.len()
    };

    if !labels.is_empty() && labels.len() != dim {
        return Err(format!(
            "line {}: `${kind_name} @labels` declares {} names but the block has dimension {}",
            rb.start_line,
            labels.len(),
            dim
        ));
    }
    if labels.is_empty() {
        // Auto-generate names with sequential numbering across blocks.
        let existing = match kind {
            OmegaKind::Omega => m.omegas.iter().map(|b| b.dim()).sum::<usize>(),
            OmegaKind::Sigma => m.sigmas.iter().map(|b| b.dim()).sum::<usize>(),
        };
        let prefix = match kind {
            OmegaKind::Omega => "ETA",
            OmegaKind::Sigma => "EPS",
        };
        labels = (1..=dim).map(|i| format!("{prefix}{}", existing + i)).collect();
    }

    let block = OmegaBlock {
        block_form,
        labels,
        values,
    };
    match kind {
        OmegaKind::Omega => m.omegas.push(block),
        OmegaKind::Sigma => m.sigmas.push(block),
    }
    Ok(())
}

// ── $CMT / $INIT ───────────────────────────────────────────────────────────

fn parse_cmt(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let joined = join_with_header(rb);
    for entry in split_decl_list(&joined) {
        match split_eq(&entry) {
            Some((name, value)) => {
                let initial: f64 = value.parse().map_err(|_| {
                    format!(
                        "line {}: `$CMT/$INIT` value `{}` is not a number",
                        rb.start_line, value
                    )
                })?;
                m.compartments.push(Compartment {
                    name: name.to_string(),
                    initial,
                });
            }
            None => {
                m.compartments.push(Compartment {
                    name: entry.to_string(),
                    initial: 0.0,
                });
            }
        }
    }
    Ok(())
}

// ── $CAPTURE ───────────────────────────────────────────────────────────────

fn parse_capture(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let joined = join_with_header(rb);
    for entry in split_decl_list(&joined) {
        // Strip trailing `=ALIAS` since ferx exposes every named variable
        // already; aliasing is a cosmetic mrgsolve feature.
        let name = entry.split('=').next().unwrap().trim();
        if !name.is_empty() {
            m.captures.push(name.to_string());
        }
    }
    Ok(())
}

// ── $PKMODEL ───────────────────────────────────────────────────────────────

fn parse_pkmodel(rb: &RawBlock, m: &mut MrgModel) -> Result<(), String> {
    let joined = join_with_header(rb);
    let mut ncmt: Option<u8> = None;
    let mut depot: Option<bool> = None;
    let mut cmt_names: Option<Vec<String>> = None;
    for entry in split_decl_list(&joined) {
        let (key, val) = split_eq(&entry).ok_or_else(|| {
            format!(
                "line {}: `$PKMODEL` argument `{}` must be `key=value`",
                rb.start_line, entry
            )
        })?;
        match key.to_ascii_lowercase().as_str() {
            "ncmt" => {
                let n: u8 = val.parse().map_err(|_| {
                    format!("line {}: `$PKMODEL ncmt={}` is invalid", rb.start_line, val)
                })?;
                if !(1..=3).contains(&n) {
                    return Err(format!(
                        "line {}: `$PKMODEL ncmt={n}` is out of range (1-3 supported)",
                        rb.start_line
                    ));
                }
                ncmt = Some(n);
            }
            "depot" => {
                depot = Some(match val.to_ascii_uppercase().as_str() {
                    "TRUE" | "T" | "1" => true,
                    "FALSE" | "F" | "0" => false,
                    _ => {
                        return Err(format!(
                            "line {}: `$PKMODEL depot={}` must be TRUE or FALSE",
                            rb.start_line, val
                        ));
                    }
                });
            }
            "cmt" => {
                let stripped = val.trim_matches('"').trim_matches('\'');
                let names: Vec<String> = split_decl_list(stripped);
                cmt_names = Some(names);
            }
            other => {
                return Err(format!(
                    "line {}: `$PKMODEL` argument `{}` is not supported",
                    rb.start_line, other
                ));
            }
        }
    }
    let ncmt = ncmt.ok_or_else(|| {
        format!("line {}: `$PKMODEL` requires `ncmt=`", rb.start_line)
    })?;
    let depot = depot.unwrap_or(false);
    m.pkmodel = Some(PkModelSpec {
        ncmt,
        depot,
        cmt_names,
    });
    Ok(())
}

// ── Body blocks ($MAIN/$ODE/$TABLE) ────────────────────────────────────────

fn parse_body_into(
    rb: &RawBlock,
    name_map: &NameMap,
    slot: &mut Option<PreprocessedBody>,
) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!(
            "line {}: `${}` declared more than once",
            rb.start_line, rb.name
        ));
    }
    let raw = body_text(&rb.body_lines);
    let processed = cpp_preprocess::preprocess(&raw, name_map, rb.start_line)?;
    *slot = Some(PreprocessedBody {
        start_line: rb.start_line,
        text: processed,
    });
    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn join_with_header(rb: &RawBlock) -> String {
    let mut s = rb.header_args.clone();
    for line in &rb.body_lines {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(&line.text);
    }
    s
}

fn body_text(lines: &[BodyLine]) -> String {
    let mut out = String::new();
    for l in lines {
        out.push_str(&l.text);
        out.push('\n');
    }
    out
}

/// Split a declaration list on commas + whitespace, keeping `NAME = value`
/// together as one token. Quoted strings (`"..."` and `'...'`) survive
/// splitting so things like `cmt="CENT,PERIPH"` remain a single token.
///
/// Returns owned `String`s because we need to normalise the `\s*=\s*` to
/// `=` before splitting (so `TVCL = 0.2` ends up as one token).
fn split_decl_list(s: &str) -> Vec<String> {
    let normalized = EQ_SPACES.replace_all(s, "=");
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_str: Option<char> = None;
    for c in normalized.chars() {
        match (in_str, c) {
            (Some(q), c) if c == q => {
                in_str = None;
                cur.push(c);
            }
            (None, '"') | (None, '\'') => {
                in_str = Some(c);
                cur.push(c);
            }
            (None, ',') | (None, ' ') | (None, '\t') | (None, '\n') | (None, '\r') => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn split_eq(entry: &str) -> Option<(&str, &str)> {
    entry.split_once('=').map(|(k, v)| (k.trim(), v.trim()))
}

static EQ_SPACES: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"\s*=\s*").unwrap());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mrgsolve::block_extractor::extract;

    fn parse_blocks(src: &str) -> MrgModel {
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
    fn parses_basic_param() {
        let m = parse_blocks("$PARAM TVCL = 0.2, TVV = 10\n");
        assert_eq!(m.thetas.len(), 2);
        assert_eq!(m.thetas[0].name, "TVCL");
        assert_eq!(m.thetas[0].value, 0.2);
        assert_eq!(m.thetas[1].name, "TVV");
        assert_eq!(m.thetas[1].value, 10.0);
    }

    #[test]
    fn parses_omega_with_labels() {
        let m = parse_blocks("$OMEGA @labels ETA_CL ETA_V\n0.09 0.04\n");
        assert_eq!(m.omegas.len(), 1);
        assert_eq!(m.omegas[0].labels, vec!["ETA_CL", "ETA_V"]);
        assert_eq!(m.omegas[0].values, vec![0.09, 0.04]);
        assert!(!m.omegas[0].block_form);
    }

    #[test]
    fn omega_block_form() {
        let m = parse_blocks("$OMEGA @block @labels A B\n0.1 0.05 0.2\n");
        assert!(m.omegas[0].block_form);
        assert_eq!(m.omegas[0].labels, vec!["A", "B"]);
        assert_eq!(m.omegas[0].values, vec![0.1, 0.05, 0.2]);
    }

    #[test]
    fn omega_auto_labels() {
        let m = parse_blocks("$OMEGA 0.04 0.09\n$OMEGA 0.16\n");
        assert_eq!(m.omegas[0].labels, vec!["ETA1", "ETA2"]);
        assert_eq!(m.omegas[1].labels, vec!["ETA3"]);
    }

    #[test]
    fn sigma_auto_labels() {
        let m = parse_blocks("$SIGMA 0.02\n");
        assert_eq!(m.sigmas[0].labels, vec!["EPS1"]);
    }

    #[test]
    fn parses_cmt_and_init() {
        let m = parse_blocks("$CMT GUT CENT\n$INIT RESP = 25\n");
        assert_eq!(m.compartments.len(), 3);
        assert_eq!(m.compartments[0].name, "GUT");
        assert_eq!(m.compartments[0].initial, 0.0);
        assert_eq!(m.compartments[2].name, "RESP");
        assert_eq!(m.compartments[2].initial, 25.0);
    }

    #[test]
    fn parses_pkmodel() {
        let m = parse_blocks("$PKMODEL ncmt=1, depot=TRUE\n");
        let pk = m.pkmodel.as_ref().unwrap();
        assert_eq!(pk.ncmt, 1);
        assert!(pk.depot);
        assert!(pk.cmt_names.is_none());
    }

    #[test]
    fn parses_pkmodel_with_cmt_names() {
        let m = parse_blocks("$PKMODEL ncmt=2, depot=FALSE, cmt=\"CENT,PERIPH\"\n");
        let pk = m.pkmodel.as_ref().unwrap();
        assert_eq!(pk.ncmt, 2);
        assert!(!pk.depot);
        assert_eq!(
            pk.cmt_names,
            Some(vec!["CENT".to_string(), "PERIPH".to_string()])
        );
    }

    #[test]
    fn pkmodel_rejects_unknown_arg() {
        let blocks = extract("$PKMODEL ncmt=1, foo=bar\n").unwrap();
        let mut m = MrgModel::default();
        let err = dispatch_declarative(&blocks[0], &mut m).unwrap_err();
        assert!(err.contains("foo"));
    }

    #[test]
    fn parses_input_as_covariates() {
        let m = parse_blocks("$INPUT WT, AGE\n");
        assert_eq!(m.covariates, vec!["WT".to_string(), "AGE".to_string()]);
    }

    #[test]
    fn parses_main_body() {
        let m = parse_blocks(
            "\
$PARAM TVCL = 1
$OMEGA @labels ETA_CL
0.09
$MAIN
double CL = TVCL * exp(ETA(1));
",
        );
        let main = m.main_body.as_ref().unwrap();
        assert!(main.text.contains("CL = TVCL * exp(ETA_CL)"));
    }

    #[test]
    fn rejects_unsupported_block() {
        let blocks = extract("$GLOBAL #include <math.h>\n").unwrap();
        let mut m = MrgModel::default();
        let err = dispatch_declarative(&blocks[0], &mut m).unwrap_err();
        assert!(err.contains("$GLOBAL"));
    }

    #[test]
    fn rejects_duplicate_body_block() {
        let src = "\
$MAIN
double CL = 1;
$MAIN
double V = 2;
";
        let blocks = extract(src).unwrap();
        let mut m = MrgModel::default();
        let name_map = NameMap::from_model(&m);
        let _ = name_map; // suppress unused
        for rb in &blocks {
            dispatch_declarative(rb, &mut m).unwrap();
        }
        // First $MAIN OK.
        dispatch_body(&blocks[0], &mut m).unwrap();
        // Second $MAIN errors.
        let err = dispatch_body(&blocks[1], &mut m).unwrap_err();
        assert!(err.contains("more than once"));
    }
}
