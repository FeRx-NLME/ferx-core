//! Preprocess a C++-flavoured mrgsolve body into ferx DSL syntax.
//!
//! Transformations (in order):
//!
//! 1. Reject disallowed constructs (`?:`, `for`, `while`, `return`, C casts,
//!    preprocessor lines).
//! 2. Strip leading type keywords: `double X = ...;` → `X = ...;`.
//! 3. Rewrite `dxdt_NAME = ...;` → `d/dt(NAME) = ...;`.
//! 4. Resolve `ETA(N)`/`THETA(N)` to named references via [`NameMap`].
//! 5. Map `fabs` → `abs`.
//! 6. Replace `;` with newline so each ferx statement is on its own line.
//!
//! `EPS(N)` is intentionally *not* resolved here — ferx represents residual
//! error declaratively (`~ proportional(SIGMA)`), so any `EPS(N)` reference
//! belongs in the `$TABLE`/`$ERROR` classifier (see `error_pattern`).
//! If `EPS(N)` appears in a body where it's illegal (e.g. `$MAIN`), the
//! caller catches that via [`reject_eps`].

use super::name_map::NameMap;
use std::sync::LazyLock;

use regex::Regex;

/// Run the full preprocessing pipeline on a body string. Returns the
/// transformed source. `start_line` is the 1-indexed source line of the
/// `$BLOCK` opener, used only for error messages.
pub(crate) fn preprocess(
    body: &str,
    name_map: &NameMap,
    start_line: usize,
) -> Result<String, String> {
    reject_unsupported(body, start_line)?;
    let body = strip_type_keywords(body);
    let body = rewrite_dxdt(&body);
    let body = resolve_indexed_refs(&body, name_map, start_line)?;
    let body = rewrite_function_aliases(&body);
    let body = replace_semicolons(&body);
    Ok(body)
}

/// Hard-error for any C++ construct outside the supported subset.
fn reject_unsupported(body: &str, start_line: usize) -> Result<(), String> {
    // Per-line scanning so we can report the offending source line.
    for (idx, line) in body.lines().enumerate() {
        let source_line = start_line + idx + 1; // body starts on the line after the opener
        if KW_FOR.is_match(line) {
            return Err(format!(
                "line {source_line}: `for` loops are not supported by ferx"
            ));
        }
        if KW_WHILE.is_match(line) {
            return Err(format!(
                "line {source_line}: `while` loops are not supported by ferx"
            ));
        }
        if KW_RETURN.is_match(line) {
            return Err(format!(
                "line {source_line}: `return` is not supported by ferx"
            ));
        }
        if KW_PREPROC.is_match(line) {
            return Err(format!(
                "line {source_line}: preprocessor directives (`#`) are not supported by ferx"
            ));
        }
        if KW_POW_CALL.is_match(line) {
            return Err(format!(
                "line {source_line}: `pow(a, b)` is not supported by ferx; use `a^b` instead"
            ));
        }
        if KW_C_CAST.is_match(line) {
            return Err(format!(
                "line {source_line}: C-style casts (e.g. `(double)x`) are not supported by ferx"
            ));
        }
        if KW_TERNARY.is_match(line) {
            return Err(format!(
                "line {source_line}: ternary `?:` is not supported by ferx; use an inline `if`"
            ));
        }
    }
    Ok(())
}

/// Strip a leading C++ type keyword (`double`, `int`, `bool`, `float`,
/// `char`) when it precedes an assignment. Mid-line occurrences are left
/// alone (they shouldn't happen in supported syntax).
fn strip_type_keywords(body: &str) -> String {
    LEADING_TYPE.replace_all(body, "").into_owned()
}

/// `dxdt_NAME = expr;` → `d/dt(NAME) = expr;`.
fn rewrite_dxdt(body: &str) -> String {
    DXDT.replace_all(body, "d/dt($1)").into_owned()
}

/// Resolve every `ETA(N)` / `THETA(N)` reference to the corresponding ferx
/// parameter name. `EPS(N)` is left untouched (handled by error-pattern
/// classifier).
fn resolve_indexed_refs(
    body: &str,
    name_map: &NameMap,
    start_line: usize,
) -> Result<String, String> {
    let mut out = String::with_capacity(body.len());
    let mut last_end = 0usize;
    for cap in INDEXED.captures_iter(body) {
        let m = cap.get(0).unwrap();
        let kind = &cap[1]; // ETA / THETA / EPS
        let n_str = &cap[2];
        let n: usize = n_str.parse().map_err(|_| {
            format!(
                "line {start_line}: cannot parse index in `{}({})`",
                kind, n_str
            )
        })?;
        let resolved = match kind.to_ascii_uppercase().as_str() {
            "ETA" => name_map.eta(n).map_err(|e| with_line(&e, start_line))?.to_string(),
            "THETA" => name_map
                .theta(n)
                .map_err(|e| with_line(&e, start_line))?
                .to_string(),
            // Leave EPS(N) intact — the error-pattern classifier in
            // error_pattern.rs sees the original form and decides what to
            // do.
            "EPS" => body[m.start()..m.end()].to_string(),
            other => return Err(format!("internal: unexpected kind {other}")),
        };
        out.push_str(&body[last_end..m.start()]);
        out.push_str(&resolved);
        last_end = m.end();
    }
    out.push_str(&body[last_end..]);
    Ok(out)
}

fn with_line(msg: &str, line: usize) -> String {
    format!("line {line}: {msg}")
}

/// `fabs(x)` → `abs(x)`. Other aliases can go here later.
fn rewrite_function_aliases(body: &str) -> String {
    FABS_CALL.replace_all(body, "abs(").into_owned()
}

/// `expr;` → `expr\n`. Newlines separate ferx statements.
fn replace_semicolons(body: &str) -> String {
    body.replace(';', "\n")
}

// ── Regex catalogue ────────────────────────────────────────────────────────

static LEADING_TYPE: LazyLock<Regex> = LazyLock::new(|| {
    // `capture` is mrgsolve's declare-and-expose keyword in $TABLE/$ERROR.
    // For ferx purposes it's just a plain assignment.
    Regex::new(r"(?m)^\s*(?:double|int|float|bool|char|capture)\s+").unwrap()
});

static DXDT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bdxdt_(\w+)\b").unwrap());

static INDEXED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(ETA|THETA|EPS)\s*\(\s*(\d+)\s*\)").unwrap()
});

static FABS_CALL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bfabs\s*\(").unwrap());

static KW_FOR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bfor\b").unwrap());
static KW_WHILE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bwhile\b").unwrap());
static KW_RETURN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\breturn\b").unwrap());
static KW_PREPROC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*#").unwrap());
static KW_POW_CALL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bpow\s*\(").unwrap());
static KW_C_CAST: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(\s*(?:double|int|float|bool|char)\s*\)").unwrap());
static KW_TERNARY: LazyLock<Regex> = LazyLock::new(|| {
    // ferx already uses `if (cond) a else b`, never raw `cond ? a : b`. The
    // bare `?` character only appears in C++ ternaries, so this is a safe
    // hard-fail trigger.
    Regex::new(r"\?").unwrap()
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::mrgsolve::intermediate::{MrgModel, NamedValue, OmegaBlock};

    fn make_name_map() -> NameMap {
        let m = MrgModel {
            thetas: vec![
                NamedValue {
                    name: "TVCL".into(),
                    value: 0.0,
                },
                NamedValue {
                    name: "TVV".into(),
                    value: 0.0,
                },
                NamedValue {
                    name: "TVKA".into(),
                    value: 0.0,
                },
            ],
            omegas: vec![OmegaBlock {
                block_form: false,
                labels: vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()],
                values: vec![0.09, 0.04, 0.30],
            }],
            sigmas: vec![OmegaBlock {
                block_form: false,
                labels: vec!["PROP_ERR".into()],
                values: vec![0.02],
            }],
            ..MrgModel::default()
        };
        NameMap::from_model(&m)
    }

    #[test]
    fn strips_double_keyword() {
        let body = "double CL = TVCL;\ndouble V = TVV;\n";
        let out = strip_type_keywords(body);
        assert_eq!(out, "CL = TVCL;\nV = TVV;\n");
    }

    #[test]
    fn rewrites_dxdt() {
        let body = "dxdt_GUT = -KA * GUT;\ndxdt_CENT = KA * GUT - (CL/V) * CENT;\n";
        let out = rewrite_dxdt(body);
        assert!(out.contains("d/dt(GUT) = -KA * GUT"));
        assert!(out.contains("d/dt(CENT) = KA * GUT - (CL/V) * CENT"));
    }

    #[test]
    fn resolves_theta_and_eta() {
        let nm = make_name_map();
        let body = "CL = THETA(1) * exp(ETA(1));";
        let out = resolve_indexed_refs(body, &nm, 1).unwrap();
        assert_eq!(out, "CL = TVCL * exp(ETA_CL);");
    }

    #[test]
    fn preserves_eps_references() {
        let nm = make_name_map();
        let body = "Y = IPRED * (1 + EPS(1));";
        let out = resolve_indexed_refs(body, &nm, 1).unwrap();
        // EPS is left intact — error_pattern classifier handles it later.
        assert!(out.contains("EPS(1)"));
    }

    #[test]
    fn out_of_range_eta_errors() {
        let nm = make_name_map();
        let body = "CL = TVCL * exp(ETA(99));";
        let err = resolve_indexed_refs(body, &nm, 7).unwrap_err();
        assert!(err.contains("line 7"));
        assert!(err.contains("out of range"));
    }

    #[test]
    fn rejects_for_loop() {
        let body = "for (int i = 0; i < 3; i++) {}\n";
        let err = reject_unsupported(body, 10).unwrap_err();
        assert!(err.contains("for"));
        assert!(err.contains("line 11")); // start_line + 1 (first body line)
    }

    #[test]
    fn rejects_pow_call() {
        let body = "Y = pow(X, 2);\n";
        let err = reject_unsupported(body, 5).unwrap_err();
        assert!(err.contains("pow"));
        assert!(err.contains("a^b"));
    }

    #[test]
    fn rejects_ternary() {
        let body = "X = (Y > 0) ? Y : 0;\n";
        let err = reject_unsupported(body, 1).unwrap_err();
        assert!(err.contains("ternary"));
    }

    #[test]
    fn rejects_c_style_cast() {
        let body = "x = (double) y;\n";
        let err = reject_unsupported(body, 1).unwrap_err();
        assert!(err.contains("C-style cast"));
    }

    #[test]
    fn rejects_preprocessor_directive() {
        let body = "#include <math.h>\n";
        let err = reject_unsupported(body, 1).unwrap_err();
        assert!(err.contains("preprocessor"));
    }

    #[test]
    fn fabs_to_abs() {
        let body = "X = fabs(Y);\n";
        let out = rewrite_function_aliases(body);
        assert_eq!(out, "X = abs(Y);\n");
    }

    #[test]
    fn semicolons_become_newlines() {
        let body = "X = 1; Y = 2;\n";
        let out = replace_semicolons(body);
        assert_eq!(out, "X = 1\n Y = 2\n\n");
    }

    #[test]
    fn end_to_end_main_body() {
        let nm = make_name_map();
        let body = "double CL = TVCL * exp(ETA(1));\ndouble V  = TVV * exp(ETA(2));\n";
        let out = preprocess(body, &nm, 0).unwrap();
        // Type keyword stripped, ETA(N) resolved, ; → newline
        assert!(out.contains("CL = TVCL * exp(ETA_CL)"));
        assert!(out.contains("V  = TVV * exp(ETA_V)"));
        assert!(!out.contains("double "));
        assert!(!out.contains(';'));
    }
}
