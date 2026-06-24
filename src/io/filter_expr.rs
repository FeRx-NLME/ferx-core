use std::collections::HashMap;

/// A single comparison: `column op value`.
#[derive(Debug, Clone)]
pub struct FilterExpr {
    col: String, // lowercase
    op: CmpOp,
    rhs: FilterValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
enum FilterValue {
    Num(f64),
    Str(String),
}

impl FilterExpr {
    fn parse_one(s: &str) -> Result<Self, String> {
        // Try each two-char operator first (order matters: `>=` before `>`).
        let ops: &[(&str, CmpOp)] = &[
            ("==", CmpOp::Eq),
            ("!=", CmpOp::Ne),
            (">=", CmpOp::Ge),
            ("<=", CmpOp::Le),
            (">", CmpOp::Gt),
            ("<", CmpOp::Lt),
        ];
        for (sym, op) in ops {
            if let Some(pos) = s.find(sym) {
                let col = s[..pos].trim().to_lowercase();
                let rhs_raw = s[pos + sym.len()..].trim();
                if col.is_empty() || rhs_raw.is_empty() {
                    return Err(format!(
                        "malformed filter expression '{}': column or value is empty",
                        s.trim()
                    ));
                }
                // ID is a string label; ordered comparisons on it are not
                // meaningful and would otherwise silently evaluate to false.
                // Reject them up front with a clear message.
                if col == "id" && !matches!(op, CmpOp::Eq | CmpOp::Ne) {
                    return Err(format!(
                        "filter expression '{}': ordered comparisons (<, <=, >, >=) are not \
                         supported on ID (a string label); use == / != or ignore_subjects.",
                        s.trim()
                    ));
                }
                // String RHS: `"A"` or `'A'`. Require length >= 2 so a lone quote
                // (`\"` / `'`) does not underflow the `[1..len-1]` slice below.
                let is_quoted = rhs_raw.len() >= 2
                    && ((rhs_raw.starts_with('"') && rhs_raw.ends_with('"'))
                        || (rhs_raw.starts_with('\'') && rhs_raw.ends_with('\'')));
                let rhs = if is_quoted {
                    FilterValue::Str(rhs_raw[1..rhs_raw.len() - 1].to_string())
                } else if col == "id" {
                    // ID is a string label, so it is always compared as a string —
                    // even when written bare (`ID == 3`). Without this, a bare
                    // numeric would be parsed as `Num`, and `eval` short-circuits
                    // every numeric comparison against `id` to `false`, so
                    // `ID == 3` would silently never match.
                    FilterValue::Str(rhs_raw.to_string())
                } else if let Ok(n) = rhs_raw.parse::<f64>() {
                    FilterValue::Num(n)
                } else {
                    return Err(format!(
                        "malformed filter expression '{}': right-hand side '{}' is not a \
                         number or quoted string",
                        s.trim(),
                        rhs_raw
                    ));
                };
                return Ok(FilterExpr { col, op: *op, rhs });
            }
        }
        Err(format!(
            "malformed filter expression '{}': no comparison operator found (==, !=, >=, <=, >, <)",
            s.trim()
        ))
    }

    fn eval(&self, ctx: &RowContext<'_>) -> bool {
        let col = self.col.as_str();
        match &self.rhs {
            FilterValue::Num(rhs) => {
                let lhs = match col {
                    "id" => return false, // numeric comparison against ID string: never match
                    "time" => ctx.time,
                    "dv" => ctx.dv,
                    "evid" => ctx.evid as f64,
                    "amt" => ctx.amt,
                    "cmt" => ctx.cmt as f64,
                    "rate" => ctx.rate,
                    "mdv" => ctx.mdv as f64,
                    "cens" => ctx.cens as f64,
                    "ii" => ctx.ii,
                    "ss" => ctx.ss as u8 as f64,
                    // Case-insensitive lookup so `BW >= 30` matches a "BW" or "bw"
                    // column. `col` is already lowercased; `eq_ignore_ascii_case`
                    // avoids allocating a lowercased key per row in the read loop.
                    _ => match ctx
                        .covariates
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(col))
                        .map(|(_, &v)| v)
                    {
                        Some(v) => v,
                        // Unknown column: never fires (safe default).
                        None => return false,
                    },
                };
                cmp_f64(lhs, *rhs, self.op)
            }
            FilterValue::Str(rhs) => {
                let lhs: &str = match col {
                    "id" => ctx.id,
                    // String comparison against numeric columns is not meaningful.
                    _ => return false,
                };
                match self.op {
                    CmpOp::Eq => lhs == rhs,
                    CmpOp::Ne => lhs != rhs,
                    _ => false, // <, <=, >, >= on strings not supported
                }
            }
        }
    }
}

fn cmp_f64(lhs: f64, rhs: f64, op: CmpOp) -> bool {
    // Exact comparison. LHS (a CSV cell) and RHS (an expression literal) are both
    // parsed from decimal text, so `STUDY == 2` / `WT == 70.5` match exactly; an
    // absolute EPSILON band would instead wrongly equate small nonzero values to
    // zero and adjacent large integers to each other. A missing value arrives as
    // NaN and must never match: `==`/`<`/`<=`/`>`/`>=` against NaN are already
    // false, and `!=` is guarded so NaN does not spuriously satisfy it.
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => !lhs.is_nan() && lhs != rhs,
        CmpOp::Lt => lhs < rhs,
        CmpOp::Le => lhs <= rhs,
        CmpOp::Gt => lhs > rhs,
        CmpOp::Ge => lhs >= rhs,
    }
}

/// A single user-supplied expression string, which may contain one or more
/// `FilterExpr`s joined by `&&`. All sub-expressions must hold for the clause
/// to evaluate to `true`. Consistent with the existing ferx DSL which uses `&&`
/// (not `AND`/`OR` keywords) for boolean composition.
///
/// `||` within a string is rejected at parse time with a clear error message;
/// use multiple strings (via `c()` in R or repeated lines in `.ferx`) instead.
#[derive(Debug, Clone)]
pub struct FilterClause {
    exprs: Vec<FilterExpr>,
    /// Original string, preserved verbatim for logging.
    pub source: String,
}

impl FilterClause {
    /// Parse a user-supplied expression string.
    ///
    /// Accepts:
    /// - `"DV < 0.001"` (bare, from `.ferx` file) or `"\"DV < 0.001\""` (R
    ///   quoted string) — surrounding quotes are stripped.
    /// - `"BW >= 30 && BW < 48"` — `&&`-joined clauses; all must hold.
    ///
    /// Rejects:
    /// - `||` — parse error: use multiple strings instead.
    /// - `AND` / `OR` keyword forms — parse error: use `&&` / multiple strings.
    pub fn parse(raw: &str) -> Result<Self, String> {
        // Strip surrounding quotes (R passes "DV < 0.001" with quotes intact).
        // Require length >= 2 so a lone quote does not underflow the slice.
        let s = raw.trim();
        let s = if s.len() >= 2
            && ((s.starts_with('"') && s.ends_with('"'))
                || (s.starts_with('\'') && s.ends_with('\'')))
        {
            &s[1..s.len() - 1]
        } else {
            s
        };

        // Reject || early with a clear message.
        if s.contains("||") {
            return Err(format!(
                "filter expression '{}' contains '||': OR within a single expression is not \
                 supported. Use multiple ignore/accept conditions (c() in R, or repeated lines \
                 in [data_selection]) — each is an independent reason to exclude.",
                s
            ));
        }

        // Reject AND / OR keyword forms.
        {
            let upper = s.to_uppercase();
            for keyword in &[" AND ", " OR "] {
                if upper.contains(keyword) {
                    return Err(format!(
                        "filter expression '{}' contains keyword '{}': use '&&' for AND, \
                         or multiple conditions for OR.",
                        s,
                        keyword.trim()
                    ));
                }
            }
        }

        let source = s.to_string();
        let parts: Vec<&str> = s.split("&&").collect();
        let mut exprs = Vec::with_capacity(parts.len());
        for part in parts {
            exprs.push(FilterExpr::parse_one(part.trim())?);
        }
        Ok(FilterClause { exprs, source })
    }

    /// Returns `true` when all sub-expressions hold for the given row.
    pub fn eval(&self, ctx: &RowContext<'_>) -> bool {
        self.exprs.iter().all(|e| e.eval(ctx))
    }

    /// Non-standard (covariate) column names referenced by this clause, in
    /// lowercase. Standard NONMEM columns (ID/TIME/DV/...) are excluded since
    /// they are read directly into [`RowContext`] rather than via the covariate
    /// map. Used by the reader to ensure a filtered covariate column is read
    /// even when a `[covariates]` block did not declare it.
    pub fn covariate_columns(&self) -> impl Iterator<Item = &str> {
        self.exprs
            .iter()
            .map(|e| e.col.as_str())
            .filter(|c| !is_standard_column(c))
    }
}

/// True for the fixed NONMEM columns the reader handles specially, so they are
/// not mistaken for covariates by `covariate_columns()` (case-insensitive; `col`
/// is expected already lowercased). `addl` is included — it is a reader-standard
/// column, not a covariate, even though it has no [`RowContext`] field and so is
/// not a usable filter target (a condition on it is an inert no-op). The dynamic
/// occasion / IOV column cannot be listed here; filtering on it is likewise
/// unsupported (see `docs/model-file/data-selection.qmd`).
pub fn is_standard_column(col: &str) -> bool {
    matches!(
        col,
        "id" | "time"
            | "dv"
            | "evid"
            | "amt"
            | "cmt"
            | "rate"
            | "mdv"
            | "cens"
            | "ii"
            | "ss"
            | "addl"
    )
}

/// Per-row context passed to `FilterClause::eval`.
pub struct RowContext<'a> {
    pub id: &'a str,
    pub time: f64,
    pub dv: f64,
    pub evid: u32,
    pub amt: f64,
    pub cmt: usize,
    pub rate: f64,
    pub mdv: u32,
    pub cens: i8,
    pub ii: f64,
    pub ss: bool,
    /// Covariate values for this row, keyed by the original CSV header name
    /// (NOT case-folded). `FilterExpr::eval` matches case-insensitively via
    /// `eq_ignore_ascii_case`, so callers must not assume lowercased keys.
    pub covariates: &'a HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn ctx(id: &str, _dv: f64, _evid: u32, bw: f64) -> (String, HashMap<String, f64>) {
        let mut cov = HashMap::new();
        cov.insert("bw".to_string(), bw);
        (id.to_string(), cov)
    }

    fn eval(expr: &str, id: &str, dv: f64, evid: u32, bw: f64) -> bool {
        let (_, cov) = ctx(id, dv, evid, bw);
        let clause = FilterClause::parse(expr).expect("parse ok");
        clause.eval(&RowContext {
            id,
            time: 0.0,
            dv,
            evid,
            amt: 0.0,
            cmt: 1,
            rate: 0.0,
            mdv: 0,
            cens: 0,
            ii: 0.0,
            ss: false,
            covariates: &cov,
        })
    }

    // ── Operator tests ────────────────────────────────────────────────────────

    #[test]
    fn test_op_eq() {
        assert!(eval("EVID == 0", "1", 1.0, 0, 70.0));
        assert!(!eval("EVID == 0", "1", 1.0, 1, 70.0));
    }

    #[test]
    fn test_op_ne() {
        assert!(eval("EVID != 0", "1", 1.0, 1, 70.0));
        assert!(!eval("EVID != 0", "1", 1.0, 0, 70.0));
    }

    #[test]
    fn test_op_lt() {
        assert!(eval("DV < 0.001", "1", 0.0005, 0, 70.0));
        assert!(!eval("DV < 0.001", "1", 1.0, 0, 70.0));
    }

    #[test]
    fn test_op_le() {
        assert!(eval("DV <= 1.0", "1", 1.0, 0, 70.0));
        assert!(!eval("DV <= 1.0", "1", 1.1, 0, 70.0));
    }

    #[test]
    fn test_op_gt() {
        assert!(eval("BW > 80", "1", 0.0, 0, 90.0));
        assert!(!eval("BW > 80", "1", 0.0, 0, 70.0));
    }

    #[test]
    fn test_op_ge() {
        assert!(eval("BW >= 30", "1", 0.0, 0, 30.0));
        assert!(eval("BW >= 30", "1", 0.0, 0, 35.0));
        assert!(!eval("BW >= 30", "1", 0.0, 0, 29.0));
    }

    // ── Case-insensitive column names ────────────────────────────────────────

    #[test]
    fn test_case_insensitive_column() {
        assert!(eval("bw >= 30", "1", 0.0, 0, 30.0));
        assert!(eval("BW >= 30", "1", 0.0, 0, 30.0));
        assert!(eval("Bw >= 30", "1", 0.0, 0, 30.0));
    }

    // ── Quoted strings stripped (R-style) ────────────────────────────────────

    #[test]
    fn test_quoted_string_stripped() {
        assert!(eval("\"DV < 0.001\"", "1", 0.0005, 0, 70.0));
        assert!(eval("'DV < 0.001'", "1", 0.0005, 0, 70.0));
    }

    // ── ID string equality ────────────────────────────────────────────────────

    #[test]
    fn test_id_eq_string() {
        assert!(eval("ID == 3", "3", 0.0, 0, 70.0));
        assert!(!eval("ID == 3", "1", 0.0, 0, 70.0));
    }

    #[test]
    fn test_id_ne_string() {
        assert!(eval("ID != 3", "1", 0.0, 0, 70.0));
        assert!(!eval("ID != 3", "3", 0.0, 0, 70.0));
    }

    // ── && composition ────────────────────────────────────────────────────────

    #[test]
    fn test_and_composition() {
        // Both must hold.
        assert!(eval("BW >= 30 && BW < 48", "1", 0.0, 0, 35.0));
        assert!(!eval("BW >= 30 && BW < 48", "1", 0.0, 0, 25.0));
        assert!(!eval("BW >= 30 && BW < 48", "1", 0.0, 0, 50.0));
    }

    #[test]
    fn test_and_evid_dv() {
        assert!(eval("EVID == 0 && DV < 0.001", "1", 0.0005, 0, 70.0));
        assert!(!eval("EVID == 0 && DV < 0.001", "1", 0.0005, 1, 70.0));
        assert!(!eval("EVID == 0 && DV < 0.001", "1", 1.0, 0, 70.0));
    }

    // ── Unknown column ────────────────────────────────────────────────────────

    #[test]
    fn test_unknown_column_never_fires() {
        // Unknown column: no-op (does not exclude).
        assert!(!eval("NOSUCHCOL == 1", "1", 0.0, 0, 70.0));
    }

    // ── Missing value (NaN) never matches ────────────────────────────────────

    #[test]
    fn test_missing_dv_never_matches() {
        // A missing DV is carried as NaN; every comparison against it is false,
        // so e.g. a dose row (DV='.') is not caught by `DV < 0.001`.
        assert!(!eval("DV < 0.001", "1", f64::NAN, 1, 70.0));
        assert!(!eval("DV > 0.001", "1", f64::NAN, 1, 70.0));
        assert!(!eval("DV == 0", "1", f64::NAN, 1, 70.0));
        // `!=` must also not fire on a missing value.
        assert!(!eval("DV != 0", "1", f64::NAN, 1, 70.0));
    }

    #[test]
    fn test_missing_dv_does_not_break_guarded_clause() {
        // `EVID == 0 && DV < 0.001` on a dose row (NaN DV): the DV term is false,
        // so the whole && clause is false — dose row retained.
        assert!(!eval("EVID == 0 && DV < 0.001", "1", f64::NAN, 1, 70.0));
        // On a real low observation it still fires.
        assert!(eval("EVID == 0 && DV < 0.001", "1", 0.0005, 0, 70.0));
    }

    // ── Parse errors ─────────────────────────────────────────────────────────

    #[test]
    fn test_pipe_or_rejected() {
        assert!(FilterClause::parse("DV < 0.001 || EVID == 0").is_err());
    }

    #[test]
    fn test_and_keyword_rejected() {
        assert!(FilterClause::parse("DV < 0.001 AND EVID == 0").is_err());
    }

    #[test]
    fn test_or_keyword_rejected() {
        assert!(FilterClause::parse("DV < 0.001 OR EVID == 0").is_err());
    }

    #[test]
    fn test_missing_operator_rejected() {
        assert!(FilterClause::parse("DV 0.001").is_err());
    }

    #[test]
    fn test_empty_column_rejected() {
        assert!(FilterClause::parse("== 1").is_err());
    }

    #[test]
    fn test_lone_quote_does_not_panic() {
        // A lone quote char must return Err (no comparison operator), not panic
        // on a `[1..len-1]` slice underflow.
        assert!(FilterClause::parse("\"").is_err());
        assert!(FilterClause::parse("'").is_err());
        // A lone quote as the RHS must not panic either (parses as a 1-char
        // string label for ID; the point is simply that it does not crash).
        let _ = FilterClause::parse("ID == \"");
    }

    #[test]
    fn test_id_ordered_comparison_rejected() {
        // Ordered comparisons on ID are a parse error (not a silent no-op).
        assert!(FilterClause::parse("ID >= 3").is_err());
        assert!(FilterClause::parse("ID < 10").is_err());
        assert!(FilterClause::parse("ID > 0").is_err());
        assert!(FilterClause::parse("ID <= 5").is_err());
        // == / != still allowed.
        assert!(FilterClause::parse("ID == 3").is_ok());
        assert!(FilterClause::parse("ID != 3").is_ok());
    }

    #[test]
    fn test_exact_equality_small_and_large() {
        // Near-zero nonzero value must NOT equal 0 (no EPSILON band).
        assert!(!eval("DV == 0", "1", 1e-20, 0, 70.0));
        assert!(eval("DV != 0", "1", 1e-20, 0, 70.0));
        // Exact integer-coded match still works.
        assert!(eval("DV == 5", "1", 5.0, 0, 70.0));
    }
}
