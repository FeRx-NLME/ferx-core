//! Structured diagnostics for model validation.
//!
//! [`Diagnostic`] is the shared currency between the validation logic in
//! [`crate::api`] / [`crate::parser`] and the `ferx check` CLI command. It
//! replaces (well, wraps) the historical free-text `Result<_, String>` errors
//! with a machine-readable shape — a stable `code`, the owning block, a
//! block-level `line`, and an optional `suggestion` — so external callers
//! (coding agents, language bindings) can act on validation output
//! programmatically instead of regex-matching prose.
//!
//! The same `Diagnostic`s feed both `ferx check` (which collects *all* of them
//! in one pass) and `fit()` (which still hard-errors on the first one, via
//! [`first_error`]), keeping a single source of truth for the check logic.
//!
//! ## Error-code registry
//!
//! Codes are stable identifiers (prefix `E_` for errors, `W_` for warnings).
//! Add new codes here and document them in `docs/src/file-formats/check-report.md`.
//!
//! | code | meaning |
//! |------|---------|
//! | `E_PARSE`                 | the model file failed to parse |
//! | `E_MISSING_BLOCK`         | a required `[block]` is absent |
//! | `E_NN_FEATURE_DISABLED`   | a `[covariate_nn]` block needs `--features nn` |
//! | `E_MISSING_COVARIATE`     | the model references a covariate not present in the data |
//! | `E_PER_CMT_SCALING`       | an observed compartment lacks a per-CMT scaling entry |
//! | `E_PER_CMT_ERROR_MODEL`   | an observed compartment lacks a per-CMT `[error_model]` entry |
//! | `E_DATA`                  | the `--data` file could not be read or parsed |
//! | `E_SDE_INCOMPATIBLE`      | an SDE (`[diffusion]`) model used with SAEM / GN / AD |
//! | `E_AD_UNAVAILABLE`        | `gradient_method = ad` requested on a build without the `autodiff` feature |
//! | `E_IMP_CHAIN`             | `imp` mis-placed in a method chain (repeated / non-terminal) |
//! | `E_OPTIMIZER_IOV`         | `optimizer = trust_region` used with an IOV model |
//! | `W_STEADY_STATE_II`       | SS=1 dose with missing / non-positive II |
//! | `W_STEADY_STATE_INFUSION` | SS=1 infusion with `T_inf > II` (overlapping pulses) |
//! | `W_SDE_RESET`             | EVID=3/4 resets under an SDE model are not honoured |
//! | `W_EXPERIMENTAL_SDE`      | an SDE (`[diffusion]`) model uses an experimental feature (see Feature Maturity docs) |
//! | `W_EXPERIMENTAL_NN`       | a neural-network (`[covariate_nn]`) model uses an experimental feature (see Feature Maturity docs) |
//! | `W_NEGATIVE_LAGTIME`      | a lag time is negative at the initial estimates |
//! | `E_DERIVED_NAME_CONFLICT` | `[derived]` name clashes with a built-in sdtab column, theta, eta, or indiv-param name |
//! | `W_DERIVED_COVARIATE_SHADOW` | `[derived]` name shadows a covariate (allowed but may be confusing) |
//! | `W_DERIVED_STEP_IGNORED`  | `step=` given for a DV-based integral (ignored; DV integrals use observation times) |
//! | `E_OUTPUT_UNKNOWN_COLUMN` | a name in `[output]` is not recognised as any known quantity |
//! | `W_OUTPUT_DUPLICATE`      | a name in `[output]` is already in the mandatory sdtab minimum |
//! | `W_ADDL_MISSING_II`       | ADDL > 0 on a dose row but II is zero or missing; additional doses not expanded |
//! | `W_MISSING_DV`            | EVID=0 observation row with a missing DV and no MDV=1; skipped rather than scored as DV=0 |

use serde::Serialize;

/// Severity of a [`Diagnostic`]. Only `Error` affects the `ferx check` exit
/// code and is treated as fatal by [`first_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

/// A single validation finding.
///
/// `line` is **block-level** in the current implementation: it points at the
/// `[block]` header the finding belongs to, not the exact offending token.
/// Token/column spans are a deferred enhancement (see the plan and the
/// check-report docs).
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostic {
    pub severity: Severity,
    /// Stable machine-readable code, e.g. `"E_MISSING_COVARIATE"`.
    pub code: String,
    /// Human-readable description (the historical free-text message).
    pub message: String,
    /// Owning block, e.g. `"individual_parameters"`, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block: Option<String>,
    /// 1-based line of the owning block's header, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    /// Actionable hint, e.g. `"available covariates: WGT, AGE, SEX"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// An `Error`-severity diagnostic with the given code and message.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            block: None,
            line: None,
            suggestion: None,
        }
    }

    /// A `Warning`-severity diagnostic with the given code and message.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            block: None,
            line: None,
            suggestion: None,
        }
    }

    /// Attach the owning block name (builder style).
    pub fn with_block(mut self, block: impl Into<String>) -> Self {
        self.block = Some(block.into());
        self
    }

    /// Attach the owning block's header line (builder style).
    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    /// Attach an actionable suggestion (builder style).
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// True for `Error`-severity diagnostics.
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// The full result of a `ferx check` run.
#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    /// True when no `Error`-severity diagnostics are present.
    pub valid: bool,
    /// Model name / file stem.
    pub model: String,
    /// Data file path, when `--data` was supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    pub diagnostics: Vec<Diagnostic>,
}

impl CheckReport {
    /// Build a report from collected diagnostics; `valid` is derived as
    /// "no error-severity diagnostics present".
    pub fn new(
        model: impl Into<String>,
        data: Option<String>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        let valid = !diagnostics.iter().any(Diagnostic::is_error);
        CheckReport {
            valid,
            model: model.into(),
            data,
            diagnostics,
        }
    }

    /// Count of `Error`-severity diagnostics.
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// Count of `Warning`-severity diagnostics.
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| !d.is_error()).count()
    }
}

/// Collapse a slice of diagnostics to the historical `Result<(), String>`:
/// `Err` with the first error-severity message, else `Ok`. This lets `fit()`
/// keep its fail-fast behavior and identical error strings while sharing the
/// diagnostic-producing validators with `ferx check`.
pub fn first_error(diagnostics: &[Diagnostic]) -> Result<(), String> {
    match diagnostics.iter().find(|d| d.is_error()) {
        Some(d) => Err(d.message.clone()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_fields() {
        let d = Diagnostic::error("E_MISSING_COVARIATE", "covariate 'WT' not found")
            .with_block("individual_parameters")
            .with_line(11)
            .with_suggestion("available: WGT, AGE");
        assert!(d.is_error());
        assert_eq!(d.code, "E_MISSING_COVARIATE");
        assert_eq!(d.block.as_deref(), Some("individual_parameters"));
        assert_eq!(d.line, Some(11));
        assert_eq!(d.suggestion.as_deref(), Some("available: WGT, AGE"));
    }

    #[test]
    fn report_validity_derives_from_errors() {
        let ok = CheckReport::new("m", None, vec![Diagnostic::warning("W_X", "heads up")]);
        assert!(ok.valid);
        assert_eq!(ok.error_count(), 0);
        assert_eq!(ok.warning_count(), 1);

        let bad = CheckReport::new("m", None, vec![Diagnostic::error("E_X", "nope")]);
        assert!(!bad.valid);
        assert_eq!(bad.error_count(), 1);
    }

    #[test]
    fn first_error_returns_first_error_message() {
        let diags = vec![
            Diagnostic::warning("W_A", "warn first"),
            Diagnostic::error("E_B", "second is the error"),
            Diagnostic::error("E_C", "third"),
        ];
        assert_eq!(first_error(&diags), Err("second is the error".to_string()));
    }

    #[test]
    fn first_error_ok_when_no_errors() {
        let diags = vec![Diagnostic::warning("W_A", "just a warning")];
        assert_eq!(first_error(&diags), Ok(()));
    }

    #[test]
    fn optional_fields_omitted_in_json() {
        let d = Diagnostic::error("E_PARSE", "bad");
        let json = serde_json::to_string(&d).unwrap();
        // block / line / suggestion are None → skipped.
        assert_eq!(
            json,
            r#"{"severity":"error","code":"E_PARSE","message":"bad"}"#
        );
    }
}
