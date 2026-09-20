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
//! Add new codes here and document them in `docs/file-formats/check-report.qmd`.
//!
//! | code | meaning |
//! |------|---------|
//! | `E_PARSE`                 | the model file failed to parse |
//! | `E_MISSING_BLOCK`         | a required `[block]` is absent |
//! | `E_UNKNOWN_BLOCK`         | a `[block]` header is not a recognised block name |
//! | `E_DEPRECATED_BLOCK`      | a `[block]` that was ferx syntax and is no longer read |
//! | `E_BLOCK_INSTANCE_NAME`   | a `[block NAME]` instance name is present where none is taken, or missing where one is required |
//! | `E_BLOCK_FEATURE_DISABLED`| a `[block]` needs a cargo feature this binary was not built with |
//! | `E_NN_FEATURE_DISABLED`   | a `[covariate_nn]` block needs `--features nn` |
//! | `E_MISSING_COVARIATE`     | the model references a covariate not present in the data |
//! | `E_PER_CMT_SCALING`       | an observed compartment lacks a per-CMT scaling entry |
//! | `E_PER_CMT_ERROR_MODEL`   | an observed compartment lacks a per-CMT `[error_model]` entry |
//! | `W_PER_CMT_UNMATCHED`     | the other direction of the two above: a **declared** per-CMT entry — `[scaling] obs_scale[CMT=N]`, `[error_model] CMT=N:`, or a `y[CMT=N]` readout on either engine — that no observation is recorded on, so the entry is inert. A warning rather than an error because nothing goes `NaN`, and one code for all three channels, with the channel named in the message. Names `[data_selection]` as a candidate cause only for the dead compartments a clause actually emptied. Silent on an observation-free population, so `ferx check` without `--data` does not report every declared entry as dead (#1405) |
//! | `E_ENDPOINT_UNROUTED`     | a CMT declared as a non-Gaussian endpoint carries Gaussian observations — the population was read without the model's endpoint routing (#1199) |
//! | `E_ENDPOINT_NO_RECORDS`   | a declared non-Gaussian endpoint has no row routed to it (typically a missing `CMT` column) |
//! | `E_DATA`                  | the `--data` file could not be read or parsed |
//! | `E_SDE_INCOMPATIBLE`      | an SDE (`[diffusion]`) model used with SAEM / GN |
//! | `E_AD_RETIRED`            | `gradient_method = ad` requested; the Enzyme AD path was retired (use `auto` / `fd`) |
//! | `W_AUTO_OPTIMIZER_FOLLOWS_GRADIENT` | `gradient = fd` with `optimizer` left at `auto` on a model whose analytic gradient *is* in scope: `auto` follows the gradient, so the one line moved the optimizer too (#1381) |
//! | `E_IMP_CHAIN`             | `imp` mis-placed in a method chain (repeated / non-terminal) |
//! | `E_SAEM_NO_RANDOM_EFFECTS`| `method = saem` anywhere in a chain on a model with `n_eta = 0` |
//! | `E_METHOD_NO_RANDOM_EFFECTS` | `method = imp` / `impmap` / `bayes` anywhere in a chain on a model with `n_eta = 0` |
//! | `W_GN_NO_RANDOM_EFFECTS`  | `method = gn` as the last estimating stage on a model with `n_eta = 0` (start-sensitive; prefer `gn_hybrid`) |
//! | `E_OPTIMIZER_IOV`         | `optimizer = trust_region` used with an IOV model |
//! | `E_OPTIMIZER_AGQ`         | `optimizer = trust_region` used with a quadrature stage (`laplace`, or `focei` with `n_agq > 1`) |
//! | `E_SIGMA_ORDER_MISMATCH`  | a single-endpoint `[error_model]` names its sigmas in an order other than the `[parameters]` declaration order |
//! | `E_BLOCK_VARIANCE_ONLY`   | a scale tag (`(sd)` / `(variance)` / `(var)`) on a `block_omega` / `block_sigma` / `block_kappa` declaration, whose lower triangle mixes variances and covariances and so takes no single scale. The repair depends on which tag was written and rides along in `suggestion`: an `(sd)` author squares each SD into a variance and writes the off-diagonals as covariances; a `(variance)` / `(var)` author deletes the tag, which claimed nothing the lower triangle did not already say (#1377) |
//! | `E_OMEGA_INIT_AT_RAIL`    | a **free** `omega` / `kappa` / `[mixture] omega(k)` variance whose initial value packs onto the optimizer's `-6` lower rail (variance ≤ 6.1e-6, `~ 0.0` included) — clamped there and not estimable; `FIX` it or start it higher (#1229) |
//! | `E_THETA_INIT_OUTSIDE_BOUNDS` | a `theta` whose initial value is strictly outside its **own declared** range, compared on the declared scale so a lower bound of `0` is not confused with ferx's `1e-10` packing floor — clamped into the box and fitted from there; NM-TRAN refuses the identical stream (error 24) (#1251) |
//! | `E_INIT_BOUNDS_INVERTED` | a coordinate whose packed box is **empty** — for a `theta`, bounds swapped or a declared range lying entirely outside ferx's `1e-10` / `1e9` packing caps. No start can be placed in it, and the clamp has no interval to clamp into. Only θ reaches it through the parser, but the code is kind-neutral. The only start-side check with no `maxiter = 0` exemption (#1251) |
//! | `W_INIT_OUTSIDE_BOUNDS`   | an initial estimate strictly outside one of ferx's **internal** rails (the hidden `1e9` θ cap, the Ω `±6` / off-diagonal `±10` guards, the Σ `[-8, 5]` guard) — clamped there before the first objective evaluation (#1251) |
//! | `W_STEADY_STATE_II`       | SS=1 dose with missing / non-positive II |
//! | `W_STEADY_STATE_INFUSION` | SS=1 infusion with `T_inf > II` (overlapping pulses) |
//! | `W_STEADY_STATE_ABSOLUTE_TIME` | SS=1 dose on an `[odes]` PK block reading an absolute clock (`TAFD`, or `T` / `TIME`) — the run-in expands the train on a cycle-local clock, so there is no periodic limit to converge to: `TAFD` reads `NaN`, `T` / `TIME` return NONMEM's value (#1139) |
//! | `W_SDE_RESET`             | EVID=3/4 resets under an SDE model are not honoured |
//! | `W_SDE_LAGTIME`           | an absorption lag time under an SDE model is not honoured |
//! | `W_EXPERIMENTAL_SDE`      | an SDE (`[diffusion]`) model uses an experimental feature; the filter is covariance-only, so the state mean is never corrected by the data (see Feature Maturity and the SDE page) |
//! | `W_EXPERIMENTAL_NN`       | a neural-network (`[covariate_nn]`) model uses an experimental feature (see Feature Maturity docs) |
//! | `W_NEGATIVE_LAGTIME`      | a lag time is negative at the initial estimates |
//! | `E_DERIVED_NAME_CONFLICT` | `[derived]` name clashes with a built-in sdtab column, theta, eta, or indiv-param name |
//! | `W_DERIVED_COVARIATE_SHADOW` | `[derived]` name shadows a covariate (allowed but may be confusing) |
//! | `W_DERIVED_STEP_IGNORED`  | `step=` given for a DV-based integral (ignored; DV integrals use observation times) |
//! | `W_ABSORPTION_TWIN_DECLINED` | an analytic transit / IG model's ODE twin could not be built; the model stays closed-form with no ODE fallback (#1008) |
//! | `E_OUTPUT_UNKNOWN_COLUMN` | a name in `[output]` is not recognised as any known quantity |
//! | `W_OUTPUT_DUPLICATE`      | a name in `[output]` is already in the mandatory sdtab minimum |
//! | `W_ADDL_MISSING_II`       | ADDL > 0 on a dose row but II is zero or missing; additional doses not expanded |
//! | `W_COMPARTMENT_FREE_DOSES` | dose records in the dataset of a compartment-free (`$PRED`-equivalent) model, which applies no dose — reported once with the event (post-`ADDL`) and dosed-subject counts; the dose-level checks that read a topology (`E_MODELED_*_NO_PARAM`, `E_DOSE_CMT_*`) are skipped for this class, so the rows land here (#1443) |
//! | `W_MISSING_DV`            | EVID=0 observation row with a missing DV and no MDV=1; skipped rather than scored as DV=0 |
//! | `W_IOV_OCC_MISSING`       | rows in the IOV occasion column had a missing or unparseable value and were assigned occasion 0 |
//! | `W_CENS_UNEXPECTED`       | an observation row's `CENS` cell is a value other than `-1` (above ULOQ), `0` (quantified) or `1` (below LLOQ). The M3 likelihood's `m3_logcdf` treats **every** nonzero flag as a left tail, so the row is scored as censored rather than rejected — flagged rather than silently mis-scored. Reported once per subject |
//! | `W_FILTER_COLUMN_ABSENT`  | a `[data_selection]` condition names a column the dataset does not have; the comparison never matches, so the condition does not select what was meant. The message names the missing column(s) and lists the headers the file does have. Raised once per read, whichever key the condition came from — and what "never matches" costs depends on that key: an `ignore` excludes nothing, an `accept` excludes everything. Reaches `ferx check` only since [#1465](https://github.com/FeRx-NLME/ferx-core/issues/1465), which made the check compile the clauses at all |
//! | `W_AMT_NOT_DOSED`         | record(s) carry `AMT != 0` but were **not** read as dose events, because `EVID` is neither 1 nor 4; their `AMT` was ignored. With no `EVID` column a nonzero `AMT` is enough to infer a dose, so this is a dataset that has one. Reported with the record and subject counts, and it wins over `W_NO_DOSES`, which is the generic backstop for a dataset carrying no `AMT` signal at all (#262) |
//! | `W_NO_DOSES`              | zero dose events parsed across every subject **although scored observations are present**. Silent on a dataset with no scored observations (an all-`EVID=2` / covariate-only file is not a fit), and silent when any subject carries a non-Gaussian observation — a TTE / survival or discrete endpoint legitimately has no PK dose, so warning there would be a false positive (#262) |
//! | `W_ALL_DOSES_ZERO`        | an `AMT` column is present and at least one dose event was parsed, but every parsed dose amount is exactly zero — so no drug enters the system and the objective is flat: the parameters will not move from their initial values. A warning rather than an error because the column is there and a genuinely dose-free-but-`AMT`-columned dataset is conceivable ([#753](https://github.com/FeRx-NLME/ferx-core/issues/753)) |
//! | `W_CMT_DEFAULTED`         | dose / observation rows assigned compartment 1 because the dataset has no `CMT` column, or the cell is missing or unparseable; reported when `CMT` selects something — more than one compartment a dose can reach (multi-state `[odes]`, or an analytical model whose `CMT=2` is a real target), a per-CMT scaling / error model / readout on either engine, an endpoint the row routes to, or a `[data_selection]` clause comparing `CMT` (the scope is the `CmtConsumer` enumeration in `api::validation`) |
//! | `E_COVSTAT_UNRESOLVED`    | a `[covariate_model]` relation still needs data-derived statistics (`center = median`, `levels = auto`, or a form whose default bounds come from the data) |
//! | `W_COVSTAT_UNBOUND`       | the same, reported without a `--data` file — the model is fine, it just cannot be built until a dataset is supplied |

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
    /// The `[individual_parameters]` block as the `[covariate_model]` desugar
    /// rewrote it (#1111), one line per assignment. Empty for a model that
    /// declares no such block.
    ///
    /// `ferx check` prints this so the expression the block actually built is
    /// visible — a covariate model stated declaratively is otherwise
    /// unauditable against NONMEM, since nothing in the file spells out the
    /// centring constant, the missing-value guard or where the factor landed.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub desugared_individual_parameters: Vec<String>,
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
            desugared_individual_parameters: Vec::new(),
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
