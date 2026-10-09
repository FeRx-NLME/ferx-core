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
//! | `E_NO_SCORED_OBSERVATIONS` | the population has no record the likelihood scores — no Gaussian observation and no endpoint record on any subject — so the data contribute nothing to the objective. One population-level finding, last in `check_model_data_rule` so the specific data codes win `fit()`'s first error; the message names `[data_selection]` when it removed observation rows, and otherwise says what makes a row scored (#1491) |
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
//! | `W_COV_ANALYTIC_SALVAGE` | the covariance step assembled the exact analytic R-matrix for the in-scope subjects and finite-differenced the information terms of the subjects outside its scope, rather than dropping the whole population onto the objective stencil (#1514) |
//! | `W_ABSORPTION_TWIN_DECLINED` | an analytic transit / IG model's ODE twin could not be built; the model stays closed-form with no ODE fallback (#1008) |
//! | `E_OUTPUT_UNKNOWN_COLUMN` | a name in `[output]` is not recognised as any known quantity |
//! | `W_OUTPUT_DUPLICATE`      | a name in `[output]` is already in the mandatory sdtab minimum |
//! | `W_ADDL_MISSING_II`       | ADDL > 0 on a dose row but II is zero or missing; additional doses not expanded |
//! | `W_COMPARTMENT_FREE_DOSES` | dose records in the dataset of a compartment-free (`$PRED`-equivalent) model, which applies no dose — reported once with the event (post-`ADDL`) and dosed-subject counts; the dose-level checks that read a topology (`E_MODELED_*_NO_PARAM`, `E_DOSE_CMT_*`) are skipped for this class, so the rows land here (#1443) |
//! | `W_MISSING_DV`            | EVID=0 observation row with a missing DV and no MDV=1; skipped rather than scored as DV=0 |
//! | `W_IOV_OCC_MISSING`       | rows in the IOV occasion column had a missing or unparseable value and were assigned occasion 0 |
//! | `W_CENS_UNEXPECTED`       | an observation row's `CENS` cell is a whole number other than `-1` (above ULOQ), `0` (quantified) or `1` (below LLOQ), such as `7` or `-2`. Every engine reads only the flag's sign, so under `bloq_method = m3` a positive flag is scored on the lower tail, like `1`, and a negative one on the upper tail, like `-1`; under `bloq_method = drop` the row is scored as an ordinary observation. The message quotes the cell as written (`200`, not the `127` it saturates to) and is reported once per subject for each sign. It is not raised on a simulation design row with no `DV`, which is never scored. A cell that is **not a whole number** (`1.5`, `abc`, `inf`) is not this warning but a read error — `E_DATA` in `ferx check` — on a Gaussian observation row that carries a `DV` and that `[data_selection]` keeps; on a dose row, a row whose `DV` is missing, or a row the filter removes, the cell is not read (#1496) |
//! | `W_FILTER_COLUMN_ABSENT`  | a `[data_selection]` condition names a column the dataset does not have; the comparison never matches, so the condition does not select what was meant. The message names the missing column(s) and lists the headers the file does have. Raised once per read, whichever key the condition came from — and what "never matches" costs depends on that key: an `ignore` excludes nothing, an `accept` excludes everything. Reaches `ferx check` only since [#1465](https://github.com/FeRx-NLME/ferx-core/issues/1465), which made the check compile the clauses at all |
//! | `W_AMT_NOT_DOSED`         | record(s) carry `AMT != 0` but were **not** read as dose events, because `EVID` is neither 1 nor 4; their `AMT` was ignored. With no `EVID` column a nonzero `AMT` is enough to infer a dose, so this is a dataset that has one. Reported with the record and subject counts, and it wins over `W_NO_DOSES`, which is the generic backstop for a dataset carrying no `AMT` signal at all (#262) |
//! | `W_NO_DOSES`              | zero dose events parsed across every subject **although scored observations are present**. Silent on a dataset with no scored observations (an all-`EVID=2` / covariate-only file is not a fit), and silent when any subject carries a non-Gaussian observation — a TTE / survival or discrete endpoint legitimately has no PK dose, so warning there would be a false positive (#262) |
//! | `W_ALL_DOSES_ZERO`        | an `AMT` column is present and at least one dose event was parsed, but every parsed dose amount is exactly zero — so no drug enters the system and the objective is flat: the parameters will not move from their initial values. A warning rather than an error because the column is there and a genuinely dose-free-but-`AMT`-columned dataset is conceivable ([#753](https://github.com/FeRx-NLME/ferx-core/issues/753)) |
//! | `W_CMT_DEFAULTED`         | dose / observation rows assigned compartment 1 because the dataset has no `CMT` column, or the cell is missing or unparseable; reported when `CMT` selects something — more than one compartment a dose can reach (multi-state `[odes]`, or an analytical model whose `CMT=2` is a real target), a per-CMT scaling / error model / readout on either engine, an endpoint the row routes to, or a `[data_selection]` clause comparing `CMT` (the scope is the `CmtConsumer` enumeration in `api::validation`) |
//! | `E_COVSTAT_UNRESOLVED`    | a `[covariate_model]` relation still needs data-derived statistics (`center = median`, `levels = auto`, or a form whose default bounds come from the data) |
//! | `W_COVSTAT_UNBOUND`       | the same, reported without a `--data` file — the model is fine, it just cannot be built until a dataset is supplied |
//! | `E_COVARIATE_STATS_BINDING` | a `[covariate_model]` statistic could not be bound, either on the data (a `levels = auto` relation whose data carry one level, which leaves nothing to estimate, or a covariate with no non-missing value) or from a fit (statistics that lack a covariate a symbolic relation reads, carry one no relation reads, or a fit that recorded none on a model with no level block). `ferx check --data` reports the first kind; `bind_covariate_stats`, `bind_from_fit` and `layout_from_fit` return both, and `run_sir` / `run_covariance` return the second, since they call the from-fit binder. Block `covariate_model`; the message is the binder's own (#1739, #1773) |
//! | `E_THETA_LEVEL_BINDING`   | a `[parameters]` level block (`theta NAME[COL, ...]`) could not be bound, either to the data's own levels (a level column missing or non-finite, no observation rows, a contrast the data cannot carry) or to a fit's level layout (a data level the fit estimated no θ for, level bindings that do not match the model's blocks, or a fit that recorded none). `ferx check --data` reports the first kind; the binders (`bind_theta_levels`, `bind_from_fit`, `layout_from_fit`, `bind_theta_levels_from_fit`) return both, and so do `run_sir` / `run_covariance`, which call them. A fit with no bindings, on a model with a level block and a symbolic statistic, is this code. Block `parameters` (#1773) |
//! | `E_THETA_LEVELS_DATA_UNBOUND` | a model whose `theta NAME[...]` block is bound was handed a population never bound for it, so no record carries an index into the block's levels: the usual "fit, then predict or simulate new data" without binding the new data. Raised by `fit()`, `predict` / `predict_diag`, the `simulate*` and adaptive entry points and `compute_npde_npd`. The binder named is `bind_from_fit` on the paths that run a θ already laid out, and `bind_theta_levels` on a fresh parse for `fit()`; the engine-internal index column is never named (#1647). Also raised by `predict_survival` / `predict_categorical` (#1763). A population only some of whose subjects carry the index gets "`k` of `n` subjects" and the first such ID instead of "never bound" (#1762) |
//! | `E_THETA_LEVELS_DATA_MISMATCH` | a population that carries a bound block's index was bound for other levels than the model's (#1762): a record's index is not the position, in the model's level table, of the label its own level columns spell — a population bound on its own for another dataset's studies or times, or a subset of the model's levels with its own count. Every unseen label is listed; or the first misindexed record (subject, time, label, both indices) is named, with the same binder as `E_THETA_LEVELS_DATA_UNBOUND`; or an index without its level column is named, since its level cannot be checked. Raised by the same entry points. Refused, never re-indexed |
//! | `E_MODEL_READ`            | the model file could not be read: it does not exist, cannot be opened, or is not UTF-8. Raised by `ferx check` before any parse; the message names the path and the operating system's reason. No block, since no block is at fault. The file is read once per check, and that one read is parsed and bound to the data, so a file moved or edited while the check runs is not read again (#1743, #1752) |
//! | `E_COV_LEVEL_UNKNOWN`     | the data carry a value of a categorical `[covariate_model]` covariate that is not one of the relation's levels, which the generated factor would model as the reference. Raised by `fit()`, `ferx check`, the `simulate*` and adaptive entry points, `predict` / `predict_diag` / `predict_survival` / `predict_categorical`, `compute_npde_npd`, and `run_sir` / `run_covariance` (#1111, #1740); not by the per-method estimators `fit()` dispatches to, which rely on its check |
//! | `E_THETA_LENGTH`          | an entry point taking a parameter vector was handed a θ whose length is not the model's; refused rather than read by position (a short vector read `0.0` past its end). Raised by the `simulate*` and adaptive entry points (#1614), `predict` / `predict_diag` / `predict_survival` / `predict_categorical` and `compute_npde_npd` (#1615); never by `ferx check`, which has no parameter vector. On a level-block model the message says the θ count is set by the bound data, so a fit's θ fits only a design bound against that fit's level bindings (#1614); it names no function, since the R wrapper reaches it too (#1623). `fit()` refuses its initial parameters with the same message (#1764) |
//! | `E_PARAM_SHAPE`           | an entry point was handed an Ω, σ or Ω_IOV whose dimension is not the model's, an Ω_IOV on a model with no κ, or none on a model with κ; refused rather than read by position (a mis-sized block panicked; a long σ or a missing Ω_IOV was silently ignored). Raised by `fit()` (as its message), the `simulate*` and adaptive entry points, `simulate_with_uncertainty` and `compute_npde_npd`, and — for σ and Ω_IOV — `run_sir` / `run_covariance` (#1764). Not by `predict*`, which read none of the three; on `simulate*` a missing Ω_IOV keeps its own uncoded message (#1019), and `compute_npde_npd` falls back to κ = 0 for one, as documented there |

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

/// Collapse a slice of diagnostics to a fail-fast `Result`: `Err` carrying the
/// first error-severity [`Diagnostic`], else `Ok`. This lets the entry points
/// keep their fail-fast behavior and identical error strings (the `Display` of
/// the returned [`EngineError`] is the diagnostic's message) while sharing the
/// diagnostic-producing validators with `ferx check` — and, since #1746, while
/// handing the caller the same code, block and suggestion `ferx check` reports.
pub fn first_error(diagnostics: &[Diagnostic]) -> Result<(), EngineError> {
    match diagnostics.iter().find(|d| d.is_error()) {
        Some(d) => Err(EngineError::from_diagnostic(d.clone())),
        None => Ok(()),
    }
}

/// The error a non-`fit` entry point (`predict*`, `simulate*`, `compute_npde_npd`,
/// [`crate::run_sir`], [`crate::run_covariance`], `inits_from_nca`) returns when it
/// refuses its input (#1746).
///
/// When `ferx check` reports the same refusal with a code, the error carries that
/// [`Diagnostic`] — so a caller matches on [`code`](Self::code) instead of parsing
/// the message. A refusal `ferx check` has no code for carries none
/// ([`diagnostic`](Self::diagnostic) is `None`); a code is never invented at the
/// entry point.
///
/// `Display` is the historical `String` error byte for byte: the message, preceded
/// by `"{context}: "` when the refusal names its entry point (`run_sir`,
/// `run_covariance`). `.to_string()` therefore recovers the pre-#1746 `Err`.
///
/// One refusal's historical text also folds its suggestion in after the message
/// (`predict`'s unbound `theta NAME[...]` block). There `Display` appends it, while
/// [`message`](Self::message) stays the bare message and the suggestion is only in
/// [`suggestion`](Self::suggestion). So a renderer shows either `to_string()` alone,
/// or `message()` and `suggestion()` side by side — never the advice twice.
#[derive(Debug, Clone)]
pub struct EngineError {
    diagnostic: Option<Diagnostic>,
    message: String,
    context: Option<String>,
    /// `Display` appends the diagnostic's suggestion as a sentence after the message.
    suggestion_in_display: bool,
}

impl EngineError {
    /// An error carrying `d`, whose message is `d.message`.
    pub(crate) fn from_diagnostic(d: Diagnostic) -> Self {
        EngineError {
            message: d.message.clone(),
            diagnostic: Some(d),
            context: None,
            suggestion_in_display: false,
        }
    }

    /// An error carrying `d` whose `Display` folds the suggestion in after the
    /// message, capitalised and closed with a full stop — for a refusal whose
    /// historical text read that way. [`message`](Self::message) stays `d.message`.
    pub(crate) fn with_suggestion_in_display(d: Diagnostic) -> Self {
        EngineError {
            suggestion_in_display: true,
            ..EngineError::from_diagnostic(d)
        }
    }

    /// This error, attributed to the entry point `context` (e.g. `"run_sir"`).
    pub(crate) fn in_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// The diagnostic `ferx check` reports for this refusal, if it has one.
    pub fn diagnostic(&self) -> Option<&Diagnostic> {
        self.diagnostic.as_ref()
    }

    /// The diagnostic's stable code (e.g. `"E_COV_LEVEL_UNKNOWN"`), if any.
    pub fn code(&self) -> Option<&str> {
        self.diagnostic.as_ref().map(|d| d.code.as_str())
    }

    /// The model block the diagnostic belongs to, if known.
    pub fn block(&self) -> Option<&str> {
        self.diagnostic.as_ref().and_then(|d| d.block.as_deref())
    }

    /// The diagnostic's actionable hint, if any.
    pub fn suggestion(&self) -> Option<&str> {
        self.diagnostic
            .as_ref()
            .and_then(|d| d.suggestion.as_deref())
    }

    /// The message, without the entry-point context and without the suggestion.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The entry point the refusal names, e.g. `Some("run_sir")`.
    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(c) = &self.context {
            write!(f, "{c}: ")?;
        }
        f.write_str(&self.message)?;
        if self.suggestion_in_display {
            if let Some(s) = self.suggestion() {
                let mut chars = s.chars();
                if let Some(first) = chars.next() {
                    write!(f, " {}{}.", first.to_uppercase(), chars.as_str())?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for EngineError {}

/// A failure with no diagnostic: [`EngineError::diagnostic`] is `None`.
impl From<String> for EngineError {
    fn from(message: String) -> Self {
        EngineError {
            diagnostic: None,
            message,
            context: None,
            suggestion_in_display: false,
        }
    }
}

/// A failure with no diagnostic: [`EngineError::diagnostic`] is `None`.
impl From<&str> for EngineError {
    fn from(message: &str) -> Self {
        message.to_string().into()
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
        let err = first_error(&diags).expect_err("an error-severity diagnostic");
        assert_eq!(err.to_string(), "second is the error");
    }

    /// #1746: the `Err` carries the whole diagnostic `ferx check` reports — code,
    /// block, line and suggestion — and its `Display` is the message alone, as the
    /// `String` it replaced was. A failure with no diagnostic carries no code.
    ///
    /// Mutations: build the error from `d.message` only (a `From<String>`) → the
    /// `code()` assert dies; drop `suggestion` / `block` in the copy → that assert
    /// dies; prefix `Display` with the code → the `to_string()` asserts die.
    #[test]
    fn first_error_carries_the_diagnostic_and_displays_its_message() {
        let mut coded = Diagnostic::error("E_B", "second is the error")
            .with_block("covariate_model")
            .with_suggestion("bind it first");
        coded.line = Some(7);
        let diags = vec![Diagnostic::warning("W_A", "warn first"), coded];
        let err = first_error(&diags).expect_err("an error-severity diagnostic");
        assert_eq!(err.code(), Some("E_B"));
        assert_eq!(err.block(), Some("covariate_model"));
        assert_eq!(err.suggestion(), Some("bind it first"));
        assert_eq!(err.diagnostic().and_then(|d| d.line), Some(7));
        assert_eq!(err.message(), "second is the error");
        assert_eq!(err.context(), None);
        assert_eq!(err.to_string(), "second is the error");

        let plain = EngineError::from("no diagnostic here".to_string());
        assert!(plain.diagnostic().is_none());
        assert_eq!(plain.code(), None);
        assert_eq!(plain.to_string(), "no diagnostic here");

        // Context is printed ahead of the message and kept out of `message()`.
        let attributed = plain.in_context("run_sir");
        assert_eq!(attributed.context(), Some("run_sir"));
        assert_eq!(attributed.message(), "no diagnostic here");
        assert_eq!(attributed.to_string(), "run_sir: no diagnostic here");

        // A folded refusal prints the advice once, after the message, while `message()`
        // and `suggestion()` keep them apart (review r1 #6). Mutations: fold into
        // `message` instead → the `message()` assert dies; drop the fold from `Display`
        // → the `to_string()` assert dies; skip the capital → it dies too.
        let folded = EngineError::with_suggestion_in_display(
            Diagnostic::error("E_X", "Short.").with_suggestion("with the advice folded in"),
        )
        .in_context("predict");
        assert_eq!(folded.code(), Some("E_X"));
        assert_eq!(folded.message(), "Short.");
        assert_eq!(folded.suggestion(), Some("with the advice folded in"));
        assert_eq!(
            folded.to_string(),
            "predict: Short. With the advice folded in."
        );
        // The fold is the exception: an ordinary diagnostic with a suggestion prints
        // its message alone.
        let plain_coded = EngineError::from_diagnostic(
            Diagnostic::error("E_X", "Short.").with_suggestion("advice"),
        );
        assert_eq!(plain_coded.to_string(), "Short.");
    }

    #[test]
    fn first_error_ok_when_no_errors() {
        let diags = vec![Diagnostic::warning("W_A", "just a warning")];
        assert!(first_error(&diags).is_ok());
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
