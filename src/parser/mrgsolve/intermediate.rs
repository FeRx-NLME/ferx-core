//! Intermediate representation of a parsed mrgsolve model.
//!
//! `MrgModel` is the structured form produced by the block parsers and
//! consumed by the `.ferx` emitter. C++ body blocks (`$MAIN`, `$ODE`,
//! `$TABLE`) are kept as preprocessed strings — `ETA(N)`/`THETA(N)`/`EPS(N)`
//! resolved to names, `dxdt_X` rewritten as `d/dt(X)`, `double`/`int`/`bool`
//! stripped, `;` replaced with newline — so the emitter can write them
//! straight into the corresponding `.ferx` block without further surgery.

#[derive(Debug, Clone, Default)]
pub(crate) struct MrgModel {
    /// Model name from `$PROB` or the source filename stem.
    pub(crate) model_name: String,
    /// `$PARAM NAME = value, ...` and `$THETA value, value, ...` declarations,
    /// in declaration order. `THETA(N)` (1-indexed) resolves to the Nth
    /// entry here.
    pub(crate) thetas: Vec<NamedValue>,
    /// `$INPUT NAME, ...` declarations (variables without a `= value`).
    /// These are model covariates — they come from the per-record data
    /// rather than from the parameter vector.
    pub(crate) covariates: Vec<String>,
    /// `$OMEGA` blocks, in declaration order. `ETA(N)` indexes into the
    /// flattened sequence of all blocks' labels.
    pub(crate) omegas: Vec<OmegaBlock>,
    /// `$SIGMA` blocks, in declaration order. `EPS(N)` indexes into the
    /// flattened sequence of all blocks' labels.
    pub(crate) sigmas: Vec<OmegaBlock>,
    /// `$CMT`/`$INIT` compartments, in declaration order.
    pub(crate) compartments: Vec<Compartment>,
    /// `$MAIN` / `$PK` body, preprocessed to ferx syntax.
    pub(crate) main_body: Option<PreprocessedBody>,
    /// `$ODE` body, preprocessed to ferx syntax. Each `dxdt_X = ...;` line
    /// has been rewritten to `d/dt(X) = ...`.
    pub(crate) ode_body: Option<PreprocessedBody>,
    /// `$TABLE` / `$ERROR` body, preprocessed to ferx syntax.
    pub(crate) error_body: Option<PreprocessedBody>,
    /// `$CAPTURE NAME, ...` declarations.
    pub(crate) captures: Vec<String>,
    /// `$PKMODEL ncmt=N, depot=BOOL, cmt="..."` if present.
    pub(crate) pkmodel: Option<PkModelSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NamedValue {
    pub(crate) name: String,
    pub(crate) value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OmegaBlock {
    /// `true` if declared with `@block` (lower-triangle matrix).
    pub(crate) block_form: bool,
    /// Names for the etas/epsilons in this block. From `@labels` if given,
    /// else auto-generated as `ETA1..ETAn` / `EPS1..EPSn` numbered across
    /// all blocks.
    pub(crate) labels: Vec<String>,
    /// Diagonal: n variance values. Block: n(n+1)/2 lower-triangular entries
    /// (row-major: ω11, ω21, ω22, ω31, ω32, ω33, ...).
    pub(crate) values: Vec<f64>,
}

impl OmegaBlock {
    /// Number of etas/epsilons this block declares.
    pub(crate) fn dim(&self) -> usize {
        self.labels.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Compartment {
    pub(crate) name: String,
    pub(crate) initial: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PkModelSpec {
    pub(crate) ncmt: u8,
    pub(crate) depot: bool,
    /// Optional `cmt="A,B,..."` override for compartment names.
    pub(crate) cmt_names: Option<Vec<String>>,
}

/// A C++ body block that has been preprocessed into ferx-compatible source.
/// Carries the original `start_line` so downstream parse errors can be
/// reported back at the source file.
#[derive(Debug, Clone)]
pub(crate) struct PreprocessedBody {
    /// 1-indexed line number of the `$BLOCK` opener in the source file.
    /// Reserved for use by future source-tracking error messages.
    #[allow(dead_code)]
    pub(crate) start_line: usize,
    /// Ferx-syntax body lines, ready to drop into an `[individual_parameters]`
    /// / `[odes]` / `[error_model]` block.
    pub(crate) text: String,
}
