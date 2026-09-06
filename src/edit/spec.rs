//! The edit vocabulary: [`ModelEdit`] and the value types its variants carry.
//!
//! Each type here is an *authoring* shape — what a caller states — and knows
//! how to render itself as the one `.ferx` line it owns. The parse-side
//! structures in [`crate::types`] are the complement: they carry what a parse
//! *derived* (resolved centres, generated θ, the verbatim source line), which
//! is exactly the information an edit must not have to supply.

use crate::types::{CovariateForm, CovariateRelation, CovariateStat, FitResult};

/// One transformation of a [`ModelText`](super::ModelText).
///
/// The variants are the moves a model-space search makes: swap the structural
/// model, add or drop a covariate relation, add or drop an η, block two ηs
/// together, change the residual error model, carry the parent's estimates
/// into the child, or set a fit option.
///
/// No `PartialEq`: [`SeedInits`](ModelEdit::SeedInits) borrows a
/// [`FitResult`], which is not comparable (it carries `f64` estimates, an
/// optional covariance matrix and per-subject diagnostics — "the same fit" is
/// not a well-defined question).
#[derive(Debug, Clone)]
pub enum ModelEdit<'a> {
    /// Replace the `pk NAME(...)` line, adding the `[parameters]` and
    /// `[individual_parameters]` declarations the new template needs and
    /// pruning the ones nothing references any more.
    SetStructural(StructuralSpec),
    /// Add one `[covariate_model]` line, creating the block if absent.
    AddCovariateRelation(Relation),
    /// Remove the `[covariate_model]` line for one `(parameter, covariate)`
    /// pair, and any `[parameters]` θ it leaves unreferenced.
    DropCovariateRelation { param: String, cov: String },
    /// Give `param` an η, in canonical `TVP * exp(ETA_P)` form.
    AddIiv { param: String, form: IivForm },
    /// Take `param`'s η away: drop the `exp(ETA_P)` factor and the `omega`
    /// declaration that goes with it.
    DropIiv { param: String },
    /// Replace the named ηs' diagonal `omega` declarations with one
    /// `block_omega (A, B, …) = [...]`.
    SetOmegaBlock(Vec<String>),
    /// Replace the `[error_model]` block, reconciling `[parameters]` σ
    /// declarations with the σ the new model names.
    SetErrorModel(ErrorSpecText),
    /// Carry a parent fit's estimates into this model's initial values,
    /// **keyed by name** — `theta_names` / `eta_names` / `sigma_names`, never
    /// by position, because a candidate's parameter vector differs from its
    /// parent's. This is Pharmpy's `update_inits` between search steps, and
    /// what `bootstrap` already does per replicate.
    SeedInits(&'a FitResult),
    /// Set one `[fit_options]` key, creating the block if absent.
    SetFitOption { key: String, value: String },
}

/// The `[structural_model]` `pk NAME(role=VAR, …)` line, plus the defaults for
/// any `[individual_parameters]` value the new template needs and the model
/// does not yet have.
///
/// The defaults are stated rather than inferred on purpose: a search that
/// widens `one_cpt_oral` to `two_cpt_oral` has to choose an init and bounds
/// for `Q` and `V2`, and choosing them here — visibly, per candidate — is
/// better than an engine-internal table nobody can see or override.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralSpec {
    /// A `pk` template name, e.g. `two_cpt_oral`. Validated against the same
    /// name table the parser uses, so a typo fails at edit time.
    pub template: String,
    /// `role = variable` bindings in the order they should be written, e.g.
    /// `[("cl", "CL"), ("v1", "V1"), ("q", "Q"), ("v2", "V2"), ("ka", "KA")]`.
    pub bindings: Vec<(String, String)>,
    /// Declarations to create for bindings that name an
    /// `[individual_parameters]` value the model does not have. A binding onto
    /// an existing name needs no entry; an entry for a name that already
    /// exists is ignored.
    pub new_parameters: Vec<NewParameter>,
}

/// One `[individual_parameters]` value to create, with the θ (and optional η)
/// behind it — written in the canonical `P = TVP * exp(ETA_P)` form.
///
/// `#[non_exhaustive]`: build one with [`NewParameter::new`] and the
/// [`with_iiv`](NewParameter::with_iiv) / [`fixed`](NewParameter::fixed)
/// builders, so the next field this struct grows (as `fixed` did on #1181)
/// does not break a caller's struct literal. The fields stay public to read.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NewParameter {
    /// The individual-parameter name, e.g. `Q`.
    pub name: String,
    /// Its typical-value θ, e.g. `TVQ`.
    pub theta: String,
    pub init: f64,
    pub lower: f64,
    pub upper: f64,
    /// `(eta name, omega variance)` when the new parameter gets an η.
    pub iiv: Option<(String, f64)>,
    /// Declare the θ `FIX`: `theta NAME(init, lower, upper) FIX`. A structural
    /// constant stated as a parameter — a fixed transit-compartment count
    /// (`TRANSITS(3)`) is the case — so that it still has a name the ODE twin
    /// and the estimates file can see, unlike a literal `n=3` binding.
    pub fixed: bool,
}

impl NewParameter {
    /// A free θ with no η: `NAME = THETA`, `theta THETA(init, lower, upper)`.
    pub fn new(
        name: impl Into<String>,
        theta: impl Into<String>,
        init: f64,
        lower: f64,
        upper: f64,
    ) -> Self {
        NewParameter {
            name: name.into(),
            theta: theta.into(),
            init,
            lower,
            upper,
            iiv: None,
            fixed: false,
        }
    }

    /// Give the parameter an η: `NAME = THETA * exp(ETA)`, `omega ETA ~ variance`.
    pub fn with_iiv(mut self, eta: impl Into<String>, variance: f64) -> Self {
        self.iiv = Some((eta.into(), variance));
        self
    }

    /// Declare the θ `FIX`.
    pub fn fixed(mut self) -> Self {
        self.fixed = true;
        self
    }
}

/// How an η enters a parameter's expression.
///
/// Only the exponential form is canonical for this module: it is the one
/// [`ModelEdit::DropIiv`] can reverse by pattern, and the one the SAEM
/// mu-reference detector reads (#619). The other two are accepted because
/// models are written with them, but a parameter carrying one cannot be
/// searched over — dropping it is a hard error naming the parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum IivForm {
    /// `P = TVP * exp(ETA_P)` — log-normal IIV, the canonical form.
    Exponential { eta: String, variance: f64 },
    /// `P = TVP + ETA_P` — normal IIV.
    Additive { eta: String, variance: f64 },
    /// `P = TVP * (1 + ETA_P)` — proportional IIV.
    Proportional { eta: String, variance: f64 },
}

impl IivForm {
    /// The η this form introduces.
    pub fn eta(&self) -> &str {
        match self {
            IivForm::Exponential { eta, .. }
            | IivForm::Additive { eta, .. }
            | IivForm::Proportional { eta, .. } => eta,
        }
    }

    /// Its `omega` initial variance.
    pub fn variance(&self) -> f64 {
        match self {
            IivForm::Exponential { variance, .. }
            | IivForm::Additive { variance, .. }
            | IivForm::Proportional { variance, .. } => *variance,
        }
    }
}

/// A residual-error model as it is written in `[error_model]`.
///
/// `Text` in the name is the point: this is the *source* form, not
/// [`crate::types::ErrorSpec`], which is the compiled one with σ resolved to
/// indices into the global σ vector.
///
/// Beyond the form and its σ, the spec carries the three residual-error
/// features a search over the error model proposes (#1182): a `power(...)`
/// exponent, `iiv_on_ruv`, and Pharmpy's time-varying magnitude. Each renders
/// in **one canonical spelling**, and [`ErrorSpecText::read`] reads exactly
/// those spellings back — so a tool can take a model's `[error_model]` apart,
/// change one feature, and write it again without knowing the block's text.
///
/// `#[non_exhaustive]`: build one with [`ErrorSpecText::new`] and the
/// `with_*` builders, so the next feature added here does not break a caller's
/// struct literal. The fields stay public to read.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ErrorSpecText {
    /// The observed quantity, almost always `DV`.
    pub endpoint: String,
    pub form: ErrorForm,
    /// The σ the form takes, in order, with the init to declare for any that
    /// `[parameters]` does not already have.
    pub sigmas: Vec<SigmaDecl>,
    /// The exponent θ of a [`ErrorForm::Power`] form — `DV ~ power(σ, P)`,
    /// NONMEM's `Y = F + EPS(1) * F**THETA`. Required for that form, refused
    /// for every other.
    pub exponent: Option<ThetaDecl>,
    /// `iiv_on_ruv = ETA`, with the `omega ETA ~ variance` to declare when
    /// `[parameters]` does not already have it.
    pub iiv_on_ruv: Option<EtaDecl>,
    /// Pharmpy's time-varying residual error (`set_time_varying_error_model`):
    /// every σ of the form is multiplied by `θ` for records with
    /// `TAD < cutoff`, written as the magnitude expression
    /// `σ * (if (TAD < cutoff) θ else 1.0)`.
    pub time_varying: Option<TimeVaryingDecl>,
}

impl ErrorSpecText {
    /// A plain form with its σ and none of the optional features.
    pub fn new(endpoint: impl Into<String>, form: ErrorForm, sigmas: Vec<SigmaDecl>) -> Self {
        ErrorSpecText {
            endpoint: endpoint.into(),
            form,
            sigmas,
            exponent: None,
            iiv_on_ruv: None,
            time_varying: None,
        }
    }

    /// The exponent θ of a `power(...)` form.
    pub fn with_exponent(mut self, theta: ThetaDecl) -> Self {
        self.exponent = Some(theta);
        self
    }

    /// `iiv_on_ruv = ETA`.
    pub fn with_iiv_on_ruv(mut self, eta: EtaDecl) -> Self {
        self.iiv_on_ruv = Some(eta);
        self
    }

    /// The time-varying magnitude `σ * (if (TAD < cutoff) θ else 1.0)`.
    pub fn with_time_varying(mut self, tv: TimeVaryingDecl) -> Self {
        self.time_varying = Some(tv);
        self
    }

    /// Read a model's `[error_model]` back into authoring form — the inverse
    /// of [`ModelEdit::SetErrorModel`](super::ModelEdit::SetErrorModel).
    ///
    /// Reads exactly the spellings the edit writes: one
    /// `DV ~ form(σ, …[, P])` statement whose σ arguments are bare names or the
    /// canonical time-varying magnitude `σ * (if (TAD < c) θ else 1.0)`, plus
    /// an optional `iiv_on_ruv = ETA` line, with each σ / θ / ω looked up in
    /// `[parameters]` for its init. `Ok(None)` when the model has no
    /// `[error_model]` block at all.
    ///
    /// # Errors
    ///
    /// A block this cannot represent — per-CMT or covariate-selected
    /// endpoints, a log-transform, a `weight =` modifier, a hand-written
    /// magnitude expression, a σ in a `block_sigma`, an η in a `block_omega`,
    /// a declaration the statement names but `[parameters]` lacks — is an
    /// error naming what was found, never a lossy reading. A tool that edits
    /// the error model must refuse such a base rather than rewrite it.
    pub fn read(text: &super::ModelText) -> Result<Option<Self>, String> {
        super::apply::read_error_spec(text)
    }

    /// The σ argument of slot `i` as the block spells it — the bare name, or
    /// the time-varying magnitude around it.
    pub(crate) fn render_sigma_arg(&self, i: usize) -> String {
        let name = &self.sigmas[i].name;
        match &self.time_varying {
            Some(tv) => format!(
                "{name} * (if (TAD < {}) {} else 1.0)",
                num(tv.cutoff),
                tv.theta.name
            ),
            None => name.clone(),
        }
    }

    /// The `DV ~ form(...)` statement.
    pub fn render_statement(&self) -> String {
        let mut args: Vec<String> = (0..self.sigmas.len())
            .map(|i| self.render_sigma_arg(i))
            .collect();
        if let Some(p) = &self.exponent {
            args.push(p.name.clone());
        }
        format!(
            "{} ~ {}({})",
            self.endpoint,
            self.form.label(),
            args.join(", ")
        )
    }
}

/// The residual-error forms `[error_model]` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorForm {
    Additive,
    Proportional,
    Combined,
    /// `power(σ, P)`: the proportional loading raised to the θ `P` (#1182).
    Power,
}

impl ErrorForm {
    /// The spelling used in the block.
    pub fn label(&self) -> &'static str {
        match self {
            ErrorForm::Additive => "additive",
            ErrorForm::Proportional => "proportional",
            ErrorForm::Combined => "combined",
            ErrorForm::Power => "power",
        }
    }

    /// How many σ the form takes.
    pub fn n_sigma(&self) -> usize {
        match self {
            ErrorForm::Additive | ErrorForm::Proportional | ErrorForm::Power => 1,
            ErrorForm::Combined => 2,
        }
    }

    /// The form spelled `label`, if any.
    pub fn from_label(label: &str) -> Option<ErrorForm> {
        match label.to_ascii_lowercase().as_str() {
            "additive" => Some(ErrorForm::Additive),
            "proportional" => Some(ErrorForm::Proportional),
            "combined" => Some(ErrorForm::Combined),
            "power" => Some(ErrorForm::Power),
            _ => None,
        }
    }
}

/// One `theta NAME(init, lower, upper)` declaration an error-model feature
/// needs — the `power(...)` exponent or the time-varying multiplier.
#[derive(Debug, Clone, PartialEq)]
pub struct ThetaDecl {
    pub name: String,
    pub init: f64,
    pub lower: f64,
    pub upper: f64,
}

impl ThetaDecl {
    pub fn new(name: impl Into<String>, init: f64, lower: f64, upper: f64) -> Self {
        ThetaDecl {
            name: name.into(),
            init,
            lower,
            upper,
        }
    }

    /// `theta NAME(init, lower, upper)` — or `theta NAME(init)` when both
    /// bounds are infinite, which is how [`ErrorSpecText::read`] reports a
    /// declaration written without bounds.
    pub(crate) fn render(&self) -> String {
        if !self.lower.is_finite() && !self.upper.is_finite() {
            return format!("theta {}({})", self.name, num(self.init));
        }
        format!(
            "theta {}({}, {}, {})",
            self.name,
            num(self.init),
            num(self.lower),
            num(self.upper)
        )
    }
}

/// One `omega NAME ~ variance` declaration — the `iiv_on_ruv` η.
#[derive(Debug, Clone, PartialEq)]
pub struct EtaDecl {
    pub name: String,
    /// The initial **variance**.
    pub variance: f64,
}

impl EtaDecl {
    pub fn new(name: impl Into<String>, variance: f64) -> Self {
        EtaDecl {
            name: name.into(),
            variance,
        }
    }

    /// `omega NAME ~ variance`.
    pub(crate) fn render(&self) -> String {
        format!("omega {} ~ {}", self.name, num(self.variance))
    }
}

/// Pharmpy's time-varying residual error: the σ multiplied by `theta` for
/// records whose time after dose is below `cutoff`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeVaryingDecl {
    /// The `TAD` cutoff, in the data's time unit.
    pub cutoff: f64,
    /// The multiplier θ.
    pub theta: ThetaDecl,
}

impl TimeVaryingDecl {
    pub fn new(cutoff: f64, theta: ThetaDecl) -> Self {
        TimeVaryingDecl { cutoff, theta }
    }
}

/// One `sigma NAME ~ init [(sd)]` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct SigmaDecl {
    pub name: String,
    /// The initial value, on the scale `as_sd` names.
    pub init: f64,
    /// `true` writes `(sd)` and treats `init` as a standard deviation; `false`
    /// writes the bare variance form. The annotation is preserved on an
    /// existing declaration rather than taken from here.
    pub as_sd: bool,
}

impl SigmaDecl {
    /// `sigma NAME ~ init (sd)`.
    pub(crate) fn render(&self) -> String {
        let scale = if self.as_sd { " (sd)" } else { "" };
        format!("sigma {} ~ {}{scale}", self.name, num(self.init))
    }
}

/// One `[covariate_model]` line, in authoring form.
///
/// The parse-side [`CovariateRelation`] carries the θ a parse *generated* and
/// the centre it *resolved*; neither is something a caller can state, so this
/// type omits both. [`From<&CovariateRelation>`] converts a relation read back
/// off a fit into an edit, which is how a search re-proposes a relation it
/// found in the parent model.
#[derive(Debug, Clone, PartialEq)]
pub struct Relation {
    /// The `[individual_parameters]` name the factor multiplies into, e.g. `CL`.
    pub parameter: String,
    /// The `[covariates]` column, e.g. `WT`.
    pub covariate: String,
    pub form: CovariateForm,
    /// `center = …` / `breakpoint = …` / `ref = …`. `None` writes no keyword
    /// and takes the block's default (the covariate's median, or its mode for
    /// `categorical`); `none` and `expr(...)` take no constant at all.
    pub center: Option<CovariateStat>,
    /// `fix = v`, when the relation's θ are to be held.
    pub fix: Option<f64>,
    /// An explicit `=> NAME(init, lower, upper)[, …]` clause. Empty leaves the
    /// θ auto-named with the block's default init and bounds.
    pub thetas: Vec<RelationTheta>,
}

/// One θ of an explicit `=> …` clause on a [`Relation`].
#[derive(Debug, Clone, PartialEq)]
pub struct RelationTheta {
    pub name: String,
    pub init: f64,
    pub lower: f64,
    pub upper: f64,
}

impl From<&CovariateRelation> for Relation {
    fn from(r: &CovariateRelation) -> Self {
        Relation {
            parameter: r.parameter.clone(),
            covariate: r.covariate.clone(),
            form: r.form.clone(),
            center: r.center,
            fix: r.fix,
            thetas: r
                .thetas
                .iter()
                .map(|t| RelationTheta {
                    name: t.name.clone(),
                    init: t.init,
                    lower: t.lower,
                    upper: t.upper,
                })
                .collect(),
        }
    }
}

impl Relation {
    /// `PARAM ~ COV form(kwargs) [=> THETA(init, lower, upper), …]`.
    pub(crate) fn render(&self) -> String {
        let mut line = format!(
            "{} ~ {} {}",
            self.parameter,
            self.covariate,
            self.form_src()
        );
        if !self.thetas.is_empty() {
            let clause: Vec<String> = self
                .thetas
                .iter()
                .map(|t| {
                    format!(
                        "{}({}, {}, {})",
                        t.name,
                        num(t.init),
                        num(t.lower),
                        num(t.upper)
                    )
                })
                .collect();
            line.push_str(" => ");
            line.push_str(&clause.join(", "));
        }
        line
    }

    /// The `form(...)` token, with the centring keyword the form actually
    /// takes — `breakpoint` for `hockey`, `ref` for `categorical`, `center`
    /// for the rest. Getting this wrong is a parse error, not a silent
    /// mis-centre, but there is no reason to make the caller carry it.
    fn form_src(&self) -> String {
        if let CovariateForm::Expr(src) = &self.form {
            return format!("expr(\"{src}\")");
        }
        let keyword = match self.form {
            CovariateForm::Hockey => "breakpoint",
            CovariateForm::Categorical => "ref",
            _ => "center",
        };
        let mut args: Vec<String> = Vec::new();
        // `none` declares no θ and has no constant, so neither keyword applies.
        if !matches!(self.form, CovariateForm::None) {
            if let Some(stat) = self.center {
                args.push(format!("{keyword} = {}", stat.label()));
            }
            if let Some(v) = self.fix {
                args.push(format!("fix = {}", num(v)));
            }
        }
        if args.is_empty() {
            self.form.label().to_string()
        } else {
            format!("{}({})", self.form.label(), args.join(", "))
        }
    }
}

/// Format a float for a `.ferx` file: the shortest round-tripping form of the
/// value rounded to 15 significant digits, with a `.0` tail so an integral
/// value still reads as a number rather than a count.
///
/// The rounding is the point. An estimate that went through the optimizer's
/// log/exp packing comes back one ULP off the number it started from —
/// `10.000000000000002` for an evaluation at `10.0` — and written verbatim
/// that reads as a bug in the generated file. Fifteen significant digits
/// keep every value a user could have typed (a `.ferx` file carries far
/// fewer) and drop the last-ULP noise; the value that is read back can
/// differ from the estimate by at most one part in 10¹⁵, which is below
/// anything a fit can tell apart.
pub(crate) fn num(v: f64) -> String {
    let v = if v.is_finite() && v != 0.0 {
        // `.is_finite()` on the *result*, not just the input: rounding a value
        // within one ULP of `f64::MAX` rounds it up past the maximum, and
        // Rust's float parser returns `Ok(f64::INFINITY)` on overflow rather
        // than `Err`, so `unwrap_or` alone would let `inf` through and write
        // it into the file as a bound.
        format!("{v:.14e}")
            .parse::<f64>()
            .ok()
            .filter(|r| r.is_finite())
            .unwrap_or(v)
    } else {
        v
    };
    let s = format!("{v}");
    if s.contains(['.', 'e', 'E', 'n', 'i']) {
        s
    } else {
        format!("{s}.0")
    }
}

/// Reject a value that cannot be written back and read as itself.
///
/// `NaN` and `±inf` format as `NaN` / `inf`, which the `[0-9eE.+-]+` numeric
/// regexes do not match — the declaration would silently stop being a
/// declaration. Catching it here names the field instead.
pub(crate) fn finite(v: f64, what: &str) -> Result<(), String> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(format!(
            "ferx-core::edit: {what} is {v}, which cannot be written as a `.ferx` number"
        ))
    }
}
