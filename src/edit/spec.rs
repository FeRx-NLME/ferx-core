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
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorSpecText {
    /// The observed quantity, almost always `DV`.
    pub endpoint: String,
    pub form: ErrorForm,
    /// The σ the form takes, in order, with the init to declare for any that
    /// `[parameters]` does not already have.
    pub sigmas: Vec<SigmaDecl>,
}

/// The residual-error forms `[error_model]` accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorForm {
    Additive,
    Proportional,
    Combined,
}

impl ErrorForm {
    /// The spelling used in the block.
    pub fn label(&self) -> &'static str {
        match self {
            ErrorForm::Additive => "additive",
            ErrorForm::Proportional => "proportional",
            ErrorForm::Combined => "combined",
        }
    }

    /// How many σ the form takes.
    pub fn n_sigma(&self) -> usize {
        match self {
            ErrorForm::Additive | ErrorForm::Proportional => 1,
            ErrorForm::Combined => 2,
        }
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

/// Format a float for a `.ferx` file: shortest round-tripping form, with a
/// `.0` tail so an integral value still reads as a number rather than a count.
pub(crate) fn num(v: f64) -> String {
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
