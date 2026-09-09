//! Allometric scaling — Pharmpy's `allometry` tool (#1180).
//!
//! Multiply every clearance-like parameter by `(WT / 70)^0.75` and every
//! volume-like one by `(WT / 70)^1.0`, fit, and report. There is no new
//! machinery: each scaling is one `[covariate_model]` line,
//! `CL ~ WT power(center = 70, fix = 0.75)`, and the tool is the *convention*
//! — which parameters get which exponent, at which reference weight — plus
//! the optional variant that estimates the exponents instead of fixing them.
//!
//! Parameters are classified by the role they are bound to on the base
//! model's `pk NAME(...)` / `ode_template NAME(...)` line: `cl`, `q`, `q2`,
//! `q3` are clearances, `v`, `v1`, `v2`, `v3` volumes. Nothing else — `ka`, a
//! lag time, a transit count — is scaled unless named explicitly, which is
//! Pharmpy's `find_clearance_parameters` / `find_volume_parameters` default.
//! A model without a template line (an `[odes]` model) has to name its
//! parameters and exponents.
//!
//! The base and the scaled model are fitted together through the candidate
//! runner so the report can show the ΔOFV; Pharmpy fits only the scaled
//! model and leaves the comparison to the caller.

use std::path::PathBuf;

use ferx_core::edit::{ModelEdit, ModelText, Relation, RelationTheta};
use ferx_core::{CancelFlag, CovariateForm, CovariateStat};
use serde::Deserialize;

use crate::search::{
    BaseModel, Candidate, CandidateResult, Criterion, Feature, FeatureVector, ModelContext, Runner,
    SearchConfig,
};

/// Template roles Pharmpy classes as clearances (exponent 0.75).
pub const CLEARANCE_ROLES: &[&str] = &["cl", "q", "q2", "q3"];
/// Template roles Pharmpy classes as volumes (exponent 1.0).
pub const VOLUME_ROLES: &[&str] = &["v", "v1", "v2", "v3"];

/// The `[allometry]` section of a `.ferxsearch` file, or the equivalent
/// command-line options.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllometryOptions {
    /// The size covariate, `WT` by default.
    #[serde(default = "default_covariate")]
    pub covariate: String,
    /// The reference value the covariate is divided by, 70 by default.
    #[serde(default = "default_reference")]
    pub reference: f64,
    /// The parameters to scale; `None` takes every clearance and volume the
    /// template line binds.
    #[serde(default)]
    pub parameters: Option<Vec<String>>,
    /// One exponent per entry of `parameters`; `None` uses 0.75 for a
    /// clearance and 1.0 for a volume, which needs every named parameter to
    /// have a template role.
    #[serde(default)]
    pub exponents: Option<Vec<f64>>,
    /// Fix the exponents (the default) or estimate them from the given
    /// values, bounded by `lower` / `upper`.
    #[serde(default = "default_fixed")]
    pub fixed: bool,
    #[serde(default = "default_lower")]
    pub lower: f64,
    #[serde(default = "default_upper")]
    pub upper: f64,
}

fn default_covariate() -> String {
    "WT".into()
}
fn default_reference() -> f64 {
    70.0
}
fn default_fixed() -> bool {
    true
}
fn default_lower() -> f64 {
    0.0
}
fn default_upper() -> f64 {
    2.0
}

impl Default for AllometryOptions {
    fn default() -> Self {
        AllometryOptions {
            covariate: default_covariate(),
            reference: default_reference(),
            parameters: None,
            exponents: None,
            fixed: default_fixed(),
            lower: default_lower(),
            upper: default_upper(),
        }
    }
}

impl AllometryOptions {
    /// Read the options from a loaded `.ferxsearch` file: the covariate and
    /// reference from its one `ALLOMETRY(cov[, ref])` statement, the rest
    /// from `[allometry]`.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        let mut statements = config.mfl.features().filter_map(|f| match f {
            Feature::Allometry {
                covariate,
                reference,
            } => Some((covariate.clone(), reference.clone())),
            _ => None,
        });
        let Some((covariate, reference)) = statements.next() else {
            return Err(
                "[space] mfl: allometry needs one `ALLOMETRY(WT, 70)` statement naming the \
                 size covariate and its reference value"
                    .into(),
            );
        };
        if statements.next().is_some() {
            return Err("[space] mfl: more than one ALLOMETRY statement; state it once".into());
        }
        let mut options = match config.tools.get("allometry") {
            Some(table) => table
                .clone()
                .try_into::<AllometryOptions>()
                .map_err(|e| format!("[allometry]: {e}"))?,
            None => AllometryOptions::default(),
        };
        options.covariate = covariate;
        if let Some(reference) = reference {
            options.reference = reference.parse().map_err(|_| {
                format!("[space] mfl: ALLOMETRY reference `{reference}` is not a number")
            })?;
        }
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.reference.is_finite() && self.reference > 0.0) {
            return Err(format!(
                "allometry: the reference value must be a positive number, not {}",
                self.reference
            ));
        }
        if let (Some(p), Some(e)) = (&self.parameters, &self.exponents) {
            if p.len() != e.len() {
                return Err(format!(
                    "allometry: {} parameter{} but {} exponent{}; give one exponent per parameter",
                    p.len(),
                    if p.len() == 1 { "" } else { "s" },
                    e.len(),
                    if e.len() == 1 { "" } else { "s" }
                ));
            }
        }
        if self.exponents.is_some() && self.parameters.is_none() {
            return Err("allometry: `exponents` needs `parameters` to say which is which".into());
        }
        if !self.fixed && self.lower >= self.upper {
            return Err(format!(
                "allometry: lower = {} must be below upper = {}",
                self.lower, self.upper
            ));
        }
        Ok(())
    }
}

/// One parameter's scaling.
#[derive(Debug, Clone, PartialEq)]
pub struct Scaling {
    pub parameter: String,
    pub exponent: f64,
    pub fixed: bool,
    /// The θ the relation declares when the exponent is estimated.
    pub theta: Option<String>,
}

impl Scaling {
    fn relation(&self, options: &AllometryOptions) -> Relation {
        Relation {
            parameter: self.parameter.clone(),
            covariate: options.covariate.clone(),
            form: CovariateForm::Power,
            center: Some(CovariateStat::Literal(options.reference)),
            fix: self.fixed.then_some(self.exponent),
            thetas: match &self.theta {
                Some(name) => vec![RelationTheta {
                    name: name.clone(),
                    init: self.exponent,
                    lower: options.lower,
                    upper: options.upper,
                }],
                None => Vec::new(),
            },
        }
    }
}

/// The scaled model and what was done to it.
#[derive(Debug, Clone)]
pub struct AllometricModel {
    pub model: ModelText,
    pub scalings: Vec<Scaling>,
    /// Parameters that already carried a relation on the covariate and were
    /// left alone, as Pharmpy does.
    pub notes: Vec<String>,
}

/// Build the allometric model from a base model.
pub fn allometric_model(
    base: &BaseModel,
    options: &AllometryOptions,
) -> Result<AllometricModel, String> {
    options.validate()?;
    let ctx =
        ModelContext::from_model(&base.prepared.parsed, &base.text, &base.prepared.population)?;
    if !ctx.all_covariates().contains(&options.covariate) {
        return Err(format!(
            "allometry: `{}` is not a covariate of the base model; known: {}",
            options.covariate,
            ctx.all_covariates().join(", ")
        ));
    }
    let (clearances, volumes) = classify(&ctx)?;

    let named: Vec<(String, Option<f64>)> = match &options.parameters {
        Some(params) => params
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), options.exponents.as_ref().map(|e| e[i])))
            .collect(),
        None => {
            let list: Vec<(String, Option<f64>)> = clearances
                .iter()
                .chain(volumes.iter())
                .map(|p| (p.clone(), None))
                .collect();
            if list.is_empty() {
                return Err(
                    "allometry: the template line binds no clearance or volume parameter, so \
                     there is nothing to scale by default; name `parameters` and `exponents`"
                        .into(),
                );
            }
            list
        }
    };

    let existing: Vec<(String, String)> = base
        .prepared
        .parsed
        .model
        .covariate_model
        .as_ref()
        .map(|spec| {
            spec.relations
                .iter()
                .map(|r| (r.parameter.clone(), r.covariate.clone()))
                .collect()
        })
        .unwrap_or_default();

    let mut model = base.text.clone();
    let mut scalings = Vec::new();
    let mut notes = Vec::new();
    for (parameter, exponent) in named {
        if !ctx.parameters.contains(&parameter) {
            return Err(format!(
                "allometry: `{parameter}` is not an individual parameter of the base model; it \
                 has: {}",
                ctx.parameters.join(", ")
            ));
        }
        let exponent = match exponent {
            Some(e) => e,
            None if clearances.contains(&parameter) => 0.75,
            None if volumes.contains(&parameter) => 1.0,
            None => {
                return Err(format!(
                    "allometry: `{parameter}` is bound to neither a clearance nor a volume role \
                     on the template line, so it has no default exponent; give `exponents`"
                ))
            }
        };
        if existing
            .iter()
            .any(|(p, c)| *p == parameter && *c == options.covariate)
        {
            notes.push(format!(
                "`{parameter}` already carries a relation on `{}`; left as written",
                options.covariate
            ));
            continue;
        }
        let scaling = Scaling {
            theta: (!options.fixed).then(|| format!("THETA_{parameter}_{}", options.covariate)),
            parameter,
            exponent,
            fixed: options.fixed,
        };
        model.apply(ModelEdit::AddCovariateRelation(scaling.relation(options)))?;
        scalings.push(scaling);
    }
    if scalings.is_empty() {
        return Err(format!(
            "allometry: every parameter already carries a relation on `{}`; nothing to add",
            options.covariate
        ));
    }
    Ok(AllometricModel {
        model,
        scalings,
        notes,
    })
}

/// The clearance-like and volume-like individual parameters, from the
/// template line's roles.
fn classify(ctx: &ModelContext) -> Result<(Vec<String>, Vec<String>), String> {
    let Some(template) = &ctx.template else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut clearances = Vec::new();
    let mut volumes = Vec::new();
    for (role, var) in &template.bindings {
        let Some(param) = ctx.parameters.iter().find(|p| p.eq_ignore_ascii_case(var)) else {
            continue;
        };
        let into = if CLEARANCE_ROLES.contains(&role.as_str()) {
            &mut clearances
        } else if VOLUME_ROLES.contains(&role.as_str()) {
            &mut volumes
        } else {
            continue;
        };
        if !into.contains(param) {
            into.push(param.clone());
        }
    }
    Ok((clearances, volumes))
}

/// What one [`run_allometry`] call takes beyond the model and options.
#[derive(Default)]
pub struct AllometryRun {
    /// Where the runner journals the two fits; `None` keeps them in memory.
    pub dir: Option<PathBuf>,
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    /// Retries, strictness and resume; the criterion is forced to OFV.
    pub run_options: crate::search::RunOptions,
}

/// The outcome: both fits, side by side.
#[derive(Debug, Clone)]
pub struct AllometryResult {
    pub scalings: Vec<Scaling>,
    pub model: ModelText,
    pub base: CandidateResult,
    pub scaled: CandidateResult,
    pub notes: Vec<String>,
    pub cancelled: bool,
}

impl AllometryResult {
    /// `OFV_base − OFV_scaled`, when both fits exist.
    pub fn dofv(&self) -> Option<f64> {
        Some(self.base.ofv? - self.scaled.ofv?)
    }
}

/// Build the allometric model and fit it beside the base model.
pub fn run_allometry(
    base: &BaseModel,
    options: &AllometryOptions,
    run: AllometryRun,
) -> Result<AllometryResult, String> {
    let built = allometric_model(base, options)?;
    let mut features = FeatureVector::new();
    for s in &built.scalings {
        features.set(
            format!("{}-{}", s.parameter, options.covariate),
            if s.fixed {
                format!("power(fix={})", s.exponent)
            } else {
                "power".to_string()
            },
        );
    }
    let candidates = vec![
        Candidate::new("base", base.text.clone()),
        Candidate::new("allometric", built.model.clone())
            .parent("base")
            .features(features),
    ];
    let mut runner = Runner::new();
    if let Some(t) = run.threads {
        runner = runner.threads(t);
    }
    if let Some(dir) = &run.dir {
        runner = runner.cache_dir(dir.clone());
    }
    if let Some(flag) = &run.cancel {
        runner = runner.cancel(flag.clone());
    }
    let mut run_options = run.run_options.clone();
    run_options.criterion = Criterion::Ofv;
    let report = runner.run(&candidates, &base.prepared.population, &run_options)?;
    let mut notes = built.notes;
    notes.extend(report.warnings.iter().cloned());
    let find = |id: &str| -> Result<CandidateResult, String> {
        report
            .results
            .iter()
            .find(|r| r.id == id)
            .cloned()
            .ok_or_else(|| format!("the {id} model was not fitted (run cancelled)"))
    };
    Ok(AllometryResult {
        scalings: built.scalings,
        model: built.model,
        base: find("base")?,
        scaled: find("allometric")?,
        notes,
        cancelled: report.cancelled,
    })
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
