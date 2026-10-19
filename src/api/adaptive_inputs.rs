//! What an adaptive run is given beyond the model, the population and the options
//! (#1594): the parameter point and, optionally, each known subject's own η.
//!
//! [`AdaptiveInputs`] carries them to [`simulate_adaptive_with_inputs`] and
//! [`simulate_adaptive_from_spec_with_inputs`]. It is `#[non_exhaustive]` and built
//! through a constructor and builder methods, so a later input (#946, #2154) is a new
//! method, not a breaking field on every caller's struct literal.
//!
//! [`simulate_adaptive_with_inputs`]: crate::simulate_adaptive_with_inputs
//! [`simulate_adaptive_from_spec_with_inputs`]: crate::simulate_adaptive_from_spec_with_inputs

use crate::diagnostics::EngineError;
use crate::types::{CompiledModel, FitResult, ModelParameters, Population};
use std::collections::HashMap;

/// Each known subject's between-subject random effects (η), keyed by subject ID.
///
/// A supplied subject is simulated at its own η instead of a draw from Ω. Build one
/// from a fit's empirical Bayes estimates with [`SuppliedEtas::from_fit`], or from
/// your own values with [`SuppliedEtas::from_table`]. Hand it to an adaptive run
/// with [`AdaptiveInputs::with_etas`].
///
/// A row gives every η of the model, in the model's order. There is no partial row:
/// under a block Ω the missing η would need a draw conditional on the given ones,
/// and a marginal draw would be silently wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct SuppliedEtas {
    ids: Vec<String>,
    names: Vec<String>,
    rows: Vec<Vec<f64>>,
    index: HashMap<String, usize>,
}

/// What an adaptive run does with a simulated subject whose ID the
/// [`SuppliedEtas`] do not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum UnmatchedSubjects {
    /// Refuse the run (`E_SUPPLIED_ETAS`), listing the unmatched IDs. The default:
    /// a population that should have been the fitted one but is keyed differently
    /// would otherwise quietly simulate every subject as a new patient.
    #[default]
    Refuse,
    /// Draw the subject's η from Ω, as for a run without supplied η: known patients
    /// and virtual ones in one population.
    Draw,
}

/// The inputs of an adaptive run besides the model, the population and the options.
///
/// ```ignore
/// let inputs = AdaptiveInputs::new(&params)
///     .with_etas(&etas)
///     .with_unmatched(UnmatchedSubjects::Draw);
/// ```
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AdaptiveInputs<'a> {
    /// The parameter point: θ, Ω (the η draw), σ (a `Dv` monitor's assay), Ω_IOV.
    pub params: &'a ModelParameters,
    /// Each known subject's η. `None` draws every subject's η from Ω.
    pub etas: Option<&'a SuppliedEtas>,
    /// What to do with a simulated subject the supplied η do not cover.
    pub unmatched: UnmatchedSubjects,
}

impl<'a> AdaptiveInputs<'a> {
    /// Simulate at `params`, drawing every subject's η from Ω.
    pub fn new(params: &'a ModelParameters) -> Self {
        Self {
            params,
            etas: None,
            unmatched: UnmatchedSubjects::default(),
        }
    }

    /// Simulate each subject the `etas` carry at its own η. A supplied η stays fixed
    /// across replicates; κ, the assay and the residual noise are still drawn.
    pub fn with_etas(mut self, etas: &'a SuppliedEtas) -> Self {
        self.etas = Some(etas);
        self
    }

    /// What to do with a simulated subject the supplied η do not cover. Without
    /// supplied η it has no effect.
    pub fn with_unmatched(mut self, policy: UnmatchedSubjects) -> Self {
        self.unmatched = policy;
        self
    }

    /// Each population subject's supplied η, in population order (`None`: draw it).
    pub(crate) fn resolve(
        &self,
        model: &CompiledModel,
        population: &Population,
    ) -> Result<Vec<Option<Vec<f64>>>, EngineError> {
        let Some(etas) = self.etas else {
            return Ok(vec![None; population.subjects.len()]);
        };
        if etas.names != model.eta_names {
            return Err(refusal(format!(
                "the supplied η are named [{}] but the model's are [{}]; supply every η of \
                 the model, in the model's order",
                etas.names.join(", "),
                model.eta_names.join(", ")
            )));
        }
        let resolved: Vec<Option<Vec<f64>>> = population
            .subjects
            .iter()
            .map(|s| etas.get(&s.id).map(<[f64]>::to_vec))
            .collect();
        if self.unmatched == UnmatchedSubjects::Refuse {
            let missing: Vec<&str> = population
                .subjects
                .iter()
                .zip(&resolved)
                .filter(|(_, r)| r.is_none())
                .map(|(s, _)| s.id.as_str())
                .collect();
            if !missing.is_empty() {
                return Err(refusal(format!(
                    "{} simulated subject(s) have no supplied η: {}. Supply their η, or pass \
                     `UnmatchedSubjects::Draw` to draw theirs from Ω",
                    missing.len(),
                    list_ids(&missing)
                )));
            }
        }
        Ok(resolved)
    }
}

impl SuppliedEtas {
    /// One row of η per subject: `rows[i]` is subject `ids[i]`'s η, in the order of
    /// `names`, which must be the model's η names in the model's order (checked when
    /// the run starts).
    ///
    /// # Errors
    ///
    /// `E_SUPPLIED_ETAS` when `ids` and `rows` differ in length, a row's length is
    /// not the number of `names`, an ID repeats, or a value is not finite.
    pub fn from_table(
        ids: Vec<String>,
        names: Vec<String>,
        rows: Vec<Vec<f64>>,
    ) -> Result<Self, EngineError> {
        if ids.len() != rows.len() {
            return Err(refusal(format!(
                "{} subject ID(s) but {} row(s) of η",
                ids.len(),
                rows.len()
            )));
        }
        let mut index = HashMap::with_capacity(ids.len());
        for (i, (id, row)) in ids.iter().zip(&rows).enumerate() {
            if row.len() != names.len() {
                return Err(refusal(format!(
                    "subject '{id}' has {} η but {} are named; a row gives every η",
                    row.len(),
                    names.len()
                )));
            }
            if let Some(k) = row.iter().position(|v| !v.is_finite()) {
                return Err(refusal(format!(
                    "subject '{id}' has a non-finite {} = {}",
                    names[k], row[k]
                )));
            }
            if index.insert(id.clone(), i).is_some() {
                return Err(refusal(format!("subject ID '{id}' appears more than once")));
            }
        }
        Ok(Self {
            ids,
            names,
            rows,
            index,
        })
    }

    /// A fit's empirical Bayes estimates (`fit.subjects[i].eta`), one row per fitted
    /// subject, named by `fit.eta_names`.
    ///
    /// The η are the fit's point estimates: a supplied η̂ is treated as exact, which
    /// is optimistic for a subject with few observations (high shrinkage).
    ///
    /// # Errors
    ///
    /// `E_SUPPLIED_ETAS` when the fit's η names are not `model`'s, or as
    /// [`from_table`](Self::from_table).
    pub fn from_fit(fit: &FitResult, model: &CompiledModel) -> Result<Self, EngineError> {
        if fit.eta_names != model.eta_names {
            return Err(refusal(format!(
                "the fit's η are named [{}] but the model's are [{}]; the fit is of another \
                 model",
                fit.eta_names.join(", "),
                model.eta_names.join(", ")
            )));
        }
        Self::from_table(
            fit.subjects.iter().map(|s| s.id.clone()).collect(),
            fit.eta_names.clone(),
            fit.subjects
                .iter()
                .map(|s| s.eta.iter().copied().collect())
                .collect(),
        )
    }

    /// The subject IDs, in the order given.
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// The η names, in the order of each row.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Subject `id`'s η, or `None` when it is not supplied.
    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.index.get(id).map(|&i| self.rows[i].as_slice())
    }
}

/// At most 10 IDs, then a count of the rest.
fn list_ids(ids: &[&str]) -> String {
    const SHOWN: usize = 10;
    let head = ids
        .iter()
        .take(SHOWN)
        .map(|id| format!("'{id}'"))
        .collect::<Vec<_>>()
        .join(", ");
    if ids.len() > SHOWN {
        format!("{head} and {} more", ids.len() - SHOWN)
    } else {
        head
    }
}

fn refusal(message: String) -> EngineError {
    super::entry_refusals::supplied_etas(message)
}
