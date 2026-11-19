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
use crate::estimation::uncertainty_samples::UncertaintyMethod;
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
    /// Parameter uncertainty (#946): the fit the draws come from and how many.
    pub(crate) uncertainty: Option<(&'a FitResult, UncertaintySpec)>,
    /// A test seam (#946): the parameter draws themselves, bypassing the sampler, so a
    /// draw can equal the point estimate exactly, which no public input produces.
    pub(crate) injected_draws: Option<&'a [ModelParameters]>,
}

/// How an adaptive run draws parameter uncertainty (#946): see
/// [`AdaptiveInputs::with_uncertainty`].
///
/// `Default` is the asymptotic method with `n_draws = 0`, which the run refuses: set
/// `n_draws` when spreading `Default::default()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UncertaintySpec {
    /// How the parameter sets are drawn: asymptotic MVN on the fit's covariance matrix
    /// (the default), or the fit's retained SIR resamples.
    pub method: UncertaintyMethod,
    /// Number of parameter sets. Each is simulated for every subject and every one of
    /// the run's `n_sim` replicates. Must be at least 1.
    pub n_draws: usize,
}

/// Purpose tag of the parameter-draw stream (#946), so it is disjoint from the η stream
/// and from the κ, assay and mixture-class substreams rooted at the same run seed.
pub(super) const PARAM_PURPOSE_SALT: u64 = 0x9A8A_3D8A_3D5E_0001;

impl<'a> AdaptiveInputs<'a> {
    /// Simulate at `params`, drawing every subject's η from Ω.
    pub fn new(params: &'a ModelParameters) -> Self {
        Self {
            params,
            etas: None,
            unmatched: UnmatchedSubjects::default(),
            uncertainty: None,
            injected_draws: None,
        }
    }

    /// Propagate the fit's parameter uncertainty (#946): draw `spec.n_draws` parameter
    /// sets (θ, Ω, σ, Ω_IOV) from `fit`, with the sampler `simulate_with_uncertainty`
    /// uses, and simulate every subject and replicate under each. Every row is then
    /// tagged with its real `draw`, and the result's `parameters` holds each draw.
    ///
    /// The draws are centred on the fit's point estimate, so `params` must carry the
    /// fit's values (θ, Ω, Ω_IOV, σ, residual ρ); the run refuses a `params` that
    /// differs from them. The draws come from a stream of their own, so draw 1 uses the
    /// standard normals the same seed gives without uncertainty; its η are those normals
    /// under draw 1's Ω, so a draw at the point estimate reproduces that run. The η
    /// stream continues into the next draw: each draw simulates new virtual patients.
    ///
    /// Cannot be combined with [`with_etas`](Self::with_etas): each supplied η was
    /// estimated at the fit's θ, so pairing it with a drawn θ would not describe the
    /// same patient.
    pub fn with_uncertainty(mut self, fit: &'a FitResult, spec: UncertaintySpec) -> Self {
        self.uncertainty = Some((fit, spec));
        self
    }

    /// Hand the run its parameter draws directly (#946), bypassing the sampler: the
    /// seam the degenerate oracle and the cross-check against `simulate_with_uncertainty`
    /// run through.
    #[cfg(test)]
    pub(crate) fn with_injected_draws(mut self, draws: &'a [ModelParameters]) -> Self {
        self.injected_draws = Some(draws);
        self
    }

    /// Whether the run draws parameter uncertainty, through the sampler or the seam.
    pub(crate) fn draws_uncertainty(&self) -> bool {
        self.uncertainty.is_some() || self.injected_draws.is_some()
    }

    /// The parameter draws of an uncertainty run (#946), `None` without uncertainty.
    /// `root` is the run's root seed (the run seed, or entropy for an unseeded run): the
    /// draws come from their own stream rooted there, never from the η stream.
    ///
    /// Refuses `n_draws = 0` and a `params` whose values are not the fit's
    /// (`E_ADAPTIVE_OPTIONS`), an asymptotic draw from a fit without a covariance matrix
    /// (`E_FIT_INPUT_MISSING`, naming the remedy the fit allows), and whatever
    /// `fitted_params_from_result` and the sampler refuse, with their codes.
    pub(crate) fn resolve_draws(
        &self, model: &CompiledModel, root: u64,
    ) -> Result<Option<Vec<ModelParameters>>, EngineError> {
        if let Some(draws) = self.injected_draws {
            debug_assert!(!draws.is_empty(), "the seam is handed at least one draw");
            return Ok(Some(draws.to_vec()));
        }
        let Some((fit, spec)) = self.uncertainty else {
            return Ok(None);
        };
        if spec.n_draws == 0 {
            return Err(super::entry_refusals::adaptive_options(
                "`UncertaintySpec::n_draws` must be at least 1; omit `with_uncertainty` to \
                 simulate at the point estimate",
            ));
        }
        let template =
            crate::estimation::uncertainty_samples::fitted_params_from_result(fit, model)?;
        if let Some(what) = first_value_difference(self.params, &template) {
            return Err(super::entry_refusals::adaptive_options(format!(
                "`with_uncertainty` draws around the fit's point estimate, but the run's \
                 parameters differ from it: {what}. Pass the fit's parameters (for example \
                 `fitted_params_from_result(&fit, &model)`) as the run's `params`"
            )));
        }
        if spec.method == UncertaintyMethod::Asymptotic && fit.covariance_matrix.is_none() {
            let remedy = if fit.sir_resamples_packed.is_some() {
                "This fit carries a SIR resample pool: set `method` to `UncertaintyMethod::Sir` \
                 to draw from it."
            } else {
                "Refit with `covariance_fallback = sir` and `sir_keep_samples = true` to draw \
                 from the SIR pool instead (#223)."
            };
            return Err(super::entry_refusals::fit_input_missing(format!(
                "Asymptotic uncertainty requires FitResult.covariance_matrix, and this fit has \
                 none (its covariance step did not run or did not succeed). {remedy}"
            )));
        }
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(crate::sim::adaptive::combine_seed(
            root ^ PARAM_PURPOSE_SALT,
            0,
        ));
        crate::estimation::uncertainty_samples::draw_parameter_samples_coded(
            fit,
            &template,
            spec.n_draws,
            spec.method,
            &mut rng,
        )
        .map(Some)
    }

    /// Simulate each subject the `etas` carry at its own η. A supplied η stays fixed
    /// across replicates; κ and a `Dv` monitor's assay are still drawn. Adaptive
    /// trajectories carry no residual error, so without IOV or an assay the replicates
    /// of a supplied subject are identical.
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
        &self, model: &CompiledModel, population: &Population,
    ) -> Result<Vec<Option<Vec<f64>>>, EngineError> {
        let Some(etas) = self.etas else {
            return Ok(vec![None; population.subjects.len()]);
        };
        // Under either unmatched policy, ahead of every other check of the η (#946).
        if self.draws_uncertainty() {
            return Err(refusal(
                "parameter uncertainty (`with_uncertainty`) cannot be combined with supplied \
                 random effects (`with_etas`): each supplied η was estimated at the fit's θ, so \
                 pairing it with a drawn θ does not describe the same patient. Simulate the \
                 supplied subjects without uncertainty, or drop the supplied η; #2154 tracks \
                 drawing η from its posterior under each draw"
                    .to_string(),
            ));
        }
        resolve_supplied_etas(model, population, etas, SuppliedEtasEntry::Adaptive(self.unmatched))
    }
}

/// The entry point a [`SuppliedEtas`] is resolved for: it picks the refusal wording and
/// whether an unmatched subject may be drawn.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SuppliedEtasEntry {
    /// An adaptive run, with its policy for a subject the η do not cover.
    Adaptive(UnmatchedSubjects),
    /// `predict_survival_at_etas_diag` (#1885): an individual curve needs every
    /// subject's η, so an unmatched subject is always refused.
    #[cfg(feature = "survival")]
    PredictSurvival,
}

/// Each population subject's supplied η, in population order (`None`: not supplied,
/// which only an adaptive run under [`UnmatchedSubjects::Draw`] returns). Refuses a
/// `[mixture]` model, η names that are not the model's, and (unless the entry may draw)
/// a population subject with no row. A supplied ID the population lacks is ignored.
pub(crate) fn resolve_supplied_etas(
    model: &CompiledModel, population: &Population, etas: &SuppliedEtas, entry: SuppliedEtasEntry,
) -> Result<Vec<Option<Vec<f64>>>, EngineError> {
    // A mixture fit estimates each subject's η under its own class, but the adaptive
    // run draws each subject's class from the mixing probabilities (#2163), so a
    // class-2 η̂ could be simulated under class 1 (#2162 review, finding 1). A survival
    // prediction has no class at all to evaluate it under.
    if model.mixture.is_some() {
        return Err(refusal(match entry {
            SuppliedEtasEntry::Adaptive(_) => {
                "supplied η are not supported on a [mixture] model. Each subject's η was \
                 estimated under its own mixture class, and an adaptive run draws each \
                 subject's class from the mixing probabilities, so the η could be simulated \
                 under another class"
                    .to_string()
            }
            #[cfg(feature = "survival")]
            SuppliedEtasEntry::PredictSurvival => {
                "supplied η are not supported on a [mixture] model: each subject's η was \
                 estimated under its own mixture class, and `predict_survival_at_etas_diag` \
                 has no class to evaluate it under"
                    .to_string()
            }
        }));
    }
    if etas.names != model.eta_names {
        return Err(refusal(format!(
            "the supplied η are named [{}] but the model's are [{}]; supply every η of \
             the model, in the model's order",
            etas.names.join(", "),
            model.eta_names.join(", ")
        )));
    }
    let resolved: Vec<Option<Vec<f64>>> =
        population.subjects.iter().map(|s| etas.get(&s.id).map(<[f64]>::to_vec)).collect();
    if matches!(entry, SuppliedEtasEntry::Adaptive(UnmatchedSubjects::Draw)) {
        return Ok(resolved);
    }
    let missing: Vec<&str> = population
        .subjects
        .iter()
        .zip(&resolved)
        .filter(|(_, r)| r.is_none())
        .map(|(s, _)| s.id.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(refusal(match entry {
            SuppliedEtasEntry::Adaptive(_) => format!(
                "{} simulated subject(s) have no supplied η: {}. Supply their η, or pass \
                 `UnmatchedSubjects::Draw` to draw theirs from Ω",
                missing.len(),
                list_ids(&missing)
            ),
            #[cfg(feature = "survival")]
            SuppliedEtasEntry::PredictSurvival => format!(
                "{} subject(s) in the population have no supplied η: {}. An individual \
                 curve needs each subject's η: supply theirs, drop them from the population, \
                 or call `predict_survival_diag` for the population (η = 0) curve",
                missing.len(),
                list_ids(&missing)
            ),
        }));
    }
    Ok(resolved)
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
        ids: Vec<String>, names: Vec<String>, rows: Vec<Vec<f64>>,
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
        Ok(Self { ids, names, rows, index })
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
            fit.subjects.iter().map(|s| s.eta.iter().copied().collect()).collect(),
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

/// The first parameter whose **value** differs between the run's `params` and the fit's
/// template (#946), or `None` when every value is bit-equal: θ, Ω, Ω_IOV, σ and the
/// `block_sigma` ρ. Masks, FIX flags, names and bounds are not compared: ferx-r builds
/// its `params` through `OmegaMatrix::from_matrix` with the template's flags, and the
/// draws pin FIX coordinates from the fit either way.
pub(super) fn first_value_difference(
    run: &ModelParameters, fit: &ModelParameters,
) -> Option<String> {
    let bits_eq = |a: f64, b: f64| a.to_bits() == b.to_bits();
    let first_in = |a: &[f64], b: &[f64]| -> Option<Option<usize>> {
        if a.len() != b.len() {
            return Some(None);
        }
        a.iter().zip(b).position(|(&x, &y)| !bits_eq(x, y)).map(Some)
    };
    // A vector parameter by the fit's name for the element: `θ TVCL is …`.
    let named = |kind: &str,
                 name: &dyn Fn(usize) -> String,
                 k: Option<usize>,
                 a: &[f64],
                 b: &[f64]| {
        match k {
            None => format!("{kind} has {} value(s), the fit's {}", a.len(), b.len()),
            Some(k) => format!("{kind} {} is {}, the fit's is {}", name(k), a[k], b[k]),
        }
    };
    let by_name = |names: &[String]| {
        let names = names.to_vec();
        move |k: usize| names.get(k).cloned().unwrap_or_else(|| (k + 1).to_string())
    };
    if let Some(k) = first_in(&run.theta, &fit.theta) {
        let (a, b) = (&run.theta, &fit.theta);
        return Some(named("θ", &by_name(&fit.theta_names), k, a, b));
    }
    let matrix = |name: &str, a: &nalgebra::DMatrix<f64>, b: &nalgebra::DMatrix<f64>| {
        if a.shape() != b.shape() {
            return Some(format!("{name} is {:?}, the fit's is {:?}", a.shape(), b.shape()));
        }
        (0..a.nrows())
            .flat_map(|i| (0..a.ncols()).map(move |j| (i, j)))
            .find(|&(i, j)| !bits_eq(a[(i, j)], b[(i, j)]))
            .map(|(i, j)| {
                format!("{name}[{}, {}] is {}, the fit's is {}", i + 1, j + 1, a[(i, j)], b[(i, j)])
            })
    };
    if let Some(d) = matrix("Ω", &run.omega.matrix, &fit.omega.matrix) {
        return Some(d);
    }
    match (&run.omega_iov, &fit.omega_iov) {
        (Some(a), Some(b)) => {
            if let Some(d) = matrix("Ω_IOV", &a.matrix, &b.matrix) {
                return Some(d);
            }
        }
        (None, None) => {}
        (a, _) => {
            return Some(format!(
                "the run {} an Ω_IOV and the fit {}",
                if a.is_some() { "has" } else { "has no" },
                if a.is_some() { "has none" } else { "has one" },
            ))
        }
    }
    if let Some(k) = first_in(&run.sigma.values, &fit.sigma.values) {
        let (a, b) = (&run.sigma.values, &fit.sigma.values);
        return Some(named("σ", &by_name(&fit.sigma.names), k, a, b));
    }
    let rho = |p: &ModelParameters| -> Vec<f64> {
        p.residual_correlations.iter().map(|c| c.rho).collect()
    };
    let (ra, rb) = (rho(run), rho(fit));
    // A ρ by the σ pair it correlates: `residual ρ (PROP, ADD)`.
    let sigma = by_name(&fit.sigma.names);
    let pair = |k: usize| {
        fit.residual_correlations.get(k).map_or_else(
            || (k + 1).to_string(),
            |c| format!("({}, {})", sigma(c.sigma_i), sigma(c.sigma_j)),
        )
    };
    first_in(&ra, &rb).map(|k| named("residual ρ", &pair, k, &ra, &rb))
}

/// At most 10 IDs, then a count of the rest.
pub(crate) fn list_ids(ids: &[&str]) -> String {
    const SHOWN: usize = 10;
    let head = ids.iter().take(SHOWN).map(|id| format!("'{id}'")).collect::<Vec<_>>().join(", ");
    if ids.len() > SHOWN {
        format!("{head} and {} more", ids.len() - SHOWN)
    } else {
        head
    }
}

fn refusal(message: String) -> EngineError {
    super::entry_refusals::supplied_etas(message)
}
