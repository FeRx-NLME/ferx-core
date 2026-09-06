//! Stepwise covariate modelling — PsN `scm` forward / forward-then-backward,
//! Pharmpy `covsearch` (#1180).
//!
//! The first search tool of the #1175 epic, and the one `[covariate_model]`
//! (#1111) was built for: one relation per line, no cross-line dependencies,
//! so adding or dropping a candidate effect is a line insert or delete on one
//! block and the rest of the file is invariant. Everything else the search
//! needs already exists — [`ModelText`] to make the edit, the
//! [`Runner`](crate::search::Runner) to fit a step's candidates in parallel
//! with dedup, journal and strictness, and the `.ferxsearch` file
//! ([`SearchConfig`]) to say what to search over.
//!
//! # The algorithm
//!
//! **Forward.** From the base model, every remaining candidate effect is
//! added on its own and fitted. Each child is compared with its parent by a
//! likelihood-ratio test ([`Lrt`]) at `p_forward`; among the children that
//! pass the strictness gate *and* are significant, the one with the lowest
//! OFV — the largest drop — becomes the parent of the next step, and every
//! other form on that (parameter, covariate) pair leaves the candidate list.
//! The phase ends when no child is significant, the list is empty, or
//! `max_steps` is reached. This is the selection rule of both PsN's
//! `gof_pval` and Pharmpy's `modelrank(rank_type='lrt')`.
//!
//! **Backward** (`scm-forward-then-backward`, the default). From the forward
//! phase's final model, every effect the forward phase added is removed on
//! its own and fitted. The removal with the *smallest* OFV increase is taken
//! if that increase is not significant at `p_backward`; the phase ends when
//! every remaining effect is significant. Effects forced by a `COVARIATE(...)`
//! statement, and relations the base model already declared, are never
//! candidates for removal.
//!
//! **Adaptive scope reduction** (SCM+, off by default as in Pharmpy). During
//! the forward phase, every effect that was fitted and found insignificant is
//! stashed and not re-fitted at later steps, which is where the saving is.
//! Once the ordinary forward phase is exhausted, the stash — minus anything on
//! a pair that has since been added — is searched again with the same rule,
//! so an effect that only shows once another is in the model still gets its
//! chance.
//!
//! **Between steps** each child starts from its parent's final estimates
//! ([`ModelEdit::SeedInits`], Pharmpy's `update_initial_estimates`); the new
//! relation's θ takes the block's PsN default for its form.
//!
//! # What the report says
//!
//! Every candidate of every step is a [`StepRow`]: the ΔOFV, the degrees of
//! freedom, the p-value, the decision — and beside them the fit's
//! convergence and the strictness verdict, never omitted. Under automation a
//! candidate that stalled at its initial estimates (#751) and "lost" a
//! forward step is a *selection error*, not a result, and the row has to say
//! that it was excluded and why rather than showing an OFV that says nothing
//! about the model.
//!
//! # Deviations, stated
//!
//! * The base model's own `[covariate_model]` relations are kept as they are.
//!   Pharmpy removes any that also appear in the search space so it can
//!   re-explore them; ferx treats a relation the author wrote as structural,
//!   drops the matching exploratory effects from the candidate list with a
//!   note, and never removes it in the backward phase.
//! * A child that adds no free parameter — a relation whose θ is `FIX`ed, or
//!   one that compiled to the parent's parameter vector — is reported as
//!   unjudgeable and cannot win. Pharmpy runs it through `χ²(0)` and would
//!   accept it on any non-negative ΔOFV.
//! * The adaptive stash is handled by exact effect. Pharmpy removes every
//!   form on a stashed *pair* from the running candidate list — including a
//!   form that was significant but not that step's winner — and, before the
//!   adaptive pass, drops a stashed effect whenever its parameter *and* its
//!   covariate each appear in some step, which also drops `CL-CRCL` after
//!   `CL-WT` and `V-CRCL` were added. ferx removes only the stashed effects
//!   themselves, and before the adaptive pass drops only those on a pair the
//!   model now carries.

use std::path::{Path, PathBuf};

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::{CancelFlag, FitResult};
use serde::Deserialize;

use crate::search::fitter::{RunnerFitter, StepFitter};
use crate::search::seed::seed_from;
use crate::search::{
    BaseModel, Candidate, CandidateResult, Criterion, FeatureVector, RankType, RunReport,
    SearchConfig,
};

mod effects;
mod lrt;
mod report;

pub use effects::Effect;
pub use lrt::{chi_square_isf, chi_square_sf, Lrt};
pub use report::{final_model_path, render_summary, steps_path, write_report, STEP_COLUMNS};

/// `[covsearch] algorithm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
pub enum Algorithm {
    /// Forward selection only.
    #[serde(rename = "scm-forward")]
    ScmForward,
    /// Forward selection, then backward elimination from the forward result
    /// — Pharmpy's default and PsN's `scm` with both `p_forward` and
    /// `p_backward`.
    #[default]
    #[serde(rename = "scm-forward-then-backward")]
    ScmForwardThenBackward,
}

impl Algorithm {
    pub fn label(&self) -> &'static str {
        match self {
            Algorithm::ScmForward => "scm-forward",
            Algorithm::ScmForwardThenBackward => "scm-forward-then-backward",
        }
    }
}

/// The `[covsearch]` section of a `.ferxsearch` file. Every key has
/// Pharmpy's default.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CovsearchOptions {
    #[serde(default)]
    pub algorithm: Algorithm,
    /// Significance level a forward step's winner must reach.
    #[serde(default = "default_p_forward")]
    pub p_forward: f64,
    /// Level at which a backward step's removal is *refused*: an effect whose
    /// removal is significant at this level stays in.
    #[serde(default = "default_p_backward")]
    pub p_backward: f64,
    /// Most steps per phase; `None` (the default) is unlimited, Pharmpy's
    /// `-1`. The backward phase is additionally bounded by the number of
    /// removable effects.
    #[serde(default)]
    pub max_steps: Option<usize>,
    /// SCM+: stash the effects a forward step finds insignificant and
    /// re-test them once the significant ones are exhausted.
    #[serde(default)]
    pub adaptive_scope_reduction: bool,
}

fn default_p_forward() -> f64 {
    0.01
}

fn default_p_backward() -> f64 {
    0.001
}

impl Default for CovsearchOptions {
    fn default() -> Self {
        CovsearchOptions {
            algorithm: Algorithm::default(),
            p_forward: default_p_forward(),
            p_backward: default_p_backward(),
            max_steps: None,
            adaptive_scope_reduction: false,
        }
    }
}

impl CovsearchOptions {
    /// Read the `[covsearch]` section of a loaded file — the defaults when it
    /// is absent — and check that the rest of the file is consistent with a
    /// likelihood-ratio search.
    ///
    /// covsearch ranks by the LRT on OFV and nothing else, so a `[rank]`
    /// section asking for a BIC or a cutoff is an error rather than something
    /// to ignore: a file that says `type = "bic"` and gets an LRT would be
    /// lying about how its winner was chosen.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        config.require_space("covsearch", "COVARIATE / COVARIATE? statements")?;
        let options = match config.tools.get("covsearch") {
            Some(table) => table
                .clone()
                .try_into::<CovsearchOptions>()
                .map_err(|e| format!("[covsearch]: {e}"))?,
            None => CovsearchOptions::default(),
        };
        options.validate()?;
        if let Some(kind) = config.rank.kind {
            if kind != RankType::Ofv {
                return Err(format!(
                    "[rank] type = \"{}\": covsearch selects by the likelihood-ratio test on \
                     the OFV at [covsearch] p_forward / p_backward; a BIC ranking does not \
                     apply. Remove the key, or set it to \"ofv\"",
                    rank_label(kind)
                ));
            }
        }
        if let Some(cutoff) = config.rank.cutoff {
            return Err(format!(
                "[rank] cutoff = {cutoff}: covsearch takes its thresholds as p-values — \
                 [covsearch] p_forward and p_backward — not as a ΔOFV. Remove the key"
            ));
        }
        Ok(options)
    }

    /// The bounds Pharmpy enforces: both levels in `(0, 1]`.
    pub fn validate(&self) -> Result<(), String> {
        for (key, p) in [
            ("p_forward", self.p_forward),
            ("p_backward", self.p_backward),
        ] {
            if !(p > 0.0 && p <= 1.0) {
                return Err(format!(
                    "[covsearch] {key} = {p}: must be a probability in (0, 1]"
                ));
            }
        }
        if self.max_steps == Some(0) {
            return Err(
                "[covsearch] max_steps = 0: a search with no steps is the base model; omit the \
                 key for an unlimited search"
                    .into(),
            );
        }
        Ok(())
    }
}

fn rank_label(kind: RankType) -> &'static str {
    match kind {
        RankType::Ofv => "ofv",
        RankType::Aic => "aic",
        RankType::Bic => "bic",
        RankType::BicMixed => "bic_mixed",
        RankType::BicIiv => "bic_iiv",
        RankType::BicRandom => "bic_random",
        RankType::BicFixed => "bic_fixed",
        RankType::Penalized => "penalized",
    }
}

/// Which pass of the search a step belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Forward,
    /// The second forward pass over the adaptive stash.
    Adaptive,
    Backward,
}

impl Phase {
    pub fn label(&self) -> &'static str {
        match self {
            Phase::Forward => "forward",
            Phase::Adaptive => "adaptive",
            Phase::Backward => "backward",
        }
    }

    fn is_forward(&self) -> bool {
        !matches!(self, Phase::Backward)
    }
}

/// One candidate of one step, as the step table reports it.
#[derive(Debug, Clone)]
pub struct StepRow {
    /// 1-based step number, counting across phases.
    pub step: usize,
    pub phase: Phase,
    /// The candidate's id in that step's runner directory.
    pub candidate: String,
    /// The effect this candidate adds (forward) or removes (backward).
    pub effect: Effect,
    /// The parent's OFV — the model the candidate is compared with.
    pub parent_ofv: f64,
    /// The candidate's OFV, `None` when it produced no fit.
    pub ofv: Option<f64>,
    /// The likelihood-ratio comparison, when it could be made. In the
    /// backward phase the *candidate* is the reduced model and the parent
    /// the extended one, so `dofv` is the OFV *increase* on removal and
    /// `significant` means the effect has to stay.
    pub lrt: Option<Lrt>,
    /// Why no comparison could be made, when `lrt` is `None` and the fit
    /// exists — the candidate added no free parameter, or the parent fit was
    /// unavailable.
    pub note: Option<String>,
    /// Whether this candidate became the next parent.
    pub selected: bool,
    /// [`CandidateResult::converged`], carried so the table never shows a
    /// ΔOFV without its termination status.
    pub converged: Option<bool>,
    /// The strictness verdict.
    pub passed: bool,
    pub failures: Vec<String>,
    pub seconds: f64,
}

impl StepRow {
    /// The p-value, `NaN` when there is none.
    pub fn p_value(&self) -> f64 {
        self.lrt.map(|t| t.p_value).unwrap_or(f64::NAN)
    }
}

/// One relation of the final model.
#[derive(Debug, Clone, PartialEq)]
pub struct Included {
    pub effect: Effect,
    /// How it got there.
    pub origin: Origin,
}

/// Where a relation in the final model came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Declared in the base model's own `[covariate_model]`.
    Base,
    /// A `COVARIATE(...)` statement without `?`.
    Forced,
    /// Added by the forward phase at this step.
    Forward(usize),
}

impl Origin {
    pub fn label(&self) -> String {
        match self {
            Origin::Base => "base model".into(),
            Origin::Forced => "forced".into(),
            Origin::Forward(step) => format!("forward step {step}"),
        }
    }
}

/// What a running search reports about its progress.
#[derive(Debug, Clone, PartialEq)]
pub enum CovsearchEvent {
    /// The base model is about to be fitted.
    BaseStarted,
    /// The base model's fit.
    BaseFinished { ofv: f64, n_parameters: usize },
    /// A step is about to fit `candidates` models.
    StepStarted {
        step: usize,
        phase: Phase,
        candidates: usize,
    },
    /// A step finished. `selected` names the effect that won, with the new
    /// OFV; `None` when nothing was accepted and the phase ends.
    StepFinished {
        step: usize,
        phase: Phase,
        selected: Option<(Effect, f64)>,
    },
}

/// A progress callback: plain data, no terminal concern of any kind.
pub type ProgressFn<'a> = &'a (dyn Fn(CovsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct CovsearchResult {
    pub options: CovsearchOptions,
    /// The base model as fitted — with any forced effects added.
    pub base_model: ModelText,
    pub base_ofv: f64,
    /// Every candidate of every step, in step order.
    pub steps: Vec<StepRow>,
    /// The model the search ended on.
    pub final_model: ModelText,
    pub final_ofv: f64,
    /// The final model's fit; `None` only when the outcome was read back
    /// from a journal whose cached fit is gone.
    pub final_fit: Option<FitResult>,
    /// Which step produced the final model; `0` is the base model.
    pub final_step: usize,
    /// The final model's covariate relations and where each came from.
    pub included: Vec<Included>,
    /// Things a user should see once: effects dropped from the space
    /// because the base model already has them, symbols that resolved to
    /// nothing, a step whose journal could not be written.
    pub notes: Vec<String>,
    /// The search stopped on a flipped [`CancelFlag`]; `steps` and
    /// `final_model` are what it reached.
    pub cancelled: bool,
}

impl CovsearchResult {
    /// The rows of one step.
    pub fn step_rows(&self, step: usize) -> impl Iterator<Item = &StepRow> {
        self.steps.iter().filter(move |r| r.step == step)
    }

    /// The number of steps taken, in every phase.
    pub fn n_steps(&self) -> usize {
        self.steps.iter().map(|r| r.step).max().unwrap_or(0)
    }
}

/// Everything a [`run_covsearch`] call takes beyond the file.
#[derive(Default)]
pub struct CovsearchRun<'a> {
    /// Where the per-step journals, `steps.csv` and `final.ferx` go. `None`
    /// keeps everything in memory: no resume, no files.
    pub dir: Option<PathBuf>,
    /// Overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Run a covariate search from a loaded `.ferxsearch` file and its base
/// model.
///
/// The file's `[space]` supplies the candidate effects (`COVARIATE?`) and
/// the forced ones (`COVARIATE`), `[covsearch]` the algorithm and levels,
/// `[strictness]` the gate and `[run]` the retries and resume flag. Writes
/// `steps.csv` and `final.ferx` into `run.dir` when given.
pub fn run_covsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: CovsearchRun<'_>,
) -> Result<CovsearchResult, String> {
    let options = CovsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = Criterion::Ofv;
    let fitter = RunnerFitter {
        threads: run.threads.or(config.run.threads).unwrap_or(0),
        dir: run.dir.clone(),
        cancel: run.cancel.clone(),
        data: &base.prepared.population,
        options: run_options,
    };
    let space = Space::from_config(config, base)?;
    let result = search(&fitter, space, &options, run.progress)?;
    if let Some(dir) = &run.dir {
        write_report(dir, &result)?;
    }
    Ok(result)
}

/// The search space after resolution against the base model: the model to
/// start from and the effects to explore.
#[derive(Debug)]
pub(crate) struct Space {
    pub base_model: ModelText,
    pub candidates: Vec<Effect>,
    /// Relations the base model carries before the search: its own, and the
    /// forced ones added here.
    pub included: Vec<Included>,
    pub notes: Vec<String>,
}

impl Space {
    fn from_config(config: &SearchConfig, base: &BaseModel) -> Result<Space, String> {
        let resolved = config.resolve_space(base)?;
        // Pharmpy's `COVSEARCH_STATEMENT_TYPES`: `LET` and `COVARIATE` only.
        // A structural feature in a covariate search is a file meant for
        // another tool; running the covariate part of it silently would
        // report a covariate model for a structural question.
        for feature in resolved.mfl.features() {
            if !matches!(feature, crate::search::Feature::Covariate { .. }) {
                return Err(format!(
                    "[space] mfl: `{}` is not a covariate statement; covsearch takes \
                     COVARIATE / COVARIATE? (and LET) only. Structural and variability \
                     features belong to modelsearch, iivsearch and iovsearch (#1175)",
                    feature.keyword()
                ));
            }
        }
        let mut notes = resolved.notes.clone();
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
        let mut included: Vec<Included> = base
            .prepared
            .parsed
            .model
            .covariate_model
            .as_ref()
            .map(|spec| {
                spec.relations
                    .iter()
                    .map(|r| Included {
                        effect: Effect {
                            parameter: r.parameter.clone(),
                            covariate: r.covariate.clone(),
                            form: r.form.clone(),
                        },
                        origin: Origin::Base,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut base_model = base.text.clone();
        let mut candidates = Vec::new();
        for spec in &resolved.covariate_effects {
            let effect = Effect::from_spec(spec)?;
            let in_base = existing
                .iter()
                .any(|(p, c)| *p == effect.parameter && *c == effect.covariate);
            if spec.optional {
                if in_base {
                    notes.push(format!(
                        "not explored: {} — the base model already declares `{} ~ {}`, which \
                         the search keeps as written",
                        effect.label(),
                        effect.parameter,
                        effect.covariate
                    ));
                    continue;
                }
                candidates.push(effect);
            } else if in_base {
                notes.push(format!(
                    "forced effect {} is already in the base model as `{} ~ {}`; kept as written",
                    effect.label(),
                    effect.parameter,
                    effect.covariate
                ));
            } else {
                base_model
                    .apply(ModelEdit::AddCovariateRelation(effect.relation()))
                    .map_err(|e| format!("forcing {} into the base model: {e}", effect.label()))?;
                included.push(Included {
                    effect,
                    origin: Origin::Forced,
                });
            }
        }
        Ok(Space {
            base_model,
            candidates,
            included,
            notes,
        })
    }
}

/// The model the search currently stands on.
struct Parent {
    id: String,
    model: ModelText,
    fit: Option<FitResult>,
    ofv: f64,
    n_parameters: usize,
    step: usize,
}

impl Parent {
    fn from_result(result: &CandidateResult, model: ModelText, step: usize) -> Parent {
        let fit = result.fit.clone();
        Parent {
            id: result.id.clone(),
            n_parameters: fit.as_ref().map(|f| f.n_parameters).unwrap_or(0),
            ofv: result.ofv.unwrap_or(f64::NAN),
            model,
            fit,
            step,
        }
    }
}

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &CovsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<CovsearchResult, String> {
    let emit = |event: CovsearchEvent| {
        if let Some(p) = progress {
            p(event);
        }
    };
    let Space {
        base_model,
        candidates,
        mut included,
        mut notes,
    } = space;

    // ── the base model ──────────────────────────────────────────────────
    emit(CovsearchEvent::BaseStarted);
    let base_candidate =
        Candidate::new("base", base_model.clone()).features(features_of(&included, None, None));
    let report = fitter.fit_step("base", std::slice::from_ref(&base_candidate))?;
    notes.extend(report.warnings.iter().cloned());
    let mut parent = {
        let result = report
            .results
            .first()
            .ok_or("the base model was not fitted")?;
        if let Some(e) = &result.error {
            return Err(format!("the base model could not be fitted: {e}"));
        }
        if !result.verdict.passed {
            return Err(format!(
                "the base model fails the strictness gate ({}); a search from a fit that \
                 cannot be trusted has nothing to compare against. Fix the base fit or \
                 relax [strictness]",
                result.verdict.failures.join("; ")
            ));
        }
        if result.fit.is_none() {
            return Err(
                "the base model's fit is missing from the resumed journal, so nothing can be \
                 seeded from it; refit without `resume`"
                    .into(),
            );
        }
        Parent::from_result(result, base_model.clone(), 0)
    };
    let base_ofv = parent.ofv;
    emit(CovsearchEvent::BaseFinished {
        ofv: base_ofv,
        n_parameters: parent.n_parameters,
    });
    if report.cancelled {
        return Ok(finish(
            options,
            base_model,
            base_ofv,
            Vec::new(),
            parent,
            included,
            notes,
            true,
        ));
    }

    let mut rows: Vec<StepRow> = Vec::new();
    let mut step = 0usize;
    let mut cancelled = false;

    // ── forward ─────────────────────────────────────────────────────────
    let mut remaining = candidates;
    let mut stash: Vec<Effect> = Vec::new();
    let forward = Pass {
        fitter,
        options,
        emit: &emit,
    };
    let outcome = forward.run(
        Phase::Forward,
        &mut parent,
        &mut remaining,
        &mut included,
        options.adaptive_scope_reduction.then_some(&mut stash),
        &mut rows,
        &mut step,
        &mut notes,
    )?;
    cancelled |= outcome.cancelled;

    // ── adaptive ────────────────────────────────────────────────────────
    if !cancelled && options.adaptive_scope_reduction && !stash.is_empty() {
        let mut adaptive: Vec<Effect> = stash
            .into_iter()
            .filter(|e| {
                !included
                    .iter()
                    .any(|i| i.effect.parameter == e.parameter && i.effect.covariate == e.covariate)
            })
            .collect();
        if !adaptive.is_empty() {
            let outcome = forward.run(
                Phase::Adaptive,
                &mut parent,
                &mut adaptive,
                &mut included,
                None,
                &mut rows,
                &mut step,
                &mut notes,
            )?;
            cancelled |= outcome.cancelled;
        }
    }

    // ── backward ────────────────────────────────────────────────────────
    if !cancelled && options.algorithm == Algorithm::ScmForwardThenBackward {
        let outcome =
            forward.backward(&mut parent, &mut included, &mut rows, &mut step, &mut notes)?;
        cancelled |= outcome.cancelled;
    }

    Ok(finish(
        options, base_model, base_ofv, rows, parent, included, notes, cancelled,
    ))
}

#[allow(clippy::too_many_arguments)]
fn finish(
    options: &CovsearchOptions,
    base_model: ModelText,
    base_ofv: f64,
    steps: Vec<StepRow>,
    parent: Parent,
    included: Vec<Included>,
    notes: Vec<String>,
    cancelled: bool,
) -> CovsearchResult {
    CovsearchResult {
        options: options.clone(),
        base_model,
        base_ofv,
        steps,
        final_model: parent.model,
        final_ofv: parent.ofv,
        final_fit: parent.fit,
        final_step: parent.step,
        included,
        notes,
        cancelled,
    }
}

/// What one phase reports back to the driver.
struct Outcome {
    cancelled: bool,
}

/// The state shared by every phase.
struct Pass<'a> {
    fitter: &'a dyn StepFitter,
    options: &'a CovsearchOptions,
    emit: &'a dyn Fn(CovsearchEvent),
}

impl Pass<'_> {
    fn steps_left(&self, taken: usize) -> bool {
        match self.options.max_steps {
            Some(max) => taken < max,
            None => true,
        }
    }

    /// A forward pass — the ordinary one or the adaptive one — over
    /// `remaining`, mutating the parent as effects are accepted.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        phase: Phase,
        parent: &mut Parent,
        remaining: &mut Vec<Effect>,
        included: &mut Vec<Included>,
        mut stash: Option<&mut Vec<Effect>>,
        rows: &mut Vec<StepRow>,
        step: &mut usize,
        notes: &mut Vec<String>,
    ) -> Result<Outcome, String> {
        let mut taken = 0usize;
        while !remaining.is_empty() && self.steps_left(taken) {
            *step += 1;
            let dir = format!("{}-{}", phase.label(), *step);
            (self.emit)(CovsearchEvent::StepStarted {
                step: *step,
                phase,
                candidates: remaining.len(),
            });

            let mut candidates = Vec::with_capacity(remaining.len());
            for effect in remaining.iter() {
                let mut model = parent.model.clone();
                if let Some(fit) = &parent.fit {
                    seed_from(&mut model, fit)?;
                }
                model
                    .apply(ModelEdit::AddCovariateRelation(effect.relation()))
                    .map_err(|e| format!("step {}: adding {}: {e}", *step, effect.label()))?;
                candidates.push(
                    Candidate::new(candidate_id(phase, *step, effect), model)
                        .parent(parent.id.clone())
                        .features(features_of(included, Some(effect), None)),
                );
            }

            let report = self.fitter.fit_step(&dir, &candidates)?;
            notes.extend(report.warnings.iter().cloned());
            let step_rows = judge(
                phase,
                *step,
                parent,
                remaining,
                &candidates,
                &report,
                self.options.p_forward,
            );
            // A cancelled step is reported, never judged: the runner may not
            // have reached every candidate, so its rows are partial and the
            // best of what finished is not the step's winner. Nothing is
            // selected, and the event says so.
            if report.cancelled {
                rows.extend(step_rows);
                (self.emit)(CovsearchEvent::StepFinished {
                    step: *step,
                    phase,
                    selected: None,
                });
                return Ok(Outcome { cancelled: true });
            }
            // A complete step has one row per candidate, in candidate order,
            // so a row index is a `remaining` index.
            debug_assert_eq!(step_rows.len(), remaining.len());

            // Lowest OFV among the candidates that passed the gate and the
            // test — PsN's largest significant drop.
            let winner = step_rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.passed && r.lrt.is_some_and(|t| t.significant))
                .min_by(|(_, a), (_, b)| a.ofv.unwrap().total_cmp(&b.ofv.unwrap()))
                .map(|(i, _)| i);
            let mut step_rows = step_rows;
            if let Some(i) = winner {
                step_rows[i].selected = true;
            }
            let selected = winner.map(|i| (remaining[i].clone(), step_rows[i].ofv.unwrap()));
            (self.emit)(CovsearchEvent::StepFinished {
                step: *step,
                phase,
                selected: selected.clone(),
            });

            // SCM+: stash what was fitted and found wanting, so it is not
            // fitted again at every later step. Only while the phase is still
            // going somewhere — Pharmpy stashes after an accepted step, and
            // a step that accepted nothing ends the phase anyway.
            if let (Some(stash), Some(_)) = (stash.as_mut(), winner) {
                for (row, effect) in step_rows.iter().zip(remaining.iter()) {
                    if !row.selected && row.passed && row.lrt.is_some_and(|t| !t.significant) {
                        stash.push(effect.clone());
                    }
                }
            }

            rows.extend(step_rows);
            let Some(i) = winner else {
                return Ok(Outcome { cancelled: false });
            };
            let effect = remaining.remove(i);
            let result = report
                .results
                .iter()
                .find(|r| r.id == candidates[i].id)
                .expect("the winner is a row of the report");
            *parent = Parent::from_result(result, candidates[i].model.clone(), *step);
            if parent.fit.is_none() {
                notes.push(format!(
                    "step {}: the winning fit for {} is not in the journal cache, so the next \
                     step starts from the file's initial estimates instead of the parent's",
                    *step,
                    effect.label()
                ));
            }
            included.push(Included {
                effect: effect.clone(),
                origin: Origin::Forward(*step),
            });
            // One line per (parameter, covariate) pair: the other forms on the
            // pair that just went in are no longer candidates. Nor is anything
            // stashed — that is the whole saving of SCM+. Exact effects only:
            // Pharmpy's `filter_effects` drops every form on a stashed *pair*,
            // which also throws away a form that was significant but not the
            // winner; a significant effect stays a candidate here.
            remaining.retain(|e| e.pair() != effect.pair());
            if let Some(stash) = stash.as_mut() {
                remaining.retain(|e| !stash.contains(e));
            }
            taken += 1;
        }
        Ok(Outcome { cancelled: false })
    }

    /// Backward elimination over the effects the forward phases added.
    fn backward(
        &self,
        parent: &mut Parent,
        included: &mut Vec<Included>,
        rows: &mut Vec<StepRow>,
        step: &mut usize,
        notes: &mut Vec<String>,
    ) -> Result<Outcome, String> {
        let mut taken = 0usize;
        loop {
            let removable: Vec<Effect> = included
                .iter()
                .filter(|i| matches!(i.origin, Origin::Forward(_)))
                .map(|i| i.effect.clone())
                .collect();
            if removable.is_empty() || !self.steps_left(taken) {
                return Ok(Outcome { cancelled: false });
            }
            *step += 1;
            let phase = Phase::Backward;
            let dir = format!("{}-{}", phase.label(), *step);
            (self.emit)(CovsearchEvent::StepStarted {
                step: *step,
                phase,
                candidates: removable.len(),
            });

            let mut candidates = Vec::with_capacity(removable.len());
            for effect in &removable {
                let mut model = parent.model.clone();
                if let Some(fit) = &parent.fit {
                    seed_from(&mut model, fit)?;
                }
                model
                    .apply(ModelEdit::DropCovariateRelation {
                        param: effect.parameter.clone(),
                        cov: effect.covariate.clone(),
                    })
                    .map_err(|e| format!("step {}: removing {}: {e}", *step, effect.label()))?;
                candidates.push(
                    Candidate::new(candidate_id(phase, *step, effect), model)
                        .parent(parent.id.clone())
                        .features(features_of(included, None, Some(effect))),
                );
            }

            let report = self.fitter.fit_step(&dir, &candidates)?;
            notes.extend(report.warnings.iter().cloned());
            let mut step_rows = judge(
                phase,
                *step,
                parent,
                &removable,
                &candidates,
                &report,
                self.options.p_backward,
            );
            // Same as the forward pass: a cancelled step's rows are partial,
            // so nothing is selected from them.
            if report.cancelled {
                rows.extend(step_rows);
                (self.emit)(CovsearchEvent::StepFinished {
                    step: *step,
                    phase,
                    selected: None,
                });
                return Ok(Outcome { cancelled: true });
            }
            debug_assert_eq!(step_rows.len(), removable.len());

            // The smallest increase among the removals that are *not*
            // significant — and that passed the gate.
            let winner = step_rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.passed && r.lrt.is_some_and(|t| !t.significant))
                .min_by(|(_, a), (_, b)| a.ofv.unwrap().total_cmp(&b.ofv.unwrap()))
                .map(|(i, _)| i);
            if let Some(i) = winner {
                step_rows[i].selected = true;
            }
            let selected = winner.map(|i| (removable[i].clone(), step_rows[i].ofv.unwrap()));
            (self.emit)(CovsearchEvent::StepFinished {
                step: *step,
                phase,
                selected,
            });
            rows.extend(step_rows);
            let Some(i) = winner else {
                return Ok(Outcome { cancelled: false });
            };
            let effect = &removable[i];
            let result = report
                .results
                .iter()
                .find(|r| r.id == candidates[i].id)
                .expect("the winner is a row of the report");
            *parent = Parent::from_result(result, candidates[i].model.clone(), *step);
            if parent.fit.is_none() {
                notes.push(format!(
                    "step {}: the winning fit for removing {} is not in the journal cache, so \
                     the next step starts from the file's initial estimates",
                    *step,
                    effect.label()
                ));
            }
            included.retain(|inc| inc.effect != *effect);
            taken += 1;
        }
    }
}

/// Compare every candidate of a step with its parent.
///
/// `alpha` is the level of the phase. In the forward phases the candidate is
/// the extended model; in the backward phase the roles swap and the row's
/// `lrt` describes the *removal*: `dofv` is the increase, `significant` means
/// the effect has to stay.
fn judge(
    phase: Phase,
    step: usize,
    parent: &Parent,
    effects: &[Effect],
    candidates: &[Candidate],
    report: &RunReport,
    alpha: f64,
) -> Vec<StepRow> {
    let mut rows = Vec::with_capacity(candidates.len());
    for (effect, candidate) in effects.iter().zip(candidates) {
        let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
            // A cancelled run never reached this candidate.
            continue;
        };
        let (lrt, note) = judge_one(phase, parent, result, alpha);
        rows.push(StepRow {
            step,
            phase,
            candidate: candidate.id.clone(),
            effect: effect.clone(),
            parent_ofv: parent.ofv,
            ofv: result.ofv,
            lrt,
            note,
            selected: false,
            converged: result.converged,
            passed: result.verdict.passed && result.error.is_none(),
            failures: result.verdict.failures.clone(),
            seconds: result.seconds,
        });
    }
    rows
}

fn judge_one(
    phase: Phase,
    parent: &Parent,
    result: &CandidateResult,
    alpha: f64,
) -> (Option<Lrt>, Option<String>) {
    if result.error.is_some() {
        return (None, None);
    }
    let Some(ofv) = result.ofv else {
        return (None, Some("no OFV".into()));
    };
    // `n_parameters` comes off the fit, which a degraded resume can lose.
    let Some(n) = result.fit.as_ref().map(|f| f.n_parameters) else {
        return (
            None,
            Some("the fit is not in the journal cache, so its parameter count is unknown".into()),
        );
    };
    if parent.fit.is_none() {
        return (
            None,
            Some(
                "the parent's fit is not in the journal cache, so its parameter count is unknown"
                    .into(),
            ),
        );
    }
    let test = if phase.is_forward() {
        Lrt::forward(parent.ofv, parent.n_parameters, ofv, n, alpha)
    } else {
        // The candidate is the reduced model. `dofv` from `Lrt::forward` is
        // `reduced − extended` = the increase on removal, and `significant`
        // says the removed effect was carrying its weight.
        Lrt::forward(ofv, n, parent.ofv, parent.n_parameters, alpha)
    };
    match test {
        Ok(t) => (Some(t), None),
        Err(e) => (None, Some(e)),
    }
}

/// `f1-CL-WT-power`, `b3-CL-WT-power`: unique within a step, readable in a
/// directory listing.
fn candidate_id(phase: Phase, step: usize, effect: &Effect) -> String {
    let p = match phase {
        Phase::Forward => "f",
        Phase::Adaptive => "a",
        Phase::Backward => "b",
    };
    format!("{p}{step}-{}", effect.label())
}

/// The candidate's relation set as a feature vector: every included pair, plus
/// `adding`, minus `removing`.
fn features_of(
    included: &[Included],
    adding: Option<&Effect>,
    removing: Option<&Effect>,
) -> FeatureVector {
    let mut v = FeatureVector::new();
    for inc in included {
        if removing.is_some_and(|r| *r == inc.effect) {
            continue;
        }
        v.set(inc.effect.pair_key(), inc.effect.form_label());
    }
    if let Some(e) = adding {
        v.set(e.pair_key(), e.form_label());
    }
    v
}

/// Where a search run's files go by default: `<config stem>-covsearch` next
/// to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    let stem = config_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("search");
    config_path.with_file_name(format!("{stem}-covsearch"))
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
