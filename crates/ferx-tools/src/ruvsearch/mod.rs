//! Residual-error model search — Pharmpy `ruvsearch` (#1182).
//!
//! The third search tool of the #1175 epic, over the block the other two
//! leave alone: `[error_model]`. Its candidates are not MFL features but the
//! four residual-error structures Pharmpy's `ruvsearch` tries — IIV on the
//! residual error, a power form, a combined form, and a time-varying
//! magnitude — each one [`ModelEdit::SetErrorModel`] away from its parent,
//! so a candidate is the parent with one feature added and the rest of the
//! file invariant. The `power(σ, P)` form is the piece of core that landed
//! with this tool; the other three were already expressible.
//!
//! # The algorithm
//!
//! **The base.** Pharmpy's candidates are defined on a *proportional* base.
//! An input model whose `[error_model]` is anything else — additive,
//! combined, or proportional with an `iiv_on_ruv` — is first rewritten as a
//! plain proportional model and fitted; that fit is the parent of iteration
//! one, and the input is kept for the final comparison. An input that is
//! already plain proportional is its own base.
//!
//! **An iteration.** Every candidate not yet skipped is derived from the
//! parent and fitted. Each is compared with the parent by a likelihood-ratio
//! test at `p_value` (`df` = the parameters it adds, always one); among the
//! candidates that pass the strictness gate *and* are significant, the one
//! with the lowest OFV becomes the next parent, and its family leaves the
//! candidate list — selecting `power` also retires `combined`, and vice
//! versa, since Pharmpy never tests one after the other (they are
//! near-equivalent in shape). Up to `max_iter` iterations; an iteration
//! that accepts nothing ends the search.
//!
//! **The final comparison.** Pharmpy's: the selected model must beat the
//! *input* by the `df = 1` χ² cutoff at `p_value`, or the input is returned;
//! when a proportional base was fitted, it must beat that too, or the base is
//! returned.
//!
//! # The CWRES pre-screen (`cwres_prescreen = true`)
//!
//! Pharmpy does not fit the candidates to the data. It fits them to the
//! **conditional weighted residuals** of the parent: a dataset with
//! `DV = CWRES`, the parent's `IPRED` and `TAD` as columns, and a model
//! `Y = θ + η + ε` whose residual error carries the candidate structure —
//! orders of magnitude cheaper than a PK fit. The candidate whose CWRES model
//! improves most over the CWRES base (by more than the cutoff) is then built
//! on the real model, with the shape parameters the CWRES fit estimated as
//! its initial values, and that one candidate is fitted and judged by the
//! same likelihood-ratio test. Off by default here: the full refit is the
//! path that is trivially right, the pre-screen the one that is cheap, and
//! [`crate::ruvsearch`]'s slow test asserts the two select the same feature
//! on the fixture set. See [`cwres`] for the screening models.
//!
//! # What a candidate becomes
//!
//! | candidate | edit on the parent | new parameter, init |
//! |---|---|---|
//! | `IIV_on_RUV` | `iiv_on_ruv = ETA_RUV` | `omega ETA_RUV ~ 0.09` |
//! | `power` | `proportional(σ)` → `power(σ, RUV_POW)` | `theta RUV_POW(1.0, 0.01, 10.0)` |
//! | `combined` | `proportional(σ)` → `combined(σ, ADD_ERR)` | `sigma ADD_ERR ~ (min DV / 2)²` |
//! | `time_varying{i}` | every σ `* (if (TAD < c_i) RUV_TV else 1.0)` | `theta RUV_TV(1.0, 0.01, 10.0)` |
//!
//! `c_i` is the `i / groups` quantile of the data's time after dose, Pharmpy's
//! `TAD.quantile(i / groups)`. The inits are Pharmpy's where it states one
//! for the full model (`0.09`, `(min DV / 2)²`, `power = 1` on a proportional
//! base); the time-varying θ starts at the neutral `1.0` rather than
//! Pharmpy's CWRES-model `0.1`, because a full refit has no CWRES estimate to
//! start from. The pre-screen path does, and uses it.
//!
//! # Deviations from Pharmpy, stated
//!
//! * The full-refit path fits every candidate to the data. Pharmpy has only
//!   the CWRES pre-screen; `cwres_prescreen = true` is that path.
//! * Pharmpy also fits an *additive* twin of a `power` or `combined` winner
//!   and ranks it alongside; ferx does not — `combined` with its
//!   proportional σ on the floor is that model, and the strictness gate
//!   reports it.
//! * `IIV_on_RUV` needs an interaction or Monte-Carlo method; on a
//!   `method = foce` base it is not tested, and the notes say so, rather than
//!   fitted under a method that would reject it.
//! * A candidate is judged by the same likelihood-ratio test and strictness
//!   gate as every other search here; Pharmpy's `strictness` string is the
//!   `[strictness]` section.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ferx_core::edit::{
    ErrorForm, ErrorSpecText, EtaDecl, ModelEdit, ModelText, SigmaDecl, ThetaDecl, TimeVaryingDecl,
};
use ferx_core::{CancelFlag, EstimationMethod, FitResult, Population};
use serde::Deserialize;

use crate::covsearch::{chi_square_isf, Lrt};
use crate::search::fitter::{RunnerFitter, StepFitter};
use crate::search::seed::seed_from;
use crate::search::{
    BaseModel, Candidate, CandidateResult, Criterion, FeatureVector, RankType, RunReport,
    SearchConfig,
};

pub mod cwres;
mod report;

pub use report::{
    final_model_path, models_dir, render_summary, steps_path, write_report, STEP_COLUMNS,
};

/// One residual-error feature the search can add to its parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RuvFeature {
    /// `iiv_on_ruv = ETA_RUV`: a log-normal per-subject scale on the residual SD.
    IivOnRuv,
    /// `power(σ, P)`: the proportional loading raised to an estimated exponent.
    Power,
    /// `combined(σ, σ_add)`: an additive component beside the proportional one.
    Combined,
    /// `time_varying{i}`: every σ scaled by θ for records with
    /// `TAD < c_i`, the `i / groups` quantile of time after dose.
    TimeVarying(usize),
}

impl RuvFeature {
    /// Pharmpy's model name for the candidate: `IIV_on_RUV`, `power`,
    /// `combined`, `time_varying{i}`.
    pub fn label(&self) -> String {
        match self {
            RuvFeature::IivOnRuv => "IIV_on_RUV".into(),
            RuvFeature::Power => "power".into(),
            RuvFeature::Combined => "combined".into(),
            RuvFeature::TimeVarying(i) => format!("time_varying{i}"),
        }
    }

    /// The family a `skip` entry names, and that a selection retires.
    pub fn family(&self) -> Family {
        match self {
            RuvFeature::IivOnRuv => Family::IivOnRuv,
            RuvFeature::Power => Family::Power,
            RuvFeature::Combined => Family::Combined,
            RuvFeature::TimeVarying(_) => Family::TimeVarying,
        }
    }
}

/// Pharmpy's `skip` categories — the four candidate families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Deserialize)]
pub enum Family {
    #[serde(rename = "IIV_on_RUV", alias = "iiv_on_ruv")]
    IivOnRuv,
    #[serde(rename = "power")]
    Power,
    #[serde(rename = "combined")]
    Combined,
    #[serde(rename = "time_varying")]
    TimeVarying,
}

impl Family {
    pub fn label(&self) -> &'static str {
        match self {
            Family::IivOnRuv => "IIV_on_RUV",
            Family::Power => "power",
            Family::Combined => "combined",
            Family::TimeVarying => "time_varying",
        }
    }
}

/// The `[ruvsearch]` section as written.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default = "default_groups")]
    groups: usize,
    #[serde(default = "default_p_value")]
    p_value: f64,
    #[serde(default)]
    skip: Vec<Family>,
    #[serde(default = "default_max_iter")]
    max_iter: usize,
    #[serde(default)]
    cwres_prescreen: bool,
}

fn default_groups() -> usize {
    4
}

fn default_p_value() -> f64 {
    0.001
}

fn default_max_iter() -> usize {
    3
}

impl Default for Section {
    fn default() -> Self {
        Section {
            groups: default_groups(),
            p_value: default_p_value(),
            skip: Vec::new(),
            max_iter: default_max_iter(),
            cwres_prescreen: false,
        }
    }
}

/// The `[ruvsearch]` section of a `.ferxsearch` file. Every key has Pharmpy's
/// default.
#[derive(Debug, Clone, PartialEq)]
pub struct RuvsearchOptions {
    /// The number of time-after-dose bins the time-varying candidates cut at:
    /// `groups − 1` candidates, at the `i / groups` quantiles.
    pub groups: usize,
    /// The likelihood-ratio level a candidate must reach, and the level of the
    /// final comparison's `df = 1` cutoff.
    pub p_value: f64,
    /// Families never tested.
    pub skip: Vec<Family>,
    /// Iterations, `1..=3` as in Pharmpy.
    pub max_iter: usize,
    /// Screen the candidates on the parent's CWRES first and refit only the
    /// winner (Pharmpy's path); `false` fits every candidate to the data.
    pub cwres_prescreen: bool,
}

impl Default for RuvsearchOptions {
    fn default() -> Self {
        Self::from_section(Section::default())
    }
}

impl RuvsearchOptions {
    fn from_section(s: Section) -> Self {
        RuvsearchOptions {
            groups: s.groups,
            p_value: s.p_value,
            skip: s.skip,
            max_iter: s.max_iter,
            cwres_prescreen: s.cwres_prescreen,
        }
    }

    /// Read the `[ruvsearch]` section of a loaded file — the defaults when it
    /// is absent — and check the rest of the file is consistent with a
    /// likelihood-ratio search. Like covsearch, a `[rank]` asking for a BIC
    /// or a `cutoff` is refused rather than ignored.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        let section = match config.tools.get("ruvsearch") {
            Some(table) => table
                .clone()
                .try_into::<Section>()
                .map_err(|e| format!("[ruvsearch]: {e}"))?,
            None => Section::default(),
        };
        let options = Self::from_section(section);
        options.validate()?;
        if let Some(kind) = config.rank.kind {
            if kind != RankType::Ofv {
                return Err(format!(
                    "[rank] type = \"{}\": ruvsearch selects by the likelihood-ratio test on \
                     the OFV at [ruvsearch] p_value; a BIC ranking does not apply. Remove the \
                     key, or set it to \"ofv\"",
                    rank_label(kind)
                ));
            }
        }
        if let Some(cutoff) = config.rank.cutoff {
            return Err(format!(
                "[rank] cutoff = {cutoff}: ruvsearch takes its threshold as a p-value — \
                 [ruvsearch] p_value — not as a ΔOFV. Remove the key"
            ));
        }
        if config.has_space() {
            return Err(
                "[space]: ruvsearch has no search space to declare — its candidates are the \
                 residual-error forms (IIV_on_RUV, power, combined, time_varying), chosen with \
                 [ruvsearch] skip. Remove the section"
                    .into(),
            );
        }
        Ok(options)
    }

    /// Pharmpy's `validate_input`: `groups ≥ 2`, `0 < p ≤ 1`, `1 ≤ max_iter ≤ 3`.
    pub fn validate(&self) -> Result<(), String> {
        if self.groups < 2 {
            return Err(format!(
                "[ruvsearch] groups = {}: must be at least 2",
                self.groups
            ));
        }
        if !(self.p_value > 0.0 && self.p_value <= 1.0) {
            return Err(format!(
                "[ruvsearch] p_value = {}: must be a probability in (0, 1]",
                self.p_value
            ));
        }
        if !(1..=3).contains(&self.max_iter) {
            return Err(format!(
                "[ruvsearch] max_iter = {}: must be 1, 2 or 3",
                self.max_iter
            ));
        }
        Ok(())
    }

    /// The `df = 1` χ² cutoff at `p_value` — Pharmpy's `cutoff`.
    pub fn cutoff(&self) -> f64 {
        chi_square_isf(self.p_value, 1)
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

/// One fitted model of the search, as the table reports it.
#[derive(Debug, Clone)]
pub struct StepRow {
    /// `0` for the input and the proportional base; the iteration otherwise.
    pub iteration: usize,
    /// The candidate's id in its step's runner directory.
    pub candidate: String,
    /// The feature this row tests; `None` for the input and the base.
    pub feature: Option<RuvFeature>,
    /// The row is a CWRES pre-screen fit — its `ofv` is on the CWRES scale
    /// and `cwres_dofv` its improvement over the CWRES base.
    pub screened: bool,
    /// The parent's OFV — the model the candidate is compared with. `NaN` for
    /// a row with no parent.
    pub parent_ofv: f64,
    pub ofv: Option<f64>,
    /// The likelihood-ratio comparison with the parent, when one was made.
    pub lrt: Option<Lrt>,
    /// The CWRES base OFV minus this row's, for a screened row.
    pub cwres_dofv: Option<f64>,
    /// Why no comparison could be made.
    pub note: Option<String>,
    /// Whether this row became the next parent (a screened row: whether its
    /// feature was the one refitted).
    pub selected: bool,
    pub converged: Option<bool>,
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

/// What a running search reports about its progress.
#[derive(Debug, Clone, PartialEq)]
pub enum RuvsearchEvent {
    /// The input model is being fitted.
    InputStarted,
    InputFinished {
        ofv: f64,
    },
    /// The input is not plain proportional; the proportional base is being
    /// fitted.
    BaseStarted,
    BaseFinished {
        ofv: f64,
    },
    /// An iteration is about to fit `candidates` models — CWRES screening
    /// models when `screening`.
    IterationStarted {
        iteration: usize,
        candidates: usize,
        screening: bool,
    },
    /// The pre-screen picked `feature` to refit (`None`: nothing beat the
    /// cutoff).
    Screened {
        iteration: usize,
        feature: Option<RuvFeature>,
    },
    /// An iteration finished; `selected` is the accepted feature and the new
    /// OFV, `None` when nothing was accepted.
    IterationFinished {
        iteration: usize,
        selected: Option<(RuvFeature, f64)>,
    },
    /// The final comparison returned `to` (`input` or `base`) instead of the
    /// last accepted model.
    Reverted {
        to: String,
        ofv: f64,
    },
}

/// A progress callback: plain data, no terminal concern of any kind.
pub type ProgressFn<'a> = &'a (dyn Fn(RuvsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct RuvsearchResult {
    pub options: RuvsearchOptions,
    /// The file's model, unedited.
    pub input_model: ModelText,
    pub input_ofv: f64,
    /// The parent of iteration one: `input`, or `base` when a proportional
    /// base had to be derived.
    pub base_id: String,
    pub base_ofv: f64,
    /// Every fitted model, in order.
    pub rows: Vec<StepRow>,
    pub final_id: String,
    pub final_model: ModelText,
    pub final_ofv: f64,
    /// The final model's fit; `None` only when the outcome was read back
    /// from a journal whose cached fit is gone.
    pub final_fit: Option<FitResult>,
    /// The features the final model carries beyond the base, in the order
    /// they were accepted. Empty when the search returned the input or base.
    pub features: Vec<RuvFeature>,
    /// Every fitted model's text, by id.
    pub models: BTreeMap<String, ModelText>,
    /// Things a user should read once: a family not tested and why, a
    /// reversion, a journal warning.
    pub notes: Vec<String>,
    /// The search stopped on a flipped [`CancelFlag`]; `rows` is partial.
    pub cancelled: bool,
}

impl RuvsearchResult {
    /// The rows of one iteration.
    pub fn iteration_rows(&self, iteration: usize) -> impl Iterator<Item = &StepRow> {
        self.rows.iter().filter(move |r| r.iteration == iteration)
    }

    /// The number of iterations run.
    pub fn n_iterations(&self) -> usize {
        self.rows.iter().map(|r| r.iteration).max().unwrap_or(0)
    }
}

/// Everything a [`run_ruvsearch`] call takes beyond the file.
#[derive(Default)]
pub struct RuvsearchRun<'a> {
    /// Where the per-step journals, `steps.csv`, `models/` and `final.ferx`
    /// go. `None` keeps everything in memory: no resume, no files.
    pub dir: Option<PathBuf>,
    /// Overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Run a residual-error search from a loaded `.ferxsearch` file and its
/// base model. Writes `steps.csv`, `models/<id>.ferx` and `final.ferx` into
/// `run.dir` when given.
pub fn run_ruvsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: RuvsearchRun<'_>,
) -> Result<RuvsearchResult, String> {
    let options = RuvsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = Criterion::Ofv;
    let fitter = RunnerFitter {
        threads: run.threads.or(config.run.threads).unwrap_or(0),
        dir: run.dir.clone(),
        cancel: run.cancel.clone(),
        data: &base.prepared.population,
        options: run_options,
    };
    let space = Space::from_base(base, &options)?;
    let result = search(&fitter, space, &options, run.progress)?;
    if let Some(dir) = &run.dir {
        write_report(dir, &result)?;
    }
    Ok(result)
}

/// Where a search run's files go by default: `<config stem>-ruvsearch` next
/// to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    let stem = config_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("search");
    config_path.with_file_name(format!("{stem}-ruvsearch"))
}

/// What the search knows about the data and the input before it fits
/// anything.
#[derive(Debug)]
pub(crate) struct Space {
    pub input_model: ModelText,
    /// The input's `[error_model]`, read back.
    pub input_spec: ErrorSpecText,
    /// The time-varying cutoffs `c_1 .. c_{groups−1}`: the `i / groups`
    /// quantiles of the data's time after dose. Empty when the data has no
    /// dose to measure from.
    pub tad_cutoffs: Vec<f64>,
    /// Pharmpy's additive-σ init for a `combined` candidate: `(min DV / 2)²`,
    /// or `0.01` when the smallest observation is zero.
    pub add_sigma_init: f64,
    /// Whether the input's estimation method admits `iiv_on_ruv`.
    pub interaction_ok: bool,
    /// The input's dataset, for the CWRES pre-screen.
    pub population: Population,
    pub notes: Vec<String>,
}

impl Space {
    pub(crate) fn from_base(base: &BaseModel, options: &RuvsearchOptions) -> Result<Space, String> {
        let input_spec = ErrorSpecText::read(&base.text)
            .map_err(|e| format!("ruvsearch cannot read the base model's [error_model]: {e}"))?
            .ok_or_else(|| {
                "ruvsearch: the base model has no [error_model] block; a residual-error search \
                 needs a residual error to start from"
                    .to_string()
            })?;
        if !input_spec.endpoint.eq_ignore_ascii_case("DV") {
            return Err(format!(
                "ruvsearch: the base model's error statement is on `{}`; only a `DV ~ ...` \
                 statement is searched",
                input_spec.endpoint
            ));
        }
        let population = &base.prepared.population;
        let fo = &base.prepared.parsed.fit_options;
        let interaction_ok = !matches!(fo.method, EstimationMethod::Foce)
            && !(matches!(
                fo.method,
                EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid
            ) && !fo.interaction);
        Ok(Self::build(
            base.text.clone(),
            input_spec,
            population.clone(),
            interaction_ok,
            options,
        ))
    }

    /// The space from its parts — the seam the unit tests use.
    pub(crate) fn build(
        input_model: ModelText,
        input_spec: ErrorSpecText,
        population: Population,
        interaction_ok: bool,
        options: &RuvsearchOptions,
    ) -> Space {
        let mut notes = Vec::new();
        let tad_cutoffs = tad_cutoffs(&population, options.groups);
        if tad_cutoffs.is_empty() && !options.skip.contains(&Family::TimeVarying) {
            notes.push(
                "time_varying not tested: no observation follows a dose, so there is no time \
                 after dose to cut at"
                    .into(),
            );
        }
        if !interaction_ok && !options.skip.contains(&Family::IivOnRuv) {
            notes.push(
                "IIV_on_RUV not tested: [fit_options] method has no η–ε interaction, and \
                 `iiv_on_ruv` needs one (focei, imp, impmap or saem)"
                    .into(),
            );
        }
        let add_sigma_init = additive_sigma_init(&population);
        Space {
            input_model,
            input_spec,
            tad_cutoffs,
            add_sigma_init,
            interaction_ok,
            population,
            notes,
        }
    }
}

/// The `i / groups` quantiles (`i = 1 .. groups − 1`) of every observation's
/// time after dose, linearly interpolated as pandas does; empty when no
/// observation follows a dose.
pub(crate) fn tad_cutoffs(population: &Population, groups: usize) -> Vec<f64> {
    let mut tads: Vec<f64> = population
        .subjects
        .iter()
        .flat_map(|s| (0..s.observations.len()).map(|j| s.time_after_dose(j)))
        .filter(|t| t.is_finite())
        .collect();
    if tads.is_empty() {
        return Vec::new();
    }
    tads.sort_by(f64::total_cmp);
    (1..groups)
        .map(|i| ferx_core::stats::convergence::quantile_sorted(&tads, i as f64 / groups as f64))
        .collect()
}

/// Pharmpy's `_get_prop_init`: `(min DV / 2)²`, or `0.01` when the smallest
/// observation is zero (a variance).
fn additive_sigma_init(population: &Population) -> f64 {
    let dv_min = population
        .subjects
        .iter()
        .flat_map(|s| s.observations.iter().copied())
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !dv_min.is_finite() || dv_min == 0.0 {
        0.01
    } else {
        (dv_min / 2.0) * (dv_min / 2.0)
    }
}

/// The model the search currently stands on.
#[derive(Debug, Clone)]
struct Node {
    id: String,
    model: ModelText,
    spec: ErrorSpecText,
    fit: Option<FitResult>,
    ofv: f64,
    n_parameters: usize,
    features: Vec<RuvFeature>,
}

impl Node {
    fn from_result(
        result: &CandidateResult,
        model: ModelText,
        spec: ErrorSpecText,
        features: Vec<RuvFeature>,
    ) -> Node {
        let fit = result.fit.clone();
        Node {
            id: result.id.clone(),
            n_parameters: fit.as_ref().map(|f| f.n_parameters).unwrap_or(0),
            ofv: result.ofv.unwrap_or(f64::NAN),
            model,
            spec,
            fit,
            features,
        }
    }
}

/// A name for a new declaration that no `[parameters]` line already uses.
fn fresh_name(model: &ModelText, base: &str) -> String {
    let taken: BTreeSet<String> = model
        .block_lines("parameters")
        .iter()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let kind = it.next()?;
            if !matches!(
                kind.to_ascii_lowercase().as_str(),
                "theta" | "omega" | "sigma" | "kappa"
            ) {
                return None;
            }
            let name: String = it
                .next()?
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            Some(name)
        })
        .collect();
    if !taken.contains(base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|n| !taken.contains(n))
        .expect("an unused suffix exists")
}

/// Pharmpy's inits for the four features on the full model.
const IIV_ON_RUV_INIT: f64 = 0.09;
const POWER_INIT: f64 = 1.0;
const POWER_LOWER: f64 = 0.01;
const POWER_UPPER: f64 = 10.0;
const TIME_VARYING_INIT: f64 = 1.0;
const TIME_VARYING_LOWER: f64 = 0.01;
const TIME_VARYING_UPPER: f64 = 10.0;

/// The shape parameter a feature's candidate starts from — Pharmpy's full-model
/// init, or the CWRES pre-screen's estimate when the screen ran.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Init {
    pub value: f64,
}

/// The candidate that adds `feature` to `parent`, or `None` when the parent's
/// error model cannot take it (a `power` on a model that is not proportional,
/// a second `iiv_on_ruv`). `init` overrides the feature's default init.
fn derive(
    id: &str,
    parent: &Node,
    feature: RuvFeature,
    space: &Space,
    init: Option<Init>,
) -> Result<Option<Candidate>, String> {
    let mut spec = parent.spec.clone();
    let model_for_names = &parent.model;
    match feature {
        RuvFeature::IivOnRuv => {
            if spec.iiv_on_ruv.is_some() {
                return Ok(None);
            }
            let v = init.map(|i| i.value).unwrap_or(IIV_ON_RUV_INIT);
            spec = spec.with_iiv_on_ruv(EtaDecl::new(fresh_name(model_for_names, "ETA_RUV"), v));
        }
        RuvFeature::Power => {
            if spec.form != ErrorForm::Proportional {
                return Ok(None);
            }
            spec.form = ErrorForm::Power;
            let p = init.map(|i| i.value).unwrap_or(POWER_INIT);
            spec = spec.with_exponent(ThetaDecl::new(
                fresh_name(model_for_names, "RUV_POW"),
                p,
                POWER_LOWER,
                POWER_UPPER,
            ));
        }
        RuvFeature::Combined => {
            if spec.form != ErrorForm::Proportional {
                return Ok(None);
            }
            spec.form = ErrorForm::Combined;
            spec.sigmas.push(SigmaDecl {
                name: fresh_name(model_for_names, "ADD_ERR"),
                init: init.map(|i| i.value).unwrap_or(space.add_sigma_init),
                as_sd: false,
            });
        }
        RuvFeature::TimeVarying(i) => {
            if spec.time_varying.is_some() {
                return Ok(None);
            }
            let Some(&cutoff) = i.checked_sub(1).and_then(|k| space.tad_cutoffs.get(k)) else {
                return Ok(None);
            };
            let t = init.map(|i| i.value).unwrap_or(TIME_VARYING_INIT);
            spec = spec.with_time_varying(TimeVaryingDecl::new(
                cutoff,
                ThetaDecl::new(
                    fresh_name(model_for_names, "RUV_TV"),
                    t,
                    TIME_VARYING_LOWER,
                    TIME_VARYING_UPPER,
                ),
            ));
        }
    }
    let mut model = parent.model.clone();
    if let Some(fit) = &parent.fit {
        seed_from(&mut model, fit)?;
    }
    model
        .apply(ModelEdit::SetErrorModel(spec.clone()))
        .map_err(|e| format!("{id}: adding {}: {e}", feature.label()))?;
    let mut features = parent.features.clone();
    features.push(feature);
    Ok(Some(
        Candidate::new(id, model)
            .parent(parent.id.clone())
            .features(feature_vector(&spec)),
    ))
}

/// The error model as a feature vector, for the runner's dedup and table.
fn feature_vector(spec: &ErrorSpecText) -> FeatureVector {
    let mut v = FeatureVector::new().with("ruv", spec.form.label());
    if let Some(eta) = &spec.iiv_on_ruv {
        v.set("iiv_on_ruv", eta.name.clone());
    }
    if let Some(tv) = &spec.time_varying {
        v.set("time_varying", format!("{}", tv.cutoff));
    }
    v
}

/// The features an iteration tests, in Pharmpy's order, minus the skipped
/// families and the ones the parent cannot take. Time-varying cutoffs that
/// coincide (a coarse `TAD` distribution) yield one candidate, not two
/// identical models.
fn candidates_for(parent: &Node, space: &Space, skip: &BTreeSet<Family>) -> Vec<RuvFeature> {
    let mut out = Vec::new();
    if !skip.contains(&Family::IivOnRuv) && space.interaction_ok && parent.spec.iiv_on_ruv.is_none()
    {
        out.push(RuvFeature::IivOnRuv);
    }
    // Pharmpy tests `power` and `combined` together or not at all.
    if !skip.contains(&Family::Power)
        && !skip.contains(&Family::Combined)
        && parent.spec.form == ErrorForm::Proportional
    {
        out.push(RuvFeature::Power);
        out.push(RuvFeature::Combined);
    }
    if !skip.contains(&Family::TimeVarying) && parent.spec.time_varying.is_none() {
        let mut seen: Vec<f64> = Vec::new();
        for (k, &c) in space.tad_cutoffs.iter().enumerate() {
            if seen.contains(&c) {
                continue;
            }
            seen.push(c);
            out.push(RuvFeature::TimeVarying(k + 1));
        }
    }
    out
}

/// After `feature` is accepted, the families no longer tested.
fn retire(skip: &mut BTreeSet<Family>, feature: RuvFeature) {
    match feature.family() {
        Family::Power | Family::Combined => {
            skip.insert(Family::Power);
            skip.insert(Family::Combined);
        }
        f => {
            skip.insert(f);
        }
    }
}

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &RuvsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<RuvsearchResult, String> {
    let emit = |event: RuvsearchEvent| {
        if let Some(p) = progress {
            p(event);
        }
    };
    let mut notes = space.notes.clone();
    let mut rows: Vec<StepRow> = Vec::new();
    let mut models: BTreeMap<String, ModelText> = BTreeMap::new();
    let cutoff = options.cutoff();

    // ── the input ───────────────────────────────────────────────────────
    emit(RuvsearchEvent::InputStarted);
    let input_candidate = Candidate::new("input", space.input_model.clone())
        .features(feature_vector(&space.input_spec));
    let report = fitter.fit_step("input", std::slice::from_ref(&input_candidate))?;
    notes.extend(report.warnings.iter().cloned());
    let input = root_node(&report, &input_candidate, &space.input_spec, "input")?;
    rows.push(row_of(
        &report,
        &input_candidate.id,
        0,
        None,
        false,
        f64::NAN,
        None,
    ));
    models.insert(input.id.clone(), input.model.clone());
    emit(RuvsearchEvent::InputFinished { ofv: input.ofv });
    if report.cancelled {
        return Ok(finish(
            options,
            &space,
            &input,
            None,
            rows,
            input.clone(),
            models,
            notes,
            true,
        )
        .0);
    }

    // ── the proportional base, when the input is not one ────────────────
    let is_plain_proportional = space.input_spec.form == ErrorForm::Proportional
        && space.input_spec.exponent.is_none()
        && space.input_spec.iiv_on_ruv.is_none()
        && space.input_spec.time_varying.is_none();
    let mut base: Option<Node> = None;
    let mut parent = input.clone();
    if !is_plain_proportional {
        emit(RuvsearchEvent::BaseStarted);
        let spec = proportional_spec(&space.input_spec, &space.input_model);
        let mut model = space.input_model.clone();
        if let Some(fit) = &input.fit {
            seed_from(&mut model, fit)?;
        }
        model
            .apply(ModelEdit::SetErrorModel(spec.clone()))
            .map_err(|e| format!("base: rewriting the input as proportional: {e}"))?;
        notes.push(format!(
            "the input's error model is `{}`; the search starts from the proportional base \
             `{}`, and the final comparison considers both",
            space.input_spec.render_statement(),
            spec.render_statement()
        ));
        let candidate = Candidate::new("base", model)
            .parent("input")
            .features(feature_vector(&spec));
        let report = fitter.fit_step("base", std::slice::from_ref(&candidate))?;
        notes.extend(report.warnings.iter().cloned());
        let node = root_node(&report, &candidate, &spec, "base")?;
        rows.push(row_of(
            &report,
            &candidate.id,
            0,
            None,
            false,
            input.ofv,
            None,
        ));
        models.insert(node.id.clone(), node.model.clone());
        emit(RuvsearchEvent::BaseFinished { ofv: node.ofv });
        parent = node.clone();
        base = Some(node);
        if report.cancelled {
            return Ok(finish(
                options,
                &space,
                &input,
                base.as_ref(),
                rows,
                parent,
                models,
                notes,
                true,
            )
            .0);
        }
    }

    // ── the iterations ──────────────────────────────────────────────────
    let mut skip: BTreeSet<Family> = options.skip.iter().copied().collect();
    let mut cancelled = false;
    for iteration in 1..=options.max_iter {
        let features = candidates_for(&parent, &space, &skip);
        if features.is_empty() {
            notes.push(format!("iteration {iteration}: no candidate left to test"));
            break;
        }
        let outcome = if options.cwres_prescreen {
            screened_iteration(
                fitter,
                &space,
                &parent,
                &features,
                iteration,
                options,
                cutoff,
                &emit,
                &mut rows,
                &mut models,
                &mut notes,
            )?
        } else {
            full_iteration(
                fitter,
                &space,
                &parent,
                &features,
                iteration,
                options,
                &emit,
                &mut rows,
                &mut models,
                &mut notes,
            )?
        };
        match outcome {
            Outcome::Cancelled => {
                cancelled = true;
                break;
            }
            Outcome::Nothing => break,
            Outcome::Rejected(feature) => {
                // Pharmpy: the screened pick did not survive the refit; its
                // family is retired and the search goes on from the parent.
                retire(&mut skip, feature);
            }
            Outcome::Accepted(node, feature) => {
                retire(&mut skip, feature);
                parent = node;
            }
        }
    }

    let (result, reverted) = finish(
        options,
        &space,
        &input,
        base.as_ref(),
        rows,
        parent,
        models,
        notes,
        cancelled,
    );
    if let Some((to, ofv)) = reverted {
        emit(RuvsearchEvent::Reverted { to, ofv });
    }
    Ok(result)
}

/// The plain proportional error model the search starts from when the input
/// is anything else: the input's proportional σ kept as declared, or a new
/// one at Pharmpy's `0.09` variance.
fn proportional_spec(input: &ErrorSpecText, model: &ModelText) -> ErrorSpecText {
    let prop = match input.form {
        ErrorForm::Proportional | ErrorForm::Combined | ErrorForm::Power => input.sigmas[0].clone(),
        ErrorForm::Additive => SigmaDecl {
            name: fresh_name(model, "PROP_ERR"),
            init: 0.09,
            as_sd: false,
        },
    };
    ErrorSpecText::new(input.endpoint.clone(), ErrorForm::Proportional, vec![prop])
}

/// The node for a root fit — input or base — or the error that stops the
/// search: a root that could not be fitted, failed the gate, or has no fit
/// to seed from.
fn root_node(
    report: &RunReport,
    candidate: &Candidate,
    spec: &ErrorSpecText,
    what: &str,
) -> Result<Node, String> {
    let result = report
        .results
        .iter()
        .find(|r| r.id == candidate.id)
        .ok_or_else(|| format!("the {what} model was not fitted"))?;
    if let Some(e) = &result.error {
        return Err(format!("the {what} model could not be fitted: {e}"));
    }
    if !result.verdict.passed {
        return Err(format!(
            "the {what} model fails the strictness gate ({}); a search from a fit that cannot \
             be trusted has nothing to compare against. Fix the fit or relax [strictness]",
            result.verdict.failures.join("; ")
        ));
    }
    if result.fit.is_none() {
        return Err(format!(
            "the {what} model's fit is missing from the resumed journal, so nothing can be \
             seeded from it; refit without `resume`"
        ));
    }
    Ok(Node::from_result(
        result,
        candidate.model.clone(),
        spec.clone(),
        Vec::new(),
    ))
}

/// What one iteration reports back to the driver.
enum Outcome {
    Cancelled,
    /// Nothing was significant: the search ends.
    Nothing,
    /// The pre-screen's pick was refitted and not accepted; its family is
    /// retired and the search continues.
    Rejected(RuvFeature),
    Accepted(Node, RuvFeature),
}

/// One full-refit iteration: every candidate fitted to the data.
#[allow(clippy::too_many_arguments)]
fn full_iteration(
    fitter: &dyn StepFitter,
    space: &Space,
    parent: &Node,
    features: &[RuvFeature],
    iteration: usize,
    options: &RuvsearchOptions,
    emit: &dyn Fn(RuvsearchEvent),
    rows: &mut Vec<StepRow>,
    models: &mut BTreeMap<String, ModelText>,
    notes: &mut Vec<String>,
) -> Result<Outcome, String> {
    let mut candidates: Vec<(RuvFeature, Candidate)> = Vec::with_capacity(features.len());
    for &feature in features {
        let id = format!("{}-{iteration}", feature.label());
        if let Some(c) = derive(&id, parent, feature, space, None)? {
            candidates.push((feature, c));
        }
    }
    if candidates.is_empty() {
        return Ok(Outcome::Nothing);
    }
    emit(RuvsearchEvent::IterationStarted {
        iteration,
        candidates: candidates.len(),
        screening: false,
    });
    let dir = format!("iteration-{iteration}");
    let list: Vec<Candidate> = candidates.iter().map(|(_, c)| c.clone()).collect();
    let report = fitter.fit_step(&dir, &list)?;
    notes.extend(report.warnings.iter().cloned());
    let mut step_rows: Vec<(RuvFeature, StepRow)> = Vec::new();
    for (feature, c) in &candidates {
        if report.results.iter().any(|r| r.id == c.id) {
            step_rows.push((
                *feature,
                row_of(
                    &report,
                    &c.id,
                    iteration,
                    Some(*feature),
                    false,
                    parent.ofv,
                    Some((parent, options.p_value)),
                ),
            ));
            models.insert(c.id.clone(), c.model.clone());
        }
    }
    if report.cancelled {
        rows.extend(step_rows.into_iter().map(|(_, r)| r));
        emit(RuvsearchEvent::IterationFinished {
            iteration,
            selected: None,
        });
        return Ok(Outcome::Cancelled);
    }
    // Lowest OFV among the candidates that passed the gate and the test.
    let winner = step_rows
        .iter()
        .enumerate()
        .filter(|(_, (_, r))| r.passed && r.lrt.is_some_and(|t| t.significant))
        .min_by(|(_, (_, a)), (_, (_, b))| {
            a.ofv
                .unwrap_or(f64::NAN)
                .total_cmp(&b.ofv.unwrap_or(f64::NAN))
        })
        .map(|(i, _)| i);
    if let Some(i) = winner {
        step_rows[i].1.selected = true;
    }
    let selected = winner.map(|i| (step_rows[i].0, step_rows[i].1.ofv.unwrap_or(f64::NAN)));
    emit(RuvsearchEvent::IterationFinished {
        iteration,
        selected,
    });
    let Some(i) = winner else {
        rows.extend(step_rows.into_iter().map(|(_, r)| r));
        return Ok(Outcome::Nothing);
    };
    let (feature, candidate) = &candidates[i];
    let result = report
        .results
        .iter()
        .find(|r| r.id == candidate.id)
        .expect("the winner is a row of the report");
    let spec =
        ErrorSpecText::read(&candidate.model)?.expect("a derived candidate has an [error_model]");
    let mut features = parent.features.clone();
    features.push(*feature);
    let node = Node::from_result(result, candidate.model.clone(), spec, features);
    if node.fit.is_none() {
        notes.push(format!(
            "iteration {iteration}: the winning fit for {} is not in the journal cache, so the \
             next iteration starts from the file's initial estimates instead of the parent's",
            feature.label()
        ));
    }
    rows.extend(step_rows.into_iter().map(|(_, r)| r));
    Ok(Outcome::Accepted(node, *feature))
}

/// One pre-screened iteration: the candidates fitted to the parent's CWRES,
/// the best of them refitted on the data.
#[allow(clippy::too_many_arguments)]
fn screened_iteration(
    fitter: &dyn StepFitter,
    space: &Space,
    parent: &Node,
    features: &[RuvFeature],
    iteration: usize,
    options: &RuvsearchOptions,
    cutoff: f64,
    emit: &dyn Fn(RuvsearchEvent),
    rows: &mut Vec<StepRow>,
    models: &mut BTreeMap<String, ModelText>,
    notes: &mut Vec<String>,
) -> Result<Outcome, String> {
    let Some(fit) = &parent.fit else {
        return Err(format!(
            "iteration {iteration}: the parent's fit is not available (a resumed journal whose \
             cached fit is gone), so its CWRES cannot be screened; refit without `resume`, or \
             set cwres_prescreen = false"
        ));
    };
    let screen = cwres::Screen::build(
        fit,
        &space.population,
        features,
        &space.tad_cutoffs,
        iteration,
    )?;
    emit(RuvsearchEvent::IterationStarted {
        iteration,
        candidates: screen.candidates.len() - 1,
        screening: true,
    });
    let report = fitter.fit_step_on(
        &format!("screen-{iteration}"),
        &screen.candidates,
        &screen.population,
    )?;
    notes.extend(report.warnings.iter().cloned());
    for c in &screen.candidates {
        models.insert(c.id.clone(), c.model.clone());
    }
    let base_ofv = report
        .results
        .iter()
        .find(|r| r.id == screen.base_id)
        .and_then(|r| r.ofv);
    // The CWRES base is a row too, so the table shows what the screened
    // drops are measured from.
    if report.results.iter().any(|r| r.id == screen.base_id) {
        rows.push(row_of(
            &report,
            &screen.base_id,
            iteration,
            None,
            true,
            f64::NAN,
            None,
        ));
    }
    let mut screened: Vec<(RuvFeature, StepRow, Option<Init>)> = Vec::new();
    for (feature, id) in &screen.features {
        let Some(result) = report.results.iter().find(|r| r.id == *id) else {
            continue;
        };
        let dofv = match (base_ofv, result.ofv) {
            (Some(b), Some(o)) => Some(b - o),
            _ => None,
        };
        let mut row = row_of(
            &report,
            id,
            iteration,
            Some(*feature),
            true,
            base_ofv.unwrap_or(f64::NAN),
            None,
        );
        row.cwres_dofv = dofv;
        if base_ofv.is_none() {
            row.note = Some("the CWRES base model produced no OFV".into());
        }
        let init = result
            .fit
            .as_ref()
            .map(|f| cwres::init_from_screen(*feature, f));
        screened.push((*feature, row, init.flatten()));
    }
    if report.cancelled {
        rows.extend(screened.into_iter().map(|(_, r, _)| r));
        emit(RuvsearchEvent::Screened {
            iteration,
            feature: None,
        });
        emit(RuvsearchEvent::IterationFinished {
            iteration,
            selected: None,
        });
        return Ok(Outcome::Cancelled);
    }
    // Pharmpy's `_create_best_model`: the largest CWRES improvement, if any
    // beats the cutoff. Strictness is not consulted on the screen — the
    // refit is what is judged.
    let best = screened
        .iter()
        .enumerate()
        .filter(|(_, (_, r, _))| r.error_free() && r.cwres_dofv.is_some_and(|d| d > cutoff))
        .max_by(|(_, (_, a, _)), (_, (_, b, _))| {
            a.cwres_dofv.unwrap().total_cmp(&b.cwres_dofv.unwrap())
        })
        .map(|(i, _)| i);
    let Some(i) = best else {
        emit(RuvsearchEvent::Screened {
            iteration,
            feature: None,
        });
        emit(RuvsearchEvent::IterationFinished {
            iteration,
            selected: None,
        });
        rows.extend(screened.into_iter().map(|(_, r, _)| r));
        return Ok(Outcome::Nothing);
    };
    screened[i].1.selected = true;
    let (feature, _, init) = &screened[i];
    let feature = *feature;
    let init = *init;
    emit(RuvsearchEvent::Screened {
        iteration,
        feature: Some(feature),
    });
    rows.extend(screened.into_iter().map(|(_, r, _)| r));

    // The refit of the pick, on the data, from the screen's estimate.
    let id = format!("{}-{iteration}", feature.label());
    let Some(candidate) = derive(&id, parent, feature, space, init)? else {
        return Ok(Outcome::Nothing);
    };
    let dir = format!("iteration-{iteration}");
    let report = fitter.fit_step(&dir, std::slice::from_ref(&candidate))?;
    notes.extend(report.warnings.iter().cloned());
    models.insert(candidate.id.clone(), candidate.model.clone());
    let mut row = row_of(
        &report,
        &candidate.id,
        iteration,
        Some(feature),
        false,
        parent.ofv,
        Some((parent, options.p_value)),
    );
    if report.cancelled {
        rows.push(row);
        emit(RuvsearchEvent::IterationFinished {
            iteration,
            selected: None,
        });
        return Ok(Outcome::Cancelled);
    }
    let accepted = row.passed && row.lrt.is_some_and(|t| t.significant);
    row.selected = accepted;
    let ofv = row.ofv;
    rows.push(row);
    emit(RuvsearchEvent::IterationFinished {
        iteration,
        selected: accepted.then(|| (feature, ofv.unwrap_or(f64::NAN))),
    });
    if !accepted {
        notes.push(format!(
            "iteration {iteration}: the pre-screen picked {} but its refit was not accepted; \
             the family is not tested again",
            feature.label()
        ));
        return Ok(Outcome::Rejected(feature));
    }
    let result = report
        .results
        .iter()
        .find(|r| r.id == candidate.id)
        .expect("the refit is a row of the report");
    let spec =
        ErrorSpecText::read(&candidate.model)?.expect("a derived candidate has an [error_model]");
    let mut features = parent.features.clone();
    features.push(feature);
    Ok(Outcome::Accepted(
        Node::from_result(result, candidate.model.clone(), spec, features),
        feature,
    ))
}

impl StepRow {
    /// The fit exists and produced an OFV.
    fn error_free(&self) -> bool {
        self.ofv.is_some()
    }
}

/// The table row for one result of a report, judged against `judge` when
/// given: the parent and the level.
fn row_of(
    report: &RunReport,
    id: &str,
    iteration: usize,
    feature: Option<RuvFeature>,
    screened: bool,
    parent_ofv: f64,
    judge: Option<(&Node, f64)>,
) -> StepRow {
    let result = report
        .results
        .iter()
        .find(|r| r.id == id)
        .expect("row_of is called for a result the report holds");
    let (lrt, note) = match judge {
        Some((parent, alpha)) => judge_one(parent, result, alpha),
        None => (None, None),
    };
    StepRow {
        iteration,
        candidate: result.id.clone(),
        feature,
        screened,
        parent_ofv,
        ofv: result.ofv,
        lrt,
        cwres_dofv: None,
        note,
        selected: false,
        converged: result.converged,
        passed: result.verdict.passed && result.error.is_none(),
        failures: result.verdict.failures.clone(),
        seconds: result.seconds,
    }
}

fn judge_one(parent: &Node, result: &CandidateResult, alpha: f64) -> (Option<Lrt>, Option<String>) {
    if let Some(e) = &result.error {
        return (None, Some(e.message.clone()));
    }
    let Some(ofv) = result.ofv else {
        return (None, Some("no OFV".into()));
    };
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
    match Lrt::forward(parent.ofv, parent.n_parameters, ofv, n, alpha) {
        Ok(t) => (Some(t), None),
        Err(e) => (None, Some(e)),
    }
}

/// Pharmpy's final comparison, then the result.
#[allow(clippy::too_many_arguments)]
fn finish(
    options: &RuvsearchOptions,
    space: &Space,
    input: &Node,
    base: Option<&Node>,
    rows: Vec<StepRow>,
    selected: Node,
    models: BTreeMap<String, ModelText>,
    mut notes: Vec<String>,
    cancelled: bool,
) -> (RuvsearchResult, Option<(String, f64)>) {
    let cutoff = options.cutoff();
    let mut final_node = selected;
    let mut reverted: Option<&str> = None;
    // The selected model must beat the input by the cutoff.
    if final_node.id != input.id && !(input.ofv - final_node.ofv >= cutoff) {
        final_node = input.clone();
        reverted = Some("input");
    }
    // And the proportional base, when there was one.
    if let Some(b) = base {
        if final_node.id != b.id && !(b.ofv - final_node.ofv >= cutoff) {
            final_node = b.clone();
            reverted = Some("base");
        }
    }
    if let Some(to) = reverted {
        notes.push(format!(
            "the selected model did not beat the {to} model (OFV {:.3}) by the df = 1 cutoff \
             {cutoff:.3} at p = {}; the {to} model is returned",
            final_node.ofv, options.p_value
        ));
    }
    let base_node = base.unwrap_or(input);
    let event = reverted.map(|to| (to.to_string(), final_node.ofv));
    (
        RuvsearchResult {
            options: options.clone(),
            input_model: space.input_model.clone(),
            input_ofv: input.ofv,
            base_id: base_node.id.clone(),
            base_ofv: base_node.ofv,
            rows,
            final_id: final_node.id.clone(),
            final_model: final_node.model,
            final_ofv: final_node.ofv,
            final_fit: final_node.fit,
            features: final_node.features,
            models,
            notes,
            cancelled,
        },
        event,
    )
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
