//! Variability-structure search — Pharmpy `iivsearch` (#1183).
//!
//! The fourth search tool of the #1175 epic, over the block the others leave
//! alone: which `[individual_parameters]` carry an η, and how their `omega`
//! declarations are blocked. Its moves are [`ModelEdit::DropIiv`] /
//! [`ModelEdit::AddIiv`] (the number of η) and [`ModelEdit::SetOmegaBlock`]
//! / [`ModelEdit::SplitOmegaBlock`] (the correlation structure), each on a
//! parameter written in the canonical `P = TVP * exp(ETA_P)` form — a
//! parameter written any other way is refused **by name** before anything
//! is fitted, never mis-edited (`VariabilityText::read`).
//!
//! # The space
//!
//! ```text
//! IIV?([CL,V,KA], EXP); COVARIANCE?(IIV, [CL,V,KA])
//! ```
//!
//! An `IIV?` statement names the parameters whose η is *searched*; a plain
//! `IIV(CL, EXP)` names one that is kept — Pharmpy's forced feature, and
//! what a modeller means by "CL always has variability". `COVARIANCE?`
//! names the parameters whose η may be blocked together; a plain
//! `COVARIANCE(IIV, [CL,V])` is a block every candidate carries. Only the
//! exponential form is searchable: it is the one the edits can reverse.
//! Pharmpy's default space is `IIV?(@IIV,EXP);COVARIANCE?(IIV,@IIV)` —
//! every η the input has, every correlation among them.
//!
//! # The algorithms
//!
//! All Pharmpy's (`tools/iivsearch/algorithms.py`), run as two stages: the
//! **number of η** under `algorithm`, then the **block structure** under
//! `correlation_algorithm` on the winner of the first.
//!
//! * **`top_down_exhaustive`** (the default) — the base carries every η the
//!   space names; one candidate per subset of the searched η, largest
//!   subsets first, the naive-pooled model included when nothing is kept.
//! * **`bottom_up_stepwise`** — the base carries only the kept η; each
//!   step adds one η to each parameter that lacks one, and the best step
//!   model becomes the next parent until no step improves on its parent.
//! * **`simultaneous_stepwise`** — bottom-up, but each new η is also tried
//!   *inside* each existing block and *paired* with each single η, so the
//!   block structure is decided as the η are added; there is no second
//!   stage.
//! * **`skip`** — no η are added or removed; only the block structure is
//!   searched.
//!
//! The block stage is `top_down_exhaustive`: over the parameters the
//! `COVARIANCE` statement names that carry an η, one candidate per
//! **single full block** of two or more of them (beside the kept blocks),
//! plus the all-diagonal model — Pharmpy's `_is_valid_block_combination`
//! admits exactly the cliques, so `[CL,V]+[KA,F]` is not a candidate; the
//! deviation is stated below.
//!
//! # Ranking
//!
//! Each step ranks its candidates *and its parent* on `[rank] type` among
//! the models that pass the strictness gate; the parent wins a tie, and a
//! `cutoff` is the improvement over the parent a candidate must show. The
//! default is the **BIC(iiv)** — `OFV + n_ω·ln(n_subjects)`, Pharmpy's
//! `bic_iiv`, which is what `type = "bic"` means for this tool (#1177: the
//! observation-count BIC mis-ranks variability structures; the mixed BIC is
//! `bic_mixed`). At the end the selected model is compared with the
//! **input** the same way, and the input is returned when it ranks better.
//!
//! # Starting values
//!
//! Every candidate is seeded from its parent's estimates. A new block
//! starts at the parent's variances with its correlations taken from the
//! parent's empirical Bayes estimates (Pharmpy's `create_joint_distribution`
//! with `individual_estimates`), shrunk towards zero until the block is
//! positive definite; a block over three or more η is fitted with extra
//! starts (`block_retries` per η beyond two), since a full block is a
//! moderate local-minimum risk (`docs/examples/multistart.qmd`).
//!
//! # Deviations from Pharmpy, stated
//!
//! * A parameter the space does not name is left as it is. Pharmpy's
//!   `transform_into_search_space` removes an η the space does not mention;
//!   here a category the space does not name is not a request to change it
//!   — the same reading `modelsearch` takes.
//! * The simultaneous algorithm joins a new η into an existing block as one
//!   block. Pharmpy 2.2.0 builds that candidate from pairwise covariance
//!   features and, measured on the anchor, writes `[KA,V]+[CL]` where
//!   `[CL,V,KA]` was meant.
//! * `[rank] cutoff` is an improvement over the parent. Pharmpy's `cutoff`
//!   is read only by its likelihood-ratio ranking.
//! * A step whose parent fails the strictness gate keeps the parent when
//!   no candidate passes, with a note, rather than aborting.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use ferx_core::edit::{IivForm, ModelEdit, ModelText, VariabilityText};
use ferx_core::{CancelFlag, FitResult};
use serde::Deserialize;

use crate::search::fitter::{RunnerFitter, StepFitter};
use crate::search::mfl::{Feature, Mode, Modes, Operand, VariabilityEffect, VariabilityLevel};
use crate::search::seed::seed_from;
use crate::search::{
    BaseModel, Candidate, CandidateError, CandidateResult, Criterion, FeatureVector, RankType,
    RunReport, SearchConfig,
};

mod report;

pub use report::{
    final_model_path, models_dir, models_path, render_summary, write_report, MODEL_COLUMNS,
};

/// `[iivsearch] algorithm` — how the number of η is searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    #[default]
    TopDownExhaustive,
    BottomUpStepwise,
    SimultaneousStepwise,
    Skip,
}

impl Algorithm {
    pub fn label(&self) -> &'static str {
        match self {
            Algorithm::TopDownExhaustive => "top_down_exhaustive",
            Algorithm::BottomUpStepwise => "bottom_up_stepwise",
            Algorithm::SimultaneousStepwise => "simultaneous_stepwise",
            Algorithm::Skip => "skip",
        }
    }
}

/// `[iivsearch] correlation_algorithm` — how the block structure is searched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrelationAlgorithm {
    TopDownExhaustive,
    Skip,
}

impl CorrelationAlgorithm {
    pub fn label(&self) -> &'static str {
        match self {
            CorrelationAlgorithm::TopDownExhaustive => "top_down_exhaustive",
            CorrelationAlgorithm::Skip => "skip",
        }
    }
}

/// The `[iivsearch]` section as written.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default)]
    algorithm: Algorithm,
    #[serde(default)]
    correlation_algorithm: Option<CorrelationAlgorithm>,
    #[serde(default)]
    as_fullblock: bool,
    #[serde(default = "default_block_retries")]
    block_retries: usize,
}

fn default_block_retries() -> usize {
    2
}

impl Default for Section {
    fn default() -> Self {
        Section {
            algorithm: Algorithm::default(),
            correlation_algorithm: None,
            as_fullblock: false,
            block_retries: default_block_retries(),
        }
    }
}

/// The `[iivsearch]` section of a `.ferxsearch` file, plus the `[rank]` and
/// `[run]` keys the tool reads. Every key has Pharmpy's default.
#[derive(Debug, Clone, PartialEq)]
pub struct IivsearchOptions {
    pub algorithm: Algorithm,
    /// `None` — the default — is Pharmpy's derivation: `top_down_exhaustive`
    /// after any algorithm but `simultaneous_stepwise`, which has no block
    /// stage.
    pub correlation_algorithm: Option<CorrelationAlgorithm>,
    /// Pharmpy's `as_fullblock`: a bottom-up candidate blocks every η it
    /// carries rather than adding the new one diagonally.
    pub as_fullblock: bool,
    /// Extra starts per η beyond two in a candidate's largest block, on top
    /// of the run's starts: a 3-block gets `block_retries` more, a 4-block
    /// twice that.
    pub block_retries: usize,
    /// The run's starts per candidate (`[run] retries + 1`), which the
    /// block scaling is added to.
    pub starts: usize,
    /// `[rank] type`; the BIC(iiv) when the file does not say, and what
    /// `bic` means for this tool.
    pub rank: RankType,
    /// `[rank] cutoff`: the improvement over the parent a candidate must
    /// show to replace it. `None` — Pharmpy's default — takes any
    /// improvement.
    pub cutoff: Option<f64>,
}

impl Default for IivsearchOptions {
    fn default() -> Self {
        IivsearchOptions {
            algorithm: Algorithm::default(),
            correlation_algorithm: None,
            as_fullblock: false,
            block_retries: default_block_retries(),
            starts: crate::search::RunOptions::default().n_starts,
            rank: RankType::BicIiv,
            cutoff: None,
        }
    }
}

impl IivsearchOptions {
    /// Read `[iivsearch]`, `[rank]` and `[run]` off a loaded file.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        config.require_space("iivsearch", "IIV / COVARIANCE statements")?;
        let section = match config.tools.get("iivsearch") {
            Some(table) => table
                .clone()
                .try_into::<Section>()
                .map_err(|e| format!("[iivsearch]: {e}"))?,
            None => Section::default(),
        };
        let options = IivsearchOptions {
            algorithm: section.algorithm,
            correlation_algorithm: section.correlation_algorithm,
            as_fullblock: section.as_fullblock,
            block_retries: section.block_retries,
            starts: config.run.retries + 1,
            rank: match config.rank.kind {
                // Pharmpy: `bic` is `bic_iiv` for this tool.
                None | Some(RankType::Bic) => RankType::BicIiv,
                Some(other) => other,
            },
            cutoff: config.rank.cutoff,
        };
        options.validate()?;
        Ok(options)
    }

    /// Pharmpy's `validate_input`, plus the file's own consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.algorithm == Algorithm::Skip
            && matches!(
                self.correlation_algorithm,
                None | Some(CorrelationAlgorithm::Skip)
            )
        {
            return Err(
                "[iivsearch] algorithm = \"skip\" needs correlation_algorithm = \
                 \"top_down_exhaustive\"; with both skipped there is nothing to search"
                    .into(),
            );
        }
        if self.algorithm == Algorithm::SimultaneousStepwise && self.correlation_algorithm.is_some()
        {
            return Err(
                "[iivsearch] correlation_algorithm cannot be set with algorithm = \
                 \"simultaneous_stepwise\": that algorithm decides the block structure as \
                 it adds η, and has no second stage"
                    .into(),
            );
        }
        if self.as_fullblock && self.algorithm == Algorithm::SimultaneousStepwise {
            return Err(
                "[iivsearch] as_fullblock does not apply to simultaneous_stepwise, which \
                 tries the block structures itself"
                    .into(),
            );
        }
        if let Some(c) = self.cutoff {
            if !(c.is_finite() && c >= 0.0) {
                return Err(format!(
                    "[rank] cutoff = {c}: must be a finite, non-negative improvement on the \
                     criterion's own scale"
                ));
            }
        }
        if self.starts == 0 {
            return Err("[run] retries: the starts per candidate must be at least 1".into());
        }
        self.rank.criterion()?;
        Ok(())
    }

    /// Whether the block structure is searched after the number of η.
    pub fn block_stage(&self) -> bool {
        match self.algorithm {
            Algorithm::SimultaneousStepwise => false,
            _ => !matches!(self.correlation_algorithm, Some(CorrelationAlgorithm::Skip)),
        }
    }

    /// The runner criterion this ranks on.
    pub fn criterion(&self) -> Criterion {
        self.rank
            .criterion()
            .expect("validated: the rank type has a criterion")
    }

    /// The starts a candidate whose largest block has `block_size` η gets.
    pub fn starts_for(&self, block_size: usize) -> usize {
        self.starts + self.block_retries * block_size.saturating_sub(2)
    }
}

/// A variability structure over the model's parameters: which carry an η,
/// and which η are blocked together.
///
/// The parameters are the *searched* ones plus any other the input's η
/// fall on; order is alphabetical, Pharmpy's, so the candidate numbering
/// matches its `iivsearch_run{n}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IivStructure {
    /// The parameters carrying an η, alphabetical.
    pub etas: Vec<String>,
    /// The blocks, each two or more of `etas`, in the order they were
    /// declared or made; a parameter in no block is a diagonal η.
    pub blocks: Vec<Vec<String>>,
}

impl IivStructure {
    fn new(mut etas: Vec<String>, blocks: Vec<Vec<String>>) -> Self {
        etas.sort();
        etas.dedup();
        let blocks = blocks
            .into_iter()
            .map(|mut b| {
                b.retain(|p| etas.contains(p));
                b.sort();
                b
            })
            .filter(|b| b.len() > 1)
            .collect();
        IivStructure { etas, blocks }
    }

    /// Pharmpy's description: blocks first, then the diagonal η, each
    /// bracketed — `[CL,V]+[KA]`; the empty string for no η at all.
    pub fn description(&self) -> String {
        let mut parts: Vec<String> = self
            .blocks
            .iter()
            .map(|b| format!("[{}]", b.join(",")))
            .collect();
        for p in &self.etas {
            if !self.blocks.iter().any(|b| b.contains(p)) {
                parts.push(format!("[{p}]"));
            }
        }
        parts.join("+")
    }

    /// The size of the largest block; `1` when every η is diagonal.
    pub fn largest_block(&self) -> usize {
        self.blocks.iter().map(Vec::len).max().unwrap_or(1)
    }

    /// Whether the structure has a block *and* an η outside every block —
    /// the mixed ω whose structural zeros FOCE / FOCEI do not honour (#1018).
    pub fn is_partial_block(&self) -> bool {
        !self.blocks.is_empty()
            && self
                .etas
                .iter()
                .any(|p| !self.blocks.iter().any(|b| b.contains(p)))
    }

    /// The block a parameter is in, if any.
    pub fn block_of(&self, param: &str) -> Option<&[String]> {
        self.blocks
            .iter()
            .find(|b| b.iter().any(|p| p == param))
            .map(Vec::as_slice)
    }

    fn has_eta(&self, param: &str) -> bool {
        self.etas.iter().any(|p| p == param)
    }

    /// The search-space coordinates for the runner's table.
    pub fn feature_vector(&self) -> FeatureVector {
        FeatureVector::new().with("iiv", self.etas.join(",")).with(
            "blocks",
            self.blocks
                .iter()
                .map(|b| format!("[{}]", b.join(",")))
                .collect::<Vec<_>>()
                .join("+"),
        )
    }
}

/// What the search knows about the input before it fits anything.
#[derive(Debug)]
pub(crate) struct Space {
    pub input_model: ModelText,
    pub input_structure: IivStructure,
    /// The parameters an `IIV` statement names, alphabetical, with whether
    /// the η is kept (`IIV`) or searched (`IIV?`).
    pub iiv: Vec<(String, bool)>,
    /// The parameters a `COVARIANCE` statement names, alphabetical.
    pub cov: Vec<String>,
    /// The pairs every candidate keeps blocked (`COVARIANCE(IIV, …)`).
    pub forced_pairs: Vec<(String, String)>,
    /// The η name and initial variance a parameter's η takes when it is
    /// added: the input's own, or a fresh `ETA_<P>` at Pharmpy's 0.09.
    pub eta_of: BTreeMap<String, (String, f64)>,
    /// The base model is estimated by FOCE / FOCEI, whose outer optimizer
    /// estimates the full lower triangle of a mixed block + diagonal ω
    /// (#1018): a candidate with a block beside a standalone η is then
    /// fitted as a larger block than its description says, and the search
    /// says so on every such candidate.
    pub outer_full_triangle: bool,
    pub notes: Vec<String>,
}

/// Pharmpy's `add_iiv` default initial variance.
const NEW_ETA_VARIANCE: f64 = 0.09;

impl Space {
    fn from_config(config: &SearchConfig, base: &BaseModel) -> Result<Space, String> {
        let resolved = config.resolve_space(base)?;
        let mut space = Self::build(base.text.clone(), resolved.mfl.features(), resolved.notes)?;
        space.outer_full_triangle =
            base.prepared
                .parsed
                .fit_options
                .method_chain()
                .iter()
                .any(|m| {
                    matches!(
                        m,
                        ferx_core::EstimationMethod::Foce | ferx_core::EstimationMethod::FoceI
                    )
                });
        Ok(space)
    }

    /// The space from the input model and the *resolved* features — the
    /// seam the unit tests use.
    pub(crate) fn build<'f>(
        input_model: ModelText,
        features: impl Iterator<Item = &'f Feature>,
        mut notes: Vec<String>,
    ) -> Result<Space, String> {
        let variability = VariabilityText::read(&input_model)
            .map_err(|e| format!("iivsearch cannot read the base model's variability: {e}"))?;
        let mut iiv: BTreeMap<String, bool> = BTreeMap::new();
        let mut cov: BTreeSet<String> = BTreeSet::new();
        let mut forced_pairs: Vec<(String, String)> = Vec::new();
        let mut any = false;
        for f in features {
            match f {
                Feature::Iiv {
                    optional,
                    parameters,
                    effects,
                } => {
                    any = true;
                    if let Modes::List(list) = effects {
                        if let Some(e) = list.iter().find(|e| **e != VariabilityEffect::Exp) {
                            return Err(format!(
                                "iivsearch: `{f}` asks for the {} form; only the exponential \
                                 form (`P = TVP * exp(ETA_P)`) is searchable, since it is the \
                                 one the η edits can reverse",
                                e.label()
                            ));
                        }
                    }
                    for p in names_of(parameters, f)? {
                        // A kept η wins over a searched one for the same
                        // parameter (Pharmpy's forced feature).
                        let forced = !*optional;
                        iiv.entry(p)
                            .and_modify(|kept| *kept |= forced)
                            .or_insert(forced);
                    }
                }
                Feature::Covariance {
                    optional,
                    level,
                    parameters,
                } => {
                    let levels = level.expand();
                    if !levels.contains(&VariabilityLevel::Iiv) {
                        return Err(format!(
                            "iivsearch: `{f}` is an IOV covariance; iivsearch searches the η \
                             blocks (`COVARIANCE(IIV, …)`), and iovsearch the κ"
                        ));
                    }
                    if levels.contains(&VariabilityLevel::Iov) {
                        notes.push(format!(
                            "`{f}`: the IOV level is ignored by iivsearch (iovsearch's move)"
                        ));
                    }
                    any = true;
                    let names = names_of(parameters, f)?;
                    if names.len() < 2 {
                        return Err(format!(
                            "iivsearch: `{f}` names {} parameter{}; a covariance needs two",
                            names.len(),
                            if names.len() == 1 { "" } else { "s" }
                        ));
                    }
                    for p in &names {
                        cov.insert(p.clone());
                    }
                    if !*optional {
                        let mut sorted = names.clone();
                        sorted.sort();
                        for (i, a) in sorted.iter().enumerate() {
                            for b in sorted.iter().skip(i + 1) {
                                forced_pairs.push((a.clone(), b.clone()));
                            }
                        }
                    }
                }
                other => {
                    return Err(format!(
                        "iivsearch: `{other}` is not a variability feature; the space takes \
                         IIV / COVARIANCE statements only ({}search is the tool for it)",
                        match other {
                            Feature::Covariate { .. } | Feature::Allometry { .. } => "cov",
                            Feature::Iov { .. } => "iov",
                            _ => "model",
                        }
                    ))
                }
            }
        }
        if !any {
            return Err(
                "iivsearch: the space has no IIV or COVARIANCE statement, so there is no \
                 variability structure to search"
                    .into(),
            );
        }

        // Every named parameter must exist, and be written in the canonical
        // form the edits can rewrite — refused by name, never mis-edited.
        let mut eta_of = BTreeMap::new();
        for p in iiv.keys().chain(cov.iter()) {
            let Some(pv) = variability.parameter(p) else {
                return Err(format!(
                    "iivsearch: `{p}` is not an [individual_parameters] name of the base model"
                ));
            };
            if !pv.canonical {
                return Err(format!(
                    "iivsearch: `{p}` is not written in the canonical form `{p} = TVP * \
                     exp(ETA_{p})` (it carries {}), so its η cannot be searched by rewriting \
                     the line. Rewrite `{p}` as a product ending in `exp(<eta>)`, or leave it \
                     out of the space",
                    if pv.mentions.is_empty() {
                        "no random effect in a form an η can be appended to".to_string()
                    } else {
                        pv.mentions
                            .iter()
                            .map(|m| format!("`{m}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ));
            }
            let (name, variance) = match &pv.eta {
                Some(eta) => (
                    eta.clone(),
                    variability.variance(eta).unwrap_or(NEW_ETA_VARIANCE),
                ),
                None => (fresh_eta_name(&variability, p), NEW_ETA_VARIANCE),
            };
            eta_of.insert(p.clone(), (name, variance));
        }
        // Every *other* parameter that carries an η is named too, at the η
        // it already has. It is never added or dropped — the space does not
        // name it — but it can share a `block_omega` with one that is, and
        // splitting that block has to name every member. A lookup that
        // silently missed one would half-split the block.
        for pv in variability.with_eta() {
            let eta = pv.eta.clone().expect("with_eta");
            eta_of.entry(pv.name.clone()).or_insert((
                eta.clone(),
                variability.variance(&eta).unwrap_or(NEW_ETA_VARIANCE),
            ));
        }
        // A `FIX`ed η is a statement about the model: never removed, never
        // blocked (Pharmpy excludes fixed η from both moves).
        let mut fixed: Vec<String> = Vec::new();
        for (p, kept) in iiv.iter_mut() {
            if let Some(eta) = variability.parameter(p).and_then(|v| v.eta.clone()) {
                if variability.is_fixed(&eta) && !*kept {
                    *kept = true;
                    fixed.push(p.clone());
                }
            }
        }
        cov.retain(|p| {
            variability
                .parameter(p)
                .and_then(|v| v.eta.clone())
                .is_none_or(|eta| !variability.is_fixed(&eta))
        });
        if !fixed.is_empty() {
            notes.push(format!(
                "the η of {} {} declared `FIX` and {} kept rather than searched",
                fixed.join(", "),
                if fixed.len() == 1 { "is" } else { "are" },
                if fixed.len() == 1 { "is" } else { "are" }
            ));
        }
        for (a, b) in &forced_pairs {
            for p in [a, b] {
                if let Some(eta) = variability.parameter(p).and_then(|v| v.eta.clone()) {
                    if variability.is_fixed(&eta) {
                        return Err(format!(
                            "iivsearch: `COVARIANCE(IIV, …)` asks to block `{p}`, whose η \
                             `{eta}` is declared `FIX`; a block cannot mix fixed and free η"
                        ));
                    }
                }
            }
        }

        // The structure is read over every parameter, not only the space's:
        // the description lists every η the model carries, as Pharmpy's does.
        let input_structure =
            structure_of(&variability, variability.parameters.iter().map(|p| &p.name));
        Ok(Space {
            input_model,
            input_structure,
            iiv: iiv.into_iter().collect(),
            cov: cov.into_iter().collect(),
            forced_pairs,
            eta_of,
            outer_full_triangle: false,
            notes,
        })
    }

    /// The parameters whose η is searched, alphabetical.
    fn searched(&self) -> Vec<&str> {
        self.iiv
            .iter()
            .filter(|(_, kept)| !kept)
            .map(|(p, _)| p.as_str())
            .collect()
    }

    /// The parameters whose η is kept, alphabetical.
    fn kept(&self) -> Vec<&str> {
        self.iiv
            .iter()
            .filter(|(_, kept)| *kept)
            .map(|(p, _)| p.as_str())
            .collect()
    }

    /// The blocks every candidate carries: the connected components of the
    /// forced pairs among `etas`.
    fn forced_blocks(&self, etas: &[String]) -> Vec<Vec<String>> {
        components(
            self.forced_pairs
                .iter()
                .filter(|(a, b)| etas.contains(a) && etas.contains(b))
                .cloned(),
        )
    }
}

/// The explicit names a resolved operand carries.
fn names_of(operand: &Operand, feature: &Feature) -> Result<Vec<String>, String> {
    match operand {
        Operand::Names(names) => Ok(names.clone()),
        other => Err(format!(
            "iivsearch: `{feature}` still carries `{other}`; resolve the space against the \
             base model first"
        )),
    }
}

/// An η name no `[parameters]` declaration uses: `ETA_<P>`, then `ETA_<P>_2`, …
fn fresh_eta_name(v: &VariabilityText, param: &str) -> String {
    let taken: BTreeSet<&str> = v
        .omegas
        .iter()
        .map(|d| d.name.as_str())
        .chain(
            v.omega_blocks
                .iter()
                .flat_map(|b| b.names.iter().map(String::as_str)),
        )
        .chain(v.kappas.iter().map(|d| d.name.as_str()))
        .chain(
            v.kappa_blocks
                .iter()
                .flat_map(|b| b.names.iter().map(String::as_str)),
        )
        .collect();
    let base = format!("ETA_{param}");
    if !taken.contains(base.as_str()) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|n| !taken.contains(n.as_str()))
        .expect("an unused suffix exists")
}

/// The structure a model text has over `params`, read off its declarations.
fn structure_of<'a>(v: &VariabilityText, params: impl Iterator<Item = &'a String>) -> IivStructure {
    let params: BTreeSet<&String> = params.collect();
    let mut etas = Vec::new();
    let mut eta_to_param: HashMap<String, String> = HashMap::new();
    for p in &params {
        if let Some(eta) = v.parameter(p).and_then(|pv| pv.eta.clone()) {
            etas.push((*p).clone());
            eta_to_param.insert(eta, (*p).clone());
        }
    }
    let blocks: Vec<Vec<String>> = v
        .omega_blocks
        .iter()
        .map(|b| {
            b.names
                .iter()
                .filter_map(|n| eta_to_param.get(n).cloned())
                .collect::<Vec<_>>()
        })
        .filter(|b| b.len() > 1)
        .collect();
    IivStructure::new(etas, blocks)
}

/// Connected components of a set of pairs, each sorted, in order of first
/// appearance.
pub(crate) fn components(pairs: impl Iterator<Item = (String, String)>) -> Vec<Vec<String>> {
    let mut groups: Vec<Vec<String>> = Vec::new();
    for (a, b) in pairs {
        let ia = groups.iter().position(|g| g.contains(&a));
        let ib = groups.iter().position(|g| g.contains(&b));
        match (ia, ib) {
            (Some(i), Some(j)) if i == j => {}
            (Some(i), Some(j)) => {
                let (lo, hi) = (i.min(j), i.max(j));
                let moved = groups.remove(hi);
                groups[lo].extend(moved);
            }
            (Some(i), None) => groups[i].push(b),
            (None, Some(j)) => groups[j].push(a),
            (None, None) => groups.push(vec![a, b]),
        }
    }
    for g in &mut groups {
        g.sort();
        g.dedup();
    }
    groups
}

/// One fitted model of the search, as the table reports it.
#[derive(Debug, Clone)]
pub struct ModelRow {
    /// `input`, `base`, or `run{n}` in generation order — Pharmpy's
    /// `iivsearch_run{n}`.
    pub id: String,
    pub parent: Option<String>,
    /// The step the model was fitted in: `0` for the input and the base.
    pub step: usize,
    pub structure: IivStructure,
    pub ofv: Option<f64>,
    pub n_parameters: Option<usize>,
    /// The ranking criterion; `NaN` without a fit.
    pub criterion: f64,
    /// `criterion − the step's parent criterion`, when both exist.
    /// Negative is better.
    pub d_criterion: Option<f64>,
    /// The model's rank within its step, among the eligible models (the
    /// parent included); `None` when not eligible.
    pub rank: Option<usize>,
    pub converged: Option<bool>,
    /// Passed the strictness gate and has a fit.
    pub passed: bool,
    pub failures: Vec<String>,
    pub error: Option<CandidateError>,
    pub seconds: f64,
    /// The starts the candidate was fitted with.
    pub starts: usize,
    /// The model the search selected.
    pub selected: bool,
    /// The outcome came from the journal of an earlier run, not a fit.
    pub reused: bool,
}

impl ModelRow {
    /// Eligible for ranking: fitted, passed the gate, finite criterion.
    pub fn eligible(&self) -> bool {
        self.error.is_none() && self.passed && self.criterion.is_finite()
    }
}

/// What kind of step a [`StepSummary`] describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    /// The number of η (top-down, or one bottom-up step).
    NumberOfEtas,
    /// The block structure.
    BlockStructure,
    /// A simultaneous step: a new η and its block at once.
    Simultaneous,
    /// The final comparison with the input model.
    Input,
}

impl StepKind {
    pub fn label(&self) -> &'static str {
        match self {
            StepKind::NumberOfEtas => "no_of_etas",
            StepKind::BlockStructure => "block_structure",
            StepKind::Simultaneous => "simultaneous",
            StepKind::Input => "compare_to_input",
        }
    }
}

/// One step's ranking: the parent and the candidates it was compared with.
#[derive(Debug, Clone)]
pub struct StepSummary {
    pub step: usize,
    pub kind: StepKind,
    /// The model the candidates were compared with.
    pub parent: String,
    /// The ids ranked, best first — eligible models only, the parent
    /// included; the ineligible candidates follow in generation order.
    pub ranked: Vec<StepRank>,
    /// The step's winner: a candidate, or the parent when nothing beat it.
    pub best: String,
}

/// One model's place in a step.
#[derive(Debug, Clone)]
pub struct StepRank {
    pub id: String,
    pub criterion: f64,
    /// `parent criterion − this criterion`: positive is better.
    pub d_criterion: Option<f64>,
    pub rank: Option<usize>,
}

/// What a search reports as it runs, for a CLI progress line.
#[derive(Debug, Clone)]
pub enum IivsearchEvent {
    InputStarted,
    InputFinished {
        ofv: f64,
        criterion: f64,
    },
    /// A base had to be derived from the input and is being fitted.
    BaseStarted,
    BaseFinished {
        ofv: f64,
        criterion: f64,
    },
    StepStarted {
        step: usize,
        kind: StepKind,
        candidates: usize,
    },
    /// `best` is the step's winner and its criterion; `improved` whether it
    /// beat the parent.
    StepFinished {
        step: usize,
        best: (String, f64),
        improved: bool,
    },
    /// The final comparison returned the input.
    Reverted {
        criterion: f64,
    },
}

/// A progress callback.
pub type ProgressFn<'a> = &'a (dyn Fn(IivsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct IivsearchResult {
    pub options: IivsearchOptions,
    pub criterion: Criterion,
    /// The file's model, unedited.
    pub input_model: ModelText,
    pub input_structure: IivStructure,
    /// The root of the search: `input`, or `base` when one had to be
    /// derived.
    pub base_id: String,
    /// Every fitted model, in generation order.
    pub rows: Vec<ModelRow>,
    /// Every step's ranking, in order; the last is the comparison with the
    /// input.
    pub steps: Vec<StepSummary>,
    pub final_id: String,
    pub final_model: ModelText,
    pub final_structure: IivStructure,
    pub final_fit: Option<FitResult>,
    pub final_criterion: f64,
    /// Every fitted model's text, by id.
    pub models: BTreeMap<String, ModelText>,
    /// Things a user should read once.
    pub notes: Vec<String>,
    /// The search stopped on a cancel flag; `rows` is partial.
    pub cancelled: bool,
}

impl IivsearchResult {
    pub fn row(&self, id: &str) -> Option<&ModelRow> {
        self.rows.iter().find(|r| r.id == id)
    }
}

/// Everything a [`run_iivsearch`] call takes beyond the file.
#[derive(Default)]
pub struct IivsearchRun<'a> {
    /// Where the per-step journals, `models.csv`, `models/` and `final.ferx`
    /// go. `None` keeps everything in memory: no resume, no files.
    pub dir: Option<PathBuf>,
    /// Overrides `[run] threads`.
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Run a variability-structure search from a loaded `.ferxsearch` file and
/// its base model. Writes `models.csv`, `models/<id>.ferx` and `final.ferx`
/// into `run.dir` when given.
pub fn run_iivsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: IivsearchRun<'_>,
) -> Result<IivsearchResult, String> {
    let options = IivsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = options.criterion();
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

/// Where a search run's files go by default: `<config stem>-iivsearch` next
/// to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    crate::search::default_dir(config_path, "iivsearch")
}

/// A fitted model the search may move from.
#[derive(Debug, Clone)]
struct Node {
    id: String,
    model: ModelText,
    fit: Option<FitResult>,
    structure: IivStructure,
    criterion: f64,
    eligible: bool,
}

impl Node {
    fn from_result(
        result: &CandidateResult,
        report: &RunReport,
        model: ModelText,
        structure: IivStructure,
        criterion: Criterion,
    ) -> Result<Node, String> {
        let fit = resolve_fit(result, report)?;
        Ok(Node {
            id: result.id.clone(),
            criterion: match &fit {
                Some(f) => criterion.of(f),
                None => result.criterion,
            },
            eligible: result.eligible(),
            fit,
            model,
            structure,
        })
    }
}

/// The fit a result stands for, so a child can be seeded from it — a
/// duplicate's representative's, a failed candidate's none, and a resumed
/// row whose cached fit is gone an error naming the fix (the reasoning is
/// `modelsearch::resolve_fit`'s).
pub(crate) fn resolve_fit(
    result: &CandidateResult,
    report: &RunReport,
) -> Result<Option<FitResult>, String> {
    if let Some(fit) = &result.fit {
        return Ok(Some(fit.clone()));
    }
    if result.error.is_some() {
        return Ok(None);
    }
    if let Some(rep) = &result.duplicate_of {
        return report
            .results
            .iter()
            .find(|r| r.id == *rep)
            .and_then(|r| r.fit.clone())
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "{}: a duplicate of {rep}, whose fit is not available to seed from",
                    result.id
                )
            });
    }
    Err(format!(
        "{}: the fit is not in the journal cache (a resumed row whose `fits/<hash>.json` is \
         missing or unreadable), so nothing can be seeded from it; refit without `resume`",
        result.id
    ))
}

/// The candidate with `target`'s structure, derived from `parent` by the η
/// and block edits and seeded from the parent's estimates.
fn derive(
    id: &str,
    parent: &Node,
    target: &IivStructure,
    space: &Space,
    options: &IivsearchOptions,
) -> Result<Candidate, String> {
    let mut model = parent.model.clone();
    if let Some(fit) = &parent.fit {
        seed_from(&mut model, fit)?;
    }
    let from = &parent.structure;
    let err = |e: String| format!("{id}: {e}");
    // The η that go first: a block shrinks around its survivors, which is
    // then the block the text carries.
    for p in &from.etas {
        if !target.has_eta(p) {
            model
                .apply(ModelEdit::DropIiv { param: p.clone() })
                .map_err(err)?;
        }
    }
    // A surviving block the target does not carry comes apart, so its η can
    // be re-blocked from diagonal declarations; the target's blocks are made
    // at the end.
    let survived = IivStructure::new(
        from.etas
            .iter()
            .filter(|p| target.has_eta(p))
            .cloned()
            .collect(),
        from.blocks.clone(),
    );
    let mut to_split: Vec<String> = Vec::new();
    for block in &survived.blocks {
        if !target.blocks.contains(block) {
            for p in block {
                if let Some((eta, _)) = space.eta_of.get(p) {
                    to_split.push(eta.clone());
                }
            }
        }
    }
    if !to_split.is_empty() {
        model
            .apply(ModelEdit::SplitOmegaBlock(to_split))
            .map_err(err)?;
    }
    for p in &target.etas {
        if !from.has_eta(p) {
            let (eta, variance) = space
                .eta_of
                .get(p)
                .cloned()
                .ok_or_else(|| format!("{id}: `{p}` has no η to add"))?;
            model
                .apply(ModelEdit::AddIiv {
                    param: p.clone(),
                    form: IivForm::Exponential { eta, variance },
                })
                .map_err(err)?;
        }
    }
    let mut new_blocks: Vec<Vec<String>> = Vec::new();
    for block in &target.blocks {
        if survived.blocks.contains(block) {
            continue;
        }
        // Every member is named, or the block would be written around fewer
        // η than it says — a different model, silently. `Space::build` names
        // every parameter that carries an η, so this is a guard, not a path.
        let etas: Vec<String> = block
            .iter()
            .map(|p| {
                space
                    .eta_of
                    .get(p)
                    .map(|(e, _)| e.clone())
                    .ok_or_else(|| format!("{id}: `{p}` has no η to block"))
            })
            .collect::<Result<_, _>>()?;
        model
            .apply(ModelEdit::SetOmegaBlock(etas.clone()))
            .map_err(err)?;
        new_blocks.push(etas);
    }
    // A new block's correlations start from the parent's empirical Bayes
    // estimates rather than the edit's flat 0.1 — Pharmpy's
    // `create_joint_distribution(individual_estimates=…)`.
    if !new_blocks.is_empty() {
        if let Some(fit) = &parent.fit {
            if let Some(seeded) = block_seed(fit, &new_blocks) {
                seed_from(&mut model, &seeded)?;
            }
        }
    }
    Ok(Candidate::new(id, model)
        .parent(parent.id.clone())
        .features(target.feature_vector())
        .starts(options.starts_for(target.largest_block())))
}

/// The parent's fit with each new block's off-diagonals set from the
/// correlation of its subjects' η estimates — `None` when the fit has no
/// per-subject η to read (a journal-loaded fit, an evaluation with no
/// subjects), so the edit's own seed stands.
///
/// A correlation matrix read off shrunken EBEs can fail to be positive
/// definite once its diagonal is the fitted variances; the block's
/// correlations are then halved until it is, so the child always starts.
/// Only the *new* blocks are touched: an inherited block keeps the parent's
/// fitted covariances.
fn block_seed(fit: &FitResult, new_blocks: &[Vec<String>]) -> Option<FitResult> {
    if fit.subjects.len() < 3 {
        return None;
    }
    let index = |name: &str| fit.eta_names.iter().position(|n| n == name);
    let mut omega = fit.omega.clone();
    let mut changed = false;
    for block in new_blocks {
        let idx: Option<Vec<usize>> = block.iter().map(|n| index(n)).collect();
        let Some(idx) = idx else { continue };
        if idx.iter().any(|&k| k >= omega.nrows()) {
            continue;
        }
        let corr = ebe_correlation(fit, &idx)?;
        let mut shrink = 1.0;
        for _ in 0..8 {
            let mut sub = nalgebra::DMatrix::zeros(idx.len(), idx.len());
            for (a, &ia) in idx.iter().enumerate() {
                for (b, &ib) in idx.iter().enumerate() {
                    sub[(a, b)] = if a == b {
                        fit.omega[(ia, ia)]
                    } else {
                        shrink * corr[(a, b)] * (fit.omega[(ia, ia)] * fit.omega[(ib, ib)]).sqrt()
                    };
                }
            }
            if sub.iter().all(|v| v.is_finite()) && nalgebra::Cholesky::new(sub.clone()).is_some() {
                for (a, &ia) in idx.iter().enumerate() {
                    for (b, &ib) in idx.iter().enumerate() {
                        if a != b {
                            omega[(ia, ib)] = sub[(a, b)];
                        }
                    }
                }
                changed = true;
                break;
            }
            shrink *= 0.5;
        }
    }
    if !changed {
        return None;
    }
    let mut out = fit.clone();
    out.omega = omega;
    Some(out)
}

/// The sample correlation matrix of the subjects' η estimates at `idx`.
fn ebe_correlation(fit: &FitResult, idx: &[usize]) -> Option<nalgebra::DMatrix<f64>> {
    let n = fit.subjects.len() as f64;
    let k = idx.len();
    let mut mean = vec![0.0; k];
    for s in &fit.subjects {
        for (a, &ia) in idx.iter().enumerate() {
            mean[a] += *s.eta.get(ia)?;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut cov: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(k, k);
    for s in &fit.subjects {
        for (a, &ia) in idx.iter().enumerate() {
            for (b, &ib) in idx.iter().enumerate() {
                cov[(a, b)] += (s.eta[ia] - mean[a]) * (s.eta[ib] - mean[b]);
            }
        }
    }
    let mut corr: nalgebra::DMatrix<f64> = nalgebra::DMatrix::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            let d = (cov[(a, a)] * cov[(b, b)]).sqrt();
            corr[(a, b)] = if a == b {
                1.0
            } else if d > 0.0 && d.is_finite() {
                (cov[(a, b)] / d).clamp(-0.95, 0.95)
            } else {
                // A collapsed η has no correlation to read; a small seed keeps
                // the block estimable, as the edit's own default does.
                0.1
            };
        }
    }
    Some(corr)
}

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &IivsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<IivsearchResult, String> {
    let mut driver = Driver {
        fitter,
        space: &space,
        options,
        progress,
        criterion: options.criterion(),
        rows: Vec::new(),
        steps: Vec::new(),
        store: HashMap::new(),
        notes: space.notes.clone(),
        next_run: 0,
        next_step: 0,
        cancelled: false,
    };
    driver.run()
}

/// The state one search carries between its steps.
struct Driver<'a> {
    fitter: &'a dyn StepFitter,
    space: &'a Space,
    options: &'a IivsearchOptions,
    progress: Option<ProgressFn<'a>>,
    criterion: Criterion,
    rows: Vec<ModelRow>,
    steps: Vec<StepSummary>,
    /// Every model's text and fit, by id.
    store: HashMap<String, (ModelText, Option<FitResult>)>,
    notes: Vec<String>,
    next_run: usize,
    next_step: usize,
    cancelled: bool,
}

impl Driver<'_> {
    fn emit(&self, event: IivsearchEvent) {
        if let Some(p) = self.progress {
            p(event);
        }
    }

    fn new_id(&mut self) -> String {
        self.next_run += 1;
        format!("run{}", self.next_run)
    }

    fn push_note(&mut self, note: String) {
        if !self.notes.contains(&note) {
            self.notes.push(note);
        }
    }

    /// The whole search: input, base, the stages, the final comparison.
    fn run(&mut self) -> Result<IivsearchResult, String> {
        let space = self.space;
        // ── the input ───────────────────────────────────────────────────
        self.emit(IivsearchEvent::InputStarted);
        let candidate = Candidate::new("input", space.input_model.clone())
            .features(space.input_structure.feature_vector())
            .starts(
                self.options
                    .starts_for(space.input_structure.largest_block()),
            );
        let report = self
            .fitter
            .fit_step("input", std::slice::from_ref(&candidate))?;
        self.notes.extend(report.warnings.iter().cloned());
        let result = report
            .results
            .first()
            .ok_or("the input model was not fitted")?;
        if let Some(e) = &result.error {
            return Err(format!("the input model could not be fitted: {e}"));
        }
        let input = Node::from_result(
            result,
            &report,
            space.input_model.clone(),
            space.input_structure.clone(),
            self.criterion,
        )?;
        self.rows
            .push(self.row_of(result, 0, None, &input.structure, &candidate));
        self.store
            .insert(input.id.clone(), (input.model.clone(), input.fit.clone()));
        self.emit(IivsearchEvent::InputFinished {
            ofv: input.fit.as_ref().map(|f| f.ofv).unwrap_or(f64::NAN),
            criterion: input.criterion,
        });
        self.cancelled |= report.cancelled;

        // ── the base ────────────────────────────────────────────────────
        let base_structure = self.base_structure();
        let mut current = input.clone();
        if !self.cancelled && base_structure != input.structure {
            self.emit(IivsearchEvent::BaseStarted);
            let candidate = derive("base", &input, &base_structure, space, self.options)?;
            let report = self
                .fitter
                .fit_step("base", std::slice::from_ref(&candidate))?;
            self.notes.extend(report.warnings.iter().cloned());
            // A run cancelled *during* this one fit comes back with no result
            // at all: `Runner` drops a candidate whose `fit()` failed while
            // the flag was set, since that failure cannot be told from the
            // flag unwinding it. The input is already fitted, so the search
            // keeps it as the last completed model and returns a cancelled
            // result, rather than throwing the run away as an error (a CLI
            // can then still exit 130).
            match report.results.first() {
                None if report.cancelled => {
                    self.cancelled = true;
                    self.push_note(
                        "the search was cancelled while the base model was being fitted; the \
                         input is the last completed model"
                            .into(),
                    );
                }
                None => return Err("the base model was not fitted".into()),
                Some(result) => {
                    if let Some(e) = &result.error {
                        return Err(format!("the base model could not be fitted: {e}"));
                    }
                    let base = Node::from_result(
                        result,
                        &report,
                        candidate.model.clone(),
                        base_structure.clone(),
                        self.criterion,
                    )?;
                    self.rows.push(self.row_of(
                        result,
                        0,
                        Some("input"),
                        &base.structure,
                        &candidate,
                    ));
                    self.store
                        .insert(base.id.clone(), (base.model.clone(), base.fit.clone()));
                    self.emit(IivsearchEvent::BaseFinished {
                        ofv: base.fit.as_ref().map(|f| f.ofv).unwrap_or(f64::NAN),
                        criterion: base.criterion,
                    });
                    self.push_note(format!(
                        "the input model ({}) is not the search's base; the base is the input \
                         with the space's η structure ({})",
                        describe(&input.structure),
                        describe(&base.structure)
                    ));
                    self.cancelled |= report.cancelled;
                    current = base;
                }
            }
        }
        let base_id = current.id.clone();
        if !current.eligible {
            self.push_note(format!(
                "the base model ({}) fails the strictness gate; candidates are ranked among \
                 themselves",
                current.id
            ));
        }

        // ── the stages ──────────────────────────────────────────────────
        if !self.cancelled {
            current = match self.options.algorithm {
                Algorithm::TopDownExhaustive => self.top_down_etas(current)?,
                Algorithm::BottomUpStepwise => self.bottom_up_etas(current)?,
                Algorithm::SimultaneousStepwise => self.simultaneous(current)?,
                Algorithm::Skip => current,
            };
        }
        if !self.cancelled && self.options.block_stage() {
            current = self.top_down_blocks(current)?;
        }

        // ── the final comparison with the input ─────────────────────────
        let mut selected = current;
        if selected.id != input.id {
            let step = self.next_step + 1;
            self.next_step = step;
            let (ranked, best) = self.rank(&input, std::slice::from_ref(&selected));
            let reverted = best == input.id;
            self.steps.push(StepSummary {
                step,
                kind: StepKind::Input,
                parent: input.id.clone(),
                ranked,
                best: best.clone(),
            });
            if reverted {
                self.push_note(format!(
                    "the selected model {} ({}) does not rank better than the input on {}; \
                     the input is returned",
                    selected.id,
                    describe(&selected.structure),
                    self.criterion.label()
                ));
                self.emit(IivsearchEvent::Reverted {
                    criterion: input.criterion,
                });
                selected = input.clone();
            }
        }
        let final_id = selected.id.clone();
        if let Some(r) = self.rows.iter_mut().find(|r| r.id == final_id) {
            r.selected = true;
        }
        let models: BTreeMap<String, ModelText> = self
            .store
            .iter()
            .map(|(id, (text, _))| (id.clone(), text.clone()))
            .collect();
        let (final_model, final_fit) = self
            .store
            .remove(&final_id)
            .expect("every fitted model is stored");
        Ok(IivsearchResult {
            options: self.options.clone(),
            criterion: self.criterion,
            input_model: space.input_model.clone(),
            input_structure: space.input_structure.clone(),
            base_id,
            rows: std::mem::take(&mut self.rows),
            steps: std::mem::take(&mut self.steps),
            final_id,
            final_model,
            final_structure: selected.structure,
            final_fit,
            final_criterion: selected.criterion,
            models,
            notes: std::mem::take(&mut self.notes),
            cancelled: self.cancelled,
        })
    }

    /// The structure the search starts from: every space η for top-down,
    /// the kept η only for bottom-up, with the forced blocks; the input's
    /// own η outside the space, and its blocks inside it, as they are.
    fn base_structure(&self) -> IivStructure {
        let space = self.space;
        let input = &space.input_structure;
        let mut etas: Vec<String> = input.etas.clone();
        match self.options.algorithm {
            Algorithm::TopDownExhaustive | Algorithm::Skip => {
                for (p, _) in &space.iiv {
                    if !etas.contains(p) {
                        etas.push(p.clone());
                    }
                }
            }
            Algorithm::BottomUpStepwise | Algorithm::SimultaneousStepwise => {
                etas.retain(|p| !space.searched().contains(&p.as_str()));
                for p in space.kept() {
                    if !etas.iter().any(|e| e == p) {
                        etas.push(p.to_string());
                    }
                }
            }
        }
        etas.sort();
        // The input's blocks survive where their members do — Pharmpy keeps
        // a block the space allows — and the forced pairs are added.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for block in &input.blocks {
            for (i, a) in block.iter().enumerate() {
                for b in block.iter().skip(i + 1) {
                    if etas.contains(a) && etas.contains(b) {
                        pairs.push((a.clone(), b.clone()));
                    }
                }
            }
        }
        pairs.extend(
            space
                .forced_pairs
                .iter()
                .filter(|(a, b)| etas.contains(a) && etas.contains(b))
                .cloned(),
        );
        IivStructure::new(etas, components(pairs.into_iter()))
    }

    /// Fit one step's candidates and rank them against `parent`; returns
    /// the winner (the parent when nothing beat it).
    fn step(
        &mut self,
        kind: StepKind,
        parent: &Node,
        targets: Vec<IivStructure>,
    ) -> Result<Node, String> {
        if targets.is_empty() {
            return Ok(parent.clone());
        }
        let step = self.next_step + 1;
        self.next_step = step;
        if self.space.outer_full_triangle && targets.iter().any(IivStructure::is_partial_block) {
            self.push_note(
                "a candidate with a block beside a standalone η — a mixed block + diagonal ω — \
                 is estimated by FOCE / FOCEI with its cross-block covariances free (#1018): \
                 the model fitted is a larger block than the description says, and its \
                 parameter count and BIC do not include those covariances. Read such rows with \
                 that in mind, or estimate with saem / gn, which honour the declared structure"
                    .into(),
            );
        }
        let mut candidates = Vec::with_capacity(targets.len());
        for target in &targets {
            let id = self.new_id();
            candidates.push(derive(&id, parent, target, self.space, self.options)?);
        }
        self.emit(IivsearchEvent::StepStarted {
            step,
            kind,
            candidates: candidates.len(),
        });
        let dir = format!("step-{step}");
        let report = self.fitter.fit_step(&dir, &candidates)?;
        self.notes.extend(report.warnings.iter().cloned());
        let mut nodes: Vec<Node> = Vec::new();
        for (candidate, target) in candidates.iter().zip(&targets) {
            let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                continue; // cancelled before this candidate was reached
            };
            self.rows
                .push(self.row_of(result, step, Some(&parent.id), target, candidate));
            let node = Node::from_result(
                result,
                &report,
                candidate.model.clone(),
                target.clone(),
                self.criterion,
            )?;
            self.store
                .insert(node.id.clone(), (node.model.clone(), node.fit.clone()));
            nodes.push(node);
        }
        let (ranked, best) = self.rank(parent, &nodes);
        for r in &ranked {
            if let Some(row) = self
                .rows
                .iter_mut()
                .find(|row| row.id == r.id && row.step == step)
            {
                row.d_criterion = r.d_criterion.map(|d| -d);
                row.rank = r.rank;
            }
        }
        let winner = nodes
            .iter()
            .find(|n| n.id == best)
            .cloned()
            .unwrap_or_else(|| parent.clone());
        self.steps.push(StepSummary {
            step,
            kind,
            parent: parent.id.clone(),
            ranked,
            best: best.clone(),
        });
        self.emit(IivsearchEvent::StepFinished {
            step,
            best: (best.clone(), winner.criterion),
            improved: best != parent.id,
        });
        if report.cancelled {
            self.cancelled = true;
        }
        Ok(winner)
    }

    /// [`rank_models`] with the parent as reference.
    fn rank(&self, parent: &Node, candidates: &[Node]) -> (Vec<StepRank>, String) {
        let cands: Vec<(String, f64, bool)> = candidates
            .iter()
            .map(|n| (n.id.clone(), n.criterion, n.eligible))
            .collect();
        rank_models(
            (&parent.id, parent.criterion, parent.eligible),
            &cands,
            self.options.cutoff,
        )
    }

    /// `top_down_exhaustive` over the number of η: one candidate per subset
    /// of the searched η, largest first, lexicographic within a size; the
    /// parent's own structure skipped.
    fn top_down_etas(&mut self, parent: Node) -> Result<Node, String> {
        let searched: Vec<String> = self
            .space
            .searched()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut targets = Vec::new();
        for size in (0..=searched.len()).rev() {
            for combo in combinations(&searched, size) {
                let etas: Vec<String> = parent
                    .structure
                    .etas
                    .iter()
                    .filter(|p| !searched.contains(p) || combo.contains(p))
                    .cloned()
                    .collect();
                let target = self.restrict(&parent.structure, etas);
                if target != parent.structure {
                    targets.push(target);
                }
            }
        }
        self.step(StepKind::NumberOfEtas, &parent, targets)
    }

    /// `parent` with its η set replaced by `etas`: the surviving blocks
    /// shrink to their remaining members.
    fn restrict(&self, from: &IivStructure, etas: Vec<String>) -> IivStructure {
        let mut blocks: Vec<Vec<String>> = from.blocks.clone();
        blocks.extend(self.space.forced_blocks(&etas));
        IivStructure::new(etas, blocks)
    }

    /// `bottom_up_stepwise`: add one η at a time to the parameter that
    /// improves the parent most, until none does.
    fn bottom_up_etas(&mut self, mut parent: Node) -> Result<Node, String> {
        let searched: Vec<String> = self
            .space
            .searched()
            .iter()
            .map(|s| s.to_string())
            .collect();
        loop {
            let to_add: Vec<&String> = searched
                .iter()
                .filter(|p| !parent.structure.has_eta(p))
                .collect();
            if to_add.is_empty() || self.cancelled {
                return Ok(parent);
            }
            let mut targets = Vec::new();
            for p in to_add {
                let mut etas = parent.structure.etas.clone();
                etas.push(p.clone());
                let mut target = self.restrict(&parent.structure, etas);
                if self.options.as_fullblock {
                    target = self.full_block(target);
                }
                targets.push(target);
            }
            let best = self.step(StepKind::NumberOfEtas, &parent, targets)?;
            if best.id == parent.id {
                return Ok(parent);
            }
            parent = best;
        }
    }

    /// `as_fullblock`: every free η of the space in one block.
    fn full_block(&self, structure: IivStructure) -> IivStructure {
        let members: Vec<String> = structure
            .etas
            .iter()
            .filter(|p| self.space.iiv.iter().any(|(q, _)| q == *p))
            .cloned()
            .collect();
        if members.len() < 2 {
            return structure;
        }
        let mut blocks: Vec<Vec<String>> = structure
            .blocks
            .iter()
            .filter(|b| !b.iter().any(|p| members.contains(p)))
            .cloned()
            .collect();
        blocks.push(members);
        IivStructure::new(structure.etas, blocks)
    }

    /// `simultaneous_stepwise`: each new η tried diagonal, inside each
    /// existing block, and paired with each single η.
    fn simultaneous(&mut self, mut parent: Node) -> Result<Node, String> {
        let searched: Vec<String> = self
            .space
            .searched()
            .iter()
            .map(|s| s.to_string())
            .collect();
        loop {
            let to_add: Vec<String> = searched
                .iter()
                .filter(|p| !parent.structure.has_eta(p))
                .cloned()
                .collect();
            if to_add.is_empty() || self.cancelled {
                return Ok(parent);
            }
            let mut targets = Vec::new();
            for p in &to_add {
                let mut etas = parent.structure.etas.clone();
                etas.push(p.clone());
                let diagonal = self.restrict(&parent.structure, etas);
                targets.push(diagonal.clone());
                if !self.space.cov.contains(p) {
                    continue;
                }
                let mut singles: Vec<String> = diagonal
                    .etas
                    .iter()
                    .filter(|q| {
                        *q != p && self.space.cov.contains(q) && diagonal.block_of(q).is_none()
                    })
                    .cloned()
                    .collect();
                for block in &diagonal.blocks {
                    if !block.iter().all(|q| self.space.cov.contains(q)) {
                        continue;
                    }
                    let mut joined = block.clone();
                    joined.push(p.clone());
                    let mut blocks: Vec<Vec<String>> = diagonal
                        .blocks
                        .iter()
                        .filter(|b| *b != block)
                        .cloned()
                        .collect();
                    blocks.push(joined);
                    targets.push(IivStructure::new(diagonal.etas.clone(), blocks));
                    singles.retain(|q| !block.contains(q));
                }
                for q in singles {
                    let mut blocks = diagonal.blocks.clone();
                    blocks.push(vec![p.clone(), q]);
                    targets.push(IivStructure::new(diagonal.etas.clone(), blocks));
                }
            }
            let best = self.step(StepKind::Simultaneous, &parent, targets)?;
            if best.id == parent.id {
                return Ok(parent);
            }
            parent = best;
        }
    }

    /// `top_down_exhaustive` over the block structure: beside the forced
    /// blocks, the all-diagonal model and one full block per subset of two
    /// or more of the parameters the `COVARIANCE` statement names that carry
    /// an η; the parent's own structure skipped.
    fn top_down_blocks(&mut self, parent: Node) -> Result<Node, String> {
        // A block the parent carries that names a parameter the space does
        // not is **not** the search's to take apart — "a parameter the space
        // does not name is left as it is" — so it rides along in every
        // target, and its members are not offered to the cliques below.
        let untouchable: Vec<Vec<String>> = parent
            .structure
            .blocks
            .iter()
            .filter(|b| b.iter().any(|p| !self.space.cov.contains(p)))
            .cloned()
            .collect();
        if !untouchable.is_empty() {
            self.push_note(format!(
                "the block{} {} name{} a parameter the space's COVARIANCE statement does not, \
                 so {} carried unchanged into every candidate rather than restructured",
                if untouchable.len() == 1 { "" } else { "s" },
                untouchable
                    .iter()
                    .map(|b| format!("[{}]", b.join(",")))
                    .collect::<Vec<_>>()
                    .join(" and "),
                if untouchable.len() == 1 { "s" } else { "" },
                if untouchable.len() == 1 {
                    "it is"
                } else {
                    "they are"
                }
            ));
        }
        let members: Vec<String> = self
            .space
            .cov
            .iter()
            .filter(|p| parent.structure.has_eta(p) && !untouchable.iter().any(|b| b.contains(p)))
            .cloned()
            .collect();
        let forced = self.space.forced_blocks(&parent.structure.etas);
        let mut targets = Vec::new();
        let mut push = |blocks: Vec<Vec<String>>| {
            let blocks = untouchable.iter().cloned().chain(blocks).collect();
            let target = IivStructure::new(parent.structure.etas.clone(), blocks);
            if target != parent.structure && !targets.contains(&target) {
                targets.push(target);
            }
        };
        push(forced.clone());
        for size in 2..=members.len() {
            for combo in combinations(&members, size) {
                let mut pairs: Vec<(String, String)> = Vec::new();
                for (i, a) in combo.iter().enumerate() {
                    for b in combo.iter().skip(i + 1) {
                        pairs.push((a.clone(), b.clone()));
                    }
                }
                for block in &forced {
                    for (i, a) in block.iter().enumerate() {
                        for b in block.iter().skip(i + 1) {
                            pairs.push((a.clone(), b.clone()));
                        }
                    }
                }
                push(components(pairs.into_iter()));
            }
        }
        if members.len() < 2 && targets.is_empty() {
            self.push_note(
                "block structure not searched: fewer than two of the COVARIANCE parameters \
                 carry an η"
                    .into(),
            );
        }
        self.step(StepKind::BlockStructure, &parent, targets)
    }

    fn row_of(
        &self,
        result: &CandidateResult,
        step: usize,
        parent: Option<&str>,
        structure: &IivStructure,
        candidate: &Candidate,
    ) -> ModelRow {
        ModelRow {
            id: result.id.clone(),
            parent: parent.map(str::to_string),
            step,
            structure: structure.clone(),
            ofv: result.ofv,
            n_parameters: result.fit.as_ref().map(|f| f.n_parameters),
            criterion: match &result.fit {
                Some(fit) => self.criterion.of(fit),
                None => result.criterion,
            },
            d_criterion: None,
            rank: None,
            converged: result.converged,
            passed: result.verdict.passed && result.error.is_none(),
            failures: result.verdict.failures.clone(),
            error: result.error.clone(),
            seconds: result.seconds,
            starts: candidate.n_starts.unwrap_or(self.options.starts),
            selected: false,
            reused: result.reused,
        }
    }
}

/// Pharmpy's `rank_models` with `parent` as the reference model: the
/// eligible models by criterion, the parent first on a tie (a stable sort
/// from parent-first order), and a `cutoff` as the improvement over the
/// parent a candidate needs. Each model is `(id, criterion, eligible)`.
/// Returns every model's place — the eligible ones ranked, the rest after
/// them in the order given — and the winner's id: a candidate that beats
/// the parent, else the parent (which is the winner too when it fails the
/// gate and nothing passes).
pub(crate) fn rank_models(
    parent: (&str, f64, bool),
    candidates: &[(String, f64, bool)],
    cutoff: Option<f64>,
) -> (Vec<StepRank>, String) {
    let (parent_id, parent_criterion, parent_eligible) = parent;
    let cutoff = cutoff.unwrap_or(0.0);
    let all: Vec<(&str, f64, bool)> =
        std::iter::once((parent_id, parent_criterion, parent_eligible))
            .chain(candidates.iter().map(|(id, c, e)| (id.as_str(), *c, *e)))
            .collect();
    let mut eligible: Vec<(&str, f64)> = all
        .iter()
        .filter(|(_, c, e)| *e && c.is_finite())
        .map(|(id, c, _)| (*id, *c))
        .collect();
    eligible.sort_by(|a, b| a.1.total_cmp(&b.1));
    let delta = |c: f64| (parent_eligible && c.is_finite()).then_some(parent_criterion - c);
    let mut ranked: Vec<StepRank> = eligible
        .iter()
        .enumerate()
        .map(|(i, (id, c))| StepRank {
            id: id.to_string(),
            criterion: *c,
            d_criterion: delta(*c),
            rank: Some(i + 1),
        })
        .collect();
    for (id, c, e) in &all {
        if !(*e && c.is_finite()) {
            ranked.push(StepRank {
                id: id.to_string(),
                criterion: *c,
                d_criterion: delta(*c),
                rank: None,
            });
        }
    }
    let best = eligible
        .iter()
        .find(|(id, c)| {
            *id == parent_id
                || !parent_eligible
                || (*c < parent_criterion && parent_criterion - c >= cutoff)
        })
        .map(|(id, _)| id.to_string())
        .unwrap_or_else(|| parent_id.to_string());
    (ranked, best)
}

/// `[CL,V]+[KA]`, or `no η` for an empty structure.
pub fn describe(s: &IivStructure) -> String {
    let d = s.description();
    if d.is_empty() {
        "no η".to_string()
    } else {
        d
    }
}

/// Every `size`-subset of `items`, in lexicographic order of positions.
pub(crate) fn combinations<T: Clone>(items: &[T], size: usize) -> Vec<Vec<T>> {
    fn go<T: Clone>(
        items: &[T],
        size: usize,
        start: usize,
        cur: &mut Vec<T>,
        out: &mut Vec<Vec<T>>,
    ) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for i in start..items.len() {
            cur.push(items[i].clone());
            go(items, size, i + 1, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    if size <= items.len() {
        go(items, size, 0, &mut Vec::new(), &mut out);
    }
    out
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
