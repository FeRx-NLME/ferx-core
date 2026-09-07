//! Inter-occasion variability search — Pharmpy `iovsearch` (#1183).
//!
//! The κ counterpart of [`crate::iivsearch`]: which `[individual_parameters]`
//! carry an inter-occasion random effect, and which η become redundant once
//! they do. Its moves are [`ModelEdit::AddIov`] / [`ModelEdit::DropIov`]
//! and [`ModelEdit::DropIiv`], on parameters written in the canonical
//! `P = TVP * exp(ETA_P [+ KAPPA_P])` form — a parameter written any other
//! way is refused by name before anything is fitted.
//!
//! # The algorithm
//!
//! Pharmpy's (`tools/iovsearch/tool.py`), two steps of brute force:
//!
//! 1. **IOV.** The input is fitted, then a model with κ on *every* candidate
//!    parameter (`run1`, the full-IOV model), then one candidate per
//!    non-empty proper subset of those κ removed, each derived from the
//!    full-IOV model and seeded from its fit. The input, the full-IOV
//!    model and the candidates are ranked together on `[rank] type`; if the
//!    input ranks best the search ends on it.
//! 2. **IIV.** From the step-1 winner, one candidate per non-empty subset of
//!    the η whose parameter also carries a κ, with those η removed — an
//!    occasion-level random effect can make the subject-level one
//!    redundant. The winner and the candidates are ranked; the best is the
//!    final model.
//!
//! The criterion defaults to the **BIC(random)** — `OFV + n_parameters ·
//! ln(n_subjects)`, Pharmpy's `bic_random`, which is what `type = "bic"`
//! means for this tool; the `[strictness]` gate and `[rank] cutoff` apply
//! as in every search here.
//!
//! # The candidates
//!
//! Which parameters get a κ comes from the `.ferxsearch` `[space]`:
//! `IOV?([CL,V], EXP)` names them, a plain `IOV(CL, EXP)` names one every
//! candidate keeps. With no `[space]` the candidates are Pharmpy's default —
//! every parameter carrying a free η. A parameter that already carries a κ
//! is left as it is. A new κ starts at a tenth of the η's fitted variance
//! (Pharmpy's `add_iov`), and the κ are declared as `distribution` says:
//! `disjoint` (diagonal), `joint` (one `block_kappa`), `same-as-iiv` (the
//! default: κ blocked as their η are), or `explicit` with `groups`.
//!
//! The dataset's occasions come from the base model's `iov_column`, which
//! `prepare_run` reads at load time — so the base must declare it even
//! though it has no κ yet.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use ferx_core::edit::{ModelEdit, ModelText, VariabilityText};
use ferx_core::{CancelFlag, FitResult, Population};
use serde::Deserialize;

use crate::iivsearch::{combinations, components, rank_models, resolve_fit, StepRank, StepSummary};
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

/// `[iovsearch] distribution` — how the added κ are declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Distribution {
    /// One `kappa` line each.
    Disjoint,
    /// One `block_kappa` over every added κ.
    Joint,
    /// κ blocked as their η are: a `block_omega` over some η gives a
    /// `block_kappa` over their κ.
    #[default]
    SameAsIiv,
    /// The `groups` key says which κ are blocked together.
    Explicit,
}

impl Distribution {
    pub fn label(&self) -> &'static str {
        match self {
            Distribution::Disjoint => "disjoint",
            Distribution::Joint => "joint",
            Distribution::SameAsIiv => "same-as-iiv",
            Distribution::Explicit => "explicit",
        }
    }
}

/// The `[iovsearch]` section as written.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default)]
    column: Option<String>,
    #[serde(default)]
    distribution: Distribution,
    #[serde(default)]
    groups: Vec<Vec<String>>,
    #[serde(default = "default_block_retries")]
    block_retries: usize,
}

fn default_block_retries() -> usize {
    2
}

impl Default for Section {
    fn default() -> Self {
        Section {
            column: None,
            distribution: Distribution::default(),
            groups: Vec::new(),
            block_retries: default_block_retries(),
        }
    }
}

/// The `[iovsearch]` section of a `.ferxsearch` file, plus the `[rank]` and
/// `[run]` keys the tool reads. Every key has Pharmpy's default.
#[derive(Debug, Clone, PartialEq)]
pub struct IovsearchOptions {
    /// The occasion column. Checked against the base model's `iov_column`,
    /// which is what actually reads the occasions; `None` takes the base's.
    pub column: Option<String>,
    pub distribution: Distribution,
    /// The κ blocks for `distribution = "explicit"`, by parameter name.
    pub groups: Vec<Vec<String>>,
    /// Extra starts per κ beyond two in a candidate's largest `block_kappa`.
    pub block_retries: usize,
    /// The run's starts per candidate (`[run] retries + 1`).
    pub starts: usize,
    /// `[rank] type`; the BIC(random) when the file does not say, and what
    /// `bic` means for this tool.
    pub rank: RankType,
    /// `[rank] cutoff`: the improvement over the parent a candidate must
    /// show to replace it.
    pub cutoff: Option<f64>,
}

impl Default for IovsearchOptions {
    fn default() -> Self {
        IovsearchOptions {
            column: None,
            distribution: Distribution::default(),
            groups: Vec::new(),
            block_retries: default_block_retries(),
            starts: crate::search::RunOptions::default().n_starts,
            rank: RankType::BicRandom,
            cutoff: None,
        }
    }
}

impl IovsearchOptions {
    /// Read `[iovsearch]`, `[rank]` and `[run]` off a loaded file.
    pub fn from_config(config: &SearchConfig) -> Result<Self, String> {
        let section = match config.tools.get("iovsearch") {
            Some(table) => table
                .clone()
                .try_into::<Section>()
                .map_err(|e| format!("[iovsearch]: {e}"))?,
            None => Section::default(),
        };
        let options = IovsearchOptions {
            column: section.column,
            distribution: section.distribution,
            groups: section.groups,
            block_retries: section.block_retries,
            starts: config.run.retries + 1,
            rank: match config.rank.kind {
                // Pharmpy: `bic` is `bic_random` for this tool.
                None | Some(RankType::Bic) => RankType::BicRandom,
                Some(other) => other,
            },
            cutoff: config.rank.cutoff,
        };
        options.validate()?;
        Ok(options)
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.distribution {
            Distribution::Explicit if self.groups.is_empty() => {
                return Err(
                    "[iovsearch] distribution = \"explicit\" needs `groups`, the κ blocks as \
                     lists of parameter names (e.g. groups = [[\"CL\", \"V\"], [\"KA\"]])"
                        .into(),
                )
            }
            Distribution::Explicit => {}
            _ if !self.groups.is_empty() => {
                return Err(format!(
                    "[iovsearch] groups is read with distribution = \"explicit\" only (the \
                     file says \"{}\")",
                    self.distribution.label()
                ))
            }
            _ => {}
        }
        let mut seen: Vec<&String> = Vec::new();
        for g in &self.groups {
            if g.is_empty() {
                return Err("[iovsearch] groups: an empty group names no parameter".into());
            }
            for p in g {
                if seen.contains(&p) {
                    return Err(format!("[iovsearch] groups: `{p}` appears in two groups"));
                }
                seen.push(p);
            }
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

    /// The runner criterion this ranks on.
    pub fn criterion(&self) -> Criterion {
        self.rank
            .criterion()
            .expect("validated: the rank type has a criterion")
    }

    /// The starts a candidate whose largest κ block has `block_size` κ gets.
    pub fn starts_for(&self, block_size: usize) -> usize {
        self.starts + self.block_retries * block_size.saturating_sub(2)
    }
}

/// A variability structure over the searched parameters: which carry an η,
/// which carry a κ, and how each family is blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IovStructure {
    /// The parameters carrying an η, alphabetical.
    pub etas: Vec<String>,
    pub eta_blocks: Vec<Vec<String>>,
    /// The parameters carrying a κ, alphabetical.
    pub kappas: Vec<String>,
    pub kappa_blocks: Vec<Vec<String>>,
}

impl IovStructure {
    fn new(
        mut etas: Vec<String>,
        eta_blocks: Vec<Vec<String>>,
        mut kappas: Vec<String>,
        kappa_blocks: Vec<Vec<String>>,
    ) -> Self {
        etas.sort();
        etas.dedup();
        kappas.sort();
        kappas.dedup();
        let tidy = |blocks: Vec<Vec<String>>, members: &[String]| -> Vec<Vec<String>> {
            blocks
                .into_iter()
                .map(|mut b| {
                    b.retain(|p| members.contains(p));
                    b.sort();
                    b
                })
                .filter(|b| b.len() > 1)
                .collect()
        };
        let eta_blocks = tidy(eta_blocks, &etas);
        let kappa_blocks = tidy(kappa_blocks, &kappas);
        IovStructure {
            etas,
            eta_blocks,
            kappas,
            kappa_blocks,
        }
    }

    /// Pharmpy's description: `IIV([CL]+[V]+[KA]);IOV([CL])`.
    pub fn description(&self) -> String {
        format!(
            "IIV({});IOV({})",
            family(&self.etas, &self.eta_blocks),
            family(&self.kappas, &self.kappa_blocks)
        )
    }

    pub fn has_kappa(&self, param: &str) -> bool {
        self.kappas.iter().any(|p| p == param)
    }

    pub fn has_eta(&self, param: &str) -> bool {
        self.etas.iter().any(|p| p == param)
    }

    /// The size of the largest `block_kappa`; `1` when every κ is diagonal.
    pub fn largest_kappa_block(&self) -> usize {
        self.kappa_blocks.iter().map(Vec::len).max().unwrap_or(1)
    }

    /// The search-space coordinates for the runner's table.
    pub fn feature_vector(&self) -> FeatureVector {
        FeatureVector::new()
            .with("iiv", self.etas.join(","))
            .with("iov", self.kappas.join(","))
            .with(
                "iov_blocks",
                self.kappa_blocks
                    .iter()
                    .map(|b| format!("[{}]", b.join(",")))
                    .collect::<Vec<_>>()
                    .join("+"),
            )
    }
}

/// `[CL,V]+[KA]`: blocks first, then the diagonal members; `[]` when empty.
fn family(members: &[String], blocks: &[Vec<String>]) -> String {
    if members.is_empty() {
        return "[]".into();
    }
    let mut parts: Vec<String> = blocks
        .iter()
        .map(|b| format!("[{}]", b.join(",")))
        .collect();
    for p in members {
        if !blocks.iter().any(|b| b.contains(p)) {
            parts.push(format!("[{p}]"));
        }
    }
    parts.join("+")
}

/// What the search knows about the input before it fits anything.
#[derive(Debug)]
pub(crate) struct Space {
    pub input_model: ModelText,
    pub input_structure: IovStructure,
    /// The parameters a κ is tried on, alphabetical, with whether the κ is
    /// kept in every candidate (`IOV(p, EXP)`) or searched (`IOV?`).
    pub params: Vec<(String, bool)>,
    /// The κ blocks the candidates declare, by parameter (each of size ≥ 2).
    pub groups: Vec<Vec<String>>,
    /// Each parameter's η name (κ is added beside it) and κ name.
    pub names: BTreeMap<String, (Option<String>, String)>,
    /// The parameters whose η may be removed in step 2: those that carry a
    /// free η.
    pub free_eta: Vec<String>,
    pub notes: Vec<String>,
}

/// Pharmpy's `add_iov`: a new κ starts at a tenth of its η's variance.
const KAPPA_FRACTION: f64 = 0.1;

impl Space {
    fn from_config(
        config: &SearchConfig,
        base: &BaseModel,
        options: &IovsearchOptions,
    ) -> Result<Space, String> {
        let column = base.prepared.parsed.fit_options.iov_column.clone();
        let features: Vec<Feature> = if config.has_space() {
            config
                .resolve_space(base)?
                .mfl
                .features()
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        Self::build(
            base.text.clone(),
            &features,
            column.as_deref(),
            &base.prepared.population,
            options,
            Vec::new(),
        )
    }

    /// The space from its parts — the seam the unit tests use. `features`
    /// are the resolved `[space]` statements (empty for Pharmpy's default).
    pub(crate) fn build(
        input_model: ModelText,
        features: &[Feature],
        base_column: Option<&str>,
        population: &Population,
        options: &IovsearchOptions,
        mut notes: Vec<String>,
    ) -> Result<Space, String> {
        // The occasions are read by `prepare_run` from the base's own
        // `iov_column`; a base without one has a population without
        // occasions, and no candidate could be fitted.
        let Some(column) = base_column else {
            return Err(
                "iovsearch: the base model declares no `iov_column` in [fit_options], so the \
                 dataset's occasions were not read. Add `iov_column = OCC` (the occasion \
                 column) to the base model; it needs no `kappa` of its own"
                    .into(),
            );
        };
        if let Some(wanted) = &options.column {
            if !wanted.eq_ignore_ascii_case(column) {
                return Err(format!(
                    "[iovsearch] column = \"{wanted}\" but the base model reads its occasions \
                     from `iov_column = {column}`; the two must agree (or drop the key)"
                ));
            }
        }
        let mut levels: Vec<u32> = population
            .subjects
            .iter()
            .flat_map(|s| s.occasions.iter().copied())
            .collect();
        levels.sort_unstable();
        levels.dedup();
        if levels.len() < 2 {
            return Err(format!(
                "iovsearch: the `{column}` column has {} occasion value{}; inter-occasion \
                 variability needs at least two",
                levels.len(),
                if levels.len() == 1 { "" } else { "s" }
            ));
        }

        let variability = VariabilityText::read(&input_model)
            .map_err(|e| format!("iovsearch cannot read the base model's variability: {e}"))?;
        // The candidate parameters: the space's `IOV` statements, or every
        // parameter with a free η.
        let mut params: BTreeMap<String, bool> = BTreeMap::new();
        let mut stated_groups: Vec<Vec<String>> = Vec::new();
        for f in features {
            match f {
                Feature::Iov {
                    optional,
                    parameters,
                    effects,
                } => {
                    if let Modes::List(list) = effects {
                        if let Some(e) = list.iter().find(|e| **e != VariabilityEffect::Exp) {
                            return Err(format!(
                                "iovsearch: `{f}` asks for the {} form; only the exponential \
                                 form (`P = TVP * exp(ETA_P + KAPPA_P)`) is searchable",
                                e.label()
                            ));
                        }
                    }
                    for p in names_of(parameters, f)? {
                        let forced = !*optional;
                        params
                            .entry(p)
                            .and_modify(|kept| *kept |= forced)
                            .or_insert(forced);
                    }
                }
                Feature::Covariance {
                    optional,
                    level,
                    parameters,
                } => {
                    if *optional {
                        return Err(format!(
                            "iovsearch: `{f}` — the κ block structure is not searched; state it \
                             with [iovsearch] distribution (or a plain `COVARIANCE(IOV, [...])` \
                             for an explicit block)"
                        ));
                    }
                    if !level.expand().contains(&VariabilityLevel::Iov) {
                        return Err(format!(
                            "iovsearch: `{f}` is an η covariance; iivsearch searches the η \
                             blocks, iovsearch the κ (`COVARIANCE(IOV, …)`)"
                        ));
                    }
                    stated_groups.push(names_of(parameters, f)?);
                }
                other => {
                    return Err(format!(
                        "iovsearch: `{other}` is not an IOV feature; the space takes IOV \
                         statements (and `COVARIANCE(IOV, …)` for an explicit block)"
                    ))
                }
            }
        }
        let from_space = !params.is_empty();
        if !from_space {
            for p in variability.with_eta() {
                let eta = p.eta.as_deref().expect("with_eta");
                if !variability.is_fixed(eta) {
                    params.insert(p.name.clone(), false);
                }
            }
            if params.is_empty() {
                return Err(
                    "iovsearch: the base model has no parameter with a free η to add \
                     inter-occasion variability to, and the file names none (`[space] mfl = \
                     \"IOV?([CL,V], EXP)\"`)"
                        .into(),
                );
            }
        }
        // A parameter that already carries a κ is left as it is.
        let already: Vec<String> = params
            .keys()
            .filter(|p| variability.parameter(p).is_some_and(|v| v.kappa.is_some()))
            .cloned()
            .collect();
        for p in &already {
            params.remove(p);
        }
        if !already.is_empty() {
            notes.push(format!(
                "{} already {} a κ in the base model and {} left as {}",
                already.join(", "),
                if already.len() == 1 {
                    "carries"
                } else {
                    "carry"
                },
                if already.len() == 1 { "is" } else { "are" },
                if already.len() == 1 {
                    "it is"
                } else {
                    "they are"
                }
            ));
        }
        if params.is_empty() {
            return Err(
                "iovsearch: every candidate parameter already carries a κ; there is nothing \
                 to add"
                    .into(),
            );
        }
        let mut names = BTreeMap::new();
        for p in params.keys() {
            let Some(pv) = variability.parameter(p) else {
                return Err(format!(
                    "iovsearch: `{p}` is not an [individual_parameters] name of the base model"
                ));
            };
            if !pv.canonical {
                return Err(format!(
                    "iovsearch: `{p}` is not written in the canonical form `{p} = TVP * \
                     exp(ETA_{p})` (it carries {}), so a κ cannot be added by rewriting the \
                     line. Rewrite `{p}` as a product ending in `exp(<eta>)`, or leave it out \
                     of the space",
                    if pv.mentions.is_empty() {
                        "no random effect in a form a κ can be appended to".to_string()
                    } else {
                        pv.mentions
                            .iter()
                            .map(|m| format!("`{m}`"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ));
            }
            names.insert(
                p.clone(),
                (pv.eta.clone(), fresh_kappa_name(&variability, p)),
            );
        }
        // The κ blocks the candidates declare.
        let member = |p: &String| params.contains_key(p);
        let groups: Vec<Vec<String>> = match options.distribution {
            Distribution::Disjoint => Vec::new(),
            Distribution::Joint => vec![params.keys().cloned().collect()],
            Distribution::SameAsIiv => variability
                .omega_blocks
                .iter()
                .map(|b| {
                    let mut g: Vec<String> = b
                        .names
                        .iter()
                        .filter_map(|eta| {
                            variability
                                .parameters
                                .iter()
                                .find(|pv| pv.eta.as_deref() == Some(eta))
                                .map(|pv| pv.name.clone())
                        })
                        .filter(member)
                        .collect();
                    g.sort();
                    g
                })
                .collect(),
            Distribution::Explicit => {
                for g in &options.groups {
                    for p in g {
                        if !member(p) {
                            return Err(format!(
                                "[iovsearch] groups names `{p}`, which is not a candidate \
                                 parameter (the candidates are {})",
                                params.keys().cloned().collect::<Vec<_>>().join(", ")
                            ));
                        }
                    }
                }
                options.groups.clone()
            }
        };
        let mut groups: Vec<Vec<String>> = groups
            .into_iter()
            .chain(stated_groups)
            .map(|mut g| {
                g.retain(member);
                g.sort();
                g.dedup();
                g
            })
            .filter(|g| g.len() > 1)
            .collect();
        // Two stated blocks sharing a member are one block.
        let pairs: Vec<(String, String)> = groups
            .iter()
            .flat_map(|g| {
                let g = g.clone();
                (0..g.len()).flat_map(move |i| {
                    let g = g.clone();
                    (i + 1..g.len()).map(move |j| (g[i].clone(), g[j].clone()))
                })
            })
            .collect();
        groups = components(pairs.into_iter());

        let free_eta: Vec<String> = params
            .keys()
            .filter(|p| {
                names[*p]
                    .0
                    .as_deref()
                    .is_some_and(|eta| !variability.is_fixed(eta))
            })
            .cloned()
            .collect();
        // The structure is read over every parameter, not only the
        // candidates: the description lists every η the model carries.
        let input_structure =
            structure_of(&variability, variability.parameters.iter().map(|p| &p.name));
        Ok(Space {
            input_model,
            input_structure,
            params: params.into_iter().collect(),
            groups,
            names,
            free_eta,
            notes,
        })
    }

    /// The κ every candidate keeps, alphabetical.
    fn kept(&self) -> Vec<String> {
        self.params
            .iter()
            .filter(|(_, kept)| *kept)
            .map(|(p, _)| p.clone())
            .collect()
    }

    /// The κ the search may remove, alphabetical.
    fn searched(&self) -> Vec<String> {
        self.params
            .iter()
            .filter(|(_, kept)| !kept)
            .map(|(p, _)| p.clone())
            .collect()
    }
}

/// The explicit names a resolved operand carries.
fn names_of(operand: &Operand, feature: &Feature) -> Result<Vec<String>, String> {
    match operand {
        Operand::Names(names) => Ok(names.clone()),
        other => Err(format!(
            "iovsearch: `{feature}` still carries `{other}`; resolve the space against the \
             base model first"
        )),
    }
}

/// A κ name no `[parameters]` declaration uses: `KAPPA_<P>`, then
/// `KAPPA_<P>_2`, …
fn fresh_kappa_name(v: &VariabilityText, param: &str) -> String {
    let taken: Vec<&str> = v
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
    let base = format!("KAPPA_{param}");
    if !taken.contains(&base.as_str()) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}_{n}"))
        .find(|n| !taken.contains(&n.as_str()))
        .expect("an unused suffix exists")
}

/// The structure a model text has over `params`, read off its declarations.
fn structure_of<'a>(v: &VariabilityText, params: impl Iterator<Item = &'a String>) -> IovStructure {
    let params: Vec<&String> = params.collect();
    let mut etas = Vec::new();
    let mut kappas = Vec::new();
    let mut eta_to_param: HashMap<String, String> = HashMap::new();
    let mut kappa_to_param: HashMap<String, String> = HashMap::new();
    for p in &params {
        if let Some(pv) = v.parameter(p) {
            if let Some(eta) = &pv.eta {
                etas.push((*p).clone());
                eta_to_param.insert(eta.clone(), (*p).clone());
            }
            if let Some(kappa) = &pv.kappa {
                kappas.push((*p).clone());
                kappa_to_param.insert(kappa.clone(), (*p).clone());
            }
        }
    }
    let map_blocks = |blocks: &[ferx_core::edit::RandomEffectBlock],
                      to: &HashMap<String, String>| {
        blocks
            .iter()
            .map(|b| {
                b.names
                    .iter()
                    .filter_map(|n| to.get(n).cloned())
                    .collect::<Vec<_>>()
            })
            .filter(|b| b.len() > 1)
            .collect::<Vec<_>>()
    };
    IovStructure::new(
        etas,
        map_blocks(&v.omega_blocks, &eta_to_param),
        kappas,
        map_blocks(&v.kappa_blocks, &kappa_to_param),
    )
}

/// One fitted model of the search, as the table reports it.
#[derive(Debug, Clone)]
pub struct ModelRow {
    /// `input`, or `run{n}` in generation order — Pharmpy's
    /// `iovsearch_run{n}`, `run1` being the full-IOV model.
    pub id: String,
    pub parent: Option<String>,
    /// `0` for the input, `1` for the IOV step, `2` for the IIV step.
    pub step: usize,
    pub structure: IovStructure,
    pub ofv: Option<f64>,
    pub n_parameters: Option<usize>,
    /// The ranking criterion; `NaN` without a fit.
    pub criterion: f64,
    /// `criterion − the step's parent criterion`, when both exist.
    pub d_criterion: Option<f64>,
    /// The model's rank within its step, among the eligible models.
    pub rank: Option<usize>,
    pub converged: Option<bool>,
    pub passed: bool,
    pub failures: Vec<String>,
    pub error: Option<CandidateError>,
    pub seconds: f64,
    pub starts: usize,
    pub selected: bool,
    pub reused: bool,
}

impl ModelRow {
    pub fn eligible(&self) -> bool {
        self.error.is_none() && self.passed && self.criterion.is_finite()
    }
}

/// What a search reports as it runs.
#[derive(Debug, Clone)]
pub enum IovsearchEvent {
    InputStarted,
    InputFinished {
        ofv: f64,
        criterion: f64,
    },
    /// The full-IOV model is being fitted.
    FullIovStarted {
        parameters: usize,
    },
    FullIovFinished {
        ofv: f64,
        criterion: f64,
    },
    StepStarted {
        step: usize,
        candidates: usize,
    },
    StepFinished {
        step: usize,
        best: (String, f64),
        improved: bool,
    },
}

/// A progress callback.
pub type ProgressFn<'a> = &'a (dyn Fn(IovsearchEvent) + Send + Sync);

/// The outcome of a search.
#[derive(Debug, Clone)]
pub struct IovsearchResult {
    pub options: IovsearchOptions,
    pub criterion: Criterion,
    pub input_model: ModelText,
    pub input_structure: IovStructure,
    /// Every fitted model, in generation order.
    pub rows: Vec<ModelRow>,
    /// The two steps' rankings.
    pub steps: Vec<StepSummary>,
    pub final_id: String,
    pub final_model: ModelText,
    pub final_structure: IovStructure,
    pub final_fit: Option<FitResult>,
    pub final_criterion: f64,
    pub models: BTreeMap<String, ModelText>,
    pub notes: Vec<String>,
    pub cancelled: bool,
}

impl IovsearchResult {
    pub fn row(&self, id: &str) -> Option<&ModelRow> {
        self.rows.iter().find(|r| r.id == id)
    }
}

/// Everything a [`run_iovsearch`] call takes beyond the file.
#[derive(Default)]
pub struct IovsearchRun<'a> {
    pub dir: Option<PathBuf>,
    pub threads: Option<usize>,
    pub cancel: Option<CancelFlag>,
    pub progress: Option<ProgressFn<'a>>,
}

/// Run an inter-occasion variability search from a loaded `.ferxsearch`
/// file and its base model. Writes `models.csv`, `models/<id>.ferx` and
/// `final.ferx` into `run.dir` when given.
pub fn run_iovsearch(
    config: &SearchConfig,
    base: &BaseModel,
    run: IovsearchRun<'_>,
) -> Result<IovsearchResult, String> {
    let options = IovsearchOptions::from_config(config)?;
    let mut run_options = config.run_options();
    run_options.criterion = options.criterion();
    let fitter = RunnerFitter {
        threads: run.threads.or(config.run.threads).unwrap_or(0),
        dir: run.dir.clone(),
        cancel: run.cancel.clone(),
        data: &base.prepared.population,
        options: run_options,
    };
    let space = Space::from_config(config, base, &options)?;
    let result = search(&fitter, space, &options, run.progress)?;
    if let Some(dir) = &run.dir {
        write_report(dir, &result)?;
    }
    Ok(result)
}

/// Where a search run's files go by default: `<config stem>-iovsearch` next
/// to the config file.
pub fn default_dir(config_path: &Path) -> PathBuf {
    crate::search::default_dir(config_path, "iovsearch")
}

/// A fitted model the search may move from.
#[derive(Debug, Clone)]
struct Node {
    id: String,
    model: ModelText,
    fit: Option<FitResult>,
    structure: IovStructure,
    criterion: f64,
    eligible: bool,
}

impl Node {
    fn from_result(
        result: &CandidateResult,
        report: &RunReport,
        model: ModelText,
        structure: IovStructure,
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

/// The candidate with `target`'s structure, derived from `parent` by the κ
/// and η edits and seeded from the parent's estimates.
fn derive(
    id: &str,
    parent: &Node,
    target: &IovStructure,
    space: &Space,
    options: &IovsearchOptions,
) -> Result<Candidate, String> {
    let mut model = parent.model.clone();
    if let Some(fit) = &parent.fit {
        seed_from(&mut model, fit)?;
    }
    let from = &parent.structure;
    let err = |e: String| format!("{id}: {e}");
    // κ that go: their block shrinks around the survivors.
    for p in &from.kappas {
        if !target.has_kappa(p) {
            model
                .apply(ModelEdit::DropIov { param: p.clone() })
                .map_err(err)?;
        }
    }
    // A surviving κ block the target does not carry comes apart.
    let survived = IovStructure::new(
        from.etas.clone(),
        from.eta_blocks.clone(),
        from.kappas
            .iter()
            .filter(|p| target.has_kappa(p))
            .cloned()
            .collect(),
        from.kappa_blocks.clone(),
    );
    let mut to_split: Vec<String> = Vec::new();
    for block in &survived.kappa_blocks {
        if !target.kappa_blocks.contains(block) {
            for p in block {
                if let Some((_, kappa)) = space.names.get(p) {
                    to_split.push(kappa.clone());
                }
            }
        }
    }
    if !to_split.is_empty() {
        model
            .apply(ModelEdit::SplitKappaBlock(to_split))
            .map_err(err)?;
    }
    // η that go — the κ stays in the `exp(…)`.
    for p in &from.etas {
        if !target.has_eta(p) {
            model
                .apply(ModelEdit::DropIiv { param: p.clone() })
                .map_err(err)?;
        }
    }
    // κ that come, at a tenth of the η's variance as the parent fitted it.
    for p in &target.kappas {
        if from.has_kappa(p) {
            continue;
        }
        let (eta, kappa) = space
            .names
            .get(p)
            .cloned()
            .ok_or_else(|| format!("{id}: `{p}` has no κ to add"))?;
        let variance = eta_variance(parent, eta.as_deref(), &model) * KAPPA_FRACTION;
        model
            .apply(ModelEdit::AddIov {
                param: p.clone(),
                kappa,
                variance,
            })
            .map_err(err)?;
    }
    for block in &target.kappa_blocks {
        if survived.kappa_blocks.contains(block) {
            continue;
        }
        let kappas: Vec<String> = block
            .iter()
            .filter_map(|p| space.names.get(p).map(|(_, k)| k.clone()))
            .collect();
        model.apply(ModelEdit::SetKappaBlock(kappas)).map_err(err)?;
    }
    Ok(Candidate::new(id, model)
        .parent(parent.id.clone())
        .features(target.feature_vector())
        .starts(options.starts_for(target.largest_kappa_block())))
}

/// The variance of `eta` the parent fitted (its declared init when the fit
/// is not at hand), or Pharmpy's 0.09 for a parameter with no η.
fn eta_variance(parent: &Node, eta: Option<&str>, model: &ModelText) -> f64 {
    let Some(eta) = eta else {
        return 0.09;
    };
    if let Some(fit) = &parent.fit {
        if let Some(k) = fit.eta_names.iter().position(|n| n == eta) {
            let v = fit.omega[(k, k)];
            if v.is_finite() && v > 0.0 {
                return v;
            }
        }
    }
    VariabilityText::read(model)
        .ok()
        .and_then(|v| v.variance(eta))
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(0.09)
}

/// The search proper, over an injected fitter.
pub(crate) fn search(
    fitter: &dyn StepFitter,
    space: Space,
    options: &IovsearchOptions,
    progress: Option<ProgressFn<'_>>,
) -> Result<IovsearchResult, String> {
    let emit = |event: IovsearchEvent| {
        if let Some(p) = progress {
            p(event);
        }
    };
    let criterion = options.criterion();
    let mut notes = space.notes.clone();
    let mut rows: Vec<ModelRow> = Vec::new();
    let mut steps: Vec<StepSummary> = Vec::new();
    let mut store: HashMap<String, (ModelText, Option<FitResult>)> = HashMap::new();
    let mut cancelled = false;
    let mut next_run = 0usize;
    let mut new_id = || {
        next_run += 1;
        format!("run{next_run}")
    };

    // ── the input ───────────────────────────────────────────────────────
    emit(IovsearchEvent::InputStarted);
    let candidate = Candidate::new("input", space.input_model.clone())
        .features(space.input_structure.feature_vector())
        .starts(options.starts_for(space.input_structure.largest_kappa_block()));
    let report = fitter.fit_step("input", std::slice::from_ref(&candidate))?;
    notes.extend(report.warnings.iter().cloned());
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
        criterion,
    )?;
    rows.push(row_of(
        result,
        0,
        None,
        &input.structure,
        &candidate,
        options,
        criterion,
    ));
    store.insert(input.id.clone(), (input.model.clone(), input.fit.clone()));
    emit(IovsearchEvent::InputFinished {
        ofv: input.fit.as_ref().map(|f| f.ofv).unwrap_or(f64::NAN),
        criterion: input.criterion,
    });
    cancelled |= report.cancelled;
    if !input.eligible {
        notes.push(
            "the input model fails the strictness gate; candidates are ranked among \
             themselves"
                .into(),
        );
    }

    // ── step 1: the full-IOV model, then every proper subset removed ─────
    let mut selected = input.clone();
    if !cancelled {
        let all: Vec<String> = space.params.iter().map(|(p, _)| p.clone()).collect();
        // A κ the input already carries is not a candidate: it stays in every
        // model, block and all, beside the ones the search adds.
        let existing: Vec<String> = input.structure.kappas.clone();
        let with_existing = |kappas: Vec<String>| -> Vec<String> {
            existing.iter().cloned().chain(kappas).collect()
        };
        let kappa_blocks: Vec<Vec<String>> = input
            .structure
            .kappa_blocks
            .iter()
            .cloned()
            .chain(space.groups.iter().cloned())
            .collect();
        let full = IovStructure::new(
            input.structure.etas.clone(),
            input.structure.eta_blocks.clone(),
            with_existing(all.clone()),
            kappa_blocks.clone(),
        );
        emit(IovsearchEvent::FullIovStarted {
            parameters: all.len(),
        });
        let id = new_id();
        let candidate = derive(&id, &input, &full, &space, options)?;
        let report = fitter.fit_step("iov-all", std::slice::from_ref(&candidate))?;
        notes.extend(report.warnings.iter().cloned());
        let result = report
            .results
            .first()
            .ok_or("the full-IOV model was not fitted")?;
        if let Some(e) = &result.error {
            return Err(format!("the full-IOV model could not be fitted: {e}"));
        }
        let full_node = Node::from_result(
            result,
            &report,
            candidate.model.clone(),
            full.clone(),
            criterion,
        )?;
        rows.push(row_of(
            result,
            1,
            Some("input"),
            &full,
            &candidate,
            options,
            criterion,
        ));
        store.insert(
            full_node.id.clone(),
            (full_node.model.clone(), full_node.fit.clone()),
        );
        emit(IovsearchEvent::FullIovFinished {
            ofv: full_node.fit.as_ref().map(|f| f.ofv).unwrap_or(f64::NAN),
            criterion: full_node.criterion,
        });
        cancelled |= report.cancelled;

        let mut nodes = vec![full_node.clone()];
        if !cancelled {
            // Every non-empty proper subset of the searched κ removed,
            // smallest removals first, from the full-IOV model.
            let searched = space.searched();
            let kept = space.kept();
            let mut targets = Vec::new();
            for size in 1..searched.len() {
                for removed in combinations(&searched, size) {
                    let kappas: Vec<String> = all
                        .iter()
                        .filter(|p| kept.contains(p) || !removed.contains(p))
                        .cloned()
                        .collect();
                    targets.push(IovStructure::new(
                        full.etas.clone(),
                        full.eta_blocks.clone(),
                        with_existing(kappas),
                        kappa_blocks.clone(),
                    ));
                }
            }
            // With one searched κ there is no proper subset to remove; the
            // full model is the only candidate.
            let mut candidates = Vec::with_capacity(targets.len());
            for target in &targets {
                let id = new_id();
                candidates.push(derive(&id, &full_node, target, &space, options)?);
            }
            emit(IovsearchEvent::StepStarted {
                step: 1,
                candidates: candidates.len() + 1,
            });
            if !candidates.is_empty() {
                let report = fitter.fit_step("step-1", &candidates)?;
                notes.extend(report.warnings.iter().cloned());
                for (candidate, target) in candidates.iter().zip(&targets) {
                    let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                        continue;
                    };
                    rows.push(row_of(
                        result,
                        1,
                        Some(&full_node.id),
                        target,
                        candidate,
                        options,
                        criterion,
                    ));
                    let node = Node::from_result(
                        result,
                        &report,
                        candidate.model.clone(),
                        target.clone(),
                        criterion,
                    )?;
                    store.insert(node.id.clone(), (node.model.clone(), node.fit.clone()));
                    nodes.push(node);
                }
                cancelled |= report.cancelled;
            }
        }
        let (ranked, best) = rank_nodes(&input, &nodes, options.cutoff);
        annotate(&mut rows, 1, &ranked);
        let winner = nodes
            .iter()
            .find(|n| n.id == best)
            .cloned()
            .unwrap_or_else(|| input.clone());
        emit(IovsearchEvent::StepFinished {
            step: 1,
            best: (best.clone(), winner.criterion),
            improved: best != input.id,
        });
        steps.push(StepSummary {
            step: 1,
            kind: crate::iivsearch::StepKind::NumberOfEtas,
            parent: input.id.clone(),
            ranked,
            best: best.clone(),
        });
        if best == input.id {
            notes
                .push("no IOV candidate ranks better than the input; the input is returned".into());
        }
        selected = winner;
    }

    // ── step 2: the η with a κ beside them, every subset removed ─────────
    if !cancelled && selected.id != input.id {
        let removable: Vec<String> = space
            .free_eta
            .iter()
            .filter(|p| selected.structure.has_kappa(p) && selected.structure.has_eta(p))
            .cloned()
            .collect();
        let mut targets = Vec::new();
        for size in 1..=removable.len() {
            for removed in combinations(&removable, size) {
                let etas: Vec<String> = selected
                    .structure
                    .etas
                    .iter()
                    .filter(|p| !removed.contains(p))
                    .cloned()
                    .collect();
                targets.push(IovStructure::new(
                    etas,
                    selected.structure.eta_blocks.clone(),
                    selected.structure.kappas.clone(),
                    selected.structure.kappa_blocks.clone(),
                ));
            }
        }
        if targets.is_empty() {
            notes.push("IIV step skipped: no parameter with a κ carries a free η to remove".into());
        } else {
            let mut candidates = Vec::with_capacity(targets.len());
            for target in &targets {
                let id = new_id();
                candidates.push(derive(&id, &selected, target, &space, options)?);
            }
            emit(IovsearchEvent::StepStarted {
                step: 2,
                candidates: candidates.len(),
            });
            let report = fitter.fit_step("step-2", &candidates)?;
            notes.extend(report.warnings.iter().cloned());
            let mut nodes = Vec::new();
            for (candidate, target) in candidates.iter().zip(&targets) {
                let Some(result) = report.results.iter().find(|r| r.id == candidate.id) else {
                    continue;
                };
                rows.push(row_of(
                    result,
                    2,
                    Some(&selected.id),
                    target,
                    candidate,
                    options,
                    criterion,
                ));
                let node = Node::from_result(
                    result,
                    &report,
                    candidate.model.clone(),
                    target.clone(),
                    criterion,
                )?;
                store.insert(node.id.clone(), (node.model.clone(), node.fit.clone()));
                nodes.push(node);
            }
            cancelled |= report.cancelled;
            let (ranked, best) = rank_nodes(&selected, &nodes, options.cutoff);
            annotate(&mut rows, 2, &ranked);
            let winner = nodes
                .iter()
                .find(|n| n.id == best)
                .cloned()
                .unwrap_or_else(|| selected.clone());
            emit(IovsearchEvent::StepFinished {
                step: 2,
                best: (best.clone(), winner.criterion),
                improved: best != selected.id,
            });
            steps.push(StepSummary {
                step: 2,
                kind: crate::iivsearch::StepKind::NumberOfEtas,
                parent: selected.id.clone(),
                ranked,
                best,
            });
            selected = winner;
        }
    }

    let final_id = selected.id.clone();
    if let Some(r) = rows.iter_mut().find(|r| r.id == final_id) {
        r.selected = true;
    }
    let models: BTreeMap<String, ModelText> = store
        .iter()
        .map(|(id, (text, _))| (id.clone(), text.clone()))
        .collect();
    let (final_model, final_fit) = store
        .remove(&final_id)
        .expect("every fitted model is stored");
    Ok(IovsearchResult {
        options: options.clone(),
        criterion,
        input_model: space.input_model,
        input_structure: space.input_structure,
        rows,
        steps,
        final_id,
        final_model,
        final_structure: selected.structure,
        final_fit,
        final_criterion: selected.criterion,
        models,
        notes,
        cancelled,
    })
}

/// Pharmpy's `rank_models` with `parent` as reference.
fn rank_nodes(parent: &Node, candidates: &[Node], cutoff: Option<f64>) -> (Vec<StepRank>, String) {
    let cands: Vec<(String, f64, bool)> = candidates
        .iter()
        .map(|n| (n.id.clone(), n.criterion, n.eligible))
        .collect();
    rank_models(
        (&parent.id, parent.criterion, parent.eligible),
        &cands,
        cutoff,
    )
}

/// Write a step's ranks and Δ onto its rows.
fn annotate(rows: &mut [ModelRow], step: usize, ranked: &[StepRank]) {
    for r in ranked {
        if let Some(row) = rows
            .iter_mut()
            .find(|row| row.id == r.id && row.step == step)
        {
            row.d_criterion = r.d_criterion.map(|d| -d);
            row.rank = r.rank;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn row_of(
    result: &CandidateResult,
    step: usize,
    parent: Option<&str>,
    structure: &IovStructure,
    candidate: &Candidate,
    options: &IovsearchOptions,
    criterion: Criterion,
) -> ModelRow {
    ModelRow {
        id: result.id.clone(),
        parent: parent.map(str::to_string),
        step,
        structure: structure.clone(),
        ofv: result.ofv,
        n_parameters: result.fit.as_ref().map(|f| f.n_parameters),
        criterion: match &result.fit {
            Some(fit) => criterion.of(fit),
            None => result.criterion,
        },
        d_criterion: None,
        rank: None,
        converged: result.converged,
        passed: result.verdict.passed && result.error.is_none(),
        failures: result.verdict.failures.clone(),
        error: result.error.clone(),
        seconds: result.seconds,
        starts: candidate.n_starts.unwrap_or(options.starts),
        selected: false,
        reused: result.reused,
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
