//! The `.ferxsearch` file: what to search over, and how to fit and judge each
//! candidate (#1179).
//!
//! `docs/model-file/covariate-model.qmd` fixes the boundary this file sits
//! on: what PsN calls `[test_relations]`, `valid_states` and `p_forward` is
//! *search configuration, not model*. So the space lives here, in a TOML
//! file next to the model, and never in a `.ferx` block.
//!
//! ```toml
//! base = "warfarin.ferx"
//! data = "warfarin.csv"
//!
//! [space]
//! mfl = """
//! ABSORPTION([FO,ZO]); PERIPHERALS(0..1); LAGTIME([OFF,ON])
//! COVARIATE?(@IIV, @CONTINUOUS, [pow,lin]); COVARIATE?(@IIV, @CATEGORICAL, cat)
//! """
//!
//! [rank]
//! type   = "bic"      # ofv | aic | bic | bic_mixed | bic_iiv | bic_random | bic_fixed
//! cutoff = 3.84
//!
//! [strictness]
//! require_converged    = true
//! max_condition_number = 1000.0
//! max_correlation      = 0.95
//! reject_on_boundary   = true
//!
//! [run]
//! threads   = 8
//! retries   = 3
//! cache_dir = ".ferx-search"
//! ```
//!
//! Loading a file does three things beyond deserialising it: it parses the
//! MFL (a syntax error names the statement), it checks every feature against
//! the [coverage table](super::coverage) (an unsupported feature is a hard
//! error naming it — never a narrowed search), and it resolves `base` and
//! `data` relative to the file's own directory, so the file can be moved
//! with its model.
//!
//! Tool-specific sections — one of [`TOOL_SECTIONS`], e.g. `[covsearch]` or
//! `[modelsearch]` — are kept as raw tables in [`SearchConfig::tools`] for
//! the tool that owns them to read; this module does not know their schemas.
//! Any other table is an error, so a misspelt `[strictnes]` cannot land in
//! `tools` and silently leave the gate at its defaults.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferx_core::edit::ModelText;
use ferx_core::{prepare_run, BicType, PreparedRun, Strictness};
use serde::Deserialize;

use super::candidate::{Criterion, RunOptions};
use super::coverage::check_coverage;
use super::mfl::Mfl;
use super::penalty::Penalties;
use super::resolve::{resolve, ModelContext, Resolved};
use super::runner::Runner;

/// The file extension a search configuration is expected to carry.
pub const EXTENSION: &str = "ferxsearch";

/// The tool sections a `.ferxsearch` file may carry, by Pharmpy's tool
/// names (#1175). The loader keeps them verbatim in [`SearchConfig::tools`];
/// a table with any other name is an error, since the alternative — a
/// misspelt core section quietly filed under `tools` — is a gate that never
/// runs.
pub const TOOL_SECTIONS: &[&str] = &[
    "amd",
    "covsearch",
    "globalsearch",
    "modelsearch",
    "iivsearch",
    "iovsearch",
    "ruvsearch",
    "structsearch",
    "allometry",
];

/// The core sections, for error messages.
const CORE_SECTIONS: &[&str] = &["space", "rank", "strictness", "run"];

/// A loaded `.ferxsearch` file.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// The base model, resolved to an absolute (or config-relative) path.
    pub base: PathBuf,
    /// The dataset, likewise; `None` defers to the model's `[data]` block.
    pub data: Option<PathBuf>,
    /// The `[space] mfl` text, verbatim.
    pub mfl_source: String,
    /// The parsed space. Coverage-checked; not yet resolved against a model.
    pub mfl: Mfl,
    pub rank: RankConfig,
    pub strictness: StrictnessConfig,
    pub run: RunConfig,
    /// Every other `[section]`, for the tool that owns it.
    pub tools: BTreeMap<String, toml::Table>,
    /// The directory the file was loaded from, which `base`, `data` and
    /// `run.cache_dir` are relative to.
    pub dir: PathBuf,
}

/// `[rank]`.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RankConfig {
    /// What candidates are ranked on. `None` when the file does not say,
    /// which every tool reads as its own default — the mixed BIC for a
    /// structural or variability search, the likelihood-ratio test for
    /// covsearch. Kept as an `Option` rather than defaulted at load so a tool
    /// that ranks one way only can tell "unset" from "asked for something
    /// else" and refuse the latter out loud.
    #[serde(rename = "type", default)]
    pub kind: Option<RankType>,
    /// The improvement a candidate must show over its parent to be accepted,
    /// on the criterion's own scale. Its use is the tool's: an SCM step reads
    /// it as a ΔOFV, a BIC search as a ΔBIC. `None` leaves it to the tool's
    /// default.
    #[serde(default)]
    pub cutoff: Option<f64>,
    /// `[rank.penalties]`: the schedule a `penalized` criterion charges
    /// (#1185). Every key is optional and overlays pyDarwin's defaults;
    /// `None` is the defaults. Read whatever the type, since the two
    /// search-level charges (non-influential genes, crashes) apply to a
    /// global search under any criterion.
    #[serde(default)]
    pub penalties: Option<Penalties>,
}

/// `[rank] type`. `bic` is the mixed BIC, Pharmpy's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RankType {
    Ofv,
    Aic,
    #[default]
    Bic,
    BicMixed,
    BicIiv,
    BicRandom,
    BicFixed,
    /// pyDarwin-style penalized fitness (#1185): `OFV` plus the
    /// [`Penalties`] schedule, `[rank.penalties]` overlaying the defaults.
    Penalized,
}

impl RankConfig {
    /// The rank type, with the file's silence read as [`RankType::default`].
    pub fn kind_or_default(&self) -> RankType {
        self.kind.unwrap_or_default()
    }

    /// The penalty schedule: `[rank.penalties]` over pyDarwin's defaults.
    pub fn penalties(&self) -> Penalties {
        self.penalties.unwrap_or_default()
    }

    /// The runner criterion this file ranks on, with its own schedule
    /// behind a `penalized` type. Fails on a schedule that does not
    /// validate.
    pub fn criterion(&self) -> Result<Criterion, String> {
        self.penalties().validate()?;
        Ok(self.kind_or_default().criterion_with(self.penalties()))
    }
}

impl RankType {
    /// The `[rank] type` spelling of this variant, for messages.
    pub fn label(&self) -> &'static str {
        match self {
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

    /// The runner criterion this ranks on, a `penalized` type at pyDarwin's
    /// default schedule. A tool holding the file's `[rank.penalties]` uses
    /// [`criterion_with`](Self::criterion_with) — or [`RankConfig::criterion`]
    /// — so the file's charges are the ones applied.
    ///
    /// `Result` for the tools that validate a rank type they cannot honour
    /// and report it as a load error; every variant currently maps.
    pub fn criterion(&self) -> Result<Criterion, String> {
        Ok(self.criterion_with(Penalties::default()))
    }

    /// The runner criterion, with `penalties` behind a `penalized` type
    /// (ignored by every other type).
    pub fn criterion_with(&self, penalties: Penalties) -> Criterion {
        match self {
            RankType::Ofv => Criterion::Ofv,
            RankType::Aic => Criterion::Aic,
            RankType::Bic | RankType::BicMixed => Criterion::Bic(BicType::Mixed),
            RankType::BicIiv => Criterion::Bic(BicType::Iiv),
            RankType::BicRandom => Criterion::Bic(BicType::Random),
            RankType::BicFixed => Criterion::Bic(BicType::Fixed),
            RankType::Penalized => Criterion::Penalized(penalties),
        }
    }
}

/// `[strictness]`. Every key is optional and overlays
/// [`Strictness::default`], so a file states only what it changes.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct StrictnessConfig {
    #[serde(default)]
    pub require_converged: Option<bool>,
    #[serde(default)]
    pub require_covariance: Option<bool>,
    #[serde(default)]
    pub max_condition_number: Option<f64>,
    #[serde(default)]
    pub max_correlation: Option<f64>,
    #[serde(default)]
    pub reject_on_boundary: Option<bool>,
    #[serde(default)]
    pub reject_init_stall: Option<bool>,
}

impl StrictnessConfig {
    /// The gate, with unspecified keys at their `ferx-core` defaults.
    pub fn strictness(&self) -> Strictness {
        let mut s = Strictness::default();
        if let Some(v) = self.require_converged {
            s.require_converged = v;
        }
        if let Some(v) = self.require_covariance {
            s.require_covariance = v;
        }
        if let Some(v) = self.max_condition_number {
            s.max_condition_number = Some(v);
        }
        if let Some(v) = self.max_correlation {
            s.max_correlation = Some(v);
        }
        if let Some(v) = self.reject_on_boundary {
            s.reject_on_boundary = v;
        }
        if let Some(v) = self.reject_init_stall {
            s.reject_init_stall = v;
        }
        s
    }
}

/// `[run]`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Worker threads for the candidate pool; `None` lets the runner choose.
    #[serde(default)]
    pub threads: Option<usize>,
    /// Perturbed restarts per candidate on top of the fit from the exact
    /// initials — Pharmpy's *retries*, so a value copied from a Pharmpy call
    /// means the same thing. The engine's `n_starts` is `retries + 1`, start
    /// 0 being the exact initials; `retries = 0` is a single start. Defaults
    /// to 2, which is the runner's default of 3 starts.
    #[serde(default = "default_retries")]
    pub retries: usize,
    /// Journal / cache directory, relative to the config file.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Reuse journalled candidates from `cache_dir`.
    #[serde(default)]
    pub resume: bool,
    /// Other searches' directories — relative to the config file — whose
    /// cached fits every step of this run may reuse (#1185): a candidate
    /// another tool already fitted to the same data with the same settings
    /// is re-scored under this file's `[rank]` and `[strictness]` rather
    /// than fitted again. Read recursively, so a tool's run directory is one
    /// entry.
    #[serde(default)]
    pub reuse_from: Vec<PathBuf>,
}

fn default_retries() -> usize {
    RunOptions::default().n_starts - 1
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            threads: None,
            retries: default_retries(),
            cache_dir: None,
            resume: false,
            reuse_from: Vec::new(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpaceSection {
    mfl: String,
}

#[derive(Deserialize)]
struct RawConfig {
    base: PathBuf,
    #[serde(default)]
    data: Option<PathBuf>,
    /// Optional since #1182: ruvsearch's candidates are not MFL features, so
    /// its file has no `[space]`. A tool that searches a space checks for one
    /// itself ([`SearchConfig::require_space`]).
    #[serde(default)]
    space: Option<SpaceSection>,
    #[serde(default)]
    rank: RankConfig,
    #[serde(default)]
    strictness: StrictnessConfig,
    #[serde(default)]
    run: RunConfig,
    #[serde(flatten)]
    rest: BTreeMap<String, toml::Value>,
}

/// The base model and its dataset, loaded once for symbol resolution and for
/// every candidate the search derives from it.
pub struct BaseModel {
    pub prepared: PreparedRun,
    pub text: ModelText,
}

impl SearchConfig {
    /// Load and validate a `.ferxsearch` file.
    pub fn load(path: impl AsRef<Path>) -> Result<SearchConfig, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_str(&text, &dir).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parse config text whose relative paths are taken against `dir`.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str, dir: &Path) -> Result<SearchConfig, String> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| format!("{e}"))?;
        let mut tools = BTreeMap::new();
        for (key, value) in raw.rest {
            match value {
                toml::Value::Table(table) if TOOL_SECTIONS.contains(&key.as_str()) => {
                    tools.insert(key, table);
                }
                toml::Value::Table(_) => {
                    let sections = |names: &[&str]| {
                        names
                            .iter()
                            .map(|s| format!("[{s}]"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    };
                    return Err(format!(
                        "unknown section `[{key}]`; the file takes {} and the tool sections {}",
                        sections(CORE_SECTIONS),
                        sections(TOOL_SECTIONS)
                    ));
                }
                _ => {
                    return Err(format!(
                        "unknown top-level key `{key}`; the top level takes `base`, `data` and \
                         the [space], [rank], [strictness] and [run] sections"
                    ))
                }
            }
        }
        let (mfl_source, mfl) = match raw.space {
            Some(space) => {
                let mfl = Mfl::parse(&space.mfl).map_err(|e| format!("[space] mfl: {e}"))?;
                if mfl.features().next().is_none() {
                    return Err(
                        "[space] mfl: the search space is empty — no feature statement".into(),
                    );
                }
                check_coverage(&mfl).map_err(|e| format!("[space] mfl: {e}"))?;
                (space.mfl, mfl)
            }
            None => (
                String::new(),
                Mfl {
                    statements: Vec::new(),
                },
            ),
        };
        // A bad penalty schedule fails at load, not after the first fit.
        raw.rank.criterion()?;
        Ok(SearchConfig {
            base: dir.join(&raw.base),
            data: raw.data.map(|d| dir.join(d)),
            mfl_source,
            mfl,
            rank: raw.rank,
            strictness: raw.strictness,
            run: raw.run,
            tools,
            dir: dir.to_path_buf(),
        })
    }

    /// Whether the file has a `[space]` with at least one feature statement.
    pub fn has_space(&self) -> bool {
        self.mfl.features().next().is_some()
    }

    /// Refuse a file with no `[space]` on behalf of a tool that searches one
    /// — covsearch, modelsearch — naming the tool and what its space is.
    pub fn require_space(&self, tool: &str, what: &str) -> Result<(), String> {
        if self.has_space() {
            Ok(())
        } else {
            Err(format!(
                "{tool} needs a [space] section: `mfl = \"...\"` with {what}. Only ruvsearch \
                 (whose candidates are the residual-error forms) and iovsearch (every \
                 parameter with a free η by default) run without one"
            ))
        }
    }

    /// The runner options this file asks for.
    pub fn run_options(&self) -> RunOptions {
        RunOptions {
            criterion: self
                .rank
                .criterion()
                .expect("rank config validated at load"),
            strictness: self.strictness.strictness(),
            // Pharmpy's retries are on top of the exact-initials fit.
            n_starts: self.run.retries + 1,
            resume: self.run.resume,
            fit_options: None,
        }
    }

    /// A [`Runner`] with this file's thread count, cache directory and
    /// reuse directories.
    pub fn runner(&self) -> Runner {
        let mut runner = Runner::new();
        if let Some(t) = self.run.threads {
            runner = runner.threads(t);
        }
        if let Some(dir) = &self.run.cache_dir {
            runner = runner.cache_dir(self.dir.join(dir));
        }
        for dir in self.reuse_dirs() {
            runner = runner.reuse_from(dir);
        }
        runner
    }

    /// `[run] reuse_from`, resolved against the file's directory.
    pub fn reuse_dirs(&self) -> Vec<PathBuf> {
        self.run
            .reuse_from
            .iter()
            .map(|d| self.dir.join(d))
            .collect()
    }

    /// Read the base model and its dataset.
    pub fn load_base(&self) -> Result<BaseModel, String> {
        let base = self.base.to_string_lossy().into_owned();
        let data = self.data.as_ref().map(|d| d.to_string_lossy().into_owned());
        let prepared = prepare_run(&base, data.as_deref())?;
        let source = std::fs::read_to_string(&self.base)
            .map_err(|e| format!("cannot read {}: {e}", self.base.display()))?;
        let text = ModelText::parse(&source)?;
        Ok(BaseModel { prepared, text })
    }

    /// Resolve the space's symbols and wildcards against the base model.
    pub fn resolve_space(&self, base: &BaseModel) -> Result<Resolved, String> {
        let ctx =
            ModelContext::from_model(&base.prepared.parsed, &base.text, &base.prepared.population)?;
        resolve(&self.mfl, &ctx)
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
