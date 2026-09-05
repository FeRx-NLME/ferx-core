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
//! Tool-specific sections (`[covsearch]`, `[modelsearch]`, …) are kept as raw
//! tables in [`SearchConfig::tools`] for the tool that owns them to read;
//! this module does not know their schemas.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ferx_core::edit::ModelText;
use ferx_core::{prepare_run, BicType, PreparedRun, Strictness};
use serde::Deserialize;

use super::candidate::{Criterion, RunOptions};
use super::coverage::check_coverage;
use super::mfl::Mfl;
use super::resolve::{resolve, ModelContext, Resolved};
use super::runner::Runner;

/// The file extension a search configuration is expected to carry.
pub const EXTENSION: &str = "ferxsearch";

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
    /// What candidates are ranked on.
    #[serde(rename = "type", default)]
    pub kind: RankType,
    /// The improvement a candidate must show over its parent to be accepted,
    /// on the criterion's own scale. Its use is the tool's: an SCM step reads
    /// it as a ΔOFV, a BIC search as a ΔBIC. `None` leaves it to the tool's
    /// default.
    #[serde(default)]
    pub cutoff: Option<f64>,
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
    /// pyDarwin-style penalized fitness — parsed so the file format is
    /// stable, but not implemented until #1175 P6. Loading a file that asks
    /// for it is an error.
    Penalized,
}

impl RankType {
    /// The runner criterion this ranks on.
    pub fn criterion(&self) -> Result<Criterion, String> {
        Ok(match self {
            RankType::Ofv => Criterion::Ofv,
            RankType::Aic => Criterion::Aic,
            RankType::Bic | RankType::BicMixed => Criterion::Bic(BicType::Mixed),
            RankType::BicIiv => Criterion::Bic(BicType::Iiv),
            RankType::BicRandom => Criterion::Bic(BicType::Random),
            RankType::BicFixed => Criterion::Bic(BicType::Fixed),
            RankType::Penalized => {
                return Err(
                    "[rank] type = \"penalized\" is not implemented yet (#1175 P6); use ofv, \
                     aic or one of the bic variants"
                        .into(),
                )
            }
        })
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
    /// `n_starts` per candidate (Pharmpy's *retries*). Defaults to the
    /// runner's 3.
    #[serde(default = "default_retries")]
    pub retries: usize,
    /// Journal / cache directory, relative to the config file.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Reuse journalled candidates from `cache_dir`.
    #[serde(default)]
    pub resume: bool,
}

fn default_retries() -> usize {
    RunOptions::default().n_starts
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            threads: None,
            retries: default_retries(),
            cache_dir: None,
            resume: false,
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
    space: SpaceSection,
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
                toml::Value::Table(table) => {
                    tools.insert(key, table);
                }
                _ => {
                    return Err(format!(
                        "unknown top-level key `{key}`; the top level takes `base`, `data` and \
                         the [space], [rank], [strictness] and [run] sections"
                    ))
                }
            }
        }
        let mfl = Mfl::parse(&raw.space.mfl).map_err(|e| format!("[space] mfl: {e}"))?;
        if mfl.features().next().is_none() {
            return Err("[space] mfl: the search space is empty — no feature statement".into());
        }
        check_coverage(&mfl).map_err(|e| format!("[space] mfl: {e}"))?;
        // Fail on an unimplemented rank type at load, not after the first fit.
        raw.rank.kind.criterion()?;
        Ok(SearchConfig {
            base: dir.join(&raw.base),
            data: raw.data.map(|d| dir.join(d)),
            mfl_source: raw.space.mfl,
            mfl,
            rank: raw.rank,
            strictness: raw.strictness,
            run: raw.run,
            tools,
            dir: dir.to_path_buf(),
        })
    }

    /// The runner options this file asks for.
    pub fn run_options(&self) -> RunOptions {
        RunOptions {
            criterion: self
                .rank
                .kind
                .criterion()
                .expect("rank type validated at load"),
            strictness: self.strictness.strictness(),
            n_starts: self.run.retries.max(1),
            resume: self.run.resume,
            fit_options: None,
        }
    }

    /// A [`Runner`] with this file's thread count and cache directory.
    pub fn runner(&self) -> Runner {
        let mut runner = Runner::new();
        if let Some(t) = self.run.threads {
            runner = runner.threads(t);
        }
        if let Some(dir) = &self.run.cache_dir {
            runner = runner.cache_dir(self.dir.join(dir));
        }
        runner
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
