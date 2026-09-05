//! Model-space search: the shared machinery every search tool sits on (#1175).
//!
//! `ferx-core::edit` can *generate* a candidate model, `ferx_core::fit` can fit
//! one and `ferx_core::bic` / `ferx_core::check_strictness` can rank and judge
//! one. What every search tool in the epic — covsearch, modelsearch, iivsearch,
//! ruvsearch — additionally needs is the orchestration between those verbs:
//! fit N candidates in parallel without oversubscribing the machine,
//! deduplicate the ones that render to the same model, journal each outcome so
//! an interrupted overnight search resumes, and report every candidate
//! including the ones that failed.
//!
//! That is [`Runner`] (#1178). It is written once here rather than four times
//! in the tools above it. What to search over comes from a `.ferxsearch` file
//! ([`SearchConfig`], #1179): a TOML file whose space is a Pharmpy MFL string
//! ([`mfl`]), checked against what ferx can build ([`coverage`]) and resolved
//! against the base model's parameters and covariates ([`mod@resolve`]).
//!
//! ```no_run
//! use ferx_core::edit::ModelText;
//! use ferx_tools::search::{Candidate, FeatureVector, RunOptions, Runner};
//!
//! # fn demo(base: &ModelText, data: &ferx_core::Population) -> Result<(), String> {
//! let candidates = vec![
//!     Candidate::new("base", base.clone()),
//!     Candidate::new("cl-wt-pow", base.clone())
//!         .parent("base")
//!         .features(FeatureVector::new().with("CL-WT", "pow")),
//! ];
//!
//! let report = Runner::new()
//!     .threads(8)
//!     .cache_dir(".ferx-search")
//!     .run(&candidates, data, &RunOptions::default())?;
//!
//! if let Some(best) = report.best() {
//!     println!("winner: {} ({:.3})", best.id, best.criterion);
//! }
//! # Ok(())
//! # }
//! ```

mod candidate;
pub mod config;
pub mod coverage;
pub mod journal;
pub mod mfl;
mod output;
pub mod resolve;
mod runner;
#[cfg(test)]
mod test_support;

pub use candidate::{
    Candidate, CandidateError, CandidateResult, Criterion, FeatureVector, RunOptions,
};
pub use config::{BaseModel, RankConfig, RankType, RunConfig, SearchConfig, StrictnessConfig};
pub use coverage::{check_coverage, CoverageError, Gap};
pub use journal::{CandidateRecord, SearchManifest};
pub use mfl::{Feature, Mfl, Statement};
pub use output::{partial_table_path, table_path, COLUMNS as TABLE_COLUMNS};
pub use resolve::{resolve, CovariateEffectSpec, ModelContext, PkTemplate, Resolved};
pub use runner::{RunReport, Runner};
