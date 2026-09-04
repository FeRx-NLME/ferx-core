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
//! in the tools above it.
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
pub mod journal;
mod output;
mod runner;
#[cfg(test)]
mod test_support;

pub use candidate::{Candidate, CandidateResult, Criterion, FeatureVector, RunOptions};
pub use journal::{CandidateRecord, SearchManifest};
pub use output::{table_path, COLUMNS as TABLE_COLUMNS};
pub use runner::{RunReport, Runner};
