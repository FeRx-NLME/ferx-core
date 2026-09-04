//! The shared candidate runner (#1178): fit N candidates in parallel,
//! deduplicate, cache, journal, and report every one of them.
//!
//! Every tool in the #1175 search epic — covsearch, modelsearch, iivsearch,
//! ruvsearch — is *generate candidates → fit them → rank them*, and the middle
//! verb is the same code four times over. This is that code once.
//!
//! # What the runner is responsible for
//!
//! * **Identity.** A candidate is its rendered model text, keyed by
//!   [`ModelText::canonical_hash`](ferx_core::edit::ModelText::canonical_hash).
//!   Two candidates the search reached by different step orders render to the
//!   same canonical text and are fitted **once** — pyDarwin's "model
//!   uniqueness", for free, and the same key the journal and the fit cache use.
//! * **Parallelism that does not oversubscribe.** Candidates are embarrassingly
//!   parallel and `fit()` already `par_iter`s over subjects, so a naive outer
//!   `par_iter` nests two pools competing for the same workers. The split is a
//!   [`PoolPlan`] (#1115), which also gives the outer pool the ferx worker stack
//!   rather than Rayon's 2 MiB default.
//! * **Retries before the gate.** Each candidate is fitted with
//!   [`RunOptions::n_starts`] starts, and only then judged by
//!   [`check_strictness`]. Under automation an init stall (#751) or an inner-EBE
//!   mode (#864, #891) is a *model-selection* error: it ranks a model on an OFV
//!   that says nothing about the model.
//! * **Nothing is dropped silently.** A candidate that failed to compile,
//!   failed to fit or failed the gate comes back with its reason attached and
//!   appears in `candidates.csv`.
//! * **Resume.** With a cache directory, each outcome is journalled as it
//!   finishes and a resumed run refits only what is missing — see
//!   [`super::journal`].
//! * **Cancellation.** A flipped [`CancelFlag`] stops the run *between*
//!   candidates and returns what finished, with [`RunReport::cancelled`] set.
//!
//! # What it is not responsible for
//!
//! Generating candidates, deciding which to keep, and the step logic on top —
//! those are the individual search tools. The runner takes a list and returns
//! the same list scored.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ferx_core::cancel::is_cancelled;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{
    bind_theta_levels, check_strictness, fit, CancelFlag, FitResult, GradientMethod, PoolPlan,
    Population, StrictnessVerdict,
};
use rayon::prelude::*;

use super::candidate::{Candidate, CandidateResult, RunOptions};
use super::journal::{self, CandidateRecord, Journal, SearchManifest};
use super::output;

/// What one [`Runner::run`] did.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// One entry per candidate, in the order they were submitted — minus any
    /// the run never reached because it was cancelled.
    pub results: Vec<CandidateResult>,
    /// The run stopped early on a flipped [`CancelFlag`]; `results` is partial.
    pub cancelled: bool,
    /// Candidates actually fitted in this run.
    pub fitted: usize,
    /// Candidates whose outcome came from the journal.
    pub reused: usize,
    /// Candidates that shared another candidate's canonical text and were
    /// therefore not fitted — the fits this run did not have to run.
    pub deduped: usize,
}

impl RunReport {
    /// The eligible result with the lowest criterion — every
    /// [`Criterion`](super::Criterion) is lower-is-better — or `None` when
    /// nothing passed the gate.
    ///
    /// Ties go to the earlier candidate, so a search that generates its
    /// candidates in a deterministic order gets a deterministic winner.
    pub fn best(&self) -> Option<&CandidateResult> {
        self.results
            .iter()
            .filter(|r| r.eligible())
            .min_by(|a, b| a.criterion.total_cmp(&b.criterion))
    }
}

/// The runner: a thread budget, an optional cache directory, an optional
/// cancellation flag.
///
/// ```no_run
/// use ferx_tools::search::{Candidate, RunOptions, Runner};
/// # fn demo(candidates: Vec<Candidate>, data: &ferx_core::Population) -> Result<(), String> {
/// let report = Runner::new()
///     .threads(8)
///     .cache_dir(".ferx-search")
///     .run(&candidates, data, &RunOptions::default())?;
/// println!("{} fitted, {} reused, {} deduped", report.fitted, report.reused, report.deduped);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct Runner {
    threads: usize,
    cache_dir: Option<PathBuf>,
    cancel: Option<CancelFlag>,
}

impl Runner {
    /// A runner on the engine's default thread budget, with no cache directory
    /// (so no journal, no resume and no `candidates.csv`) and no cancellation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total worker threads across both levels of parallelism. `0` — the
    /// default — means the engine default, i.e. whatever an unpinned `fit()`
    /// would run on. See [`PoolPlan::from_budget`] for how the budget splits.
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Where the journal, the fit cache and `candidates.csv` are written.
    /// Without one the run holds everything in memory and is not resumable.
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(dir.into());
        self
    }

    /// The flag that stops the run between candidates. It is also handed to
    /// each `fit()`, so a long individual fit unwinds rather than running to
    /// completion after the user has asked to stop.
    pub fn cancel(mut self, flag: CancelFlag) -> Self {
        self.cancel = Some(flag);
        self
    }

    /// Fit every candidate against `data` and return them scored.
    ///
    /// Each candidate's model text is parsed on its own, bound to the data
    /// (`theta NAME[COL]` level counts and symbolic `[covariate_model]`
    /// statistics), and fitted from its own `[fit_options]` with the runner's
    /// quiet / thread / start / cancel overrides applied on top.
    ///
    /// # Errors
    ///
    /// Only for failures of the *run*, not of a candidate: duplicate candidate
    /// ids, an unusable cache directory, an incompatible resume. A candidate
    /// that does not compile or does not fit is a [`CandidateResult`] carrying
    /// the reason.
    pub fn run(
        &self,
        candidates: &[Candidate],
        data: &Population,
        options: &RunOptions,
    ) -> Result<RunReport, String> {
        self.run_with_fitter(candidates, data, options, |candidate, threads_per_fit| {
            compile_and_fit(candidate, data, options, threads_per_fit, &self.cancel)
        })
    }

    /// The body of [`run`](Self::run), with the fit itself injected.
    ///
    /// The seam exists so the orchestration — dedup, thread budget, journal,
    /// resume, cancellation — can be tested against a fitter that counts its
    /// callers instead of against real fits, which would put every one of those
    /// tests in the slow tier and make the concurrency assertion untestable
    /// (there is no way to observe "two fits are in flight" from outside
    /// `fit()`).
    fn run_with_fitter<F>(
        &self,
        candidates: &[Candidate],
        data: &Population,
        options: &RunOptions,
        fitter: F,
    ) -> Result<RunReport, String>
    where
        F: Fn(&Candidate, usize) -> Result<FitResult, String> + Sync,
    {
        reject_duplicate_ids(candidates)?;

        // ── identity ────────────────────────────────────────────────────────
        // The representative of a hash is the first candidate that carries it;
        // every later one is a duplicate and is never fitted.
        let hashes: Vec<String> = candidates.iter().map(Candidate::hash).collect();
        let mut representative: HashMap<&str, usize> = HashMap::new();
        let mut duplicate_of: Vec<Option<usize>> = vec![None; candidates.len()];
        for (i, hash) in hashes.iter().enumerate() {
            match representative.get(hash.as_str()) {
                Some(&first) => duplicate_of[i] = Some(first),
                None => {
                    representative.insert(hash.as_str(), i);
                }
            }
        }
        let deduped = duplicate_of.iter().filter(|d| d.is_some()).count();

        // ── resume ──────────────────────────────────────────────────────────
        let manifest = SearchManifest::new(options, data);
        let kept = match (&self.cache_dir, options.resume) {
            (Some(dir), true) => load_resumable(dir, &manifest)?,
            _ => Vec::new(),
        };
        let reused: HashMap<&str, &CandidateRecord> = kept
            .iter()
            .map(|record| (record.hash.as_str(), record))
            .collect();

        // Checked before the journal is opened, not after: opening it rewrites
        // the journal file, and a run cancelled before it started must not be
        // what truncates the previous run's recovery data.
        if is_cancelled(&self.cancel) {
            return Ok(RunReport {
                results: Vec::new(),
                cancelled: true,
                fitted: 0,
                reused: 0,
                deduped: 0,
            });
        }

        // ── the journal ─────────────────────────────────────────────────────
        // Opened before the first fit so that a kill at any point from here on
        // leaves a directory the next resume can read.
        let journal = match &self.cache_dir {
            Some(dir) => {
                let j = Journal::create(dir, &kept)?;
                manifest.write(&journal::manifest_path(dir))?;
                Some(j)
            }
            None => None,
        };

        // ── what has to be fitted ───────────────────────────────────────────
        let todo: Vec<usize> = (0..candidates.len())
            .filter(|i| duplicate_of[*i].is_none())
            .filter(|i| !reused.contains_key(hashes[*i].as_str()))
            .collect();

        let plan = PoolPlan::from_budget(self.threads, todo.len());
        let threads_per_fit = plan.threads_per_fit();
        let n_fitted = AtomicUsize::new(0);

        let fit_one = |&i: &usize| -> Option<(usize, CandidateResult)> {
            // Between candidates, not inside one: this is the granularity the
            // cancellation contract promises, and it is why a cancelled run can
            // still return everything that finished.
            if is_cancelled(&self.cancel) {
                return None;
            }
            let candidate = &candidates[i];
            let started = Instant::now();
            let outcome = fitter(candidate, threads_per_fit);
            let seconds = started.elapsed().as_secs_f64();
            let result = match outcome {
                Ok(fitted) => {
                    let verdict = check_strictness(&fitted, &options.strictness);
                    let criterion = options.criterion.of(&fitted);
                    CandidateResult {
                        id: candidate.id.clone(),
                        hash: hashes[i].clone(),
                        parent: candidate.parent.clone(),
                        features: candidate.features.clone(),
                        fit: Some(fitted),
                        verdict,
                        criterion,
                        seconds,
                        error: None,
                        duplicate_of: None,
                        reused: false,
                    }
                }
                // A fit that failed while the flag was set cannot be told apart
                // from one the flag itself unwound, so it is dropped rather
                // than journalled — journalling it would make the next resume
                // carry a failure forward that was never the candidate's.
                Err(_) if is_cancelled(&self.cancel) => return None,
                Err(e) => CandidateResult {
                    id: candidate.id.clone(),
                    hash: hashes[i].clone(),
                    parent: candidate.parent.clone(),
                    features: candidate.features.clone(),
                    fit: None,
                    verdict: failed_verdict(&e),
                    criterion: f64::NAN,
                    seconds,
                    error: Some(e),
                    duplicate_of: None,
                    reused: false,
                },
            };
            if let Some(j) = &journal {
                j.append(&record_of(&result), result.fit.as_ref());
            }
            n_fitted.fetch_add(1, Ordering::Relaxed);
            Some((i, result))
        };

        let fitted: Vec<(usize, CandidateResult)> = if todo.is_empty() {
            Vec::new()
        } else {
            plan.install(|| todo.par_iter().filter_map(fit_one).collect())?
        };

        // Closed before the table is written, and the point at which a write
        // failure inside the parallel loop surfaces.
        if let Some(j) = journal {
            j.into_result()?;
        }

        // ── assembly, in submission order ───────────────────────────────────
        let mut by_index: HashMap<usize, CandidateResult> = fitted.into_iter().collect();
        let mut outcomes: HashMap<usize, CandidateResult> = HashMap::new();
        let mut n_reused = 0usize;
        for i in (0..candidates.len()).filter(|i| duplicate_of[*i].is_none()) {
            if let Some(result) = by_index.remove(&i) {
                outcomes.insert(i, result);
            } else if let Some(record) = reused.get(hashes[i].as_str()) {
                n_reused += 1;
                outcomes.insert(
                    i,
                    reused_result(&candidates[i], record, self.cache_dir.as_deref()),
                );
            }
        }

        let mut results = Vec::with_capacity(candidates.len());
        for i in 0..candidates.len() {
            match duplicate_of[i] {
                None => {
                    if let Some(result) = outcomes.get(&i) {
                        results.push(result.clone());
                    }
                }
                // A duplicate carries the representative's verdict and
                // criterion but not its `FitResult`: the fit is one object and
                // cloning it per duplicate would cost more than the answer is
                // worth. `duplicate_of` names where to find it.
                Some(first) => {
                    if let Some(source) = outcomes.get(&first) {
                        results.push(CandidateResult {
                            id: candidates[i].id.clone(),
                            hash: hashes[i].clone(),
                            parent: candidates[i].parent.clone(),
                            features: candidates[i].features.clone(),
                            fit: None,
                            verdict: source.verdict.clone(),
                            criterion: source.criterion,
                            seconds: 0.0,
                            error: source.error.clone(),
                            duplicate_of: Some(source.id.clone()),
                            reused: source.reused,
                        });
                    }
                }
            }
        }

        if let Some(dir) = &self.cache_dir {
            output::write_table(dir, &results)?;
        }

        Ok(RunReport {
            results,
            cancelled: is_cancelled(&self.cancel),
            fitted: n_fitted.load(Ordering::Relaxed),
            reused: n_reused,
            deduped,
        })
    }
}

/// Ids are how a search refers back to its candidates and how the table is
/// keyed, so two candidates sharing one is a bug in the caller, not something
/// to paper over by renaming.
fn reject_duplicate_ids(candidates: &[Candidate]) -> Result<(), String> {
    let mut seen: HashSet<&str> = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.id.trim().is_empty() {
            return Err("a candidate has an empty id".to_string());
        }
        if !seen.insert(candidate.id.as_str()) {
            return Err(format!(
                "two candidates share the id `{}`; ids must be unique within one run",
                candidate.id
            ));
        }
    }
    Ok(())
}

/// The verdict of a candidate that produced no fit at all. Not `passed`, and
/// the reason is the failure itself.
fn failed_verdict(error: &str) -> StrictnessVerdict {
    StrictnessVerdict {
        passed: false,
        failures: vec![format!("no fit: {error}")],
        skipped: Vec::new(),
    }
}

fn record_of(result: &CandidateResult) -> CandidateRecord {
    CandidateRecord {
        id: result.id.clone(),
        hash: result.hash.clone(),
        parent: result.parent.clone(),
        features: result.features.clone(),
        criterion: result.criterion.is_finite().then_some(result.criterion),
        ofv: result
            .fit
            .as_ref()
            .map(|f| f.ofv)
            .filter(|ofv| ofv.is_finite()),
        converged: result.fit.as_ref().is_some_and(|f| f.converged),
        passed: result.verdict.passed,
        failures: result.verdict.failures.clone(),
        skipped: result.verdict.skipped.clone(),
        seconds: result.seconds,
        error: result.error.clone(),
        has_fit: result.fit.is_some(),
    }
}

/// Rebuild a result from a journalled row, loading the cached fit when it is
/// still readable. A missing cache file costs the `FitResult`, not the row.
fn reused_result(
    candidate: &Candidate,
    record: &CandidateRecord,
    dir: Option<&Path>,
) -> CandidateResult {
    let fit = match (record.has_fit, dir) {
        (true, Some(dir)) => journal::load_fit(dir, &record.hash),
        _ => None,
    };
    CandidateResult {
        id: candidate.id.clone(),
        hash: record.hash.clone(),
        parent: candidate.parent.clone(),
        features: candidate.features.clone(),
        fit,
        verdict: StrictnessVerdict {
            passed: record.passed,
            failures: record.failures.clone(),
            skipped: record.skipped.clone(),
        },
        criterion: record.criterion.unwrap_or(f64::NAN),
        seconds: record.seconds,
        error: record.error.clone(),
        duplicate_of: None,
        reused: true,
    }
}

/// Read an interrupted run's finished candidates back out of its directory.
///
/// Anything dropped here is simply refitted, so the read errs towards dropping.
/// The one unsafe outcome is *keeping* a row that does not belong to this run,
/// which is what [`SearchManifest::check_compatible`] rules out.
fn load_resumable(dir: &Path, manifest: &SearchManifest) -> Result<Vec<CandidateRecord>, String> {
    let manifest_path = journal::manifest_path(dir);
    if !manifest_path.exists() {
        // Nothing to be incompatible with — a directory that was never written
        // resumes as an empty run rather than failing.
        return Ok(Vec::new());
    }
    let disk = SearchManifest::read(&manifest_path)?;
    manifest.check_compatible(&disk, dir)?;

    let mut seen: HashSet<String> = HashSet::new();
    let mut kept = Vec::new();
    for record in journal::read_records(&journal::journal_path(dir)) {
        // A hash appearing twice — two runs over the same directory — keeps the
        // first row, so a resume is not sensitive to the order of the log.
        if seen.insert(record.hash.clone()) {
            kept.push(record);
        }
    }
    Ok(kept)
}

/// Compile one candidate against the data and fit it.
///
/// The population is cloned per candidate because binding mutates it:
/// `theta NAME[COL]` synthesizes a per-record level-index column (#1064), and
/// the binder that does it takes `&mut Population`. A candidate whose model has
/// no such block is unchanged by the clone, but the runner cannot know that
/// without compiling first, and a shared mutable population across concurrent
/// candidates would be a data race by construction.
fn compile_and_fit(
    candidate: &Candidate,
    data: &Population,
    options: &RunOptions,
    threads_per_fit: usize,
    cancel: &Option<CancelFlag>,
) -> Result<FitResult, String> {
    let text = candidate.model.render();
    let mut parsed = parse_full_model(&text)
        .map_err(|e| format!("candidate `{}` does not compile: {e}", candidate.id))?;
    let mut population = data.clone();
    bind_theta_levels(&mut parsed, &text, &mut population)?;
    ferx_core::api::bind_covariate_stats(&mut parsed, &text, &population)?;

    // The caller's settings replace the file's wholesale when given; the four
    // overrides below are the runner's own and always win.
    let base = options
        .fit_options
        .clone()
        .unwrap_or_else(|| parsed.fit_options.clone());

    // Mirrors `prepare_run`: the file's `gradient = ...` reaches the engine
    // through `model.gradient_method`, and an SDE model is forced to FD.
    parsed.model.gradient_method = if parsed.model.is_sde() {
        GradientMethod::Fd
    } else {
        base.gradient_method
    };

    let init_params = parsed.model.default_params.clone();
    let mut fit_options = base.quiet();
    fit_options.n_starts = options.n_starts.max(1);
    fit_options.threads = Some(threads_per_fit);
    fit_options.cancel = cancel.clone();

    fit(&parsed.model, &population, &init_params, &fit_options)
}

#[cfg(test)]
#[path = "runner_tests.rs"]
mod tests;
