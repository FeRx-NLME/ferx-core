//! The candidate journal — what makes an interrupted search resumable (#1178).
//!
//! A model-space search is hours of `fit()` calls, and a search that has to
//! start over from candidate 1 after a kill is a search nobody runs overnight.
//! So the runner writes each candidate's outcome **as it finishes**, and a
//! resumed run reads that file and refits only what is missing. This is the
//! `bootstrap/` journal (#1143) with two shape changes:
//!
//! * **The key is the canonical hash, not an index.** A bootstrap replicate is
//!   identified by `(seed, index)`; a candidate is identified by the model text
//!   it renders to, so a candidate the search reaches by a different step order
//!   — or under a different id — hits the same journal row. That is dedup and
//!   resume off one key.
//! * **The row is JSON, one object per line.** A candidate's outcome is a
//!   verdict with a variable-length list of reasons, which is not a CSV row.
//!   The `candidates.csv` beside it is the human-readable *table*, rewritten in
//!   candidate order once the run finishes; this file is the log.
//!
//! Three properties carried over from #1143 unchanged:
//!
//! * **Every line is flushed.** A lost tail row is a lost fit, and the flush
//!   costs microseconds against a fit measured in seconds.
//! * **A truncated final line is dropped, not fatal.** A hard kill lands
//!   mid-write; [`read_records`] parses what it can and ignores the rest, and
//!   [`Journal::create`] then *rewrites* the file from what it read rather than
//!   appending onto the fragment.
//! * **The rewrite goes through a `.part` sibling** renamed into place, so the
//!   interrupted run's journal is never the only copy of its own recovery data.
//!
//! The fits themselves live beside the journal as `fits/<hash>.json`. They are
//! a *cache*: a missing or unreadable one costs the reused candidate its
//! `FitResult`, never its journalled criterion and verdict, so a corrupt fit
//! file degrades a resume instead of failing it.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use ferx_core::{FitOptions, FitResult, Population, Strictness};
use serde::{Deserialize, Serialize};

use super::candidate::FeatureVector;
use super::RunOptions;

/// One finished candidate, as stored in the journal.
///
/// `criterion` and `ofv` are `Option<f64>` rather than `f64` because JSON has
/// no `NaN`: `serde_json` writes a non-finite float as `null` and then refuses
/// to read it back as an `f64`, so a failed candidate's row would be a row that
/// cannot be resumed from. `None` *is* the failed case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    pub id: String,
    pub hash: String,
    pub parent: Option<String>,
    pub features: FeatureVector,
    pub criterion: Option<f64>,
    pub ofv: Option<f64>,
    pub converged: bool,
    pub passed: bool,
    pub failures: Vec<String>,
    pub skipped: Vec<String>,
    pub seconds: f64,
    pub error: Option<String>,
    /// Whether `error` describes the run rather than the candidate, in which
    /// case a resumed run refits instead of trusting this row — see
    /// [`CandidateError`](super::CandidateError).
    ///
    /// Defaults to `true` so a row written before the field existed is refitted
    /// rather than believed. That is the safe direction: a needless refit costs
    /// time, and a wrongly trusted failure costs the candidate.
    #[serde(default = "retryable_by_default")]
    pub retryable: bool,
    /// Whether a `fits/<hash>.json` was written for this row. `false` for a
    /// failed fit; `true` does not promise the file is still readable.
    pub has_fit: bool,
}

fn retryable_by_default() -> bool {
    true
}

pub fn journal_path(dir: &Path) -> PathBuf {
    dir.join("search_journal.jsonl")
}

pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("search_run.json")
}

pub fn fits_dir(dir: &Path) -> PathBuf {
    dir.join("fits")
}

pub fn fit_path(dir: &Path, hash: &str) -> PathBuf {
    fits_dir(dir).join(format!("{hash}.json"))
}

pub fn lock_path(dir: &Path) -> PathBuf {
    dir.join("search.lock")
}

/// The temp sibling a file is rewritten through, in the same directory so the
/// `rename` that swaps it in is atomic.
///
/// The name carries this process's id and a per-call counter, because a
/// *fixed* `.part` name is shared state between processes: two runs pointed at
/// one cache directory would `write` the same temp file concurrently and then
/// rename it twice, publishing a byte-mix of two fits under a hash that names
/// neither. [`DirLock`] is what normally keeps them apart; this is the second
/// lock on that door, since a `.part` collision fails silently.
fn part_path(path: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let n = NONCE.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.{n}.part", std::process::id()));
    path.with_file_name(name)
}

/// An exclusive claim on a search directory, held for the length of one run and
/// released when it is dropped.
///
/// Two runs sharing a cache directory do not merely interleave — they lose
/// data silently. `Journal::create` *renames* a fresh file over
/// `search_journal.jsonl`, so the run that was already appending keeps writing
/// into an unlinked inode and every row it produces from then on is gone, with
/// nothing on disk to say so. A lock file turns that into a message.
///
/// It is deliberately a plain `O_EXCL` file rather than an OS advisory lock, so
/// that it needs no new dependency and leaves something a user can see. The
/// price of that choice is a *stale* lock: a hard kill releases no file, and
/// resuming after a hard kill is the whole point of the journal — a lock that
/// had to be deleted by hand would break the flagship case. So the holder's pid
/// is written into the file and a lock whose owner is gone is taken over
/// automatically.
///
/// The one thing it will not do is take a lock from a *live* pid, so the
/// failure direction is "refused a run that could have proceeded", never "two
/// runs in one directory". A recycled pid therefore reads as live and refuses;
/// the message says which file to delete.
pub struct DirLock {
    path: PathBuf,
}

impl DirLock {
    /// Claim `dir`, taking over a lock whose owner is no longer running.
    pub fn acquire(dir: &Path) -> Result<DirLock, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
        let path = lock_path(dir);
        match Self::claim(&path) {
            Ok(lock) => Ok(lock),
            // Taken. If the holder is gone this is the debris of a hard kill,
            // and one retry is enough: losing the race to a third process just
            // means that process holds a lock we then report honestly.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let owner = std::fs::read_to_string(&path).unwrap_or_default();
                if !owner_is_running(&owner) {
                    let _ = std::fs::remove_file(&path);
                    if let Ok(lock) = Self::claim(&path) {
                        return Ok(lock);
                    }
                }
                let owner = owner.trim();
                let owner = if owner.is_empty() {
                    "owner unknown"
                } else {
                    owner
                };
                Err(format!(
                    "the search directory `{}` is already in use ({owner}): two runs sharing one \
                     directory overwrite each other's journal, so one run's fits would be lost. \
                     Point this run at a different directory — or, if that process is not \
                     actually running, delete `{}` and retry.",
                    dir.display(),
                    path.display()
                ))
            }
            Err(e) => Err(format!("cannot create `{}`: {e}", path.display())),
        }
    }

    fn claim(path: &Path) -> Result<DirLock, std::io::Error> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        // Best-effort: the lock is the file's existence, not its contents, so a
        // failed write costs the takeover check and nothing else — and an
        // unreadable owner reads as "still running", which refuses rather than
        // steals.
        let _ = writeln!(file, "pid {}", std::process::id());
        Ok(DirLock {
            path: path.to_path_buf(),
        })
    }
}

/// Whether the process named in a lock file is still alive.
///
/// Unparseable contents count as running: the takeover is the only path that
/// deletes another run's lock, so everything it cannot positively establish has
/// to fall on the refusing side.
fn owner_is_running(contents: &str) -> bool {
    let Some(pid) = contents
        .trim()
        .strip_prefix("pid ")
        .and_then(|rest| rest.trim().parse::<u32>().ok())
    else {
        return true;
    };
    process_is_running(pid)
}

#[cfg(unix)]
fn process_is_running(pid: u32) -> bool {
    // Signal 0 performs the permission and existence checks without delivering
    // anything. `ESRCH` — and only `ESRCH` — means there is no such process;
    // `EPERM` means one exists that we may not signal, which is still alive.
    let alive = unsafe { libc::kill(pid as libc::pid_t, 0) };
    alive == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn process_is_running(_pid: u32) -> bool {
    // No dependency-free liveness check, so a lock is never taken over and the
    // error tells the user which file to delete.
    true
}

impl Drop for DirLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Every well-formed record in `path`, in file order.
///
/// Malformed lines — including the truncated final line of a hard kill — are
/// dropped. Dropping a row only costs a refit; keeping a half-parsed one would
/// cost a wrong answer. A missing file is an empty journal, not an error: that
/// is what a first run looks like.
pub fn read_records(path: &Path) -> Vec<CandidateRecord> {
    let Ok(file) = File::open(path) else {
        return Vec::new();
    };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<CandidateRecord>(&line).ok())
        .collect()
}

/// Read a cached fit, or `None` when it is absent or unreadable.
pub fn load_fit(dir: &Path, hash: &str) -> Option<FitResult> {
    let text = std::fs::read_to_string(fit_path(dir, hash)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Write a fit to the cache, through a `.part` sibling so a reader never sees a
/// half-written file.
fn store_fit(dir: &Path, hash: &str, fit: &FitResult) -> Result<(), String> {
    let final_path = fit_path(dir, hash);
    std::fs::create_dir_all(fits_dir(dir))
        .map_err(|e| format!("cannot create `{}`: {e}", fits_dir(dir).display()))?;
    let temp = part_path(&final_path);
    let text = serde_json::to_string(fit)
        .map_err(|e| format!("cannot serialize the fit of `{hash}`: {e}"))?;
    std::fs::write(&temp, text).map_err(|e| format!("cannot write `{}`: {e}", temp.display()))?;
    std::fs::rename(&temp, &final_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("cannot rename `{}`: {e}", temp.display())
    })
}

/// An append-as-you-go log of finished candidates.
pub struct Journal {
    dir: PathBuf,
    file: Mutex<File>,
    /// The first write failure, reported once by [`Journal::into_result`]
    /// rather than from inside the parallel fit loop, where there is nothing
    /// useful to do with it.
    error: Mutex<Option<String>>,
}

impl Journal {
    /// Open the journal in `dir`, rewritten from `kept`.
    ///
    /// The file is rewritten rather than appended to even on a resume, which is
    /// what makes a truncated trailing row self-healing: the dropped row is
    /// simply absent from `kept`, so the rewritten file is well-formed and the
    /// next append lands on a line boundary.
    pub fn create(dir: &Path, kept: &[CandidateRecord]) -> Result<Journal, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create search directory `{}`: {e}", dir.display()))?;
        let final_path = journal_path(dir);
        let temp = part_path(&final_path);
        {
            let mut out = File::create(&temp)
                .map_err(|e| format!("cannot create `{}`: {e}", temp.display()))?;
            for record in kept {
                write_line(&mut out, record)?;
            }
            out.flush()
                .map_err(|e| format!("cannot flush `{}`: {e}", temp.display()))?;
        }
        std::fs::rename(&temp, &final_path).map_err(|e| {
            let _ = std::fs::remove_file(&temp);
            format!("cannot rename `{}`: {e}", temp.display())
        })?;
        let file = OpenOptions::new()
            .append(true)
            .open(&final_path)
            .map_err(|e| format!("cannot open `{}` for appending: {e}", final_path.display()))?;
        Ok(Journal {
            dir: dir.to_path_buf(),
            file: Mutex::new(file),
            error: Mutex::new(None),
        })
    }

    /// Append one finished candidate, and cache its fit when there is one.
    ///
    /// The fit is written **before** the row, so a journal row claiming
    /// `has_fit` can only be read after the file it names is complete. Failures
    /// are recorded, not returned: the caller is inside the parallel loop and
    /// aborting it would throw away the fits still in flight.
    pub fn append(&self, record: &CandidateRecord, fit: Option<&FitResult>) {
        if let Some(fit) = fit {
            if let Err(e) = store_fit(&self.dir, &record.hash, fit) {
                self.record_error(e);
                return;
            }
        }
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = write_line(&mut *guard, record).and_then(|()| {
            guard
                .flush()
                .map_err(|e| format!("cannot flush the search journal: {e}"))
        }) {
            drop(guard);
            self.record_error(e);
        }
    }

    fn record_error(&self, message: String) {
        let mut slot = match self.error.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if slot.is_none() {
            *slot = Some(message);
        }
    }

    /// Close the journal, surfacing the first write failure seen inside the
    /// parallel loop.
    pub fn into_result(self) -> Result<(), String> {
        match self.error.into_inner() {
            Ok(Some(e)) => Err(e),
            Ok(None) => Ok(()),
            Err(poisoned) => match poisoned.into_inner() {
                Some(e) => Err(e),
                None => Ok(()),
            },
        }
    }
}

fn write_line(out: &mut File, record: &CandidateRecord) -> Result<(), String> {
    let line = serde_json::to_string(record)
        .map_err(|e| format!("cannot serialize the journal row of `{}`: {e}", record.id))?;
    writeln!(out, "{line}").map_err(|e| format!("cannot write the search journal: {e}"))
}

/// What a search directory was created from — the resume compatibility check.
///
/// Reusing a journalled candidate means trusting a *number another process
/// computed*. Its criterion was evaluated under one [`Criterion`](super::Criterion),
/// and its verdict under one [`Strictness`]; reusing rows written under either
/// of those set differently gives a ranking whose entries do not mean the same
/// thing, and nothing downstream could detect it. Same for the dataset: the
/// same candidate fitted to different data is a different fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchManifest {
    /// [`Criterion::label`](super::Criterion::label) — stable across a refactor
    /// of the enum, and readable when a user opens the file to see why their
    /// resume was refused.
    pub criterion: String,
    pub n_starts: usize,
    pub strictness: Strictness,
    pub n_subjects: usize,
    pub n_observations: usize,
    /// SHA-256 over the dataset's contents: every field of every subject, with
    /// covariate maps written in key order so the digest is stable across runs.
    pub data_fingerprint: String,
    /// SHA-256 over [`RunOptions::fit_options`], when the caller supplied them.
    ///
    /// A candidate's identity is the hash of its *model text*, which carries the
    /// `[fit_options]` block — so with `fit_options: None` the fit settings are
    /// already part of the cache key. An override is not: it lives outside
    /// `ModelText`, so resuming a run under a different method, iteration cap,
    /// optimizer, tolerance, seed or covariance setting would reuse scores the
    /// override never produced. `None` here records "no override", which is
    /// itself incompatible with a run that had one.
    pub fit_options_fingerprint: Option<String>,
}

impl SearchManifest {
    pub fn new(options: &RunOptions, data: &Population) -> Self {
        Self {
            criterion: options.criterion.label().to_string(),
            n_starts: options.n_starts.max(1),
            strictness: options.strictness.clone(),
            n_subjects: data.subjects.len(),
            n_observations: data.subjects.iter().map(|s| s.observations.len()).sum(),
            data_fingerprint: data_fingerprint(data),
            fit_options_fingerprint: options.fit_options.as_ref().map(fit_options_fingerprint),
        }
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| format!("cannot serialize the search manifest: {e}"))?;
        std::fs::write(path, text).map_err(|e| format!("cannot write `{}`: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("cannot parse `{}`: {e}", path.display()))
    }

    /// Refuse a resume whose inputs differ from the ones on disk, naming the
    /// field that differs.
    pub fn check_compatible(&self, disk: &SearchManifest, dir: &Path) -> Result<(), String> {
        let refuse = |field: &str, disk: String, now: String| {
            Err(format!(
                "cannot resume the search in `{}`: {field} differs (on disk: {disk}, now: {now}). \
                 Point `--resume` at the directory it was created in, or start a fresh one.",
                dir.display()
            ))
        };
        if disk.criterion != self.criterion {
            return refuse(
                "the ranking criterion",
                disk.criterion.clone(),
                self.criterion.clone(),
            );
        }
        if disk.n_starts != self.n_starts {
            return refuse(
                "the number of starts per candidate",
                disk.n_starts.to_string(),
                self.n_starts.to_string(),
            );
        }
        if disk.strictness != self.strictness {
            return refuse(
                "the strictness gate",
                format!("{:?}", disk.strictness),
                format!("{:?}", self.strictness),
            );
        }
        if disk.fit_options_fingerprint != self.fit_options_fingerprint {
            let describe = |f: &Option<String>| match f {
                Some(hash) => hash.clone(),
                None => "the candidates' own `[fit_options]`".to_string(),
            };
            return refuse(
                "the fit settings",
                describe(&disk.fit_options_fingerprint),
                describe(&self.fit_options_fingerprint),
            );
        }
        if disk.data_fingerprint != self.data_fingerprint
            || disk.n_subjects != self.n_subjects
            || disk.n_observations != self.n_observations
        {
            return refuse(
                "the dataset",
                format!(
                    "{} subjects / {} observations ({})",
                    disk.n_subjects, disk.n_observations, disk.data_fingerprint
                ),
                format!(
                    "{} subjects / {} observations ({})",
                    self.n_subjects, self.n_observations, self.data_fingerprint
                ),
            );
        }
        Ok(())
    }
}

/// SHA-256 over the fit settings that can change a candidate's *result*.
///
/// Taken off the `Debug` rendering of the options as the runner will actually
/// apply them — `quiet()` included — with the four knobs the runner overrides
/// per candidate normalised away, since they are properties of *this* run's
/// scheduling rather than of the numbers it produces:
///
/// * `threads` — the [`PoolPlan`](ferx_core::PoolPlan) share, and pinned per run;
/// * `cancel` — the run's own flag;
/// * `n_starts` — recorded separately as [`SearchManifest::n_starts`];
/// * `user_set_keys` — bookkeeping for the "key not consumed by this method"
///   warning, and order-dependent on how the caller happened to build the
///   options, so including it would refuse resumes that are in fact identical.
///
/// `FitOptions` implements neither `Hash` nor `Serialize`, and its `Debug` holds
/// no maps, so its rendering is both complete and deterministic — a field added
/// to it lands in this digest automatically.
fn fit_options_fingerprint(options: &FitOptions) -> String {
    let mut normalised = options.clone().quiet();
    normalised.threads = None;
    normalised.cancel = None;
    normalised.n_starts = 1;
    normalised.user_set_keys.clear();
    ferx_core::io::hash::sha256_bytes(format!("{normalised:?}").as_bytes())
}

/// SHA-256 over the dataset's **contents**, not its shape.
///
/// The first version hashed subject ids, per-subject observation counts and the
/// DV column, and that is a silent wrong-result path: change a DV value, an
/// observation time, a dose, a censoring flag or a covariate while keeping the
/// ids and the counts, and the fingerprint does not move. A resume then finds
/// every candidate hash in the journal, fits nothing, and returns criteria
/// computed from the *previous* dataset. Nothing downstream could detect it.
///
/// So every field of every subject is hashed. Two properties make that
/// trustworthy rather than approximately trustworthy:
///
/// * **The subject is destructured exhaustively, with no `..`.** A field added
///   to `ferx_core::Subject` stops this file compiling, which forces whoever
///   adds it to decide whether it belongs in the fingerprint. A `..` here would
///   silently un-hash it, which is exactly the failure this function exists to
///   prevent.
/// * **Every map is written in key order.** `HashMap`'s iteration order varies
///   per process, so hashing its `Debug` directly would produce a different
///   digest on every run and refuse every resume.
///
/// `Population::warnings` and `exclusions` are deliberately excluded: they
/// describe what the *reader* did on the way to this population, not what the
/// fit sees, and `exclusions` has already been applied to `subjects`.
fn data_fingerprint(data: &Population) -> String {
    use std::fmt::Write as _;

    let mut buf = String::new();
    let _ = writeln!(buf, "dv={}", data.dv_column);
    let _ = writeln!(buf, "covariate_names={:?}", data.covariate_names);
    let _ = writeln!(buf, "input_columns={:?}", data.input_columns);

    for subject in &data.subjects {
        // No `..`: see the note above — a new `Subject` field must fail to
        // compile here rather than quietly leave the fingerprint blind to it.
        let ferx_core::Subject {
            id,
            doses,
            obs_times,
            obs_raw_times,
            observations,
            obs_cmts,
            covariates,
            dose_covariates,
            obs_covariates,
            pk_only_times,
            pk_only_covariates,
            reset_times,
            reset_covariates,
            cens,
            occasions,
            obs_l2,
            dose_occasions,
            reset_occasions,
            fremtype,
            obs_records,
        } = subject;

        let _ = writeln!(buf, "id={id}");
        let _ = writeln!(buf, "doses={doses:?}");
        let _ = writeln!(buf, "obs_times={obs_times:?}");
        let _ = writeln!(buf, "obs_raw_times={obs_raw_times:?}");
        let _ = writeln!(buf, "observations={observations:?}");
        let _ = writeln!(buf, "obs_cmts={obs_cmts:?}");
        let _ = writeln!(buf, "pk_only_times={pk_only_times:?}");
        let _ = writeln!(buf, "reset_times={reset_times:?}");
        let _ = writeln!(buf, "cens={cens:?}");
        let _ = writeln!(buf, "occasions={occasions:?}");
        let _ = writeln!(buf, "obs_l2={obs_l2:?}");
        let _ = writeln!(buf, "dose_occasions={dose_occasions:?}");
        let _ = writeln!(buf, "reset_occasions={reset_occasions:?}");
        let _ = writeln!(buf, "fremtype={fremtype:?}");
        let _ = writeln!(buf, "obs_records={obs_records:?}");

        write_covariates(&mut buf, "covariates", std::slice::from_ref(covariates));
        write_covariates(&mut buf, "dose_covariates", dose_covariates);
        write_covariates(&mut buf, "obs_covariates", obs_covariates);
        write_covariates(&mut buf, "pk_only_covariates", pk_only_covariates);
        write_covariates(&mut buf, "reset_covariates", reset_covariates);
    }

    ferx_core::io::hash::sha256_bytes(buf.as_bytes())
}

/// Append covariate snapshots in **key order**, so the digest does not depend on
/// `HashMap`'s per-process iteration order.
fn write_covariates(
    buf: &mut String,
    label: &str,
    maps: &[std::collections::HashMap<String, f64>],
) {
    use std::fmt::Write as _;

    for (i, map) in maps.iter().enumerate() {
        let mut entries: Vec<(&String, &f64)> = map.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let _ = write!(buf, "{label}[{i}]=");
        for (name, value) in entries {
            let _ = write!(buf, "{name}:{value:?};");
        }
        buf.push('\n');
    }
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
