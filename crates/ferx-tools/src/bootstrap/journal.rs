//! Writing the per-replicate files *as the replicates finish* (#1143).
//!
//! Before this, `output::write_all` ran once after every fit was done, so a run
//! killed at replicate 190 of 200 left nothing behind and had to start over. The
//! journal turns the same four per-replicate files into an append-as-you-go log,
//! which is all `--resume` needs: because a draw is a pure function of
//! `(seed, index)` (see [`super::resample::replicate_seed`]), a replicate
//! recorded here is exactly the one a fresh run would have produced, so reusing
//! it is not an approximation.
//!
//! Three properties the implementation has to keep:
//!
//! * **Rows stay aligned across the four files.** `raw_results`,
//!   `included_individuals`, `included_keys` and `sample_keys` are written under
//!   one lock per replicate, so a reader still finds the same replicate at row
//!   *j* of each — #1141's guarantee, now under concurrency.
//! * **Every row is flushed.** An unflushed tail row is a lost fit, and losing
//!   it silently is worse than the microseconds the flush costs against a fit
//!   measured in seconds.
//! * **The order is completion order, not index order.** The fits run on a Rayon
//!   pool, so the journal cannot be sorted. That is fine for recovery — the
//!   reader keys on the `sample` column — and the final `output::write_all`
//!   rewrites all four files in index order, so a *completed* run's artefacts
//!   are byte-identical whatever order its replicates finished in.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::output;
use super::{BootstrapOptions, Replicate, ReplicateResult};

/// The four per-replicate files, open for appending.
struct Files {
    raw: csv::Writer<File>,
    included_individuals: csv::Writer<File>,
    included_keys: csv::Writer<File>,
    sample_keys: csv::Writer<File>,
}

/// An append-as-you-go writer for the per-replicate artefacts.
pub struct Journal {
    files: Mutex<Files>,
    /// The first write failure, reported once by [`Journal::into_result`]
    /// rather than from inside the parallel fit loop, where there is nothing
    /// useful to do with it.
    error: Mutex<Option<String>>,
    n_params: usize,
    subject_ids: Vec<String>,
    /// Kept whole because it is what fixes the *column shape* of a row; only
    /// `keep_covariance` and `dofv` are actually read.
    options: BootstrapOptions,
}

fn append_writer(path: &Path) -> Result<csv::Writer<File>, String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("cannot open `{}` for appending: {e}", path.display()))?;
    Ok(csv::Writer::from_writer(file))
}

fn truncate(path: &Path) -> Result<(), String> {
    File::create(path)
        .map(|_| ())
        .map_err(|e| format!("cannot create `{}`: {e}", path.display()))
}

/// The temp sibling a per-replicate file is rewritten through. It sits in the
/// same directory, so the `rename` that swaps it in is atomic.
fn part_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    path.with_file_name(name)
}

/// Best-effort cleanup of the temp siblings after a failed rewrite. A leftover
/// `.part` file is harmless — the next `create` truncates it — but noisy.
fn remove_parts(paths: &[PathBuf; 4]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

pub fn raw_results_path(dir: &Path) -> PathBuf {
    dir.join("raw_results.csv")
}

pub fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("bootstrap_run.json")
}

impl Journal {
    /// Open the journal in `dir`, seeding it with the replicates being reused.
    ///
    /// The four files are **rewritten from `kept`**, not appended to, even on a
    /// resume. That is what makes a truncated trailing row self-healing: the
    /// dropped row is simply absent from `kept`, so the rewritten file is
    /// well-formed and the next append lands on a record boundary. Appending
    /// straight onto a file whose last line was cut mid-field would splice the
    /// fragment and the new row into one corrupt record.
    ///
    /// The rewrite goes through `.part` siblings that are renamed into place
    /// only once they are complete, so the live files are never the only copy
    /// of the recovery data: a failure or a kill part-way through leaves the
    /// interrupted run's artefacts exactly as they were.
    ///
    /// `draws` must cover every index in `kept` — it always does, because the
    /// draws are recomputed for `1..=samples` before anything is fitted.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        dir: &Path,
        parameter_names: &[String],
        subject_ids: &[String],
        kept_original: Option<&ReplicateResult>,
        kept: &[ReplicateResult],
        draws: &[Replicate],
        options: &BootstrapOptions,
    ) -> Result<Journal, String> {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create bootstrap directory `{}`: {e}", dir.display()))?;

        let final_paths = [
            raw_results_path(dir),
            dir.join("included_individuals1.csv"),
            dir.join("included_keys1.csv"),
            dir.join("sample_keys1.csv"),
        ];
        let temp_paths = [
            part_path(&final_paths[0]),
            part_path(&final_paths[1]),
            part_path(&final_paths[2]),
            part_path(&final_paths[3]),
        ];

        let journal = match Journal::seed(
            &temp_paths,
            parameter_names,
            subject_ids,
            kept_original,
            kept,
            draws,
            options,
        ) {
            Ok(journal) => journal,
            Err(e) => {
                remove_parts(&temp_paths);
                return Err(e);
            }
        };

        for (temp, final_path) in temp_paths.iter().zip(final_paths.iter()) {
            if let Err(e) = std::fs::rename(temp, final_path) {
                remove_parts(&temp_paths);
                return Err(format!(
                    "cannot move `{}` into place at `{}`: {e}",
                    temp.display(),
                    final_path.display()
                ));
            }
        }
        Ok(journal)
    }

    /// Write the headers and the reused rows into `paths` — the temp siblings
    /// [`Journal::create`] renames into place once they are complete.
    ///
    /// The returned journal's handles are the ones opened here: a `rename`
    /// follows the inode, not the name, so they keep appending to the same
    /// files after the swap and never have to be reopened.
    #[allow(clippy::too_many_arguments)]
    fn seed(
        paths: &[PathBuf; 4],
        parameter_names: &[String],
        subject_ids: &[String],
        kept_original: Option<&ReplicateResult>,
        kept: &[ReplicateResult],
        draws: &[Replicate],
        options: &BootstrapOptions,
    ) -> Result<Journal, String> {
        for path in paths {
            truncate(path)?;
        }

        let journal = Journal {
            files: Mutex::new(Files {
                raw: append_writer(&paths[0])?,
                included_individuals: append_writer(&paths[1])?,
                included_keys: append_writer(&paths[2])?,
                sample_keys: append_writer(&paths[3])?,
            }),
            error: Mutex::new(None),
            n_params: parameter_names.len(),
            subject_ids: subject_ids.to_vec(),
            options: options.clone(),
        };

        {
            let mut files = journal.files.lock().expect("journal lock");
            let header = output::raw_results_header(parameter_names, options);
            files
                .raw
                .write_record(&header)
                .map_err(|e| format!("cannot write `{}`: {e}", paths[0].display()))?;
            // `sample_keys1.csv` is the only per-replicate file with a header:
            // one column per original subject.
            files
                .sample_keys
                .write_record(subject_ids)
                .map_err(|e| format!("cannot write `{}`: {e}", paths[3].display()))?;
            // Flush the headers here rather than leaving them to the first
            // `append`. A run killed during the base fit — often the longest
            // single fit — would otherwise leave a `raw_results.csv` that exists
            // but is empty, which `--resume` rejects as a malformed header
            // instead of reporting that there is nothing to resume.
            for flush in [files.raw.flush(), files.sample_keys.flush()] {
                flush.map_err(|e| format!("cannot flush the bootstrap journal: {e}"))?;
            }
        }

        if let Some(original) = kept_original {
            journal.append(original, None);
        }
        for r in kept {
            let draw = draws.iter().find(|d| d.index == r.index);
            journal.append(r, draw);
        }
        journal.take_error().map_or(Ok(()), Err)?;
        Ok(journal)
    }

    /// Record one finished fit. `draw` is `None` for the base model, which has
    /// no row in the three draw files.
    ///
    /// Deliberately infallible from the caller's side: this runs inside the
    /// Rayon fit loop, where propagating an error would either abandon fits that
    /// already succeeded or need a fallible closure through the whole pipeline.
    /// The first failure is stashed and surfaced by [`Journal::into_result`]
    /// once the loop is done.
    pub fn append(&self, result: &ReplicateResult, draw: Option<&Replicate>) {
        let mut files = match self.files.lock() {
            Ok(f) => f,
            Err(_) => return self.record_error("the bootstrap journal lock was poisoned".into()),
        };
        let row = output::raw_results_row(result, self.n_params, &self.options);
        if let Err(e) = files.raw.write_record(&row) {
            return self.record_error(format!("cannot append to raw_results.csv: {e}"));
        }
        if let Some(draw) = draw {
            let individuals = output::included_individuals_row(draw, &self.subject_ids);
            let keys = output::included_keys_row(draw);
            let counts = output::sample_keys_row(draw, self.subject_ids.len());
            if let Err(e) = files.included_individuals.write_record(&individuals) {
                return self
                    .record_error(format!("cannot append to included_individuals1.csv: {e}"));
            }
            if let Err(e) = files.included_keys.write_record(&keys) {
                return self.record_error(format!("cannot append to included_keys1.csv: {e}"));
            }
            if let Err(e) = files.sample_keys.write_record(&counts) {
                return self.record_error(format!("cannot append to sample_keys1.csv: {e}"));
            }
        }
        // Flush every row. Buffering the tail is exactly the fit a hard kill
        // would lose, which is the one this file exists to keep.
        for flush in [
            files.raw.flush(),
            files.included_individuals.flush(),
            files.included_keys.flush(),
            files.sample_keys.flush(),
        ] {
            if let Err(e) = flush {
                return self.record_error(format!("cannot flush the bootstrap journal: {e}"));
            }
        }
    }

    fn record_error(&self, message: String) {
        if let Ok(mut slot) = self.error.lock() {
            slot.get_or_insert(message);
        }
    }

    fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut e| e.take())
    }

    /// Close the journal, surfacing the first write failure if there was one.
    ///
    /// Called before `output::write_all` rewrites the same paths: the journal's
    /// handles must be dropped first, or a truncating rewrite would race the
    /// append offsets they still hold.
    pub fn into_result(self) -> Result<(), String> {
        match self.take_error() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
