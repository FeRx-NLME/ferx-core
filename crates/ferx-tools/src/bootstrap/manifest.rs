//! `bootstrap_run.json` — what a run directory was created from (#1143).
//!
//! Resuming means appending new replicates to files another process wrote. If
//! that other process fitted a *different* model, or drew from a different
//! seed, the resulting `raw_results.csv` is not a bootstrap of anything: the
//! rows would be labelled with one model's parameter names and hold another
//! model's estimates. Nothing downstream could detect it — the file would parse,
//! the statistics would compute, and the intervals would be wrong.
//!
//! So the run records what it was made from, and a resume refuses to proceed
//! when any of it differs, naming the field.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{BootstrapOptions, SampleSize};

/// The inputs and options a bootstrap run directory was created under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub seed: u64,
    pub samples: usize,
    /// [`SampleSize`] rendered as a stable string — see [`describe_sample_size`].
    pub sample_size: String,
    pub stratify_on: Option<String>,
    /// SHA-256 of the model file, as `ferx_core::PreparedRun` computed it.
    pub model_hash: Option<String>,
    /// SHA-256 of the dataset.
    pub data_hash: Option<String>,
    /// Recorded because it changes the *column shape* of `raw_results.csv`:
    /// appending rows of a different width would corrupt the file.
    pub keep_covariance: bool,
    /// Recorded for the same reason — `--dofv` adds a column.
    pub dofv: bool,
    /// The flat parameter vector's names, in order. The last line of defence
    /// when neither file could be hashed.
    pub parameter_names: Vec<String>,
}

/// Render a [`SampleSize`] as the string the manifest stores.
///
/// Written out rather than derived through serde so the on-disk form is stable
/// against a future refactor of the enum, and readable when a user opens the
/// file to see why their resume was refused.
pub fn describe_sample_size(size: &SampleSize) -> String {
    match size {
        SampleSize::Original => "original".to_string(),
        SampleSize::Total(n) => format!("total:{n}"),
        SampleSize::PerStratum(map) => {
            let parts: Vec<String> = map.iter().map(|(k, v)| format!("{k}=>{v}")).collect();
            format!("per_stratum:{}", parts.join(","))
        }
    }
}

impl RunManifest {
    /// The manifest describing the run `options` is about to perform.
    pub fn new(
        options: &BootstrapOptions,
        model_hash: Option<String>,
        data_hash: Option<String>,
        parameter_names: &[String],
    ) -> Self {
        RunManifest {
            seed: options.seed,
            samples: options.samples,
            sample_size: describe_sample_size(&options.sample_size),
            stratify_on: options.stratify_on.clone(),
            model_hash,
            data_hash,
            keep_covariance: options.keep_covariance,
            dofv: options.dofv,
            parameter_names: parameter_names.to_vec(),
        }
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("cannot serialize `{}`: {e}", path.display()))?;
        std::fs::write(path, json + "\n")
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            format!(
                "--resume needs `{}`, written by the run being resumed: {e}",
                path.display()
            )
        })?;
        serde_json::from_str(&text).map_err(|e| format!("cannot parse `{}`: {e}", path.display()))
    }

    /// Refuse a resume whose options or inputs differ from the run on disk.
    ///
    /// `self` is what is being asked for now, `disk` is what the directory was
    /// created under. The first difference wins and is named — a message saying
    /// only "the run does not match" leaves the user to diff two files by hand.
    ///
    /// The two hashes are compared **only when both sides have one**:
    /// `ferx_core::PreparedRun` leaves them `None` when a file could not be
    /// hashed, and refusing a resume because of that would turn a missing check
    /// into a hard failure. `parameter_names` is checked unconditionally and
    /// catches the same class of mistake when it does.
    pub fn check_compatible(&self, disk: &RunManifest, dir: &Path) -> Result<(), String> {
        let refuse = |field: &str, disk_value: String, now: String| {
            Err(format!(
                "--resume: `{}` was created with {field} {disk_value}, but this run asks for \
                 {now}. Resuming would mix two different runs' replicates into one \
                 raw_results.csv. Use a fresh --directory, or drop the conflicting option.",
                dir.display()
            ))
        };
        if self.seed != disk.seed {
            return refuse("--seed", disk.seed.to_string(), self.seed.to_string());
        }
        if self.samples != disk.samples {
            return refuse(
                "--samples",
                disk.samples.to_string(),
                self.samples.to_string(),
            );
        }
        if self.sample_size != disk.sample_size {
            return refuse(
                "--sample-size",
                disk.sample_size.clone(),
                self.sample_size.clone(),
            );
        }
        if self.stratify_on != disk.stratify_on {
            let show = |v: &Option<String>| match v {
                Some(c) => format!("`{c}`"),
                None => "no stratification".to_string(),
            };
            return refuse(
                "--stratify-on",
                show(&disk.stratify_on),
                show(&self.stratify_on),
            );
        }
        if self.keep_covariance != disk.keep_covariance {
            return refuse(
                "--keep-covariance",
                disk.keep_covariance.to_string(),
                self.keep_covariance.to_string(),
            );
        }
        if self.dofv != disk.dofv {
            return refuse("--dofv", disk.dofv.to_string(), self.dofv.to_string());
        }
        if let (Some(a), Some(b)) = (&self.model_hash, &disk.model_hash) {
            if a != b {
                return refuse("a model file whose hash was", b.clone(), a.clone());
            }
        }
        if let (Some(a), Some(b)) = (&self.data_hash, &disk.data_hash) {
            if a != b {
                return refuse("a dataset whose hash was", b.clone(), a.clone());
            }
        }
        if self.parameter_names != disk.parameter_names {
            return refuse(
                "the parameter vector",
                format!("{:?}", disk.parameter_names),
                format!("{:?}", self.parameter_names),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
