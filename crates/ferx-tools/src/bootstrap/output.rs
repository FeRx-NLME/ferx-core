//! The CSV artefacts of a bootstrap run.
//!
//! File names follow PsN so an existing PsN user's habits and downstream scripts
//! carry over. Two deliberate departures, both noted on the functions below:
//! the results table is tidy (one row per parameter) rather than PsN's stacked
//! statistic blocks, and the run-level diagnostics live in their own file rather
//! than as a second block inside `bootstrap_results.csv` — a single CSV with two
//! different header shapes is not machine-readable, which defeats the point of
//! writing it.
//!
//! Row order is consistent across `raw_results`, `included_individuals`,
//! `included_keys` and `sample_keys`, exactly as PsN documents: row *j* concerns
//! the same replicate in every file.

use std::path::Path;

use super::{BootstrapOptions, BootstrapResult};

fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.10}")
    } else {
        String::new()
    }
}

fn opt(v: Option<f64>) -> String {
    v.map(fmt).unwrap_or_default()
}

/// Write every artefact into `dir`, creating it if needed.
pub fn write_all(
    dir: &Path,
    result: &BootstrapResult,
    options: &BootstrapOptions,
) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("cannot create bootstrap directory `{}`: {e}", dir.display()))?;
    write_raw_results(&dir.join("raw_results.csv"), result, options)?;
    write_bootstrap_results(&dir.join("bootstrap_results.csv"), result)?;
    write_diagnostics(&dir.join("bootstrap_diagnostics.csv"), result)?;
    write_all_individuals(&dir.join("all_individuals1.csv"), result)?;
    write_included_individuals(&dir.join("included_individuals1.csv"), result)?;
    write_included_keys(&dir.join("included_keys1.csv"), result)?;
    write_sample_keys(&dir.join("sample_keys1.csv"), result)?;
    if options.dofv {
        write_delta_ofv(&dir.join("delta_ofv.csv"), result)?;
    }
    Ok(())
}

fn writer(path: &Path) -> Result<csv::Writer<std::fs::File>, String> {
    csv::Writer::from_path(path).map_err(|e| format!("cannot write `{}`: {e}", path.display()))
}

fn finish(mut w: csv::Writer<std::fs::File>, path: &Path) -> Result<(), String> {
    w.flush()
        .map_err(|e| format!("cannot flush `{}`: {e}", path.display()))
}

/// One row per fit, **first row the original dataset** (`sample = 0`), as in
/// PsN.
///
/// Carries the termination diagnostics alongside the estimates, not just the
/// estimates: that is what makes `--summarize` possible — the exclusion filters
/// are re-applied to this file rather than baked into it.
pub fn write_raw_results(
    path: &Path,
    result: &BootstrapResult,
    options: &BootstrapOptions,
) -> Result<(), String> {
    let mut w = writer(path)?;
    let mut header = vec![
        "sample".to_string(),
        "minimization_successful".to_string(),
        "estimate_near_boundary".to_string(),
        "covariance_step_successful".to_string(),
        "covariance_step_warnings".to_string(),
        "ofv".to_string(),
        "seconds".to_string(),
    ];
    header.extend(result.parameter_names.iter().cloned());
    if options.keep_covariance {
        header.extend(result.parameter_names.iter().map(|n| format!("se_{n}")));
    }
    if options.dofv {
        header.push("delta_ofv".to_string());
    }
    header.push("error".to_string());
    w.write_record(&header)
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;

    let n_params = result.parameter_names.len();
    let rows = result.original.iter().chain(result.replicates.iter());
    for r in rows {
        let mut row = vec![
            r.index.to_string(),
            (r.converged as u8).to_string(),
            (r.estimate_near_boundary as u8).to_string(),
            (r.covariance_step_successful as u8).to_string(),
            (r.covariance_step_warnings as u8).to_string(),
            fmt(r.ofv),
            format!("{:.3}", r.seconds),
        ];
        for j in 0..n_params {
            row.push(opt(r.estimates.get(j).copied()));
        }
        if options.keep_covariance {
            for j in 0..n_params {
                row.push(opt(r
                    .standard_errors
                    .as_ref()
                    .and_then(|s| s.get(j).copied())
                    .flatten()));
            }
        }
        if options.dofv {
            row.push(opt(r.delta_ofv));
        }
        row.push(r.error.clone().unwrap_or_default());
        w.write_record(&row)
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// The statistics table: one row per parameter.
///
/// PsN stacks each statistic as its own block of rows sharing one header. Tidy
/// is chosen here instead because the file's job is to be read by a script or a
/// data frame, and column names that mean "means" in one block and "bias" twenty
/// rows later cannot be.
pub fn write_bootstrap_results(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let level = result.summary.confidence_level;
    let tail = (100.0 - level) / 2.0;
    let mut w = writer(path)?;
    w.write_record([
        "parameter".to_string(),
        "original".to_string(),
        "mean".to_string(),
        "bias".to_string(),
        "standard.error".to_string(),
        "median".to_string(),
        format!("percentile.{tail}"),
        format!("percentile.{}", 100.0 - tail),
        format!("se.ci.lower.{level}"),
        format!("se.ci.upper.{level}"),
    ])
    .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for p in &result.summary.parameters {
        w.write_record([
            p.name.clone(),
            opt(p.original),
            fmt(p.mean),
            opt(p.bias),
            fmt(p.standard_error),
            fmt(p.median),
            opt(p.ci_percentile.map(|(lo, _)| lo)),
            opt(p.ci_percentile.map(|(_, hi)| hi)),
            opt(p.ci_standard_error.map(|(lo, _)| lo)),
            opt(p.ci_standard_error.map(|(_, hi)| hi)),
        ])
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// Run-level counts and PsN's `diagnostic.means`.
pub fn write_diagnostics(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let mut w = writer(path)?;
    w.write_record(["statistic", "value"])
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    let s = &result.summary;
    let mut rows = vec![
        (
            "samples_requested".to_string(),
            result.replicates.len() as f64,
        ),
        ("samples_completed".to_string(), s.n_completed as f64),
        ("samples_included".to_string(), s.n_included as f64),
        (
            "chi_square_df".to_string(),
            result.n_estimated_parameters as f64,
        ),
    ];
    for (reason, n) in &s.excluded_by {
        rows.push((format!("excluded: {reason}"), *n as f64));
    }
    for (name, value) in &s.diagnostic_means {
        rows.push((format!("mean: {name}"), *value));
    }
    for (name, value) in rows {
        w.write_record([name, fmt(value)])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// Every subject of the original dataset, in dataset order.
pub fn write_all_individuals(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let mut w = writer(path)?;
    w.write_record(["ID"])
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for id in &result.subject_ids {
        w.write_record([id])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// One row per replicate: the IDs drawn, repeats included, in dataset order.
///
/// The IDs are the *original* ones — the `#2` suffix that keeps duplicate copies
/// independent inside the fit is an implementation detail of the replicate
/// population, and writing it here would make the file disagree with
/// `all_individuals1.csv`.
pub fn write_included_individuals(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let mut w = writer(path)?;
    for draw in &result.draws {
        let row: Vec<String> = draw
            .keys
            .iter()
            .map(|&k| result.subject_ids[k].clone())
            .collect();
        w.write_record(&row)
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// One row per replicate: the internal 1..N ordinals of the drawn subjects.
pub fn write_included_keys(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let mut w = writer(path)?;
    for draw in &result.draws {
        let row: Vec<String> = draw.keys.iter().map(|&k| (k + 1).to_string()).collect();
        w.write_record(&row)
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// One row per replicate, one count per **original** subject: how many times it
/// was drawn.
pub fn write_sample_keys(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let n = result.subject_ids.len();
    let mut w = writer(path)?;
    w.write_record(result.subject_ids.iter())
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for draw in &result.draws {
        let row: Vec<String> = draw.counts(n).iter().map(|c| c.to_string()).collect();
        w.write_record(&row)
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// `--dofv` only: bootstrap sample id and `ofv_bs,maxeval=0 − ofv_original`.
pub fn write_delta_ofv(path: &Path, result: &BootstrapResult) -> Result<(), String> {
    let mut w = writer(path)?;
    w.write_record(["sample", "delta_ofv"])
        .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    for r in &result.replicates {
        w.write_record([r.index.to_string(), opt(r.delta_ofv)])
            .map_err(|e| format!("cannot write `{}`: {e}", path.display()))?;
    }
    finish(w, path)
}

/// Read one statistic back out of a `bootstrap_diagnostics.csv`.
///
/// Used by `--summarize` to carry forward the values that describe the *model*
/// rather than the filtering — the chi-square degrees of freedom for the Δofv
/// reference — which are not recoverable from `raw_results.csv` and have not
/// changed just because the exclusion criteria did.
pub fn read_diagnostic(path: &Path, key: &str) -> Option<f64> {
    let mut reader = csv::Reader::from_path(path).ok()?;
    for record in reader.records() {
        let row = record.ok()?;
        if row.get(0)? == key {
            return row.get(1)?.trim().parse().ok();
        }
    }
    None
}

/// Read a `raw_results.csv` back into replicate results.
///
/// The counterpart of [`write_raw_results`], and the whole mechanism behind
/// `--summarize`: because the raw file carries each replicate's *diagnostics*
/// and not only its estimates, the exclusion filters can be re-applied to a
/// finished run under different settings without refitting anything. PsN
/// supports the same move for the same reason.
///
/// Returns `(parameter_names, original, replicates)`; `original` is the
/// `sample = 0` row when the base model was run.
#[allow(clippy::type_complexity)]
pub fn read_raw_results(
    path: &Path,
) -> Result<
    (
        Vec<String>,
        Option<super::ReplicateResult>,
        Vec<super::ReplicateResult>,
    ),
    String,
> {
    let mut reader = csv::Reader::from_path(path)
        .map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
    let header: Vec<String> = reader
        .headers()
        .map_err(|e| format!("cannot read the header of `{}`: {e}", path.display()))?
        .iter()
        .map(str::to_string)
        .collect();

    // Everything between `seconds` and the first `se_`/`delta_ofv`/`error`
    // column is a parameter, so the reader does not need to be told the model.
    let fixed_head = [
        "sample",
        "minimization_successful",
        "estimate_near_boundary",
        "covariance_step_successful",
        "covariance_step_warnings",
        "ofv",
        "seconds",
    ];
    for (i, expected) in fixed_head.iter().enumerate() {
        if header.get(i).map(String::as_str) != Some(expected) {
            return Err(format!(
                "`{}` does not look like a bootstrap raw_results file: expected column {} to be \
                 `{expected}`, found `{}`",
                path.display(),
                i + 1,
                header.get(i).map(String::as_str).unwrap_or("<missing>")
            ));
        }
    }
    let names: Vec<String> = header[fixed_head.len()..]
        .iter()
        .take_while(|h| !h.starts_with("se_") && h.as_str() != "delta_ofv" && h.as_str() != "error")
        .cloned()
        .collect();
    let n_params = names.len();
    let param_start = fixed_head.len();
    let se_start = header
        .iter()
        .position(|h| h.starts_with("se_"))
        .filter(|_| header.iter().filter(|h| h.starts_with("se_")).count() == n_params);
    let dofv_at = header.iter().position(|h| h == "delta_ofv");
    let error_at = header.iter().position(|h| h == "error");

    let num = |row: &csv::StringRecord, i: usize| -> f64 {
        row.get(i)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    };
    let flag = |row: &csv::StringRecord, i: usize| -> bool {
        matches!(row.get(i).map(str::trim), Some("1") | Some("true"))
    };

    let mut original = None;
    let mut replicates = Vec::new();
    for record in reader.records() {
        let row = record.map_err(|e| format!("malformed row in `{}`: {e}", path.display()))?;
        let index = row
            .get(0)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .ok_or_else(|| format!("non-numeric `sample` column in `{}`", path.display()))?;
        let error = error_at
            .and_then(|i| row.get(i))
            .map(str::to_string)
            .filter(|s| !s.is_empty());
        let estimates: Vec<f64> = if error.is_some() {
            Vec::new()
        } else {
            (0..n_params).map(|j| num(&row, param_start + j)).collect()
        };
        let result = super::ReplicateResult {
            index,
            estimates,
            standard_errors: se_start.map(|start| {
                // An empty cell means core reported no SE for that coordinate
                // (an IOV covariance, say). It is not a zero and not a NaN, and
                // it must come back as `None` so the column stays aligned with
                // the names it was written under.
                (0..n_params)
                    .map(|j| num(&row, start + j))
                    .map(|v| v.is_finite().then_some(v))
                    .collect()
            }),
            ofv: num(&row, 5),
            converged: flag(&row, 1),
            estimate_near_boundary: flag(&row, 2),
            covariance_step_successful: flag(&row, 3),
            covariance_step_warnings: flag(&row, 4),
            seconds: num(&row, 6),
            error,
            delta_ofv: dofv_at.map(|i| num(&row, i)).filter(|v| v.is_finite()),
        };
        if index == 0 {
            original = Some(result);
        } else {
            replicates.push(result);
        }
    }
    Ok((names, original, replicates))
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
