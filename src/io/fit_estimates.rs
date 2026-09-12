//! Read the parameter estimates and standard errors of a **previous fit** back
//! off disk (#254 phase 2).
//!
//! This is the input side of `[priors] from_fit = "…"`: the model-updating path
//! where a published or previously-run model becomes the prior for a fit on new,
//! sparse data. The caller ([`crate::estimation::priors`]) turns each estimate
//! into `prior(value, rse = se/value)` — exactly the object a user would have
//! typed by hand off the source run's parameter table.
//!
//! # The one scale rule
//!
//! A [`FitResult`] reports Ω and Ω_IOV as **variances** and Σ as an **SD**,
//! whatever scale the source model *declared* them on — `omega X ~ 0.09` and
//! `omega X ~ 0.3 (sd)` produce byte-identical output. So a source fit's own
//! `(sd)` spelling is invisible here and must not be guessed at; the only
//! declared scale that matters is the **target** model's, which the caller
//! applies. Getting this backwards would be a factor of two on half the priors
//! and would still look plausible, so the scales are stated per field on
//! [`SourceEstimate::value`] and pinned by
//! `omega_declared_as_sd_in_the_source_is_invisible`.
//!
//! # Formats
//!
//! Three, dispatched on the file extension:
//!
//! | Extension | Reader | Precision |
//! |---|---|---|
//! | `.fitrx` | [`crate::io::fitrx::load_fit`] | exact |
//! | `.json` | serde, via [`FitResult`]'s own derives | exact |
//! | `.yaml` / `.yml` | [`parse_estimates_yaml`] | 6 decimals |
//!
//! The YAML one is the curated human-readable file
//! ([`crate::io::output::write_estimates_yaml`]), which is what a plain `ferx
//! model.ferx --data …` run writes and therefore what a user has to hand. It is
//! formatted `{:.6}`, so an estimate below ~1e-4 loses significant figures and
//! one below 5e-7 rounds to zero — which is why a zero or non-finite estimate is
//! reported as an error naming the parameter rather than silently becoming a
//! degenerate prior. Prefer `.fitrx` or `.json` when the numbers are small.

use crate::types::FitResult;
use std::path::Path;

/// Which parameter family a source estimate belongs to.
///
/// Carried so the caller can match a source estimate to a target coordinate on
/// *name and kind*. Name alone is not enough: a source θ called `CL` and a
/// target Ω called `CL` are both legal, and importing the θ's natural value as
/// an Ω variance prior would be silently wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EstimateKind {
    Theta,
    Omega,
    Sigma,
    Kappa,
}

impl EstimateKind {
    /// The `[parameters]` keyword this family is declared with, for diagnostics.
    pub(crate) fn keyword(self) -> &'static str {
        match self {
            EstimateKind::Theta => "theta",
            EstimateKind::Omega => "omega",
            EstimateKind::Sigma => "sigma",
            EstimateKind::Kappa => "kappa",
        }
    }
}

/// One parameter's estimate and standard error, as read from a previous fit.
#[derive(Debug, Clone)]
pub(crate) struct SourceEstimate {
    pub(crate) kind: EstimateKind,
    /// Parameter name as the source model declared it.
    pub(crate) name: String,
    /// The estimate, on the scale a [`FitResult`] reports that family on:
    /// **variance** for `Omega` / `Kappa`, **SD** for `Sigma`, the natural value
    /// for `Theta`. Never the source model's declared scale — see the module
    /// docs.
    pub(crate) value: f64,
    /// Standard error on the same scale as [`Self::value`]. `None` when the
    /// source fit reported none (no covariance step, or a FIXed parameter).
    pub(crate) se: Option<f64>,
}

/// Read every priorable estimate out of the fit file at `path`.
///
/// Dispatches on the extension; an unrecognised one is an error rather than a
/// guess, because reading a `.csv` as YAML would return an empty set and look
/// like "the source model has no parameters".
pub(crate) fn read_fit_estimates(path: &Path) -> Result<Vec<SourceEstimate>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match ext.as_str() {
        "fitrx" => {
            let loaded = crate::io::fitrx::load_fit(path)
                .map_err(|e| format!("failed to read `{}`: {e}", path.display()))?;
            Ok(from_fit_result(&loaded.fit))
        }
        "json" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read `{}`: {e}", path.display()))?;
            let fit: FitResult = serde_json::from_str(&text).map_err(|e| {
                format!(
                    "`{}` is not a ferx fit JSON (written by `--output-format json`): {e}",
                    path.display()
                )
            })?;
            Ok(from_fit_result(&fit))
        }
        "yaml" | "yml" => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read `{}`: {e}", path.display()))?;
            parse_estimates_yaml(&text)
        }
        other => Err(format!(
            "`{}`: unsupported fit file extension `{other}`. Expected one of \
             `.fitrx`, `.json` (`--output-format json`) or `.yaml` (the \
             `{{model}}-fit.yaml` every run writes).",
            path.display()
        )),
    }
}

/// Flatten a [`FitResult`] into the priorable estimates.
///
/// The single derivation shared by the `.fitrx` and `.json` readers, and the
/// reference the YAML reader is round-tripped against.
///
/// **The set is exactly what the fit report lists**, which is why the two θ
/// exclusions below are shared with [`crate::io::output::write_estimates_yaml`]
/// rather than spelled again: a `[covariate_nn]` weight and a large `theta
/// NAME[COL]` level block are generated coefficients, summarized in blocks
/// rather than named one per line, and neither is something a user could have
/// written a `prior(...)` on in `[parameters]` — phase 1 rejects a hand-written
/// prior on a level block for that reason. Importing them from `.json` while the
/// `.yaml` path could not see them at all would make `from_fit` mean two
/// different things depending on which file the user happened to keep.
pub(crate) fn from_fit_result(fit: &FitResult) -> Vec<SourceEstimate> {
    let mut out = Vec::new();

    #[cfg(feature = "nn")]
    let nn_theta: std::collections::HashSet<usize> = fit
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn_theta: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let blocked_theta: std::collections::HashSet<usize> =
        crate::io::output::compact_theta_blocks(&fit.theta_names)
            .into_iter()
            .flat_map(|(_, r)| r)
            .collect();

    for (i, name) in fit.theta_names.iter().enumerate() {
        if nn_theta.contains(&i) || blocked_theta.contains(&i) {
            continue;
        }
        out.push(SourceEstimate {
            kind: EstimateKind::Theta,
            name: name.clone(),
            value: fit.theta.get(i).copied().unwrap_or(f64::NAN),
            // A FIXed θ has no meaningful SE; the writer emits `~` for it and
            // the caller drops it either way, so read it as absent here too
            // rather than letting a stale number through on one path only.
            se: se_of(fit.se_theta.as_ref().and_then(|v| v.get(i).copied()))
                .filter(|_| !fit.theta_fixed.get(i).copied().unwrap_or(false)),
        });
    }

    let n_eta = fit.omega.nrows();
    for i in 0..n_eta {
        out.push(SourceEstimate {
            kind: EstimateKind::Omega,
            name: fit
                .eta_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("omega_{}_{}", i + 1, i + 1)),
            value: fit.omega[(i, i)],
            se: se_of(crate::types::omega_se_at(&fit.se_omega, n_eta, i, i))
                .filter(|_| !fit.omega_fixed.get(i).copied().unwrap_or(false)),
        });
    }

    for (i, &value) in fit.sigma.iter().enumerate() {
        out.push(SourceEstimate {
            kind: EstimateKind::Sigma,
            name: fit
                .sigma_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("sigma_{}", i + 1)),
            value,
            se: se_of(fit.se_sigma.as_ref().and_then(|v| v.get(i).copied()))
                .filter(|_| !fit.sigma_fixed.get(i).copied().unwrap_or(false)),
        });
    }

    if let Some(iov) = fit.omega_iov.as_ref() {
        for i in 0..iov.nrows() {
            out.push(SourceEstimate {
                kind: EstimateKind::Kappa,
                name: fit
                    .kappa_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("kappa_{}", i + 1)),
                value: iov[(i, i)],
                se: se_of(fit.se_kappa.as_ref().and_then(|v| v.get(i).copied()))
                    .filter(|_| !fit.kappa_fixed.get(i).copied().unwrap_or(false)),
            });
        }
    }

    out
}

/// Normalise a reported SE: a non-finite or non-positive one carries no
/// information and is treated as absent.
///
/// `0.0` is the value `se_theta` and friends carry for a coordinate the
/// covariance step could not reach (`idx < n` fails, or the variance came back
/// non-positive), so it is "no SE", not "an infinitely sharp prior".
fn se_of(se: Option<f64>) -> Option<f64> {
    se.filter(|s| s.is_finite() && *s > 0.0)
}

/// Parse the curated `{model}-fit.yaml` written by
/// [`crate::io::output::write_estimates_yaml`].
///
/// Deliberately *not* a general YAML parser — the crate has no YAML dependency
/// and adding one to read a file this crate also writes would be a compile-time
/// cost (#971) for no correctness gain. It reads exactly the four sections it
/// needs, at exactly the indentation the writer emits, and is pinned to the
/// writer by `yaml_round_trip_matches_the_fit_result`.
///
/// Structure it relies on, all of it produced by one `writeln!` each:
///
/// ```text
/// theta:                      ← column 0, a top-level section
///   TVCL:                     ← 2 spaces, an entry
///     estimate: 0.132917      ← 4 spaces, a field
///     se: 0.011000
/// ```
///
/// Entries that lack the section's value key are skipped, which is what excludes
/// the `A__B` off-diagonal entries (`covariance:`) from `omega:` / `omega_iov:`
/// without needing to recognise the `__` spelling.
pub(crate) fn parse_estimates_yaml(text: &str) -> Result<Vec<SourceEstimate>, String> {
    // (section name, the kind it yields, the key holding the estimate).
    // `omega`/`omega_iov` report a variance, `sigma` an SD — the same scales
    // `from_fit_result` produces, which is what makes the round-trip test an
    // oracle rather than a tautology.
    const SECTIONS: &[(&str, EstimateKind, &str)] = &[
        ("theta", EstimateKind::Theta, "estimate"),
        ("omega", EstimateKind::Omega, "variance"),
        ("sigma", EstimateKind::Sigma, "estimate"),
        ("omega_iov", EstimateKind::Kappa, "variance"),
    ];

    let mut out: Vec<SourceEstimate> = Vec::new();
    let mut section: Option<(EstimateKind, &'static str)> = None;
    // The entry being accumulated: (name, estimate, se, fixed).
    let mut entry: Option<(String, Option<f64>, Option<f64>, bool)> = None;

    // Close the open entry, pushing it when the section's value key was seen.
    fn flush(
        out: &mut Vec<SourceEstimate>,
        kind: EstimateKind,
        entry: Option<(String, Option<f64>, Option<f64>, bool)>,
    ) {
        if let Some((name, Some(value), se, fixed)) = entry {
            out.push(SourceEstimate {
                kind,
                name,
                value,
                se: se_of(se).filter(|_| !fixed),
            });
        }
    }

    for raw in text.lines() {
        let line = strip_comment(raw);
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim();

        if indent == 0 {
            // A new top-level section ends whatever entry was open.
            if let Some((kind, _)) = section {
                flush(&mut out, kind, entry.take());
            }
            entry = None;
            let key = trimmed.strip_suffix(':').unwrap_or(trimmed);
            section = SECTIONS
                .iter()
                .find(|(name, _, _)| *name == key)
                .map(|(_, kind, value_key)| (*kind, *value_key));
            continue;
        }

        let Some((kind, value_key)) = section else {
            continue;
        };

        if indent == 2 {
            flush(&mut out, kind, entry.take());
            // A list item (`  - name: …`, as `parameter_priors:` uses) is not an
            // entry; none of the four sections above emits one, but ignoring it
            // keeps a mis-identified section from inventing a parameter called
            // `- name`.
            if let Some(name) = trimmed.strip_suffix(':').filter(|n| !n.starts_with('-')) {
                entry = Some((name.to_string(), None, None, false));
            }
            continue;
        }

        if indent == 4 {
            let Some((key, value)) = trimmed.split_once(':') else {
                continue;
            };
            let Some(e) = entry.as_mut() else { continue };
            let (key, value) = (key.trim(), value.trim());
            if key == value_key {
                e.1 = parse_scalar(value);
            } else if key == "se" {
                e.2 = parse_scalar(value);
            } else if key == "fixed" {
                e.3 = value == "true";
            }
        }
    }
    if let Some((kind, _)) = section {
        flush(&mut out, kind, entry.take());
    }

    if out.is_empty() {
        return Err(
            "no `theta:` / `omega:` / `sigma:` entries found — is this a ferx \
             `{model}-fit.yaml`?"
                .to_string(),
        );
    }
    Ok(out)
}

/// `~` (the writer's null) and anything unparseable read as absent.
fn parse_scalar(value: &str) -> Option<f64> {
    value.parse::<f64>().ok()
}

/// Drop a trailing `#` comment. The writer emits two (`sigma:  # error model: …`
/// and the two `# aic/bic are computed from ofv_data` lines), and the first of
/// those sits on a section header this reader must still recognise.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

#[cfg(test)]
#[path = "fit_estimates_tests.rs"]
mod tests;
