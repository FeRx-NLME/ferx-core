//! Write a fit's estimates into the exported model's `[parameters]` declarations
//! (#1331), so [`super::export_dcm_from_fit`] hands back the fitted model and not
//! the fitted network inside the model's starting values.
//!
//! Each declaration keeps the scale it was written on. A `(sd)` tag, and an
//! inline `prior(...)` declared on that scale, mean what they meant before; only
//! the number changes. A [`crate::types::FitResult`] reports Ω and κ as
//! variances and Σ as an SD, so a value written on the other scale goes through
//! a square or a square root, and reads back within a few ULP of the estimate
//! rather than bit for bit.
//!
//! Nothing here is trusted to have found every declaration. The export is parsed
//! again and every estimate of the fit compared with the value the export now
//! declares: a declaration form this pass does not write (a vector θ, a
//! `[mixture]` class override) is an error naming the parameter, not a silent
//! starting value.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

use crate::parser::model_parser::{block_header, strip_line_comment};
use crate::types::{CompiledModel, FitResult, OmegaMatrix};

static THETA_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(\s*theta\s+(\w+)\s*\(\s*)([0-9eE.+-]+)").unwrap());
static SCALAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(\s*(omega|kappa|sigma)\s+(\w+)\s*~\s*)([0-9eE.+-]+)(.*)$").unwrap()
});
static BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(\s*(block_omega|block_kappa|block_sigma)\s*\(([^)]+)\)\s*=\s*\[)([^\]]*)(\].*)$",
    )
    .unwrap()
});
static SD_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i-u)\(\s*sd\s*\)").unwrap());

/// `text` with every `[parameters]` declaration the fit estimates set to the
/// fit's value.
pub(super) fn write_fitted_values(text: &str, fit: &FitResult) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_parameters = false;
    // A `block_*` declaration whose `[` … `]` spans several lines, joined onto
    // one, with the comments of its lines kept after it.
    let mut pending: Option<String> = None;
    let mut comments: Vec<&str> = Vec::new();
    for line in text.lines() {
        let code = strip_line_comment(line);
        if let Some(open) = pending.as_mut() {
            open.push(' ');
            open.push_str(code.trim());
            comments.push(line[code.len()..].trim());
            if code.contains(']') {
                let joined = pending.take().expect("set above");
                out.push_str(&rewrite_line(&joined, fit));
                for c in comments.drain(..).filter(|c| !c.is_empty()) {
                    out.push_str("  ");
                    out.push_str(c);
                }
                out.push('\n');
            }
            continue;
        }
        if let Some((ty, _)) = block_header(code.trim()) {
            in_parameters = ty == "parameters";
        } else if in_parameters {
            if BLOCK_RE.is_match(code) || !is_open_block(code) {
                let comment = &line[code.len()..];
                out.push_str(&rewrite_line(code, fit));
                out.push_str(comment);
                out.push('\n');
            } else {
                pending = Some(code.trim_end().to_string());
                comments.push(line[code.len()..].trim());
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    // An unclosed block cannot parse either; hand it back for the parser to name.
    if let Some(open) = pending {
        out.push_str(&open);
        out.push('\n');
    }
    out
}

/// Does `code` open a `block_*` declaration it does not close?
fn is_open_block(code: &str) -> bool {
    let t = code.trim_start().to_ascii_lowercase();
    t.starts_with("block_") && code.contains('[') && !code.contains(']')
}

fn rewrite_line(code: &str, fit: &FitResult) -> String {
    if let Some(c) = THETA_RE.captures(code) {
        let Some(v) = theta_value(fit, &c[2]) else {
            return code.to_string();
        };
        let whole = c.get(0).expect("match");
        return format!("{}{v:?}{}", &c[1], &code[whole.end()..]);
    }
    if let Some(c) = SCALAR_RE.captures(code) {
        let sd = SD_TAG_RE.is_match(&c[5]);
        let name = &c[3];
        let value = match c[2].to_ascii_lowercase().as_str() {
            "omega" => diag(&fit.omega, &fit.eta_names, name).map(|v| on_scale(v, sd, true)),
            "kappa" => fit
                .omega_iov
                .as_ref()
                .and_then(|m| diag(m, &fit.kappa_names, name))
                .map(|v| on_scale(v, sd, true)),
            _ => sigma_sd(fit, name).map(|s| on_scale(s, sd, false)),
        };
        return match value {
            Some(v) => format!("{}{v:?}{}", &c[1], &c[5]),
            None => code.to_string(),
        };
    }
    if let Some(c) = BLOCK_RE.captures(code) {
        let names: Vec<&str> = c[3].split(',').map(str::trim).collect();
        let kind = c[2].to_ascii_lowercase();
        let mut values = Vec::with_capacity(names.len() * (names.len() + 1) / 2);
        for (r, a) in names.iter().enumerate() {
            for b in &names[..=r] {
                let v = match kind.as_str() {
                    "block_omega" => entry(&fit.omega, &fit.eta_names, a, b),
                    "block_kappa" => fit
                        .omega_iov
                        .as_ref()
                        .and_then(|m| entry(m, &fit.kappa_names, a, b)),
                    _ => sigma_covariance(fit, a, b),
                };
                let Some(v) = v else {
                    return code.to_string();
                };
                values.push(format!("{v:?}"));
            }
        }
        return format!("{}{}{}", &c[1], values.join(", "), &c[5]);
    }
    code.to_string()
}

fn theta_value(fit: &FitResult, name: &str) -> Option<f64> {
    let i = fit.theta_names.iter().position(|n| n == name)?;
    fit.theta.get(i).copied()
}

/// `stored` (a variance, or for Σ an SD) on the declaration's scale.
fn on_scale(stored: f64, sd_tag: bool, stored_is_variance: bool) -> f64 {
    match (sd_tag, stored_is_variance) {
        (true, true) => stored.sqrt(),
        (false, false) => stored * stored,
        _ => stored,
    }
}

fn index(names: &[String], name: &str) -> Option<usize> {
    names.iter().position(|n| n == name)
}

fn diag(m: &nalgebra::DMatrix<f64>, names: &[String], name: &str) -> Option<f64> {
    let i = index(names, name)?;
    (i < m.nrows()).then(|| m[(i, i)])
}

fn entry(m: &nalgebra::DMatrix<f64>, names: &[String], a: &str, b: &str) -> Option<f64> {
    let (i, j) = (index(names, a)?, index(names, b)?);
    (i < m.nrows() && j < m.ncols()).then(|| m[(i, j)])
}

fn sigma_sd(fit: &FitResult, name: &str) -> Option<f64> {
    fit.sigma.get(index(&fit.sigma_names, name)?).copied()
}

/// The `block_sigma` entry for `(a, b)`: a variance on the diagonal, else
/// `rho · sd_a · sd_b`, with `rho = 0` for a pair the fit does not correlate.
fn sigma_covariance(fit: &FitResult, a: &str, b: &str) -> Option<f64> {
    let (i, j) = (index(&fit.sigma_names, a)?, index(&fit.sigma_names, b)?);
    let (si, sj) = (*fit.sigma.get(i)?, *fit.sigma.get(j)?);
    if i == j {
        return Some(si * si);
    }
    let rho = fit
        .residual_correlations
        .iter()
        .find(|c| (c.sigma_i, c.sigma_j) == (i, j) || (c.sigma_i, c.sigma_j) == (j, i))
        .map_or(0.0, |c| c.rho);
    Some(rho * si * sj)
}

/// Equal, or apart by no more than a scale conversion's rounding.
fn close(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a - b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs())
}

/// Confirm `model` — the export, parsed — declares every estimate of `fit`.
/// `weights` are the network θ, which a `Literal` export no longer declares.
pub(super) fn check_fitted_values(
    model: &CompiledModel,
    fit: &FitResult,
    weights: &HashSet<String>,
) -> Result<(), String> {
    let p = &model.default_params;
    let refuse = |what: String| {
        Err(format!(
            "cannot write the fit's {what} into the export; set `network_only` \
             (`ferx export-dcm --network-only`) to keep the model file's values instead"
        ))
    };
    for (name, &want) in fit.theta_names.iter().zip(&fit.theta) {
        match index(&p.theta_names, name) {
            Some(i) if close(p.theta[i], want) => {}
            Some(i) => {
                return refuse(format!(
                    "theta {name} = {want:?} (declared {:?})",
                    p.theta[i]
                ))
            }
            None if weights.contains(name) => {}
            None => {
                return refuse(format!(
                    "theta {name}: the model declares no theta of that name"
                ))
            }
        }
    }
    check_matrix("omega", &p.omega, &fit.omega, &fit.eta_names).or_else(|e| refuse(e))?;
    if let Some(want) = &fit.omega_iov {
        match &p.omega_iov {
            Some(got) => {
                check_matrix("kappa", got, want, &fit.kappa_names).or_else(|e| refuse(e))?
            }
            None => return refuse("kappa variances: the export declares no kappa".into()),
        }
    }
    for (name, &want) in fit.sigma_names.iter().zip(&fit.sigma) {
        match index(&p.sigma.names, name) {
            Some(i) if close(p.sigma.values[i], want) => {}
            _ => return refuse(format!("sigma {name} = {want:?} (SD)")),
        }
    }
    for c in &fit.residual_correlations {
        let (a, b) = (&fit.sigma_names[c.sigma_i], &fit.sigma_names[c.sigma_j]);
        let found = p.residual_correlations.iter().find(|r| {
            let pair = (&p.sigma.names[r.sigma_i], &p.sigma.names[r.sigma_j]);
            pair == (a, b) || pair == (b, a)
        });
        if !found.is_some_and(|r| close(r.rho, c.rho)) {
            return refuse(format!("residual correlation of {a} and {b} = {:?}", c.rho));
        }
    }
    if let Some(m) = &p.mixture {
        if !m.omega_override_addr.is_empty() || !m.sigma_override_addr.is_empty() {
            return refuse("[mixture] per-class omega / sigma: a fit does not report them".into());
        }
    }
    Ok(())
}

fn check_matrix(
    kind: &str,
    got: &OmegaMatrix,
    want: &nalgebra::DMatrix<f64>,
    names: &[String],
) -> Result<(), String> {
    for (i, a) in names.iter().enumerate() {
        for (j, b) in names.iter().enumerate().take(i + 1) {
            let w = want[(i, j)];
            let g = match (index(&got.eta_names, a), index(&got.eta_names, b)) {
                (Some(gi), Some(gj)) => got.matrix[(gi, gj)],
                _ => {
                    return Err(format!(
                        "{kind} {a}: the model declares no {kind} of that name"
                    ))
                }
            };
            if !close(g, w) {
                return Err(if i == j {
                    format!("{kind} {a} = {w:?} (variance)")
                } else {
                    format!("{kind} covariance of {a} and {b} = {w:?}")
                });
            }
        }
    }
    Ok(())
}
