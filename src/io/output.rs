use crate::types::*;

fn fixed_label(name: &str) -> String {
    format!("{} [FIX]", name)
}

/// Summary statistics over an NN's flat weight vector for the compact
/// `neural_networks:` section in the fit YAML / CLI output. Empty input
/// yields all zeros (defensive — shouldn't happen in practice because
/// the parser refuses zero-weight NNs).
#[cfg(feature = "nn")]
fn weight_summary(w: &[f64]) -> (f64, f64, f64, f64) {
    if w.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = w.len() as f64;
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &v in w {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
        sum += v;
    }
    let mean = sum / n;
    let var = w.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mn, mx, mean, var.sqrt())
}

/// Print NONMEM-style results to stderr
pub fn print_results(result: &FitResult) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("NONLINEAR MIXED EFFECTS MODEL ESTIMATION");
    eprintln!("{}", "=".repeat(60));

    eprintln!(
        "\nConverged: {}",
        if result.converged { "YES" } else { "NO" }
    );
    if result.method_chain.len() > 1 {
        let chain: Vec<&str> = result.method_chain.iter().map(|m| m.label()).collect();
        eprintln!("Estimation chain:  {}", chain.join(" → "));
    } else {
        eprintln!("Estimation method: {}", result.method.label());
    }
    eprintln!("Iterations: {}", result.n_iterations);

    eprintln!("\n--- Objective Function ---");
    eprintln!("OFV:  {:.4}", result.ofv);
    eprintln!("AIC:  {:.4}", result.aic);
    eprintln!("BIC:  {:.4}", result.bic);

    eprintln!(
        "\nSubjects: {}  Observations: {}  Parameters: {}",
        result.n_subjects, result.n_obs, result.n_parameters
    );

    // NN weight thetas — listed as a compact summary block below instead
    // of one row per weight (Option E; see plans/dcm-and-low-dim-node.md).
    #[cfg(feature = "nn")]
    let nn_theta_indices: std::collections::HashSet<usize> = result
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn_theta_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Theta estimates
    eprintln!("\n--- THETA Estimates ---");
    eprintln!(
        "{:<16} {:>12} {:>12} {:>10}",
        "Parameter", "Estimate", "SE", "%RSE"
    );
    eprintln!("{}", "-".repeat(52));
    for (i, name) in result.theta_names.iter().enumerate() {
        if nn_theta_indices.contains(&i) {
            continue;
        }
        let est = result.theta[i];
        let is_fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let label = if is_fixed {
            fixed_label(name)
        } else {
            name.clone()
        };
        let (se_str, rse_str) = if is_fixed {
            ("---".to_string(), "---".to_string())
        } else {
            match &result.se_theta {
                Some(se) => {
                    let se_val = se[i];
                    let rse = if est.abs() > 1e-12 {
                        (se_val / est.abs()) * 100.0
                    } else {
                        f64::NAN
                    };
                    (format!("{:.6}", se_val), format!("{:.1}", rse))
                }
                None => ("N/A".to_string(), "N/A".to_string()),
            }
        };
        eprintln!("{:<16} {:>12.6} {:>12} {:>10}", label, est, se_str, rse_str);
    }

    // Compact NN-weight summary block (Option E). Skipped when no
    // `[covariate_nn]` blocks were declared.
    #[cfg(feature = "nn")]
    if !result.neural_networks.is_empty() {
        eprintln!("\n--- NEURAL NETWORKS ---");
        for nn in &result.neural_networks {
            let w = &result.theta[nn.weights_offset..nn.weights_offset + nn.n_weights];
            let (mn, mx, mean, sd) = weight_summary(w);
            let shape: Vec<String> = nn.shape.iter().map(|s| s.to_string()).collect();
            eprintln!(
                "{}  shape=[{}]  activation={}/{}  n_weights={}",
                nn.name,
                shape.join(", "),
                nn.hidden_activation,
                nn.output_activation,
                nn.n_weights,
            );
            eprintln!(
                "  inputs:  [{}]   outputs: [{}]",
                nn.input_names.join(", "),
                nn.output_names.join(", "),
            );
            eprintln!(
                "  weights: min {:.4}  max {:.4}  mean {:.4}  std {:.4}",
                mn, mx, mean, sd
            );
        }
    }

    // Omega estimates
    eprintln!("\n--- OMEGA Estimates ---");
    let n_eta = result.omega.nrows();
    let show_cv = !matches!(
        result.covariance_status,
        CovarianceStatus::Failed | CovarianceStatus::SirFallback
    );
    // Check if omega has off-diagonal elements
    let has_offdiag = (0..n_eta).any(|i| (0..i).any(|j| result.omega[(i, j)].abs() > 1e-15));
    for i in 0..n_eta {
        let var = result.omega[(i, i)];
        let eta_name = result.eta_names.get(i).map(|s| s.as_str()).unwrap_or("ETA");
        let is_fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
        let label = if is_fixed {
            fixed_label(eta_name)
        } else {
            eta_name.to_string()
        };
        let se_str = if is_fixed {
            "---".to_string()
        } else {
            match crate::types::omega_se_at(&result.se_omega, n_eta, i, i) {
                Some(s) => format!("{:.6}", s),
                None => "N/A".to_string(),
            }
        };
        if show_cv {
            let cv = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
            eprintln!(
                "  {:<20} = {:.6}  (CV% = {:.1})  SE = {}",
                label, var, cv, se_str
            );
        } else {
            eprintln!("  {:<20} = {:.6}  SE = {}", label, var, se_str);
        }
    }
    if has_offdiag {
        eprintln!("  --- Correlations ---");
        for i in 0..n_eta {
            for j in 0..i {
                let cov = result.omega[(i, j)];
                if cov.abs() <= 1e-15 {
                    continue;
                }
                let name_i = result.eta_names.get(i).map(|s| s.as_str()).unwrap_or("ETA");
                let name_j = result.eta_names.get(j).map(|s| s.as_str()).unwrap_or("ETA");
                let param_corr = result
                    .omega_param_corr
                    .as_ref()
                    .map(|m| m[(i, j)])
                    .unwrap_or_else(|| {
                        let var_i = result.omega[(i, i)];
                        let var_j = result.omega[(j, j)];
                        if var_i > 0.0 && var_j > 0.0 {
                            cov / (var_i.sqrt() * var_j.sqrt())
                        } else {
                            0.0
                        }
                    });
                let se_cov = crate::types::omega_se_at(&result.se_omega, n_eta, i, j);
                let se_str = match se_cov {
                    Some(s) => format!("  SE = {:.6}", s),
                    None => String::new(),
                };
                eprintln!(
                    "  {} × {} = {:.6}  (param corr = {:.4}){}",
                    name_i, name_j, cov, param_corr, se_str,
                );
            }
        }
    }

    // Sigma estimates
    let err_type = match result.error_model {
        ErrorModel::Additive => "additive",
        ErrorModel::Proportional => "proportional",
        ErrorModel::Combined => "combined",
    };
    eprintln!("\n--- SIGMA Estimates ({}) ---", err_type);
    // Sigma is stored on the SD scale for both proportional and additive
    // components (see src/stats/residual_error.rs). For the proportional
    // component CV% = sigma * 100 directly; for the additive component the
    // value is in observation units and no CV% applies. `result.sigma_types`
    // is already parallel to `result.sigma`.
    for (i, &s) in result.sigma.iter().enumerate() {
        let sig_name = result
            .sigma_names
            .get(i)
            .map(|n| n.as_str())
            .unwrap_or("EPS");
        let is_fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let label = if is_fixed {
            fixed_label(sig_name)
        } else {
            sig_name.to_string()
        };
        let se_str = if is_fixed {
            "---".to_string()
        } else {
            match &result.se_sigma {
                Some(se) if i < se.len() => format!("{:.6}", se[i]),
                _ => "N/A".to_string(),
            }
        };
        match result.sigma_types.get(i).copied() {
            Some(SigmaType::Proportional) => eprintln!(
                "  {:<20} = {:.6}  (CV% = {:.1})  SE = {}",
                label,
                s,
                s * 100.0,
                se_str,
            ),
            _ => eprintln!("  {:<20} = {:.6}  SE = {}", label, s, se_str),
        }
    }

    // IOV (KAPPA) estimates
    if let Some(ref iov) = result.omega_iov {
        eprintln!("\n--- KAPPA (IOV) Estimates ---");
        let n_kappa = iov.nrows();
        for i in 0..n_kappa {
            let var = iov[(i, i)];
            let is_fixed = result.kappa_fixed.get(i).copied().unwrap_or(false);
            let name = result
                .kappa_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("KAPPA");
            let label = if is_fixed {
                fixed_label(name)
            } else {
                name.to_string()
            };
            let se_str = if is_fixed {
                "---".to_string()
            } else {
                match &result.se_kappa {
                    Some(se) if i < se.len() => format!("{:.6}", se[i]),
                    _ => "N/A".to_string(),
                }
            };
            if show_cv {
                let cv = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
                eprintln!(
                    "  {:<20} = {:.6}  (CV% = {:.1})  SE = {}",
                    label, var, cv, se_str
                );
            } else {
                eprintln!("  {:<20} = {:.6}  SE = {}", label, var, se_str);
            }
        }
        // Off-diagonal covariances/correlations (block_kappa)
        let has_offdiag = (0..n_kappa).any(|i| (0..i).any(|j| iov[(i, j)].abs() > 1e-15));
        if has_offdiag {
            eprintln!("  --- Correlations ---");
            for i in 0..n_kappa {
                for j in 0..i {
                    let cov = iov[(i, j)];
                    if cov.abs() <= 1e-15 {
                        continue;
                    }
                    let name_i = result
                        .kappa_names
                        .get(i)
                        .map(|s| s.as_str())
                        .unwrap_or("KAPPA");
                    let name_j = result
                        .kappa_names
                        .get(j)
                        .map(|s| s.as_str())
                        .unwrap_or("KAPPA");
                    let param_corr = result
                        .omega_iov_param_corr
                        .as_ref()
                        .map(|m| m[(i, j)])
                        .unwrap_or_else(|| {
                            let var_i = iov[(i, i)];
                            let var_j = iov[(j, j)];
                            if var_i > 0.0 && var_j > 0.0 {
                                cov / (var_i.sqrt() * var_j.sqrt())
                            } else {
                                0.0
                            }
                        });
                    eprintln!(
                        "  {} × {} = {:.6}  (param corr = {:.4})",
                        name_i, name_j, cov, param_corr,
                    );
                }
            }
        }
    }

    // Importance sampling marginal log-likelihood
    if let Some(ref imp) = result.importance_sampling {
        eprintln!("\n--- Importance Sampling (marginal log-likelihood) ---");
        eprintln!(
            "  -2 log L (IS): {:.4}  (MC SE = {:.4}, K = {}, ν = {})",
            imp.minus2_log_likelihood, imp.mc_standard_error, imp.n_samples, imp.proposal_df,
        );
        eprintln!(
            "  ESS / K: min = {:.3}, median = {:.3}",
            imp.ess_min, imp.ess_median
        );
        match imp.kappa_treatment {
            KappaTreatment::FixedAtMode => {
                eprintln!(
                    "  Note: κ fixed at EBE (partial marginal — not fully comparable to NONMEM IMP)"
                );
            }
            KappaTreatment::Marginalized => {
                eprintln!("  κ marginalised over IS proposal");
            }
            KappaTreatment::NotApplicable => {}
        }
        if !imp.low_ess_subjects.is_empty() {
            eprintln!(
                "  Low-ESS subjects ({}): {}",
                imp.low_ess_subjects.len(),
                imp.low_ess_subjects
                    .iter()
                    .take(5)
                    .map(|(id, frac)| format!("{}={:.2}", id, frac))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    // Bayesian posterior
    if let Some(ref b) = result.bayes {
        eprintln!("\n--- Bayesian posterior (Gibbs-within-HMC) ---");
        eprintln!(
            "  chains = {}, warmup = {}, draws/chain = {}, max R-hat = {:.3}",
            b.n_chains, b.n_warmup, b.n_draws_per_chain, b.max_rhat
        );
        if b.n_divergent > 0 {
            eprintln!("  divergent transitions: {}", b.n_divergent);
        }
        eprintln!(
            "  {:<14} {:>10} {:>10} {:>10} {:>10} {:>6} {:>8}",
            "param", "mean", "sd", "2.5%", "97.5%", "Rhat", "ESS"
        );
        for s in &b.summaries {
            eprintln!(
                "  {:<14} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>6.3} {:>8.0}",
                s.name, s.mean, s.sd, s.q025, s.q975, s.rhat, s.ess_bulk
            );
        }
    }

    // SIR results
    if let Some(ess) = result.sir_ess {
        eprintln!("\n--- SIR Uncertainty (95% CI) ---");
        eprintln!("Effective sample size: {:.1}", ess);
        if let Some(ref ci) = result.sir_ci_theta {
            for (i, name) in result.theta_names.iter().enumerate() {
                if i < ci.len() {
                    eprintln!("  {} : [{:.6}, {:.6}]", name, ci[i].0, ci[i].1);
                }
            }
        }
        if let Some(ref ci) = result.sir_ci_omega {
            let n_eta = result.omega.nrows();
            for i in 0..n_eta.min(ci.len()) {
                let name = result.eta_names.get(i).map(|s| s.as_str()).unwrap_or("ETA");
                eprintln!("  {} : [{:.6}, {:.6}]", name, ci[i].0, ci[i].1);
            }
        }
        if let Some(ref ci) = result.sir_ci_sigma {
            for (i, (lo, hi)) in ci.iter().enumerate() {
                let name = result
                    .sigma_names
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("EPS");
                eprintln!("  {} : [{:.6}, {:.6}]", name, lo, hi);
            }
        }
    }

    // Shrinkage
    if !result.shrinkage_eta.is_empty() {
        eprintln!("\n--- Shrinkage ---");
        for (k, &sh) in result.shrinkage_eta.iter().enumerate() {
            if sh.is_finite() {
                let name = result.eta_names.get(k).map(|s| s.as_str()).unwrap_or("ETA");
                eprintln!("  {} shrinkage: {:.1}%", name, sh * 100.0);
            }
        }
        if result.shrinkage_eps.is_finite() {
            eprintln!("  EPS shrinkage:  {:.1}%", result.shrinkage_eps * 100.0);
        }
    }
    if !result.shrinkage_kappa.is_empty() {
        eprintln!("\n--- Kappa Shrinkage (pooled) ---");
        for (k, &sh) in result.shrinkage_kappa.iter().enumerate() {
            let name = result
                .kappa_names
                .get(k)
                .map(|s| s.as_str())
                .unwrap_or("KAPPA");
            if sh.is_finite() {
                eprintln!("  {} shrinkage: {:.1}%", name, sh * 100.0);
            } else {
                eprintln!("  {} shrinkage: NaN", name);
            }
        }
        if !result.shrinkage_kappa_by_occ.is_empty() {
            eprintln!("  Per-occasion:");
            for (occ_idx, occ_sh) in result.shrinkage_kappa_by_occ.iter().enumerate() {
                let parts: Vec<String> = occ_sh
                    .iter()
                    .enumerate()
                    .map(|(k, &sh)| {
                        let name = result
                            .kappa_names
                            .get(k)
                            .map(|s| s.as_str())
                            .unwrap_or("KAPPA");
                        if sh.is_finite() {
                            format!("{} {:.1}%", name, sh * 100.0)
                        } else {
                            format!("{} NaN", name)
                        }
                    })
                    .collect();
                eprintln!("    Occasion slot {}: {}", occ_idx + 1, parts.join(", "));
            }
        }
    }

    // Run info
    eprintln!("\n--- Run Info ---");
    let cov_str = match result.covariance_status {
        crate::types::CovarianceStatus::Computed => "computed",
        crate::types::CovarianceStatus::Failed => "FAILED",
        crate::types::CovarianceStatus::NotRequested => "not requested",
        crate::types::CovarianceStatus::SirFallback => "SIR fallback",
    };
    eprintln!("  Covariance: {}", cov_str);
    eprintln!("  Wall time:  {:.1}s", result.wall_time_secs);
    eprintln!("  ferx v{}", result.ferx_version);

    // Data selection exclusions
    if let Some(excl) = &result.exclusions {
        eprintln!("\n--- Data Selection ---");
        eprintln!(
            "  Records read: {}  Obs excluded: {}  Doses excluded: {}  Other excluded: {}",
            excl.n_records_total, excl.n_obs_excluded, excl.n_dose_excluded, excl.n_other_excluded
        );
        if !excl.excluded_subject_ids.is_empty() {
            eprintln!(
                "  Subjects excluded entirely: {}",
                excl.excluded_subject_ids.join(", ")
            );
        }
        if !excl.fired_ignore.is_empty() {
            eprintln!("  Fired ignore conditions:");
            for c in &excl.fired_ignore {
                eprintln!("    * {}", c);
            }
        }
        if !excl.fired_accept.is_empty() {
            eprintln!("  Fired accept conditions (failed):");
            for c in &excl.fired_accept {
                eprintln!("    * {}", c);
            }
        }
    }

    // Warnings
    if !result.warnings.is_empty() {
        eprintln!("\n--- Warnings ---");
        for w in &result.warnings {
            eprintln!("  * {}", w);
        }
    }

    eprintln!("{}\n", "=".repeat(60));
}

/// Render a concise, `psn::sumo`-style summary of a completed fit as a
/// `String`: parameter estimates plus basic run information. Unlike
/// [`print_results`] (which streams the full, kitchen-sink report to stderr),
/// this *returns text* so it can be sent to stdout, redirected to a file, or
/// asserted on in a test. It stays focused on the headline numbers a modeller
/// scans first — convergence, objective function, the parameter table with
/// %RSE, key diagnostics (condition number, shrinkage) and run metadata — and
/// is what the `ferx summary <run.fitrx>` subcommand prints.
pub fn format_summary(result: &FitResult) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let bar = "=".repeat(60);

    let _ = writeln!(out, "{bar}");
    let _ = writeln!(out, "ferx run summary — {}", result.model_name);
    let _ = writeln!(out, "{bar}");

    // --- Run status ---
    if result.method_chain.len() > 1 {
        let chain: Vec<&str> = result.method_chain.iter().map(|m| m.label()).collect();
        let _ = writeln!(out, "Method:     {}", chain.join(" → "));
    } else {
        let _ = writeln!(out, "Method:     {}", result.method.label());
    }
    let _ = writeln!(
        out,
        "Converged:  {}",
        if result.converged { "YES" } else { "NO" }
    );
    let _ = writeln!(out, "Iterations: {}", result.n_iterations);
    let _ = writeln!(
        out,
        "Subjects: {}   Observations: {}   Parameters: {}",
        result.n_subjects, result.n_obs, result.n_parameters
    );

    // --- Objective function ---
    let _ = writeln!(out, "\n--- Objective Function ---");
    let _ = writeln!(out, "  OFV:  {:.4}", result.ofv);
    let _ = writeln!(out, "  AIC:  {:.4}", result.aic);
    let _ = writeln!(out, "  BIC:  {:.4}", result.bic);

    // Failed / SIR-fallback covariance make the SE columns meaningless, so we
    // suppress the derived CV% (mirrors `print_results`).
    let show_cv = !matches!(
        result.covariance_status,
        CovarianceStatus::Failed | CovarianceStatus::SirFallback
    );

    // NN weight thetas are excluded from the per-parameter table (they can
    // number in the hundreds); `print_results` skips them the same way.
    #[cfg(feature = "nn")]
    let nn_theta_indices: std::collections::HashSet<usize> = result
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn_theta_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // --- THETA ---
    let _ = writeln!(out, "\n--- THETA ---");
    let _ = writeln!(
        out,
        "  {:<16} {:>12} {:>12} {:>8}",
        "Parameter", "Estimate", "SE", "%RSE"
    );
    for (i, name) in result.theta_names.iter().enumerate() {
        if nn_theta_indices.contains(&i) {
            continue;
        }
        let est = result.theta[i];
        let is_fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let label = if is_fixed {
            fixed_label(name)
        } else {
            name.clone()
        };
        let (se_str, rse_str) = if is_fixed {
            ("---".to_string(), "---".to_string())
        } else {
            match &result.se_theta {
                Some(se) => {
                    let se_val = se[i];
                    let rse = if est.abs() > 1e-12 {
                        (se_val / est.abs()) * 100.0
                    } else {
                        f64::NAN
                    };
                    (format!("{:.6}", se_val), format!("{:.1}", rse))
                }
                None => ("N/A".to_string(), "N/A".to_string()),
            }
        };
        let _ = writeln!(
            out,
            "  {:<16} {:>12.6} {:>12} {:>8}",
            label, est, se_str, rse_str
        );
    }

    // --- OMEGA ---
    let n_eta = result.omega.nrows();
    if n_eta > 0 {
        let _ = writeln!(out, "\n--- OMEGA ---");
        for i in 0..n_eta {
            let var = result.omega[(i, i)];
            let name = result.eta_names.get(i).map(|s| s.as_str()).unwrap_or("ETA");
            let is_fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
            let label = if is_fixed {
                fixed_label(name)
            } else {
                name.to_string()
            };
            let se_str = if is_fixed {
                "---".to_string()
            } else {
                match crate::types::omega_se_at(&result.se_omega, n_eta, i, i) {
                    Some(s) => format!("{:.6}", s),
                    None => "N/A".to_string(),
                }
            };
            if show_cv {
                let cv = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
                let _ = writeln!(
                    out,
                    "  {:<16} = {:.6}  (CV% = {:.1})  SE = {}",
                    label, var, cv, se_str
                );
            } else {
                let _ = writeln!(out, "  {:<16} = {:.6}  SE = {}", label, var, se_str);
            }
        }
        // Off-diagonal correlations (block omega).
        let has_offdiag = (0..n_eta).any(|i| (0..i).any(|j| result.omega[(i, j)].abs() > 1e-15));
        if has_offdiag {
            for i in 0..n_eta {
                for j in 0..i {
                    let cov = result.omega[(i, j)];
                    if cov.abs() <= 1e-15 {
                        continue;
                    }
                    let ni = result.eta_names.get(i).map(|s| s.as_str()).unwrap_or("ETA");
                    let nj = result.eta_names.get(j).map(|s| s.as_str()).unwrap_or("ETA");
                    let corr = result
                        .omega_param_corr
                        .as_ref()
                        .map(|m| m[(i, j)])
                        .unwrap_or_else(|| {
                            let vi = result.omega[(i, i)];
                            let vj = result.omega[(j, j)];
                            if vi > 0.0 && vj > 0.0 {
                                cov / (vi.sqrt() * vj.sqrt())
                            } else {
                                0.0
                            }
                        });
                    let _ = writeln!(out, "  corr({}, {}) = {:.4}", ni, nj, corr);
                }
            }
        }
    }

    // --- SIGMA ---
    let err_type = match result.error_model {
        ErrorModel::Additive => "additive",
        ErrorModel::Proportional => "proportional",
        ErrorModel::Combined => "combined",
    };
    let _ = writeln!(out, "\n--- SIGMA ({}) ---", err_type);
    for (i, &s) in result.sigma.iter().enumerate() {
        let name = result
            .sigma_names
            .get(i)
            .map(|n| n.as_str())
            .unwrap_or("EPS");
        let is_fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let label = if is_fixed {
            fixed_label(name)
        } else {
            name.to_string()
        };
        let se_str = if is_fixed {
            "---".to_string()
        } else {
            match &result.se_sigma {
                Some(se) if i < se.len() => format!("{:.6}", se[i]),
                _ => "N/A".to_string(),
            }
        };
        match result.sigma_types.get(i).copied() {
            Some(SigmaType::Proportional) => {
                let _ = writeln!(
                    out,
                    "  {:<16} = {:.6}  (CV% = {:.1})  SE = {}",
                    label,
                    s,
                    s * 100.0,
                    se_str
                );
            }
            _ => {
                let _ = writeln!(out, "  {:<16} = {:.6}  SE = {}", label, s, se_str);
            }
        }
    }

    // --- Diagnostics ---
    let _ = writeln!(out, "\n--- Diagnostics ---");
    let cov_str = match result.covariance_status {
        CovarianceStatus::Computed => "computed",
        CovarianceStatus::Failed => "FAILED",
        CovarianceStatus::NotRequested => "not requested",
        CovarianceStatus::SirFallback => "SIR fallback",
    };
    let _ = writeln!(out, "  Covariance: {}", cov_str);
    if let Some(cn) = result.cov_condition_number {
        let _ = writeln!(out, "  Condition number: {:.1}", cn);
    }
    if !result.shrinkage_eta.is_empty() {
        let parts: Vec<String> = result
            .shrinkage_eta
            .iter()
            .enumerate()
            .filter(|(_, sh)| sh.is_finite())
            .map(|(k, sh)| {
                let name = result.eta_names.get(k).map(|s| s.as_str()).unwrap_or("ETA");
                format!("{} {:.1}%", name, sh * 100.0)
            })
            .collect();
        if !parts.is_empty() {
            let _ = writeln!(out, "  Eta shrinkage: {}", parts.join(", "));
        }
    }
    if result.shrinkage_eps.is_finite() {
        let _ = writeln!(out, "  EPS shrinkage: {:.1}%", result.shrinkage_eps * 100.0);
    }

    // --- Run info ---
    let _ = writeln!(out, "\n--- Run Info ---");
    let _ = writeln!(out, "  Wall time: {:.1}s", result.wall_time_secs);
    let _ = writeln!(out, "  ferx v{}", result.ferx_version);
    if !result.warnings.is_empty() {
        let _ = writeln!(out, "  Warnings: {}", result.warnings.len());
        for w in &result.warnings {
            let _ = writeln!(out, "    * {}", w);
        }
    }
    let _ = writeln!(out, "{bar}");

    out
}

/// Generate SDTAB-like output as vectors of (header, values) pairs
pub fn sdtab(result: &FitResult, population: &Population) -> Vec<(String, Vec<f64>)> {
    let n_total: usize = result.subjects.iter().map(|s| s.ipred.len()).sum();

    let any_cens = result
        .subjects
        .iter()
        .any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = population.subjects.iter().any(|s| !s.occasions.is_empty());
    let any_multicmt = population
        .subjects
        .iter()
        .any(|s| s.obs_cmts.iter().any(|&c| c != 1));

    let mut ids = Vec::with_capacity(n_total);
    let mut times = Vec::with_capacity(n_total);
    let mut dvs = Vec::with_capacity(n_total);
    let mut cens_col = Vec::with_capacity(n_total);
    let mut occ_col = Vec::with_capacity(n_total);
    let mut cmt_col = Vec::with_capacity(n_total);
    let mut preds = Vec::with_capacity(n_total);
    let mut ipreds = Vec::with_capacity(n_total);
    let mut cwres_vec = Vec::with_capacity(n_total);
    let mut iwres_vec = Vec::with_capacity(n_total);
    let mut npde_vec = Vec::with_capacity(n_total);
    let mut npd_vec = Vec::with_capacity(n_total);
    let mut ebe_ofv_col = Vec::with_capacity(n_total);
    let mut n_obs_col = Vec::with_capacity(n_total);
    // NPDE/NPD are only computed when `[fit_options] npde_nsim > 0`; emit the
    // columns only when at least one subject carries them.
    let any_npde = result
        .subjects
        .iter()
        .any(|s| !s.npde.is_empty() || !s.npd.is_empty());

    for (si, sr) in result.subjects.iter().enumerate() {
        let subj = &population.subjects[si];
        for j in 0..sr.ipred.len() {
            ids.push(sr.id.parse::<f64>().unwrap_or(si as f64 + 1.0));
            // Report the raw data TIME (so sdtab joins back to the input CSV and
            // to covtab, which is also raw); `obs_times` may be the internal
            // shifted timeline for stacked reset occasions. Falls back to
            // `obs_times` when no raw vector was recorded (in-memory subjects).
            times.push(
                subj.obs_raw_times
                    .get(j)
                    .copied()
                    .unwrap_or(subj.obs_times[j]),
            );
            dvs.push(subj.observations[j]);
            cens_col.push(sr.cens.get(j).copied().unwrap_or(0) as f64);
            occ_col.push(subj.occasions.get(j).copied().unwrap_or(0) as f64);
            cmt_col.push(subj.obs_cmts.get(j).copied().unwrap_or(1) as f64);
            preds.push(sr.pred[j]);
            ipreds.push(sr.ipred[j]);
            cwres_vec.push(sr.cwres[j]);
            iwres_vec.push(sr.iwres[j]);
            if any_npde {
                npde_vec.push(sr.npde.get(j).copied().unwrap_or(f64::NAN));
                npd_vec.push(sr.npd.get(j).copied().unwrap_or(f64::NAN));
            }
            ebe_ofv_col.push(sr.ofv_contribution);
            n_obs_col.push(sr.n_obs as f64);
        }
    }

    let mut cols = vec![
        ("ID".to_string(), ids),
        ("TIME".to_string(), times),
        ("DV".to_string(), dvs),
    ];
    if any_cens {
        cols.push(("CENS".to_string(), cens_col));
    }
    if any_occ {
        cols.push(("OCC".to_string(), occ_col));
    }
    if any_multicmt {
        cols.push(("CMT".to_string(), cmt_col));
    }
    cols.extend([
        ("PRED".to_string(), preds),
        ("IPRED".to_string(), ipreds),
        ("CWRES".to_string(), cwres_vec),
        ("IWRES".to_string(), iwres_vec),
    ]);
    if any_npde {
        cols.push(("NPDE".to_string(), npde_vec));
        cols.push(("NPD".to_string(), npd_vec));
    }
    cols.extend([
        ("EBE_OFV".to_string(), ebe_ofv_col),
        ("N_OBS".to_string(), n_obs_col),
    ]);

    // NOTE: sdtab intentionally does NOT emit ETA1..ETAn columns. Per-subject
    // EBEs live in `fit$ebe_etas` on the R side; sdtab is strictly
    // per-observation diagnostic data (see tests/map_estimates_outputs.rs::
    // sdtab_omits_eta_columns_after_fit).

    // TAFD — mandatory, computed from dose records. Measured per reset-occasion
    // (`occasion_first_dose_time`) so stacked occasions each restart their TAFD;
    // `obs_times` is the internal clock, so the difference is offset-invariant
    // and equals the raw time-after-first-dose.
    {
        let vals: Vec<f64> = result
            .subjects
            .iter()
            .zip(population.subjects.iter())
            .flat_map(|(_sr, subj)| {
                subj.obs_times.iter().map(move |&t| {
                    let first_dose = subj.occasion_first_dose_time(t);
                    if first_dose.is_finite() {
                        t - first_dose
                    } else {
                        f64::NAN
                    }
                })
            })
            .collect();
        cols.push(("TAFD".to_string(), vals));
    }

    // TAD — mandatory, SS-aware. When compute_extra_output_columns has run
    // (lagtime models or models with [derived]/[output] blocks), per_obs_tad
    // already reflects the individual lagtime; fall back to lagtime=0 otherwise.
    {
        let vals: Vec<f64> = result
            .subjects
            .iter()
            .zip(population.subjects.iter())
            .flat_map(|(sr, subj)| {
                (0..sr.ipred.len()).map(|j| {
                    if !sr.per_obs_tad.is_empty() {
                        return sr.per_obs_tad[j];
                    }
                    let obs_t = subj.obs_times[j];
                    let last_eff = subj
                        .doses
                        .iter()
                        .filter(|d| d.time <= obs_t + 1e-12)
                        .map(|d| {
                            if d.ss && d.ii > 0.0 {
                                let elapsed = obs_t - d.time;
                                obs_t - elapsed.rem_euclid(d.ii)
                            } else {
                                d.time
                            }
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    if last_eff.is_finite() {
                        obs_t - last_eff
                    } else {
                        f64::NAN
                    }
                })
            })
            .collect();
        cols.push(("TAD".to_string(), vals));
    }

    // extra_columns from [derived] and [output] blocks.
    if let Some(first_with_extra) = result
        .subjects
        .iter()
        .find(|sr| !sr.extra_columns.is_empty())
    {
        let extra_names: Vec<String> = first_with_extra
            .extra_columns
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        for col_name in &extra_names {
            let vals: Vec<f64> = result
                .subjects
                .iter()
                .flat_map(|sr| {
                    sr.extra_columns
                        .iter()
                        .find(|(n, _)| n == col_name)
                        .map(|(_, v)| v.as_slice())
                        .unwrap_or(&[])
                        .to_vec()
                })
                .collect();
            cols.push((col_name.clone(), vals));
        }
    }

    cols
}

/// Write SDTAB as a CSV file
pub fn write_sdtab_csv(
    result: &FitResult,
    population: &Population,
    path: &str,
) -> Result<(), String> {
    let cols = sdtab(result, population);
    if cols.is_empty() {
        return Err("No data to write".to_string());
    }

    let n_rows = cols[0].1.len();
    let mut f =
        std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

    use std::io::Write;

    // Header
    let header: Vec<&str> = cols.iter().map(|(name, _)| name.as_str()).collect();
    writeln!(f, "{}", header.join(",")).map_err(|e| e.to_string())?;

    // Rows. Any non-finite value — NaN (BLOQ IWRES/CWRES) or ±Inf (e.g. from a
    // derived column) — is written as an empty cell so downstream tools handle
    // it as missing rather than a sentinel. This is intentionally broader than
    // the covariate-table writer below (which only blanks NaN via fmt_num), so
    // adopting fmt_num here would silently change derived-column output.
    for row in 0..n_rows {
        let vals: Vec<String> = cols
            .iter()
            .map(|(_, values)| {
                let v = values[row];
                if !v.is_finite() {
                    String::new()
                } else {
                    format!("{:.6}", v)
                }
            })
            .collect();
        writeln!(f, "{}", vals.join(",")).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Write the covariate table (from a `[covariates]` block) as a CSV file.
///
/// Columns: `ID, TIME, EVID, <declared covariates...>`, one row per input
/// dataset record. Missing values (`f64::NAN`) are written as empty cells so
/// downstream tools read them as missing. Uses the `csv` writer so a subject ID
/// (a free-form string) containing a comma or quote is properly escaped rather
/// than corrupting column alignment.
pub fn write_covtab_csv(table: &crate::types::CovariateTable, path: &str) -> Result<(), String> {
    let mut wtr = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|e| format!("Failed to create {}: {}", path, e))?;

    let mut header: Vec<String> = vec!["ID".into(), "TIME".into(), "EVID".into()];
    header.extend(table.names.iter().cloned());
    wtr.write_record(&header).map_err(|e| e.to_string())?;

    for row in &table.rows {
        let mut rec: Vec<String> = Vec::with_capacity(3 + row.values.len());
        rec.push(row.id.clone());
        rec.push(fmt_num(row.time));
        rec.push(row.evid.to_string());
        rec.extend(row.values.iter().map(|&v| fmt_num(v)));
        wtr.write_record(&rec).map_err(|e| e.to_string())?;
    }

    wtr.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write the SAEM conditional-distribution summary (`saem_conddist = true`,
/// #257) as a long-format CSV: one row per subject per random effect.
///
/// Columns: `ID, ETA, COND_MEAN, COND_SD, COND_MODE` — the conditional mean and
/// SD from the MCMC pass (`p(η_i | y_i; θ̂)`) alongside the conditional **mode**
/// (the EBE, `SubjectResult.eta`) for direct comparison. This is the ferx
/// analogue of saemix's `cond.mean.phi` / `cond.sd` and Monolix's
/// `*_mean` / `*_sd` columns. Returns `Err` if the fit carries no
/// conditional-distribution results.
pub fn write_conddist_csv(result: &FitResult, path: &str) -> Result<(), String> {
    let cd = result
        .cond_dist
        .as_ref()
        .ok_or("fit has no conditional-distribution results (saem_conddist was not enabled)")?;

    let mut wtr = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|e| format!("Failed to create {}: {}", path, e))?;
    wtr.write_record(["ID", "ETA", "COND_MEAN", "COND_SD", "COND_MODE"])
        .map_err(|e| e.to_string())?;

    for (i, subj) in result.subjects.iter().enumerate() {
        let mean_i = cd.cond_mean.get(i);
        let sd_i = cd.cond_sd.get(i);
        for (j, eta_name) in result.eta_names.iter().enumerate() {
            let mean = mean_i.and_then(|m| m.get(j)).copied().unwrap_or(f64::NAN);
            let sd = sd_i.and_then(|s| s.get(j)).copied().unwrap_or(f64::NAN);
            // The conditional mode is the EBE already on the subject result.
            let mode = subj.eta.get(j).copied().unwrap_or(f64::NAN);
            wtr.write_record([
                subj.id.clone(),
                eta_name.clone(),
                fmt_num(mean),
                fmt_num(sd),
                fmt_num(mode),
            ])
            .map_err(|e| e.to_string())?;
        }
    }

    wtr.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write the retained per-subject conditional-distribution draws
/// (`saem_conddist_keep_samples = true`) as a wide CSV: one row per draw.
///
/// Columns: `ID, SAMPLE, <eta_names...>`. The analogue of Monolix's
/// `simulatedRandomEffects.txt`. Returns `Err` if no samples were retained.
pub fn write_conddist_samples_csv(result: &FitResult, path: &str) -> Result<(), String> {
    let cd = result
        .cond_dist
        .as_ref()
        .ok_or("fit has no conditional-distribution results (saem_conddist was not enabled)")?;
    if cd.samples.iter().all(|s| s.is_empty()) {
        return Err(
            "no conditional-distribution draws retained (saem_conddist_keep_samples was false)"
                .to_string(),
        );
    }

    let mut wtr = csv::WriterBuilder::new()
        .from_path(path)
        .map_err(|e| format!("Failed to create {}: {}", path, e))?;
    let mut header: Vec<String> = vec!["ID".into(), "SAMPLE".into()];
    header.extend(result.eta_names.iter().cloned());
    wtr.write_record(&header).map_err(|e| e.to_string())?;

    for (i, subj) in result.subjects.iter().enumerate() {
        let Some(draws) = cd.samples.get(i) else {
            continue;
        };
        for (s, draw) in draws.iter().enumerate() {
            let mut rec: Vec<String> = Vec::with_capacity(2 + draw.len());
            rec.push(subj.id.clone());
            rec.push((s + 1).to_string());
            rec.extend(draw.iter().map(|&v| fmt_num(v)));
            wtr.write_record(&rec).map_err(|e| e.to_string())?;
        }
    }

    wtr.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Emit the conditional-distribution CSV outputs (#257) for a completed fit,
/// returning the human-facing status/warning lines for the caller to print.
///
/// Returns an empty `Vec` when the fit carries no conditional-distribution
/// results (the pass was not enabled). Otherwise it always attempts
/// `{model}-conddist.csv`, and writes `{model}-conddist-samples.csv` only when
/// draws were retained — surfacing any real write error rather than swallowing
/// it. Pulling this out of the CLI `main()` keeps the dispatch unit-testable.
pub fn write_conddist_outputs(result: &FitResult, model_name: &str) -> Vec<String> {
    let mut messages = Vec::new();
    let Some(cd) = result.cond_dist.as_ref() else {
        return messages;
    };

    let cd_path = format!("{}-conddist.csv", model_name);
    match write_conddist_csv(result, &cd_path) {
        Ok(()) => messages.push(format!("Conditional distribution written to {}", cd_path)),
        Err(e) => messages.push(format!("Warning: failed to write conddist: {}", e)),
    }

    // Raw draws — only write when samples were retained; skip quietly otherwise.
    if !cd.samples.iter().all(|s| s.is_empty()) {
        let samples_path = format!("{}-conddist-samples.csv", model_name);
        match write_conddist_samples_csv(result, &samples_path) {
            Ok(()) => messages.push(format!(
                "Conditional-distribution draws written to {}",
                samples_path
            )),
            Err(e) => messages.push(format!("Warning: failed to write conddist draws: {}", e)),
        }
    }

    messages
}

/// Format a numeric cell for CSV output: NaN → empty (missing), else 6 dp.
fn fmt_num(v: f64) -> String {
    if v.is_nan() {
        String::new()
    } else {
        format!("{:.6}", v)
    }
}

/// Format a covariance matrix entry in YAML 1.1-compatible scientific notation.
///
/// Rust's `{:.6e}` omits the `+` sign on non-negative exponents (e.g. `1.0` →
/// `"1.000000e0"`), which YAML 1.1 parsers (R yaml, PyYAML) reject as a float
/// and read back as a string.  This helper always emits an explicit sign:
/// `"1.000000e+0"`, `"1.234567e-4"`, etc.
fn fmt_cov_entry(v: f64) -> String {
    let s = format!("{:.6e}", v);
    // Insert '+' before a bare non-negative exponent, e.g. "e0" → "e+0".
    if let Some(pos) = s.find('e') {
        let exp = &s[pos + 1..];
        if !exp.starts_with('-') {
            return format!("{}e+{}", &s[..pos], exp);
        }
    }
    s
}

/// Build the ordered parameter name list that matches `pack_params` layout:
/// `[theta..., omega_packed..., sigma..., kappa_packed...]`.
///
/// For a diagonal omega/kappa each entry is `log_chol_{eta_name}` — the packed
/// value is `log(L_ii)` where `omega = L Lᵀ` (Cholesky diagonal, log-transformed).
/// For a full-block omega/kappa the column-major lower-triangle entries are
/// `log_chol_{eta_i}` on the diagonal (`log(L_ii)`) and `chol_{eta_i}_{eta_j}`
/// (i > j) off-diagonal (`L_ij`, not log-transformed).
fn packed_param_names(result: &FitResult, n: usize) -> Vec<String> {
    let n_theta = result.theta_names.len();
    let n_eta = result.omega.nrows();
    let n_sigma = result.sigma_names.len();
    let n_kappa = result.kappa_names.len();

    let n_omega_diag = n_eta;
    let n_omega_full = n_eta * (n_eta + 1) / 2;
    let n_kappa_diag = n_kappa;
    let n_kappa_full = if n_kappa > 0 {
        n_kappa * (n_kappa + 1) / 2
    } else {
        0
    };
    let n_remaining = n.saturating_sub(n_theta + n_sigma);

    // Try all four diagonal/block combinations; take the first match.
    //
    // Known ambiguity: when n_eta == n_kappa > 1 and the two structures differ
    // (one diagonal, one full-block), both (false, true) and (true, false)
    // yield the same n_remaining and the code always picks (false, true) —
    // full-block omega + diagonal kappa.  In that rare case the packed names
    // will be wrong for the minority of models where kappa is full-block while
    // omega is diagonal.  A proper fix requires storing `omega_is_block` /
    // `kappa_is_block` on `FitResult` so the output layer doesn't have to
    // infer the structure.
    let combos = [
        (true, true, n_omega_diag + n_kappa_diag),
        (false, true, n_omega_full + n_kappa_diag),
        (true, false, n_omega_diag + n_kappa_full),
        (false, false, n_omega_full + n_kappa_full),
    ];
    let (omega_diagonal, kappa_diagonal) = combos
        .iter()
        .find(|(_, _, size)| *size == n_remaining)
        .map(|(od, kd, _)| (*od, *kd))
        .unwrap_or((n_eta <= 1, n_kappa <= 1));

    let mut names: Vec<String> = Vec::with_capacity(n);

    names.extend(result.theta_names.iter().cloned());

    if omega_diagonal {
        for name in &result.eta_names {
            names.push(format!("log_chol_{name}"));
        }
    } else {
        for j in 0..n_eta {
            for i in j..n_eta {
                if i == j {
                    names.push(format!("log_chol_{}", result.eta_names[i]));
                } else {
                    names.push(format!(
                        "chol_{}_{}",
                        result.eta_names[i], result.eta_names[j]
                    ));
                }
            }
        }
    }

    names.extend(result.sigma_names.iter().cloned());

    if n_kappa > 0 {
        if kappa_diagonal {
            for name in &result.kappa_names {
                names.push(format!("log_chol_{name}"));
            }
        } else {
            for j in 0..n_kappa {
                for i in j..n_kappa {
                    if i == j {
                        names.push(format!("log_chol_{}", result.kappa_names[i]));
                    } else {
                        names.push(format!(
                            "chol_{}_{}",
                            result.kappa_names[i], result.kappa_names[j]
                        ));
                    }
                }
            }
        }
    }

    // Pad with generic names if the covariance matrix is larger than expected
    // (shouldn't happen in practice, but prevents index-out-of-bounds).
    while names.len() < n {
        names.push(format!("param_{}", names.len() + 1));
    }
    names.truncate(n);
    names
}

/// Write parameter estimates and uncertainty as YAML
pub fn write_estimates_yaml(result: &FitResult, path: &str) -> Result<(), String> {
    use std::io::Write;

    let mut f =
        std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

    writeln!(f, "model:").map_err(|e| e.to_string())?;
    writeln!(f, "  converged: {}", result.converged).map_err(|e| e.to_string())?;
    writeln!(f, "  method: {}", result.method.label()).map_err(|e| e.to_string())?;
    if result.method_chain.len() > 1 {
        let chain: Vec<&str> = result.method_chain.iter().map(|m| m.label()).collect();
        writeln!(f, "  method_chain: [{}]", chain.join(", ")).map_err(|e| e.to_string())?;
    }
    writeln!(f, "  uses_ode_solver: {}", result.uses_ode_solver).map_err(|e| e.to_string())?;
    writeln!(f, "  uses_sde: {}", result.uses_sde).map_err(|e| e.to_string())?;
    if let Some(n_hmc) = result.saem_n_subjects_hmc {
        let n_mh = result.n_subjects.saturating_sub(n_hmc);
        writeln!(f, "  saem_n_subjects_hmc: {n_hmc}").map_err(|e| e.to_string())?;
        writeln!(f, "  saem_n_subjects_mh: {n_mh}").map_err(|e| e.to_string())?;
    }
    // Effective seed of the NPDE/NPD simulation (only when those diagnostics
    // ran), so the sdtab NPDE/NPD columns can be regenerated exactly.
    if let Some(seed) = result.npde_seed {
        writeln!(f, "  npde_seed: {seed}").map_err(|e| e.to_string())?;
    }

    writeln!(f, "\nobjective_function:").map_err(|e| e.to_string())?;
    writeln!(f, "  ofv: {:.6}", result.ofv).map_err(|e| e.to_string())?;
    writeln!(f, "  aic: {:.6}", result.aic).map_err(|e| e.to_string())?;
    writeln!(f, "  bic: {:.6}", result.bic).map_err(|e| e.to_string())?;

    writeln!(f, "\ndata:").map_err(|e| e.to_string())?;
    writeln!(f, "  n_subjects: {}", result.n_subjects).map_err(|e| e.to_string())?;
    writeln!(f, "  n_observations: {}", result.n_obs).map_err(|e| e.to_string())?;
    writeln!(f, "  n_parameters: {}", result.n_parameters).map_err(|e| e.to_string())?;

    // Identify indices that belong to a `[covariate_nn]` weight block so we
    // can skip them in the per-theta listing below and emit them in a
    // compact `neural_networks:` block instead. Empty set when no NN blocks
    // are present (Option E: see plans/dcm-and-low-dim-node.md).
    #[cfg(feature = "nn")]
    let nn_theta_indices: std::collections::HashSet<usize> = result
        .neural_networks
        .iter()
        .flat_map(|nn| nn.weights_offset..nn.weights_offset + nn.n_weights)
        .collect();
    #[cfg(not(feature = "nn"))]
    let nn_theta_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

    writeln!(f, "\ntheta:").map_err(|e| e.to_string())?;
    for (i, name) in result.theta_names.iter().enumerate() {
        if nn_theta_indices.contains(&i) {
            continue;
        }
        let est = result.theta[i];
        let is_fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_theta.as_ref().map(|v| v[i]);
        let rse = se.map(|s| {
            if est.abs() > 1e-12 {
                (s / est.abs()) * 100.0
            } else {
                f64::NAN
            }
        });
        writeln!(f, "  {}:", name).map_err(|e| e.to_string())?;
        writeln!(f, "    estimate: {:.6}", est).map_err(|e| e.to_string())?;
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
            writeln!(f, "    rse_pct: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(s) => {
                    writeln!(f, "    se: {:.6}", s).map_err(|e| e.to_string())?;
                    writeln!(f, "    rse_pct: {:.2}", rse.unwrap()).map_err(|e| e.to_string())?;
                }
                None => {
                    writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
                    writeln!(f, "    rse_pct: ~").map_err(|e| e.to_string())?;
                }
            }
        }
    }

    // Compact NN-weights summary (Option E). Lists shape, activations,
    // weight count, and basic statistics over the trained weight values
    // — none of which need to be one-line-per-weight to be useful for
    // sanity-checking a fit. Full per-weight values are still recoverable
    // from `result.theta` slices keyed by `weights_offset` + `n_weights`.
    #[cfg(feature = "nn")]
    if !result.neural_networks.is_empty() {
        writeln!(f, "\nneural_networks:").map_err(|e| e.to_string())?;
        for nn in &result.neural_networks {
            writeln!(f, "  {}:", nn.name).map_err(|e| e.to_string())?;
            writeln!(
                f,
                "    shape: [{}]",
                nn.shape
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .map_err(|e| e.to_string())?;
            writeln!(f, "    hidden_activation: {}", nn.hidden_activation)
                .map_err(|e| e.to_string())?;
            writeln!(f, "    output_activation: {}", nn.output_activation)
                .map_err(|e| e.to_string())?;
            writeln!(f, "    inputs: [{}]", nn.input_names.join(", "))
                .map_err(|e| e.to_string())?;
            writeln!(f, "    outputs: [{}]", nn.output_names.join(", "))
                .map_err(|e| e.to_string())?;
            writeln!(f, "    n_weights: {}", nn.n_weights).map_err(|e| e.to_string())?;
            // Summary statistics over the trained weight values.
            let w_slice = &result.theta[nn.weights_offset..nn.weights_offset + nn.n_weights];
            let (mn, mx, mean, sd) = weight_summary(w_slice);
            writeln!(f, "    weights_summary:").map_err(|e| e.to_string())?;
            writeln!(f, "      min:  {:.6}", mn).map_err(|e| e.to_string())?;
            writeln!(f, "      max:  {:.6}", mx).map_err(|e| e.to_string())?;
            writeln!(f, "      mean: {:.6}", mean).map_err(|e| e.to_string())?;
            writeln!(f, "      std:  {:.6}", sd).map_err(|e| e.to_string())?;
        }
    }

    let n_eta = result.omega.nrows();
    writeln!(f, "\nomega:").map_err(|e| e.to_string())?;
    for i in 0..n_eta {
        let var = result.omega[(i, i)];
        let cv_pct = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
        let is_fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
        let se = crate::types::omega_se_at(&result.se_omega, n_eta, i, i);
        let key = result
            .eta_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("omega_{}_{}", i + 1, i + 1));
        writeln!(f, "  {}:", key).map_err(|e| e.to_string())?;
        writeln!(f, "    variance: {:.6}", var).map_err(|e| e.to_string())?;
        writeln!(f, "    cv_pct: {:.2}", cv_pct).map_err(|e| e.to_string())?;
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(s) => writeln!(f, "    se: {:.6}", s).map_err(|e| e.to_string())?,
                None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
            }
        }
    }
    // Off-diagonal covariances
    for i in 0..n_eta {
        for j in 0..i {
            let cov = result.omega[(i, j)];
            if cov.abs() > 1e-15 {
                let name_i = result
                    .eta_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("eta_{}", i + 1));
                let name_j = result
                    .eta_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("eta_{}", j + 1));
                let param_corr = result
                    .omega_param_corr
                    .as_ref()
                    .map(|m| m[(i, j)])
                    .unwrap_or_else(|| {
                        let var_i = result.omega[(i, i)];
                        let var_j = result.omega[(j, j)];
                        if var_i > 0.0 && var_j > 0.0 {
                            cov / (var_i.sqrt() * var_j.sqrt())
                        } else {
                            0.0
                        }
                    });
                let se_cov = crate::types::omega_se_at(&result.se_omega, n_eta, i, j);
                writeln!(f, "  {}__{}:", name_i, name_j).map_err(|e| e.to_string())?;
                writeln!(f, "    covariance: {:.6}", cov).map_err(|e| e.to_string())?;
                match se_cov {
                    Some(s) => writeln!(f, "    se: {:.6}", s).map_err(|e| e.to_string())?,
                    None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
                }
                writeln!(f, "    correlation: {:.6}", param_corr).map_err(|e| e.to_string())?;
            }
        }
    }

    let err_type_str = match result.error_model {
        ErrorModel::Additive => "additive",
        ErrorModel::Proportional => "proportional",
        ErrorModel::Combined => "combined",
    };
    writeln!(f, "\nsigma:  # error model: {}", err_type_str).map_err(|e| e.to_string())?;
    // Sigma is stored on the SD scale (see src/stats/residual_error.rs).
    // `variance` is therefore `estimate^2` for both component types. `cv_pct`
    // is only emitted for proportional components, where `sigma * 100` is
    // the coefficient of variation directly; an additive sigma's value lives
    // in observation units and has no scale-free CV interpretation.
    // `result.sigma_types` is parallel to `result.sigma`.
    for (i, &s) in result.sigma.iter().enumerate() {
        let is_fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_sigma.as_ref().and_then(|v| v.get(i).copied());
        let key = result
            .sigma_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("sigma_{}", i + 1));
        let sigma_type = result.sigma_types.get(i).copied();
        let kind_str = match sigma_type {
            Some(SigmaType::Proportional) => "proportional",
            Some(SigmaType::Additive) => "additive",
            None => "unknown",
        };
        writeln!(f, "  {}:", key).map_err(|e| e.to_string())?;
        writeln!(f, "    estimate: {:.6}", s).map_err(|e| e.to_string())?;
        writeln!(f, "    variance: {:.6}", s * s).map_err(|e| e.to_string())?;
        writeln!(f, "    type: {}", kind_str).map_err(|e| e.to_string())?;
        if matches!(sigma_type, Some(SigmaType::Proportional)) {
            writeln!(f, "    cv_pct: {:.2}", s * 100.0).map_err(|e| e.to_string())?;
        }
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(sv) => writeln!(f, "    se: {:.6}", sv).map_err(|e| e.to_string())?,
                None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
            }
        }
    }

    // IOV (KAPPA) block
    if let Some(ref iov) = result.omega_iov {
        writeln!(f, "\nomega_iov:").map_err(|e| e.to_string())?;
        let n_kappa = iov.nrows();
        for i in 0..n_kappa {
            let var = iov[(i, i)];
            let cv_pct = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
            let is_fixed = result.kappa_fixed.get(i).copied().unwrap_or(false);
            let se = result.se_kappa.as_ref().and_then(|v| v.get(i).copied());
            let name = result
                .kappa_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("kappa_{}", i + 1));
            writeln!(f, "  {}:", name).map_err(|e| e.to_string())?;
            writeln!(f, "    variance: {:.6}", var).map_err(|e| e.to_string())?;
            writeln!(f, "    cv_pct: {:.2}", cv_pct).map_err(|e| e.to_string())?;
            if is_fixed {
                writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
                writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
            } else {
                match se {
                    Some(sv) => writeln!(f, "    se: {:.6}", sv).map_err(|e| e.to_string())?,
                    None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
                }
            }
        }
        // Off-diagonal covariances/correlations (block_kappa). Keyed as
        // `{name_i}__{name_j}` to keep the per-name structure of the diagonal
        // entries; falls back to numeric indices if names are missing.
        for i in 0..n_kappa {
            for j in 0..i {
                let cov = iov[(i, j)];
                if cov.abs() <= 1e-15 {
                    continue;
                }
                let name_i = result
                    .kappa_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("kappa_{}", i + 1));
                let name_j = result
                    .kappa_names
                    .get(j)
                    .cloned()
                    .unwrap_or_else(|| format!("kappa_{}", j + 1));
                let param_corr = result
                    .omega_iov_param_corr
                    .as_ref()
                    .map(|m| m[(i, j)])
                    .unwrap_or_else(|| {
                        let var_i = iov[(i, i)];
                        let var_j = iov[(j, j)];
                        if var_i > 0.0 && var_j > 0.0 {
                            cov / (var_i.sqrt() * var_j.sqrt())
                        } else {
                            0.0
                        }
                    });
                writeln!(f, "  {}__{}:", name_i, name_j).map_err(|e| e.to_string())?;
                writeln!(f, "    covariance: {:.6}", cov).map_err(|e| e.to_string())?;
                writeln!(f, "    correlation: {:.6}", param_corr).map_err(|e| e.to_string())?;
            }
        }
    }

    // Importance-sampling marginal log-likelihood section
    if let Some(ref imp) = result.importance_sampling {
        writeln!(f, "\nimportance_sampling:").map_err(|e| e.to_string())?;
        writeln!(
            f,
            "  minus2_log_likelihood: {:.6}",
            imp.minus2_log_likelihood
        )
        .map_err(|e| e.to_string())?;
        writeln!(f, "  mc_standard_error: {:.6}", imp.mc_standard_error)
            .map_err(|e| e.to_string())?;
        writeln!(f, "  n_samples: {}", imp.n_samples).map_err(|e| e.to_string())?;
        writeln!(f, "  proposal_df: {:.4}", imp.proposal_df).map_err(|e| e.to_string())?;
        writeln!(f, "  ess_min: {:.4}", imp.ess_min).map_err(|e| e.to_string())?;
        writeln!(f, "  ess_median: {:.4}", imp.ess_median).map_err(|e| e.to_string())?;
        let kt = match imp.kappa_treatment {
            KappaTreatment::NotApplicable => "not_applicable",
            KappaTreatment::FixedAtMode => "fixed_at_mode",
            KappaTreatment::Marginalized => "marginalized",
        };
        writeln!(f, "  kappa_treatment: {}", kt).map_err(|e| e.to_string())?;
        if imp.low_ess_subjects.is_empty() {
            writeln!(f, "  low_ess_subjects: []").map_err(|e| e.to_string())?;
        } else {
            writeln!(f, "  low_ess_subjects:").map_err(|e| e.to_string())?;
            for (id, frac) in &imp.low_ess_subjects {
                writeln!(f, "    - id: \"{}\"", id).map_err(|e| e.to_string())?;
                writeln!(f, "      ess_fraction: {:.4}", frac).map_err(|e| e.to_string())?;
            }
        }
    }

    // Bayesian posterior section
    if let Some(ref b) = result.bayes {
        writeln!(f, "\nbayes:").map_err(|e| e.to_string())?;
        writeln!(f, "  n_chains: {}", b.n_chains).map_err(|e| e.to_string())?;
        writeln!(f, "  n_warmup: {}", b.n_warmup).map_err(|e| e.to_string())?;
        writeln!(f, "  n_draws_per_chain: {}", b.n_draws_per_chain).map_err(|e| e.to_string())?;
        writeln!(f, "  n_divergent: {}", b.n_divergent).map_err(|e| e.to_string())?;
        writeln!(f, "  max_rhat: {:.4}", b.max_rhat).map_err(|e| e.to_string())?;
        writeln!(f, "  parameters:").map_err(|e| e.to_string())?;
        for s in &b.summaries {
            writeln!(f, "    - name: \"{}\"", s.name).map_err(|e| e.to_string())?;
            writeln!(f, "      mean: {:.6}", s.mean).map_err(|e| e.to_string())?;
            writeln!(f, "      sd: {:.6}", s.sd).map_err(|e| e.to_string())?;
            writeln!(f, "      q025: {:.6}", s.q025).map_err(|e| e.to_string())?;
            writeln!(f, "      median: {:.6}", s.median).map_err(|e| e.to_string())?;
            writeln!(f, "      q975: {:.6}", s.q975).map_err(|e| e.to_string())?;
            writeln!(f, "      rhat: {:.4}", s.rhat).map_err(|e| e.to_string())?;
            writeln!(f, "      ess_bulk: {:.1}", s.ess_bulk).map_err(|e| e.to_string())?;
            writeln!(f, "      ess_tail: {:.1}", s.ess_tail).map_err(|e| e.to_string())?;
            writeln!(f, "      mcse: {:.6}", s.mcse).map_err(|e| e.to_string())?;
        }
    }

    // SIR section
    if let Some(ess) = result.sir_ess {
        writeln!(f, "\nsir:").map_err(|e| e.to_string())?;
        writeln!(f, "  effective_sample_size: {:.1}", ess).map_err(|e| e.to_string())?;
        if let Some(ref ci) = result.sir_ci_theta {
            writeln!(f, "  ci_theta:").map_err(|e| e.to_string())?;
            for (i, name) in result.theta_names.iter().enumerate() {
                if i < ci.len() {
                    writeln!(f, "    {}:", name).map_err(|e| e.to_string())?;
                    writeln!(f, "      lower: {:.6}", ci[i].0).map_err(|e| e.to_string())?;
                    writeln!(f, "      upper: {:.6}", ci[i].1).map_err(|e| e.to_string())?;
                }
            }
        }
        if let Some(ref ci) = result.sir_ci_omega {
            writeln!(f, "  ci_omega:").map_err(|e| e.to_string())?;
            for (i, (lo, hi)) in ci.iter().enumerate() {
                let key = result
                    .eta_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("omega_{}_{}", i + 1, i + 1));
                writeln!(f, "    {}:", key).map_err(|e| e.to_string())?;
                writeln!(f, "      lower: {:.6}", lo).map_err(|e| e.to_string())?;
                writeln!(f, "      upper: {:.6}", hi).map_err(|e| e.to_string())?;
            }
        }
        if let Some(ref ci) = result.sir_ci_sigma {
            writeln!(f, "  ci_sigma:").map_err(|e| e.to_string())?;
            for (i, (lo, hi)) in ci.iter().enumerate() {
                let key = result
                    .sigma_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("sigma_{}", i + 1));
                writeln!(f, "    {}:", key).map_err(|e| e.to_string())?;
                writeln!(f, "      lower: {:.6}", lo).map_err(|e| e.to_string())?;
                writeln!(f, "      upper: {:.6}", hi).map_err(|e| e.to_string())?;
            }
        }
    }

    if let Some(excl) = &result.exclusions {
        writeln!(f, "\nexclusions:").map_err(|e| e.to_string())?;
        writeln!(f, "  n_records_total: {}", excl.n_records_total).map_err(|e| e.to_string())?;
        writeln!(f, "  n_obs_excluded: {}", excl.n_obs_excluded).map_err(|e| e.to_string())?;
        writeln!(f, "  n_dose_excluded: {}", excl.n_dose_excluded).map_err(|e| e.to_string())?;
        writeln!(f, "  n_other_excluded: {}", excl.n_other_excluded).map_err(|e| e.to_string())?;
        if !excl.excluded_subject_ids.is_empty() {
            writeln!(f, "  excluded_subject_ids:").map_err(|e| e.to_string())?;
            for id in &excl.excluded_subject_ids {
                writeln!(f, "    - \"{}\"", id).map_err(|e| e.to_string())?;
            }
        }
        if !excl.fired_ignore.is_empty() {
            writeln!(f, "  fired_ignore:").map_err(|e| e.to_string())?;
            for c in &excl.fired_ignore {
                writeln!(f, "    - \"{}\"", c).map_err(|e| e.to_string())?;
            }
        }
        if !excl.fired_accept.is_empty() {
            writeln!(f, "  fired_accept:").map_err(|e| e.to_string())?;
            for c in &excl.fired_accept {
                writeln!(f, "    - \"{}\"", c).map_err(|e| e.to_string())?;
            }
        }
    }

    // Covariance matrix block (optimizer parameterization).
    // Emitted when the covariance step succeeded or was regularized, giving
    // downstream tools (bootstrap, SIR, uncertainty propagation) the full
    // parameter covariance without re-running the fit.
    if let Some(ref cov) = result.covariance_matrix {
        let n = cov.nrows();
        let names = packed_param_names(result, n);
        writeln!(f, "\ncovariance_matrix:").map_err(|e| e.to_string())?;
        writeln!(
            f,
            "  # optimizer parameterization: theta log-transformed when lower bound >= 0 \
             (identity otherwise), sigma log-transformed, omega/kappa Cholesky-factored"
        )
        .map_err(|e| e.to_string())?;
        writeln!(f, "  parameters: [{}]", names.join(", ")).map_err(|e| e.to_string())?;
        writeln!(f, "  rows:").map_err(|e| e.to_string())?;
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| fmt_cov_entry(cov[(i, j)])).collect();
            writeln!(f, "    {}: [{}]", names[i], row.join(", ")).map_err(|e| e.to_string())?;
        }
    }

    if !result.warnings.is_empty() {
        writeln!(f, "\nwarnings:").map_err(|e| e.to_string())?;
        for w in &result.warnings {
            writeln!(f, "  - \"{}\"", w).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

/// Parameter table as text
pub fn parameter_table(result: &FitResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{:<20} {:>12} {:>12} {:>10} {:>8}",
        "Parameter", "Estimate", "SE", "%RSE", "Type"
    ));
    lines.push("-".repeat(64));

    for (i, name) in result.theta_names.iter().enumerate() {
        let est = result.theta[i];
        let (se_str, rse_str) = match &result.se_theta {
            Some(se) => {
                let se_val = se[i];
                let rse = if est.abs() > 1e-12 {
                    (se_val / est.abs()) * 100.0
                } else {
                    f64::NAN
                };
                (format!("{:.6}", se_val), format!("{:.1}", rse))
            }
            None => ("---".to_string(), "---".to_string()),
        };
        lines.push(format!(
            "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
            name, est, se_str, rse_str, "THETA"
        ));
    }

    let n_eta = result.omega.nrows();
    for i in 0..n_eta {
        for j in 0..=i {
            let val = result.omega[(i, j)];
            let name = format!("OMEGA({},{})", i + 1, j + 1);
            lines.push(format!(
                "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
                name, val, "---", "---", "OMEGA"
            ));
        }
    }

    for (i, &s) in result.sigma.iter().enumerate() {
        let name = format!("SIGMA({})", i + 1);
        lines.push(format!(
            "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
            name, s, "---", "---", "SIGMA"
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    /// Build a near-empty `FitResult` with just enough data for the YAML
    /// emitter to produce a sigma block. All fields the sigma block does not
    /// read are zeroed / empty.
    fn make_sigma_only_result(error_model: ErrorModel, sigma: Vec<f64>) -> FitResult {
        let sigma_types = error_model.sigma_types();
        let n = sigma.len();
        FitResult {
            method: EstimationMethod::Foce,
            method_chain: vec![EstimationMethod::Foce],
            converged: true,
            ofv: 0.0,
            aic: 0.0,
            bic: 0.0,
            theta: Vec::new(),
            theta_names: Vec::new(),
            eta_names: Vec::new(),
            omega: DMatrix::zeros(0, 0),
            sigma,
            sigma_names: (0..n).map(|i| format!("EPS_{}", i + 1)).collect(),
            error_model,
            covariance_matrix: None,
            se_theta: None,
            se_omega: None,
            se_sigma: None,
            theta_fixed: Vec::new(),
            omega_fixed: Vec::new(),
            sigma_fixed: vec![false; n],
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false; n],
            subjects: Vec::new(),
            n_obs: 0,
            n_subjects: 0,
            n_parameters: 0,
            n_iterations: 0,
            interaction: false,
            warnings: Vec::new(),
            warnings_structured: Vec::new(),
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            impmap_trace: None,
            bayes: None,
            omega_iov: None,
            kappa_names: Vec::new(),
            kappa_fixed: Vec::new(),
            kappa_init_as_sd: Vec::new(),
            se_kappa: None,
            shrinkage_kappa: Vec::new(),
            shrinkage_kappa_by_occ: Vec::new(),
            ebe_kappas: Vec::new(),
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: String::new(),
            gradient_method_outer: String::new(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 1,
            nlopt_missing_algorithms: Vec::new(),
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: CovarianceStatus::NotRequested,
            shrinkage_eta: Vec::new(),
            cond_dist: None,
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: "test".to_string(),
            ferx_version: env!("CARGO_PKG_VERSION").to_string(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            sigma_types,
            cov_eigenvalues: None,
            cov_condition_number: None,
            eta_log_transformed: Vec::new(),
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            model_text: None,
            theta_init: Vec::new(),
            omega_init: DMatrix::zeros(0, 0),
            sigma_init: Vec::new(),
            obs_time_range: None,
            final_gradient: None,
            optimizer: "bobyqa".to_string(),
            n_starts: 1,
            multi_start_seed: None,
            saem_seed: None,
            sir_seed: None,
            imp_seed: None,
            npde_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 0,
            outer_gtol: 0.0,
            inits_from_nca: None,
            covariate_names: Vec::new(),
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
            exclusions: None,
        }
    }

    #[test]
    fn format_summary_contains_key_sections() {
        // Build a small-but-representative fit: two thetas (one fixed), a
        // diagonal 2×2 omega, one proportional sigma, a condition number,
        // shrinkage, and a warning — enough to exercise every section and the
        // fixed-label / CV% / %RSE branches.
        let mut r = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r.method = EstimationMethod::FoceI;
        r.method_chain = vec![EstimationMethod::FoceI];
        r.converged = true;
        r.ofv = 123.45;
        r.aic = 130.0;
        r.bic = 140.0;
        r.n_subjects = 12;
        r.n_obs = 96;
        r.n_parameters = 4;
        r.n_iterations = 7;
        r.theta = vec![1.5, 10.0];
        r.theta_names = vec!["CL".into(), "V".into()];
        r.theta_fixed = vec![false, true];
        r.se_theta = Some(vec![0.15, 0.0]);
        r.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.0, 0.0, 0.04]);
        r.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
        r.omega_fixed = vec![false, false];
        r.covariance_status = CovarianceStatus::Computed;
        r.cov_condition_number = Some(42.0);
        r.shrinkage_eta = vec![0.05, 0.10];
        r.shrinkage_eps = 0.08;
        r.wall_time_secs = 3.2;
        r.warnings = vec!["heads up".into()];

        let s = format_summary(&r);
        assert!(s.contains("ferx run summary — test"));
        assert!(s.contains("Method:     FOCEI"));
        assert!(s.contains("Converged:  YES"));
        assert!(s.contains("Subjects: 12   Observations: 96   Parameters: 4"));
        assert!(s.contains("OFV:  123.4500"));
        assert!(s.contains("--- THETA ---"));
        assert!(s.contains("%RSE"));
        // Free theta shows an SE; fixed theta is labelled and SE-suppressed.
        assert!(s.contains("CL"));
        assert!(s.contains("V [FIX]"));
        assert!(s.contains("--- OMEGA ---"));
        assert!(s.contains("ETA_CL"));
        assert!(s.contains("(CV% ="));
        assert!(s.contains("--- SIGMA (proportional) ---"));
        assert!(s.contains("Condition number: 42.0"));
        assert!(s.contains("Eta shrinkage:"));
        assert!(s.contains("EPS shrinkage: 8.0%"));
        assert!(s.contains("Warnings: 1"));
        assert!(s.contains("heads up"));
    }

    #[test]
    fn format_summary_block_omega_shows_correlation() {
        // A correlated (block) omega must surface a `corr(...)` line.
        let mut r = make_sigma_only_result(ErrorModel::Additive, vec![1.0]);
        r.theta = vec![2.0];
        r.theta_names = vec!["CL".into()];
        r.theta_fixed = vec![false];
        r.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.03, 0.03, 0.04]);
        r.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
        r.omega_fixed = vec![false, false];

        let s = format_summary(&r);
        assert!(s.contains("corr(ETA_V, ETA_CL) ="));
    }

    #[test]
    fn format_summary_no_cov_suppresses_cv_and_rse() {
        // Failed covariance → no SE table, CV% suppressed, method chain shown,
        // fixed omega/sigma labelled, no condition number, no warnings.
        let mut r = make_sigma_only_result(ErrorModel::Additive, vec![0.5]);
        r.method = EstimationMethod::FoceI;
        r.method_chain = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
        r.converged = false;
        r.theta = vec![1.5];
        r.theta_names = vec!["CL".into()];
        r.theta_fixed = vec![false];
        r.se_theta = None;
        r.omega = DMatrix::from_row_slice(1, 1, &[0.09]);
        r.eta_names = vec!["ETA_CL".into()];
        r.omega_fixed = vec![true];
        r.sigma_fixed = vec![true];
        r.covariance_status = CovarianceStatus::Failed;
        r.cov_condition_number = None;
        r.shrinkage_eta = vec![f64::NAN];
        r.shrinkage_eps = f64::NAN;

        let s = format_summary(&r);
        // Method chain joined with the arrow.
        assert!(s.contains("SAEM → FOCEI"));
        assert!(s.contains("Converged:  NO"));
        // No se_theta → N/A in the %RSE column, no CV% under failed covariance.
        assert!(s.contains("N/A"));
        assert!(!s.contains("CV% ="));
        // Fixed omega/sigma are [FIX]-labelled; condition number omitted.
        assert!(s.contains("ETA_CL [FIX]"));
        assert!(!s.contains("Condition number"));
        // No finite shrinkage or warnings → those lines are absent.
        assert!(!s.contains("Eta shrinkage:"));
        assert!(!s.contains("EPS shrinkage:"));
        assert!(!s.contains("Warnings:"));
    }

    fn yaml_for(error_model: ErrorModel, sigma: Vec<f64>) -> String {
        let result = make_sigma_only_result(error_model, sigma);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&result, path.to_str().unwrap()).expect("yaml write");
        std::fs::read_to_string(&path).expect("yaml read")
    }

    // ─── Option E: NN-aware fit YAML rendering ───────────────────────────

    /// Build a minimal FitResult with one `[covariate_nn]` block worth of
    /// theta entries so we can exercise the YAML writer's compact rendering.
    #[cfg(feature = "nn")]
    fn make_nn_result() -> FitResult {
        // 1 user theta (TVKA) + a 2->3->2 NN: W_1 6, b_1 3, W_2 6, b_2 2 → 17 weights.
        let mut theta = vec![1.5_f64];
        let mut theta_names = vec!["TVKA".to_string()];
        // NN weight names — pattern: W_<NAME>_<l>_<i>_<j>, B_<NAME>_<l>_<i>.
        for l in 1..3 {
            let (n_l, n_lm1) = if l == 1 { (3, 2) } else { (2, 3) };
            for i in 1..=n_l {
                for j in 1..=n_lm1 {
                    theta.push(0.1 * (i + j) as f64);
                    theta_names.push(format!("W_TYPICAL_PK_{}_{}_{}", l, i, j));
                }
            }
            for i in 1..=n_l {
                theta.push(0.01 * i as f64);
                theta_names.push(format!("B_TYPICAL_PK_{}_{}", l, i));
            }
        }
        let n = theta.len();
        let theta_fixed = vec![false; n];

        let mut base = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        base.theta = theta;
        base.theta_names = theta_names;
        base.theta_fixed = theta_fixed;
        base.neural_networks = vec![NeuralNetworkInfo {
            name: "TYPICAL_PK".to_string(),
            shape: vec![2, 3, 2],
            hidden_activation: "tanh".to_string(),
            output_activation: "softplus".to_string(),
            n_weights: 17,
            weights_offset: 1,
            input_names: vec!["WT".to_string(), "CRCL".to_string()],
            output_names: vec!["CL".to_string(), "V".to_string()],
        }];
        base
    }

    #[cfg(feature = "nn")]
    #[test]
    fn yaml_collapses_nn_weight_thetas_into_summary_section() {
        let result = make_nn_result();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&result, path.to_str().unwrap()).expect("yaml write");
        let yaml = std::fs::read_to_string(&path).expect("yaml read");

        // The user-declared theta (TVKA) is still in the per-theta listing.
        assert!(
            yaml.contains("  TVKA:"),
            "TVKA must appear in theta section:\n{yaml}"
        );

        // None of the 17 NN-weight thetas appear in the per-theta listing.
        for layer in 1..3 {
            let n_l = if layer == 1 { 3 } else { 2 };
            let n_lm1 = if layer == 1 { 2 } else { 3 };
            for i in 1..=n_l {
                for j in 1..=n_lm1 {
                    let name = format!("W_TYPICAL_PK_{}_{}_{}:", layer, i, j);
                    assert!(
                        !yaml.contains(&name),
                        "NN weight `{}` should NOT appear in per-theta section:\n{yaml}",
                        name
                    );
                }
            }
        }

        // The compact `neural_networks:` section IS present.
        assert!(
            yaml.contains("\nneural_networks:"),
            "neural_networks section missing:\n{yaml}"
        );
        assert!(
            yaml.contains("  TYPICAL_PK:"),
            "NN block name missing:\n{yaml}"
        );
        assert!(
            yaml.contains("    shape: [2, 3, 2]"),
            "shape missing:\n{yaml}"
        );
        assert!(
            yaml.contains("hidden_activation: tanh"),
            "hidden activation missing:\n{yaml}"
        );
        assert!(
            yaml.contains("output_activation: softplus"),
            "output activation missing:\n{yaml}"
        );
        assert!(
            yaml.contains("inputs: [WT, CRCL]"),
            "inputs missing:\n{yaml}"
        );
        assert!(
            yaml.contains("outputs: [CL, V]"),
            "outputs missing:\n{yaml}"
        );
        assert!(yaml.contains("n_weights: 17"), "n_weights missing:\n{yaml}");
        assert!(
            yaml.contains("weights_summary:"),
            "weights_summary missing:\n{yaml}"
        );
        assert!(yaml.contains("      min:"), "min stat missing:\n{yaml}");
        assert!(yaml.contains("      max:"), "max stat missing:\n{yaml}");
    }

    /// Regression: `format_summary` must exclude NN weight thetas from the
    /// THETA table (they can number in the hundreds), matching the YAML writer
    /// and `print_results`. Without the `nn_theta_indices` skip this dumps one
    /// row per weight.
    #[cfg(feature = "nn")]
    #[test]
    fn format_summary_excludes_nn_weight_thetas() {
        let result = make_nn_result();
        let s = format_summary(&result);

        // The user-declared theta is still listed.
        assert!(s.contains("TVKA"), "TVKA must appear in THETA table:\n{s}");

        // None of the 17 NN-weight thetas appear.
        for layer in 1..3 {
            let n_l = if layer == 1 { 3 } else { 2 };
            let n_lm1 = if layer == 1 { 2 } else { 3 };
            for i in 1..=n_l {
                for j in 1..=n_lm1 {
                    let name = format!("W_TYPICAL_PK_{}_{}_{}", layer, i, j);
                    assert!(
                        !s.contains(&name),
                        "NN weight `{}` should NOT appear in THETA table:\n{s}",
                        name
                    );
                }
            }
        }
    }

    /// Sanity: when no `[covariate_nn]` blocks are declared, the YAML
    /// should NOT have a `neural_networks:` section at all.
    #[cfg(feature = "nn")]
    #[test]
    fn yaml_no_neural_networks_section_when_block_absent() {
        let result = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&result, path.to_str().unwrap()).expect("yaml write");
        let yaml = std::fs::read_to_string(&path).expect("yaml read");
        assert!(
            !yaml.contains("neural_networks:"),
            "neural_networks section should be absent when no NN blocks:\n{yaml}"
        );
    }

    #[test]
    fn sigma_yaml_proportional_emits_variance_and_cv_pct() {
        let yaml = yaml_for(ErrorModel::Proportional, vec![0.1]);
        // sigma is on the SD scale: variance = 0.1² = 0.01, cv_pct = 10.
        assert!(yaml.contains("estimate: 0.100000"), "yaml=\n{}", yaml);
        assert!(yaml.contains("variance: 0.010000"), "yaml=\n{}", yaml);
        assert!(yaml.contains("cv_pct: 10.00"), "yaml=\n{}", yaml);
        assert!(yaml.contains("type: proportional"), "yaml=\n{}", yaml);
    }

    #[test]
    fn sigma_yaml_additive_emits_variance_but_no_cv_pct() {
        let yaml = yaml_for(ErrorModel::Additive, vec![0.5]);
        assert!(yaml.contains("estimate: 0.500000"), "yaml=\n{}", yaml);
        assert!(yaml.contains("variance: 0.250000"), "yaml=\n{}", yaml);
        assert!(yaml.contains("type: additive"), "yaml=\n{}", yaml);
        // Additive sigma is in observation units — no scale-free CV applies.
        assert!(
            !yaml.lines().any(|l| l.trim_start().starts_with("cv_pct:")),
            "yaml unexpectedly contains cv_pct:\n{}",
            yaml
        );
    }

    #[test]
    fn sigma_yaml_combined_distinguishes_components() {
        let yaml = yaml_for(ErrorModel::Combined, vec![0.2, 0.5]);
        // First sigma is proportional (CV%), second is additive (no CV).
        assert!(yaml.contains("type: proportional"), "yaml=\n{}", yaml);
        assert!(yaml.contains("type: additive"), "yaml=\n{}", yaml);
        assert!(yaml.contains("cv_pct: 20.00"), "yaml=\n{}", yaml);
        // Variances: 0.2² = 0.04 and 0.5² = 0.25.
        assert!(yaml.contains("variance: 0.040000"), "yaml=\n{}", yaml);
        assert!(yaml.contains("variance: 0.250000"), "yaml=\n{}", yaml);
        // Exactly one cv_pct line for the prop component.
        assert_eq!(
            yaml.lines()
                .filter(|l| l.trim_start().starts_with("cv_pct:"))
                .count(),
            1,
            "yaml=\n{}",
            yaml
        );
    }

    // ── sdtab helpers ────────────────────────────────────────────────────────

    fn sdtab_subject_result(id: &str, n_obs: usize) -> SubjectResult {
        SubjectResult {
            id: id.to_string(),
            eta: nalgebra::DVector::zeros(0),
            ipred: vec![1.0; n_obs],
            pred: vec![1.0; n_obs],
            iwres: vec![0.0; n_obs],
            cwres: vec![0.0; n_obs],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 0.0,
            cens: vec![0; n_obs],
            n_obs,
            extra_columns: vec![],
            per_obs_tad: vec![],
            compartment_states: vec![],
        }
    }

    /// Round-trip the conditional-distribution CSV writers (#257): a populated
    /// `cond_dist` produces the documented `-conddist.csv` / `-conddist-samples.csv`
    /// layouts, and the two error paths (no `cond_dist`, no retained samples)
    /// return `Err` without creating a file.
    #[test]
    fn write_conddist_csv_and_samples_roundtrip() {
        let mut s0 = sdtab_subject_result("1", 2);
        s0.eta = nalgebra::DVector::from_vec(vec![0.10, -0.20]);
        let mut s1 = sdtab_subject_result("2", 2);
        s1.eta = nalgebra::DVector::from_vec(vec![0.30, 0.40]);
        let mut r = minimal_sdtab_result(vec![s0, s1]);
        r.eta_names = vec!["ETA_CL".to_string(), "ETA_V".to_string()];
        r.cond_dist = Some(CondDist {
            cond_mean: vec![vec![0.11, -0.19], vec![0.29, 0.41]],
            cond_sd: vec![vec![0.05, 0.06], vec![0.07, 0.08]],
            samples: vec![
                vec![vec![0.10, -0.20], vec![0.12, -0.18]],
                vec![vec![0.28, 0.40], vec![0.30, 0.42]],
            ],
            shrinkage: vec![0.2, 0.3],
            nsamp: 2,
            burnin: 1,
        });

        let dir = tempfile::tempdir().expect("tempdir");
        let model = dir.path().join("m");
        let model_name = model.to_str().unwrap();

        // Drive the CLI dispatch helper: with retained draws it writes both files
        // and reports both.
        let msgs = write_conddist_outputs(&r, model_name);
        assert_eq!(
            msgs.len(),
            2,
            "expected conddist + samples messages: {msgs:?}"
        );
        assert!(msgs[0].starts_with("Conditional distribution written to"));
        assert!(msgs[1].starts_with("Conditional-distribution draws written to"));

        // --- conditional mean/SD/mode CSV ---
        let lines: Vec<String> = std::fs::read_to_string(format!("{model_name}-conddist.csv"))
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        assert_eq!(lines[0], "ID,ETA,COND_MEAN,COND_SD,COND_MODE");
        // header + 2 subjects × 2 eta
        assert_eq!(lines.len(), 5);
        // subject 1, ETA_CL: mean 0.11, sd 0.05, mode = eta[0] = 0.10
        assert_eq!(lines[1], "1,ETA_CL,0.110000,0.050000,0.100000");
        // subject 2, ETA_V: mean 0.41, sd 0.08, mode = eta[1] = 0.40
        assert_eq!(lines[4], "2,ETA_V,0.410000,0.080000,0.400000");

        // --- raw draws CSV ---
        let slines: Vec<String> =
            std::fs::read_to_string(format!("{model_name}-conddist-samples.csv"))
                .unwrap()
                .lines()
                .map(|l| l.to_string())
                .collect();
        assert_eq!(slines[0], "ID,SAMPLE,ETA_CL,ETA_V");
        // header + 2 subjects × 2 draws
        assert_eq!(slines.len(), 5);
        assert_eq!(slines[1], "1,1,0.100000,-0.200000");
        assert_eq!(slines[4], "2,2,0.300000,0.420000");

        // --- dispatch: no conditional-distribution results → no files, no messages ---
        let mut r_none = minimal_sdtab_result(vec![]);
        r_none.cond_dist = None;
        assert!(write_conddist_outputs(&r_none, model_name).is_empty());
        // The writers themselves return Err on a missing cond_dist.
        assert!(write_conddist_csv(&r_none, "unused").is_err());
        assert!(write_conddist_samples_csv(&r_none, "unused").is_err());

        // --- dispatch: cond_dist present but no draws retained → only the mean/SD
        // file is written (one message); the samples writer is skipped quietly. ---
        let mut s = sdtab_subject_result("1", 1);
        s.eta = nalgebra::DVector::from_vec(vec![0.0]);
        let mut r_nodraws = minimal_sdtab_result(vec![s]);
        r_nodraws.eta_names = vec!["ETA_CL".to_string()];
        r_nodraws.cond_dist = Some(CondDist {
            cond_mean: vec![vec![0.0]],
            cond_sd: vec![vec![0.1]],
            samples: vec![vec![]],
            shrinkage: vec![f64::NAN],
            nsamp: 0,
            burnin: 0,
        });
        let model2 = dir.path().join("n");
        let msgs2 = write_conddist_outputs(&r_nodraws, model2.to_str().unwrap());
        assert_eq!(msgs2.len(), 1, "samples skipped when no draws: {msgs2:?}");
        // And the samples writer returns Err directly for the no-draws case.
        assert!(write_conddist_samples_csv(&r_nodraws, "unused").is_err());
    }

    fn sdtab_subject(id: &str, n_obs: usize, obs_cmts: Vec<usize>) -> Subject {
        use std::collections::HashMap;
        Subject {
            id: id.to_string(),
            doses: vec![],
            obs_times: (0..n_obs).map(|j| j as f64 + 1.0).collect(),
            obs_raw_times: Vec::new(),
            observations: vec![1.0; n_obs],
            obs_cmts,
            covariates: HashMap::new(),
            dose_covariates: vec![],
            obs_covariates: vec![],
            pk_only_times: vec![],
            pk_only_covariates: vec![],
            reset_times: vec![],
            cens: vec![0; n_obs],
            occasions: vec![],
            dose_occasions: vec![],
            fremtype: Vec::new(),
            #[cfg(feature = "survival")]
            obs_records: vec![],
        }
    }

    fn minimal_sdtab_result(subjects: Vec<SubjectResult>) -> FitResult {
        let sigma_types = ErrorModel::Proportional.sigma_types();
        FitResult {
            method: EstimationMethod::Foce,
            method_chain: vec![EstimationMethod::Foce],
            converged: true,
            ofv: 0.0,
            aic: 0.0,
            bic: 0.0,
            theta: Vec::new(),
            theta_names: Vec::new(),
            eta_names: Vec::new(),
            omega: DMatrix::zeros(0, 0),
            sigma: vec![0.1],
            sigma_names: vec!["eps".to_string()],
            error_model: ErrorModel::Proportional,
            covariance_matrix: None,
            se_theta: None,
            se_omega: None,
            se_sigma: None,
            theta_fixed: Vec::new(),
            omega_fixed: Vec::new(),
            sigma_fixed: vec![false],
            omega_init_as_sd: Vec::new(),
            sigma_init_as_sd: vec![false],
            subjects,
            n_obs: 0,
            n_subjects: 0,
            n_parameters: 0,
            n_iterations: 0,
            interaction: false,
            warnings: Vec::new(),
            warnings_structured: Vec::new(),
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            impmap_trace: None,
            bayes: None,
            omega_iov: None,
            kappa_names: Vec::new(),
            kappa_fixed: Vec::new(),
            kappa_init_as_sd: Vec::new(),
            se_kappa: None,
            shrinkage_kappa: Vec::new(),
            shrinkage_kappa_by_occ: Vec::new(),
            ebe_kappas: Vec::new(),
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: String::new(),
            gradient_method_outer: String::new(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 1,
            nlopt_missing_algorithms: Vec::new(),
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: CovarianceStatus::NotRequested,
            shrinkage_eta: Vec::new(),
            cond_dist: None,
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: "test".to_string(),
            ferx_version: env!("CARGO_PKG_VERSION").to_string(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            sigma_types,
            cov_eigenvalues: None,
            cov_condition_number: None,
            eta_log_transformed: Vec::new(),
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            model_text: None,
            theta_init: Vec::new(),
            omega_init: DMatrix::zeros(0, 0),
            sigma_init: Vec::new(),
            obs_time_range: None,
            final_gradient: None,
            optimizer: "bobyqa".to_string(),
            n_starts: 1,
            multi_start_seed: None,
            saem_seed: None,
            sir_seed: None,
            imp_seed: None,
            npde_seed: None,
            bloq_method: "drop".to_string(),
            outer_maxiter: 0,
            outer_gtol: 0.0,
            inits_from_nca: None,
            covariate_names: Vec::new(),
            input_columns: vec![],
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
            covariate_table: None,
            exclusions: None,
        }
    }

    // ── Step 1: sdtab ID column uses the subject's original numeric ID ────────

    #[test]
    fn sdtab_id_column_uses_subject_id_not_loop_index() {
        // Subjects with non-consecutive IDs — the classic clinical-data case.
        let result = minimal_sdtab_result(vec![
            sdtab_subject_result("101", 1),
            sdtab_subject_result("202", 1),
            sdtab_subject_result("303", 1),
        ]);
        let population = Population {
            subjects: vec![
                sdtab_subject("101", 1, vec![1]),
                sdtab_subject("202", 1, vec![1]),
                sdtab_subject("303", 1, vec![1]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        let id_col = cols
            .iter()
            .find(|(name, _)| name == "ID")
            .map(|(_, v)| v.clone())
            .expect("ID column missing");

        assert_eq!(
            id_col,
            vec![101.0, 202.0, 303.0],
            "expected original subject IDs, got {:?}",
            id_col
        );
    }

    // ── Step 2: sdtab CMT column appears only for multi-endpoint datasets ─────

    #[test]
    fn sdtab_cmt_column_present_for_multi_cmt() {
        let result = minimal_sdtab_result(vec![sdtab_subject_result("1", 2)]);
        let population = Population {
            subjects: vec![sdtab_subject("1", 2, vec![1, 2])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        let cmt_col = cols
            .iter()
            .find(|(name, _)| name == "CMT")
            .map(|(_, v)| v.clone())
            .expect("CMT column should be present for multi-endpoint data");

        assert_eq!(cmt_col, vec![1.0, 2.0]);
    }

    #[test]
    fn sdtab_npde_columns_present_when_populated() {
        let mut sr = sdtab_subject_result("1", 2);
        sr.npde = vec![0.3, -1.2];
        sr.npd = vec![0.4, -1.0];
        let result = minimal_sdtab_result(vec![sr]);
        let population = Population {
            subjects: vec![sdtab_subject("1", 2, vec![1, 1])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        let npde = cols
            .iter()
            .find(|(name, _)| name == "NPDE")
            .map(|(_, v)| v.clone())
            .expect("NPDE column should be present when populated");
        let npd = cols
            .iter()
            .find(|(name, _)| name == "NPD")
            .map(|(_, v)| v.clone())
            .expect("NPD column should be present when populated");
        assert_eq!(npde, vec![0.3, -1.2]);
        assert_eq!(npd, vec![0.4, -1.0]);
    }

    #[test]
    fn sdtab_npde_columns_absent_by_default() {
        // sdtab_subject_result leaves npde/npd empty (the npde_nsim = 0 default).
        let result = minimal_sdtab_result(vec![sdtab_subject_result("1", 2)]);
        let population = Population {
            subjects: vec![sdtab_subject("1", 2, vec![1, 1])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        assert!(
            cols.iter().all(|(name, _)| name != "NPDE" && name != "NPD"),
            "NPDE/NPD columns should be absent when npde_nsim = 0"
        );
    }

    #[test]
    fn sdtab_cmt_column_absent_for_single_cmt() {
        let result = minimal_sdtab_result(vec![sdtab_subject_result("1", 2)]);
        let population = Population {
            subjects: vec![sdtab_subject("1", 2, vec![1, 1])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        assert!(
            cols.iter().all(|(name, _)| name != "CMT"),
            "CMT column should be absent when all obs_cmts == 1"
        );
    }

    #[test]
    fn sdtab_omits_eta_columns_even_when_etas_present() {
        // Fast (no-fit) mirror of the slow-tests integration guard
        // `tests/map_estimates_outputs.rs::sdtab_omits_eta_columns_after_fit`.
        // sdtab is strictly per-observation; per-subject EBEs live in
        // `ebe_etas` on the R side, so even a BSV model with named etas must
        // NOT surface ETA* columns. This runs under `cargo test --lib`, so the
        // contract is enforced on every PR — not only in the nightly slow-tests
        // (a column-shape regression slipped in once because the only guard
        // required a full fit; this catches that class pre-merge).
        let mut sr = sdtab_subject_result("1", 2);
        sr.eta = nalgebra::DVector::from_vec(vec![0.1, -0.2]);
        let mut result = minimal_sdtab_result(vec![sr]);
        result.eta_names = vec!["ETA_CL".to_string(), "ETA_V".to_string()];

        let population = Population {
            subjects: vec![sdtab_subject("1", 2, vec![1, 1])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();

        let eta_cols: Vec<&&str> = names.iter().filter(|n| n.starts_with("ETA")).collect();
        assert!(
            eta_cols.is_empty(),
            "sdtab must not contain ETA columns (EBEs live in ebe_etas); found: {eta_cols:?}"
        );

        // Per-observation contract still holds (a future accidental column drop
        // also fails here).
        for required in [
            "ID", "TIME", "DV", "PRED", "IPRED", "CWRES", "IWRES", "EBE_OFV", "N_OBS", "TAFD",
            "TAD",
        ] {
            assert!(
                names.contains(&required),
                "sdtab missing required column `{required}`; have: {names:?}"
            );
        }
    }

    // ── Fix 4: non-numeric subject IDs fall back to 1-based loop index ───────

    /// When a subject ID cannot be parsed as f64 the sdtab ID column falls
    /// back to the 1-based loop index rather than panicking or silently
    /// emitting 0.  This test pins the fallback behavior; a separate warning
    /// is issued by fit_inner() for callers that go through the estimation
    /// path.
    #[test]
    fn sdtab_id_column_falls_back_for_non_numeric_ids() {
        let result = minimal_sdtab_result(vec![
            sdtab_subject_result("PT-001", 1),
            sdtab_subject_result("PT-002", 1),
        ]);
        let population = Population {
            subjects: vec![
                sdtab_subject("PT-001", 1, vec![1]),
                sdtab_subject("PT-002", 1, vec![1]),
            ],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let cols = sdtab(&result, &population);
        let id_col = cols
            .iter()
            .find(|(name, _)| name == "ID")
            .map(|(_, v)| v.clone())
            .expect("ID column missing");

        // Fallback: 1-based loop indices (1.0, 2.0), not NaN or 0.
        assert_eq!(
            id_col,
            vec![1.0, 2.0],
            "non-numeric IDs should fall back to 1-based index, got {:?}",
            id_col
        );
    }

    // ── fmt_num / fixed_label: pure formatting helpers ───────────────────────

    #[test]
    fn fmt_num_nan_is_blank_finite_is_six_dp() {
        assert_eq!(fmt_num(f64::NAN), "");
        assert_eq!(fmt_num(1.5), "1.500000");
        assert_eq!(fmt_num(0.0), "0.000000");
        assert_eq!(fmt_num(-2.25), "-2.250000");
    }

    #[test]
    fn fixed_label_appends_fix_tag() {
        assert_eq!(fixed_label("CL"), "CL [FIX]");
    }

    // ── write_sdtab_csv: round-trips through the file writer ──────────────────

    #[test]
    fn write_sdtab_csv_emits_header_and_blanks_nan_residuals() {
        // One subject, two observations; the second observation is BLOQ, so its
        // CWRES/IWRES are NaN and must serialize to an empty cell, not "NaN".
        let sr = SubjectResult {
            id: "7".to_string(),
            eta: nalgebra::DVector::zeros(0),
            ipred: vec![10.0, 5.0],
            pred: vec![11.0, 6.0],
            iwres: vec![0.25, f64::NAN],
            cwres: vec![-0.5, f64::NAN],
            npde: vec![],
            npd: vec![],
            ofv_contribution: 3.0,
            cens: vec![0, 1],
            n_obs: 2,
            extra_columns: Vec::new(),
            per_obs_tad: Vec::new(),
            compartment_states: Vec::new(),
        };
        let result = minimal_sdtab_result(vec![sr]);
        let population = Population {
            subjects: vec![sdtab_subject("7", 2, vec![1, 1])],
            covariate_names: vec![],
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: Vec::new(),
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sdtab.csv");
        write_sdtab_csv(&result, &population, path.to_str().unwrap()).expect("sdtab write");
        let csv = std::fs::read_to_string(&path).expect("sdtab read");

        let mut lines = csv.lines();
        let header = lines.next().expect("header line");
        assert!(header.starts_with("ID,TIME,DV"), "header={header}");
        // Subject is BLOQ on one row → CENS column appears.
        assert!(header.contains("CENS"), "CENS column expected: {header}");
        assert!(header.contains("CWRES") && header.contains("IWRES"));

        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 2, "two observation rows expected");
        // Numeric ID is preserved verbatim (parses to 7.0).
        assert!(rows[0].starts_with("7.000000,"));
        // BLOQ row: trailing CWRES/IWRES cells are blank (empty between commas).
        assert!(
            rows[1].contains(",,"),
            "NaN residuals should be blank cells, got: {}",
            rows[1]
        );
    }

    // ── write_covtab_csv: CSV escaping + NaN-as-blank ────────────────────────

    #[test]
    fn write_covtab_csv_escapes_ids_and_blanks_missing() {
        use crate::types::{CovariateKind, CovariateRow, CovariateTable};
        let table = CovariateTable {
            names: vec!["WT".to_string(), "SEX".to_string()],
            kinds: vec![CovariateKind::Continuous, CovariateKind::Categorical],
            rows: vec![
                CovariateRow {
                    id: "A,1".to_string(), // comma must be quoted by the csv writer
                    time: 0.0,
                    evid: 1,
                    values: vec![70.0, 1.0],
                },
                CovariateRow {
                    id: "B".to_string(),
                    time: 1.5,
                    evid: 0,
                    values: vec![f64::NAN, 0.0], // missing WT → blank cell
                },
            ],
        };

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("covtab.csv");
        write_covtab_csv(&table, path.to_str().unwrap()).expect("covtab write");
        let csv = std::fs::read_to_string(&path).expect("covtab read");

        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines[0], "ID,TIME,EVID,WT,SEX");
        // ID with a comma is quoted, not split across columns.
        assert!(
            lines[1].starts_with("\"A,1\",0.000000,1,70.000000,1.000000"),
            "row 1 escaping wrong: {}",
            lines[1]
        );
        // Missing WT serializes to an empty cell.
        assert_eq!(lines[2], "B,1.500000,0,,0.000000");
    }

    // ── parameter_table: theta / omega / sigma rows ──────────────────────────

    fn full_param_result(se_theta: Option<Vec<f64>>) -> FitResult {
        let mut r = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r.theta = vec![2.0, 0.0];
        r.theta_names = vec!["CL".to_string(), "ZERO".to_string()];
        r.theta_fixed = vec![false, true];
        r.se_theta = se_theta;
        r.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.01, 0.01, 0.04]);
        r.eta_names = vec!["ETA_CL".to_string(), "ETA_V".to_string()];
        r
    }

    #[test]
    fn parameter_table_lists_theta_omega_sigma_with_se() {
        let table = parameter_table(&full_param_result(Some(vec![0.2, 0.0])));
        assert!(table.contains("Parameter") && table.contains("Type"));
        // THETA row carries an SE and a finite %RSE.
        let cl = table.lines().find(|l| l.starts_with("CL")).expect("CL row");
        assert!(cl.contains("THETA") && cl.contains("0.200000"), "{cl}");
        // OMEGA lower triangle: (1,1),(2,1),(2,2).
        assert!(table.contains("OMEGA(1,1)"));
        assert!(table.contains("OMEGA(2,1)"));
        assert!(table.contains("OMEGA(2,2)"));
        assert!(table.contains("SIGMA(1)"));
    }

    #[test]
    fn parameter_table_dashes_se_when_absent() {
        let table = parameter_table(&full_param_result(None));
        let cl = table.lines().find(|l| l.starts_with("CL")).expect("CL row");
        // No covariance run → SE/%RSE rendered as "---".
        assert!(cl.contains("---"), "{cl}");
    }

    // ── print_results: stderr smoke test (exercises both header branches) ────

    #[test]
    fn print_results_smoke_single_and_chain() {
        // Single-method, with SEs and a fixed theta.
        print_results(&full_param_result(Some(vec![0.2, 0.0])));
        // Method chain (>1) + no SEs → the alternate header + "N/A" branch.
        let mut chained = full_param_result(None);
        chained.method_chain = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
        chained.converged = false;
        print_results(&chained);
    }

    // ── comprehensive report fixtures: exercise every optional section ────────

    use crate::types::{ImportanceSamplingResult, KappaTreatment};

    /// A FitResult with *every* optional reporting section populated: free +
    /// fixed thetas with SEs, a 2×2 block omega (off-diagonal correlation),
    /// a combined (proportional + additive) sigma, a 2×2 block omega_iov,
    /// importance-sampling results with a low-ESS subject, SIR CIs for
    /// theta/omega/sigma, eta/eps/kappa shrinkage (with a NaN entry), a
    /// computed covariance, and a warning. Drives the maximal branch coverage
    /// of both `print_results` and `write_estimates_yaml` in one pass.
    fn comprehensive_result() -> FitResult {
        let mut r = make_sigma_only_result(ErrorModel::Combined, vec![0.1, 0.5]);
        // theta: one free (with SE), one fixed.
        r.theta = vec![2.0, 0.0];
        r.theta_names = vec!["CL".into(), "BASE".into()];
        r.theta_fixed = vec![false, true];
        r.se_theta = Some(vec![0.2, 0.0]);
        // omega: 2×2 block with an off-diagonal; second eta fixed; param_corr
        // left None so the correlation fallback path is taken.
        r.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.04]);
        r.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
        r.omega_fixed = vec![false, true];
        r.se_omega = Some(vec![0.01, 0.0]);
        r.omega_param_corr = None;
        // sigma: combined → [Proportional, Additive]; one fixed.
        r.sigma_fixed = vec![false, true];
        r.se_sigma = Some(vec![0.01, 0.02]);
        // omega_iov: 2×2 block kappa with an off-diagonal; second kappa fixed.
        r.omega_iov = Some(DMatrix::from_row_slice(2, 2, &[0.05, 0.01, 0.01, 0.03]));
        r.kappa_names = vec!["KAPPA_CL".into(), "KAPPA_V".into()];
        r.kappa_fixed = vec![false, true];
        r.se_kappa = Some(vec![0.005, 0.0]);
        r.omega_iov_param_corr = None;
        // importance sampling, with a low-ESS subject.
        r.importance_sampling = Some(ImportanceSamplingResult {
            minus2_log_likelihood: -1234.5,
            mc_standard_error: 1.2,
            low_ess_subjects: vec![("S3".into(), 0.05)],
            n_samples: 500,
            proposal_df: 5.0,
            ess_min: 0.05,
            ess_median: 0.6,
            kappa_treatment: KappaTreatment::FixedAtMode,
        });
        // SIR CIs for theta/omega/sigma.
        r.sir_ess = Some(123.4);
        r.sir_ci_theta = Some(vec![(1.8, 2.2), (-0.1, 0.1)]);
        r.sir_ci_omega = Some(vec![(0.07, 0.11), (0.03, 0.05)]);
        r.sir_ci_sigma = Some(vec![(0.08, 0.12), (0.4, 0.6)]);
        // Shrinkage (a NaN entry exercises the is_finite guard).
        r.shrinkage_eta = vec![0.12, f64::NAN];
        r.shrinkage_eps = 0.05;
        r.shrinkage_kappa = vec![0.20, f64::NAN];
        r.shrinkage_kappa_by_occ = vec![vec![0.10, f64::NAN]];
        r.covariance_status = CovarianceStatus::Computed;
        // 2 theta + 3 omega (full 2×2) + 2 sigma + 3 kappa (full 2×2) = 10
        r.covariance_matrix = Some(DMatrix::identity(10, 10));
        r.warnings = vec!["example warning".into()];
        r.bayes = Some(crate::types::BayesResult {
            summaries: vec![crate::types::PosteriorSummary {
                name: "CL".into(),
                mean: 2.0,
                sd: 0.2,
                q025: 1.6,
                median: 2.0,
                q975: 2.4,
                rhat: 1.01,
                ess_bulk: 1500.0,
                ess_tail: 1400.0,
                mcse: 0.005,
            }],
            n_chains: 4,
            n_warmup: 1000,
            n_draws_per_chain: 1000,
            n_divergent: 0,
            max_rhat: 1.01,
            draws: None,
        });
        r
    }

    #[test]
    fn write_estimates_yaml_emits_all_sections() {
        let r = comprehensive_result();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&r, path.to_str().unwrap()).expect("yaml write");
        let yaml = std::fs::read_to_string(&path).expect("yaml read");

        // Theta: free with SE, and fixed entry.
        assert!(yaml.contains("  CL:") && yaml.contains("    se: 0.200000"));
        assert!(yaml.contains("  BASE:") && yaml.contains("    fixed: true"));
        // Omega diagonal + off-diagonal covariance block.
        assert!(yaml.contains("\nomega:"));
        assert!(yaml.contains("  ETA_V__ETA_CL:"));
        assert!(yaml.contains("    covariance: 0.020000"));
        // Sigma: combined → proportional emits cv_pct, additive does not.
        assert!(yaml.contains("error model: combined"));
        assert!(yaml.contains("    type: proportional") && yaml.contains("    type: additive"));
        // IOV block + kappa off-diagonal.
        assert!(yaml.contains("\nomega_iov:"));
        assert!(yaml.contains("  KAPPA_V__KAPPA_CL:"));
        // Importance sampling block.
        assert!(yaml.contains("\nimportance_sampling:"));
        assert!(yaml.contains("  kappa_treatment: fixed_at_mode"));
        assert!(yaml.contains("  low_ess_subjects:") && yaml.contains("- id: \"S3\""));
        // Bayesian posterior block.
        assert!(yaml.contains("\nbayes:"));
        assert!(yaml.contains("  max_rhat: 1.0100"));
        assert!(yaml.contains("  parameters:") && yaml.contains("    - name: \"CL\""));
        // SIR section with all three CI blocks.
        assert!(yaml.contains("\nsir:"));
        assert!(
            yaml.contains("  ci_theta:")
                && yaml.contains("  ci_omega:")
                && yaml.contains("  ci_sigma:")
        );
        // Covariance matrix block.
        assert!(
            yaml.contains("\ncovariance_matrix:"),
            "covariance_matrix block missing"
        );
        assert!(
            yaml.contains("  parameters: [CL, BASE, log_chol_ETA_CL, chol_ETA_V_ETA_CL, log_chol_ETA_V, EPS_1, EPS_2, log_chol_KAPPA_CL, chol_KAPPA_V_KAPPA_CL, log_chol_KAPPA_V]"),
            "covariance_matrix parameter list wrong:\n{yaml}"
        );
        // Warnings.
        assert!(yaml.contains("\nwarnings:") && yaml.contains("- \"example warning\""));
    }

    #[test]
    fn write_yaml_emits_npde_seed_only_when_present() {
        // When NPDE ran, the resolved seed is recorded; otherwise the line is absent.
        let mut r = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r.npde_seed = Some(20240601);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&r, path.to_str().unwrap()).expect("yaml write");
        assert!(
            std::fs::read_to_string(&path)
                .unwrap()
                .contains("  npde_seed: 20240601"),
            "npde_seed should be emitted when set"
        );

        r.npde_seed = None;
        write_estimates_yaml(&r, path.to_str().unwrap()).expect("yaml write");
        assert!(
            !std::fs::read_to_string(&path)
                .unwrap()
                .contains("npde_seed"),
            "npde_seed line should be absent when NPDE did not run"
        );
    }

    #[test]
    fn print_results_smoke_comprehensive() {
        // Exercises the maximal-section path of the stderr printer.
        print_results(&comprehensive_result());
    }

    #[test]
    fn write_yaml_and_print_handle_no_se_and_failed_covariance() {
        // No SEs on any non-fixed parameter (→ `se: ~` / N/A arms), a failed
        // covariance step (→ show_cv=false / "FAILED"), an IS block with the
        // Marginalized κ treatment and no low-ESS subjects, and no SIR /
        // shrinkage / warnings.
        let mut r = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r.theta = vec![2.0];
        r.theta_names = vec!["CL".into()];
        r.theta_fixed = vec![false];
        r.se_theta = None;
        r.omega = DMatrix::from_row_slice(1, 1, &[0.09]);
        r.eta_names = vec!["ETA_CL".into()];
        r.omega_fixed = vec![false];
        r.se_omega = None;
        r.sigma_fixed = vec![false];
        r.se_sigma = None;
        r.omega_iov = Some(DMatrix::from_row_slice(1, 1, &[0.04]));
        r.kappa_names = vec!["KAPPA_CL".into()];
        r.kappa_fixed = vec![false];
        r.se_kappa = None;
        r.importance_sampling = Some(ImportanceSamplingResult {
            minus2_log_likelihood: -10.0,
            mc_standard_error: 0.5,
            low_ess_subjects: Vec::new(),
            n_samples: 100,
            proposal_df: 4.0,
            ess_min: 0.9,
            ess_median: 0.95,
            kappa_treatment: KappaTreatment::Marginalized,
        });
        r.covariance_status = CovarianceStatus::Failed;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&r, path.to_str().unwrap()).expect("yaml write");
        let yaml = std::fs::read_to_string(&path).expect("yaml read");
        assert!(
            yaml.contains("    se: ~"),
            "missing-SE entries render as `se: ~`"
        );
        assert!(yaml.contains("  kappa_treatment: marginalized"));
        assert!(yaml.contains("  low_ess_subjects: []"));

        // print_results: failed-covariance branch (no CV%, "FAILED").
        print_results(&r);

        // Flip the IS κ treatment to NotApplicable to cover that match arm too.
        if let Some(is) = r.importance_sampling.as_mut() {
            is.kappa_treatment = KappaTreatment::NotApplicable;
        }
        print_results(&r);
    }

    #[test]
    fn write_yaml_emits_covariance_matrix_block() {
        // Diagonal omega (1 eta), 1 theta, 1 sigma → packed: [CL, log_chol_ETA_CL, EPS_1]
        let mut r = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r.theta = vec![2.0];
        r.theta_names = vec!["CL".into()];
        r.theta_fixed = vec![false];
        r.omega = DMatrix::from_row_slice(1, 1, &[0.09]);
        r.eta_names = vec!["ETA_CL".into()];
        r.omega_fixed = vec![false];
        // 3×3 identity covariance matrix
        r.covariance_matrix = Some(DMatrix::identity(3, 3));
        r.covariance_status = CovarianceStatus::Computed;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("fit.yaml");
        write_estimates_yaml(&r, path.to_str().unwrap()).expect("yaml write");
        let yaml = std::fs::read_to_string(&path).expect("yaml read");

        assert!(
            yaml.contains("\ncovariance_matrix:"),
            "block header missing"
        );
        assert!(
            yaml.contains("  parameters: [CL, log_chol_ETA_CL, EPS_1]"),
            "parameter list wrong:\n{yaml}"
        );
        assert!(yaml.contains("    CL: ["), "CL row missing");
        assert!(
            yaml.contains("    log_chol_ETA_CL: ["),
            "log_chol_ETA_CL row missing"
        );
        assert!(yaml.contains("    EPS_1: ["), "EPS_1 row missing");
        // Identity matrix: diagonal 1 → "1.000000e+0", off-diagonal 0 → "0.000000e+0"
        assert!(
            yaml.contains("1.000000e+0"),
            "diagonal 1 missing or wrong exponent format"
        );
        assert!(
            yaml.contains("0.000000e+0"),
            "off-diagonal 0 missing or wrong exponent format"
        );

        // Full-block omega (2 etas): packed names include chol_ off-diagonal entry
        let mut r2 = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        r2.theta = vec![2.0];
        r2.theta_names = vec!["CL".into()];
        r2.theta_fixed = vec![false];
        r2.omega = DMatrix::from_row_slice(2, 2, &[0.09, 0.02, 0.02, 0.04]);
        r2.eta_names = vec!["ETA_CL".into(), "ETA_V".into()];
        r2.omega_fixed = vec![false, false];
        // 1 theta + 3 omega (full 2×2) + 1 sigma = 5
        r2.covariance_matrix = Some(DMatrix::identity(5, 5));
        r2.covariance_status = CovarianceStatus::Computed;

        let path2 = dir.path().join("fit2.yaml");
        write_estimates_yaml(&r2, path2.to_str().unwrap()).expect("yaml write");
        let yaml2 = std::fs::read_to_string(&path2).expect("yaml read");

        assert!(
            yaml2.contains(
                "  parameters: [CL, log_chol_ETA_CL, chol_ETA_V_ETA_CL, log_chol_ETA_V, EPS_1]"
            ),
            "full-block omega parameter list wrong:\n{yaml2}"
        );
        // No covariance_matrix block when covariance_matrix is None
        let r3 = make_sigma_only_result(ErrorModel::Proportional, vec![0.1]);
        let path3 = dir.path().join("fit3.yaml");
        write_estimates_yaml(&r3, path3.to_str().unwrap()).expect("yaml write");
        let yaml3 = std::fs::read_to_string(&path3).expect("yaml read");
        assert!(
            !yaml3.contains("covariance_matrix:"),
            "block should be absent when covariance_matrix is None"
        );
    }

    #[cfg(feature = "nn")]
    #[test]
    fn print_results_smoke_with_neural_network_block() {
        // Covers the `--- NEURAL NETWORKS ---` print branch.
        print_results(&make_nn_result());
    }
}
