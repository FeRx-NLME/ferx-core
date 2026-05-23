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
    let show_cv = result.covariance_status != CovarianceStatus::Failed;
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
            match &result.se_omega {
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
                eprintln!(
                    "  {} × {} = {:.6}  (param corr = {:.4})",
                    name_i, name_j, cov, param_corr,
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

    // Run info
    eprintln!("\n--- Run Info ---");
    let cov_str = match result.covariance_status {
        crate::types::CovarianceStatus::Computed => "computed",
        crate::types::CovarianceStatus::Failed => "FAILED",
        crate::types::CovarianceStatus::NotRequested => "not requested",
    };
    eprintln!("  Covariance: {}", cov_str);
    eprintln!("  Wall time:  {:.1}s", result.wall_time_secs);
    eprintln!("  ferx v{}", result.ferx_version);

    // Warnings
    if !result.warnings.is_empty() {
        eprintln!("\n--- Warnings ---");
        for w in &result.warnings {
            eprintln!("  * {}", w);
        }
    }

    eprintln!("{}\n", "=".repeat(60));
}

/// Generate SDTAB-like output as vectors of (header, values) pairs
pub fn sdtab(result: &FitResult, population: &Population) -> Vec<(String, Vec<f64>)> {
    let n_total: usize = result.subjects.iter().map(|s| s.ipred.len()).sum();

    let any_cens = result
        .subjects
        .iter()
        .any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = population.subjects.iter().any(|s| !s.occasions.is_empty());

    let mut ids = Vec::with_capacity(n_total);
    let mut times = Vec::with_capacity(n_total);
    let mut dvs = Vec::with_capacity(n_total);
    let mut cens_col = Vec::with_capacity(n_total);
    let mut occ_col = Vec::with_capacity(n_total);
    let mut preds = Vec::with_capacity(n_total);
    let mut ipreds = Vec::with_capacity(n_total);
    let mut cwres_vec = Vec::with_capacity(n_total);
    let mut iwres_vec = Vec::with_capacity(n_total);
    let mut ebe_ofv_col = Vec::with_capacity(n_total);
    let mut n_obs_col = Vec::with_capacity(n_total);

    for (si, sr) in result.subjects.iter().enumerate() {
        let subj = &population.subjects[si];
        for j in 0..sr.ipred.len() {
            ids.push(si as f64 + 1.0);
            times.push(subj.obs_times[j]);
            dvs.push(subj.observations[j]);
            cens_col.push(sr.cens.get(j).copied().unwrap_or(0) as f64);
            occ_col.push(subj.occasions.get(j).copied().unwrap_or(0) as f64);
            preds.push(sr.pred[j]);
            ipreds.push(sr.ipred[j]);
            cwres_vec.push(sr.cwres[j]);
            iwres_vec.push(sr.iwres[j]);
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
    cols.extend([
        ("PRED".to_string(), preds),
        ("IPRED".to_string(), ipreds),
        ("CWRES".to_string(), cwres_vec),
        ("IWRES".to_string(), iwres_vec),
        ("EBE_OFV".to_string(), ebe_ofv_col),
        ("N_OBS".to_string(), n_obs_col),
    ]);

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

    // Rows. NaN (used for BLOQ IWRES/CWRES) is written as an empty cell so
    // downstream tools handle it as missing rather than a sentinel.
    for row in 0..n_rows {
        let vals: Vec<String> = cols
            .iter()
            .map(|(_, values)| {
                let v = values[row];
                if v.is_nan() {
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
        let se = result.se_omega.as_ref().and_then(|v| v.get(i).copied());
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
                writeln!(f, "  {}__{}:", name_i, name_j).map_err(|e| e.to_string())?;
                writeln!(f, "    covariance: {:.6}", cov).map_err(|e| e.to_string())?;
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
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            omega_iov: None,
            kappa_names: Vec::new(),
            kappa_fixed: Vec::new(),
            kappa_init_as_sd: Vec::new(),
            se_kappa: None,
            shrinkage_kappa: Vec::new(),
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
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
        }
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
}
