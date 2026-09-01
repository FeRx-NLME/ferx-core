//! `ferx gam` — GAM-based covariate pre-screening (#1114).
//!
//! Thin by construction: parse flags, call [`ferx_tools::gam::gam_screen`],
//! print a ranked table, optionally write a CSV. All math lives in
//! `ferx-tools`.

use ferx_tools::gam::{gam_screen, write_gam_csv, GamOptions};

pub const GAM_USAGE: &str = "\
Usage: ferx gam <model.ferx> --data <data.csv> [options]

Fits a base model (no covariates in the structural parameters) and screens
each declared covariate against each ETA using independent GAM regressions.
Covariates are ranked by ΔAIC = AIC_null − AIC_best; a positive ΔAIC means
the covariate improves the model for that ETA.

This is the Rust equivalent of Xpose4's xpose.gam() (Jonsson & Karlsson 1999).

  --output PATH   write the ranked table as a CSV (default: off)
  --spline-df N   try natural-spline basis with df=N (repeatable, default: 2,3)
  --no-linear     do not include the linear form as a candidate
  --shrink FRAC   shrinkage warning threshold, 0–1 (default: 0.30)
  --threads N     rayon worker count for parallelism (default: auto)

  -h, --help      print this help and exit
";

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn value<'a>(args: &'a [String], name: &str) -> Option<&'a String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
}

fn parse_spline_df(args: &[String]) -> Result<Vec<usize>, String> {
    let mut dfs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--spline-df" {
            match args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
                Some(df) if df >= 1 => {
                    dfs.push(df);
                    i += 2;
                    continue;
                }
                _ => {
                    return Err(
                        "--spline-df requires a positive integer (e.g. --spline-df 3)".into(),
                    )
                }
            }
        }
        i += 1;
    }
    if dfs.is_empty() {
        dfs = vec![2, 3];
    }
    Ok(dfs)
}

fn parse_shrink(args: &[String]) -> Result<f64, String> {
    match value(args, "--shrink") {
        None => Ok(0.30),
        Some(raw) => raw
            .parse::<f64>()
            .ok()
            .filter(|&v| (0.0..=1.0).contains(&v))
            .ok_or_else(|| format!("--shrink requires a fraction between 0 and 1; got '{raw}'")),
    }
}

fn parse_threads(args: &[String]) -> Option<usize> {
    value(args, "--threads")?
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
}

/// Entry point for `ferx gam ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args.get(2).map(String::as_str) == Some("-h")
        || args.get(2).map(String::as_str) == Some("--help")
    {
        print!("{GAM_USAGE}");
        return 0;
    }

    match run_gam(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_gam(args: &[String]) -> Result<i32, String> {
    // First positional after `gam` is the model file.
    let model_path = args
        .get(2)
        .filter(|a| !a.starts_with('-'))
        .ok_or("Usage: ferx gam <model.ferx> --data <data.csv>")?;

    let data_path = value(args, "--data").map(String::as_str);

    let spline_df = parse_spline_df(args)?;
    let shrinkage_warn = parse_shrink(args)?;
    let include_linear = !flag(args, "--no-linear");
    let output_path = value(args, "--output").cloned();

    if let Some(n) = parse_threads(args) {
        if let Err(e) = ferx_core::configure_global_thread_pool(n) {
            eprintln!("Warning: {e}");
        }
    }

    // Prepare: parse model + read data.
    let mut prepared = ferx_core::prepare_run(model_path, data_path)?;
    if let Some(w) = &prepared.data_path_warning {
        eprintln!("Warning: {w}");
    }
    prepared.parsed.fit_options.verbose = false;
    prepared.parsed.fit_options.run_covariance_step = false;

    eprintln!(
        "Model: {}  ({} subjects, {} observations)",
        prepared.parsed.model.name,
        prepared.population.subjects.len(),
        prepared.population.n_obs(),
    );
    eprintln!("Covariates: {:?}", prepared.population.covariate_names);

    // Fit the base model.
    let started = std::time::Instant::now();
    let fit = ferx_core::fit(
        &prepared.parsed.model,
        &prepared.population,
        &prepared.init_params,
        &prepared.parsed.fit_options,
    )?;
    eprintln!(
        "Fit: converged={}, OFV={:.3}, {:.1}s",
        fit.converged,
        fit.ofv,
        started.elapsed().as_secs_f64(),
    );

    // GAM screening.
    let opts = GamOptions {
        spline_df,
        include_linear,
        shrinkage_warn_threshold: shrinkage_warn,
        ..GamOptions::default()
    };

    let t0 = std::time::Instant::now();
    let gam = gam_screen(&fit, &prepared.population, &opts);
    eprintln!(
        "GAM: {:.1} ms  ({} ETAs × {} covariates)",
        t0.elapsed().as_secs_f64() * 1000.0,
        gam.eta_results.len(),
        gam.eta_results
            .first()
            .map_or(0, |e| e.covariate_scores.len()),
    );

    // Warnings.
    for w in &gam.warnings {
        eprintln!("Warning: {w}");
    }

    // Print ranked table.
    for eta_res in &gam.eta_results {
        println!(
            "\n{} (shrinkage {:.1}%, null AIC {:.2})",
            eta_res.eta_name,
            eta_res.shrinkage * 100.0,
            eta_res.aic_null,
        );
        if eta_res.covariate_scores.is_empty() {
            println!("  (no covariates screened)");
        } else {
            println!(
                "  {:>12}  {:>10}  {:>8}  {:>8}  form",
                "covariate", "delta_aic", "aic", "r2"
            );
            for s in &eta_res.covariate_scores {
                let form = match &s.best_form {
                    ferx_tools::gam::CovariateForm::Linear => "Linear".to_string(),
                    ferx_tools::gam::CovariateForm::Spline { df } => format!("Spline(df={df})"),
                    ferx_tools::gam::CovariateForm::Categorical => "Categorical".to_string(),
                };
                println!(
                    "  {:>12}  {:>10.3}  {:>8.2}  {:>8.4}  {}",
                    s.covariate, s.delta_aic, s.aic, s.r_squared, form,
                );
            }
        }
    }

    // Optional CSV output.
    if let Some(ref path) = output_path {
        write_gam_csv(&gam, path)?;
        eprintln!("Wrote GAM results to {path}");
    }

    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["ferx".to_string(), "gam".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn help_exits_zero() {
        assert_eq!(run(&args(&["-h"])), 0);
        assert_eq!(run(&args(&["--help"])), 0);
    }

    #[test]
    fn missing_model_exits_one() {
        assert_eq!(run(&args(&[])), 1);
    }

    #[test]
    fn bad_model_exits_one() {
        assert_eq!(run(&args(&["definitely-not-a-model.ferx"])), 1);
    }

    #[test]
    fn parse_spline_df_defaults_to_2_3() {
        let df = parse_spline_df(&args(&["m.ferx"])).unwrap();
        assert_eq!(df, vec![2, 3]);
    }

    #[test]
    fn parse_spline_df_accepts_repeated_flag() {
        let df =
            parse_spline_df(&args(&["m.ferx", "--spline-df", "2", "--spline-df", "4"])).unwrap();
        assert_eq!(df, vec![2, 4]);
    }

    #[test]
    fn parse_shrink_defaults_to_30_percent() {
        assert_eq!(parse_shrink(&args(&["m.ferx"])).unwrap(), 0.30);
    }

    #[test]
    fn parse_shrink_rejects_out_of_range() {
        assert!(parse_shrink(&args(&["m.ferx", "--shrink", "1.5"])).is_err());
    }
}
