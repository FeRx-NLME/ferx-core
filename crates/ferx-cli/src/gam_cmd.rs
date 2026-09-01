//! `ferx gam` — GAM-based covariate pre-screening (#1114).
//!
//! Thin by construction: parse flags, call [`ferx_tools::gam::gam_screen`],
//! print a ranked table, write a CSV. All math lives in `ferx-tools`.

use ferx_tools::gam::{gam_screen, write_gam_csv, GamOptions};

pub const GAM_USAGE: &str = "\
Usage: ferx gam <model.ferx> --data <data.csv> [--no-fit] [options]
       ferx gam --from-fit <run.fitrx>         [--data <data.csv>] [options]

Screens each declared covariate against each ETA using independent GAM
regressions. Covariates are ranked by delta-AIC = AIC_null - AIC_best; a
positive delta-AIC means the covariate improves the model for that ETA.

This is the Rust equivalent of Xpose4's xpose.gam() (Jonsson & Karlsson 1999).

  --no-fit          skip the optimization loop; compute EBEs at initial
                    parameter values only (equivalent to NONMEM MAXEVAL=0)
  --from-fit PATH   load an existing .fitrx bundle and skip fitting entirely;
                    --data is required when the bundle was saved without it
  --csv PATH        write the ranked table as a CSV
                    (default: {model}-gam.csv)
  --spline-df N     try natural-spline basis with df=N (repeatable, default: 2,3)
  --no-linear       do not include the linear form as a candidate
  --shrink FRAC     shrinkage warning threshold, 0-1 (default: 0.30)
  --threads N       rayon worker count for parallelism (default: auto)

  -h, --help        print this help and exit
";

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Return the value following `name`, or an error when the next token is absent
/// or starts with `-` (which means a missing value, not a file named `-foo`).
fn value<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    let Some(i) = args.iter().position(|a| a == name) else {
        return Ok(None);
    };
    match args.get(i + 1) {
        Some(v) if !v.starts_with('-') => Ok(Some(v.as_str())),
        Some(v) => Err(format!("{name} requires a value but got '{v}'")),
        None => Err(format!("{name} requires a value")),
    }
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
    } else {
        dfs.sort_unstable();
        dfs.dedup();
    }
    Ok(dfs)
}

fn parse_shrink(args: &[String]) -> Result<f64, String> {
    match value(args, "--shrink")? {
        None => Ok(0.30),
        Some(raw) => raw
            .parse::<f64>()
            .ok()
            .filter(|&v| (0.0..=1.0).contains(&v))
            .ok_or_else(|| format!("--shrink requires a fraction between 0 and 1; got '{raw}'")),
    }
}

fn parse_threads(args: &[String]) -> Result<Option<usize>, String> {
    match value(args, "--threads")? {
        None => Ok(None),
        Some(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .map(Some)
            .ok_or_else(|| format!("--threads requires a positive integer; got '{raw}'")),
    }
}

/// Reject any flag that this subcommand does not recognise.
fn check_unknown_flags(args: &[String]) -> Result<(), String> {
    // Flags that consume the following token as their value.
    const VALUE_FLAGS: &[&str] = &[
        "--data",
        "--from-fit",
        "--csv",
        "--spline-df",
        "--shrink",
        "--threads",
    ];
    // Boolean (no-value) flags.
    const BOOL_FLAGS: &[&str] = &["--no-fit", "--no-linear", "-h", "--help"];

    let mut i = 2; // skip `ferx`, `gam`
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with('-') {
            i += 1; // positional (model file)
            continue;
        }
        if VALUE_FLAGS.contains(&a.as_str()) {
            i += 2; // flag + value
            continue;
        }
        if BOOL_FLAGS.contains(&a.as_str()) {
            i += 1;
            continue;
        }
        return Err(format!("unknown flag: {a}"));
    }
    Ok(())
}

/// Entry point for `ferx gam ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    // Help is recognised anywhere in the argument list.
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
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
    check_unknown_flags(args)?;

    let spline_df = parse_spline_df(args)?;
    let shrinkage_warn = parse_shrink(args)?;
    let include_linear = !flag(args, "--no-linear");
    let csv_flag = value(args, "--csv")?;
    let from_fit = value(args, "--from-fit")?;
    let data_path = value(args, "--data")?;
    let no_fit = flag(args, "--no-fit");

    if let Some(n) = parse_threads(args)? {
        if let Err(e) = ferx_core::configure_global_thread_pool(n) {
            eprintln!("Warning: {e}");
        }
    }

    let opts = GamOptions {
        spline_df,
        include_linear,
        shrinkage_warn_threshold: shrinkage_warn,
        ..GamOptions::default()
    };

    // Obtain (FitResult, Population, model_stem) according to mode.
    let (fit, population, model_stem) = if let Some(bundle) = from_fit {
        // ── --from-fit: load an existing .fitrx bundle, skip fitting ─────────
        let loaded = ferx_core::io::fitrx::load_fit(std::path::Path::new(bundle))
            .map_err(|e| format!("failed to load '{bundle}': {e}"))?;

        let population = match loaded.population {
            Some(p) => p,
            None => {
                // Bundle was saved without --include-data; recover the population
                // by re-parsing the embedded model source against --data.
                let dp = data_path.ok_or(
                    "the .fitrx bundle has no embedded population; \
                     re-run with --data <data.csv>",
                )?;
                // Write model source to a temp file so prepare_run can parse it.
                let tmp_path =
                    std::env::temp_dir().join(format!("ferx_gam_{}.ferx", std::process::id()));
                std::fs::write(&tmp_path, &loaded.model_source)
                    .map_err(|e| format!("cannot write temporary model file: {e}"))?;
                let pop_result = ferx_core::prepare_run(tmp_path.to_str().unwrap(), Some(dp));
                let _ = std::fs::remove_file(&tmp_path);
                pop_result?.population
            }
        };

        let stem = std::path::Path::new(bundle)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gam")
            .to_string();
        eprintln!("Loaded fit from '{bundle}'");
        (loaded.fit, population, stem)
    } else {
        // ── Normal / --no-fit: parse model and fit (or eval-only) ────────────
        let model_path = args
            .iter()
            .skip(2)
            .find(|a| !a.starts_with('-'))
            .ok_or("Usage: ferx gam <model.ferx> --data <data.csv>")?;

        let mut prepared = ferx_core::prepare_run(model_path, data_path)?;
        if let Some(w) = &prepared.data_path_warning {
            eprintln!("Warning: {w}");
        }
        prepared.parsed.fit_options.verbose = false;
        prepared.parsed.fit_options.run_covariance_step = false;
        if no_fit {
            // NONMEM MAXEVAL=0 equivalent: compute EBEs at initial params only.
            prepared.parsed.fit_options.outer_maxiter = 0;
        }

        eprintln!(
            "Model: {}  ({} subjects, {} observations)",
            prepared.parsed.model.name,
            prepared.population.subjects.len(),
            prepared.population.n_obs(),
        );
        eprintln!("Covariates: {:?}", prepared.population.covariate_names);

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

        let stem = std::path::Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("gam")
            .to_string();
        (fit, prepared.population, stem)
    };

    // GAM screening.
    let t0 = std::time::Instant::now();
    let gam = gam_screen(&fit, &population, &opts);
    eprintln!(
        "GAM: {:.1} ms  ({} ETAs x {} covariates)",
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

    // Write CSV — explicit --csv path, or default to {stem}-gam.csv.
    let out_path = csv_flag
        .map(String::from)
        .unwrap_or_else(|| format!("{model_stem}-gam.csv"));
    write_gam_csv(&gam, &out_path)?;
    eprintln!("GAM results written to {out_path}");

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

    // ── help ──────────────────────────────────────────────────────────────────

    #[test]
    fn help_exits_zero_at_any_position() {
        assert_eq!(run(&args(&["-h"])), 0);
        assert_eq!(run(&args(&["--help"])), 0);
        assert_eq!(run(&args(&["m.ferx", "--data", "d.csv", "-h"])), 0);
        assert_eq!(run(&args(&["m.ferx", "--help", "--data", "d.csv"])), 0);
    }

    #[test]
    fn missing_model_exits_one() {
        assert_eq!(run(&args(&[])), 1);
    }

    #[test]
    fn bad_model_exits_one() {
        assert_eq!(run(&args(&["definitely-not-a-model.ferx"])), 1);
    }

    // ── unknown flags ─────────────────────────────────────────────────────────

    #[test]
    fn unknown_flag_is_rejected() {
        assert!(check_unknown_flags(&args(&["m.ferx", "--bogus"])).is_err());
        assert!(check_unknown_flags(&args(&["m.ferx", "--data", "d.csv", "--bogus"])).is_err());
    }

    #[test]
    fn known_flags_are_accepted() {
        assert!(check_unknown_flags(&args(&[
            "m.ferx",
            "--data",
            "d.csv",
            "--no-fit",
            "--no-linear",
            "--shrink",
            "0.3",
            "--threads",
            "4",
            "--csv",
            "out.csv",
            "--spline-df",
            "2",
            "--spline-df",
            "3",
        ]))
        .is_ok());
        assert!(
            check_unknown_flags(&args(&["--from-fit", "run.fitrx", "--data", "d.csv"])).is_ok()
        );
    }

    // ── parse_spline_df ───────────────────────────────────────────────────────

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
    fn parse_spline_df_deduplicates() {
        let df =
            parse_spline_df(&args(&["m.ferx", "--spline-df", "3", "--spline-df", "3"])).unwrap();
        assert_eq!(df, vec![3]);
    }

    #[test]
    fn parse_spline_df_rejects_zero() {
        assert!(parse_spline_df(&args(&["m.ferx", "--spline-df", "0"])).is_err());
    }

    // ── parse_shrink ──────────────────────────────────────────────────────────

    #[test]
    fn parse_shrink_defaults_to_30_percent() {
        assert_eq!(parse_shrink(&args(&["m.ferx"])).unwrap(), 0.30);
    }

    #[test]
    fn parse_shrink_rejects_out_of_range() {
        assert!(parse_shrink(&args(&["m.ferx", "--shrink", "1.5"])).is_err());
    }

    #[test]
    fn parse_shrink_rejects_flag_as_value() {
        assert!(parse_shrink(&args(&["m.ferx", "--shrink", "--no-linear"])).is_err());
    }

    #[test]
    fn parse_shrink_rejects_missing_value() {
        assert!(parse_shrink(&args(&["m.ferx", "--shrink"])).is_err());
    }

    // ── parse_threads ─────────────────────────────────────────────────────────

    #[test]
    fn parse_threads_absent_is_none() {
        assert_eq!(parse_threads(&args(&["m.ferx"])).unwrap(), None);
    }

    #[test]
    fn parse_threads_positive_integer_works() {
        assert_eq!(
            parse_threads(&args(&["m.ferx", "--threads", "4"])).unwrap(),
            Some(4)
        );
    }

    #[test]
    fn parse_threads_rejects_zero() {
        assert!(parse_threads(&args(&["m.ferx", "--threads", "0"])).is_err());
    }

    #[test]
    fn parse_threads_rejects_non_integer() {
        assert!(parse_threads(&args(&["m.ferx", "--threads", "auto"])).is_err());
    }

    #[test]
    fn parse_threads_rejects_flag_as_value() {
        assert!(parse_threads(&args(&["m.ferx", "--threads", "--no-linear"])).is_err());
    }

    // ── value helper ──────────────────────────────────────────────────────────

    #[test]
    fn value_absent_is_none() {
        assert_eq!(value(&args(&["m.ferx"]), "--csv").unwrap(), None);
    }

    #[test]
    fn value_returns_next_token() {
        assert_eq!(
            value(&args(&["m.ferx", "--csv", "out.csv"]), "--csv").unwrap(),
            Some("out.csv")
        );
    }

    #[test]
    fn value_rejects_flag_as_value() {
        assert!(value(&args(&["m.ferx", "--csv", "--no-linear"]), "--csv").is_err());
    }

    #[test]
    fn value_rejects_missing_value() {
        assert!(value(&args(&["m.ferx", "--csv"]), "--csv").is_err());
    }
}
