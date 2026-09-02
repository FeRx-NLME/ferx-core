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
  --csv PATH        write the ranked table to PATH instead of the default
                    (default: {model}-gam.csv, alongside {model}-sdtab.csv)
  --no-csv          print the table only; write no file
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

/// Flags that consume the following token as their value.
const VALUE_FLAGS: &[&str] = &[
    "--data",
    "--from-fit",
    "--csv",
    "--spline-df",
    "--shrink",
    "--threads",
];
/// Boolean (no-value) flags.
const BOOL_FLAGS: &[&str] = &["--no-fit", "--no-linear", "--no-csv", "-h", "--help"];

/// Walk the argument list once, rejecting unknown flags and returning the
/// positional arguments in order.
///
/// The walk has to know which flags consume the following token, because
/// "the first token that does not start with `-`" is not the model file: in
/// `ferx gam --threads 4 model.ferx` that rule picks `4`, and in
/// `ferx gam --data d.csv model.ferx` it picks `d.csv`. Both then fail while
/// parsing the wrong file, with an error naming a path the user never gave as a
/// model. Skipping each value flag's argument is what makes the positional list
/// mean what it says.
fn scan_args(args: &[String]) -> Result<Vec<&str>, String> {
    let mut positionals = Vec::new();
    let mut i = 2; // skip `ferx`, `gam`
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with('-') {
            positionals.push(a.as_str());
            i += 1;
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
    Ok(positionals)
}

/// The model file: the single positional argument, if there is exactly one.
fn model_path(args: &[String]) -> Result<&str, String> {
    let positionals = scan_args(args)?;
    match positionals.len() {
        0 => Err("Usage: ferx gam <model.ferx> --data <data.csv>".into()),
        1 => Ok(positionals[0]),
        _ => Err(format!(
            "expected one model file but got {}: {}. \
             (Options take their value after the flag, e.g. --data <data.csv>.)",
            positionals.len(),
            positionals.join(", ")
        )),
    }
}

/// Returns `true` when `--no-fit` (outer_maxiter=0) is safe for this method.
///
/// The `outer_maxiter == 0` short-circuit lives in `optimize_population` (the
/// `_ =>` arm in `api/fit.rs`), so it only applies to methods that actually
/// reach that arm. Two families do not:
///
/// - **SAEM / IMP / IMPMAP / Bayes / VI** each dispatch to their own runner
///   first, so `outer_maxiter = 0` has no effect at all. The exception is
///   `imp_eval_only` (NONMEM `EONLY=1`), which evaluates `-2 log L` at the
///   fixed input parameters without updating them — that is already what
///   `--no-fit` asks for, so it is allowed.
/// - **`gn_hybrid`** is dispatched to `gauss_newton::run_foce_gn` alongside
///   `gn`, bypassing `optimize_population` entirely. Its GN loop is
///   `0..outer_maxiter` and so does run zero iterations — but it then falls
///   through to the FOCEI polish, which **hard-sets**
///   `polish_options.outer_maxiter = 100` (`estimation/gauss_newton.rs`)
///   rather than deriving it from `options`. So `--no-fit` on a `gn_hybrid`
///   model would quietly perform a 100-iteration estimation. Pure `gn` is safe,
///   but only because it returns before reaching that polish.
fn no_fit_supported(m: ferx_core::EstimationMethod, imp_eval_only: bool) -> bool {
    use ferx_core::EstimationMethod::*;
    match m {
        Imp => imp_eval_only,
        Saem | Impmap | Bayes | Vi | FoceGnHybrid => false,
        _ => true,
    }
}

/// `gam_screen` pairs `fit.subjects[i]` with `population.subjects[i]` **by
/// index**. That holds by construction everywhere the two come from the same
/// run, but not for `--from-fit <bundle> --data <csv>`, where the bundle was
/// saved without its population and the CSV is whatever the user passed.
///
/// Left unchecked, a differing subject count reaches `gam_screen_raw`'s length
/// assertion and aborts the CLI with a raw panic and backtrace; worse, a
/// matching count in a different order (a re-sorted CSV, a different
/// `[data_selection]` filter) pairs every eta with the wrong subject's
/// covariates and silently produces a wrong ranking — which the gam module's
/// own docs call the worst outcome for a tool whose entire output is an
/// ordering. Compare the ids and refuse instead.
fn check_subject_alignment(
    fit: &ferx_core::FitResult,
    population: &ferx_core::Population,
) -> Result<(), String> {
    if fit.subjects.len() != population.subjects.len() {
        return Err(format!(
            "the saved fit has {} subjects but the data in --data has {}. \
             The bundle was saved without its population, so --data must be the \
             same dataset the fit was run on.",
            fit.subjects.len(),
            population.subjects.len()
        ));
    }
    for (i, (f, p)) in fit.subjects.iter().zip(&population.subjects).enumerate() {
        if f.id != p.id {
            return Err(format!(
                "subject {} of the saved fit is '{}' but subject {} of --data is '{}'. \
                 The eta estimates are matched to covariates by position, so a \
                 reordered or filtered dataset would produce a silently wrong ranking.",
                i + 1,
                f.id,
                i + 1,
                p.id
            ));
        }
    }
    Ok(())
}

/// Re-read a population by parsing `model_source` against `data_path`.
///
/// `prepare_run` takes a path, so the embedded source has to reach the disk
/// first. It goes into a **freshly created private directory** rather than a
/// name derived from the pid: `env::temp_dir().join(format!("ferx_gam_{pid}.ferx"))`
/// is guessable, and `fs::write` follows symlinks, so on a shared machine
/// another user can pre-place a symlink at that path and have this process
/// overwrite the target. `create_dir` fails rather than following an existing
/// entry, which makes claiming the directory the atomic step.
fn population_from_model_source(
    model_source: &str,
    data_path: &str,
) -> Result<ferx_core::Population, String> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("ferx-gam-{}-{nanos:x}", std::process::id()));
    std::fs::create_dir(&dir)
        .map_err(|e| format!("cannot create temporary directory {}: {e}", dir.display()))?;

    let model_file = dir.join("model.ferx");
    let result = (|| {
        std::fs::write(&model_file, model_source)
            .map_err(|e| format!("cannot write temporary model file: {e}"))?;
        let path = model_file
            .to_str()
            .ok_or("temporary model path is not valid UTF-8")?;
        Ok(ferx_core::prepare_run(path, Some(data_path))?.population)
    })();

    let _ = std::fs::remove_dir_all(&dir);
    result
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
    scan_args(args)?;

    let spline_df = parse_spline_df(args)?;
    let shrinkage_warn = parse_shrink(args)?;
    let include_linear = !flag(args, "--no-linear");
    let csv_flag = value(args, "--csv")?;
    let no_csv = flag(args, "--no-csv");
    let from_fit = value(args, "--from-fit")?;
    let data_path = value(args, "--data")?;
    let no_fit = flag(args, "--no-fit");

    if no_csv && csv_flag.is_some() {
        return Err("--no-csv and --csv are mutually exclusive".into());
    }

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
                let pop = population_from_model_source(&loaded.model_source, dp)?;
                // The fit and this population now come from different places, so
                // nothing guarantees they describe the same subjects in the same
                // order. `gam_screen` pairs them by index.
                check_subject_alignment(&loaded.fit, &pop)?;
                pop
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
        let model_path = model_path(args)?;

        let mut prepared = ferx_core::prepare_run(model_path, data_path)?;
        if let Some(w) = &prepared.data_path_warning {
            eprintln!("Warning: {w}");
        }
        prepared.parsed.fit_options.verbose = false;
        prepared.parsed.fit_options.run_covariance_step = false;
        if no_fit {
            // NONMEM MAXEVAL=0 equivalent: compute EBEs at initial params only.
            // Guard: the outer_maxiter=0 short-circuit only applies to methods
            // that go through optimize_population (FOCE, Laplace, GN, …); see
            // `no_fit_supported` for the two families that do not.
            let opts = &prepared.parsed.fit_options;
            let eval_only = opts.imp_eval_only;
            if let Some(bad) = opts
                .method_chain()
                .into_iter()
                .find(|m| !no_fit_supported(*m, eval_only))
            {
                return Err(format!(
                    "--no-fit is not supported with method '{}': \
                     outer_maxiter=0 only short-circuits the FOCE/BFGS \
                     outer loop. SAEM, IMP, IMPMAP, Bayes, and VI have their \
                     own runners, and gn_hybrid falls through to a FOCEI polish \
                     with a hard-coded 100 iterations — all of which would run a \
                     full estimation. Run the fit first and use --from-fit instead.",
                    bad.label()
                ));
            }
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

    print_gam_table(&gam);

    // Write CSV — `--no-csv` suppresses it, `--csv PATH` redirects it, and the
    // default `{stem}-gam.csv` matches how a plain `ferx` run writes
    // `{model}-sdtab.csv` next to the model.
    if !no_csv {
        let out_path = csv_flag
            .map(String::from)
            .unwrap_or_else(|| format!("{model_stem}-gam.csv"));
        write_gam_csv(&gam, &out_path)?;
        eprintln!("GAM results written to {out_path}");
    }

    Ok(0)
}

/// Run the screen on an already-computed fit and write `{model_stem}-gam.csv`.
///
/// This is the body of the main fit path's `--gam` flag. It lives here, rather
/// than inline in `main`, so it is reachable from a test — `main` itself is not.
pub fn run_after_fit(
    fit: &ferx_core::FitResult,
    population: &ferx_core::Population,
    model_stem: &str,
) -> Result<String, String> {
    let gam = gam_screen(fit, population, &GamOptions::default());
    for w in &gam.warnings {
        eprintln!("GAM Warning: {w}");
    }
    print_gam_table(&gam);
    let out_path = format!("{model_stem}-gam.csv");
    write_gam_csv(&gam, &out_path)?;
    Ok(out_path)
}

/// Print the ranked delta-AIC table, one block per ETA.
fn print_gam_table(gam: &ferx_tools::gam::GamResult) {
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
        assert!(scan_args(&args(&["m.ferx", "--bogus"])).is_err());
        assert!(scan_args(&args(&["m.ferx", "--data", "d.csv", "--bogus"])).is_err());
    }

    #[test]
    fn known_flags_are_accepted() {
        assert!(scan_args(&args(&[
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
        assert!(scan_args(&args(&["--from-fit", "run.fitrx", "--data", "d.csv"])).is_ok());
        assert!(scan_args(&args(&["m.ferx", "--data", "d.csv", "--no-csv"])).is_ok());
    }

    // ── positional arguments ──────────────────────────────────────────────────

    #[test]
    fn a_flag_value_is_not_mistaken_for_the_model_file() {
        // The regression: "first token not starting with -" picks `4` here.
        assert_eq!(
            model_path(&args(&["--threads", "4", "m.ferx", "--data", "d.csv"])).unwrap(),
            "m.ferx"
        );
        // ... and `d.csv` here.
        assert_eq!(
            model_path(&args(&["--data", "d.csv", "m.ferx"])).unwrap(),
            "m.ferx"
        );
        // A repeated value flag consumes each of its own values.
        assert_eq!(
            model_path(&args(&[
                "--spline-df",
                "2",
                "--spline-df",
                "3",
                "m.ferx",
                "--data",
                "d.csv",
            ]))
            .unwrap(),
            "m.ferx"
        );
    }

    #[test]
    fn the_model_file_is_found_in_the_ordinary_position() {
        assert_eq!(
            model_path(&args(&["m.ferx", "--data", "d.csv"])).unwrap(),
            "m.ferx"
        );
    }

    #[test]
    fn no_model_file_is_an_error() {
        assert!(model_path(&args(&["--data", "d.csv"])).is_err());
    }

    #[test]
    fn two_model_files_are_rejected_rather_than_silently_picking_one() {
        let err = model_path(&args(&["a.ferx", "b.ferx", "--data", "d.csv"])).unwrap_err();
        assert!(err.contains("a.ferx") && err.contains("b.ferx"), "{err}");
    }

    // ── --no-csv ──────────────────────────────────────────────────────────────

    #[test]
    fn no_csv_and_csv_together_are_rejected() {
        let err = run_gam(&args(&["m.ferx", "--no-csv", "--csv", "out.csv"])).unwrap_err();
        assert!(err.contains("mutually exclusive"), "{err}");
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

    // ── no_fit_supported ──────────────────────────────────────────────────────

    #[test]
    fn no_fit_supported_allows_foce_variants() {
        use ferx_core::EstimationMethod::*;
        assert!(no_fit_supported(Foce, false));
        assert!(no_fit_supported(FoceI, false));
        assert!(no_fit_supported(FoceGn, false));
        assert!(no_fit_supported(Laplace, false));
    }

    #[test]
    fn no_fit_supported_rejects_own_runner_methods() {
        use ferx_core::EstimationMethod::*;
        assert!(!no_fit_supported(Saem, false));
        assert!(!no_fit_supported(Imp, false));
        assert!(!no_fit_supported(Impmap, false));
        assert!(!no_fit_supported(Bayes, false));
        assert!(!no_fit_supported(Vi, false));
    }

    #[test]
    fn no_fit_supported_rejects_gn_hybrid() {
        // `gn_hybrid` bypasses `optimize_population` (it dispatches to
        // `gauss_newton::run_foce_gn` alongside `gn`), and its FOCEI polish
        // hard-sets `outer_maxiter = 100` rather than reading it from the
        // options — so `--no-fit` would run a full 100-iteration estimation.
        // Pure `gn` returns before that polish and stays allowed.
        use ferx_core::EstimationMethod::*;
        assert!(!no_fit_supported(FoceGnHybrid, false));
        assert!(!no_fit_supported(FoceGnHybrid, true));
        assert!(no_fit_supported(FoceGn, false));
    }

    #[test]
    fn no_fit_supported_allows_eval_only_imp() {
        // `imp_eval_only` (NONMEM EONLY=1) evaluates -2logL at the fixed input
        // parameters without updating them, which is what --no-fit asks for.
        use ferx_core::EstimationMethod::*;
        assert!(no_fit_supported(Imp, true));
        // The flag is IMP-specific: it does not excuse the other runners.
        assert!(!no_fit_supported(Saem, true));
        assert!(!no_fit_supported(Impmap, true));
        assert!(!no_fit_supported(Bayes, true));
        assert!(!no_fit_supported(Vi, true));
    }

    // ── subject alignment (--from-fit + --data) ───────────────────────────────

    const MODEL: &str = "../../examples/two_cpt_oral_cov.ferx";
    const DATA: &str = "../../data/two_cpt_oral_cov.csv";

    /// Fit the covariate model with `outer_maxiter = 0` — the `--no-fit` path —
    /// to get a real `(FitResult, Population)` pair. It declares WT and CRCL in
    /// a `[covariates]` block, so the screen actually has something to rank;
    /// `warfarin.ferx` declares none and yields a header-only CSV.
    fn cov_model_eval_only() -> (ferx_core::FitResult, ferx_core::Population) {
        let mut prepared = ferx_core::prepare_run(MODEL, Some(DATA)).expect("prepare model");
        prepared.parsed.fit_options.verbose = false;
        prepared.parsed.fit_options.run_covariance_step = false;
        prepared.parsed.fit_options.outer_maxiter = 0;
        let fit = ferx_core::fit(
            &prepared.parsed.model,
            &prepared.population,
            &prepared.init_params,
            &prepared.parsed.fit_options,
        )
        .expect("eval-only fit");
        (fit, prepared.population)
    }

    #[test]
    fn a_matching_population_passes_the_alignment_check() {
        let (fit, population) = cov_model_eval_only();
        assert!(check_subject_alignment(&fit, &population).is_ok());
    }

    #[test]
    fn a_shorter_population_is_refused_rather_than_asserted_on() {
        // Without the check this reaches `gam_screen_raw`'s length assertion and
        // aborts the CLI with a panic and a backtrace.
        let (fit, mut population) = cov_model_eval_only();
        population.subjects.pop();
        let err = check_subject_alignment(&fit, &population).unwrap_err();
        assert!(err.contains("subjects"), "{err}");
    }

    #[test]
    fn a_reordered_population_is_refused_rather_than_silently_mispaired() {
        // Same subject count, different order: every eta would be paired with
        // the wrong subject's covariates and the ranking would be wrong with no
        // sign of it.
        let (fit, mut population) = cov_model_eval_only();
        population.subjects.swap(0, 1);
        let err = check_subject_alignment(&fit, &population).unwrap_err();
        assert!(err.contains("position"), "{err}");
    }

    // ── end-to-end ────────────────────────────────────────────────────────────

    #[test]
    fn no_fit_runs_end_to_end_and_writes_the_csv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let csv = dir.path().join("gam.csv");
        let code = run(&args(&[
            MODEL,
            "--data",
            DATA,
            "--no-fit",
            "--csv",
            csv.to_str().unwrap(),
        ]));
        assert_eq!(code, 0);

        let text = std::fs::read_to_string(&csv).expect("gam csv written");
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap(),
            "eta_name,covariate,delta_aic,best_form,aic,aic_null,r_squared,shrinkage"
        );
        // At least one screened (eta, covariate) row, each with all 8 columns.
        let rows: Vec<&str> = lines.collect();
        assert!(!rows.is_empty(), "no rows written");
        for row in &rows {
            assert_eq!(row.split(',').count(), 8, "row has wrong arity: {row}");
        }
    }

    #[test]
    fn no_csv_prints_the_table_but_writes_nothing() {
        // Without `--csv` the default path is `{stem}-gam.csv` in the working
        // directory. `--no-csv` must leave it uncreated. No other test writes
        // this name (the end-to-end test directs its output into a tempdir).
        let default_path = std::path::Path::new("two_cpt_oral_cov-gam.csv");
        let _ = std::fs::remove_file(default_path);

        let code = run(&args(&[MODEL, "--data", DATA, "--no-fit", "--no-csv"]));
        assert_eq!(code, 0);
        assert!(
            !default_path.exists(),
            "--no-csv wrote {}",
            default_path.display()
        );
    }

    #[test]
    fn run_after_fit_writes_the_default_named_csv() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stem = dir.path().join("two_cpt_oral_cov");
        let (fit, population) = cov_model_eval_only();
        let path = run_after_fit(&fit, &population, stem.to_str().unwrap()).expect("gam csv");
        assert!(path.ends_with("two_cpt_oral_cov-gam.csv"), "{path}");
        let text = std::fs::read_to_string(&path).expect("csv written");
        assert!(text.starts_with("eta_name,covariate,"), "{text}");
    }
}
