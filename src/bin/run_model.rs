use ferx_core::NcaInit;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // `ferx check ...` is a separate, non-fitting subcommand: parse + validate a
    // model (optionally against data) and report structured diagnostics. Dispatch
    // before the fit/simulate path so the rest of main() is unchanged.
    if args.get(1).map(String::as_str) == Some("check") {
        std::process::exit(run_check(&args));
    }

    if args.len() < 2 {
        eprintln!("Usage: ferx <model.ferx> --data <data.csv> [--threads N|auto] [--output <run.fitrx>] [--include-data] [--inits-from-nca[=nca|nca_sweep|nca_ebe]]");
        eprintln!("       ferx <model.ferx> --simulate          [--threads N|auto] [--output <run.fitrx>]");
        eprintln!("       ferx check <model.ferx> [--data <data.csv>] [--json]");
        eprintln!();
        eprintln!("Fits a NLME model and writes sdtab.csv with residuals.");
        eprintln!("Data must be in NONMEM format (ID, TIME, DV, EVID, AMT, CMT, ...)");
        eprintln!();
        eprintln!("--threads N    use N rayon workers (N > 0)");
        eprintln!("--threads 0    use rayon default (one worker per logical CPU)");
        eprintln!("--threads auto alias for --threads 0");
        eprintln!();
        eprintln!("--output PATH  also write a portable .fitrx fit bundle (zip of JSON+CSV)");
        eprintln!("--include-data embed the input --data CSV inside the .fitrx (off by default)");
        eprintln!();
        eprintln!("--inits-from-nca[=METHOD]  derive NCA-based starting values before fitting,");
        eprintln!("               overriding the model file. METHOD is nca, nca_sweep (default),");
        eprintln!("               or nca_ebe; a bare --inits-from-nca means nca_sweep.");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let data_path = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1));
    let simulate = args.iter().any(|a| a == "--simulate");
    let threads = parse_threads_flag(&args);
    let inits_from_nca = match parse_inits_from_nca_flag(&args) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let output_path = parse_output_flag(&args);
    let include_data = args.iter().any(|a| a == "--include-data");
    if include_data && output_path.is_none() {
        eprintln!("Warning: --include-data has no effect without --output");
    }
    if include_data && data_path.is_none() {
        eprintln!("Warning: --include-data ignored (no --data file to embed)");
    }

    // Honor --threads by sizing rayon's global pool (build_global() is once-per-process,
    // correct for a CLI binary) so fit()'s default pool — sized to current_num_threads()
    // — inherits the count. The 32 MiB worker stack that wide ODE+IOV analytic gradients
    // need is applied by fit()'s own fit-scoped pool (api::default_fit_pool), so the
    // global pool keeps the platform-default stack here rather than reserving a second
    // 32 MiB × N. Without --threads, fit() sizes its pool to all cores.
    if let Some(n) = threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            eprintln!("Warning: failed to configure thread pool with {n} threads: {e}");
        }
    }

    let t_start = Instant::now();
    let result = if let Some(csv_path) = data_path {
        ferx_core::run_model_with_data_inits(model_path, csv_path, inits_from_nca)
    } else if simulate {
        ferx_core::run_model_simulate(model_path)
    } else {
        eprintln!("Error: specify --data <file.csv> or --simulate");
        std::process::exit(1);
    };
    let elapsed = t_start.elapsed();

    match result {
        Ok((fit_result, population)) => {
            // CLI prints the human-readable summary. The library `fit()` no longer
            // prints it — language bindings (e.g. ferx-r's print.ferx_fit) are the
            // single source of truth for formatted summaries (see issue #60).
            ferx_core::io::output::print_results(&fit_result);
            // Measurement only (no-op unless FERX_PROFILE=1).
            ferx_core::pk::event_driven::profile_report();
            ferx_core::sens::provider::profile_report();
            ferx_core::estimation::inner_optimizer::profile_report();

            // Derive model name from model file path
            let model_name = std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            let sdtab_path = format!("{}-sdtab.csv", model_name);
            match ferx_core::io::output::write_sdtab_csv(&fit_result, &population, &sdtab_path) {
                Ok(()) => eprintln!("Residuals written to {}", sdtab_path),
                Err(e) => eprintln!("Warning: failed to write sdtab: {}", e),
            }

            // Covariate table — only when the model declared a [covariates] block.
            if let Some(table) = &fit_result.covariate_table {
                let covtab_path = format!("{}-covtab.csv", model_name);
                match ferx_core::io::output::write_covtab_csv(table, &covtab_path) {
                    Ok(()) => eprintln!("Covariates written to {}", covtab_path),
                    Err(e) => eprintln!("Warning: failed to write covtab: {}", e),
                }
            }

            // SAEM conditional-distribution outputs (only when the pass ran).
            for msg in ferx_core::io::output::write_conddist_outputs(&fit_result, &model_name) {
                eprintln!("{}", msg);
            }

            let yaml_path = format!("{}-fit.yaml", model_name);
            match ferx_core::io::output::write_estimates_yaml(&fit_result, &yaml_path) {
                Ok(()) => eprintln!("Estimates written to {}", yaml_path),
                Err(e) => eprintln!("Warning: failed to write estimates: {}", e),
            }

            if let Some(out) = &output_path {
                let model_source = std::fs::read_to_string(model_path).unwrap_or_default();
                let include = if include_data {
                    data_path.map(std::path::PathBuf::from)
                } else {
                    None
                };
                let opts = ferx_core::io::fitrx::SaveFitOptions {
                    include_data: include,
                };
                match ferx_core::io::fitrx::save_fit(
                    &fit_result,
                    &population,
                    &model_source,
                    std::path::Path::new(out),
                    opts,
                ) {
                    Ok(()) => eprintln!("Fit bundle written to {}", out),
                    Err(e) => eprintln!("Warning: failed to write fit bundle: {}", e),
                }
            }

            let elapsed_secs = elapsed.as_secs_f64();
            eprintln!("Elapsed fit time: {:.3}s", elapsed_secs);

            // Write timing file alongside outputs
            let timing_path = format!("{}-timing.txt", model_name);
            if let Ok(()) = std::fs::write(
                &timing_path,
                format!("elapsed_seconds={:.6}\n", elapsed_secs),
            ) {
                eprintln!("Timing written to {}", timing_path);
            }

            println!("\nFit completed!");
            println!("OFV: {:.4}", fit_result.ofv);
            println!("Elapsed: {:.3}s", elapsed_secs);
            for (name, val) in fit_result.theta_names.iter().zip(fit_result.theta.iter()) {
                println!("  {} = {:.6}", name, val);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Parsed `ferx check` arguments.
#[derive(Debug, PartialEq, Eq)]
struct CheckArgs<'a> {
    model: &'a str,
    data: Option<&'a str>,
    json: bool,
}

/// Why `parse_check_args` rejected the arguments.
#[derive(Debug, PartialEq, Eq)]
enum CheckArgsError {
    /// Model path missing or flag-looking — print full usage.
    Usage,
    /// `--data` present but missing its value or followed by another flag.
    MissingDataValue,
}

/// Parse `ferx check <model> [--data <csv>] [--json]`.
///
/// `--data` is parsed like the existing `--output` / `--threads` helpers: when
/// the flag is present it must be followed by a non-flag value, otherwise we
/// reject the args rather than silently running without data (or trying to open
/// a file literally named `--json`).
fn parse_check_args(args: &[String]) -> Result<CheckArgs<'_>, CheckArgsError> {
    let model = match args.get(2) {
        Some(p) if !p.starts_with("--") => p.as_str(),
        _ => return Err(CheckArgsError::Usage),
    };
    let data = match args.iter().position(|a| a == "--data") {
        None => None,
        Some(i) => match args.get(i + 1) {
            Some(v) if !v.starts_with("--") => Some(v.as_str()),
            _ => return Err(CheckArgsError::MissingDataValue),
        },
    };
    let json = args.iter().any(|a| a == "--json");
    Ok(CheckArgs { model, data, json })
}

/// Run the `check` subcommand. Returns the process exit code:
/// `0` = valid (no errors), `1` = errors found, `2` = usage / serialization error.
fn run_check(args: &[String]) -> i32 {
    let parsed = match parse_check_args(args) {
        Ok(p) => p,
        Err(CheckArgsError::MissingDataValue) => {
            eprintln!("Error: --data requires a path (e.g. --data data.csv)");
            return 2;
        }
        Err(CheckArgsError::Usage) => {
            eprintln!("Usage: ferx check <model.ferx> [--data <data.csv>] [--json]");
            eprintln!();
            eprintln!("Validates a model file without fitting and reports structured");
            eprintln!("diagnostics. With --data, also runs data-dependent checks");
            eprintln!("(covariates present, per-CMT coverage, steady-state, lag time).");
            eprintln!("--json   emit the report as JSON to stdout");
            return 2;
        }
    };

    let report = ferx_core::validate_model_file(parsed.model, parsed.data);

    if parsed.json {
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{}", s),
            Err(e) => {
                eprintln!("Error: failed to serialize check report: {}", e);
                return 2;
            }
        }
    } else {
        print_check_human(&report);
    }

    if report.valid {
        0
    } else {
        1
    }
}

/// Print a `CheckReport` in human-readable form to stdout, one diagnostic per
/// line as `severity[CODE] block:line: message`, with an indented `help:` line
/// for any suggestion, then a one-line summary.
fn print_check_human(report: &ferx_core::CheckReport) {
    for d in &report.diagnostics {
        let sev = match d.severity {
            ferx_core::Severity::Error => "error",
            ferx_core::Severity::Warning => "warning",
        };
        let loc = match (&d.block, d.line) {
            (Some(b), Some(l)) => format!(" {}:{}", b, l),
            (Some(b), None) => format!(" [{}]", b),
            _ => String::new(),
        };
        println!("{}[{}]{}: {}", sev, d.code, loc, d.message);
        if let Some(s) = &d.suggestion {
            println!("    help: {}", s);
        }
    }
    if report.valid {
        println!(
            "ok: {} — no errors ({} warning(s))",
            report.model,
            report.warning_count()
        );
    } else {
        println!(
            "invalid: {} — {} error(s), {} warning(s)",
            report.model,
            report.error_count(),
            report.warning_count()
        );
    }
}

/// Parse the optional `--output` flag. Returns `None` when absent; exits with
/// an error message when present but missing its value, mirroring `--threads`.
fn parse_output_flag(args: &[String]) -> Option<String> {
    let idx = args.iter().position(|a| a == "--output")?;
    match args.get(idx + 1) {
        Some(v) if !v.starts_with("--") => Some(v.clone()),
        _ => {
            eprintln!("Error: --output requires a path (e.g. --output run1.fitrx)");
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--threads` flag. Returns `None` when the flag is
/// absent, when its value is `0`, or when its value is `auto` — all of which
/// mean "leave rayon's default pool alone". Exits the process on a missing
/// value or any other non-parseable input so typos don't silently fall
/// through to the default.
fn parse_threads_flag(args: &[String]) -> Option<usize> {
    let idx = args.iter().position(|a| a == "--threads")?;
    let value = args.get(idx + 1).unwrap_or_else(|| {
        eprintln!("Error: --threads requires a value (positive integer, 0, or 'auto')");
        std::process::exit(1);
    });
    if value.eq_ignore_ascii_case("auto") || value == "0" {
        return None;
    }
    match value.parse::<usize>() {
        Ok(n) if n > 0 => Some(n),
        _ => {
            eprintln!(
                "Error: --threads expects a positive integer, 0, or 'auto'; got '{}'",
                value
            );
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--inits-from-nca[=METHOD]` flag. Returns `Ok(None)` when
/// the flag is absent (use the model file's value), `Ok(Some(method))` when it
/// is present (overriding the model file). A bare `--inits-from-nca` selects
/// `nca_sweep`; an explicit method is given as `--inits-from-nca=nca_ebe`.
/// Returns `Err` for an unrecognised method.
fn parse_inits_from_nca_flag(args: &[String]) -> Result<Option<NcaInit>, String> {
    let Some(arg) = args
        .iter()
        .find(|a| *a == "--inits-from-nca" || a.starts_with("--inits-from-nca="))
    else {
        return Ok(None);
    };
    let method = match arg.split_once('=') {
        None => NcaInit::Sweep, // bare flag → default strategy
        Some((_, value)) => match value.to_ascii_lowercase().as_str() {
            "nca" => NcaInit::Nca,
            "" | "sweep" | "nca_sweep" => NcaInit::Sweep,
            "ebe" | "nca_ebe" => NcaInit::Ebe,
            other => {
                return Err(format!(
                    "--inits-from-nca: unknown method '{other}' — expected nca, nca_sweep, or nca_ebe"
                ));
            }
        },
    };
    Ok(Some(method))
}

#[cfg(test)]
mod tests {
    use super::{
        parse_check_args, parse_inits_from_nca_flag, parse_output_flag, parse_threads_flag,
        print_check_human, run_check, CheckArgsError,
    };
    use ferx_core::NcaInit;

    fn args(extra: &[&str]) -> Vec<String> {
        std::iter::once("ferx")
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn absent_flag_is_none() {
        assert_eq!(parse_threads_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn positive_integer_parses() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "4"])), Some(4));
    }

    #[test]
    fn zero_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "0"])), None);
    }

    #[test]
    fn auto_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "auto"])), None);
        assert_eq!(parse_threads_flag(&args(&["--threads", "AUTO"])), None);
    }

    #[test]
    fn output_absent_is_none() {
        assert_eq!(parse_output_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn output_returns_path() {
        assert_eq!(
            parse_output_flag(&args(&["--output", "run1.fitrx"])),
            Some("run1.fitrx".to_string())
        );
    }

    #[test]
    fn inits_absent_is_none() {
        assert_eq!(parse_inits_from_nca_flag(&args(&["model.ferx"])), Ok(None));
    }

    #[test]
    fn inits_bare_flag_defaults_to_sweep() {
        assert_eq!(
            parse_inits_from_nca_flag(&args(&["--inits-from-nca"])),
            Ok(Some(NcaInit::Sweep))
        );
    }

    #[test]
    fn inits_explicit_methods_parse() {
        assert_eq!(
            parse_inits_from_nca_flag(&args(&["--inits-from-nca=nca"])),
            Ok(Some(NcaInit::Nca))
        );
        assert_eq!(
            parse_inits_from_nca_flag(&args(&["--inits-from-nca=nca_sweep"])),
            Ok(Some(NcaInit::Sweep))
        );
        assert_eq!(
            parse_inits_from_nca_flag(&args(&["--inits-from-nca=nca_ebe"])),
            Ok(Some(NcaInit::Ebe))
        );
    }

    #[test]
    fn inits_unknown_method_errors() {
        assert!(parse_inits_from_nca_flag(&args(&["--inits-from-nca=bogus"])).is_err());
    }

    #[test]
    fn check_args_model_only() {
        let argv = args(&["check", "model.ferx"]);
        let a = parse_check_args(&argv).unwrap();
        assert_eq!(a.model, "model.ferx");
        assert_eq!(a.data, None);
        assert!(!a.json);
    }

    #[test]
    fn check_args_with_data_and_json() {
        let argv = args(&["check", "model.ferx", "--data", "d.csv", "--json"]);
        let a = parse_check_args(&argv).unwrap();
        assert_eq!(a.model, "model.ferx");
        assert_eq!(a.data, Some("d.csv"));
        assert!(a.json);
    }

    #[test]
    fn check_args_missing_model_is_usage_error() {
        assert_eq!(
            parse_check_args(&args(&["check"])),
            Err(CheckArgsError::Usage)
        );
        assert_eq!(
            parse_check_args(&args(&["check", "--json"])),
            Err(CheckArgsError::Usage)
        );
    }

    #[test]
    fn check_args_data_without_value_is_error() {
        assert_eq!(
            parse_check_args(&args(&["check", "model.ferx", "--data"])),
            Err(CheckArgsError::MissingDataValue)
        );
    }

    #[test]
    fn check_args_data_followed_by_flag_is_error() {
        assert_eq!(
            parse_check_args(&args(&["check", "model.ferx", "--data", "--json"])),
            Err(CheckArgsError::MissingDataValue)
        );
    }

    // ── run_check: in-process coverage of the `check` subcommand ──────────────
    // `run_check` (and, through it, `print_check_human`) is otherwise exercised
    // only by the `cli_binaries.rs` end-to-end tests, which spawn `ferx` as a
    // child process — coverage the instrumented build does not capture. Driving
    // it in-process with absolute fixture paths (so CWD is irrelevant) registers
    // the coverage and pins the documented exit-code contract: 0 = valid,
    // 1 = errors found, 2 = usage / bad arguments.

    const VALID_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/one_cpt_iv.ferx");
    const VALID_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/one_cpt_iv.csv");

    #[test]
    fn run_check_usage_errors_return_2() {
        // No model path, and a flag where the model should be — both usage (2).
        assert_eq!(run_check(&args(&["check"])), 2);
        assert_eq!(run_check(&args(&["check", "--json"])), 2);
    }

    #[test]
    fn run_check_missing_data_value_returns_2() {
        assert_eq!(run_check(&args(&["check", VALID_MODEL, "--data"])), 2);
    }

    #[test]
    fn run_check_valid_model_human_and_json_return_0() {
        // A clean example model has no errors → valid → 0. Covers both the human
        // (`print_check_human`) and `--json` (serde) rendering branches.
        assert_eq!(run_check(&args(&["check", VALID_MODEL])), 0);
        assert_eq!(run_check(&args(&["check", VALID_MODEL, "--json"])), 0);
    }

    #[test]
    fn run_check_with_data_runs_data_path_without_usage_error() {
        // 0 (valid) or 1 (data-dependent findings) — never 2. Exercises the
        // data-dependent branch of `validate_model_file`.
        assert_ne!(
            run_check(&args(&["check", VALID_MODEL, "--data", VALID_DATA])),
            2
        );
    }

    #[test]
    fn run_check_invalid_model_returns_1() {
        // An unparseable model → errors → invalid → 1. Covers the invalid-summary
        // and error-diagnostic branches of `print_check_human`, plus `--json` on
        // an invalid report.
        let dir = tempfile::tempdir().expect("tempdir");
        let bad = dir.path().join("bad.ferx");
        std::fs::write(&bad, "this is not a valid ferx model\n").expect("write bad model");
        let bad_path = bad.to_str().unwrap();
        assert_eq!(run_check(&args(&["check", bad_path])), 1);
        assert_eq!(run_check(&args(&["check", bad_path, "--json"])), 1);
    }

    #[test]
    fn print_check_human_covers_all_diagnostic_shapes() {
        // Drive `print_check_human` directly over diagnostics that hit every arm:
        // both severities, all three `loc` shapes (block+line / block-only /
        // none), and suggestion present/absent — branches the model-file fixtures
        // above don't deterministically reach.
        use ferx_core::{CheckReport, Diagnostic};
        let invalid = CheckReport::new(
            "m.ferx",
            Some("d.csv".to_string()),
            vec![
                Diagnostic::warning("W_X", "a warning")
                    .with_block("error_model")
                    .with_line(7)
                    .with_suggestion("try this instead"),
                Diagnostic::error("E_Y", "block-scoped error").with_block("odes"),
                Diagnostic::error("E_Z", "locationless error"),
            ],
        );
        // Must not panic; output is captured by the test harness.
        print_check_human(&invalid);
        assert!(!invalid.valid);
        assert_eq!(invalid.error_count(), 2);
        assert_eq!(invalid.warning_count(), 1);

        // A clean report exercises the valid-summary branch through the same printer.
        let ok = CheckReport::new("m.ferx", None, vec![]);
        print_check_human(&ok);
        assert!(ok.valid);
    }
}
