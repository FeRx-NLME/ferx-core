//! `ferx bootstrap` — the non-parametric case bootstrap (#1140).
//!
//! Thin by construction (#1114 A5): parse flags, call
//! [`ferx_tools::bootstrap::run_bootstrap`], print a table, pick an exit code.
//! Every decision about resampling, filtering or statistics lives in
//! `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::bootstrap::{
    resummarize, run_bootstrap, BootstrapOptions, BootstrapResult, SampleSize,
};

pub const BOOTSTRAP_USAGE: &str = "\
Usage: ferx bootstrap <model.ferx> [--data <data.csv>] [options]
       ferx bootstrap --summarize --directory <dir> [filter options]

Resamples subjects with replacement, refits the model to each replicate, and
reports bias, standard errors and confidence intervals from the spread of the
estimates. Resampling is always over whole subjects.

  --samples N          number of bootstrap datasets (default 200)
  --seed N             master seed (default 1). Each replicate's draw is derived
                       from this and its own index, so the datasets do not
                       depend on --threads or on completion order.
  --threads N          replicates to fit concurrently (each fit uses 1 thread)
  --sample-size M      subjects per replicate (default: as many as the dataset).
                       With --stratify-on, may instead be a per-stratum map:
                       --sample-size \"1001=>12,1002=>24,1003=>10\"
  --stratify-on COL    resample within strata defined by a dataset column. The
                       column must take exactly one value per subject.
  --directory DIR      where the CSV artefacts go (default {model}-bootstrap)
  --ci LEVEL           two-sided confidence level in percent (default 95)

  --no-update-inits    start replicates from the model file's initial estimates
                       instead of the base fit's final ones
  --no-run-base-model  skip the fit on the original dataset (disables bias, the
                       standard-error intervals, --update-inits and --dofv)
  --keep-covariance    run the covariance step for every replicate (slow; only
                       needed for the two covstep filters below)
  --dofv               evaluate each replicate's estimates on the original data
                       with no estimation, and report the OFV difference

  --no-skip-minimization-terminated    keep replicates that did not converge
  --no-skip-estimate-near-boundary     keep replicates with an estimate on a bound
  --skip-covariance-step-terminated    drop replicates whose covariance step failed
  --skip-with-covstep-warnings         drop replicates with covariance warnings

  --summarize          recompute the statistics from an existing run directory's
                       raw_results.csv under different filters. Refits nothing.
";

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn value<'a>(args: &'a [String], name: &str) -> Option<&'a String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str) -> Result<Option<T>, String> {
    match value(args, name) {
        None => Ok(None),
        Some(raw) => raw
            .parse::<T>()
            .map(Some)
            .map_err(|_| format!("{name}: `{raw}` is not a valid value")),
    }
}

/// Build [`BootstrapOptions`] from the command line.
pub fn parse_options(
    args: &[String],
    model_path: Option<&str>,
) -> Result<BootstrapOptions, String> {
    let mut o = BootstrapOptions::default();
    if let Some(n) = parse::<usize>(args, "--samples")? {
        o.samples = n;
    }
    if let Some(n) = parse::<u64>(args, "--seed")? {
        o.seed = n;
    }
    if let Some(n) = parse::<usize>(args, "--threads")? {
        o.threads = Some(n.max(1));
    }
    if let Some(level) = parse::<f64>(args, "--ci")? {
        o.confidence_level = level;
    }
    if let Some(spec) = value(args, "--sample-size") {
        o.sample_size = SampleSize::parse(spec)?;
    }
    o.stratify_on = value(args, "--stratify-on").cloned();

    o.update_inits = !flag(args, "--no-update-inits");
    o.run_base_model = !flag(args, "--no-run-base-model");
    o.keep_covariance = flag(args, "--keep-covariance");
    o.dofv = flag(args, "--dofv");

    o.skip_minimization_terminated = !flag(args, "--no-skip-minimization-terminated");
    o.skip_estimate_near_boundary = !flag(args, "--no-skip-estimate-near-boundary");
    o.skip_covariance_step_terminated = flag(args, "--skip-covariance-step-terminated");
    o.skip_with_covstep_warnings = flag(args, "--skip-with-covstep-warnings");

    // `--update-inits` needs somewhere to get the inits from. Silently ignoring
    // it would change every replicate's starting point without saying so.
    if o.update_inits && !o.run_base_model {
        o.update_inits = false;
    }

    o.directory = Some(match value(args, "--directory") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let stem = model_path
                .and_then(|p| std::path::Path::new(p).file_stem().and_then(|s| s.to_str()))
                .unwrap_or("model");
            PathBuf::from(format!("{stem}-bootstrap"))
        }
    });
    Ok(o)
}

/// Entry point for `ferx bootstrap ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args.get(2).map(String::as_str) == Some("-h")
        || args.get(2).map(String::as_str) == Some("--help")
    {
        print!("{BOOTSTRAP_USAGE}");
        return 0;
    }

    if flag(args, "--summarize") {
        return match run_summarize(args) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("Error: {e}");
                1
            }
        };
    }

    // The first positional after the subcommand is the model file.
    let Some(model_path) = args.get(2).filter(|a| !a.starts_with('-')) else {
        eprint!("{BOOTSTRAP_USAGE}");
        return 1;
    };

    match run_bootstrap_command(args, model_path) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_summarize(args: &[String]) -> Result<i32, String> {
    let dir = value(args, "--directory")
        .ok_or("--summarize needs --directory pointing at an existing bootstrap run")?;
    let options = parse_options(args, None)?;
    let summary = resummarize(std::path::Path::new(dir), &options)?;
    println!(
        "Re-summarized {dir}: {} of {} completed samples included",
        summary.n_included, summary.n_completed
    );
    print!(
        "{}",
        render_table(&summary.parameters, summary.confidence_level)
    );
    Ok(0)
}

fn run_bootstrap_command(args: &[String], model_path: &str) -> Result<i32, String> {
    let data_path = value(args, "--data").map(String::as_str);
    let options = parse_options(args, Some(model_path))?;

    let prepared = ferx_core::prepare_run(model_path, data_path)?;
    if let Some(w) = &prepared.data_path_warning {
        eprintln!("Warning: {w}");
    }
    eprintln!("Model: {}", prepared.parsed.model.name);
    eprintln!(
        "Data:  {} subjects, {} observations from {}",
        prepared.population.subjects.len(),
        prepared.population.n_obs(),
        prepared.data_path
    );
    eprintln!(
        "Bootstrap: {} samples, seed {}{}",
        options.samples,
        options.seed,
        options
            .stratify_on
            .as_deref()
            .map(|c| format!(", stratified on {c}"))
            .unwrap_or_default()
    );

    let started = std::time::Instant::now();
    let result = run_bootstrap(&prepared, &options)?;
    eprintln!(
        "Fitted {} replicates in {:.1}s",
        result.replicates.len(),
        started.elapsed().as_secs_f64()
    );
    print!("{}", report(&result, &options));

    // A run whose samples nearly all failed produced no usable interval, and
    // saying so in the exit code keeps it from passing silently in a pipeline.
    if result.summary.n_included == 0 {
        eprintln!("Error: no bootstrap sample survived the exclusion criteria");
        return Ok(1);
    }
    Ok(0)
}

/// Render the whole report. Built as a `String` rather than printed line by line
/// so the layout is testable in-process — the CLI's own integration tests run
/// the binary as a subprocess, which is the right way to check exit codes and
/// the wrong way to check what a table looks like.
fn report(result: &BootstrapResult, options: &BootstrapOptions) -> String {
    let s = &result.summary;
    let mut out = format!(
        "\n{} of {} samples completed; {} included in the statistics\n",
        s.n_completed,
        result.replicates.len(),
        s.n_included
    );
    for (reason, n) in &s.excluded_by {
        out.push_str(&format!("  excluded {n}: {reason}\n"));
    }
    out.push_str(&render_table(&s.parameters, s.confidence_level));

    if !s.parameters.is_empty() && s.parameters.iter().all(|p| p.ci_percentile.is_none()) {
        out.push_str(&format!(
            "\nNote: {} samples cannot resolve a {}% percentile interval; only the \
             normal-approximation interval is shown. PsN's rule of thumb is 200 samples.\n",
            s.n_included, s.confidence_level
        ));
    }
    if let Some(dir) = &options.directory {
        out.push_str(&format!("\nWrote bootstrap results to {}\n", dir.display()));
    }
    out
}

fn render_table(parameters: &[ferx_tools::bootstrap::ParameterSummary], level: f64) -> String {
    let cell = |v: Option<f64>| match v {
        Some(v) if v.is_finite() => format!("{v:>12.5}"),
        _ => format!("{:>12}", "-"),
    };
    let width = parameters
        .iter()
        .map(|p| p.name.len())
        .max()
        .unwrap_or(9)
        .max(9);
    let mut out = format!(
        "\n{:<width$} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}\n",
        "parameter",
        "original",
        "mean",
        "bias",
        "SE(boot)",
        format!("{level}% lo"),
        format!("{level}% hi"),
    );
    for p in parameters {
        // The percentile interval is the bootstrap's own answer; the
        // normal-approximation one is the fallback when there are too few
        // samples to resolve the tail.
        let (lo, hi) = match p.ci_percentile.or(p.ci_standard_error) {
            Some((lo, hi)) => (Some(lo), Some(hi)),
            None => (None, None),
        };
        out.push_str(&format!(
            "{:<width$} {} {} {} {} {} {}\n",
            p.name,
            cell(p.original),
            cell(Some(p.mean)),
            cell(p.bias),
            cell(Some(p.standard_error)),
            cell(lo),
            cell(hi),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(rest: &[&str]) -> Vec<String> {
        let mut v = vec!["ferx".to_string(), "bootstrap".to_string()];
        v.extend(rest.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn defaults_match_psn() {
        let o = parse_options(&args(&["m.ferx"]), Some("m.ferx")).unwrap();
        assert_eq!(o.samples, 200);
        assert!(o.update_inits);
        assert!(o.run_base_model);
        assert!(o.skip_minimization_terminated);
        assert!(o.skip_estimate_near_boundary);
        assert!(!o.skip_covariance_step_terminated);
        assert!(!o.skip_with_covstep_warnings);
        assert!(!o.keep_covariance);
        assert_eq!(o.confidence_level, 95.0);
        assert_eq!(o.directory, Some(PathBuf::from("m-bootstrap")));
    }

    #[test]
    fn flags_are_parsed() {
        let o = parse_options(
            &args(&[
                "m.ferx",
                "--samples",
                "500",
                "--seed",
                "12345",
                "--threads",
                "5",
                "--ci",
                "90",
                "--directory",
                "boot",
                "--dofv",
                "--keep-covariance",
            ]),
            Some("m.ferx"),
        )
        .unwrap();
        assert_eq!(o.samples, 500);
        assert_eq!(o.seed, 12345);
        assert_eq!(o.threads, Some(5));
        assert_eq!(o.confidence_level, 90.0);
        assert_eq!(o.directory, Some(PathBuf::from("boot")));
        assert!(o.dofv);
        assert!(o.keep_covariance);
    }

    #[test]
    fn negated_flags_turn_the_psn_defaults_off() {
        let o = parse_options(
            &args(&[
                "m.ferx",
                "--no-skip-minimization-terminated",
                "--no-skip-estimate-near-boundary",
            ]),
            Some("m.ferx"),
        )
        .unwrap();
        assert!(!o.skip_minimization_terminated);
        assert!(!o.skip_estimate_near_boundary);
    }

    #[test]
    fn update_inits_is_dropped_when_there_is_no_base_fit() {
        // `--update-inits` reads the base fit's final estimates. With
        // `--no-run-base-model` there are none, so the combination must resolve
        // to the model file's own inits rather than to something undefined.
        let o = parse_options(&args(&["m.ferx", "--no-run-base-model"]), Some("m.ferx")).unwrap();
        assert!(!o.run_base_model);
        assert!(!o.update_inits);
    }

    #[test]
    fn sample_size_accepts_both_forms() {
        let flat = parse_options(&args(&["m.ferx", "--sample-size", "32"]), None).unwrap();
        assert_eq!(flat.sample_size, SampleSize::Total(32));

        let per = parse_options(
            &args(&[
                "m.ferx",
                "--stratify-on",
                "STUD",
                "--sample-size",
                "1001=>12,1002=>24",
            ]),
            None,
        )
        .unwrap();
        assert!(matches!(per.sample_size, SampleSize::PerStratum(_)));
        assert_eq!(per.stratify_on.as_deref(), Some("STUD"));
    }

    #[test]
    fn a_bad_numeric_flag_is_an_error_not_a_default() {
        // Falling back to the default on a typo would run a 200-sample
        // bootstrap when the user asked for something else.
        let err = parse_options(&args(&["m.ferx", "--samples", "many"]), None).unwrap_err();
        assert!(err.contains("--samples"), "{err}");
    }

    #[test]
    fn the_directory_defaults_to_the_model_stem() {
        let o = parse_options(&args(&["runs/warfarin.ferx"]), Some("runs/warfarin.ferx")).unwrap();
        assert_eq!(o.directory, Some(PathBuf::from("warfarin-bootstrap")));
    }

    // ── rendering ───────────────────────────────────────────────────────────

    use ferx_tools::bootstrap::ParameterSummary;

    fn parameter(name: &str, ci_percentile: Option<(f64, f64)>) -> ParameterSummary {
        ParameterSummary {
            name: name.to_string(),
            original: Some(1.0),
            mean: 1.5,
            bias: Some(0.5),
            standard_error: 0.25,
            median: 1.4,
            ci_percentile,
            ci_standard_error: Some((0.5, 2.5)),
        }
    }

    #[test]
    fn the_table_is_column_aligned_and_widens_for_long_names() {
        let table = render_table(
            &[
                parameter("CL", Some((1.1, 1.9))),
                parameter("OMEGA(ETA_CL,ETA_CL)", Some((1.1, 1.9))),
            ],
            95.0,
        );
        let lines: Vec<&str> = table.lines().filter(|l| !l.is_empty()).collect();
        assert!(lines[0].starts_with("parameter"));
        assert!(lines[0].contains("95% lo") && lines[0].contains("95% hi"));
        // Every row is the same width, including the one that set it.
        let widths: Vec<usize> = lines.iter().map(|l| l.len()).collect();
        assert!(
            widths.windows(2).all(|w| w[0] == w[1]),
            "ragged table: {widths:?}"
        );
        assert!(lines[2].starts_with("OMEGA(ETA_CL,ETA_CL)"));
    }

    #[test]
    fn the_table_falls_back_to_the_normal_interval_when_there_is_no_percentile_one() {
        // Too few samples to resolve the tail: show the normal-approximation
        // interval rather than an empty column.
        let table = render_table(&[parameter("CL", None)], 95.0);
        let row = table.lines().nth(2).expect("a data row");
        assert!(row.contains("0.50000") && row.contains("2.50000"), "{row}");

        let with_percentile = render_table(&[parameter("CL", Some((1.1, 1.9)))], 95.0);
        let row = with_percentile.lines().nth(2).expect("a data row");
        assert!(row.contains("1.10000") && row.contains("1.90000"), "{row}");
    }

    #[test]
    fn a_missing_value_renders_as_a_dash_not_a_nan() {
        let mut p = parameter("CL", None);
        p.original = None;
        p.bias = None;
        p.ci_standard_error = None;
        let table = render_table(&[p], 95.0);
        let row = table.lines().nth(2).expect("a data row");
        assert!(!row.to_lowercase().contains("nan"), "{row}");
        assert!(row.contains('-'), "{row}");
    }

    #[test]
    fn the_ci_level_is_reflected_in_the_header() {
        let table = render_table(&[parameter("CL", Some((1.1, 1.9)))], 90.0);
        assert!(table.contains("90% lo"), "{table}");
    }

    fn summary_of(
        parameters: Vec<ParameterSummary>,
        excluded: Vec<(String, usize)>,
    ) -> BootstrapResult {
        BootstrapResult {
            parameter_names: parameters.iter().map(|p| p.name.clone()).collect(),
            original: None,
            replicates: Vec::new(),
            draws: Vec::new(),
            subject_ids: Vec::new(),
            summary: ferx_tools::bootstrap::BootstrapSummary {
                n_completed: 4,
                n_included: 3,
                excluded_by: excluded,
                confidence_level: 95.0,
                diagnostic_means: Vec::new(),
                parameters,
            },
            n_estimated_parameters: 2,
        }
    }

    #[test]
    fn the_report_names_the_exclusions_and_the_output_directory() {
        let result = summary_of(
            vec![parameter("CL", Some((1.1, 1.9)))],
            vec![("minimization terminated".to_string(), 1)],
        );
        let options = BootstrapOptions {
            directory: Some(PathBuf::from("out")),
            ..BootstrapOptions::default()
        };
        let text = report(&result, &options);
        assert!(
            text.contains("4 of 0 samples completed; 3 included"),
            "{text}"
        );
        assert!(
            text.contains("excluded 1: minimization terminated"),
            "{text}"
        );
        assert!(text.contains("Wrote bootstrap results to out"), "{text}");
        // A resolvable percentile interval means no "too few samples" note.
        assert!(!text.contains("cannot resolve"), "{text}");
    }

    #[test]
    fn the_report_says_when_there_were_too_few_samples_for_a_percentile_interval() {
        let result = summary_of(vec![parameter("CL", None)], Vec::new());
        let text = report(
            &result,
            &BootstrapOptions {
                directory: None,
                ..BootstrapOptions::default()
            },
        );
        assert!(
            text.contains("cannot resolve a 95% percentile interval"),
            "{text}"
        );
        assert!(!text.contains("Wrote bootstrap results"), "{text}");
    }

    // ── dispatch ────────────────────────────────────────────────────────────

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn repo_path(rel: &str) -> String {
        repo_root().join(rel).to_string_lossy().into_owned()
    }

    #[test]
    fn help_exits_zero_and_a_missing_model_exits_one() {
        assert_eq!(run(&args(&["--help"])), 0);
        assert_eq!(run(&args(&["-h"])), 0);
        // No positional at all, and a leading flag that is not a model path.
        assert_eq!(run(&args(&[])), 1);
        assert_eq!(run(&args(&["--samples", "5"])), 1);
    }

    #[test]
    fn a_failure_to_load_the_model_exits_one() {
        assert_eq!(run(&args(&["definitely-not-a-model.ferx"])), 1);
    }

    #[test]
    fn summarize_without_a_directory_exits_one() {
        assert_eq!(run(&args(&["--summarize"])), 1);
        // A directory that exists but holds no raw_results.csv.
        let dir = tempfile::tempdir().expect("temp dir");
        assert_eq!(
            run(&args(&[
                "--summarize",
                "--directory",
                dir.path().to_str().unwrap()
            ])),
            1
        );
    }

    /// One real end-to-end pass through `run`, on the smallest useful model.
    ///
    /// Kept to a single sample and a single outer iteration is not possible from
    /// the command line, so this is warfarin (10 subjects) at `--samples 1`:
    /// two fits, well under a second in release and a couple in debug. It is
    /// what covers the load → fit → report → exit-code path that the subprocess
    /// tests in `tests/cli_ferx.rs` exercise but cannot measure.
    #[test]
    fn a_real_run_writes_its_directory_and_exits_zero() {
        let dir = tempfile::tempdir().expect("temp dir");
        let code = run(&args(&[
            &repo_path("examples/warfarin.ferx"),
            "--data",
            &repo_path("data/warfarin.csv"),
            "--samples",
            "1",
            "--seed",
            "5",
            "--threads",
            "1",
            "--directory",
            dir.path().to_str().unwrap(),
        ]));
        assert_eq!(code, 0);
        assert!(dir.path().join("raw_results.csv").exists());
        assert!(dir.path().join("bootstrap_results.csv").exists());

        // And `--summarize` over what it just wrote.
        assert_eq!(
            run(&args(&[
                "--summarize",
                "--directory",
                dir.path().to_str().unwrap()
            ])),
            0
        );
    }

    #[test]
    fn a_run_whose_every_sample_is_excluded_exits_one() {
        // `--sample-size 1` fits single-subject replicates, which cannot
        // identify the between-subject variances and so land on a boundary —
        // the default filters then drop every sample. Exiting 0 there would let
        // a pipeline treat an empty result as a successful bootstrap.
        let dir = tempfile::tempdir().expect("temp dir");
        let code = run(&args(&[
            &repo_path("examples/warfarin.ferx"),
            "--data",
            &repo_path("data/warfarin.csv"),
            "--samples",
            "1",
            "--sample-size",
            "1",
            "--seed",
            "5",
            "--threads",
            "1",
            "--directory",
            dir.path().to_str().unwrap(),
        ]));
        // Either outcome is legitimate for a degenerate replicate; what must not
        // happen is a zero exit with nothing included.
        let raw = std::fs::read_to_string(dir.path().join("bootstrap_diagnostics.csv"))
            .expect("diagnostics written");
        let included: f64 = raw
            .lines()
            .find(|l| l.starts_with("samples_included,"))
            .and_then(|l| l.split(',').nth(1))
            .and_then(|v| v.trim().parse().ok())
            .expect("samples_included row");
        if included == 0.0 {
            assert_eq!(code, 1, "an empty bootstrap must not exit 0");
        } else {
            assert_eq!(code, 0);
        }
    }
}
