mod allometry_cmd;
mod amd_cmd;
mod bootstrap_cmd;
mod bootstrap_progress;
mod covsearch_cmd;
mod gam_cmd;
mod globalsearch_cmd;
mod iivsearch_cmd;
mod iovsearch_cmd;
mod modelsearch_cmd;
mod ruvsearch_cmd;

use ferx_core::NcaInit;
use std::env;
use std::time::Instant;

/// Top-level usage/help text, shared by the no-args error path (stderr, exit 1)
/// and `ferx -h`/`--help` (stdout, exit 0) so the two can't drift apart.
const MAIN_USAGE: &str = "\
Usage: ferx <model.ferx> --data <data.csv> [--threads N|auto] [--output <run.fitrx>] [--include-data] [--inits-from-nca[=nca|nca_sweep|nca_ebe]] [--output-format yaml|json|both] [--clean]
       ferx <model.ferx> --simulate          [--threads N|auto] [--output <run.fitrx>]
       ferx check <model.ferx> [--data <data.csv>] [--json]
       ferx summary <run.fitrx> [<run2.fitrx> ...]
       ferx bootstrap <model.ferx> [--data <data.csv>] [--samples N] [--seed N]
                      [--stratify-on COL] [--threads N]   (see `ferx bootstrap --help`)
       ferx gam      <model.ferx>  --data <data.csv> [--csv gam.csv] [--threads N]
                                                         (see `ferx gam --help`)
       ferx covsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx covsearch --help`)
       ferx allometry <model.ferx> --data <data.csv> [--covariate WT] [--reference 70]
                                                         (see `ferx allometry --help`)
       ferx modelsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx modelsearch --help`)
       ferx ruvsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx ruvsearch --help`)
       ferx iivsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx iivsearch --help`)
       ferx iovsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx iovsearch --help`)
       ferx amd       <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx amd --help`)
       ferx globalsearch <search.ferxsearch> [--directory DIR] [--threads N] [--resume]
                                                         (see `ferx globalsearch --help`)

Fits a NLME model and writes sdtab.csv with residuals.
Data must be in NONMEM format (ID, TIME, DV, EVID, AMT, CMT, ...)

--data is optional if the model file has a [data] block (path = ...);
       an explicit --data overrides it, with a warning if they differ.

--threads N    use N rayon workers (N > 0)
--threads 0    use the default worker count (available cores - 1, floored at 1, capped at 8)
--threads auto alias for --threads 0
               An explicit --threads (including 0/auto) overrides the model
               file's [fit_options] threads, with a warning if they differ.

--output PATH  also write a portable .fitrx fit bundle (zip of JSON+CSV)
--include-data embed the input --data CSV inside the .fitrx (off by default)

--output-format yaml|json|both  which estimates file to write (default yaml).
               json writes {model}-fit.json: the complete FitResult under a
               versioned schema, for programmatic/agent consumers.

--inits-from-nca[=METHOD]  derive NCA-based starting values before fitting,
               overriding the model file. METHOD is nca, nca_sweep (default),
               or nca_ebe; a bare --inits-from-nca means nca_sweep.

--gam          run GAM covariate pre-screening after fitting and write
               {model}-gam.csv  (same as `ferx gam` but in one step)

--clean        ignore any resume checkpoint ({model}.tmp) and start fresh.
               By default a run periodically checkpoints and, if interrupted,
               resumes automatically on the next run of the same model + data.
               Disable checkpointing entirely with [fit_options] checkpoint = false.

-h, --help     print this help and exit
";

/// True for either spelling of the help flag.
fn is_help_flag(arg: Option<&String>) -> bool {
    matches!(arg.map(String::as_str), Some("-h") | Some("--help"))
}

/// Rewrites `--flag=value` into separate `--flag`, `value` elements so the
/// rest of the parsing below (which matches flags by exact string, then reads
/// the following element as the value) sees `--data=d.csv` the same way it
/// already sees `--data d.csv` (#693 — `=`-style args were silently dropped).
/// `--inits-from-nca=METHOD` is left untouched: `parse_inits_from_nca_flag`
/// parses that combined form itself, deliberately never consuming a following
/// positional for the bare-flag case.
fn normalize_args(args: &[String]) -> Vec<String> {
    args.iter()
        .flat_map(|a| {
            if a.starts_with("--inits-from-nca") {
                return vec![a.clone()];
            }
            match a.strip_prefix("--").and_then(|rest| rest.find('=')) {
                Some(eq) => {
                    let (flag, value) = a.split_at(eq + 2);
                    vec![flag.to_string(), value[1..].to_string()]
                }
                None => vec![a.clone()],
            }
        })
        .collect()
}

/// Every `ferx <word> ...` subcommand, paired with the function that runs it
/// and returns the process exit code.
///
/// This table is the *only* list of tool names: `main` dispatches from it and
/// the unknown-tool error below enumerates it, so a tool cannot be added to
/// one and missed by the other (#1396).
const SUBCOMMANDS: &[(&str, fn(&[String]) -> i32)] = &[
    // Parse + validate a model (optionally against data) and report structured
    // diagnostics; no fit.
    ("check", run_check),
    // Read-only reporting over a saved fit bundle, like psn::sumo.
    ("summary", run_summary),
    // #1140, the first `ferx-tools` subcommand: many fits over resampled data.
    ("bootstrap", bootstrap_cmd::run),
    ("gam", gam_cmd::run),
    // `covsearch` / `allometry` (#1180): the first model-space search tools,
    // driven by a `.ferxsearch` file (or, for allometry, a model file + flags).
    ("covsearch", covsearch_cmd::run),
    ("allometry", allometry_cmd::run),
    // #1181: structural PK search over the `pk` templates.
    ("modelsearch", modelsearch_cmd::run),
    // #1182: residual-error model search.
    ("ruvsearch", ruvsearch_cmd::run),
    // #1183: variability-structure search.
    ("iivsearch", iivsearch_cmd::run),
    ("iovsearch", iovsearch_cmd::run),
    // #1184: the whole pipeline, one tool after another.
    ("amd", amd_cmd::run),
    // #1185: GA / exhaustive over one grid, penalized fitness.
    ("globalsearch", globalsearch_cmd::run),
];

/// The runner for `word`, when it names a subcommand.
fn subcommand(word: &str) -> Option<fn(&[String]) -> i32> {
    SUBCOMMANDS
        .iter()
        .find(|(name, _)| *name == word)
        .map(|(_, run)| *run)
}

/// What the first argument means once it is known not to name a [`SUBCOMMANDS`]
/// entry (#1396).
#[derive(Debug, PartialEq, Eq)]
enum FirstArg<'a> {
    /// A bare word — no dot, no path separator. The user meant a tool, and this
    /// build does not have it (a typo, or a tool that only exists in a later
    /// version). Reading it as a model path is what produced the misleading
    /// "Failed to read model file: No such file or directory".
    UnknownTool(&'a str),
    /// Either the path is there, or the filesystem would not say: hand it to
    /// the fit/simulate path, which opens it and reports whatever the OS says.
    ModelFile(&'a str),
    /// Looks like a path, and the filesystem says it is not there.
    MissingModelFile(&'a str),
    /// A flag in the model-file position.
    Flag(&'a str),
}

/// Classify the first argument. `exists` is the filesystem answer for `arg`
/// (`Path::try_exists`), passed in so the rule itself is testable without
/// touching the disk.
///
/// A file that is present is a model file whatever the name — an extension is
/// only the fallback signal for something that is *not* there, and a tool name
/// is a bare word by construction, so anything carrying a `.` or a path
/// separator is a path the user got wrong rather than a tool we do not have.
/// "Whatever the name" is bounded by the two decisions `main` takes first: a
/// [`SUBCOMMANDS`] name runs its tool, and a leading `-` is a flag, so a file
/// called `check` is reachable only by a path (`./check`).
///
/// A probe that *fails* (`Err`) is not an absent file: an unreadable parent
/// directory or a symlink loop answers neither question, and only
/// `Path::try_exists` keeps the two apart — `Path::exists` folds every
/// metadata error into `false`, which would report `EACCES` or `ELOOP` as
/// "was not found". Such a path goes to the fit path, whose reader surfaces
/// the real OS error.
fn classify_first_arg(arg: &str, exists: std::io::Result<bool>) -> FirstArg<'_> {
    if arg.starts_with('-') {
        return FirstArg::Flag(arg);
    }
    match exists {
        Ok(true) | Err(_) => return FirstArg::ModelFile(arg),
        Ok(false) => {}
    }
    if arg.contains('.') || arg.contains('/') || arg.contains('\\') {
        FirstArg::MissingModelFile(arg)
    } else {
        FirstArg::UnknownTool(arg)
    }
}

/// The second line of the unknown-tool error: a `did you mean` when one tool
/// name is within two edits of `word`, and the full list either way.
fn unknown_tool_hint(word: &str) -> String {
    let names: Vec<&str> = SUBCOMMANDS.iter().map(|(name, _)| *name).collect();
    let mut hint = String::new();
    if let Some(near) = names
        .iter()
        .map(|name| (edit_distance(word, name), *name))
        .filter(|(d, _)| *d <= 2)
        .min()
        .map(|(_, name)| name)
    {
        hint.push_str(&format!("Did you mean `{near}`?\n"));
    }
    hint.push_str(&format!("Available tools: {}.\n", names.join(", ")));
    hint.push_str("To fit a model, give its file path: ferx run1.ferx --data data.csv");
    hint
}

/// Levenshtein distance, for the `did you mean` above only.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    // One row of the DP table: prev[j] is the distance between the prefix of
    // `a` seen so far and the first j characters of `b`.
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != *cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

fn main() {
    let raw_args: Vec<String> = env::args().collect();
    let args = normalize_args(&raw_args);

    if is_help_flag(args.get(1)) {
        print!("{MAIN_USAGE}");
        std::process::exit(0);
    }

    // A subcommand (`ferx covsearch ...`) is dispatched before the fit/simulate
    // path, so the rest of main() is unchanged and a plain `ferx model.ferx`
    // still means what it always did.
    if let Some(run) = args.get(1).and_then(|w| subcommand(w)) {
        std::process::exit(run(&args));
    }

    if args.len() < 2 {
        eprint!("{MAIN_USAGE}");
        std::process::exit(1);
    }

    // Not a subcommand and not a flag: either a model file, or a tool name this
    // build does not have (#1396). Telling those apart here is what keeps a
    // mistyped or not-yet-shipped tool from being read as a model path and
    // reported as a missing file.
    match classify_first_arg(&args[1], std::path::Path::new(&args[1]).try_exists()) {
        FirstArg::UnknownTool(word) => {
            eprintln!("Error: tool `{word}` not recognized.");
            eprintln!("{}", unknown_tool_hint(word));
            std::process::exit(1);
        }
        FirstArg::MissingModelFile(path) => {
            eprintln!(
                "Error: the model `{path}` was not found at this location. \
                 Please check folder and file names."
            );
            std::process::exit(1);
        }
        FirstArg::Flag(flag) => {
            eprintln!("Error: expected a model file or a tool name, got the flag `{flag}`.");
            eprint!("{MAIN_USAGE}");
            std::process::exit(1);
        }
        FirstArg::ModelFile(_) => {}
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
    let estimates_format = parse_output_format(&args);
    let include_data = args.iter().any(|a| a == "--include-data");
    if include_data && output_path.is_none() {
        eprintln!("Warning: --include-data has no effect without --output");
    }

    // `--clean` forces a fresh start (#755): delete any resume checkpoint left
    // by an interrupted run of this model before fitting. Without it, a run
    // resumes automatically from `{model}.tmp` when present and compatible.
    if args.iter().any(|a| a == "--clean") {
        if let Some(stem) = std::path::Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
        {
            let tmp = format!("{stem}.tmp");
            if std::path::Path::new(&tmp).exists() {
                match std::fs::remove_file(&tmp) {
                    Ok(()) => eprintln!("Removed checkpoint {tmp} (--clean)"),
                    Err(e) => eprintln!("Warning: could not remove checkpoint {tmp}: {e}"),
                }
            }
        }
    }
    // Honor --threads by sizing rayon's global pool (build_global() is once-per-process,
    // correct for a CLI binary) so fit()'s default pool — sized to current_num_threads()
    // — inherits the count. The 32 MiB worker stack that wide ODE+IOV analytic gradients
    // need is applied by fit()'s own fit-scoped pool (api::default_fit_pool), so the
    // global pool keeps the platform-default stack here rather than reserving a second
    // 32 MiB × N. Without --threads, fit() applies its own default (available cores - 1,
    // floored at 1, capped at 8 — #707). `Some(0)` (`--threads 0` / `auto`) names the
    // default rather than a width, so it sizes nothing here — but it is still carried in
    // `RunOverrides` below, where it overrides a model file's `[fit_options] threads`.
    if let Some(n) = threads.filter(|&n| n > 0) {
        if let Err(e) = ferx_core::configure_global_thread_pool(n) {
            eprintln!("Warning: {e}");
        }
    }
    // Everything the command line named explicitly, which beats the model file's
    // `[fit_options]` (#1416).
    let overrides = ferx_core::RunOverrides {
        inits_from_nca,
        threads,
    };

    let t_start = Instant::now();
    // Precedence: an explicit `--data` always wins (unchanged); `--simulate`
    // is checked next (unchanged); only when neither flag is given do we fall
    // through to `run_model_with_overrides(.., None, ..)`, which resolves the
    // model's own optional `[data] path = ...` (#690) and errors if that's
    // absent too.
    let result = if data_path.is_some() {
        ferx_core::run_model_with_overrides(model_path, data_path.map(String::as_str), &overrides)
    } else if simulate {
        ferx_core::run_model_simulate_with_overrides(model_path, &overrides)
    } else {
        ferx_core::run_model_with_overrides(model_path, None, &overrides)
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
            for msg in ferx_core::io::output::write_conddist_outputs(&fit_result, model_name) {
                eprintln!("{}", msg);
            }

            if estimates_format.wants_yaml() {
                let yaml_path = format!("{}-fit.yaml", model_name);
                match ferx_core::io::output::write_estimates_yaml(&fit_result, &yaml_path) {
                    Ok(()) => eprintln!("Estimates written to {}", yaml_path),
                    Err(e) => eprintln!("Warning: failed to write estimates: {}", e),
                }
            }
            if estimates_format.wants_json() {
                let json_path = format!("{}-fit.json", model_name);
                match ferx_core::io::output::write_result_json(&fit_result, &json_path) {
                    Ok(()) => eprintln!("Estimates (JSON) written to {}", json_path),
                    Err(e) => eprintln!("Warning: failed to write JSON estimates: {}", e),
                }
            }

            if let Some(out) = &output_path {
                let model_source = std::fs::read_to_string(model_path).unwrap_or_default();
                // Use the resolved data path from the fit result, not the raw
                // `--data` flag: with a model-declared `[data]` block (#690)
                // and no `--data`, the CSV that was actually fit is only known
                // here (`run_model_with_data_inits` resolved and stamped it).
                // `--simulate` is the only case with nothing to embed: any
                // successful non-simulate fit has `resolve_data_path`-guaranteed
                // `Some` data path (an `Err` there exits before this point).
                let include = if include_data {
                    if simulate {
                        eprintln!("Warning: --include-data ignored (no data file to embed)");
                    }
                    fit_result.data_path.as_ref().map(std::path::PathBuf::from)
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

            // --gam: run GAM covariate pre-screening after fitting and write CSV.
            if args.iter().any(|a| a == "--gam") {
                match gam_cmd::run_after_fit(&fit_result, &population, model_name) {
                    Ok(path) => eprintln!("GAM results written to {path}"),
                    Err(e) => eprintln!("Warning: failed to write GAM results: {e}"),
                }
            }

            let elapsed_secs = elapsed.as_secs_f64();
            eprintln!("Elapsed fit time: {:.3}s", elapsed_secs);

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

const SUMMARY_USAGE: &str = "\
Usage: ferx summary <run.fitrx> [<run2.fitrx> ...]

With one bundle: prints a psn::sumo-style summary — parameter estimates
with %RSE plus basic run info.

With two or more bundles: prints a Markdown table comparing the runs
(method, convergence, OFV/AIC/BIC, ΔOFV, runtime, sizes, and parameter
estimates) side by side.
";

/// Run the `summary` subcommand: load one or more `.fitrx` bundles.
///
/// A single bundle prints a `psn::sumo`-style summary (parameter estimates +
/// basic run info). Two or more bundles print a Markdown comparison table.
/// The column label for each run is the bundle's file stem.
///
/// Returns the process exit code: `0` = printed (incl. `--help`), `1` = a load
/// failed, `2` = usage (missing/flag-looking path).
fn run_summary(args: &[String]) -> i32 {
    // `summary` takes only bundle paths plus `-h`/`--help`. Scan every argument:
    // help anywhere prints usage (exit 0); any other flag-looking argument is a
    // usage error (exit 2) rather than being silently ignored or mistaken for a
    // file path. Everything else is a bundle path.
    let mut paths: Vec<&str> = Vec::new();
    for arg in &args[2..] {
        if is_help_flag(Some(arg)) {
            print!("{SUMMARY_USAGE}");
            return 0;
        }
        if arg.starts_with('-') {
            eprintln!("Error: unknown option {}", arg);
            eprint!("{SUMMARY_USAGE}");
            return 2;
        }
        paths.push(arg.as_str());
    }
    if paths.is_empty() {
        eprint!("{SUMMARY_USAGE}");
        return 2;
    }

    // Load all bundles up front so a bad path fails before any output.
    let mut loaded = Vec::with_capacity(paths.len());
    for path in &paths {
        match ferx_core::io::fitrx::load_fit(std::path::Path::new(path)) {
            Ok(l) => loaded.push((*path, l)),
            Err(e) => {
                eprintln!("Error: failed to load {}: {}", path, e);
                return 1;
            }
        }
    }

    if loaded.len() == 1 {
        print!(
            "{}",
            ferx_core::io::output::format_summary(&loaded[0].1.fit)
        );
    } else {
        // Column label = file stem (e.g. `run1.fitrx` → `run1`).
        let runs: Vec<(String, &ferx_core::FitResult)> = loaded
            .iter()
            .map(|(path, l)| {
                let label = std::path::Path::new(path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(path)
                    .to_string();
                (label, &l.fit)
            })
            .collect();
        print!("{}", ferx_core::io::output::format_comparison(&runs));
    }
    0
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

const CHECK_USAGE: &str = "\
Usage: ferx check <model.ferx> [--data <data.csv>] [--json]

Validates a model file without fitting and reports structured
diagnostics. With --data, also runs data-dependent checks
(covariates present, per-CMT coverage, steady-state, lag time).
--json   emit the report as JSON to stdout
";

/// Run the `check` subcommand. Returns the process exit code:
/// `0` = valid (no errors) or `--help`, `1` = errors found, `2` = usage /
/// serialization error.
fn run_check(args: &[String]) -> i32 {
    if is_help_flag(args.get(2)) {
        print!("{CHECK_USAGE}");
        return 0;
    }
    let parsed = match parse_check_args(args) {
        Ok(p) => p,
        Err(CheckArgsError::MissingDataValue) => {
            eprintln!("Error: --data requires a path (e.g. --data data.csv)");
            return 2;
        }
        Err(CheckArgsError::Usage) => {
            eprint!("{CHECK_USAGE}");
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
    // #1111: a `[covariate_model]` block is sugar over `[individual_parameters]`,
    // so show what it built. Without this the modeller cannot check the
    // generated expression against a NONMEM control stream.
    if !report.desugared_individual_parameters.is_empty() {
        println!("\n[individual_parameters] as built from [covariate_model]:");
        for line in &report.desugared_individual_parameters {
            println!("  {}", line.trim());
        }
        println!();
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

/// Which estimates file(s) to write for a completed fit.
#[derive(Clone, Copy, PartialEq, Debug)]
enum EstimatesFormat {
    Yaml,
    Json,
    Both,
}

impl EstimatesFormat {
    fn wants_yaml(self) -> bool {
        matches!(self, EstimatesFormat::Yaml | EstimatesFormat::Both)
    }
    fn wants_json(self) -> bool {
        matches!(self, EstimatesFormat::Json | EstimatesFormat::Both)
    }
}

/// Parse `--output-format yaml|json|both` (default `yaml`, current behaviour).
/// Controls only the estimates file (`{model}-fit.yaml` / `-fit.json`); the
/// sdtab/covtab/conddist CSVs are unaffected.
fn parse_output_format(args: &[String]) -> EstimatesFormat {
    let Some(idx) = args.iter().position(|a| a == "--output-format") else {
        return EstimatesFormat::Yaml;
    };
    match args.get(idx + 1).map(String::as_str) {
        Some("yaml") => EstimatesFormat::Yaml,
        Some("json") => EstimatesFormat::Json,
        Some("both") => EstimatesFormat::Both,
        _ => {
            eprintln!("Error: --output-format requires one of: yaml, json, both");
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--threads` flag. Returns `None` only when the flag is
/// **absent**; `0` and `auto` return `Some(0)`, the spelling of "the engine's own
/// worker count". Exits the process on a missing value or any other
/// non-parseable input so typos don't silently fall through to the default.
///
/// The `None` / `Some(0)` distinction is load-bearing (#1416): it is what lets
/// `--threads auto` override a model file's `[fit_options] threads = 8` while an
/// unmentioned flag leaves the file in charge. `Some(0)` is *not* a request to
/// size the global pool — `configure_global_thread_pool` rejects `0` — it is a
/// request that the model file not pin one either.
fn parse_threads_flag(args: &[String]) -> Option<usize> {
    let idx = args.iter().position(|a| a == "--threads")?;
    let value = args.get(idx + 1).unwrap_or_else(|| {
        eprintln!("Error: --threads requires a value (positive integer, 0, or 'auto')");
        std::process::exit(1);
    });
    if value.eq_ignore_ascii_case("auto") || value == "0" {
        return Some(0);
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
        classify_first_arg, edit_distance, is_help_flag, normalize_args, parse_check_args,
        parse_inits_from_nca_flag, parse_output_flag, parse_output_format, parse_threads_flag,
        print_check_human, run_check, run_summary, subcommand, unknown_tool_hint, CheckArgsError,
        EstimatesFormat, FirstArg, SUBCOMMANDS,
    };
    use ferx_core::NcaInit;

    fn args(extra: &[&str]) -> Vec<String> {
        std::iter::once("ferx")
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn parse_output_format_defaults_to_yaml() {
        let f = parse_output_format(&args(&["model.ferx", "--data", "d.csv"]));
        assert!(f.wants_yaml() && !f.wants_json());
    }

    #[test]
    fn parse_output_format_reads_explicit_values() {
        let yaml = parse_output_format(&args(&["m.ferx", "--output-format", "yaml"]));
        assert!(yaml.wants_yaml() && !yaml.wants_json());

        let json = parse_output_format(&args(&["m.ferx", "--output-format", "json"]));
        assert!(json.wants_json() && !json.wants_yaml());

        let both = parse_output_format(&args(&["m.ferx", "--output-format", "both"]));
        assert!(both.wants_yaml() && both.wants_json());
        assert_eq!(both, EstimatesFormat::Both);
    }

    // ── first-argument classification (#1396) ───────────────────────────────

    #[test]
    fn a_bare_word_is_read_as_a_tool_name_not_a_model_path() {
        // The reported case: a tool this build does not have. Before #1396 this
        // fell through to the fit path and reported a missing model file.
        assert_eq!(
            classify_first_arg("covsearch", Ok(false)),
            FirstArg::UnknownTool("covsearch")
        );
        assert_eq!(
            classify_first_arg("bootstrap", Ok(false)),
            FirstArg::UnknownTool("bootstrap")
        );
    }

    #[test]
    fn anything_with_a_dot_or_a_separator_is_a_path() {
        // A tool name is a bare word, so a dot or a separator means the user
        // got a path wrong — the two arms must not collapse into one.
        assert_eq!(
            classify_first_arg("run1.ferx", Ok(false)),
            FirstArg::MissingModelFile("run1.ferx")
        );
        assert_eq!(
            classify_first_arg("runs/run1", Ok(false)),
            FirstArg::MissingModelFile("runs/run1")
        );
        assert_eq!(
            classify_first_arg("runs\\run1", Ok(false)),
            FirstArg::MissingModelFile("runs\\run1")
        );
    }

    #[test]
    fn a_file_that_exists_is_a_model_unless_its_name_is_reserved() {
        // Extension-less model files are legal; existence wins over the name,
        // so `ferx mymodel` still fits when `mymodel` is on disk. It does not
        // win over the two decisions `main` takes before this one — a
        // SUBCOMMANDS name and a leading `-` are settled there, which is why a
        // file called `check` needs a path.
        assert_eq!(
            classify_first_arg("mymodel", Ok(true)),
            FirstArg::ModelFile("mymodel")
        );
        assert_eq!(
            classify_first_arg("run1.ferx", Ok(true)),
            FirstArg::ModelFile("run1.ferx")
        );
    }

    #[test]
    fn a_probe_that_fails_is_not_an_absent_file() {
        // `Path::exists` folds every metadata error into `false`, which would
        // report an unreadable parent or a symlink loop as "was not found" and
        // lose the actionable OS error. Only `Ok(false)` is an absent file;
        // `Err` goes to the fit path, which opens it and reports what the OS
        // says. Both spellings below reach `Ok(false)` arms above, so the two
        // cases have to be told apart here.
        let denied = || std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            classify_first_arg("run1.ferx", Err(denied())),
            FirstArg::ModelFile("run1.ferx")
        );
        assert_eq!(
            classify_first_arg("covsearch", Err(denied())),
            FirstArg::ModelFile("covsearch")
        );
    }

    #[test]
    fn a_flag_in_the_model_position_is_neither_a_tool_nor_a_file() {
        assert_eq!(
            classify_first_arg("--data", Ok(false)),
            FirstArg::Flag("--data")
        );
        assert_eq!(
            classify_first_arg("--simulate", Ok(false)),
            FirstArg::Flag("--simulate")
        );
    }

    #[test]
    fn every_subcommand_in_the_table_dispatches() {
        // The table is the single list of tool names: dispatch reads it and the
        // unknown-tool error enumerates it. A name in the list that does not
        // resolve would be advertised and then rejected.
        for (name, _) in SUBCOMMANDS {
            assert!(
                subcommand(name).is_some(),
                "`{name}` is listed but does not dispatch"
            );
        }
        // A listed name is a bare word, so `main` must consult `subcommand`
        // *before* classifying — classification alone would call every tool
        // unrecognized. `every_listed_tool_still_dispatches` (cli_ferx.rs) runs
        // the binary to pin that ordering.
        assert_eq!(
            classify_first_arg("check", Ok(false)),
            FirstArg::UnknownTool("check")
        );
        assert!(subcommand("covsearchx").is_none());
    }

    #[test]
    fn the_hint_names_a_near_miss_and_always_lists_the_tools() {
        let typo = unknown_tool_hint("covsearh");
        assert!(
            typo.contains("Did you mean `covsearch`?"),
            "one-edit typo should suggest the tool: {typo}"
        );
        // Far from every name: no suggestion, but still the full list.
        let far = unknown_tool_hint("zzzzzzzzzz");
        assert!(
            !far.contains("Did you mean"),
            "unrelated word should not suggest anything: {far}"
        );
        for (name, _) in SUBCOMMANDS {
            assert!(far.contains(name), "hint should list `{name}`: {far}");
        }
    }

    #[test]
    fn edit_distance_counts_single_edits() {
        assert_eq!(edit_distance("covsearch", "covsearch"), 0);
        assert_eq!(edit_distance("covsearh", "covsearch"), 1); // deletion
        assert_eq!(edit_distance("covsearchh", "covsearch"), 1); // insertion
        assert_eq!(edit_distance("covsearcz", "covsearch"), 1); // substitution
        assert_eq!(edit_distance("", "amd"), 3);
        assert_eq!(edit_distance("amd", ""), 3);
    }

    #[test]
    fn normalize_args_splits_eq_form() {
        assert_eq!(
            normalize_args(&args(&["model.ferx", "--data=d.csv", "--threads=4"])),
            args(&["model.ferx", "--data", "d.csv", "--threads", "4"])
        );
    }

    #[test]
    fn normalize_args_leaves_space_form_and_bare_flags_alone() {
        assert_eq!(
            normalize_args(&args(&["model.ferx", "--data", "d.csv", "--simulate"])),
            args(&["model.ferx", "--data", "d.csv", "--simulate"])
        );
    }

    #[test]
    fn normalize_args_preserves_extra_eq_signs_in_value() {
        assert_eq!(
            normalize_args(&args(&["--output=a=b.fitrx"])),
            args(&["--output", "a=b.fitrx"])
        );
    }

    #[test]
    fn normalize_args_does_not_split_inits_from_nca() {
        // parse_inits_from_nca_flag parses the combined `--inits-from-nca=X`
        // form itself; normalize_args must leave it intact rather than
        // splitting it like other `--flag=value` args.
        assert_eq!(
            normalize_args(&args(&["--inits-from-nca=nca_ebe"])),
            args(&["--inits-from-nca=nca_ebe"])
        );
    }

    #[test]
    fn threads_flag_accepts_eq_form_after_normalize() {
        assert_eq!(
            parse_threads_flag(&normalize_args(&args(&["--threads=4"]))),
            Some(4)
        );
    }

    #[test]
    fn output_flag_accepts_eq_form_after_normalize() {
        assert_eq!(
            parse_output_flag(&normalize_args(&args(&["--output=run1.fitrx"]))),
            Some("run1.fitrx".to_string())
        );
    }

    #[test]
    fn check_args_data_accepts_eq_form_after_normalize() {
        let argv = normalize_args(&args(&["check", "model.ferx", "--data=d.csv", "--json"]));
        let a = parse_check_args(&argv).unwrap();
        assert_eq!(a.data, Some("d.csv"));
        assert!(a.json);
    }

    #[test]
    fn absent_flag_is_none() {
        assert_eq!(parse_threads_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn positive_integer_parses() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "4"])), Some(4));
    }

    // `Some(0)` rather than `None` for 0/auto (#1416): the flag was named, so it
    // overrides the model file's `[fit_options] threads`; it just names the
    // engine's own count rather than a width. `None` is reserved for an absent
    // flag — see `absent_flag_is_none` above, the other half of the pair.
    #[test]
    fn zero_means_the_default_was_asked_for_by_name() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "0"])), Some(0));
    }

    #[test]
    fn auto_means_the_default_was_asked_for_by_name() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "auto"])), Some(0));
        assert_eq!(parse_threads_flag(&args(&["--threads", "AUTO"])), Some(0));
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

    // `CARGO_MANIFEST_DIR` is `crates/ferx-cli`, so the repo-root fixtures
    // (`examples/`, `data/`) are two levels up (#1114).
    const VALID_MODEL: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/one_cpt_iv.ferx"
    );
    const VALID_DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/one_cpt_iv.csv");

    #[test]
    fn run_check_usage_errors_return_2() {
        // No model path, and a flag where the model should be — both usage (2).
        assert_eq!(run_check(&args(&["check"])), 2);
        assert_eq!(run_check(&args(&["check", "--json"])), 2);
    }

    #[test]
    fn run_check_help_returns_0() {
        assert_eq!(run_check(&args(&["check", "-h"])), 0);
        assert_eq!(run_check(&args(&["check", "--help"])), 0);
    }

    #[test]
    fn run_summary_help_returns_0() {
        assert_eq!(run_summary(&args(&["summary", "-h"])), 0);
        assert_eq!(run_summary(&args(&["summary", "--help"])), 0);
    }

    #[test]
    fn is_help_flag_recognizes_both_spellings_only() {
        assert!(is_help_flag(Some(&"-h".to_string())));
        assert!(is_help_flag(Some(&"--help".to_string())));
        assert!(!is_help_flag(Some(&"--json".to_string())));
        assert!(!is_help_flag(None));
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

    // ── run_summary: in-process coverage of the `summary` subcommand ──────────

    const WARFARIN_MODEL: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/warfarin.ferx");

    #[test]
    fn run_summary_usage_errors_return_2() {
        // No path, and a flag where the path should be — both usage (2).
        assert_eq!(run_summary(&args(&["summary"])), 2);
        assert_eq!(run_summary(&args(&["summary", "--json"])), 2);
        // An unknown flag *after* a path is rejected, not silently ignored.
        assert_eq!(
            run_summary(&args(&["summary", "/no/such/file.fitrx", "--bogus"])),
            2
        );
        // A short unknown flag (single dash) is rejected too.
        assert_eq!(run_summary(&args(&["summary", "-x"])), 2);
    }

    #[test]
    fn run_summary_help_anywhere_returns_0() {
        // Help is recognized even when it follows a path (before any load).
        assert_eq!(
            run_summary(&args(&["summary", "/no/such/file.fitrx", "-h"])),
            0
        );
        assert_eq!(
            run_summary(&args(&["summary", "/no/such/file.fitrx", "--help"])),
            0
        );
    }

    #[test]
    fn run_summary_missing_file_returns_1() {
        assert_eq!(run_summary(&args(&["summary", "/no/such/file.fitrx"])), 1);
    }

    /// Simulate the warfarin model and write it to a `.fitrx` at `path` (fast —
    /// no optimization loop), so `run_summary`'s success arms can be covered.
    fn write_warfarin_fitrx(path: &std::path::Path) {
        let (fit, pop) = ferx_core::run_model_simulate(WARFARIN_MODEL).expect("simulate warfarin");
        let src = std::fs::read_to_string(WARFARIN_MODEL).expect("read model");
        ferx_core::io::fitrx::save_fit(
            &fit,
            &pop,
            &src,
            path,
            ferx_core::io::fitrx::SaveFitOptions::default(),
        )
        .expect("save fitrx");
    }

    #[test]
    fn run_summary_valid_fitrx_returns_0() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("run.fitrx");
        write_warfarin_fitrx(&path);
        assert_eq!(run_summary(&args(&["summary", path.to_str().unwrap()])), 0);
    }

    #[test]
    fn run_summary_multiple_fitrx_returns_0() {
        // Two bundles → the comparison-table arm.
        let dir = tempfile::tempdir().expect("tempdir");
        let p1 = dir.path().join("run1.fitrx");
        let p2 = dir.path().join("run2.fitrx");
        write_warfarin_fitrx(&p1);
        write_warfarin_fitrx(&p2);
        assert_eq!(
            run_summary(&args(&[
                "summary",
                p1.to_str().unwrap(),
                p2.to_str().unwrap()
            ])),
            0
        );
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
