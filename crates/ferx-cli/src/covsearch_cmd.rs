//! `ferx covsearch` — stepwise covariate modelling (#1180).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::covsearch::run_covsearch`], print the step table,
//! pick an exit code. Every decision about candidates, tests and selection
//! lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::covsearch::{
    default_dir, render_summary, run_covsearch, CovsearchEvent, CovsearchOptions, CovsearchRun,
    Phase,
};
use ferx_tools::search::SearchConfig;

pub const COVSEARCH_USAGE: &str = "\
Usage: ferx covsearch <search.ferxsearch> [options]

Stepwise covariate modelling — PsN scm / Pharmpy covsearch. The .ferxsearch
file names the base model and data, the candidate effects as an MFL space
(`COVARIATE?(@IIV, @CONTINUOUS, [pow,lin])`; `COVARIATE(...)` without `?`
forces an effect in), and the search in a [covsearch] section:

  algorithm  = \"scm-forward-then-backward\"   # or \"scm-forward\"
  p_forward  = 0.01
  p_backward = 0.001
  max_steps  = 10                              # omit for unlimited
  adaptive_scope_reduction = false             # SCM+

Each forward step adds every remaining effect on its own, fits the
candidates in parallel, and keeps the largest OFV drop that is significant
by the likelihood-ratio test; the backward phase removes the cheapest
effect whose removal is not significant. Every candidate is fitted with
[run] retries perturbed restarts and judged by the [strictness] gate before
it can win: a fit that stalled at its initial estimates is excluded, with
the reason in the table.

  --directory DIR      where the per-step journals, steps.csv, final.ferx and
                       final-fit.yaml go (default {search}-covsearch, next to
                       the .ferxsearch file; [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print step progress to stderr

  -h, --help           print this help and exit
";

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

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

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Walk the arguments once, rejecting unknown flags and returning the
/// positionals — skipping each value flag's argument so `--threads 4
/// search.ferxsearch` does not read `4` as the file.
pub(crate) fn scan_args<'a>(
    args: &'a [String],
    value_flags: &[&str],
    bool_flags: &[&str],
) -> Result<Vec<&'a str>, String> {
    let mut positionals = Vec::new();
    let mut i = 2; // skip `ferx`, the subcommand
    while i < args.len() {
        let a = &args[i];
        if !a.starts_with('-') {
            positionals.push(a.as_str());
            i += 1;
            continue;
        }
        if value_flags.contains(&a.as_str()) {
            i += 2;
            continue;
        }
        if bool_flags.contains(&a.as_str()) {
            i += 1;
            continue;
        }
        return Err(format!("unknown flag: {a}"));
    }
    Ok(positionals)
}

pub(crate) fn parse_threads(args: &[String]) -> Result<Option<usize>, String> {
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

/// Entry point for `ferx covsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{COVSEARCH_USAGE}");
        return 0;
    }
    match run_covsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_covsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{COVSEARCH_USAGE}");
            return Ok(1);
        }
        many => {
            return Err(format!(
                "expected one .ferxsearch file, got {}: {}",
                many.len(),
                many.join(", ")
            ))
        }
    };
    let threads = parse_threads(args)?;
    let quiet = flag(args, "--quiet");

    let mut config = SearchConfig::load(&config_path)?;
    // Refuse a file covsearch cannot honour before the dataset is read.
    CovsearchOptions::from_config(&config)?;
    if flag(args, "--resume") {
        config.run.resume = true;
    }
    let dir = match value(args, "--directory")? {
        Some(d) => PathBuf::from(d),
        None => match &config.run.cache_dir {
            Some(d) => config.dir.join(d),
            None => default_dir(&config_path),
        },
    };

    let base = config.load_base()?;
    eprintln!("Base model: {}", config.base.display());
    eprintln!(
        "Data:       {} subjects, {} observations",
        base.prepared.population.subjects.len(),
        base.prepared.population.n_obs()
    );
    eprintln!("Directory:  {}", dir.display());

    let progress = |event: CovsearchEvent| {
        if quiet {
            return;
        }
        match event {
            CovsearchEvent::BaseStarted => eprintln!("Fitting the base model..."),
            CovsearchEvent::BaseFinished { ofv, n_parameters } => {
                eprintln!("Base model: OFV {ofv:.3}, {n_parameters} free parameters")
            }
            CovsearchEvent::StepStarted {
                step,
                phase,
                candidates,
            } => eprintln!(
                "Step {step} ({}): fitting {candidates} candidate{}...",
                phase.label(),
                if candidates == 1 { "" } else { "s" }
            ),
            CovsearchEvent::StepFinished {
                step,
                phase,
                selected,
            } => match selected {
                Some((effect, ofv)) => eprintln!(
                    "Step {step} ({}): {} {} (OFV {ofv:.3})",
                    phase.label(),
                    if phase == Phase::Backward {
                        "removed"
                    } else {
                        "added"
                    },
                    effect.label()
                ),
                None => eprintln!("Step {step} ({}): nothing accepted", phase.label()),
            },
        }
    };

    let result = run_covsearch(
        &config,
        &base,
        CovsearchRun {
            dir: Some(dir.clone()),
            threads,
            cancel: None,
            progress: Some(&progress),
        },
    )?;

    print!("{}", render_summary(&result));
    if let Some(fit) = &result.final_fit {
        let yaml = dir.join("final-fit.yaml");
        match ferx_core::io::output::write_estimates_yaml(fit, &yaml.to_string_lossy()) {
            Ok(()) => eprintln!("Final estimates written to {}", yaml.display()),
            Err(e) => eprintln!("Warning: failed to write {}: {e}", yaml.display()),
        }
    }
    eprintln!(
        "Step table written to {}; final model to {}",
        ferx_tools::covsearch::steps_path(&dir).display(),
        ferx_tools::covsearch::final_model_path(&dir).display()
    );
    Ok(if result.cancelled { 130 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn scan_args_skips_value_flags_and_rejects_unknown_ones() {
        let a = args(&[
            "ferx",
            "covsearch",
            "--threads",
            "4",
            "s.ferxsearch",
            "--resume",
        ]);
        assert_eq!(
            scan_args(&a, VALUE_FLAGS, BOOL_FLAGS).unwrap(),
            vec!["s.ferxsearch"]
        );
        assert_eq!(parse_threads(&a).unwrap(), Some(4));
        let e = scan_args(
            &args(&["ferx", "covsearch", "--samples", "3"]),
            VALUE_FLAGS,
            BOOL_FLAGS,
        )
        .unwrap_err();
        assert_eq!(e, "unknown flag: --samples");
        let e = parse_threads(&args(&["ferx", "covsearch", "--threads", "0"])).unwrap_err();
        assert!(e.contains("positive integer"), "{e}");
        let e = value(
            &args(&["ferx", "covsearch", "--directory", "--resume"]),
            "--directory",
        )
        .unwrap_err();
        assert!(e.contains("requires a value but got '--resume'"), "{e}");
        let e = value(&args(&["ferx", "covsearch", "--directory"]), "--directory").unwrap_err();
        assert!(e.contains("requires a value"), "{e}");
    }

    #[test]
    fn run_rejects_two_files_and_a_missing_file() {
        assert_eq!(
            run(&args(&[
                "ferx",
                "covsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "covsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "covsearch", "-h"])), 0);
    }
}
