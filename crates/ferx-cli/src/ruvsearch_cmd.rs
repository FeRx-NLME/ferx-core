//! `ferx ruvsearch` — residual-error model search (#1182).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::ruvsearch::run_ruvsearch`], print the step
//! table, pick an exit code. Every decision about candidates, tests and
//! selection lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::ruvsearch::{
    default_dir, render_summary, run_ruvsearch, RuvsearchEvent, RuvsearchOptions, RuvsearchRun,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{flag, parse_threads, scan_args, value};

pub const RUVSEARCH_USAGE: &str = "\
Usage: ferx ruvsearch <search.ferxsearch> [options]

Residual-error model search — Pharmpy ruvsearch. The .ferxsearch file names
the base model and data (it has no [space]: the candidates are the four
residual-error forms) and the search in a [ruvsearch] section:

  groups          = 4          # time-varying candidates cut at the i/groups TAD quantiles
  p_value         = 0.001      # likelihood-ratio level, and the final df = 1 cutoff
  skip            = []         # any of \"IIV_on_RUV\", \"power\", \"combined\", \"time_varying\"
  max_iter        = 3          # 1, 2 or 3
  cwres_prescreen = false      # screen candidates on the parent's CWRES first

An input whose error model is not plain proportional is first refitted as
one. Each iteration adds IIV on the residual error, a power form, a combined
form and a time-varying magnitude to the parent on their own, fits them in
parallel with [run] retries perturbed restarts, and keeps the largest OFV
drop that is significant by the likelihood-ratio test and passes the
[strictness] gate; the accepted form's family is not tested again (power and
combined retire each other). The selected model must beat the input by the
df = 1 cutoff or the input is returned. With cwres_prescreen the candidates
are fitted to the parent's CWRES first — Pharmpy's path — and only the
winner is refitted on the data.

  --directory DIR      where the per-step journals, steps.csv, models/,
                       final.ferx and final-fit.yaml go (default
                       {search}-ruvsearch, next to the .ferxsearch file;
                       [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Entry point for `ferx ruvsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{RUVSEARCH_USAGE}");
        return 0;
    }
    match run_ruvsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_ruvsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{RUVSEARCH_USAGE}");
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
    // Refuse a file ruvsearch cannot honour before the dataset is read.
    let options = RuvsearchOptions::from_config(&config)?;
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
    eprintln!(
        "Search:     p = {}, groups = {}, max_iter = {}{}{}",
        options.p_value,
        options.groups,
        options.max_iter,
        if options.cwres_prescreen {
            ", CWRES pre-screen"
        } else {
            ""
        },
        if options.skip.is_empty() {
            String::new()
        } else {
            format!(
                ", skipping {}",
                options
                    .skip
                    .iter()
                    .map(|f| f.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );
    eprintln!("Directory:  {}", dir.display());

    let progress = |event: RuvsearchEvent| {
        if quiet {
            return;
        }
        match event {
            RuvsearchEvent::InputStarted => eprintln!("Fitting the input model..."),
            RuvsearchEvent::InputFinished { ofv } => eprintln!("Input model: OFV {ofv:.3}"),
            RuvsearchEvent::BaseStarted => {
                eprintln!("Fitting the proportional base model...")
            }
            RuvsearchEvent::BaseFinished { ofv } => {
                eprintln!("Proportional base: OFV {ofv:.3}")
            }
            RuvsearchEvent::IterationStarted {
                iteration,
                candidates,
                screening,
            } => eprintln!(
                "Iteration {iteration}: {} {candidates} candidate{}...",
                if screening { "screening" } else { "fitting" },
                if candidates == 1 { "" } else { "s" }
            ),
            RuvsearchEvent::Screened { iteration, feature } => match feature {
                Some(f) => eprintln!("Iteration {iteration}: refitting {} on the data", f.label()),
                None => eprintln!("Iteration {iteration}: no candidate beat the cutoff on CWRES"),
            },
            RuvsearchEvent::IterationFinished {
                iteration,
                selected,
            } => match selected {
                Some((feature, ofv)) => {
                    eprintln!(
                        "Iteration {iteration}: added {} (OFV {ofv:.3})",
                        feature.label()
                    )
                }
                None => eprintln!("Iteration {iteration}: nothing accepted"),
            },
            RuvsearchEvent::Reverted { to, ofv } => {
                eprintln!("Final comparison: the {to} model (OFV {ofv:.3}) is returned")
            }
        }
    };

    let result = run_ruvsearch(
        &config,
        &base,
        RuvsearchRun {
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
        ferx_tools::ruvsearch::steps_path(&dir).display(),
        ferx_tools::ruvsearch::final_model_path(&dir).display()
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
    fn run_rejects_two_files_and_a_missing_file_and_prints_help() {
        assert_eq!(
            run(&args(&[
                "ferx",
                "ruvsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "ruvsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "ruvsearch", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "ruvsearch"])), 1);
        assert_eq!(
            run(&args(&[
                "ferx",
                "ruvsearch",
                "x.ferxsearch",
                "--samples",
                "3"
            ])),
            1
        );
    }
}
