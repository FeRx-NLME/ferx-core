//! `ferx modelsearch` — structural PK model search (#1181).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::modelsearch::run_modelsearch`], print the model
//! table, pick an exit code. Every decision about candidates, ranking and
//! selection lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::modelsearch::{
    default_dir, render_summary, run_modelsearch, ModelsearchEvent, ModelsearchOptions,
    ModelsearchRun,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{parse_threads, scan_args};

pub const MODELSEARCH_USAGE: &str = "\
Usage: ferx modelsearch <search.ferxsearch> [options]

Structural PK model search — Pharmpy modelsearch. The .ferxsearch file names
the base model and data, the structural space as MFL (`ABSORPTION([INST,FO]);
PERIPHERALS(0..2); TRANSITS([0,1,3], NODEPOT); LAGTIME([OFF,ON])`), the
criterion in [rank], and the search in a [modelsearch] section:

  algorithm    = \"reduced_stepwise\"     # or \"exhaustive_stepwise\", \"exhaustive\"
  iiv_strategy = \"absorption_delay\"     # or \"add_diagonal\", \"no_add\"

Every candidate is one `pk` template swap from its parent, with the new
parameters declared from the parent's estimates. Candidates are fitted in
parallel with [run] retries perturbed restarts, judged by the [strictness]
gate, and ranked on [rank] type (the mixed BIC by default); with [rank]
cutoff a candidate must beat the base by that much to be selected. The table
shows every model's structure, criterion, rank, convergence status and fit
time, and an excluded model carries its reason.

  --directory DIR      where the per-layer journals, models.csv, final.ferx and
                       final-fit.yaml go (default {search}-modelsearch, next to
                       the .ferxsearch file; [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print layer progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

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

/// Entry point for `ferx modelsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{MODELSEARCH_USAGE}");
        return 0;
    }
    match run_modelsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_modelsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{MODELSEARCH_USAGE}");
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
    // Refuse a file modelsearch cannot honour before the dataset is read.
    let options = ModelsearchOptions::from_config(&config)?;
    ModelsearchOptions::check_space(&config)?;
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
    eprintln!("Space:      {}", config.mfl.render());
    eprintln!(
        "Algorithm:  {}, ranked on {}",
        options.algorithm.label(),
        options.criterion().label()
    );
    eprintln!("Directory:  {}", dir.display());

    let progress = |event: ModelsearchEvent| {
        if quiet {
            return;
        }
        match event {
            ModelsearchEvent::InputStarted => eprintln!("Fitting the input model..."),
            ModelsearchEvent::BaseStarted => eprintln!("Fitting the base model..."),
            ModelsearchEvent::BaseFinished { ofv, criterion } => {
                eprintln!("Base model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            ModelsearchEvent::LayerStarted { layer, candidates } => eprintln!(
                "Layer {layer}: fitting {candidates} candidate{}...",
                if candidates == 1 { "" } else { "s" }
            ),
            ModelsearchEvent::LayerFinished { layer, best } => match best {
                Some((id, criterion)) => {
                    eprintln!("Layer {layer}: best {id} (criterion {criterion:.3})")
                }
                None => eprintln!("Layer {layer}: no candidate passed the gate"),
            },
        }
    };

    let result = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
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
        "Model table written to {}; final model to {}",
        ferx_tools::modelsearch::models_path(&dir).display(),
        ferx_tools::modelsearch::final_model_path(&dir).display()
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
                "modelsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "modelsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "modelsearch", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "modelsearch"])), 1);
        assert_eq!(
            run(&args(&[
                "ferx",
                "modelsearch",
                "x.ferxsearch",
                "--samples",
                "3"
            ])),
            1
        );
    }
}
