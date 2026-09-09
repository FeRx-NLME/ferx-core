//! `ferx globalsearch` — global model search, GA or exhaustive (#1185).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::globalsearch::run_globalsearch`], print the
//! model table, pick an exit code. Every decision about the grid, the
//! fitness and the selection lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::globalsearch::{
    default_dir, render_summary, run_globalsearch, GlobalsearchEvent, GlobalsearchOptions,
    GlobalsearchRun,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{flag, parse_threads, scan_args, value};

pub const GLOBALSEARCH_USAGE: &str = "\
Usage: ferx globalsearch <search.ferxsearch> [options]

Global model search — pyDarwin's genetic algorithm, or exhaustive enumeration,
over one grid of structural and covariate choices, ranked on pyDarwin's
penalized fitness. The .ferxsearch file names the base model and data, the
grid as MFL (every structural category is an axis with its values as alleles,
every `COVARIATE?` pair an axis with `none` and each of its forms), the
criterion in [rank] (`penalized` by default: OFV + 10 per estimated
parameter + 100 per failure; `[rank.penalties]` changes the charges), and the
search in a [globalsearch] section:

  algorithm    = \"ga\"                  # or \"exhaustive\"
  iiv_strategy = \"absorption_delay\"    # or \"add_diagonal\", \"no_add\"
  max_models   = 500                   # the largest grid `exhaustive` enumerates

  [globalsearch.ga]
  population_size = 20
  generations     = 10
  seed            = 12345              # …and the crossover / mutation / niche knobs

Every candidate is one grid point decoded from the input model — a `pk`
template swap plus one `[covariate_model]` line per relation — fitted in
parallel with [run] retries perturbed restarts, judged by the [strictness]
gate, and scored. A gene that changes nothing (a covariate on a parameter
the structure removed), a candidate that cannot be built, and a fit the gate
refused are charged on top. The table shows every model's genome, criterion,
fitness, rank, convergence status and fit time; an excluded model carries
its reason.

  --directory DIR      where the per-batch journals, models.csv, generations.csv,
                       final.ferx and final-fit.yaml go (default {search}-globalsearch,
                       next to the .ferxsearch file; [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory (the GA
                       is seeded, so a resumed run proposes the same genomes)
  --reuse-from DIR     another search's directory (a modelsearch or covsearch
                       run, say) whose cached fits this run may reuse: a
                       candidate already fitted there to the same data with the
                       same settings is re-scored, not refitted. Adds to the
                       file's [run] reuse_from
  --quiet              do not print batch progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads", "--reuse-from"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Entry point for `ferx globalsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{GLOBALSEARCH_USAGE}");
        return 0;
    }
    match run_globalsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_globalsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{GLOBALSEARCH_USAGE}");
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
    let reuse_from: Vec<PathBuf> = value(args, "--reuse-from")?
        .map(PathBuf::from)
        .into_iter()
        .collect();

    let mut config = SearchConfig::load(&config_path)?;
    // Refuse a file the grid cannot lay out before the dataset is read.
    let options = GlobalsearchOptions::from_config(&config)?;
    GlobalsearchOptions::check_space(&config)?;
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

    let progress = |event: GlobalsearchEvent| {
        if quiet {
            return;
        }
        match event {
            GlobalsearchEvent::Space { axes, size } => eprintln!(
                "Grid:       {size} point{} over {axes} ax{}",
                if size == 1 { "" } else { "s" },
                if axes == 1 { "is" } else { "es" }
            ),
            GlobalsearchEvent::InputStarted => eprintln!("Fitting the input model..."),
            GlobalsearchEvent::InputFinished { ofv, criterion } => {
                eprintln!("Input model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            GlobalsearchEvent::BatchStarted {
                step,
                proposed,
                candidates,
            } => eprintln!(
                "{step}: fitting {candidates} new model{} of {proposed} proposed...",
                if candidates == 1 { "" } else { "s" }
            ),
            GlobalsearchEvent::BatchFinished { step, best } => match best {
                Some((id, fitness)) => eprintln!("{step}: best {id} (fitness {fitness:.3})"),
                None => eprintln!("{step}: no candidate passed the gate"),
            },
        }
    };

    let result = run_globalsearch(
        &config,
        &base,
        GlobalsearchRun {
            dir: Some(dir.clone()),
            threads,
            cancel: None,
            progress: Some(&progress),
            reuse_from,
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
        ferx_tools::globalsearch::models_path(&dir).display(),
        ferx_tools::globalsearch::final_model_path(&dir).display()
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
                "globalsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "globalsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "globalsearch", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "globalsearch"])), 1);
        assert_eq!(
            run(&args(&[
                "ferx",
                "globalsearch",
                "x.ferxsearch",
                "--samples",
                "3"
            ])),
            1
        );
    }
}
