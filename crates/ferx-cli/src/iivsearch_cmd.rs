//! `ferx iivsearch` — variability-structure search (#1183).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::iivsearch::run_iivsearch`], print the model
//! table, pick an exit code. Every decision about candidates, ranking and
//! selection lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::iivsearch::{
    default_dir, render_summary, run_iivsearch, IivsearchEvent, IivsearchOptions, IivsearchRun,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{flag, parse_threads, scan_args, value};

pub const IIVSEARCH_USAGE: &str = "\
Usage: ferx iivsearch <search.ferxsearch> [options]

Variability-structure search — Pharmpy iivsearch. The .ferxsearch file names
the base model and data, the space as MFL (`IIV?([CL,V,KA], EXP);
COVARIANCE?(IIV, [CL,V,KA])` — a plain `IIV(CL, EXP)` keeps that η in every
candidate), the criterion in [rank], and the search in an [iivsearch] section:

  algorithm             = \"top_down_exhaustive\"   # or \"bottom_up_stepwise\",
                                                  # \"simultaneous_stepwise\", \"skip\"
  correlation_algorithm = \"top_down_exhaustive\"   # or \"skip\"; Pharmpy's default when unset
  block_retries         = 2                       # extra starts per η beyond two in a block

Two stages: the number of η under `algorithm`, then the block structure over
the COVARIANCE parameters under `correlation_algorithm`. Every candidate is
derived from its parent by the η / block edits and seeded from the parent's
estimates (a new block's correlations from the parent's EBEs). Candidates are
fitted in parallel with [run] retries perturbed restarts, judged by the
[strictness] gate, and ranked on [rank] type — the BIC(iiv) by default, which
is what `bic` means here; with [rank] cutoff a candidate must beat its parent
by that much. The final model is compared with the input, and the input is
returned when it ranks better. A parameter not in the canonical
`P = TVP * exp(ETA_P)` form is refused by name before anything is fitted.

  --directory DIR      where the per-step journals, models.csv, models/,
                       final.ferx and final-fit.yaml go (default
                       {search}-iivsearch, next to the .ferxsearch file;
                       [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print step progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Entry point for `ferx iivsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{IIVSEARCH_USAGE}");
        return 0;
    }
    match run_iivsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_iivsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{IIVSEARCH_USAGE}");
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
    // Refuse a file iivsearch cannot honour before the dataset is read.
    let options = IivsearchOptions::from_config(&config)?;
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
        "Algorithm:  {}{}, ranked on {}",
        options.algorithm.label(),
        if options.block_stage() {
            " + top_down_exhaustive blocks"
        } else {
            ""
        },
        options.criterion().label()
    );
    eprintln!("Directory:  {}", dir.display());

    let progress = |event: IivsearchEvent| {
        if quiet {
            return;
        }
        match event {
            IivsearchEvent::InputStarted => eprintln!("Fitting the input model..."),
            IivsearchEvent::InputFinished { ofv, criterion } => {
                eprintln!("Input model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            IivsearchEvent::BaseStarted => eprintln!("Fitting the base model..."),
            IivsearchEvent::BaseFinished { ofv, criterion } => {
                eprintln!("Base model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            IivsearchEvent::StepStarted {
                step,
                kind,
                candidates,
            } => eprintln!(
                "Step {step} ({}): fitting {candidates} candidate{}...",
                kind.label(),
                if candidates == 1 { "" } else { "s" }
            ),
            IivsearchEvent::StepFinished {
                step,
                best,
                improved,
            } => eprintln!(
                "Step {step}: {} {} (criterion {:.3})",
                if improved { "best" } else { "kept" },
                best.0,
                best.1
            ),
            IivsearchEvent::Reverted { criterion } => {
                eprintln!("The input model ranks better (criterion {criterion:.3}); returning it")
            }
        }
    };

    let result = run_iivsearch(
        &config,
        &base,
        IivsearchRun {
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
        ferx_tools::iivsearch::models_path(&dir).display(),
        ferx_tools::iivsearch::final_model_path(&dir).display()
    );
    Ok(if result.cancelled { 130 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The warfarin model, evaluating rather than fitting (`maxiter = 0`).
    const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
";

    /// The command end to end on the warfarin model, then again
    /// `--quiet --resume` so the journal is reused and the progress
    /// printer's early return is taken.
    ///
    /// `bottom_up_stepwise` is the algorithm that reaches every event the
    /// printer has: a derived base (the input minus the searched η), a
    /// step, the block stage, and — since an evaluation cannot improve on
    /// the input — the reversion to it.
    #[test]
    fn run_searches_writes_the_files_and_resumes_quietly() {
        let dir = tempfile::tempdir().unwrap();
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/warfarin.csv");
        std::fs::write(dir.path().join("base.ferx"), BASE).unwrap();
        let config = dir.path().join("search.ferxsearch");
        std::fs::write(
            &config,
            format!(
                "base = \"base.ferx\"\ndata = \"{}\"\n\n[space]\nmfl = \"IIV(CL,EXP);\
                 IIV?([V],EXP);COVARIANCE?(IIV,[CL,V])\"\n\n[iivsearch]\nalgorithm = \
                 \"bottom_up_stepwise\"\n\n[strictness]\nrequire_converged = \
                 false\nreject_init_stall = false\nreject_on_boundary = false\n\n[run]\nretries \
                 = 0\nthreads = 2\n",
                data.display()
            ),
        )
        .unwrap();
        let out = dir.path().join("run");
        let common = [
            "ferx".to_string(),
            "iivsearch".to_string(),
            config.to_string_lossy().into_owned(),
            "--directory".to_string(),
            out.to_string_lossy().into_owned(),
            "--threads".to_string(),
            "2".to_string(),
        ];
        assert_eq!(run(&common), 0);
        for file in ["models.csv", "final.ferx", "final-fit.yaml"] {
            assert!(out.join(file).is_file(), "{file} missing");
        }
        assert!(out.join("models/input.ferx").is_file());
        let mut again = common.to_vec();
        again.push("--quiet".to_string());
        again.push("--resume".to_string());
        assert_eq!(run(&again), 0);
    }

    /// A model the space cannot search is refused by name, before the
    /// dataset is read.
    #[test]
    fn run_reports_a_non_canonical_parameter_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/warfarin.csv");
        std::fs::write(
            dir.path().join("base.ferx"),
            BASE.replace("  V = TVV * exp(ETA_V)", "  V = TVV * (1 + ETA_V)"),
        )
        .unwrap();
        let config = dir.path().join("search.ferxsearch");
        std::fs::write(
            &config,
            format!(
                "base = \"base.ferx\"\ndata = \"{}\"\n\n[space]\nmfl = \
                 \"IIV?([CL,V],EXP)\"\n\n[run]\nretries = 0\nthreads = 2\n",
                data.display()
            ),
        )
        .unwrap();
        assert_eq!(
            run(&args(&[
                "ferx",
                "iivsearch",
                &config.to_string_lossy(),
                "--quiet"
            ])),
            1
        );
    }

    #[test]
    fn run_rejects_two_files_and_a_missing_file_and_prints_help() {
        assert_eq!(
            run(&args(&[
                "ferx",
                "iivsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "iivsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "iivsearch", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "iivsearch"])), 1);
        assert_eq!(
            run(&args(&[
                "ferx",
                "iivsearch",
                "x.ferxsearch",
                "--samples",
                "3"
            ])),
            1
        );
    }
}
