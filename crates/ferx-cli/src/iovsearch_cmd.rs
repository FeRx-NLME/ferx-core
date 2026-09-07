//! `ferx iovsearch` — inter-occasion variability search (#1183).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch`
//! file, call [`ferx_tools::iovsearch::run_iovsearch`], print the model
//! table, pick an exit code.

use std::path::PathBuf;

use ferx_tools::iovsearch::{
    default_dir, render_summary, run_iovsearch, IovsearchEvent, IovsearchOptions, IovsearchRun,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{flag, parse_threads, scan_args, value};

pub const IOVSEARCH_USAGE: &str = "\
Usage: ferx iovsearch <search.ferxsearch> [options]

Inter-occasion variability search — Pharmpy iovsearch. The .ferxsearch file
names the base model and data; an optional [space] names the parameters a κ
is tried on (`IOV?([CL,V], EXP)`; a plain `IOV(CL, EXP)` keeps that κ), the
default being every parameter with a free η; the criterion goes in [rank] and
the search in an [iovsearch] section:

  distribution  = \"same-as-iiv\"   # or \"disjoint\", \"joint\", \"explicit\" (+ groups)
  groups        = [[\"CL\",\"V\"]]     # the κ blocks, with distribution = \"explicit\"
  column        = \"OCC\"           # optional; must match the base's iov_column
  block_retries = 2               # extra starts per κ beyond two in a block

Two steps: a model with κ on every candidate parameter, then every proper
subset of those κ removed, ranked with the input on [rank] type — the
BIC(random) by default, which is what `bic` means here; then, from the
winner, every subset of the η that sit beside a κ removed. The base model
must declare `iov_column` in [fit_options] (that is what reads the occasions)
and every candidate parameter must be in the canonical
`P = TVP * exp(ETA_P)` form.

  --directory DIR      where the per-step journals, models.csv, models/,
                       final.ferx and final-fit.yaml go (default
                       {search}-iovsearch, next to the .ferxsearch file;
                       [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print step progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Entry point for `ferx iovsearch ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{IOVSEARCH_USAGE}");
        return 0;
    }
    match run_iovsearch_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_iovsearch_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{IOVSEARCH_USAGE}");
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
    let options = IovsearchOptions::from_config(&config)?;
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
    if config.has_space() {
        eprintln!("Space:      {}", config.mfl.render());
    } else {
        eprintln!("Space:      every parameter with a free η (Pharmpy's default)");
    }
    eprintln!(
        "Distribution: {}, ranked on {}",
        options.distribution.label(),
        options.criterion().label()
    );
    eprintln!("Directory:  {}", dir.display());

    let progress = |event: IovsearchEvent| {
        if quiet {
            return;
        }
        match event {
            IovsearchEvent::InputStarted => eprintln!("Fitting the input model..."),
            IovsearchEvent::InputFinished { ofv, criterion } => {
                eprintln!("Input model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            IovsearchEvent::FullIovStarted { parameters } => eprintln!(
                "Fitting the model with IOV on all {parameters} candidate parameter{}...",
                if parameters == 1 { "" } else { "s" }
            ),
            IovsearchEvent::FullIovFinished { ofv, criterion } => {
                eprintln!("Full-IOV model: OFV {ofv:.3}, criterion {criterion:.3}")
            }
            IovsearchEvent::StepStarted { step, candidates } => eprintln!(
                "Step {step}: fitting {candidates} candidate{}...",
                if candidates == 1 { "" } else { "s" }
            ),
            IovsearchEvent::StepFinished {
                step,
                best,
                improved,
            } => eprintln!(
                "Step {step}: {} {} (criterion {:.3})",
                if improved { "best" } else { "kept" },
                best.0,
                best.1
            ),
        }
    };

    let result = run_iovsearch(
        &config,
        &base,
        IovsearchRun {
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
        ferx_tools::iovsearch::models_path(&dir).display(),
        ferx_tools::iovsearch::final_model_path(&dir).display()
    );
    Ok(if result.cancelled { 130 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The warfarin model over two occasions, IIV only, evaluating rather
    /// than fitting (`maxiter = 0`), reading its occasions from `OCC`.
    const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)

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
  iov_column = OCC
";

    fn write_config(dir: &std::path::Path, space: &str, section: &str) -> std::path::PathBuf {
        let data =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/warfarin_iov.csv");
        std::fs::write(dir.join("base.ferx"), BASE).unwrap();
        let config = dir.join("search.ferxsearch");
        std::fs::write(
            &config,
            format!(
                "base = \"base.ferx\"\ndata = \"{}\"\n\n{space}{section}[strictness]\n\
                 require_converged = false\nreject_init_stall = false\nreject_on_boundary = \
                 false\n\n[run]\nretries = 0\nthreads = 2\n",
                data.display()
            ),
        )
        .unwrap();
        config
    }

    /// The command end to end, then again `--quiet --resume` so the journal
    /// is reused and the progress printer's early return is taken. One
    /// candidate parameter keeps it to two fits.
    #[test]
    fn run_searches_writes_the_files_and_resumes_quietly() {
        let dir = tempfile::tempdir().unwrap();
        let config = write_config(dir.path(), "[space]\nmfl = \"IOV?([CL],EXP)\"\n\n", "");
        let out = dir.path().join("run");
        let common = [
            "ferx".to_string(),
            "iovsearch".to_string(),
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
        assert!(out.join("models/run1.ferx").is_file());
        let mut again = common.to_vec();
        again.push("--quiet".to_string());
        again.push("--resume".to_string());
        assert_eq!(run(&again), 0);
    }

    /// With no `[space]` the candidates are every parameter with a free η,
    /// which the banner says; a base that does not read its occasions is
    /// refused with the fix named.
    #[test]
    fn run_defaults_the_space_and_refuses_a_base_without_an_occasion_column() {
        let dir = tempfile::tempdir().unwrap();
        let config = write_config(dir.path(), "", "[iovsearch]\ndistribution = \"joint\"\n\n");
        assert_eq!(
            run(&args(&[
                "ferx",
                "iovsearch",
                &config.to_string_lossy(),
                "--directory",
                &dir.path().join("run").to_string_lossy(),
                "--quiet"
            ])),
            0
        );

        let no_column = dir.path().join("no-occ");
        std::fs::create_dir(&no_column).unwrap();
        let config = write_config(&no_column, "", "");
        std::fs::write(
            no_column.join("base.ferx"),
            BASE.replace("  iov_column = OCC\n", ""),
        )
        .unwrap();
        assert_eq!(
            run(&args(&["ferx", "iovsearch", &config.to_string_lossy()])),
            1
        );
    }

    #[test]
    fn run_rejects_two_files_and_a_missing_file_and_prints_help() {
        assert_eq!(
            run(&args(&[
                "ferx",
                "iovsearch",
                "a.ferxsearch",
                "b.ferxsearch"
            ])),
            1
        );
        assert_eq!(run(&args(&["ferx", "iovsearch", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "iovsearch", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "iovsearch"])), 1);
        assert_eq!(
            run(&args(&[
                "ferx",
                "iovsearch",
                "x.ferxsearch",
                "--samples",
                "3"
            ])),
            1
        );
    }
}
