//! `ferx amd` — the automatic model development pipeline (#1184).
//!
//! Thin by construction (#1114 A5): parse flags, load the `.ferxsearch` file,
//! call [`ferx_tools::amd::run_amd`], print the report, pick an exit code.
//! Every decision about the order, the skips, the seeding and the retries
//! policy lives in `ferx-tools`.

use std::path::PathBuf;

use ferx_tools::amd::{
    default_dir, render_summary, run_amd, AmdEvent, AmdOptions, AmdRun, Context,
};
use ferx_tools::search::SearchConfig;

use crate::covsearch_cmd::{flag, parse_threads, scan_args, value};

pub const AMD_USAGE: &str = "\
Usage: ferx amd <search.ferxsearch> [options]

Automatic model development — Pharmpy amd. One .ferxsearch file drives the
whole pipeline: the base model and data, one [space] mfl holding every step's
statements, the [rank] criterion and [strictness] gate every step inherits,
and an [amd] section:

  strategy = \"default\"      # or \"reevaluation\", \"SIR\", \"SRI\", \"RSI\"
  retries  = \"all_final\"    # or \"final\", \"skip\"
  skip     = []             # any of \"structural\", \"iivsearch\", \"residual\",
                            # \"iovsearch\", \"allometry\", \"covariates\"

The default order is structural -> IIV -> residual -> IOV -> allometry ->
covariates. Each step is handed the statements its own tool accepts — the
structural search gets ABSORPTION / PERIPHERALS / TRANSITS / LAGTIME, the
covariate search COVARIATE, and so on — and the model the previous step
selected, seeded from that step's estimates. A step the space says nothing
about is skipped with its reason in the report; the residual search needs no
space, and the IOV search needs the base model's `iov_column`.

Tool-specific sections ([modelsearch], [iivsearch], [iovsearch], [ruvsearch],
[covsearch], [allometry]) are read by their own step, so a pipeline is
configured exactly as the tools are on their own.

  --directory DIR      where the per-step directories, steps.csv,
                       candidates.csv, final.ferx and final-fit.yaml go
                       (default {search}-amd, next to the .ferxsearch file;
                       [run] cache_dir when set)
  --threads N          total worker threads (overrides [run] threads)
  --resume             reuse the fits already journalled in --directory
  --quiet              do not print step progress to stderr

  -h, --help           print this help and exit
";

const VALUE_FLAGS: &[&str] = &["--directory", "--threads"];
const BOOL_FLAGS: &[&str] = &["--resume", "--quiet", "-h", "--help"];

/// Entry point for `ferx amd ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{AMD_USAGE}");
        return 0;
    }
    match run_amd_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

fn run_amd_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let config_path = match positionals.as_slice() {
        [one] => PathBuf::from(one),
        [] => {
            eprint!("{AMD_USAGE}");
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
    // Refuse a file the pipeline cannot honour before the dataset is read.
    let options = AmdOptions::from_config(&config)?;
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
        "Strategy:   {}, retries {}",
        options.strategy.label(),
        options.retries.label()
    );
    eprintln!("Directory:  {}", dir.display());

    // The plan before the first fit, so a user sees what will and will not run
    // rather than discovering a skipped step in the report afterwards.
    let plan = ferx_tools::amd::plan(&options, &config.mfl, &Context::from_base(&base))?;
    for step in &plan {
        match &step.skipped {
            Some(reason) => eprintln!(
                "  {} {:<12} skipped: {reason}",
                step.index,
                step.step.label()
            ),
            None => eprintln!("  {} {:<12} -> {}", step.index, step.step.label(), step.dir),
        }
    }

    let progress = |event: AmdEvent| {
        if quiet {
            return;
        }
        match event {
            AmdEvent::Planned { .. } => {}
            AmdEvent::StartStarted => eprintln!("Fitting the starting model..."),
            AmdEvent::StartFinished { ofv } => eprintln!(
                "Starting model: OFV {}",
                ofv.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into())
            ),
            AmdEvent::StepStarted {
                position,
                total,
                step,
                rerun,
                ..
            } => eprintln!(
                "Step {position}/{total}: {}{} ({})...",
                step.label(),
                if rerun { " (rerun)" } else { "" },
                step.tool()
            ),
            AmdEvent::StepSkipped { step, reason, .. } => {
                eprintln!("Skipping {}: {reason}", step.label())
            }
            AmdEvent::StepFailed { step, error, .. } => eprintln!(
                "Step {} failed: {error}; carrying on from the model it was handed",
                step.label()
            ),
            AmdEvent::StepFinished {
                step,
                criterion,
                before,
                after,
                selected,
                ..
            } => eprintln!(
                "  {} done: {criterion} {} -> {}; selected {}",
                step.label(),
                before
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".into()),
                after
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "-".into()),
                if selected.is_empty() {
                    "nothing".to_string()
                } else {
                    selected.join("; ")
                }
            ),
            AmdEvent::RetriesStarted { starts, .. } => {
                eprintln!("  retries: refitting the selected model with {starts} starts...")
            }
            AmdEvent::RetriesFinished { improved, ofv } => eprintln!(
                "  retries: {} (OFV {})",
                if improved { "improved" } else { "kept" },
                ofv.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into())
            ),
        }
    };

    let result = run_amd(
        &config,
        &base,
        AmdRun {
            dir: dir.clone(),
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
        "Step table written to {}; every candidate to {}; final model to {}",
        ferx_tools::amd::steps_path(&dir).display(),
        ferx_tools::amd::candidates_path(&dir).display(),
        ferx_tools::amd::final_model_path(&dir).display()
    );
    // A step that failed leaves a usable model and a complete report, so the
    // run is not thrown away — but it did not do what was asked, and a
    // scripted caller has to be able to tell.
    let failed: Vec<&str> = result.failures().map(|s| s.step.label()).collect();
    if !failed.is_empty() {
        eprintln!("Steps that failed: {}", failed.join(", "));
    }
    Ok(if result.cancelled {
        130
    } else if failed.is_empty() {
        0
    } else {
        1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    /// The warfarin model, evaluating rather than fitting (`maxiter = 0`), so
    /// the pipeline runs end to end in a unit test.
    const BASE: &str = "\
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV(40.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  omega ETA_CL ~ 0.15
  omega ETA_V ~ 0.15
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[covariates]
  WT continuous

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
";

    /// A model, a dataset, a space and an `[amd]` section, written out as a
    /// `.ferxsearch` file the command can be pointed at.
    fn write(
        dir: &std::path::Path,
        base: &str,
        data: &str,
        space: &str,
        amd: &str,
    ) -> std::path::PathBuf {
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(data);
        std::fs::write(dir.join("base.ferx"), base).unwrap();
        let config = dir.join("search.ferxsearch");
        let space = if space.is_empty() {
            String::new()
        } else {
            format!("[space]\nmfl = \"{space}\"\n\n")
        };
        std::fs::write(
            &config,
            format!(
                "base = \"base.ferx\"\ndata = \"{}\"\n\n{space}{amd}\n\
                 [strictness]\nrequire_converged = false\nreject_init_stall = \
                 false\nreject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 2\n",
                data.display()
            ),
        )
        .unwrap();
        config
    }

    fn write_config(dir: &std::path::Path, amd: &str) -> std::path::PathBuf {
        write(
            dir,
            BASE,
            "two_cpt_oral_cov.csv",
            "LAGTIME([OFF,ON]);IIV?([V],EXP);COVARIATE?(CL,WT,pow)",
            amd,
        )
    }

    /// The same model with an occasion column, for the IOV step.
    fn iov_base() -> String {
        BASE.replace("[fit_options]", "[fit_options]\n  iov_column = OCC")
            .replace("[covariates]\n  WT continuous\n\n", "")
    }

    /// Run the command, returning its exit code.
    fn drive(config: &std::path::Path, out: &std::path::Path) -> i32 {
        run(&args(&[
            "ferx",
            "amd",
            &config.to_string_lossy(),
            "--directory",
            &out.to_string_lossy(),
            "--threads",
            "2",
            "--quiet",
        ]))
    }

    /// The command end to end: three steps run, the IOV and allometry steps
    /// are skipped for cause, the two tables and the final model are written,
    /// and a second `--quiet --resume` run reuses the journals.
    #[test]
    fn run_drives_the_pipeline_writes_the_report_and_resumes_quietly() {
        let dir = tempfile::tempdir().unwrap();
        let config = write_config(
            dir.path(),
            "[amd]\nstrategy = \"SIR\"\nretries = \"skip\"\n",
        );
        let out = dir.path().join("run");
        let common = [
            "ferx".to_string(),
            "amd".to_string(),
            config.to_string_lossy().into_owned(),
            "--directory".to_string(),
            out.to_string_lossy().into_owned(),
            "--threads".to_string(),
            "2".to_string(),
        ];
        assert_eq!(run(&common), 0);
        for file in [
            "steps.csv",
            "candidates.csv",
            "final.ferx",
            "final-fit.yaml",
        ] {
            assert!(out.join(file).is_file(), "{file} missing");
        }
        for step in ["00-start", "01-modelsearch", "02-iivsearch", "03-ruvsearch"] {
            assert!(
                out.join(step).join("input.ferx").is_file(),
                "{step}/input.ferx missing"
            );
        }
        let steps = std::fs::read_to_string(out.join("steps.csv")).unwrap();
        assert_eq!(steps.lines().count(), 4, "one header, three steps");
        let candidates = std::fs::read_to_string(out.join("candidates.csv")).unwrap();
        assert!(
            candidates.lines().count() > 4,
            "every candidate of every step is in the table:\n{candidates}"
        );

        let mut again = common.to_vec();
        again.push("--quiet".to_string());
        again.push("--resume".to_string());
        assert_eq!(run(&again), 0);
    }

    /// The default strategy skips the steps the file cannot support and says
    /// why, rather than failing or running them on nothing.
    #[test]
    fn the_default_strategy_skips_iov_and_allometry_for_cause() {
        let dir = tempfile::tempdir().unwrap();
        let config = write_config(
            dir.path(),
            "[amd]\nretries = \"skip\"\nskip = [\"structural\"]\n",
        );
        let out = dir.path().join("run");
        assert_eq!(drive(&config, &out), 0);
        let steps = std::fs::read_to_string(out.join("steps.csv")).unwrap();
        assert!(steps.contains("iov_column"), "{steps}");
        assert!(steps.contains("no ALLOMETRY statement"), "{steps}");
        assert!(steps.contains("[amd] skip"), "{steps}");
        assert!(!out.join("01-modelsearch").exists(), "a skipped step ran");
    }

    /// The allometry step, which is the one step that is not a ranked search:
    /// the scaled model is adopted when it passes the gate, and both fits are
    /// in the candidate table either way.
    #[test]
    fn the_allometry_step_scales_the_model_and_reports_both_fits() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            BASE,
            "two_cpt_oral_cov.csv",
            "ALLOMETRY(WT,70)",
            "[amd]\nretries = \"skip\"\nskip = [\"structural\", \"iivsearch\", \"residual\", \
             \"covariates\"]\n",
        );
        let out = dir.path().join("run");
        assert_eq!(drive(&config, &out), 0);
        let candidates = std::fs::read_to_string(out.join("candidates.csv")).unwrap();
        assert!(candidates.contains("allometry,base,"), "{candidates}");
        assert!(candidates.contains("allometry,allometric,"), "{candidates}");
        let steps = std::fs::read_to_string(out.join("steps.csv")).unwrap();
        let row = steps
            .lines()
            .find(|l| l.starts_with("allometry,"))
            .expect("the allometry step ran");
        assert!(row.contains(",ran,"), "{row}");
        assert!(
            std::fs::read_to_string(out.join("final.ferx"))
                .unwrap()
                .contains("WT"),
            "the scaled model was not carried forward"
        );
    }

    /// The IOV step, which needs no space at all: with `iov_column` set it
    /// searches every parameter that has a free η.
    #[test]
    fn the_iov_step_runs_without_a_space_when_the_model_reads_occasions() {
        let dir = tempfile::tempdir().unwrap();
        let config = write(
            dir.path(),
            &iov_base(),
            "warfarin_iov.csv",
            "",
            "[amd]\nretries = \"skip\"\nskip = [\"residual\"]\n",
        );
        let out = dir.path().join("run");
        assert_eq!(drive(&config, &out), 0);
        let steps = std::fs::read_to_string(out.join("steps.csv")).unwrap();
        let row = steps
            .lines()
            .find(|l| l.starts_with("iovsearch,"))
            .expect("the IOV step ran");
        assert!(row.contains(",ran,"), "{row}");
        // Every other step is skipped: four because the file has no [space]
        // at all, and the residual step because `[amd] skip` names it.
        assert_eq!(steps.matches(",skipped,").count(), 5, "{steps}");
        assert!(out.join("04-iovsearch/models.csv").is_file());
    }

    #[test]
    fn run_rejects_two_files_and_a_missing_file_and_prints_help() {
        assert_eq!(
            run(&args(&["ferx", "amd", "a.ferxsearch", "b.ferxsearch"])),
            1
        );
        assert_eq!(run(&args(&["ferx", "amd", "nope.ferxsearch"])), 1);
        assert_eq!(run(&args(&["ferx", "amd", "-h"])), 0);
        assert_eq!(run(&args(&["ferx", "amd"])), 1);
        assert_eq!(
            run(&args(&["ferx", "amd", "x.ferxsearch", "--samples", "3"])),
            1
        );
    }
}
