//! `ferx allometry` — allometric scaling (#1180).
//!
//! Thin by construction: parse flags, build the options, call
//! [`ferx_tools::allometry::run_allometry`], print both fits. The convention
//! — which parameters get which exponent — lives in `ferx-tools`.

use std::path::{Path, PathBuf};

use ferx_core::edit::ModelText;
use ferx_tools::allometry::{run_allometry, AllometryOptions, AllometryRun};
use ferx_tools::search::{BaseModel, SearchConfig};

use crate::covsearch_cmd::{parse_threads, scan_args};

pub const ALLOMETRY_USAGE: &str = "\
Usage: ferx allometry <model.ferx> --data <data.csv> [options]
       ferx allometry <search.ferxsearch> [options]

Adds allometric scaling to the base model — `(WT/70)^0.75` on every
clearance the `pk` line binds (cl, q, q2, q3) and `(WT/70)^1.0` on every
volume (v, v1, v2, v3) — as `[covariate_model]` lines, then fits the base
and the scaled model side by side and reports both. This is Pharmpy's
`allometry` tool; a .ferxsearch file states the same thing as
`ALLOMETRY(WT, 70)` in its space plus an optional [allometry] section.

  --covariate COV      the size covariate (default WT)
  --reference X        the reference value it is divided by (default 70)
  --parameters A,B     scale these parameters instead of the template's
                       clearances and volumes
  --exponents x,y      one exponent per --parameters entry (default 0.75 for
                       a clearance, 1.0 for a volume)
  --estimate           estimate the exponents from those values instead of
                       fixing them, bounded by --lower / --upper (0, 2)
  --lower X, --upper X the bounds of an estimated exponent
  --data FILE          the dataset (model-file form; a .ferxsearch names its own)
  --directory DIR      where the two fits are journalled (default: in memory)
  --threads N          total worker threads
  --retries N          perturbed restarts per fit on top of the exact one
                       (default 2)

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

fn number(args: &[String], name: &str) -> Result<Option<f64>, String> {
    match value(args, name)? {
        None => Ok(None),
        Some(raw) => raw
            .parse::<f64>()
            .map(Some)
            .map_err(|_| format!("{name} requires a number; got '{raw}'")),
    }
}

const VALUE_FLAGS: &[&str] = &[
    "--covariate",
    "--reference",
    "--parameters",
    "--exponents",
    "--lower",
    "--upper",
    "--data",
    "--directory",
    "--threads",
    "--retries",
];
const BOOL_FLAGS: &[&str] = &["--estimate", "-h", "--help"];

/// Entry point for `ferx allometry ...`; returns the process exit code.
pub fn run(args: &[String]) -> i32 {
    if args[2..].iter().any(|a| a == "-h" || a == "--help") {
        print!("{ALLOMETRY_USAGE}");
        return 0;
    }
    match run_allometry_command(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            1
        }
    }
}

/// Command-line overrides applied on top of the defaults (or the file's
/// section).
fn apply_flags(args: &[String], options: &mut AllometryOptions) -> Result<(), String> {
    if let Some(c) = value(args, "--covariate")? {
        options.covariate = c.to_string();
    }
    if let Some(r) = number(args, "--reference")? {
        options.reference = r;
    }
    if let Some(p) = value(args, "--parameters")? {
        options.parameters = Some(p.split(',').map(|s| s.trim().to_string()).collect());
    }
    if let Some(e) = value(args, "--exponents")? {
        let parsed: Result<Vec<f64>, _> = e.split(',').map(|s| s.trim().parse::<f64>()).collect();
        options.exponents =
            Some(parsed.map_err(|_| format!("--exponents requires numbers; got '{e}'"))?);
    }
    if flag(args, "--estimate") {
        options.fixed = false;
    }
    if let Some(l) = number(args, "--lower")? {
        options.lower = l;
    }
    if let Some(u) = number(args, "--upper")? {
        options.upper = u;
    }
    options.validate()
}

fn run_allometry_command(args: &[String]) -> Result<i32, String> {
    let positionals = scan_args(args, VALUE_FLAGS, BOOL_FLAGS)?;
    let path = match positionals.as_slice() {
        [one] => Path::new(one),
        [] => {
            eprint!("{ALLOMETRY_USAGE}");
            return Ok(1);
        }
        many => {
            return Err(format!(
                "expected one model or .ferxsearch file, got {}: {}",
                many.len(),
                many.join(", ")
            ))
        }
    };
    let threads = parse_threads(args)?;
    let retries = match value(args, "--retries")? {
        None => None,
        Some(raw) => Some(
            raw.parse::<usize>()
                .map_err(|_| format!("--retries requires a non-negative integer; got '{raw}'"))?,
        ),
    };

    let is_search = path
        .extension()
        .is_some_and(|e| e == ferx_tools::search::config::EXTENSION);
    let (base, mut options, mut run_options) = if is_search {
        if value(args, "--data")?.is_some() {
            return Err("--data does not apply to a .ferxsearch file, which names its own".into());
        }
        let config = SearchConfig::load(path)?;
        let options = AllometryOptions::from_config(&config)?;
        (config.load_base()?, options, config.run_options())
    } else {
        let data = value(args, "--data")?;
        let prepared = ferx_core::prepare_run(&path.to_string_lossy(), data)?;
        if let Some(w) = &prepared.data_path_warning {
            eprintln!("Warning: {w}");
        }
        let text = ModelText::parse(
            &std::fs::read_to_string(path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?,
        )?;
        (
            BaseModel { prepared, text },
            AllometryOptions::default(),
            ferx_tools::search::RunOptions::default(),
        )
    };
    apply_flags(args, &mut options)?;
    if let Some(r) = retries {
        run_options.n_starts = r + 1;
    }

    eprintln!(
        "Data: {} subjects, {} observations",
        base.prepared.population.subjects.len(),
        base.prepared.population.n_obs()
    );
    let result = run_allometry(
        &base,
        &options,
        AllometryRun {
            dir: value(args, "--directory")?.map(PathBuf::from),
            threads,
            cancel: None,
            run_options,
        },
    )?;

    println!(
        "Allometric scaling on {} (reference {}):",
        options.covariate, options.reference
    );
    for s in &result.scalings {
        println!(
            "  {} ~ {} power(center = {}, {})",
            s.parameter,
            options.covariate,
            options.reference,
            if s.fixed {
                format!("fix = {}", s.exponent)
            } else {
                format!("init = {}, estimated", s.exponent)
            }
        );
    }
    let show = |label: &str, r: &ferx_tools::search::CandidateResult| {
        println!(
            "  {label:<10} OFV {:>12}  converged={}  passed={}{}",
            r.ofv
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "-".into()),
            r.converged
                .map(|c| c.to_string())
                .unwrap_or_else(|| "-".into()),
            r.verdict.passed,
            if r.verdict.failures.is_empty() {
                String::new()
            } else {
                format!("  ({})", r.verdict.failures.join("; "))
            }
        );
    };
    println!();
    show("base", &result.base);
    show("allometric", &result.scaled);
    if let Some(d) = result.dofv() {
        println!("  dOFV (base - allometric): {d:.3}");
    }
    if let Some(fit) = &result.scaled.fit {
        let estimated: Vec<String> = fit
            .covariate_relations
            .iter()
            .filter(|r| r.covariate == options.covariate)
            .flat_map(|r| r.thetas.iter().filter(|t| !t.fixed))
            .map(|t| {
                format!(
                    "{} = {:.4}{}",
                    t.name,
                    t.estimate,
                    t.se.map(|se| format!(" (SE {se:.4})")).unwrap_or_default()
                )
            })
            .collect();
        if !estimated.is_empty() {
            println!("  estimated exponents: {}", estimated.join(", "));
        }
    }
    for n in &result.notes {
        println!("  note: {n}");
    }
    let out = path.with_file_name(format!(
        "{}-allometric.ferx",
        path.file_stem().and_then(|s| s.to_str()).unwrap_or("model")
    ));
    let mut model = result.model.clone();
    if let Some(fit) = &result.scaled.fit {
        model.apply(ferx_core::edit::ModelEdit::SeedInits(fit))?;
    }
    std::fs::write(&out, model.render())
        .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
    eprintln!("Allometric model written to {}", out.display());
    Ok(if result.cancelled { 130 } else { 0 })
}
