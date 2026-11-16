//! Sweep: no committed model falls back to finite differences (#2786).
//!
//! The compiled per-subject plans are the only derivative path; a subject whose plan cannot
//! be recorded falls back to finite differences and the fit says so (`W_PLAN_FD_FALLBACK`).
//! That must be a defect of the recorder, never a property of a model the parser accepts.
//! Reading the code cannot show that for every model, and the example sweep that existed
//! before this one missed three real classes (#2949: `two_cpt_eigen_g`, the steady-state
//! pulse-train stop test, the overlapping-SS pulse count). So this runs every committed
//! model that has data through one short fit and asserts the fallback warning is absent.
//!
//! Coverage:
//! - every `examples/*.ferx` with a `[simulation]` block, fitted on its own simulated data;
//! - every other `examples/*.ferx` that has a `data/<name>.csv`;
//! - every `nonmem_anchor/**/*.ferx`, paired with its dataset by the control stream's
//!   `$DATA` (`<base>.ctl`, `base` being the stem without `_fit` / `_ode` / `_ode_fit`), or
//!   by a same-stem `.csv`.
//!
//! A model that cannot be read in this build (a feature not compiled in, a covariate a
//! simulation design does not state, no dataset) is skipped, and the sweep reports how many
//! it tried, ran, and skipped, and why. It asserts a floor on `ran`, so a sweep that quietly
//! stopped covering models is a red test.
//!
//! Tier 3 (it fits ~150 models): gated as slow-tests, run nightly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const TOKEN: &str = "W_PLAN_FD_FALLBACK";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn ferx_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "results") {
                continue;
            }
            ferx_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "ferx") {
            out.push(p);
        }
    }
}

/// The model text with a one-iteration fit: the first gradient is taken (so a plan is
/// recorded and run), nothing converges.
fn shortened(text: &str) -> String {
    let short = |key: &str, v: &str, l: &str| {
        let t = l.trim_start();
        (t.starts_with(key) && t[key.len()..].trim_start().starts_with('='))
            .then(|| format!("  {key} = {v}"))
    };
    let mut out: Vec<String> = text
        .lines()
        .map(|l| {
            short("maxiter", "1", l)
                .or_else(|| short("n_subjects", "6", l))
                .or_else(|| short("n_starts", "1", l))
                .or_else(|| short("imp_iterations", "2", l))
                .or_else(|| short("impmap_iterations", "2", l))
                .or_else(|| short("imp_samples", "50", l))
                .or_else(|| short("impmap_samples", "50", l))
                .or_else(|| short("conddist_keep_samples", "20", l))
                .or_else(|| short("conddist_burnin", "20", l))
                .or_else(|| short("covariance", "false", l))
                .unwrap_or_else(|| l.to_string())
        })
        .collect();
    if !out.iter().any(|l| l.trim_start().starts_with("maxiter")) {
        match out.iter().position(|l| l.trim() == "[fit_options]") {
            Some(i) => out.insert(i + 1, "  maxiter = 1".into()),
            None => out.extend(["[fit_options]".into(), "  maxiter = 1".into()]),
        }
    }
    out.join("\n")
}

/// The dataset a `nonmem_anchor` model is paired with.
fn anchor_data(model: &Path) -> Option<PathBuf> {
    let dir = model.parent()?;
    let stem = model.file_stem()?.to_str()?;
    let base = ["_ode_fit", "_fit", "_ode"]
        .iter()
        .find_map(|s| stem.strip_suffix(s))
        .unwrap_or(stem);
    for ctl in [
        dir.join(format!("{base}.ctl")),
        dir.join(format!("{stem}.ctl")),
    ] {
        if let Ok(t) = std::fs::read_to_string(&ctl) {
            let name = t.lines().find_map(|l| {
                let l = l.trim_start();
                l.strip_prefix("$DATA")
                    .and_then(|r| r.split_whitespace().next())
                    .map(|f| f.trim_matches(|c| c == '\'' || c == '"').to_string())
            });
            if let Some(n) = name {
                let file = Path::new(&n).file_name()?;
                let p = dir.join(file);
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }
    [base, stem]
        .iter()
        .map(|s| dir.join(format!("{s}.csv")))
        .find(|p| p.exists())
}

#[derive(Default)]
struct Tally {
    tried: usize,
    ran: usize,
    skipped: BTreeMap<String, Vec<String>>,
    fell_back: Vec<String>,
}

impl Tally {
    fn skip(&mut self, why: &str, model: &Path) {
        let why = why.lines().next().unwrap_or("").chars().take(70).collect();
        self.skipped
            .entry(why)
            .or_default()
            .push(model.display().to_string());
    }
}

fn run_one(
    tally: &std::sync::Mutex<Tally>,
    tmp: &Path,
    idx: usize,
    model: &Path,
    data: Option<PathBuf>,
    simulate: bool,
) {
    let text = std::fs::read_to_string(model).expect("read model");
    let patched = tmp.join(format!("m{idx}.ferx"));
    std::fs::write(&patched, shortened(&text)).expect("write");
    let overrides = ferx_core::RunOverrides {
        threads: Some(1),
        ..Default::default()
    };
    let path = patched.to_str().expect("utf8").to_string();
    let data_s = data.as_ref().map(|d| d.to_str().expect("utf8").to_string());
    let res = std::panic::catch_unwind(|| {
        if simulate {
            ferx_core::run_model_simulate_with_overrides(&path, &overrides)
        } else {
            ferx_core::run_model_with_overrides(&path, data_s.as_deref(), &overrides)
        }
    });
    let mut t = tally.lock().expect("tally");
    t.tried += 1;
    match res {
        Ok(Ok((fit, _))) => {
            t.ran += 1;
            let hits: Vec<_> = fit.warnings.iter().filter(|w| w.contains(TOKEN)).collect();
            if !hits.is_empty() {
                t.fell_back.push(format!("{}: {hits:?}", model.display()));
            }
        }
        Ok(Err(e)) => t.skip(&format!("{e}"), model),
        Err(_) => t.skip("panicked", model),
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: fits every committed model once; opt in with --features slow-tests"
)]
fn no_committed_model_falls_back_to_finite_differences() {
    let tmp = tempfile::tempdir().expect("tmp");
    let mut tally = Tally::default();

    let mut examples = Vec::new();
    ferx_files(&root().join("examples"), &mut examples);
    examples.sort();
    let mut anchors = Vec::new();
    ferx_files(&root().join("nonmem_anchor"), &mut anchors);
    anchors.sort();
    let (n_examples, n_anchors) = (examples.len(), anchors.len());

    let mut jobs: Vec<(PathBuf, Option<PathBuf>, bool)> = Vec::new();
    for m in &examples {
        let text = std::fs::read_to_string(m).expect("read");
        if text.lines().any(|l| l.trim() == "[simulation]") {
            jobs.push((m.clone(), None, true));
        } else if let Some(d) = example_data(m, &text) {
            jobs.push((m.clone(), Some(d), false));
        } else {
            tally.skip("no dataset and no [simulation] block", m);
        }
    }
    for m in &anchors {
        match anchor_data(m) {
            Some(d) => jobs.push((m.clone(), Some(d), false)),
            None => tally.skip("no paired dataset", m),
        }
    }
    // Fits are single-threaded (`threads = 1`); the models run side by side.
    let workers = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .min(8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let shared = std::sync::Mutex::new(tally);
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some((m, d, sim)) = jobs.get(i) else {
                    break;
                };
                run_one(&shared, tmp.path(), i, m, d.clone(), *sim);
            });
        }
    });
    let tally = shared.into_inner().expect("tally");

    eprintln!(
        "plan coverage sweep: {} models found ({} examples, {} nonmem_anchor), {} tried, {} ran, \
         {} skipped",
        n_examples + n_anchors,
        n_examples,
        n_anchors,
        tally.tried,
        tally.ran,
        tally.skipped.values().map(Vec::len).sum::<usize>()
    );
    for (why, models) in &tally.skipped {
        eprintln!("  skipped ({}): {why}", models.len());
        for m in models.iter().take(3) {
            eprintln!("    {m}");
        }
    }
    assert!(
        tally.fell_back.is_empty(),
        "models that fell back to finite differences:\n{}",
        tally.fell_back.join("\n")
    );
    // Measured floor (see the PR): a sweep that stopped covering models must not pass.
    assert!(
        tally.ran >= 100,
        "only {} of {} committed models ran",
        tally.ran,
        n_examples + n_anchors
    );
}

/// The dataset committed under `data/` with the example's name (`examples/warfarin.ferx` runs
/// on `data/warfarin.csv`, as the README and the CLI examples pair them), when there is one.
fn example_data(model: &Path, _text: &str) -> Option<PathBuf> {
    let stem = model.file_stem()?.to_str()?;
    let p = root().join("data").join(format!("{stem}.csv"));
    p.is_file().then_some(p)
}
