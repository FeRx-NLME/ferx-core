//! Tier-3: Pharmpy 2.2.0 `ruvsearch` (driven by NONMEM 7.5.1) as the
//! trajectory anchor for `ferx ruvsearch` (#1182).
//!
//! The reference run and its inputs live in `tests/pharmpy/ruvsearch_anchor/`
//! (see its README): a 40-subject oral dataset simulated with a *power*
//! residual error, a proportional-error input model, and Pharmpy's
//! `cwres_models` / `summary_tool` / final model in `pharmpy_ruvsearch.json`.
//!
//! What is anchored, per path:
//!
//! * **CWRES pre-screen** (`cwres_prescreen = true`, Pharmpy's own path):
//!   every screened candidate's CWRES dOFV within 1.0 of Pharmpy's, the same
//!   pick (`combined`), the refit landing on Pharmpy's refit to 0.01, the
//!   second iteration accepting nothing, and the same final model. The screen
//!   is fitted to *ferx's* CWRES, so this is also the end-to-end check that
//!   ferx's `CWRES` is NONMEM's (`compute_cwres`).
//! * **Full refit**: ferx's `combined` candidate lands on Pharmpy's refit to
//!   0.01 — the same real model, two engines — and the selected feature is one
//!   of the `power`/`combined` pair Pharmpy treats as a unit (ferx's full
//!   refit prefers `power`, the data's generating form, by 1.2 OFV).
//!
//! Slow-gated: the pre-screen path is cheap but the full refit is six FOCEI
//! fits on 440 observations.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ferx_tools::ruvsearch::{run_ruvsearch, RuvFeature, RuvsearchResult, RuvsearchRun};
use ferx_tools::search::SearchConfig;

const ANCHOR: &str = "tests/pharmpy/ruvsearch_anchor";

#[derive(serde::Deserialize)]
struct CwresRow {
    model: String,
    iteration: usize,
    dofv: f64,
}

#[derive(serde::Deserialize)]
struct Variant {
    base_ofv: f64,
    cwres_models: Vec<CwresRow>,
    final_ofv: f64,
    final_model_code: String,
}

fn pharmpy(variant: &str) -> Variant {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(ANCHOR)
        .join("pharmpy_ruvsearch.json");
    let text = std::fs::read_to_string(&path).expect("anchor json");
    let mut all: HashMap<String, Variant> = serde_json::from_str(&text).expect("anchor json");
    all.remove(variant).expect("variant")
}

fn run(dir: &Path, prescreen: bool) -> RuvsearchResult {
    let anchor = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR);
    let config = format!(
        "base = \"{}\"\ndata = \"{}\"\n\n[ruvsearch]\ngroups = 4\np_value = 0.001\n\
         max_iter = 3\ncwres_prescreen = {prescreen}\n\n[strictness]\n\
         require_converged = false\nreject_init_stall = false\nreject_on_boundary = false\n\n\
         [run]\nretries = 0\nthreads = 4\n",
        anchor.join("base.ferx").display(),
        anchor.join("power_sim.csv").display()
    );
    let path: PathBuf = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    run_ruvsearch(
        &config,
        &base,
        RuvsearchRun {
            dir: Some(dir.join("run")),
            ..RuvsearchRun::default()
        },
    )
    .expect("search")
}

fn label(f: RuvFeature) -> String {
    f.label()
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_cwres_prescreen_follows_pharmpys_trajectory() {
    let reference = pharmpy("all");
    assert!(reference.final_model_code.contains("EPS(1)*IPRED + EPS(2)"));
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), true);
    assert!(
        (result.input_ofv - reference.base_ofv).abs() < 0.01,
        "input OFV {} vs Pharmpy {}",
        result.input_ofv,
        reference.base_ofv
    );
    // Iteration 1: every screened dOFV within 1.0 of Pharmpy's.
    for row in reference.cwres_models.iter().filter(|r| r.iteration == 1) {
        let ours = result
            .rows
            .iter()
            .find(|r| {
                r.screened && r.iteration == 1 && r.feature.map(label) == Some(row.model.clone())
            })
            .unwrap_or_else(|| panic!("no screened row for {}", row.model));
        let d = ours.cwres_dofv.expect("screened dOFV");
        eprintln!(
            "screen {:14} pharmpy {:8.3} ferx {:8.3}",
            row.model, row.dofv, d
        );
        assert!(
            (d - row.dofv).abs() < 1.0,
            "{}: ferx dOFV {d} vs Pharmpy {}",
            row.model,
            row.dofv
        );
    }
    // The same pick, refitted to the same objective, accepted.
    let refit = result
        .rows
        .iter()
        .find(|r| !r.screened && r.iteration == 1)
        .expect("a refit");
    assert_eq!(refit.feature, Some(RuvFeature::Combined));
    assert!(refit.selected);
    assert!(
        (refit.ofv.unwrap() - reference.final_ofv).abs() < 0.01,
        "refit {} vs Pharmpy {}",
        refit.ofv.unwrap(),
        reference.final_ofv
    );
    // Iteration 2: nothing beats the cutoff, as in Pharmpy.
    for row in reference.cwres_models.iter().filter(|r| r.iteration == 2) {
        let ours = result
            .rows
            .iter()
            .find(|r| {
                r.screened && r.iteration == 2 && r.feature.map(label) == Some(row.model.clone())
            })
            .unwrap_or_else(|| panic!("no screened row for {} in iteration 2", row.model));
        let d = ours.cwres_dofv.expect("screened dOFV");
        assert!(
            (d - row.dofv).abs() < 1.0,
            "{}: {d} vs {}",
            row.model,
            row.dofv
        );
        assert!(d < result.options.cutoff());
    }
    assert!(result
        .rows
        .iter()
        .all(|r| !(r.iteration == 2 && !r.screened)));
    assert_eq!(result.features, vec![RuvFeature::Combined]);
    assert!((result.final_ofv - reference.final_ofv).abs() < 0.01);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_full_refit_lands_on_pharmpys_refit_and_selects_within_the_pair() {
    let reference = pharmpy("all");
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), false);
    let combined = result
        .rows
        .iter()
        .find(|r| r.iteration == 1 && r.feature == Some(RuvFeature::Combined))
        .expect("combined candidate");
    assert!(
        (combined.ofv.unwrap() - reference.final_ofv).abs() < 0.01,
        "ferx combined {} vs Pharmpy {}",
        combined.ofv.unwrap(),
        reference.final_ofv
    );
    let selected: Vec<RuvFeature> = result
        .rows
        .iter()
        .filter(|r| r.selected && r.iteration == 1)
        .map(|r| r.feature.unwrap())
        .collect();
    assert!(
        matches!(
            selected.as_slice(),
            [RuvFeature::Power] | [RuvFeature::Combined]
        ),
        "{selected:?}"
    );
    assert_eq!(result.features.len(), 1);
    // Whichever of the pair wins, the search ends on it.
    assert!(result
        .rows
        .iter()
        .filter(|r| r.iteration == 2)
        .all(|r| !r.selected));
}
