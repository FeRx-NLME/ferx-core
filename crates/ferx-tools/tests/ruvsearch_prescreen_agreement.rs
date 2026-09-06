//! Tier-3: the CWRES pre-screen and the full refit select the same feature on
//! the fixture set (#1182, the issue's third validation bullet) — up to the
//! `power`/`combined` pair Pharmpy treats as one candidate.
//!
//! Two runs of `ferx ruvsearch` on the Pharmpy anchor dataset
//! (`tests/pharmpy/ruvsearch_anchor/`, simulated with a power residual), fitted
//! to convergence: one with every candidate refitted on the data, one with the
//! CWRES pre-screen picking a single candidate to refit. The two paths judge
//! different objects — real fits by likelihood ratio versus residual-model fits
//! by the χ² cutoff — so agreement is a property of the data, not a tautology.
//! On this set the full refit prefers `power` (401.69) and the screen picks
//! `combined` (402.89) by a 1-unit CWRES margin; Pharmpy never tests one
//! after the other, and neither does ferx, so the pair is one decision.
//!
//! Slow-gated: about a dozen FOCEI fits on 440 observations. Run nightly.

use std::path::{Path, PathBuf};

use ferx_tools::ruvsearch::{run_ruvsearch, Family, RuvFeature, RuvsearchRun};
use ferx_tools::search::SearchConfig;

const ANCHOR: &str = "tests/pharmpy/ruvsearch_anchor";

fn write_config(dir: &Path, prescreen: bool) -> PathBuf {
    let anchor = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR);
    let config = format!(
        "base = \"{}\"\ndata = \"{}\"\n\n[ruvsearch]\nmax_iter = 3\n\
         cwres_prescreen = {prescreen}\n\n[strictness]\nrequire_converged = false\n\
         reject_init_stall = false\nreject_on_boundary = false\n\n[run]\nretries = 0\n\
         threads = 4\n",
        anchor.join("base.ferx").display(),
        anchor.join("power_sim.csv").display()
    );
    let path = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    path
}

/// The accepted features, with `power` and `combined` folded into one.
fn pair_key(f: RuvFeature) -> String {
    match f.family() {
        Family::Power | Family::Combined => "power|combined".into(),
        _ => f.label(),
    }
}

fn accepted(prescreen: bool) -> (Vec<String>, f64, Vec<String>) {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(dir.path(), prescreen);
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_ruvsearch(
        &config,
        &base,
        RuvsearchRun {
            dir: Some(dir.path().join("run")),
            ..RuvsearchRun::default()
        },
    )
    .expect("search");
    (
        result.features.iter().map(|f| pair_key(*f)).collect(),
        result.final_ofv,
        result.notes.clone(),
    )
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_prescreen_and_the_full_refit_select_the_same_feature() {
    let (full, full_ofv, full_notes) = accepted(false);
    let (screened, screened_ofv, screened_notes) = accepted(true);
    eprintln!("full refit: {full:?} at {full_ofv:.3} — {full_notes:?}");
    eprintln!("pre-screen: {screened:?} at {screened_ofv:.3} — {screened_notes:?}");
    assert_eq!(full, vec!["power|combined".to_string()]);
    assert_eq!(
        full, screened,
        "the two paths must agree on the accepted features, up to the power/combined pair"
    );
    // Near-equivalent in shape: the two members of the pair land within a
    // couple of OFV units of each other.
    assert!(
        (full_ofv - screened_ofv).abs() < 2.0,
        "{full_ofv} vs {screened_ofv}"
    );
}
