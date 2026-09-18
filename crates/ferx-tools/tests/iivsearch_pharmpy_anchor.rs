//! Tier-3: Pharmpy 2.2.0 `iivsearch` (driven by NONMEM 7.5.1) as the
//! trajectory anchor for `ferx iivsearch` (#1183).
//!
//! The reference runs and their inputs live in
//! `tests/pharmpy/iivsearch_anchor/` (see its README): a 40-subject oral
//! dataset simulated with IIV on CL and V only, correlated, and none on KA;
//! a diagonal three-η input model; and Pharmpy's `summary_tool` /
//! `summary_models` / final model for `top_down_exhaustive`,
//! `bottom_up_stepwise` and `simultaneous_stepwise` in
//! `pharmpy_iivsearch.json`.
//!
//! What is anchored, per algorithm: the candidates of every step in the
//! same numbering and with the same description; the BIC(iiv) of every
//! model both engines fitted to a proper optimum, to 0.05; the winner of
//! every step; the final model. One divergence is asserted by name: the
//! simultaneous algorithm's `[CL,V]+[KA]` candidate is a mixed ω, which
//! ferx's FOCEI fits as the full block (#1018) — better than NONMEM's failed
//! fit of the declared model — so ferx ends on it where Pharmpy ends on
//! `[CL,V]`. The assertion is on the divergence itself, so the fix of #1018
//! turns this test red and says what to flip.
//!
//! Slow-gated: about twenty FOCEI fits on 440 observations across the three
//! runs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ferx_tools::iivsearch::{run_iivsearch, IivsearchResult, IivsearchRun};
use ferx_tools::search::SearchConfig;

const ANCHOR: &str = "tests/pharmpy/iivsearch_anchor";

#[derive(serde::Deserialize)]
struct ToolRow {
    step: usize,
    model: String,
    description: String,
    bic: f64,
    rank: Option<u32>,
}

#[derive(serde::Deserialize)]
struct ModelRow {
    model: String,
    ofv: f64,
    minimization_successful: Option<bool>,
}

#[derive(serde::Deserialize)]
struct Variant {
    summary_tool: Vec<ToolRow>,
    summary_models: Vec<ModelRow>,
    final_description: String,
    final_ofv: f64,
}

#[derive(serde::Deserialize)]
struct Reference {
    base_ofv: f64,
    /// Not read; named so the flatten below does not try to parse it as a variant.
    #[allow(dead_code)]
    base_estimates: serde_json::Value,
    #[serde(flatten)]
    variants: HashMap<String, Variant>,
}

fn reference() -> Reference {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(ANCHOR)
        .join("pharmpy_iivsearch.json");
    let text = std::fs::read_to_string(&path).expect("anchor json");
    serde_json::from_str(&text).expect("anchor json")
}

/// Pharmpy's `iivsearch_run{n}` → ferx's `run{n}`; `input` and `base` as is.
fn ferx_id(pharmpy: &str) -> String {
    pharmpy.replace("iivsearch_", "")
}

fn run(dir: &Path, mfl: &str, algorithm: &str) -> IivsearchResult {
    let anchor = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR);
    let strictness = "[strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n";
    let config = format!(
        "base = \"{}\"\ndata = \"{}\"\n\n[space]\nmfl = \"{mfl}\"\n\n[iivsearch]\n\
         algorithm = \"{algorithm}\"\n\n{strictness}\n[run]\nretries = 0\nthreads = 4\n",
        anchor.join("base.ferx").display(),
        anchor.join("iiv_sim.csv").display()
    );
    let path: PathBuf = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    run_iivsearch(
        &config,
        &base,
        IivsearchRun {
            dir: Some(dir.join("run")),
            ..IivsearchRun::default()
        },
    )
    .expect("search")
}

/// Every Pharmpy candidate exists in ferx under the same id with the same
/// description — except the named `divergences`, `(ferx id, ferx
/// description)`, where the two sides are asserted to *differ* and ferx's
/// side to be as stated, so a stale entry fails — and where NONMEM's fit was
/// a proper optimum the BIC(iiv) agrees to `tol`.
fn assert_candidates(
    variant: &Variant,
    result: &IivsearchResult,
    tol: f64,
    divergences: &[(&str, &str)],
) {
    let pharmpy_ofv: HashMap<&str, &ModelRow> = variant
        .summary_models
        .iter()
        .map(|m| (m.model.as_str(), m))
        .collect();
    for row in &variant.summary_tool {
        let id = ferx_id(&row.model);
        let ours = result
            .row(&id)
            .unwrap_or_else(|| panic!("no ferx row for {} ({})", row.model, row.description));
        match divergences.iter().find(|(d, _)| *d == id) {
            Some((_, expected)) => {
                assert_eq!(ours.structure.description(), *expected, "{id}");
                assert_ne!(
                    row.description, *expected,
                    "{id}: the named divergence is gone; drop it from the list"
                );
                continue;
            }
            None => assert_eq!(
                ours.structure.description(),
                row.description,
                "{}: ferx {} vs Pharmpy {}",
                id,
                ours.structure.description(),
                row.description
            ),
        }
        let proper = pharmpy_ofv
            .get(row.model.as_str())
            .and_then(|m| m.minimization_successful)
            .unwrap_or(false)
            || row.rank == Some(1);
        eprintln!(
            "step {} {:8} {:16} pharmpy {:10.3} ferx {:10.3}{}",
            row.step,
            id,
            row.description,
            row.bic,
            ours.criterion,
            if proper {
                ""
            } else {
                "  (NONMEM: not a proper optimum)"
            }
        );
        if proper && !ours.structure.is_partial_block() {
            assert!(
                (ours.criterion - row.bic).abs() < tol,
                "{}: BIC(iiv) {} vs Pharmpy {}",
                id,
                ours.criterion,
                row.bic
            );
        }
    }
}

/// The winner of each Pharmpy step (its `rank == 1` row) is ferx's step
/// winner too.
fn assert_step_winners(variant: &Variant, result: &IivsearchResult) {
    for row in variant.summary_tool.iter().filter(|r| r.rank == Some(1)) {
        let step = result
            .steps
            .iter()
            .find(|s| s.step == row.step)
            .unwrap_or_else(|| panic!("ferx has no step {}", row.step));
        assert_eq!(
            step.best,
            ferx_id(&row.model),
            "step {}: ferx picked {} ({}), Pharmpy {} ({})",
            row.step,
            step.best,
            result
                .row(&step.best)
                .map(|r| r.structure.description())
                .unwrap_or_default(),
            row.model,
            row.description
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn top_down_exhaustive_follows_pharmpys_trajectory() {
    let reference = reference();
    let variant = &reference.variants["td"];
    let dir = tempfile::tempdir().unwrap();
    let result = run(
        dir.path(),
        "IIV?(@IIV,EXP);COVARIANCE?(IIV,@IIV)",
        "top_down_exhaustive",
    );
    let input = result.row("input").unwrap();
    assert!(
        (input.ofv.unwrap() - reference.base_ofv).abs() < 0.2,
        "input OFV {} vs Pharmpy {}",
        input.ofv.unwrap(),
        reference.base_ofv
    );
    // Seven η subsets, one block, the comparison with the input.
    assert_eq!(result.rows.iter().filter(|r| r.step == 1).count(), 7);
    assert_eq!(result.rows.iter().filter(|r| r.step == 2).count(), 1);
    assert_candidates(variant, &result, 0.05, &[]);
    assert_step_winners(variant, &result);
    assert_eq!(
        result.final_structure.description(),
        variant.final_description
    );
    let final_ofv = result.final_fit.as_ref().unwrap().ofv;
    assert!(
        (final_ofv - variant.final_ofv).abs() < 0.01,
        "final OFV {final_ofv} vs Pharmpy {}",
        variant.final_ofv
    );
    assert!(
        result.notes.iter().all(|n| !n.contains("#1018")),
        "{:?}",
        result.notes
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bottom_up_stepwise_follows_pharmpys_trajectory() {
    let reference = reference();
    let variant = &reference.variants["bu"];
    let dir = tempfile::tempdir().unwrap();
    let result = run(
        dir.path(),
        "IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,@IIV)",
        "bottom_up_stepwise",
    );
    assert_eq!(result.base_id, "base");
    assert_eq!(result.row("base").unwrap().structure.description(), "[CL]");
    assert_candidates(variant, &result, 0.05, &[]);
    assert_step_winners(variant, &result);
    assert_eq!(
        result.final_structure.description(),
        variant.final_description
    );
    let final_ofv = result.final_fit.as_ref().unwrap().ofv;
    assert!((final_ofv - variant.final_ofv).abs() < 0.01);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn simultaneous_stepwise_follows_pharmpy_on_the_mixed_omega() {
    let reference = reference();
    let variant = &reference.variants["sim"];
    let mfl = "IIV(CL,EXP);IIV?([V,KA],EXP);COVARIANCE?(IIV,@IIV)";

    // Without the boundary gate: step 1 as Pharmpy, then the divergence.
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), mfl, "simultaneous_stepwise");
    // Pharmpy 2.2.0 builds the "KA joined into the [CL,V] block" candidate
    // from pairwise covariance features and writes `[KA,V]+[CL]`; ferx
    // writes the block that was meant.
    assert_candidates(variant, &result, 0.05, &[("run6", "[CL,KA,V]")]);
    let step1 = variant
        .summary_tool
        .iter()
        .find(|r| r.step == 1 && r.rank == Some(1))
        .unwrap();
    assert_eq!(result.steps[0].best, ferx_id(&step1.model));
    assert_eq!(step1.description, "[CL,V]");
    // Step 2: `[CL,V]+[KA]` is a mixed ω. Before ferx-core #1018 the outer
    // optimizer fitted it with the cross-block covariances free — the full
    // `[CL,V,KA]` block, 647.73 against NONMEM's 655.47 — and the search ended
    // on it where Pharmpy ends on `[CL,V]`. With the structural zeros held,
    // ferx fits the model as declared: 655.02 against that NONMEM run, which
    // itself terminated with unreportable significant digits, so the anchor is
    // "within 1 OFV unit of a non-converged reference", not equality.
    let run5 = result.row("run5").expect("run5");
    assert_eq!(run5.structure.description(), "[CL,V]+[KA]");
    assert!(run5.structure.is_partial_block());
    let pharmpy_run5 = variant
        .summary_models
        .iter()
        .find(|m| m.model == "iivsearch_run5")
        .unwrap();
    assert_eq!(pharmpy_run5.minimization_successful, Some(false));
    let gap = run5.ofv.unwrap() - pharmpy_run5.ofv;
    assert!(
        gap.abs() < 1.0,
        "ferx run5 {} vs NONMEM {} (gap {gap:.3}): the declared model should now \
         fit within an OFV unit of NONMEM's own non-converged run",
        run5.ofv.unwrap(),
        pharmpy_run5.ofv
    );
    // The run no longer carries the #1018 caveat: nothing is fitted as a larger
    // block than its description.
    assert!(
        !result.notes.iter().any(|n| n.contains("#1018")),
        "{:?}",
        result.notes
    );
    // …and the search ends where Pharmpy ends.
    assert_eq!(variant.final_description, "[CL,V]");
    assert_eq!(
        result.final_structure.description(),
        variant.final_description
    );
}
