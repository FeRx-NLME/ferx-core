//! Tier-3: Pharmpy 2.2.0 `iovsearch` (driven by NONMEM 7.5.1) as the
//! trajectory anchor for `ferx iovsearch` (#1183).
//!
//! The reference runs and their inputs live in
//! `tests/pharmpy/iovsearch_anchor/` (see its README): a 40-subject dataset
//! over three dosing occasions simulated with IOV on CL only, an IIV-only
//! input model reading `OCC`, and Pharmpy's `summary_tool` /
//! `summary_models` / final model for the `disjoint` and `same-as-iiv`
//! distributions in `pharmpy_iovsearch.json`.
//!
//! What is anchored, per distribution: the same candidates (by description;
//! Pharmpy's numbering follows a Python set's order and is not asserted);
//! the BIC(random) of every model both engines fitted to a proper optimum,
//! to 0.25; the winner of each step; the final model.
//!
//! Slow-gated: about ten FOCEI fits on 840 observations per distribution.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ferx_tools::iovsearch::{run_iovsearch, IovsearchResult, IovsearchRun};
use ferx_tools::search::SearchConfig;

const ANCHOR: &str = "tests/pharmpy/iovsearch_anchor";

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
        .join("pharmpy_iovsearch.json");
    let text = std::fs::read_to_string(&path).expect("anchor json");
    serde_json::from_str(&text).expect("anchor json")
}

fn run(dir: &Path, distribution: &str) -> IovsearchResult {
    let anchor = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR);
    let config = format!(
        "base = \"{}\"\ndata = \"{}\"\n\n[iovsearch]\ndistribution = \"{distribution}\"\n\n\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 4\n",
        anchor.join("base.ferx").display(),
        anchor.join("iov_sim.csv").display()
    );
    let path: PathBuf = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    run_iovsearch(
        &config,
        &base,
        IovsearchRun {
            dir: Some(dir.join("run")),
            ..IovsearchRun::default()
        },
    )
    .expect("search")
}

/// A description with each family's members sorted — Pharmpy lists the η in
/// model order (`[CL]+[V]+[KA]`), ferx alphabetically (`[CL]+[KA]+[V]`).
fn canonical(desc: &str) -> String {
    desc.split(';')
        .map(|family| {
            let (name, rest) = family.split_once('(').unwrap_or((family, ""));
            let inner = rest.trim_end_matches(')');
            let mut members: Vec<String> = inner
                .split('+')
                .filter(|m| !m.is_empty())
                .map(|m| {
                    let mut names: Vec<&str> = m.trim_matches(['[', ']']).split(',').collect();
                    names.sort_unstable();
                    format!("[{}]", names.join(","))
                })
                .collect();
            members.sort();
            format!("{name}({})", members.join("+"))
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Pharmpy's `input` is ferx's `input`; every other model is matched by
/// its description.
fn ferx_row<'a>(result: &'a IovsearchResult, row: &ToolRow) -> &'a ferx_tools::iovsearch::ModelRow {
    if row.model == "input" {
        return result.row("input").unwrap();
    }
    result
        .rows
        .iter()
        .find(|r| {
            r.id != "input" && canonical(&r.structure.description()) == canonical(&row.description)
        })
        .unwrap_or_else(|| panic!("no ferx row for {} ({})", row.model, row.description))
}

fn check(variant: &Variant, result: &IovsearchResult, base_ofv: f64) {
    let input = result.row("input").unwrap();
    assert!(
        (input.ofv.unwrap() - base_ofv).abs() < 0.2,
        "input OFV {} vs Pharmpy {}",
        input.ofv.unwrap(),
        base_ofv
    );
    // The full model and six removals, then the one η removal.
    assert_eq!(result.rows.iter().filter(|r| r.step == 1).count(), 7);
    assert_eq!(result.rows.iter().filter(|r| r.step == 2).count(), 1);
    let proper: HashMap<&str, bool> = variant
        .summary_models
        .iter()
        .map(|m| (m.model.as_str(), m.minimization_successful.unwrap_or(false)))
        .collect();
    for row in &variant.summary_tool {
        let ours = ferx_row(result, row);
        let proper =
            proper.get(row.model.as_str()).copied().unwrap_or(false) || row.rank == Some(1);
        eprintln!(
            "step {} {:38} pharmpy {:10.3} ferx {:10.3}{}",
            row.step,
            row.description,
            row.bic,
            ours.criterion,
            if proper {
                ""
            } else {
                "  (NONMEM: not a proper optimum)"
            }
        );
        assert!(ours.ofv.is_some(), "{}: no fit: {:?}", ours.id, ours.error);
        if proper {
            assert!(
                (ours.criterion - row.bic).abs() < 0.25,
                "{}: BIC(random) {} vs Pharmpy {}",
                row.description,
                ours.criterion,
                row.bic
            );
        }
    }
    for row in variant.summary_tool.iter().filter(|r| r.rank == Some(1)) {
        let step = result.steps.iter().find(|s| s.step == row.step).unwrap();
        assert_eq!(
            canonical(&result.row(&step.best).unwrap().structure.description()),
            canonical(&row.description),
            "step {}",
            row.step
        );
    }
    assert_eq!(
        canonical(&result.final_structure.description()),
        canonical(&variant.final_description)
    );
    let final_ofv = result.final_fit.as_ref().unwrap().ofv;
    assert!(
        (final_ofv - variant.final_ofv).abs() < 0.01,
        "final OFV {final_ofv} vs Pharmpy {}",
        variant.final_ofv
    );
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn disjoint_follows_pharmpys_trajectory() {
    let reference = reference();
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), "disjoint");
    check(&reference.variants["disjoint"], &result, reference.base_ofv);
    assert!(result
        .rows
        .iter()
        .all(|r| r.structure.kappa_blocks.is_empty()));
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn same_as_iiv_follows_pharmpys_trajectory() {
    let reference = reference();
    let dir = tempfile::tempdir().unwrap();
    let result = run(dir.path(), "same-as-iiv");
    check(
        &reference.variants["same_as_iiv"],
        &result,
        reference.base_ofv,
    );
}
