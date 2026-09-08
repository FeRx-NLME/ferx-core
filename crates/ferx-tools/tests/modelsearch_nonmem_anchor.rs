//! The NONMEM anchor for modelsearch (#1181): the same base, the same
//! space and the same candidate inits, fitted by NONMEM 7.5.1 and by ferx,
//! must give the same objective and the same BIC ranking.
//!
//! The NONMEM side is committed under `tests/nonmem/modelsearch_anchor/`:
//! `base.ctl` is the warfarin one-compartment oral model under FOCEI;
//! `lag.ctl`, `p1.ctl` and `p1_lag.ctl` are the three candidates of the
//! space `PERIPHERALS(0..1); LAGTIME([OFF,ON])`, written by hand the way the
//! search writes them — seeded from `base.ext`, with `Q = CL`, `V2 = 0.05·V`,
//! `ALAG = 0.25` and an η of 0.01 on the lag. The ferx side is `base.ferx` +
//! `anchor.ferxsearch` in the same directory, run exhaustively with the
//! strictness gate off so every candidate is ranked, as NONMEM's are.
//!
//! # Two sets of NONMEM numbers
//!
//! NONMEM's own minimiser reports `MINIMIZATION TERMINATED` on all three
//! candidates. Warfarin's first sample is at 0.5 h, so the lag is barely
//! identified and its η collapses (ω → 1e-6); the peripheral of the
//! two-compartment lag model collapses too. On those flat directions NONMEM
//! stopped at −287.053 (lag), −289.073 (two-cpt) and −287.483 (two-cpt +
//! lag), while ferx reached −287.629, −289.099 and −287.629.
//!
//! So the comparison is made **at the same point**: `lag_eval.ctl`,
//! `p1_eval.ctl` and `p1_lag_eval.ctl` evaluate NONMEM's objective
//! (`MAXEVAL=0`) at ferx's estimates, and agree with ferx to 1.2e-9,
//! 3.7e-9 and 5.4e-5 (the last on a corner with `Q = 3069`, `V2 = 9e-6`).
//! And `lag_refit.ctl` re-minimises the lag model *from* ferx's estimates:
//! NONMEM declares `MINIMIZATION SUCCESSFUL` there, at −287.62917388 against
//! ferx's −287.62917388 — ferx's optimum is one NONMEM confirms, and
//! NONMEM's own run had stopped short of it.
//!
//! The ranking is on the mixed BIC, whose penalty depends only on the
//! model's parameter classes — the same on both sides — so NONMEM's BIC is
//! its OFV plus the penalty ferx computed for the same structure. The
//! ranking is asserted under *both* sets of NONMEM numbers: it is the same
//! either way, and the smallest BIC gap between ranked models (2.98,
//! base vs lag) is four orders above the worst OFV disagreement.
//!
//! # The `[odes]` candidate family (#1257)
//!
//! `the_michaelis_menten_candidate_evaluates_as_nonmem_does` is a second,
//! narrower anchor for the candidates that have no `pk` template at all. It
//! compares an **evaluation** — `MAXEVAL=0` on both sides at one stated
//! parameter vector — because there the object under test is the generated
//! `[odes]` equation and nothing else. `mm_base.ferx` / `mm_base.ctl` are the
//! first-order control at the same θ; `mm_eval.ctl` is the Michaelis-Menten
//! candidate as the search writes it.
//!
//! Slow: four FOCEI fits, plus two evaluations.

use std::path::Path;

use ferx_core::fit;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_tools::modelsearch::{
    run_modelsearch, Absorption, Elimination, ModelsearchRun, Structure,
};
use ferx_tools::search::SearchConfig;

const ANCHOR_DIR: &str = "tests/nonmem/modelsearch_anchor";

/// `OBJ` of NONMEM's own minimisations (`base.ext`, `p1.ext`, `lag.ext`,
/// `p1_lag.ext`) — the base successful, the candidates `TERMINATED`.
const NM_BASE: f64 = -286.00421948870667;
const NM_P1: f64 = -289.07268330983356;
const NM_LAG: f64 = -287.05255519150103;
const NM_P1_LAG: f64 = -287.48301406411423;

/// `OBJ` of NONMEM evaluated at ferx's estimates (`p1_eval.ext`,
/// `lag_eval.ext`, `p1_lag_eval.ext`); the base is its own minimum.
const NM_P1_AT_FERX: f64 = -289.09862517584287;
const NM_LAG_AT_FERX: f64 = -287.62917388013796;
const NM_P1_LAG_AT_FERX: f64 = -287.62922649869876;

/// Measured, not assumed: the worst |ferx − NONMEM| at the same point on
/// the run that fixed these numbers was 5.4e-5 (`p1_lag`, whose optimum is
/// a degenerate corner where the two engines' two-compartment arithmetic
/// parts in the last digits); the other three agree to 4e-9 or better.
/// The bound leaves ~20× headroom for a different optimizer path and is
/// still four orders below the smallest BIC gap the ranking depends on.
const OFV_TOL: f64 = 1e-3;

fn structure(peripherals: u32, lagtime: bool) -> Structure {
    Structure {
        absorption: Absorption::Fo,
        elimination: Elimination::Fo,
        peripherals,
        transits: None,
        lagtime,
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn modelsearch_ranks_the_warfarin_candidates_as_nonmem_does() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR_DIR);
    let config = SearchConfig::load(dir.join("anchor.ferxsearch")).unwrap();
    let base = config.load_base().unwrap();
    let out = tempfile::tempdir().unwrap();
    let result = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
            dir: Some(out.path().join("run")),
            ..ModelsearchRun::default()
        },
    )
    .expect("search");

    // (label, structure, NONMEM at ferx's point, NONMEM's own minimum)
    let expected = [
        ("base", structure(0, false), NM_BASE, NM_BASE),
        ("PERIPHERALS(1)", structure(1, false), NM_P1_AT_FERX, NM_P1),
        ("LAGTIME(ON)", structure(0, true), NM_LAG_AT_FERX, NM_LAG),
        (
            "LAGTIME(ON); PERIPHERALS(1)",
            structure(1, true),
            NM_P1_LAG_AT_FERX,
            NM_P1_LAG,
        ),
    ];
    assert_eq!(result.rows.len(), 4, "{:?}", result.rows);
    let mut worst = 0.0f64;
    let mut nm_bic_at_ferx: Vec<(String, f64)> = Vec::new();
    let mut nm_bic_own: Vec<(String, f64)> = Vec::new();
    for (label, s, nm_at_ferx, nm_own) in expected {
        let row = result
            .rows
            .iter()
            .find(|r| r.structure == s)
            .unwrap_or_else(|| panic!("no row for {label}"));
        let ofv = row.ofv.expect("fitted");
        assert!(ofv.is_finite(), "{label}: {ofv}");
        assert!(row.error.is_none(), "{label}: {:?}", row.error);
        let err = (ofv - nm_at_ferx).abs();
        assert!(err.is_finite());
        worst = worst.max(err);
        assert!(
            err < OFV_TOL,
            "{label}: ferx OFV {ofv} vs NONMEM at the same point {nm_at_ferx} (|Δ| = {err:.3e})"
        );
        // ferx never stops above NONMEM's own (terminated) minimum.
        assert!(
            ofv <= nm_own + OFV_TOL,
            "{label}: ferx OFV {ofv} is above NONMEM's own minimum {nm_own}"
        );
        assert!(row.criterion.is_finite(), "{label}");
        // The penalty is the model's, so NONMEM's BIC on the same
        // structure is its OFV plus ferx's penalty.
        let penalty = row.criterion - ofv;
        nm_bic_at_ferx.push((row.id.clone(), nm_at_ferx + penalty));
        nm_bic_own.push((row.id.clone(), nm_own + penalty));
    }
    eprintln!("worst |ferx − NONMEM at the same point| OFV: {worst:.3e}");

    // The same ranking by the mixed BIC under either NONMEM reading, and
    // the same final model.
    let ferx_order: Vec<&str> = result.ranked().iter().map(|r| r.id.as_str()).collect();
    for (what, mut nm) in [
        ("at ferx's point", nm_bic_at_ferx),
        ("own minima", nm_bic_own),
    ] {
        nm.sort_by(|a, b| a.1.total_cmp(&b.1));
        let nm_order: Vec<&str> = nm.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ferx_order, nm_order, "NONMEM {what}");
        assert_eq!(result.final_id, nm_order[0], "NONMEM {what}");
        // …and the ranking is not decided by numbers finer than the
        // engines agree on: every BIC gap is wider than the worst OFV
        // disagreement, by orders of magnitude.
        for pair in nm.windows(2) {
            assert!(
                pair[1].1 - pair[0].1 > 100.0 * OFV_TOL,
                "NONMEM {what}: {} and {} are separated by only {:.3} BIC",
                pair[0].0,
                pair[1].0,
                pair[1].1 - pair[0].1
            );
        }
    }
    assert_eq!(
        result.final_id, "base",
        "no candidate earns its extra parameters on 10 subjects"
    );
    let order: Vec<Structure> = result.ranked().iter().map(|r| r.structure).collect();
    assert_eq!(
        order,
        vec![
            structure(0, false),
            structure(0, true),
            structure(1, false),
            structure(1, true)
        ]
    );
}

// ── The `[odes]` candidate family (#1257) ──────────────────────────────────

/// NONMEM's objective for the Michaelis-Menten candidate, evaluated
/// (`MAXEVAL=0`) at the inits the search derives — `mm_eval.ext`. Its
/// first-order twin at the same θ is `mm_base.ext`.
const NM_MM_EVAL: f64 = 929.63055240774270;
const NM_MM_BASE_EVAL: f64 = -182.66670807964042;

/// Measured, not assumed: the realised |ferx − NONMEM| on the run that fixed
/// these numbers was **9.6e-6** for the Michaelis-Menten candidate and
/// **1.0e-8** for its analytic control. Both sides integrate to a stated
/// tolerance (`TOL=9 ATOL=12` against `ode_reltol = 1e-10`,
/// `ode_abstol = 1e-12`), so the residual is solver error, not model error;
/// the bound leaves ~100× headroom over it and is still five orders below
/// the 1112-unit gap between the two objectives being told apart.
const MM_OFV_TOL: f64 = 1e-3;

/// `KM`'s declaration, as the search derives it from the data: `max(DV)/2`,
/// bounded above by `1.5·max(DV)`. `max(DV) = 13.8818` on this dataset, and
/// `mm_eval.ctl`'s `$THETA (0, 6.9409, 20.8227)` is the same declaration.
const NM_MM_KM: &str = "6.9409";
const NM_MM_KM_UPPER: &str = "20.8227";

/// The `[odes]` candidate family against NONMEM (#1257).
///
/// `ELIMINATION(MM)` has no `pk` template, so the search writes it as an
/// `ode_template` line plus one `[odes]` override of the `central` equation.
/// That equation is the new numerical object, and NONMEM can state it exactly
/// (`$DES` with a `CLMM·KM/(KM + C)` clearance), so it is anchored directly.
///
/// Both sides **evaluate** at one stated parameter vector — the base's own θ
/// plus `KM = max(DV)/2` — rather than minimise. That is the point: the object
/// under test is the generated equation, and comparing two minimisations of a
/// Michaelis-Menten model would compare two optimizers on the canonical
/// multimodal surface (`docs/examples/multistart.qmd`) instead. The analytic
/// first-order candidate at the same θ is the control: it says the agreement
/// below belongs to the saturable term and is not an accident of the dataset.
///
/// The candidate is generated by the search and then evaluated **directly**,
/// with one start. Through the runner it would be fitted with the
/// saturable-elimination start budget (`MM_STARTS`), and a multi-start under
/// `maxiter = 0` reports the best of N *perturbed* evaluations rather than the
/// objective at the stated inits — which is what the budget is for, and not
/// what an equation anchor can compare.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn the_michaelis_menten_candidate_evaluates_as_nonmem_does() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR_DIR);
    let config = SearchConfig::load(dir.join("mm_anchor.ferxsearch")).unwrap();
    let base = config.load_base().unwrap();
    let out = tempfile::tempdir().unwrap();
    let result = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
            dir: Some(out.path().join("run")),
            ..ModelsearchRun::default()
        },
    )
    .expect("search");

    let mm = result
        .rows
        .iter()
        .find(|r| r.structure.elimination == Elimination::Mm)
        .expect("the Michaelis-Menten candidate is in the table");
    let text = &result.models[&mm.id];

    // The model NONMEM was handed. Pinned here rather than described, so a
    // change to the generated text has to come back through this file and
    // through `mm_eval.ctl` beside it.
    assert_eq!(
        text.block_lines("structural_model"),
        vec!["ode_template one_cpt_oral(cl=CL, v=V, ka=KA)"]
    );
    assert_eq!(
        text.block_lines("odes"),
        vec!["d/dt(central) = KA * depot - ((CL * KM / (KM + central / V)) / V) * central"]
    );
    assert!(
        text.block_lines("parameters")
            .contains(&format!("theta TVKM({NM_MM_KM}, 0.0, {NM_MM_KM_UPPER})")),
        "{:?}",
        text.block_lines("parameters")
    );

    let ofv = evaluate(&text.render(), &base.prepared.population);
    let base_ofv = evaluate(&result.models["base"].render(), &base.prepared.population);
    eprintln!(
        "MM candidate: ferx {ofv:.8}  NONMEM {NM_MM_EVAL:.8}  |Δ| = {:.3e}\n\
         FO control:   ferx {base_ofv:.8}  NONMEM {NM_MM_BASE_EVAL:.8}  |Δ| = {:.3e}",
        (ofv - NM_MM_EVAL).abs(),
        (base_ofv - NM_MM_BASE_EVAL).abs()
    );
    assert!(
        (ofv - NM_MM_EVAL).abs() < MM_OFV_TOL,
        "ferx {ofv} vs NONMEM {NM_MM_EVAL}"
    );
    assert!(
        (base_ofv - NM_MM_BASE_EVAL).abs() < MM_OFV_TOL,
        "ferx {base_ofv} vs NONMEM {NM_MM_BASE_EVAL}"
    );
    // The two objectives are 1112 units apart, so the anchor cannot pass on a
    // candidate whose override the engine ignored.
    assert!(
        (ofv - base_ofv).abs() > 1000.0,
        "{ofv} vs {base_ofv}: the saturable term must change the objective"
    );
}

/// Evaluate one model's objective at the parameters it declares, with a
/// single start — `MAXEVAL=0`, the same question NONMEM was asked.
fn evaluate(src: &str, population: &ferx_core::Population) -> f64 {
    let parsed = parse_full_model(src).unwrap_or_else(|e| panic!("{e}\n---\n{src}"));
    let mut options = parsed.fit_options.clone().quiet();
    options.n_starts = 1;
    fit(
        &parsed.model,
        population,
        &parsed.model.default_params,
        &options,
    )
    .expect("evaluation")
    .ofv
}
