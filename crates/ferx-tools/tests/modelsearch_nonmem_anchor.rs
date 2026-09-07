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
//! Slow: four FOCEI fits.

use std::path::Path;

use ferx_tools::modelsearch::{run_modelsearch, Absorption, ModelsearchRun, Structure};
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
