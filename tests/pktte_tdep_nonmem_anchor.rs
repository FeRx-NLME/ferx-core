//! NONMEM anchor for a joint PK-TTE model whose hazard is **time-dependent** and whose
//! PK block is **autonomous** — the shape #1166 is about, and the one nothing committed
//! exercised before it (a grep for a time-reading `hazard =` over `src/`, `tests/`,
//! `examples/` and `docs/` was empty).
//!
//! Reference: `nonmem_anchor/pktte_tdep.ctl`, NONMEM 7.6.0 `ADVAN13 TOL=9`, `$OMEGA 0 FIX`,
//! `MAXEVAL=0 POSTHOC`, on `nonmem_anchor/pktte_tdep.csv` (40 subjects, 8 PK observations
//! each plus one exact event). `#OBJV = 1912.025`. Outputs in `nonmem_anchor/results/`.
//!
//! Two objects are compared, and the reason for each:
//!
//!   * **`A(3)`, the cumulative hazard** — constant-free, so it is comparable without any
//!     convention reconciliation.
//!   * **Per-subject `OBJ` from the `.phi`** — comparable because the ferx↔NONMEM
//!     per-subject constant was *measured* to be zero on the GAM = 0 twin of this run:
//!     `2 × individual_nll` **is** NONMEM's `OBJ`, with no `n·log 2π` and nothing else
//!     (#1166 item 0). That reconciliation is what makes this half of the anchor sound.
//!
//! Bounds are measured, not chosen: at the `[fit_options]` tolerances in
//! `pktte_tdep_fit.ferx` (reltol 1e-9 / abstol 1e-11) the worst observed disagreement is
//! 3.57e-9 relative on `H(t)`, 4.33e-9 on `h(t)` and 6.02e-8 absolute per subject. The
//! asserted bounds sit roughly 15× above those. At ferx's *default* tolerances the same
//! comparison is only good to ~8e-4 per subject, which is why the model file pins them.
//!
//! This anchors the **value**. That the model takes the single shared solve rather than
//! the two-engine fallback is pinned separately, by
//! `stats::likelihood::tests::joint_pktte_share_admits_a_time_dependent_hazard` and
//! `…::identical_arithmetic_takes_the_identical_engine_bit_for_bit`.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::stats::likelihood::individual_nll;
use ferx_core::{predict_survival, CompiledModel, Population};

const MODEL: &str = "nonmem_anchor/pktte_tdep_fit.ferx";
const DATA: &str = "nonmem_anchor/pktte_tdep.csv";
const TAB: &str = "nonmem_anchor/results/pktte_tdep.tab";
const PHI: &str = "nonmem_anchor/results/pktte_tdep.phi";

/// The NONMEM-keyed CSV serves both sides unchanged here — observations sit on CMT 2
/// (central) and the events on CMT 3, which is exactly what the ferx model declares — so
/// unlike `per_route_lag` / `lagged_zo` there is no differently-keyed `data/` twin.
fn load(src: &str) -> (CompiledModel, Population) {
    let m = parse_model_string(src).expect("anchor model must parse");
    let (pop, _) = read_population_for(&m, &None, DATA, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    (m, pop)
}

fn model_src() -> String {
    std::fs::read_to_string(MODEL).expect("anchor model file")
}

/// `(id, time, cumulative hazard, instantaneous hazard)` for every TTE record.
fn nonmem_tte_rows() -> Vec<(i64, f64, f64, f64)> {
    std::fs::read_to_string(TAB)
        .expect("NONMEM table")
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // ID TIME CMT DV IPRED CHZ HAZ ETA1 MDV
            (f.len() >= 9 && f[2] == 3.0 && f[8] == 0.0).then(|| (f[0] as i64, f[1], f[5], f[6]))
        })
        .collect()
}

fn nonmem_obj() -> Vec<f64> {
    std::fs::read_to_string(PHI)
        .expect("NONMEM .phi")
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .nth(4)
                .expect("OBJ column")
                .parse()
                .expect("OBJ is numeric")
        })
        .collect()
}

/// ferx's `H(t)` and `h(t)` at every TTE record match NONMEM's `A(3)` and `HAZ`.
#[test]
fn cumulative_hazard_matches_nonmem_for_a_time_dependent_hazard() {
    let (m, pop) = load(&model_src());
    let rows = nonmem_tte_rows();
    assert_eq!(rows.len(), 40, "40 TTE records in the reference table");

    let mut grid: Vec<f64> = rows.iter().map(|r| r.1).collect();
    grid.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
    grid.dedup();
    let sv = predict_survival(&m, &pop, &m.default_params, &grid);

    let (mut worst_h, mut worst_haz) = (0.0f64, 0.0f64);
    for (id, t, chz, haz) in &rows {
        let r = sv
            .iter()
            .find(|r| r.id == id.to_string() && (r.time - t).abs() < 1e-9)
            .expect("every TTE time is on the grid");
        // `f64::max` RETURNS THE OTHER OPERAND when one side is NaN, so folding a
        // NaN relative error through `.max()` silently leaves the running worst at
        // 0 and every bound below passes. An anchor that goes green on NaN
        // predictions is worse than no anchor, so the values are checked first.
        assert!(
            r.cum_hazard.is_finite() && r.hazard.is_finite(),
            "subject {id} at t={t}: ferx returned a non-finite H/h \
             (H={}, h={}) — the anchor cannot compare it",
            r.cum_hazard,
            r.hazard
        );
        worst_h = worst_h.max(((r.cum_hazard - chz) / chz).abs());
        worst_haz = worst_haz.max(((r.hazard - haz) / haz).abs());
    }
    assert!(
        worst_h < 5e-8,
        "H(t) vs NONMEM A(3): worst relative error {worst_h:.3e} (measured 3.57e-9)"
    );
    assert!(
        worst_haz < 5e-8,
        "h(t) vs NONMEM HAZ: worst relative error {worst_haz:.3e} (measured 4.33e-9)"
    );
}

/// The per-subject objective matches NONMEM's `.phi` `OBJ` with **no constant**.
#[test]
fn per_subject_objective_matches_nonmem_for_a_time_dependent_hazard() {
    let (m, pop) = load(&model_src());
    assert!(
        m.eta_names.is_empty(),
        "the anchor is evaluated at eta = 0 against NONMEM's `$OMEGA 0 FIX` run"
    );
    let obj = nonmem_obj();
    assert_eq!(obj.len(), pop.subjects.len(), "one OBJ per subject");

    let p = &m.default_params;
    let mut worst = 0.0f64;
    let mut total = 0.0f64;
    for (s, o) in pop.subjects.iter().zip(obj.iter()) {
        let ferx = 2.0 * individual_nll(&m, s, &p.theta, &[], &p.omega, &p.sigma.values);
        // See the note in the sibling test: a NaN folded through `.max()` vanishes
        // and leaves `worst` at 0, so the bound below would pass on a broken solve.
        assert!(
            ferx.is_finite(),
            "subject {}: ferx objective is not finite ({ferx})",
            s.id
        );
        worst = worst.max((ferx - o).abs());
        total += ferx;
    }
    assert!(
        worst < 1e-6,
        "per-subject 2·NLL vs NONMEM OBJ: worst absolute difference {worst:.3e} \
         (measured 6.02e-8)"
    );
    assert!(
        (total - 1912.025).abs() < 1e-2,
        "summed objective {total:.6} vs NONMEM #OBJV 1912.025"
    );
}

/// Non-degeneracy: the time term is **live**, so the anchor above could not have passed
/// with the hazard's `TIME` dependence silently dropped.
///
/// The comparator is this model's own `GAM = 0` twin — the only difference is the factor
/// `exp(GAM·t)` on the integrand — and the gap it has to clear is the anchor's own bound,
/// by four to eight orders of magnitude at every record rather than only at the late ones.
#[test]
fn the_time_term_in_the_hazard_is_live_at_every_record() {
    let (m, pop) = load(&model_src());
    let src0 = model_src().replace("theta TVGAM(0.15, FIX)", "theta TVGAM(0.0, FIX)");
    assert!(src0.contains("TVGAM(0.0"), "the twin must actually differ");
    let (m0, pop0) = load(&src0);

    let rows = nonmem_tte_rows();
    let mut grid: Vec<f64> = rows.iter().map(|r| r.1).collect();
    grid.sort_by(|a, b| a.partial_cmp(b).expect("finite times"));
    grid.dedup();
    let live = predict_survival(&m, &pop, &m.default_params, &grid);
    let flat = predict_survival(&m0, &pop0, &m0.default_params, &grid);

    let (mut smallest, mut largest) = (f64::INFINITY, 0.0f64);
    for (id, t, _, _) in &rows {
        let at = |v: &[ferx_core::SurvivalPredictionResult]| {
            v.iter()
                .find(|r| r.id == id.to_string() && (r.time - t).abs() < 1e-9)
                .expect("grid point")
                .cum_hazard
        };
        let rel = ((at(&live) - at(&flat)) / at(&flat)).abs();
        // Same NaN trap as the sibling tests, and worse here: `f64::min` discards a
        // NaN too, so an all-NaN run would leave `smallest` at INFINITY and pass
        // the "the term is live" bound while proving nothing at all.
        assert!(
            rel.is_finite(),
            "subject {id} at t={t}: relative gap is {rel}"
        );
        smallest = smallest.min(rel);
        largest = largest.max(rel);
    }
    assert!(
        smallest > 5e-3,
        "the earliest TTE record must still see the time term: {smallest:.3e} \
         (measured 5.5e-3, against a 5e-8 anchor bound)"
    );
    assert!(
        largest > 0.5,
        "and the latest must see it plainly: {largest:.3e} (measured 0.85)"
    );
}
