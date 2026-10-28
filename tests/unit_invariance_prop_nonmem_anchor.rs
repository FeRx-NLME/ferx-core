//! #2441 — NONMEM anchor for a proportional-error objective with predictions below
//! `1e-6`, on the same data in two units.
//!
//! # The object
//!
//! ferx used to clamp every residual variance at an absolute `1e-12`. Under
//! proportional error `V = (σ·f)²`, so with `σ = 0.2` any row with `f < 5e-6` was
//! scored at `ln 1e-12` instead of its true variance, and the objective depended on
//! the units of `DV`. NONMEM has no floor. The fix guards only a variance that
//! cannot be scored (`NaN`, `≤ 0`, subnormal).
//!
//! Two datasets, identical but for units: `unit_invariance_prop_k1.csv` and
//! `unit_invariance_prop_k1e6.csv` (`AMT` and `DV` × 1e6, the decimal exponent
//! shifted, so the mantissas are the same bytes). 8 subjects × 10 observations of a
//! 1-cpt oral model out to 96 h; at k = 1, 11 of the 80 rows have an IPRED below
//! `5e-6` at NONMEM's EBEs (down to 2.7e-11), at k = 1e6 none do.
//!
//! | dataset | NONMEM OBJV | ferx before #2441 |
//! |---|---|---|
//! | k = 1 | −705.952035430 | −606.197508814 (off by 99.75) |
//! | k = 1e6 | 1504.529653767 | matched |
//!
//! NONMEM's two objectives differ by 2210.48169, which is `2·n·ln k` with n = 80 to
//! the printed digits — the external statement of unit invariance.
//!
//! # Why this is a sound oracle
//!
//! `MAXEVAL=0` / `maxiter = 0` removes the optimizer from both sides, so the compared
//! quantity is the FOCEI objective at the initial estimates. ADVAN2 and ferx's closed
//! form are independent statements of the same Bateman function. Non-degeneracy: the
//! k = 1e6 arm has no row near the old floor, so it pins that nothing else moved; the
//! k = 1 arm has 11 rows below it, so it is the one the floor could reach. Either alone
//! passes an implementation that is right only on one side of the threshold.
//!
//! Fast: `maxiter = 0` on 8 subjects / 80 observations, so it runs on every PR.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv};

/// NONMEM 7.5.1 OBJV, the `-1000000000` row of `nonmem_anchor/results/unit_invariance_prop_k1.ext`.
const NONMEM_OBJV_K1: f64 = -705.952_035_430_100_3;
/// The same for `unit_invariance_prop_k1e6.ext`.
const NONMEM_OBJV_K1E6: f64 = 1504.529_653_767_327_3;

fn anchor(name: &str) -> String {
    format!("{}/nonmem_anchor/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn ofv_on(data: &str) -> f64 {
    let parsed = parse_full_model_file(std::path::Path::new(&anchor(
        "unit_invariance_prop_fit.ferx",
    )))
    .expect("the anchor model parses");
    let population = read_nonmem_csv(std::path::Path::new(&anchor(data)), None, None)
        .unwrap_or_else(|e| panic!("{data} loads: {e}"));
    assert_eq!(population.subjects.len(), 8, "{data}: 8 subjects");
    let result = fit(
        &parsed.model,
        &population,
        &parsed.model.default_params,
        &parsed.fit_options,
    )
    .unwrap_or_else(|e| panic!("fit on {data} returns Ok: {e}"));
    assert!(
        result.ofv.is_finite(),
        "{data}: objective must be finite before it is compared, got {}",
        result.ofv
    );
    result.ofv
}

/// Measured |Δ OFV| against NONMEM (macOS arm64, closed form vs ADVAN2): 1.087e-7 at
/// k = 1, 3.195e-8 at k = 1e6; the two ferx objectives differ from `2·n·ln k` by 1.2e-10.
/// The bound is the worst of those with ~10× headroom — 1e4× inside the issue's 0.01
/// acceptance bound and 1e8× inside the defect it guards: with the old `max(1e-12)` floor
/// restored the k = 1 objective is −606.197508814, 99.75 from NONMEM's.
const OBJV_TOL: f64 = 1e-6;

#[test]
fn small_predictions_under_proportional_error_match_nonmem() {
    let got = ofv_on("unit_invariance_prop_k1.csv");
    let err = (got - NONMEM_OBJV_K1).abs();
    println!("k=1: ferx {got:.12}, NONMEM {NONMEM_OBJV_K1:.12}, |Δ| = {err:.3e}");
    assert!(
        err < OBJV_TOL,
        "k = 1: ferx {got:.12} vs NONMEM {NONMEM_OBJV_K1:.12} (|Δ| = {err:.3e}, bound \
         {OBJV_TOL:.1e}) — rows with IPRED < 5e-6 are not being scored at their own variance"
    );
}

#[test]
fn the_same_data_in_large_units_matches_nonmem() {
    let got = ofv_on("unit_invariance_prop_k1e6.csv");
    let err = (got - NONMEM_OBJV_K1E6).abs();
    println!("k=1e6: ferx {got:.12}, NONMEM {NONMEM_OBJV_K1E6:.12}, |Δ| = {err:.3e}");
    assert!(
        err < OBJV_TOL,
        "k = 1e6: ferx {got:.12} vs NONMEM {NONMEM_OBJV_K1E6:.12} (|Δ| = {err:.3e}, bound \
         {OBJV_TOL:.1e})"
    );
}

#[test]
fn a_change_of_units_shifts_the_objective_by_2_n_ln_k() {
    let shift = ofv_on("unit_invariance_prop_k1e6.csv") - ofv_on("unit_invariance_prop_k1.csv");
    let want = 2.0 * 80.0 * 1e6_f64.ln();
    let err = (shift - want).abs();
    println!("shift {shift:.12}, 2·n·ln k = {want:.12}, |Δ| = {err:.3e}");
    assert!(
        err < OBJV_TOL,
        "a change of units moved the objective by {shift:.9}, it owes 2·n·ln k = {want:.9}"
    );
}
