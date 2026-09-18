//! #1009 — NONMEM anchor for a dataset that does not say which compartment to
//! dose, on a model whose dosed state is not the first one.
//!
//! # The object
//!
//! `defdose_depot_second_fit.ferx` is a 1-cpt oral model written as `[odes]` with
//! `states = [central, depot]`, so the dose belongs in compartment **2**. Its
//! NONMEM twin declares the same thing the other way round —
//! `$MODEL COMP=(CENTRAL, DEFOBS) COMP=(DEPOT, DEFDOSE)` — and the two engines
//! agree to 14 digits once the compartment agrees.
//!
//! Three datasets, identical but for how they spell `CMT`:
//!
//! | file | `CMT` on dose rows | NONMEM doses | ferx doses |
//! |---|---|---|---|
//! | `defdose_cmt2.csv` | `2` | DEPOT | depot |
//! | `defdose_cmt2float.csv` | `2.0` | DEPOT | depot — *before this fix, compartment 1* |
//! | `defdose_no_cmt.csv` | *(no column)* | DEPOT (`DEFDOSE`) | compartment 1 = central |
//!
//! All three NONMEM runs return **OBJV −182.66670816329514**, because NM-TRAN
//! resolves an undecorated dose against the model's declared `DEFDOSE`. ferx has
//! no such declaration to fall back on, which is the whole of #1009: the float
//! spelling is a reader bug and is fixed; the absent column is a genuine
//! ambiguity and is now *reported* (`W_CMT_DEFAULTED`) rather than silently
//! resolved.
//!
//! # Why this is a sound oracle
//!
//! The engines are independent statements of the same ODE — ADVAN13 at `TOL=9`
//! against Dormand–Prince at `reltol 1e-9 / abstol 1e-11` — and `MAXEVAL=0` /
//! `maxiter = 0` removes the optimizer from both sides, so what is compared is the
//! FOCEI objective at the initial estimates, not two searches that happened to
//! stop nearby. The datasets committed here are the *exact bytes* NONMEM read
//! (CRLF and all), copied out of the run directories.
//!
//! Non-degeneracy: the two arms of the object are live on both sides. The `2` and
//! `2.0` datasets differ by a single character and must agree bit-for-bit — that
//! is the fix. The no-`CMT` dataset must *not* agree, and must keep the objective
//! it had before this PR, which is what pins that nothing about dose routing
//! moved. Either half alone is satisfied by an implementation that ignores `CMT`
//! entirely.
//!
//! Fast: `maxiter = 0` on 10 subjects / 110 observations, so it runs on every PR.

use ferx_core::{fit, parse_full_model_file, read_nonmem_csv, FitResult};

/// NONMEM OBJV, from the `-1000000000` row of every one of the three
/// `nonmem_anchor/results/defdose_*.ext` files (they agree exactly).
const NONMEM_OBJV: f64 = -182.666708_163_295_14;

/// The objective the no-`CMT` dataset produces when the dose lands in compartment
/// 1 (`central`) instead of `depot`. Not a NONMEM number — NONMEM cannot express
/// this run — but the ferx number this PR must not change, measured at
/// `d3f2ca29`. A bolus into a concentration-unit state is 100·e^(−(CL/V)t), which
/// is nowhere near the data; the size of the miss is the point.
const FERX_WRONG_COMPARTMENT_OFV: f64 = 154_050.610_450;

fn anchor(name: &str) -> String {
    format!("{}/nonmem_anchor/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// Fit `nonmem_anchor/<data>` with the committed anchor model at `maxiter = 0`.
fn fit_anchor(data: &str) -> FitResult {
    let parsed = parse_full_model_file(std::path::Path::new(&anchor(
        "defdose_depot_second_fit.ferx",
    )))
    .expect("the anchor model parses");
    let population = read_nonmem_csv(std::path::Path::new(&anchor(data)), None, None)
        .unwrap_or_else(|e| panic!("{data} loads: {e}"));
    assert_eq!(
        population.subjects.len(),
        10,
        "{data} must carry all 10 warfarin subjects"
    );
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
    result
}

/// Measured worst |Δ OFV| between ferx and NONMEM on the two datasets that agree
/// on the compartment: **4.084e-8** at `d3f2ca29`, a relative error of 2.2e-10 on
/// an objective of magnitude 183 — the size two independent solvers at
/// `reltol 1e-9` and `TOL=9` should differ by. Bound is that number with one
/// order of magnitude of headroom. It is 3e11 times tighter than the defect it
/// guards (a compartment swap moves the objective by 154233).
const OBJV_TOL: f64 = 5e-7;

#[test]
fn explicit_cmt2_matches_the_nonmem_objective() {
    // (i) The control arm: both engines dose DEPOT, so they must agree.
    let result = fit_anchor("defdose_cmt2.csv");
    let got = result.ofv;
    let err = (got - NONMEM_OBJV).abs();
    println!("defdose_cmt2: ferx {got:.12}, NONMEM {NONMEM_OBJV:.12}, |Δ| = {err:.3e}");
    assert!(
        err < OBJV_TOL,
        "ferx {got:.12} vs NONMEM {NONMEM_OBJV:.12} (|Δ| = {err:.3e}, bound {OBJV_TOL:.1e})"
    );
    // The warning must be silent here: the dataset said which compartment.
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("W_CMT_DEFAULTED")),
        "an explicit CMT column is not a default: {:?}",
        result.warnings
    );
}

#[test]
fn float_formatted_cmt_is_bit_identical_to_the_integer_spelling() {
    // (ii) The fix. `2.0` and `2` are the same dataset spelled two ways, so the
    // objective must be identical to the last bit — not "close". Before the fix
    // the float arm produced 154050.610450 instead, an 154233-unit miss.
    let integer = fit_anchor("defdose_cmt2.csv").ofv;
    let float = fit_anchor("defdose_cmt2float.csv").ofv;
    assert!(float.is_finite(), "float arm objective must be finite");
    assert_eq!(
        float.to_bits(),
        integer.to_bits(),
        "`2.0` must read as compartment 2: got {float:.12} against {integer:.12}"
    );
    // Both are the NONMEM number; asserting it here too means this test still
    // says something if the control arm is ever deleted.
    let err = (float - NONMEM_OBJV).abs();
    println!("defdose_cmt2float: ferx {float:.12}, NONMEM {NONMEM_OBJV:.12}, |Δ| = {err:.3e}");
    assert!(
        err < OBJV_TOL,
        "float arm: {float:.12} (|Δ| = {err:.3e}, bound {OBJV_TOL:.1e})"
    );
}

#[test]
fn absent_cmt_column_keeps_its_objective_and_is_reported() {
    // (iii) The ambiguity. ferx doses compartment 1 where NONMEM doses DEPOT, and
    // this PR does not change that — it makes it audible. Two assertions, because
    // either alone is satisfiable by the wrong implementation: a routing change
    // would move the objective, and an inverted suppression predicate would drop
    // the warning.
    let result = fit_anchor("defdose_no_cmt.csv");
    let err = (result.ofv - FERX_WRONG_COMPARTMENT_OFV).abs();
    println!(
        "defdose_no_cmt: ferx {:.6}, pre-PR {FERX_WRONG_COMPARTMENT_OFV:.6}, |Δ| = {err:.3e}; \
         NONMEM would have said {NONMEM_OBJV:.6}",
        result.ofv
    );
    // Realised |Δ| at `d3f2ca29`: 1.644e-7, which is the reference constant's own
    // rounding — it is a 6-decimal print, so ≤5e-7 of any error is the constant,
    // not the engine. Bound is that floor with headroom, ~60× tighter than the
    // 1e-3 a "routing did not move" check might casually be given.
    assert!(
        err < 1e-5,
        "dose routing must not have moved: {:.6} against {FERX_WRONG_COMPARTMENT_OFV:.6} \
         (|Δ| = {err:.3e})",
        result.ofv
    );
    // And it is nowhere near NONMEM's — the miss this warning exists to announce.
    //
    // This is *not* a second gate on the same inputs as the assertion above. That
    // one pins the measurement against a constant; this one pins the **constant**.
    // If someone later "fixes" this test by retargeting
    // `FERX_WRONG_COMPARTMENT_OFV` at the NONMEM value — the natural edit once a
    // model-side `default_dose_cmt` lands — the first assertion goes green again
    // while the fixture has silently stopped exercising the ambiguity. This one
    // fails in exactly that case, and only that case.
    assert!(
        (result.ofv - NONMEM_OBJV).abs() > 1e4,
        "the no-CMT arm must still disagree with NONMEM; if it agrees, either the \
         reference constant was retargeted or ferx grew a DEFDOSE equivalent — \
         both mean this fixture no longer tests what it was written for"
    );
    let hit = result
        .warnings
        .iter()
        .find(|w| w.starts_with("W_CMT_DEFAULTED"))
        .unwrap_or_else(|| panic!("no W_CMT_DEFAULTED in {:?}", result.warnings));
    assert!(
        hit.contains("10 dose row(s) and 110 observation row(s)"),
        "every dose and every scored observation was assigned a compartment: {hit}"
    );
}
