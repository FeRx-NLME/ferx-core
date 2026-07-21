//! NONMEM anchor for dose-compartment handling on the analytical engine (#375).
//!
//! Reference values are `PRED` from **NONMEM 7.6.0**, printed at full precision
//! (`$TABLE ... FORMAT=,1PE17.10`), `$ESTIMATION MAXEVAL=0`
//! (evaluation at fixed THETA, `$OMEGA 0 FIX`), one subject, a 100 mg dose at
//! t = 0 and observations at t = 1, 4, 8 h. Parameters throughout:
//! `CL = 5, V/V1 = 50, Q = 3, V2 = 80, Q3 = 1, V3 = 120, KA = 1`.
//!
//! The control streams are one `$PROBLEM` per case, `ADVAN{1,2,3,4,11,12}` with
//! `TRANS2`/`TRANS4`, differing only in the dose record's `CMT` (and `SS`/`II`
//! or `RATE` where noted). The control streams, their matched datasets, and the
//! verbatim NONMEM output are committed under `nonmem_anchor/` as
//! `dose_cmt_*.{ctl,csv}` and `nonmem_anchor/results/dose_cmt_*.{lst,tab}`, so
//! every number below can be re-derived rather than taken on trust — they were
//! re-run end-to-end on NONMEM 7.6.0 and reproduce at zero relative deviation.
//! See `nonmem_anchor/README.md` for the case table.
//!
//! Two things are pinned here.
//!
//! 1. **`SS=1` with `CMT=0`.** NONMEM produces *identical* `PRED` for `CMT=0`
//!    and `CMT=1` — confirming `CMT=0` means "the default dose compartment" and
//!    that it must equilibrate like any other. ferx's event-driven walk used to
//!    bail out of steady-state equilibration on `CMT=0` and return the
//!    single-dose curve (#375).
//! 2. **Dosing into a non-default compartment.** For every model, NONMEM agrees
//!    with ferx's **event-driven walk**. It disagrees sharply with the
//!    dose-superposition path, which never reads `dose.cmt` and computes every
//!    dose as going into the model's default route — by up to two orders of
//!    magnitude. Since #375 such a dose reroutes to
//!    the walk, so **every case here asserts both paths** — each against NONMEM,
//!    and against each other. That three-way agreement is the acceptance
//!    criterion; before the fix the `[default dispatch]` arm failed.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv, CompiledModel, Population};
use std::io::Write;
use tempfile::NamedTempFile;

fn pop_of(csv: &str) -> Population {
    let mut f = NamedTempFile::new().expect("temp file");
    write!(f, "{csv}").expect("write");
    f.flush().expect("flush");
    read_nonmem_csv(f.path(), None, None).expect("dataset loads")
}

fn model_src(structural: &str, params: &str, indiv: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
{params}
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
{indiv}

[structural_model]
  {structural}

[error_model]
  DV ~ proportional(PROP)
"#
    )
}

/// Predictions on the **event-driven walk**. The walk is selected by adding a
/// `WT` column that varies across rows; the model never references `WT`, so no
/// parameter value differs from the plain dataset — only the code path does.
fn walk_preds(src: &str, dose_cmt: usize, obs_cmt: usize, rate: f64, ss: u8, ii: f64) -> Vec<f64> {
    let model: CompiledModel = parse_full_model(src).expect("model parses").model;
    let dose = format!("1,0,.,1,100,{dose_cmt},{rate},1,{ss},{ii},70");
    let obs: String = [(1, 80), (4, 90), (8, 100)]
        .iter()
        .map(|(t, w)| format!("1,{t},5.0,0,.,{obs_cmt},.,0,0,0,{w}"))
        .collect::<Vec<_>>()
        .join("\n");
    let csv = format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II,WT\n{dose}\n{obs}\n");
    preds_of(&model, &csv)
}

/// Predictions on the **plain** dataset — no time-varying covariate, so the
/// dispatcher's default route. Historically this was the dose-superposition
/// path, which ignores `dose.cmt`; since #375 a dose the superposition form
/// cannot place reroutes to the same walk, so this must now agree with
/// `walk_preds` *and* with NONMEM. That agreement is the point of asserting
/// both.
fn plain_preds(src: &str, dose_cmt: usize, obs_cmt: usize, rate: f64, ss: u8, ii: f64) -> Vec<f64> {
    let model: CompiledModel = parse_full_model(src).expect("model parses").model;
    let dose = format!("1,0,.,1,100,{dose_cmt},{rate},1,{ss},{ii}");
    let obs: String = [1, 4, 8]
        .iter()
        .map(|t| format!("1,{t},5.0,0,.,{obs_cmt},.,0,0,0"))
        .collect::<Vec<_>>()
        .join("\n");
    let csv = format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II\n{dose}\n{obs}\n");
    preds_of(&model, &csv)
}

fn preds_of(model: &CompiledModel, csv: &str) -> Vec<f64> {
    predict(model, &pop_of(csv), &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

/// Default bound. The control streams print `PRED` with `FORMAT=,1PE17.10` (11
/// significant figures), so this is a strict anchor, not a loose one: at the
/// default 5-figure `$TABLE` precision several of these cases sit right at the
/// printing resolution and a slack tolerance would hide a real difference rather
/// than reveal one.
const NONMEM_TOL: f64 = 1e-9;

/// Assert **both** analytical paths reproduce NONMEM for the same dose row: the
/// event-driven walk, and the plain dispatcher route. Before #375 the second one
/// was dose superposition, which ignores `dose.cmt` and disagreed with NONMEM by
/// up to two orders of magnitude; the two paths agreeing *and* both landing on
/// NONMEM is the acceptance criterion.
fn assert_both_paths_match(
    src: &str,
    dose_cmt: usize,
    obs_cmt: usize,
    rate: f64,
    ss: u8,
    ii: f64,
    nonmem: &[f64],
    tol: f64,
    case: &str,
) {
    let walk = walk_preds(src, dose_cmt, obs_cmt, rate, ss, ii);
    let plain = plain_preds(src, dose_cmt, obs_cmt, rate, ss, ii);
    assert_close(&walk, nonmem, tol, &format!("{case} [event-driven walk]"));
    assert_close(&plain, nonmem, tol, &format!("{case} [default dispatch]"));
    assert_close(&plain, &walk, tol, &format!("{case} [paths agree]"));
}

/// Bound set by **NONMEM's** accuracy on these cases, not ferx's: the measured
/// gap is 2.1e-6 on `ADVAN11 CMT=2` and 1.1e-5 on the smallest-magnitude case
/// (`ADVAN12 CMT=4`, PRED ~1.6e-3). ferx's own agreement with an independent
/// 1e-12 integration of the same system is ~1e-11 — see
/// `tests/oral_peripheral_infusion.rs`, which is the tight oracle. This anchor's
/// job is cross-tool corroboration; that file's job is precision.
///
/// **Both** the oral and the IV cases guarded by this constant have a 1e-9
/// ODE-twin counterpart there, on identical parameters and dose schedules, so
/// nothing rests on this bound alone. That was not always true: the IV pair had
/// no tight guard until one was added, and a deliberately injected 9.4e-7 error
/// in the shared `propagate_three_cpt_core_g` passed *this* assertion while
/// failing the ODE twin — which is the margin this constant cannot see.
const NONMEM_3CPT_INFUSION_TOL: f64 = 3e-5;

/// 3-compartment **bolus** cases. Those closed forms obtain their eigenvalues by
/// solving a **cubic**; ferx and NONMEM use different root-finders, so the last
/// couple of digits legitimately differ. Still a strict anchor — 1e-8 relative on
/// an 11-significant-figure reference — just not one that pretends two
/// independent cubic solves agree to the ULP.
const NONMEM_3CPT_TOL: f64 = 1e-8;

fn assert_close(got: &[f64], nonmem: &[f64], tol: f64, case: &str) {
    assert_eq!(got.len(), nonmem.len(), "{case}: row count");
    for (j, (&g, &n)) in got.iter().zip(nonmem).enumerate() {
        let rel = (g - n).abs() / n.abs().max(1e-12);
        assert!(
            rel < tol,
            "{case} obs {j}: ferx {g:.8} vs NONMEM {n:.8} (rel {rel:.2e})"
        );
    }
}

fn one_cpt_iv() -> String {
    model_src("pk one_cpt_iv(cl=CL, v=V)", "", "")
}
fn one_cpt_oral() -> String {
    model_src(
        "pk one_cpt_oral(cl=CL, v=V, ka=KA)",
        "  theta TVKA(1.0, 0.01, 10.0)",
        "  KA = TVKA",
    )
}
fn two_cpt_iv() -> String {
    model_src(
        "pk two_cpt_iv(cl=CL, v=V, q=Q, v2=V2)",
        "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)",
        "  Q  = TVQ\n  V2 = TVV2",
    )
}
fn two_cpt_oral() -> String {
    model_src(
        "pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA)",
        "  theta TVKA(1.0, 0.01, 10.0)\n  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)",
        "  KA = TVKA\n  Q  = TVQ\n  V2 = TVV2",
    )
}
const THREE_P: &str = "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)\n  \
                       theta TVQ3(1.0, 0.1, 50.0)\n  theta TVV3(120.0, 5.0, 900.0)";
const THREE_I: &str = "  Q  = TVQ\n  V2 = TVV2\n  Q3 = TVQ3\n  V3 = TVV3";
fn three_cpt_iv() -> String {
    model_src(
        "pk three_cpt_iv(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3)",
        THREE_P,
        THREE_I,
    )
}
fn three_cpt_oral() -> String {
    model_src(
        "pk three_cpt_oral(cl=CL, v=V, q=Q, v2=V2, q3=Q3, v3=V3, ka=KA)",
        &format!("  theta TVKA(1.0, 0.01, 10.0)\n{THREE_P}"),
        &format!("  KA = TVKA\n{THREE_I}"),
    )
}

// ── 1. SS=1 with CMT=0 (ADVAN1, SS=1, II=12) ─────────────────────────────────

/// NONMEM `ADVAN1 TRANS2`, `SS=1 II=12`, dose `CMT=0` and `CMT=1`. Both runs
/// print the same `PRED`, which is also the closed form
/// `(D/V)·e^{−kt}/(1−e^{−k·II})`. Before #375 the walk returned the *single
/// dose* curve for `CMT=0` (1.8097 / 1.3406 / 0.8987) — ~30 % low here.
#[test]
fn ss_dose_cmt_zero_and_one_match_nonmem() {
    const NONMEM_SS: [f64; 3] = [2.5896677831, 1.9184730793, 1.2859909628];
    for cmt in [0usize, 1] {
        assert_both_paths_match(
            &one_cpt_iv(),
            cmt,
            1,
            0.0,
            1,
            12.0,
            &NONMEM_SS,
            NONMEM_TOL,
            &format!("ADVAN1 SS=1 CMT={cmt}"),
        );
    }
}

// ── 2. bolus into a non-default compartment, per model ───────────────────────

/// `ADVAN2` (1-cpt oral), bolus into `CMT=2` — the central compartment, i.e. an
/// IV dose written against an oral model. Superposition computes 1.19324 here
/// (it treats it as a depot dose); NONMEM and the walk agree on 1.8097.
#[test]
fn one_cpt_oral_bolus_into_central_matches_nonmem() {
    assert_both_paths_match(
        &one_cpt_oral(),
        2,
        2,
        0.0,
        0,
        0.0,
        &[1.8096748361, 1.3406400921, 0.89865792823],
        NONMEM_TOL,
        "ADVAN2 bolus CMT=2",
    );
}

/// `ADVAN3` (2-cpt IV), bolus into `CMT=2` — the peripheral compartment.
#[test]
fn two_cpt_iv_bolus_into_peripheral_matches_nonmem() {
    assert_both_paths_match(
        &two_cpt_iv(),
        2,
        1,
        0.0,
        0,
        0.0,
        &[0.068015673681, 0.20535408329, 0.29007691882],
        NONMEM_TOL,
        "ADVAN3 bolus CMT=2",
    );
}

/// `ADVAN4` (2-cpt oral), bolus into `CMT=3` — the peripheral compartment.
#[test]
fn two_cpt_oral_bolus_into_peripheral_matches_nonmem() {
    assert_both_paths_match(
        &two_cpt_oral(),
        3,
        2,
        0.0,
        0,
        0.0,
        &[0.068015673681, 0.20535408329, 0.29007691882],
        NONMEM_TOL,
        "ADVAN4 bolus CMT=3",
    );
}

/// `ADVAN11` (3-cpt IV), bolus into `CMT=3` — the second peripheral.
#[test]
fn three_cpt_iv_bolus_into_peripheral2_matches_nonmem() {
    assert_both_paths_match(
        &three_cpt_iv(),
        3,
        1,
        0.0,
        0,
        0.0,
        &[0.015193556553, 0.046937359706, 0.069431483504],
        NONMEM_3CPT_TOL,
        "ADVAN11 bolus CMT=3",
    );
}

/// `ADVAN12` (3-cpt oral), bolus into `CMT=2` — the central compartment.
#[test]
fn three_cpt_oral_bolus_into_central_matches_nonmem() {
    assert_both_paths_match(
        &three_cpt_oral(),
        2,
        2,
        0.0,
        0,
        0.0,
        &[1.6726602861, 0.99662213471, 0.53066189083],
        NONMEM_3CPT_TOL,
        "ADVAN12 bolus CMT=2",
    );
}

// ── 3. infusion into an IV peripheral (accepted by check_dose_compartments) ──

/// `ADVAN3` with `RATE=20` into `CMT=2`. This compartment *is* in
/// `infusable_compartments()`, so `check_dose_compartments` accepts it — this
/// anchor confirms the walk then computes it the way NONMEM does, i.e. that
/// accepting it is correct and not merely permissive.
#[test]
fn two_cpt_iv_infusion_into_peripheral_matches_nonmem() {
    assert_both_paths_match(
        &two_cpt_iv(),
        2,
        1,
        20.0,
        0,
        0.0,
        &[0.0070275296315, 0.09332945504, 0.24115537856],
        NONMEM_TOL,
        "ADVAN3 infusion CMT=2",
    );
}

// ── 4. infusion into an ORAL model's peripheral (#375) ───────────────────────

/// `ADVAN4` (2-cpt oral) with `RATE=20` into `CMT=3` — the **peripheral** of an
/// oral model. Previously rejected: the oral propagators had no peripheral-rate
/// forcing term where the IV ones did.
///
/// The closed form needed is the IV one. Nothing flows back into the depot, so a
/// rate into the peripheral drives exactly the central/peripheral pair a 2-cpt IV
/// model has, and the oral propagator superposes that forced response onto its
/// homogeneous evolution. NONMEM confirms the reduction exactly: this run prints
/// the *same* `PRED` as `ADVAN3` with an infusion into `CMT=2`
/// (`two_cpt_iv_infusion_into_peripheral_matches_nonmem`), because the depot is
/// empty throughout.
#[test]
fn two_cpt_oral_infusion_into_peripheral_matches_nonmem() {
    assert_both_paths_match(
        &two_cpt_oral(),
        3,
        2,
        20.0,
        0,
        0.0,
        &[0.0070275296315, 0.09332945504, 0.24115537856],
        NONMEM_TOL,
        "ADVAN4 infusion CMT=3",
    );
}

/// `ADVAN12` (3-cpt oral), infusion into the **first** peripheral (`CMT=3`).
#[test]
fn three_cpt_oral_infusion_into_peripheral1_matches_nonmem() {
    assert_both_paths_match(
        &three_cpt_oral(),
        3,
        2,
        20.0,
        0,
        0.0,
        &[0.0069821063874, 0.091126295438, 0.2297334849],
        NONMEM_3CPT_INFUSION_TOL,
        "ADVAN12 infusion CMT=3",
    );
}

/// `ADVAN12` (3-cpt oral), infusion into the **second** peripheral (`CMT=4`).
#[test]
fn three_cpt_oral_infusion_into_peripheral2_matches_nonmem() {
    assert_both_paths_match(
        &three_cpt_oral(),
        4,
        2,
        20.0,
        0,
        0.0,
        &[0.0015668828542, 0.021085941254, 0.056167606771],
        NONMEM_3CPT_INFUSION_TOL,
        "ADVAN12 infusion CMT=4",
    );
}

/// 3-cpt **IV** peripheral infusions — the same forced response the oral cases
/// above delegate to. Anchored separately so a discrepancy is attributed to the
/// right place (the shared `propagate_three_cpt_core_g`), not to the oral wiring.
/// NONMEM prints identical `PRED` for these and their `ADVAN12` twins.
#[test]
fn three_cpt_iv_infusion_into_peripherals_matches_nonmem() {
    assert_both_paths_match(
        &three_cpt_iv(),
        2,
        1,
        20.0,
        0,
        0.0,
        &[0.0069821063874, 0.091126295438, 0.2297334849],
        NONMEM_3CPT_INFUSION_TOL,
        "ADVAN11 infusion CMT=2",
    );
    assert_both_paths_match(
        &three_cpt_iv(),
        3,
        1,
        20.0,
        0,
        0.0,
        &[0.0015668828542, 0.021085941254, 0.056167606771],
        NONMEM_3CPT_INFUSION_TOL,
        "ADVAN11 infusion CMT=3",
    );
}
