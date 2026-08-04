//! NONMEM anchor for an **oral model's central (`CMT=2`) infusion** — the
//! depot-bypassing `RATE>0` case PR #350 fixed on the event-driven path (#376).
//!
//! Before this file that behaviour was anchored only *transitively*: the #350
//! test asserts `event_driven_predictions == predict_concentration` (the
//! superposition reference), and that reference chains to NONMEM through the
//! IV-into-central closed form. This file replaces the chain with a direct
//! `.ctl`-backed comparison, and adds the F≠1 interaction (#327/#349/#419) the
//! transitive check could reach only via self-consistency.
//!
//! ## Reference values
//!
//! `PRED` from **NONMEM 7.6.0**, `$ESTIMATION MAXEVAL=0` (evaluation at fixed
//! structural parameters, `$OMEGA 0 FIX`), printed at full precision
//! (`$TABLE ... FORMAT=,1PE17.10`). One subject, a 100 mg infusion at `RATE=25`
//! into `CMT=2` (central, depot bypass) at `t=0`, observations on `CMT=2` at
//! `t = 1, 3, 6, 10, 16 h`. Disposition throughout: `CL=5, V/V2=50, KA=1`
//! (2-cpt adds `Q=3, V3=80`). The infusion runs `[0, F·AMT/RATE]`: `4 h` at
//! `F=1`, `2.4 h` at `F=0.6`. The obs grid straddles **both** ends, so the same
//! five times sit in the rising phase for one `F` and the decay phase for the
//! other — a curve that agrees at every point under both `F` values cannot be
//! matching by a rate/duration mix-up on one side.
//!
//! Control streams, datasets and verbatim NONMEM output are committed under
//! `nonmem_anchor/` as `oral_central_inf_advan{2,4}_f{1,06}.{ctl}` /
//! `oral_central_inf_advan{2,4}.csv` and
//! `nonmem_anchor/results/oral_central_inf_advan{2,4}_f{1,06}.{lst,tab}`.
//!
//! ## What is pinned
//!
//! 1. **Both analytical paths land on NONMEM.** Like
//!    `nonmem_dose_compartment_anchor.rs`, every case asserts the **event-driven
//!    walk** (forced by a varying `WT` column the model never reads) *and* the
//!    **default superposition dispatch** against NONMEM, and against each other.
//!    A path that silently dropped the central input rate (the #350 bug) or
//!    applied `F` on only one path fails here where the F=1-only self-check
//!    could not.
//! 2. **F reshapes the *duration*, not the rate (#419), on the depot-bypass
//!    central route.** NONMEM holds the data `RATE` and shortens the window to
//!    `F·AMT/RATE`; the `F=1` and `F=0.6` runs therefore share their `t=1` value
//!    (still infusing at the same rate) and diverge only past `2.4 h`. The
//!    1-cpt cases additionally assert ferx against the exact ADVAN1
//!    duration-scaled closed form — a **license-free** oracle that reproduces
//!    the committed NONMEM digits independently of NONMEM.

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

/// 1- or 2-cpt oral model with a bound bioavailability `F` (initial value
/// `tvf`). `F` is bound even when `tvf == 1.0`, so the `F=1` case exercises the
/// F codepath as an identity rather than skipping it. Structural parameters
/// match the control streams; `ETA_CL ~ 0` and evaluation at the initial thetas
/// reproduce `PRED` at `η = 0`.
fn oral_model(cpt: usize, tvf: f64) -> String {
    let (structural, extra_params, extra_indiv) = if cpt == 1 {
        (
            "pk one_cpt_oral(cl=CL, v=V, ka=KA, f=F)".to_string(),
            String::new(),
            String::new(),
        )
    } else {
        (
            "pk two_cpt_oral(cl=CL, v=V, q=Q, v2=V2, ka=KA, f=F)".to_string(),
            "  theta TVQ(3.0, 0.1, 50.0)\n  theta TVV2(80.0, 5.0, 500.0)".to_string(),
            "  Q  = TVQ\n  V2 = TVV2".to_string(),
        )
    };
    format!(
        r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
  theta TVF({tvf}, 0.001, 1.5)
{extra_params}
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA
  F  = TVF
{extra_indiv}

[structural_model]
  {structural}

[error_model]
  DV ~ proportional(PROP)
"#
    )
}

/// Predictions on the **event-driven walk**, selected by a `WT` column that
/// varies across rows. The model never references `WT`, so no parameter value
/// differs from the plain dataset — only the code path does: the reader flags a
/// varying non-model column as a covariate (`datareader.rs`), which makes
/// `Subject::has_tv_covariates()` true, and `compute_predictions_with_tv`
/// (`pk/mod.rs`) then routes an analytical + TV-covariate subject to
/// `event_driven_predictions` (the walk) rather than the superposition closed
/// form `plain_preds` uses.
///
/// That routing was confirmed empirically, not just read off the source: a
/// temporary ×1.5 scale injected into the walk's sole funnel
/// (`event_driven_predictions_run`) moved **only** this dataset — `walk_preds`
/// read 1.5× the true curve while `plain_preds` stayed exactly 1.0×, proving the
/// two go through different functions (the walk vs `predict_concentration`).
/// Re-run that differential injection to re-verify if the dispatch in
/// `compute_predictions_with_tv` is ever refactored — the F=1 walk value here is
/// ~0.476 (> 0), so a walk that silently fell back to superposition would still
/// pass this file, but a walk that regressed to the #350 ~0 curve would not.
fn walk_preds(src: &str, rate: f64) -> Vec<f64> {
    let model: CompiledModel = parse_full_model(src).expect("model parses").model;
    let dose = format!("1,0,.,1,100,2,{rate},1,70");
    let obs: String = [(1, 80), (3, 90), (6, 100), (10, 110), (16, 120)]
        .iter()
        .map(|(t, w)| format!("1,{t},1.0,0,.,2,0,0,{w}"))
        .collect::<Vec<_>>()
        .join("\n");
    let csv = format!("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT\n{dose}\n{obs}\n");
    preds_of(&model, &csv)
}

/// Predictions on the **plain** dataset — no time-varying covariate, so the
/// dispatcher's default (dose-superposition) route.
fn plain_preds(src: &str, rate: f64) -> Vec<f64> {
    let model: CompiledModel = parse_full_model(src).expect("model parses").model;
    let dose = format!("1,0,.,1,100,2,{rate}");
    let obs: String = [1, 3, 6, 10, 16]
        .iter()
        .map(|t| format!("1,{t},1.0,0,.,2,0"))
        .collect::<Vec<_>>()
        .join("\n");
    let csv = format!("ID,TIME,DV,EVID,AMT,CMT,RATE\n{dose}\n{obs}\n");
    preds_of(&model, &csv)
}

fn preds_of(model: &CompiledModel, csv: &str) -> Vec<f64> {
    predict(model, &pop_of(csv), &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect()
}

/// The obs grid, shared by the datasets, the NONMEM runs and the closed-form
/// oracle.
const OBS_T: [f64; 5] = [1.0, 3.0, 6.0, 10.0, 16.0];

/// 1-cpt and 2-cpt closed forms both come from a 2×2-or-smaller system (no cubic
/// root-finder to disagree on), so ferx and NONMEM agree essentially exactly:
/// the **measured** max relative gap across all four cases is ~2e-11 (1.98e-11
/// on `ADVAN4`, at the last printed digit of the 11-significant-figure `$TABLE`
/// reference). This bound is set just above that measured floor — tighter than
/// the sibling `nonmem_dose_compartment_anchor.rs`'s 1e-9 because that file's
/// 3-cpt cases carry a cubic root-finder these do not — so a real regression
/// (which would move a digit far above 2e-11) cannot hide in slack, while the
/// ~5× headroom keeps the reference's last-digit rounding from flaking. Probing
/// at 1e-11 fails (the floor); 1e-10 passes all five.
const NONMEM_TOL: f64 = 1e-10;

fn assert_close(got: &[f64], nonmem: &[f64], tol: f64, case: &str) {
    assert_eq!(got.len(), nonmem.len(), "{case}: row count");
    for (j, (&g, &n)) in got.iter().zip(nonmem).enumerate() {
        let rel = (g - n).abs() / n.abs().max(1e-12);
        assert!(
            rel < tol,
            "{case} obs {j} (t={}): ferx {g:.10} vs NONMEM {n:.10} (rel {rel:.2e})",
            OBS_T[j]
        );
    }
}

/// Assert both analytical paths reproduce NONMEM for the same central-infusion
/// row, and agree with each other.
fn assert_both_paths_match(src: &str, rate: f64, nonmem: &[f64], case: &str) {
    let walk = walk_preds(src, rate);
    let plain = plain_preds(src, rate);
    assert_close(
        &walk,
        nonmem,
        NONMEM_TOL,
        &format!("{case} [event-driven walk]"),
    );
    assert_close(
        &plain,
        nonmem,
        NONMEM_TOL,
        &format!("{case} [default dispatch]"),
    );
    assert_close(&plain, &walk, NONMEM_TOL, &format!("{case} [paths agree]"));
}

// ── NONMEM PRED (nonmem_anchor/results/oral_central_inf_*.tab) ────────────────

/// `oral_central_inf_advan2_f1.tab` — 1-cpt oral, `CMT=2` infusion, `F=1`.
const ADVAN2_F1: [f64; 5] = [
    4.7581290982e-01,
    1.2959088966e+00,
    1.3495955849e+00,
    9.0466097461e-01,
    4.9648846959e-01,
];
/// `oral_central_inf_advan2_f06.tab` — 1-cpt oral, `CMT=2` infusion, `F2=0.6`.
const ADVAN2_F06: [f64; 5] = [
    4.7581290982e-01,
    1.0047315645e+00,
    7.4432344989e-01,
    4.9893492919e-01,
    2.7382129479e-01,
];
/// `oral_central_inf_advan4_f1.tab` — 2-cpt oral, `CMT=2` infusion, `F=1`.
const ADVAN4_F1: [f64; 5] = [
    4.6222223658e-01,
    1.1951957279e+00,
    1.0956488990e+00,
    6.2303307734e-01,
    3.1054165440e-01,
];
/// `oral_central_inf_advan4_f06.tab` — 2-cpt oral, `CMT=2` infusion, `F2=0.6`.
const ADVAN4_F06: [f64; 5] = [
    4.6222223658e-01,
    9.0910738405e-01,
    5.7799283210e-01,
    3.3374887600e-01,
    1.7144103343e-01,
];

// ── 1. The pure #350 case: 1-cpt oral central infusion, F=1, direct anchor ────

#[test]
fn one_cpt_oral_central_infusion_f1_matches_nonmem() {
    assert_both_paths_match(
        &oral_model(1, 1.0),
        25.0,
        &ADVAN2_F1,
        "ADVAN2 CMT=2 infusion F=1",
    );
}

// ── 2. F≠1 on the depot-bypass central route (#327/#349/#419) ─────────────────

#[test]
fn one_cpt_oral_central_infusion_f06_matches_nonmem() {
    assert_both_paths_match(
        &oral_model(1, 0.6),
        25.0,
        &ADVAN2_F06,
        "ADVAN2 CMT=2 infusion F=0.6",
    );
}

// ── 3-4. 2-cpt oral central infusion — closes the gap the dose_cmt anchors left
//         (they cover ADVAN4 infusion into CMT=3, the peripheral, only) ────────

#[test]
fn two_cpt_oral_central_infusion_f1_matches_nonmem() {
    assert_both_paths_match(
        &oral_model(2, 1.0),
        25.0,
        &ADVAN4_F1,
        "ADVAN4 CMT=2 infusion F=1",
    );
}

#[test]
fn two_cpt_oral_central_infusion_f06_matches_nonmem() {
    assert_both_paths_match(
        &oral_model(2, 0.6),
        25.0,
        &ADVAN4_F06,
        "ADVAN4 CMT=2 infusion F=0.6",
    );
}

// ── 5. License-free oracle: ferx (and the committed NONMEM digits) reproduce the
//       exact ADVAN1 duration-scaled closed form for the 1-cpt cases ───────────

/// Central concentration of a 1-cpt model under a `RATE`-defined infusion into
/// central with bioavailability `F` applied as **duration scaling** (#419): the
/// data rate is held and the window is `[0, F·AMT/RATE]`. With an empty depot
/// this *is* the oral model's central curve, so it doubles as an
/// engine-independent reference for the anchors above.
fn iv_infusion_f_closed_form(t: f64, cl: f64, v: f64, amt: f64, rate: f64, f: f64) -> f64 {
    let ke = cl / v;
    let plateau = rate / (v * ke);
    let dur = f * amt / rate;
    if t <= dur {
        plateau * (1.0 - (-ke * t).exp())
    } else {
        let peak = plateau * (1.0 - (-ke * dur).exp());
        peak * (-ke * (t - dur)).exp()
    }
}

#[test]
fn one_cpt_central_infusion_matches_closed_form_and_nonmem() {
    for (tvf, nonmem) in [(1.0_f64, &ADVAN2_F1), (0.6_f64, &ADVAN2_F06)] {
        let oracle: Vec<f64> = OBS_T
            .iter()
            .map(|&t| iv_infusion_f_closed_form(t, 5.0, 50.0, 100.0, 25.0, tvf))
            .collect();
        // The committed NONMEM PRED reproduces the closed form (documents the
        // depot-bypass reduction independently of NONMEM's solver). The measured
        // gap is ~1.6e-11 — both sides are pure 1-cpt closed forms with no
        // root-finder — so this bound is the reference's last-printed-digit
        // resolution, not slack that could hide a real divergence.
        assert_close(
            nonmem,
            &oracle,
            NONMEM_TOL,
            &format!("NONMEM vs closed form F={tvf}"),
        );
        // ferx reproduces it too, on both analytical paths.
        assert_both_paths_match(
            &oral_model(1, tvf),
            25.0,
            &oracle,
            &format!("closed form F={tvf}"),
        );
    }
}
