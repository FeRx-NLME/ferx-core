//! NONMEM `ADVAN5` anchor for the exact linear-ODE path (#852).
//!
//! The model is #852's motivating structure — `depot → trans1 → trans2 → liver ⇄ central ⇄
//! periph` — written on the NONMEM side as `ADVAN5` rate constants and on the ferx side as an
//! ordinary `[odes]` block, which the parser certifies linear so `ode_method = auto` solves it
//! by the matrix exponential. Two properties make it a hard fixture rather than a convenient
//! one:
//!
//! - `K12 = K23 = K34 = KTR`: one transit rate three times, so the system matrix is
//!   **defective** (a single Jordan block). The `t·e^{−kt}` / `t²·e^{−kt}` terms it produces are
//!   what an eigendecomposition-based solver gets wrong; ferx's Padé exponential never
//!   diagonalizes.
//! - The liver drains at `(QH + CLINT)/V4 ≈ 107/h` against a terminal phase near 0.02/h — a
//!   stiffness ratio in the thousands, which is what made this model slow on an integrator.
//!
//! Every dose event keeps the quantity under test live on both sides: subject 1 takes three
//! depot doses q12h that land on residual drug, with `F1 = 0.8` and `ALAG1 = 0.3`; subject 2 a
//! steady-state depot dose followed by an ordinary one; subject 3 two central infusions that
//! land on residual drug; subject 4 a steady-state central infusion sampled across the interval.
//!
//! ## Reference values
//!
//! **NONMEM 7.6.0**, `$ESTIMATION METHOD=1 INTER MAXEVAL=0 POSTHOC`, parameters at the
//! `$THETA`/`$OMEGA`/`$SIGMA` initial values. DVs were simulated by NONMEM itself from the same
//! model (`advan5_liver_sim.ctl`, seed 852). Files: `nonmem_anchor/advan5_liver{.ctl,_sim.ctl,
//! .csv}` and the verbatim output `nonmem_anchor/results/advan5_liver{.lst,.tab}`. (NONMEM
//! 7.5.1 segfaults on the evaluation in the `pharma_ai_r` image; the simulation runs on either.)
//!
//! Tier 2: one `predict()` and one evaluation-only `fit()` on four subjects; not gated.

use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::FitOptions;
use ferx_core::{fit, predict, read_nonmem_csv};

const DATA: &str = "nonmem_anchor/advan5_liver.csv";
const TABLE: &str = "nonmem_anchor/results/advan5_liver.tab";

/// NONMEM `OBJECTIVE FUNCTION VALUE` (`advan5_liver.lst`).
const NM_OFV: f64 = -340.39027326310884;

/// The same model as `advan5_liver.ctl`, `K` constants written out as equations. `{tail}` is
/// appended to `[fit_options]`.
fn model(tail: &str) -> String {
    format!(
        "[parameters]
  theta TVKTR(1.2, 0.01, 50)
  theta TVCLINT(20, 0.1, 500)
  theta TVQH(17.4, 1, 100)
  theta TVV4(0.35, 0.01, 10)
  theta TVV(30, 1, 500)
  theta TVQ(5, 0.1, 100)
  theta TVVP(200, 1, 2000)
  theta TVF1(0.8, 0.01, 1)
  theta TVALAG1(0.3, 0, 5)
  omega ETA_CLINT ~ 0.09
  omega ETA_KTR ~ 0.04
  sigma PROP ~ 0.01

[individual_parameters]
  KTR   = TVKTR * exp(ETA_KTR)
  CLINT = TVCLINT * exp(ETA_CLINT)
  QH    = TVQH
  V4    = TVV4
  V     = TVV
  Q     = TVQ
  VP    = TVVP
  F1    = TVF1
  ALAG1 = TVALAG1

[structural_model]
  ode(obs_cmt=central, states=[depot, trans1, trans2, liver, central, periph])

[odes]
  d/dt(depot)  = -KTR * depot
  d/dt(trans1) =  KTR * depot  - KTR * trans1
  d/dt(trans2) =  KTR * trans1 - KTR * trans2
  d/dt(liver)  =  KTR * trans2 - (QH / V4) * liver - (CLINT / V4) * liver + (QH / V) * central
  d/dt(central) = (QH / V4) * liver - (QH / V) * central - (Q / V) * central + (Q / VP) * periph
  d/dt(periph) =  (Q / V) * central - (Q / VP) * periph

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = focei
{tail}
"
    )
}

/// `(ID, TIME, EVID, PRED, ETA1, ETA2)` rows of the committed NONMEM table.
fn nonmem_rows() -> Vec<[f64; 6]> {
    let text = std::fs::read_to_string(TABLE).expect("the NONMEM table is committed");
    text.lines()
        .skip(2) // `TABLE NO.  1` and the column header
        .map(|l| {
            let v: Vec<f64> = l
                .split(',')
                .map(|x| x.trim().parse().expect("numeric table cell"))
                .collect();
            // ID TIME EVID PRED IPRED ETA1 ETA2
            [v[0], v[1], v[2], v[3], v[5], v[6]]
        })
        .collect()
}

/// Worst relative PRED (η = 0) difference from NONMEM over every observation.
fn worst_pred_error(tail: &str) -> f64 {
    let parsed = parse_full_model(&model(tail)).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let ferx = predict(&parsed.model, &pop, &parsed.model.default_params).unwrap();
    let nm: Vec<[f64; 6]> = nonmem_rows().into_iter().filter(|r| r[2] == 0.0).collect();
    assert_eq!(
        ferx.len(),
        nm.len(),
        "one prediction per observation record"
    );
    assert_eq!(nm.len(), 33, "the anchor must compare all 33 observations");
    nm.iter().zip(&ferx).fold(0.0_f64, |worst, (want, got)| {
        assert_eq!(got.id, format!("{}", want[0] as i64));
        assert!((got.time - want[1]).abs() < 1e-9, "row order");
        // `f64::max` would discard a NaN; a broken solver must fail here instead.
        assert!(
            got.pred.is_finite() && want[3].is_finite(),
            "non-finite PRED: ferx {} vs NONMEM {:?}",
            got.pred,
            want
        );
        worst.max((got.pred - want[3]).abs() / want[3].abs())
    })
}

#[test]
fn advan5_pred_matches_nonmem_on_every_record() {
    // Measured worst over all 33 observations: 3.1e-12 — the `$TABLE` print precision
    // (`1PE20.12`). The bound is 30× that.
    let rel = worst_pred_error("");
    assert!(
        rel < 1e-10,
        "PRED vs NONMEM ADVAN5: worst relative error {rel:e}"
    );
    // The bound's teeth: the same model integrated (`rk45`, default tolerances) misses NONMEM
    // by 2.6e-5, five orders outside it — so this test cannot pass on a route that silently
    // fell back to the integrator.
    let integrated = worst_pred_error("  ode_method = rk45");
    assert!(
        integrated > 1e3 * 1e-10,
        "the rk45 twin should carry integrator error, got {integrated:e}"
    );
}

#[test]
fn advan5_focei_objective_and_ebes_match_nonmem() {
    let parsed = parse_full_model(&model("")).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    }
    .quiet();
    let result = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(
        result.ode_linear_exact,
        "the fixture must run on the exact linear path, or this anchors the integrator"
    );
    // Measured: ferx −340.390273316, NONMEM −340.390273263 — a difference of 5.2e-8. Bound
    // 200× that, and still far inside the 1e-3 the other FOCEI anchors use.
    assert!(
        (result.ofv - NM_OFV).abs() < 1e-5,
        "FOCEI OFV: ferx {}, NONMEM {NM_OFV}",
        result.ofv
    );

    // NONMEM's POSTHOC η, on each subject's records. Measured worst absolute difference
    // 3.0e-8; bound 300× that.
    let nm = nonmem_rows();
    assert_eq!(result.subjects.len(), 4);
    for s in &result.subjects {
        let row = nm
            .iter()
            .find(|r| format!("{}", r[0] as i64) == s.id)
            .expect("subject in table");
        for (k, want) in [row[4], row[5]].into_iter().enumerate() {
            assert!(
                (s.eta[k] - want).abs() < 1e-5,
                "subject {} η{}: ferx {}, NONMEM {want}",
                s.id,
                k + 1,
                s.eta[k]
            );
        }
    }
}
