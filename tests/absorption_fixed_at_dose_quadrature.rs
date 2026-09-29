//! #1569 — the transit and inverse-Gaussian ODE twins under IOV and a time-varying covariate,
//! against an independent quadrature of the **fixed-at-dose** absorption rule.
//!
//! `pk two_cpt_transit` / `pk two_cpt_ig` with IOV on `CL`, both kernel parameters (`n`/`MTT`,
//! `MAT`/`CV2`) and `F`, a lag of 0.7, `WT` on `CL` changing mid-occasion, and four overlapping
//! q6h doses. Occasion 2 begins **on a dose record** (`t = 12`), occasion 3 **on an observation
//! record** (`t = 16.5`), each while an earlier dose is still absorbing. The reference is
//! `tests/data/absorption_fixed_at_dose_quadrature.py` — pure-stdlib Python, no ferx or NONMEM
//! code: a 2×2 spectral propagator per segment with each dose's input integrated by tanh-sinh
//! quadrature, self-checked against an independent fixed-step RK4 (≤ 2.9e-10) and a mass ledger
//! (`320.800666888905 / 320.800666888929` at `t = 400` for transit). It tabulates two rules for
//! an in-flight dose's kernel:
//!
//! * `fixed_at_dose` — each dose keeps the kernel of its own dose record (#1569);
//! * `current_interval` — every open dose uses the governing record's (ferx before #1569).
//!
//! The rules differ by up to 32 % (transit) / 15 % (IG) on `full` and coincide exactly on
//! `cl_only` (IOV on `CL` only). Both straddles are asserted on the committed file itself, so
//! neither leg can silently drift to the other side. Measured on the pre-#1569 engine
//! (`835044be`), the transit twin matched the `current_interval` column to 3.7e-12; it now
//! matches `fixed_at_dose` to the same order.
//!
//! Since #1560 the closed-form model's IOV subjects run on the exact absorption walk, not the
//! twin. Both engines are checked here against the same table: the plain model (the walk) and
//! a copy whose `CL` carries the inert factor `(1 + 0·TIME)` (a `TIME` read is outside the
//! walk, so the twin serves it). The two legs must not be bit-identical, which keeps either
//! from quietly running the other's engine.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, Subject};
use std::collections::HashMap;

/// The model for `kernel`: its structural line and the names of its two kernel parameters.
/// `twin` multiplies `CL` by the inert `(1 + 0·TIME)`, which routes it to the ODE twin.
fn model_src(kernel: &str, twin: bool) -> String {
    let time_read = if twin { " * (1 + 0 * TIME)" } else { "" };
    let (thetas, params, pk) = match kernel {
        "transit" => (
            "  theta TVA(3.0, 0.0, 30.0)\n  theta TVB(2.5, 0.05, 24.0)",
            "  NTR = TVA * exp(KAPPA_A)\n  MTT = TVB * exp(KAPPA_B)",
            "pk two_cpt_transit(cl=CL, v1=V1, q=Q, v2=V2, n=NTR, mtt=MTT, f=FB, lagtime=LAG)",
        ),
        "ig" => (
            "  theta TVA(2.5, 0.05, 24.0)\n  theta TVB(0.5, 0.01, 5.0)",
            "  MAT = TVA * exp(KAPPA_A)\n  CV2 = TVB * exp(KAPPA_B)",
            "pk two_cpt_ig(cl=CL, v1=V1, q=Q, v2=V2, mat=MAT, cv2=CV2, f=FB, lagtime=LAG)",
        ),
        other => panic!("unknown kernel {other}"),
    };
    format!(
        r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(30.0, 1.0, 500.0)
  theta TVQ(3.0, 0.1, 100.0)
  theta TVV2(60.0, 1.0, 1000.0)
{thetas}
  theta TVF(0.8, 0.01, 1.0)
  theta TVLAG(0.7, 0.0, 5.0)
  omega ETA_V ~ 0.09
  kappa KAPPA_CL ~ 0.04
  kappa KAPPA_A ~ 0.04
  kappa KAPPA_B ~ 0.04
  kappa KAPPA_F ~ 0.04
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL  = TVCL * (WT/70)^0.75 * exp(KAPPA_CL){time_read}
  V1  = TVV1 * exp(ETA_V)
  Q   = TVQ
  V2  = TVV2
{params}
  FB  = TVF * exp(KAPPA_F)
  LAG = TVLAG
[structural_model]
  {pk}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-14
"#
    )
}

const ETA_V: f64 = 0.1;

/// `(time, occasion, WT)` per dose (100 each, into compartment 1) and per observation — the
/// generator's `DOSES` / `OBS`.
const DOSES: [(f64, u32, f64); 4] = [
    (0.0, 1, 70.0),
    (6.0, 1, 72.0),
    (12.0, 2, 74.0),
    (18.0, 3, 80.0),
];
const OBS: [(f64, u32, f64); 15] = [
    (1.0, 1, 70.0),
    (2.5, 1, 70.0),
    (4.0, 1, 71.0),
    (5.5, 1, 72.0),
    (7.5, 1, 72.0),
    (9.0, 1, 73.0),
    (11.0, 1, 73.0),
    (13.5, 2, 74.0),
    (15.0, 2, 76.0),
    (16.5, 3, 77.0),
    (19.5, 3, 80.0),
    (22.0, 3, 82.0),
    (26.0, 3, 84.0),
    (36.0, 3, 86.0),
    (48.0, 3, 88.0),
];

/// κ per occasion, in declaration order `[KAPPA_CL, KAPPA_A, KAPPA_B, KAPPA_F]` — the
/// generator's `FULL` / `CL_ONLY`.
fn kappas(scenario: &str) -> Vec<Vec<f64>> {
    let (kcl, ka, kb, kf) = match scenario {
        "full" => (
            [0.10, -0.15, 0.20],
            [0.00, 0.25, -0.20],
            [0.20, -0.30, 0.35],
            [0.00, -0.10, 0.10],
        ),
        "cl_only" => ([0.10, -0.15, 0.20], [0.0; 3], [0.0; 3], [0.0; 3]),
        other => panic!("unknown scenario {other}"),
    };
    (0..3).map(|o| vec![kcl[o], ka[o], kb[o], kf[o]]).collect()
}

fn subject() -> Subject {
    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let n = OBS.len();
    Subject {
        id: "1".into(),
        doses: DOSES
            .iter()
            .map(|&(t, _, _)| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0))
            .collect(),
        obs_times: OBS.iter().map(|&(t, _, _)| t).collect(),
        obs_raw_times: vec![],
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        covariates: wt(70.0),
        dose_covariates: DOSES.iter().map(|&(_, _, w)| wt(w)).collect(),
        obs_covariates: OBS.iter().map(|&(_, _, w)| wt(w)).collect(),
        pk_only_times: vec![],
        pk_only_covariates: vec![],
        reset_times: vec![],
        reset_covariates: vec![],
        cens: vec![0; n],
        occasions: OBS.iter().map(|&(_, o, _)| o).collect(),
        obs_l2: vec![],
        dose_occasions: DOSES.iter().map(|&(_, o, _)| o).collect(),
        reset_occasions: vec![],
        fremtype: vec![],
        obs_records: vec![],
    }
}

/// `scenario → [(time, fixed_at_dose, current_interval)]`, in observation order.
fn reference() -> HashMap<String, Vec<(f64, f64, f64)>> {
    let text = std::fs::read_to_string("tests/data/absorption_fixed_at_dose_quadrature.csv")
        .expect("the frozen quadrature table loads");
    let mut out: HashMap<String, Vec<(f64, f64, f64)>> = HashMap::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 5, "malformed row: {line}");
        let parse = |s: &str| -> f64 { s.parse().expect("a float") };
        out.entry(f[0].to_string())
            .or_default()
            .push((parse(f[2]), parse(f[3]), parse(f[4])));
    }
    out
}

#[test]
fn walk_and_twin_match_the_fixed_at_dose_quadrature_under_iov_and_a_tv_covariate() {
    let table = reference();
    let subject = subject();
    for kernel in ["transit", "ig"] {
        let mut legs: Vec<Vec<f64>> = Vec::new();
        for (leg, twin) in [("walk", false), ("twin", true)] {
            let model =
                parse_model_string(&model_src(kernel, twin)).expect("the fixture model parses");
            assert!(
                model.ode_spec.is_none() && model.effective_for(&subject).ode_spec.is_some(),
                "{kernel} {leg}: a closed-form IOV model with an ODE twin"
            );
            let theta = model.default_params.theta.clone();
            for rule_set in ["full", "cl_only"] {
                let scenario = format!("{kernel}_{rule_set}");
                let rows = &table[&scenario];
                assert_eq!(rows.len(), OBS.len(), "{scenario}: one row per observation");
                let got = ferx_core::pk::predict_iov(
                    &model,
                    &subject,
                    &theta,
                    &[ETA_V],
                    &kappas(rule_set),
                );
                let mut worst = 0.0_f64;
                let mut vs_current = 0.0_f64;
                let mut split = 0.0_f64;
                for (j, (&(t, fixed, current), &pred)) in rows.iter().zip(&got).enumerate() {
                    assert_eq!(
                        t, OBS[j].0,
                        "{scenario} row {j}: the table's time must be the design's"
                    );
                    assert!(
                        pred.is_finite() && fixed.is_finite() && current.is_finite(),
                        "{scenario} {leg} obs {j}: non-finite value (ferx {pred}, table \
                         {fixed} / {current})"
                    );
                    worst = worst.max((pred - fixed).abs() / fixed.abs());
                    vs_current = vs_current.max((pred - current).abs() / current.abs());
                    split = split.max((fixed - current).abs() / fixed.abs());
                }
                println!(
                    "#1569/#1560 quadrature {scenario} ({leg}): max rel |ferx − fixed_at_dose| \
                     {worst:.3e} (vs current_interval {vs_current:.3e}); rules split {split:.3e}"
                );
                // The straddle, on the committed file itself: the two rules must differ
                // materially on `full` (else the fixture could not tell them apart) and
                // coincide on `cl_only` (the arm that must be unaffected by the change).
                if rule_set == "full" {
                    assert!(
                        split > 0.1,
                        "{scenario} no longer separates the two rules ({split:.3e})"
                    );
                } else {
                    assert!(
                        split == 0.0,
                        "{scenario}: with IOV on CL alone the two rules must coincide exactly \
                         ({split:.3e})"
                    );
                }
                // Realised ≤ 3.7e-12 on the twin at `ode_reltol = 1e-12` and ≤ 8.7e-16 on
                // the walk; the bound carries ~27x over the worse and sits nine orders inside
                // the smallest split (15 %).
                assert!(
                    worst < 1e-10,
                    "{scenario} ({leg}): ferx departs from the fixed-at-dose quadrature by \
                     {worst:.3e}"
                );
                legs.push(got);
            }
        }
        // [walk full, walk cl_only, twin full, twin cl_only]: two engines, never one bit
        // pattern.
        let differ = legs[0..2]
            .iter()
            .flatten()
            .zip(legs[2..4].iter().flatten())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differ > 0,
            "{kernel}: walk and twin legs are bit-identical — one ran the other's engine"
        );
    }
}
