//! Regression test for issue #195 — multiple `EVID=4` (reset + dose) occasions
//! stacked under one subject ID, each with its TIME column restarting at 0.
//!
//! NONMEM processes records sequentially, so a second `EVID=4` whose TIME
//! restarts at 0 opens a fresh dosing occasion that reuses the first
//! occasion's wall-clock. ferx-core represents each subject as a single
//! absolute-time event list and sorts it, which (before this fix) interleaved
//! the two occasions: both doses landed at t=0 and the subject was silently
//! double-dosed. For the Schnider propofol dataset in the issue this inflated
//! CL by ~1.8x versus NONMEM.
//!
//! The reader now shifts each restarting occasion past the previous one onto a
//! single monotonic timeline (the reset zeros all compartments at the
//! boundary, so no drug carries across). This test exercises the public path
//! parser → NONMEM CSV reader → `predict()` and asserts:
//!
//!   1. A subject given one infusion occasion and a subject given the SAME
//!      occasion twice (each opened by `EVID=4`) produce identical population
//!      predictions per occasion — i.e. no double-dosing.
//!   2. The second occasion reproduces the first exactly, confirming the reset
//!      zeros state at the occasion boundary rather than carrying drug over.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(20.0, 1.0, 200.0)

  omega ETA_CL ~ 0.0

  sigma PROP_ERR ~ 0.01 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// ID 1: a single infusion occasion (AMT=100 over 1 time unit), obs at 1,2,4,8.
// ID 2: the IDENTICAL occasion given twice, each opened by EVID=4 with TIME
//       restarting at 0 — the issue-#195 pattern.
const DATA: &str = "ID,TIME,DV,EVID,AMT,RATE\n\
                    1,0,.,1,100,100\n\
                    1,1,1.0,0,.,.\n\
                    1,2,1.0,0,.,.\n\
                    1,4,1.0,0,.,.\n\
                    1,8,1.0,0,.,.\n\
                    2,0,.,4,100,100\n\
                    2,1,1.0,0,.,.\n\
                    2,2,1.0,0,.,.\n\
                    2,4,1.0,0,.,.\n\
                    2,8,1.0,0,.,.\n\
                    2,0,.,4,100,100\n\
                    2,1,1.0,0,.,.\n\
                    2,2,1.0,0,.,.\n\
                    2,4,1.0,0,.,.\n\
                    2,8,1.0,0,.,.\n";

#[test]
fn reset_occasions_are_not_double_dosed() {
    let parsed = parse_full_model(MODEL).expect("model parses");
    let model = parsed.model;

    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(DATA.as_bytes()).unwrap();
    let population = read_nonmem_csv(f.path(), None, None).expect("dataset loads");

    // Sanity: the two-occasion subject carries two resets, segmented onto a
    // monotonic timeline (the second occasion shifted past the first).
    let subj2 = &population.subjects[1];
    assert_eq!(
        subj2.reset_times.len(),
        2,
        "subject 2 has two reset occasions"
    );
    assert!(
        subj2.obs_times.windows(2).all(|w| w[1] > w[0]),
        "subject 2 obs times must be monotonic across the occasion boundary: {:?}",
        subj2.obs_times
    );

    let preds = predict(&model, &population, &model.default_params);

    // Split predictions by subject id.
    let single: Vec<f64> = preds
        .iter()
        .filter(|p| p.id == "1")
        .map(|p| p.pred)
        .collect();
    let double: Vec<f64> = preds
        .iter()
        .filter(|p| p.id == "2")
        .map(|p| p.pred)
        .collect();

    assert_eq!(single.len(), 4, "single-occasion subject has 4 obs");
    assert_eq!(
        double.len(),
        8,
        "two-occasion subject has 8 obs (4 per occasion)"
    );

    // Each prediction is positive (the infusion produced drug).
    assert!(single.iter().all(|&c| c > 0.0));

    // Occasion 1 of the stacked subject must equal the single-occasion subject:
    // a leading EVID=4 reset on an empty system is a no-op, so the dosing is
    // identical — and crucially NOT doubled.
    for (j, (&d, &s)) in double[..4].iter().zip(&single).enumerate() {
        let rel = (d - s).abs() / s;
        assert!(
            rel < 1e-9,
            "occasion 1 obs {j}: stacked PRED {d:.8} != single-occasion PRED {s:.8} \
             (rel {rel:.2e}) — second dose is leaking into occasion 1 (double-dosing)"
        );
    }

    // Occasion 2 must reproduce occasion 1 exactly: the EVID=4 reset zeros
    // state at the boundary, so the second identical occasion sees a clean
    // system — not residual drug from occasion 1.
    for (j, (&d2, &d1)) in double[4..].iter().zip(&double[..4]).enumerate() {
        let rel = (d2 - d1).abs() / d1;
        assert!(
            rel < 1e-9,
            "occasion 2 obs {j}: PRED {d2:.8} != occasion 1 PRED {d1:.8} (rel {rel:.2e}) \
             — reset did not zero state at the occasion boundary"
        );
    }
}
