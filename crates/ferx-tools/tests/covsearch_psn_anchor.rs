//! The PsN `scm` anchor for covsearch (#1180): the same dataset, base model
//! and test relations, run through PsN 5.7.1 + NONMEM 7.6 and through ferx,
//! must walk the **same trajectory** — the same effect added at each forward
//! step, the same decision at the backward step, the same final relation set
//! — with every candidate's OFV within tolerance of NONMEM's.
//!
//! The PsN side is committed under `tests/psn/scm_anchor/`: `base.mod` and
//! `scm.conf` are what PsN ran, `scmlog.txt` is what it wrote, and
//! `base.ferx` / `scm.ferxsearch` are their ferx twins. The numbers below are
//! transcribed from `scmlog.txt`.
//!
//! Why `p = 0.05` in both directions: at PsN's usual `p_forward = 0.01` the
//! first step's runner-up, CL-WT, misses the 6.63 cutoff by 0.04 OFV, and at
//! `p_backward = 0.001` the winner is removed again — a trajectory decided by
//! numbers finer than the agreement between two engines can be, which is not
//! an anchor. At 0.05 every decision has a margin of at least 1.7 OFV except
//! the step-1 *ordering* (CL-CRCL's drop beats CL-WT's by 0.165), which is
//! asserted too: ferx's OFVs agree with NONMEM's to 1e-4 here.
//!
//! Slow: nine FOCEI fits of a two-compartment model, three starts each.

use std::path::Path;

use ferx_tools::covsearch::{run_covsearch, CovsearchRun, Phase};
use ferx_tools::search::SearchConfig;

const ANCHOR_DIR: &str = "tests/psn/scm_anchor";

/// `scmlog.txt`, step 1: every candidate against the base model.
const PSN_BASE_OFV: f64 = -696.79434;
const PSN_STEP1: &[(&str, &str, f64)] = &[
    ("CL", "CRCL", -703.55245),
    ("CL", "WT", -703.38740),
    ("V1", "CRCL", -697.02773),
    ("V1", "WT", -698.37971),
];
/// Step 2, from CL-CRCL.
const PSN_STEP2: &[(&str, &str, f64)] = &[
    ("CL", "WT", -709.14359),
    ("V1", "CRCL", -703.79309),
    ("V1", "WT", -705.13444),
];
/// Step 3, from CL-CRCL + CL-WT: nothing significant.
const PSN_STEP3: &[(&str, &str, f64)] = &[("V1", "CRCL", -709.38002), ("V1", "WT", -710.72213)];
/// The backward step: removing either effect costs more than 3.84.
const PSN_BACKWARD: &[(&str, &str, f64)] = &[("CL", "CRCL", -703.38740), ("CL", "WT", -703.55245)];

/// NONMEM and ferx agree on these fits to about 1e-4; the tolerance is
/// loose enough for a different optimizer path and tight enough that a
/// wrong centre, a wrong θ init or a wrong parameter count would fail it.
const OFV_TOL: f64 = 0.05;

fn check_step(
    result: &ferx_tools::covsearch::CovsearchResult,
    step: usize,
    phase: Phase,
    expected: &[(&str, &str, f64)],
    selected: Option<(&str, &str)>,
) {
    let rows: Vec<_> = result.step_rows(step).collect();
    assert_eq!(rows.len(), expected.len(), "step {step}: {rows:#?}");
    for (parameter, covariate, ofv) in expected {
        let row = rows
            .iter()
            .find(|r| r.effect.parameter == *parameter && r.effect.covariate == *covariate)
            .unwrap_or_else(|| panic!("step {step}: no row for {parameter}-{covariate}"));
        assert_eq!(row.phase, phase);
        let got = row.ofv.expect("every candidate fitted");
        assert!(
            (got - ofv).abs() < OFV_TOL,
            "step {step} {parameter}-{covariate}: ferx OFV {got} vs NONMEM {ofv}"
        );
        assert!(
            row.passed,
            "step {step} {parameter}-{covariate}: {:?}",
            row.failures
        );
        assert_eq!(row.converged, Some(true));
        assert_eq!(row.lrt.unwrap().df, 1);
        assert_eq!(
            row.selected,
            selected == Some((parameter, covariate)),
            "step {step} {parameter}-{covariate}: selected={}",
            row.selected
        );
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: nine FOCEI fits; opt in with --features slow-tests"
)]
fn ferx_covsearch_walks_psn_scms_trajectory() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(ANCHOR_DIR);
    let config = SearchConfig::load(dir.join("scm.ferxsearch")).expect("anchor config");
    let base = config.load_base().expect("anchor base model + data");
    let out = tempfile::tempdir().expect("tempdir");
    let result = run_covsearch(
        &config,
        &base,
        CovsearchRun {
            dir: Some(out.path().to_path_buf()),
            ..CovsearchRun::default()
        },
    )
    .expect("search");

    assert!(
        (result.base_ofv - PSN_BASE_OFV).abs() < OFV_TOL,
        "base OFV {} vs NONMEM {PSN_BASE_OFV}",
        result.base_ofv
    );

    // Forward: CL-CRCL, then CL-WT, then nothing.
    check_step(&result, 1, Phase::Forward, PSN_STEP1, Some(("CL", "CRCL")));
    check_step(&result, 2, Phase::Forward, PSN_STEP2, Some(("CL", "WT")));
    check_step(&result, 3, Phase::Forward, PSN_STEP3, None);
    // Backward: both removals cost more than the cutoff, so both stay.
    check_step(&result, 4, Phase::Backward, PSN_BACKWARD, None);
    assert_eq!(result.n_steps(), 4);

    // The same final relation set, in the order it was built.
    let included: Vec<(String, String, &str)> = result
        .included
        .iter()
        .map(|i| {
            (
                i.effect.parameter.clone(),
                i.effect.covariate.clone(),
                i.effect.form_label(),
            )
        })
        .collect();
    assert_eq!(
        included,
        vec![
            ("CL".into(), "CRCL".into(), "power"),
            ("CL".into(), "WT".into(), "power")
        ]
    );
    assert_eq!(result.final_step, 2);
    assert!((result.final_ofv - -709.14359).abs() < OFV_TOL);

    // The p-values PsN printed, at the resolution it printed them.
    let step1: Vec<_> = result.step_rows(1).collect();
    let p = |parameter: &str, covariate: &str| {
        step1
            .iter()
            .find(|r| r.effect.parameter == parameter && r.effect.covariate == covariate)
            .unwrap()
            .p_value()
    };
    assert!((p("CL", "CRCL") - 0.009332).abs() < 1e-4);
    assert!((p("CL", "WT") - 0.010238).abs() < 1e-4);
    assert!((p("V1", "WT") - 0.207990).abs() < 1e-3);
    assert!((p("V1", "CRCL") - 0.629030).abs() < 1e-3);
}
