//! #3004 — a CTMM subject whose trial point the value path repels (its `1e20` sentinel: a rate
//! that overflows the `expm` argument, a probability that underflows) is a repelled point, not a
//! failed plan: the fit carries no `W_PLAN_FD_FALLBACK`.
//!
//! The reproducer is `examples/ctmm_pd_2state.ferx` (a drug-driven generator) started where
//! every subject's trial point is repelled: `TVV = 1`, `SLOPE = 4` put the intensity at
//! `exp(-5 + 4·C)` with `C` of order 100. Before the fix each of the 60 subjects was marked bad on
//! the first evaluation ("the recorder returned None") and reported as a plan failure; on the
//! default start the same mechanism hit 23 of 60 subjects once an L-BFGS line search stepped into
//! such a region. The unit-level pins, with their controls at an ordinary point, are
//! `a_repelled_*_trial_point_does_not_fail_the_subjects_plan` in `src/markov/endpoint.rs`.
//!
//! Tier 2: two outer iterations.
#![cfg(feature = "markov")]

use ferx_core::api::run_model_with_data;
use std::io::Write;

#[test]
fn a_repelled_ctmm_trial_point_does_not_report_a_failed_plan() {
    let root = env!("CARGO_MANIFEST_DIR");
    let mut text = std::fs::read_to_string(format!("{root}/examples/ctmm_pd_2state.ferx"))
        .expect("the example is committed");
    for (from, to) in [
        ("theta TVCL(1.0,", "theta TVCL(0.05,"),
        ("theta TVV(10.0,", "theta TVV(1.0,"),
        ("theta LQ01(-1.2,", "theta LQ01(-5.0,"),
        ("theta SLOPE(0.15,", "theta SLOPE(4.0,"),
        ("  maxiter = 300\n", "  maxiter = 2\n  threads = 1\n"),
    ] {
        assert!(text.contains(from), "fixture: {from}");
        text = text.replace(from, to);
    }
    let mut f = tempfile::Builder::new().suffix(".ferx").tempfile().expect("temp model");
    f.write_all(text.as_bytes()).expect("write model");
    let data = format!("{root}/data/ctmm_pd_2state.csv");
    let (fit, _) = run_model_with_data(f.path().to_str().unwrap(), Some(&data)).expect("fit runs");
    assert!(fit.ofv >= 1e19, "fixture: the start is a repelled point ({})", fit.ofv);
    assert!(
        !fit.warnings.iter().any(|w| w.contains("W_PLAN_FD_FALLBACK")),
        "no failed plan at a repelled trial point: {:?}",
        fit.warnings
    );
}
