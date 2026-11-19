//! #3004 — an ODE subject (the linear closed-form path: a matrix exponential) whose trial point
//! makes the Padé approximant's squaring overflow is a point the value path cannot score, so
//! its recording is repelled, not reported as a failed plan: the fit carries no
//! `W_PLAN_FD_FALLBACK`.
//!
//! The reproducer is `examples/warfarin_ode_bloq.ferx` (M3 censored rows) at two outer
//! iterations on two threads: one subject's inner search visits an η whose `exp(A)` squares past
//! `f64::MAX` (the iterate stays finite until the last squaring, so no recorded value is
//! non-finite) and before the fix that subject was counted as a plan failure
//! ("a repeat call's record-time run failed"). The unit-level pin, with its controls on the
//! other side of each gate, is `an_overflowing_matrix_or_repeat_call_is_repelled_…` in
//! `src/parser/kernel_jit_plan_tests.rs`.
//!
//! Tier 2: two outer iterations.

use ferx_core::api::run_model_with_data;
use std::io::Write;

#[test]
fn an_overflowing_matrix_exponential_does_not_report_a_failed_plan() {
    let root = env!("CARGO_MANIFEST_DIR");
    let text = std::fs::read_to_string(format!("{root}/examples/warfarin_ode_bloq.ferx"))
        .expect("the example is committed");
    let anchor = "  maxiter      = 300\n";
    assert!(text.contains(anchor), "fixture: the fit options line");
    let text = text.replace(anchor, "  maxiter      = 2\n  threads      = 2\n");
    let mut f = tempfile::Builder::new()
        .suffix(".ferx")
        .tempfile()
        .expect("temp model");
    f.write_all(text.as_bytes()).expect("write model");
    let data = format!("{root}/data/warfarin.csv");
    let (fit, _) = run_model_with_data(f.path().to_str().unwrap(), Some(&data)).expect("fit runs");
    assert!(
        !fit.warnings
            .iter()
            .any(|w| w.contains("W_PLAN_FD_FALLBACK")),
        "no failed plan at an overflowed trial point: {:?}",
        fit.warnings
    );
}
