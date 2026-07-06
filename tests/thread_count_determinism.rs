//! Regression test for #703: a fit's objective must not depend on the rayon
//! worker-thread count.
//!
//! The per-subject FOCE/FOCEI likelihood is summed across subjects. The bug was
//! a parallel `ParallelIterator::sum` whose reduction tree is split along
//! boundaries that depend on the worker count; because f64 addition is not
//! associative, the OFV differed between (e.g.) 4 and 15 threads. In a
//! non-converged run those ULP-level differences steer the optimizer down
//! divergent trajectories, giving visibly different OFVs. The fix collects the
//! per-subject NLLs in subject order and sums them serially, so the objective is
//! bit-reproducible regardless of thread count.
//!
//! `FitOptions::threads` runs the whole fit inside a scoped rayon pool of the
//! requested size, so this exercises the exact user-facing knob from the issue.

use std::path::Path;

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};

fn warfarin() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn run_with_threads(n: usize, outer_maxiter: usize) -> f64 {
    let (model, population) = warfarin();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.outer_maxiter = outer_maxiter;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.threads = Some(n);
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("FOCEI fit must succeed");
    result.ofv
}

/// The same FOCEI fit under 1, 4, and 15 worker threads must produce a
/// bit-identical OFV. A short-but-nontrivial `outer_maxiter` keeps the run fast
/// while giving the optimizer enough iterations that any thread-dependent
/// rounding in the objective would have driven the trajectories apart.
#[test]
fn focei_ofv_is_independent_of_thread_count() {
    let ofv_1 = run_with_threads(1, 40);
    let ofv_4 = run_with_threads(4, 40);
    let ofv_15 = run_with_threads(15, 40);

    assert!(ofv_1.is_finite(), "OFV must be finite, got {ofv_1}");
    assert_eq!(
        ofv_1.to_bits(),
        ofv_4.to_bits(),
        "OFV differs between 1 and 4 threads: {ofv_1} vs {ofv_4}"
    );
    assert_eq!(
        ofv_1.to_bits(),
        ofv_15.to_bits(),
        "OFV differs between 1 and 15 threads: {ofv_1} vs {ofv_15}"
    );
}
