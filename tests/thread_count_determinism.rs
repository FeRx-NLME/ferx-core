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
//!
//! Four tests cover every function this PR patched: FOCEI
//! (`foce_population_nll`/`_iov` in `src/stats/likelihood.rs`), SAEM
//! non-IOV and IOV (`obs_nll_sum`/`_iov` and both branches of
//! `theta_sigma_mstep_light` in `src/estimation/saem.rs`), and IMPMAP
//! (`theta_sigma_weighted_mstep` in `src/estimation/impmap.rs`).

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

fn assert_bit_identical_across_threads(label: &str, ofv_1: f64, ofv_4: f64, ofv_15: f64) {
    assert!(
        ofv_1.is_finite(),
        "{label}: OFV must be finite, got {ofv_1}"
    );
    assert_eq!(
        ofv_1.to_bits(),
        ofv_4.to_bits(),
        "{label}: OFV differs between 1 and 4 threads: {ofv_1} vs {ofv_4}"
    );
    assert_eq!(
        ofv_1.to_bits(),
        ofv_15.to_bits(),
        "{label}: OFV differs between 1 and 15 threads: {ofv_1} vs {ofv_15}"
    );
}

fn run_saem_with_threads(n: usize) -> f64 {
    let (model, population) = warfarin();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    // Small, fixed-length exploration/convergence phases (no adaptive stop) —
    // mirrors the fast SAEM setup in tests/importance_sampling_api.rs.
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 5;
    opts.saem_seed = Some(267);
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.threads = Some(n);
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("SAEM fit must succeed");
    result.ofv
}

/// SAEM routes its M-step objective/gradient through `obs_nll_sum` and
/// `theta_sigma_mstep_light` (`src/estimation/saem.rs`), one of the sites this
/// PR patched. Same bit-identical-OFV check as the FOCEI test above, so a
/// regression in the SAEM-side fix fails here even if the FOCEI path stays
/// correct.
#[test]
fn saem_ofv_is_independent_of_thread_count() {
    let ofv_1 = run_saem_with_threads(1);
    let ofv_4 = run_saem_with_threads(4);
    let ofv_15 = run_saem_with_threads(15);
    assert_bit_identical_across_threads("SAEM", ofv_1, ofv_4, ofv_15);
}

fn warfarin_iov() -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let model = parse_model_file(Path::new("examples/warfarin_iov_saem.ferx"))
        .expect("warfarin_iov_saem model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data must load");
    (model, population)
}

fn run_saem_iov_with_threads(n: usize) -> f64 {
    let (model, population) = warfarin_iov();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 5;
    opts.saem_n_convergence = 5;
    opts.saem_seed = Some(267);
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.threads = Some(n);
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("SAEM IOV fit must succeed");
    result.ofv
}

/// IOV variant of the SAEM check above: exercises `obs_nll_sum_iov` and the
/// `obs_nll_subject_grad_iov` branch of `theta_sigma_mstep_light`, the one
/// pair of patched functions the non-IOV `warfarin.ferx` fixture above never
/// reaches (it has no `kappa` block).
#[test]
fn saem_iov_ofv_is_independent_of_thread_count() {
    let ofv_1 = run_saem_iov_with_threads(1);
    let ofv_4 = run_saem_iov_with_threads(4);
    let ofv_15 = run_saem_iov_with_threads(15);
    assert_bit_identical_across_threads("SAEM IOV", ofv_1, ofv_4, ofv_15);
}

fn run_impmap_with_threads(n: usize) -> f64 {
    let (model, population) = warfarin();
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Impmap;
    // Aggressively capped — wire-up/determinism check, not convergence quality
    // (mirrors tests/impmap_api.rs's fast Tier-2 setup).
    opts.impmap_iterations = 3;
    opts.impmap_samples = 50;
    opts.impmap_averaging = 2;
    opts.impmap_seed = Some(7);
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.threads = Some(n);
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("IMPMAP fit must succeed");
    result.ofv
}

/// IMPMAP's weighted M-step objective (`theta_sigma_weighted_mstep` in
/// `src/estimation/impmap.rs`) is the third site this PR patched. Same
/// bit-identical-OFV check, seeded so the importance-sampling draws
/// themselves are reproducible independent of thread count.
#[test]
fn impmap_ofv_is_independent_of_thread_count() {
    let ofv_1 = run_impmap_with_threads(1);
    let ofv_4 = run_impmap_with_threads(4);
    let ofv_15 = run_impmap_with_threads(15);
    assert_bit_identical_across_threads("IMPMAP", ofv_1, ofv_4, ofv_15);
}
