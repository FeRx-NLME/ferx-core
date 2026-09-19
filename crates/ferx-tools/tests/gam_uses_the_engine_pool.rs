//! #1460 — GAM screening must not run its `par_iter` on Rayon's global pool.
//!
//! Screening is reached from `ferx gam` and from scripts, i.e. from a thread that is not
//! a Rayon worker. A bare `par_iter` there lands on Rayon's *global* pool, which ferx
//! never sizes: one worker per logical CPU, whatever `--threads` asked for. Measured on a
//! 15-core host after `configure_global_thread_pool(2)`: 15 workers.
//!
//! **One test per process, on purpose.** `ThreadPoolBuilder::build_global()` succeeds
//! exactly once per process, which is what makes this able to fail: a success *after* the
//! screening proves nothing had built the global pool, i.e. the screening did not use it.
//! A second test in this file — or any earlier `rayon::current_num_threads()` — would
//! build that pool itself and make the assertion vacuous.

use ferx_core::CovariateKind;
use ferx_tools::gam::{gam_screen_raw, GamOptions};

#[test]
fn gam_screening_does_not_initialise_rayons_global_pool() {
    // 24 subjects is enough for the spline fits to be well posed; the numbers themselves
    // do not matter here, only that every eta is screened.
    let n = 24usize;
    let eta_a: Vec<f64> = (0..n).map(|i| (i as f64 - 12.0) / 12.0).collect();
    let eta_b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64 - 3.0) / 8.0).collect();
    let wt: Vec<f64> = (0..n).map(|i| 60.0 + (i % 11) as f64 * 3.0).collect();
    let crcl: Vec<f64> = (0..n).map(|i| 40.0 + (i % 13) as f64 * 5.0).collect();

    let out = gam_screen_raw(
        &["ETA_CL", "ETA_V"],
        &[&eta_a, &eta_b],
        &[0.10, 0.15],
        &["WT", "CRCL"],
        &[&wt, &crcl],
        &[CovariateKind::Continuous, CovariateKind::Continuous],
        &GamOptions::default(),
    );
    assert_eq!(out.eta_results.len(), 2, "both etas must be screened");

    assert!(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok(),
        "GAM screening initialised Rayon's global pool: its par_iter ran at one worker \
         per logical CPU instead of on the engine's pool (#1460)"
    );
}
