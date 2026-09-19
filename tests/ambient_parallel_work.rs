//! #1460 — a declared thread count must reach parallel work that is not a `fit()`.
//!
//! **One test per process, on purpose.** `configure_global_thread_pool` is
//! once-per-process and `ThreadPoolBuilder::build_global()` is too, and the second is
//! what makes this test able to fail: Rayon's global pool can be built exactly once, so
//! a *successful* `build_global` afterwards proves nothing had built it — i.e. the work
//! above did not quietly run on it. Adding a second test to this file (or any earlier
//! `rayon::current_num_threads()` in it) would initialise that pool and turn the
//! assertion into a tautology.

use rayon::prelude::*;

#[test]
fn a_declared_thread_count_reaches_parallel_work_outside_a_fit() {
    ferx_core::configure_global_thread_pool(2).expect("declare the process-wide width");

    let widths: Vec<usize> = ferx_core::install_on_engine_pool(|| {
        (0..64)
            .into_par_iter()
            .map(|_| rayon::current_num_threads())
            .collect()
    });

    assert!(!widths.is_empty(), "the parallel work did not run");
    assert!(
        widths.iter().all(|&w| w == 2),
        "declared 2 workers, work ran on a pool of {:?} — the width was lost on the way \
         to non-fit parallel work",
        widths.first()
    );

    // The regression this file exists to catch, and the reason it holds only one test:
    // the same closure written as a bare `par_iter` runs on Rayon's *global* pool at one
    // worker per logical CPU (measured: 15 on a 15-core host, where 2 was asked for).
    // Building the global pool here can only succeed if nothing already did.
    assert!(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok(),
        "Rayon's global pool already existed: something ran ambient parallel work, at \
         one worker per logical CPU rather than the declared 2"
    );
}
