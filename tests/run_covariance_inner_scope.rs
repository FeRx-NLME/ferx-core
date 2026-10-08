//! #426: `run_covariance` re-solves the EBEs with the **caller's** inner-loop settings,
//! whatever any other fit in the process sets or has set.
//!
//! Until #426 `inner_optimizer` / `ebe_warm_start` were process globals that every `fit()`
//! wrote and none restored, so a standalone covariance step scored with whichever inner solver
//! the most recent fit in the process used — in another thread too, mid-run. That is what
//! made #1790's `run_covariance` bit-identity flake on `main` once a concurrent test fitted
//! with another solver.
//!
//! Fixture: warfarin, FOCEI, covariance step, analytic (`Dual2`) inner gradients,
//! measured via `gradient_method_inner` (fit A with `inner_optimizer = lbfgs`). Mutation: drop the inner half of the scope `run_covariance`
//! opens (score under `FitOptions::default()`'s inner settings) → T4 (ii) and T5 die; write
//! a fit's inner settings to a process global again → T4 (iii) dies.

use ferx_core::{fit, prepare_run, run_covariance, FitOptions, FitResult, InnerOptimizer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

struct Fixture {
    prep: ferx_core::PreparedRun,
    lbfgs: FitOptions,
    fit_a: FitResult,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let prep =
            prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv")).expect("prepare");
        let lbfgs = FitOptions {
            verbose: false,
            method: ferx_core::EstimationMethod::FoceI,
            run_covariance_step: true,
            sir: false,
            inner_optimizer: InnerOptimizer::Lbfgs,
            ..prep.parsed.fit_options.clone()
        };
        let fit_a = fit(
            &prep.parsed.model,
            &prep.population,
            &prep.init_params,
            &lbfgs,
        )
        .expect("lbfgs fit");
        Fixture { prep, lbfgs, fit_a }
    })
}

fn bits(fit: &FitResult) -> Vec<u64> {
    let cov = fit.covariance_matrix.as_ref().expect("covariance step ran");
    assert!(cov.iter().all(|x| x.is_finite()), "non-finite covariance");
    cov.iter().map(|x| x.to_bits()).collect()
}

fn cov_under(f: &Fixture, options: &FitOptions) -> Vec<u64> {
    bits(
        &run_covariance(
            &f.fit_a,
            Some(&f.prep.parsed.model),
            Some(&f.prep.population),
            options,
        )
        .expect("run_covariance"),
    )
}

/// A short fit with another inner solver: the writer that used to move the global.
fn nelder_mead_fit(f: &Fixture) {
    let nm = FitOptions {
        verbose: false,
        inner_optimizer: InnerOptimizer::NelderMead,
        run_covariance_step: false,
        sir: false,
        outer_maxiter: 2,
        ..f.prep.parsed.fit_options.clone()
    };
    fit(
        &f.prep.parsed.model,
        &f.prep.population,
        &f.prep.init_params,
        &nm,
    )
    .expect("nm fit");
}

/// T4, deterministic. (i) premise: the inner solver moves the covariance on this fixture, so
/// (ii) and (iii) cannot pass by accident; (ii) the caller's lbfgs reproduces the fit's inline
/// covariance bit for bit; (iii) a later fit with another solver leaves (ii) unchanged. Red on
/// `main` at `1f298774`: (iii) — and (i) — read the global the Nelder–Mead fit left.
#[test]
fn run_covariance_scores_with_the_callers_inner_solver() {
    let f = fixture();
    let inline = bits(&f.fit_a);
    let lbfgs = cov_under(f, &f.lbfgs);
    assert_eq!(
        lbfgs, inline,
        "(ii) run_covariance under the fit's own options must reproduce its inline step"
    );
    nelder_mead_fit(f);
    assert_eq!(
        cov_under(f, &f.lbfgs),
        inline,
        "(iii) a later Nelder–Mead fit changed run_covariance's inner solver"
    );
    let default_inner = FitOptions {
        inner_optimizer: InnerOptimizer::Auto,
        ..f.lbfgs.clone()
    };
    assert_ne!(
        cov_under(f, &default_inner),
        inline,
        "(i) premise: the inner solver must move the covariance on this fixture"
    );
}

/// T5, the same property under a genuinely concurrent writer: a second thread fits with
/// Nelder–Mead in a loop while this one repeats the covariance step. Nondeterministic under
/// the old globals (the writer had to land mid-step), so T4 (iii) is the deterministic
/// killer; this pins the scenario #1790's flake was.
#[test]
fn run_covariance_is_unaffected_by_a_concurrent_fit() {
    let f = fixture();
    let inline = bits(&f.fit_a);
    let stop = AtomicBool::new(false);
    let fits = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let writer = s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                nelder_mead_fit(f);
                fits.fetch_add(1, Ordering::Relaxed);
            }
        });
        // Stop the writer on every exit, a failed assertion included, or the scope never joins.
        struct StopOnDrop<'a>(&'a AtomicBool);
        impl Drop for StopOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let _stop = StopOnDrop(&stop);
        // Let the writer get going before the first step.
        while fits.load(Ordering::Relaxed) == 0 && !writer.is_finished() {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        for round in 0..5 {
            assert_eq!(
                cov_under(f, &f.lbfgs),
                inline,
                "round {round}: a concurrent fit moved run_covariance's inner solver"
            );
        }
    });
}
