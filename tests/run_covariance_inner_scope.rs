//! #426: `run_covariance` re-solves the EBEs with the **caller's** inner-loop settings,
//! whatever any other fit in the process sets or has set. Since #426's second PR a setting the
//! caller leaves at its default comes from the fit's scoring record instead
//! (`tests/run_covariance_scoring_record.rs`), which T4 (i) and (iv) account for.
//!
//! Until #426 `inner_optimizer` / `ebe_warm_start` were process globals that every `fit()`
//! wrote and none restored, so a standalone covariance step scored with whichever inner solver
//! the most recent fit in the process used — in another thread too, mid-run. That is what
//! made #1790's `run_covariance` bit-identity flake on `main` once a concurrent test fitted
//! with another solver.
//!
//! Fixture: `schnider`, FOCEI, covariance step, analytic (`Dual2`) inner gradients,
//! measured via `gradient_method_inner` (fit A with `inner_optimizer = lbfgs`). Mutation: drop the inner half of the scope `run_covariance`
//! opens (score under `FitOptions::default()`'s inner settings) → T4 (i) dies here, and the
//! `inner_optimizer` row and T10 in `run_covariance_scoring_record.rs` (measured on #1847;
//! before #1847 it was T4 (ii) and T5, which since then start from the fit's own EBEs, where
//! the default dense BFGS and L-BFGS reconverge to the same bits, see T4); write a fit's inner
//! settings to a process global again → T4 (iii) dies.
//!
//! The straddle T4 (i) needs is fragile, because it rests on how the inner solvers stop. Warfarin
//! served until #2093: with the step-scaled first inner iterate, Nelder–Mead and L-BFGS
//! reconverged warfarin's EBEs to the same covariance bits, and `two_cpt_oral_cov` took over.
//! #2270 (relative BFGS pair gate, gradient-verified steps below the objective's resolution)
//! made it platform-dependent: the BFGS polish the fallback certificate runs on a Nelder–Mead
//! end point reaches L-BFGS's bits on one platform and not the other. Entries of the
//! covariance that differ, Nelder–Mead vs L-BFGS, measured on #2270:
//!
//! | fixture | macOS arm64 | Linux aarch64 |
//! |---|---|---|
//! | `two_cpt_oral_cov` | 0 / 169 | 169 / 169 |
//! | `warfarin` | 49 / 49 | 0 / 49 |
//! | `mm_oral` | 49 / 49 | 49 / 49 |
//! | `one_cpt_iv`, `emax_pkpd`, `one_cpt_infusion`, `bioavailability` | 0 | 0 |
//!
//! So `mm_oral`, the one that straddled on both: a straddle seen on one platform only would
//! leave T4 (i) red on the other.
//!
//! #2557 certified the *primary* Nelder–Mead search's end point too, not only the fallback's,
//! so under `inner_optimizer = nelder_mead` the inner solve now ends on the same certified
//! mode L-BFGS reaches, and the straddle moved again. Entries that differ, Nelder–Mead vs
//! L-BFGS, measured on #2557 (`run_covariance` from fit A's EBEs):
//!
//! | fixture | macOS arm64 | Linux aarch64 |
//! |---|---|---|
//! | `schnider` | 121 / 121 | 121 / 121 |
//! | `warfarin` | 0 / 49 | 49 / 49 |
//! | `mm_oral` | 0 / 49 | 0 / 49 |
//! | `two_cpt_oral_cov`, `one_cpt_iv`, `emax_pkpd`, `bioavailability`, `mixed_absorption`, `mm_iv`, `warfarin_iov`, `three_cpt_iv` | 0 | 0 |
//!
//! On `mm_oral` no inner setting moves the covariance any more: `nelder_mead`, `bfgs` and
//! `auto` against L-BFGS, at `cov_inner_tol` unset, 1e-3, 1e-4 and 1e-6, and `ebe_warm_start`
//! flipped all reproduce its bits on both platforms. So `schnider` now.

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
        let prep = prepare_run("examples/schnider.ferx", Some("data/schnider_propofol.csv"))
            .expect("prepare");
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
    // The premise uses the solver the leak would bring in. Since #1847 the step starts from
    // the fit's own EBEs, and from there dense BFGS (`Auto` here) and L-BFGS reconverge
    // to the same bits. On warfarin (measured on #2092, before #2093) the analytic-Hessian route reconverges once, at
    // the centre: 8 of 10 warm starts already meet the tolerance, and the other 2 move by
    // ≤1.9e-9 with identical steps under both solvers (measured on #2092). So only a
    // materially different solver can move the covariance. Before #1847 `Auto` did, through
    // the cold solve it started from. A
    // non-default caller value is scored as given, so the record stays.
    let nelder_mead = FitOptions {
        inner_optimizer: InnerOptimizer::NelderMead,
        ..f.lbfgs.clone()
    };
    assert_ne!(
        cov_under(f, &nelder_mead),
        inline,
        "(i) premise: the inner solver must move the covariance on this fixture"
    );
    // (iv) #426 PR 2: default options reproduce the inline step too, from the record.
    let defaults = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    assert_eq!(
        cov_under(f, &defaults),
        inline,
        "(iv) run_covariance with default options must reproduce the fit's inline step"
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
