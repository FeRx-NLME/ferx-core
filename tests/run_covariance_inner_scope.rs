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
//! before #1847 T4 (ii) and T5 died too); write a fit's inner settings to a process global
//! again → T4 (iii) dies, and T5; set [`DISPLACEMENT`] to 0, or have `run_covariance` start from
//! the cold solve whatever the fit reports → the fixture's start-straddle assertion dies, in
//! both tests (each measured on #2570).
//!
//! Why only (i) sees the first mutation: since #1847 the step starts from the EBEs the fit
//! reports, and from there the default dense BFGS (`Auto`) and L-BFGS reconverge to the same
//! bits. Measured on warfarin on #2092: the analytic-Hessian route reconverges once, at the
//! centre; 8 of 10 warm starts already meet the tolerance, and the other 2 move by ≤1.9e-9 with
//! identical steps under both solvers. The same holds from the displaced start below (0 of 121
//! entries differ, `Auto` vs L-BFGS, macOS arm64 probe). So a step scored under the default inner settings
//! reproduces the caller's L-BFGS step, in (ii), (iii) and T5 alike, and only a materially
//! different solver, Nelder–Mead in (i), can move the covariance.
//!
//! # Why (i), (iii) and T5 start from displaced EBEs (#2570)
//!
//! The inner solver can only reach the covariance through the step's centre re-solve, and on
//! `schnider` that is the only re-solve there is: the step takes the analytic-Hessian route,
//! so a run is 24 cold solves (the comparison in `reported_ebe_inputs`) and 24 warm ones, from
//! the EBEs the fit reports. A warm start the certificate accepts is returned as it stands
//! under every solver (#2279), and since #2557 a Nelder–Mead end point is certified the same
//! way. So from fit A's own EBEs the premise held only where some subject's warm start was
//! *not* accepted in place and the two solvers then stopped on different bits — a property of
//! how certified solvers stop, not of #426. Measured on macOS arm64 at `f23776566`: all 24 warm
//! re-solves left the EBE untouched (maximum move 0) under both Nelder–Mead and L-BFGS, the
//! covariance was bit-identical, and (i) failed; Linux CI still straddled. The fixture had
//! moved four times for that reason (warfarin → `two_cpt_oral_cov` → `mm_oral` → `schnider`),
//! each move a sweep of ten fixtures on two platforms. A covariance that cannot depend on the
//! solver also left (iii) and T5 blind on that platform: a leaked Nelder–Mead would have
//! reproduced the same bits.
//!
//! So (i), (iii) and T5 run from a copy of fit A whose every EBE is moved by
//! [`DISPLACEMENT`]. That start is off every subject's mode, so no warm start is certified in
//! place and each solver has to search to the mode; Nelder–Mead's simplex and L-BFGS's
//! quasi-Newton path end on different bits. Measured with a probe (not committed) on macOS
//! arm64; the Linux aarch64 column is this test's own printout under `tools/linux-test.sh`:
//!
//! | η displacement | warm starts searched (NM / L-BFGS) | subjects ending on different bits | entries that differ, NM vs L-BFGS, macOS arm64 | same, Linux aarch64 |
//! |---|---|---|---|---|
//! | 0 (fit A's EBEs) | 0 / 0 moved | 0 / 24 | 0 / 121 | not measured (CI straddled) |
//! | 0.01 | 24 / 24 | 24 / 24 | 121 / 121 (max relative 2.5e-5) | 121 / 121 (max relative 7.9e-6) |
//! | 0.05 | 24 / 24 | 24 / 24 | 121 / 121 (max relative 1.7e-5) | not measured |
//!
//! Every subject stays in its basin: the L-BFGS step from the displaced start lands back on fit
//! A's mode, its covariance within a maximum relative difference of 6.0e-7 (macOS arm64) and
//! 9.3e-7 (Linux aarch64) of fit A's inline one, while differing from it in all 121 entries'
//! bits on both. The fixture prints that start straddle and T4 prints (i)'s count and gap on
//! stderr, so `--nocapture` (or any failing run) shows the margin on whatever platform runs it.
//!
//! (ii) and (iv) stay on fit A's own EBEs: they pin bit-identity with the inline step, which
//! ran from those.

use ferx_core::{fit, prepare_run, run_covariance, FitOptions, FitResult, InnerOptimizer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Added to every EBE of fit A to build the start (i), (iii) and T5 run from. Far above the
/// certificate's stationarity tolerance, so no warm start is accepted in place; small enough
/// that every subject stays in its basin (the L-BFGS step lands within 6.0e-7 / 9.3e-7 relative
/// of fit A's inline covariance on macOS arm64 / Linux aarch64, see the module doc).
const DISPLACEMENT: f64 = 0.01;

struct Fixture {
    prep: ferx_core::PreparedRun,
    lbfgs: FitOptions,
    fit_a: FitResult,
    /// Fit A with every EBE moved by [`DISPLACEMENT`].
    displaced: FitResult,
    /// The caller's L-BFGS step from `displaced`, taken inside the fixture right after fit A:
    /// both tests block on the fixture until it is built, so no other fit in this binary can
    /// have run before it. A baseline taken later in a test would be scored under whatever a
    /// leak had left, and (iii) and T5 would compare a leaked step against a leaked step.
    displaced_lbfgs: Vec<u64>,
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
        let mut displaced = fit_a.clone();
        for subject in &mut displaced.subjects {
            subject.eta.add_scalar_mut(DISPLACEMENT);
        }
        let displaced_lbfgs = bits(
            &run_covariance(
                &displaced,
                Some(&prep.parsed.model),
                Some(&prep.population),
                &lbfgs,
            )
            .expect("run_covariance"),
        );
        // The displaced EBEs must be the step's start, or (i), (iii) and T5 would run from fit
        // A's own EBEs, where the solver need not matter (module doc). `reported_ebe_inputs`
        // takes the reported EBEs whenever they are not the cold solve's. Checked here, so both
        // tests carry it, run together or alone.
        let inline = bits(&fit_a);
        let n = n_differing(&displaced_lbfgs, &inline);
        eprintln!(
            "start straddle: {n} of {} entries differ from fit A's inline covariance \
             (max relative {:.2e})",
            inline.len(),
            max_relative(&displaced_lbfgs, &inline)
        );
        assert!(
            n > 0,
            "(i) premise: the step must start from the displaced EBEs, but it reproduced fit A's \
             inline covariance"
        );
        Fixture {
            prep,
            lbfgs,
            fit_a,
            displaced,
            displaced_lbfgs,
        }
    })
}

fn bits(fit: &FitResult) -> Vec<u64> {
    let cov = fit.covariance_matrix.as_ref().expect("covariance step ran");
    assert!(cov.iter().all(|x| x.is_finite()), "non-finite covariance");
    cov.iter().map(|x| x.to_bits()).collect()
}

fn cov_from(f: &Fixture, start: &FitResult, options: &FitOptions) -> Vec<u64> {
    bits(
        &run_covariance(
            start,
            Some(&f.prep.parsed.model),
            Some(&f.prep.population),
            options,
        )
        .expect("run_covariance"),
    )
}

/// How many covariance entries differ in their bits.
fn n_differing(a: &[u64], b: &[u64]) -> usize {
    assert_eq!(a.len(), b.len(), "covariance shapes differ");
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// The largest relative difference between two covariances given as bits.
fn max_relative(a: &[u64], b: &[u64]) -> f64 {
    let mut worst = 0.0_f64;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from_bits(*x), f64::from_bits(*y));
        let rel = ((x - y) / x.abs().max(y.abs()).max(f64::MIN_POSITIVE)).abs();
        assert!(rel.is_finite(), "non-finite relative gap: {x} vs {y}");
        worst = worst.max(rel);
    }
    worst
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

/// T4, deterministic. (i) premise: the inner solver moves the covariance from the displaced
/// start, so (iii) cannot pass by accident; (ii) the caller's lbfgs reproduces the fit's inline
/// covariance bit for bit; (iii) a later fit with another solver leaves the caller's lbfgs
/// step unchanged; (iv) default options reproduce the inline step from the scoring record.
/// Red on `main` at `1f298774`: (iii) — and (i) — read the global the Nelder–Mead fit left.
#[test]
fn run_covariance_scores_with_the_callers_inner_solver() {
    let f = fixture();
    let inline = bits(&f.fit_a);
    assert_eq!(
        cov_from(f, &f.fit_a, &f.lbfgs),
        inline,
        "(ii) run_covariance under the fit's own options must reproduce its inline step"
    );
    // The fixture has already checked that the displaced EBEs are the step's start.
    let displaced_lbfgs = &f.displaced_lbfgs;
    // The premise uses the solver the leak would bring in. A non-default caller value is
    // scored as given, so the scoring record does not replace it.
    let nelder_mead = FitOptions {
        inner_optimizer: InnerOptimizer::NelderMead,
        ..f.lbfgs.clone()
    };
    let displaced_nm = cov_from(f, &f.displaced, &nelder_mead);
    let n = n_differing(&displaced_nm, displaced_lbfgs);
    eprintln!(
        "(i) premise: {n} of {} entries differ, Nelder–Mead vs L-BFGS from the displaced start \
         (max relative {:.2e})",
        inline.len(),
        max_relative(&displaced_nm, displaced_lbfgs)
    );
    assert!(
        n > 0,
        "(i) premise: from EBEs displaced by {DISPLACEMENT}, Nelder–Mead and L-BFGS must end the \
         centre re-solve on different bits, so the inner solver moves the covariance; 0 of {} \
         entries differ. Either the scope no longer carries the caller's inner solver to the \
         re-solve, or a displaced warm start is now returned as it stands (certified in place), \
         or both solvers now stop on the same bits from an off-mode start. The module doc \
         records the measurement this rests on.",
        inline.len()
    );
    nelder_mead_fit(f);
    assert_eq!(
        n_differing(&cov_from(f, &f.displaced, &f.lbfgs), displaced_lbfgs),
        0,
        "(iii) a later Nelder–Mead fit changed run_covariance's inner solver"
    );
    // (iv) #426 PR 2: default options reproduce the inline step too, from the record.
    let defaults = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    assert_eq!(
        cov_from(f, &f.fit_a, &defaults),
        inline,
        "(iv) run_covariance with default options must reproduce the fit's inline step"
    );
}

/// T5, the same property under a genuinely concurrent writer: a second thread fits with
/// Nelder–Mead in a loop while this one repeats the covariance step. Nondeterministic under
/// the old globals (the writer had to land mid-step), so T4 (iii) is the deterministic
/// killer; this pins the scenario #1790's flake was. From the displaced start, where T4 (i)
/// shows the solver moves the covariance.
#[test]
fn run_covariance_is_unaffected_by_a_concurrent_fit() {
    let f = fixture();
    let baseline = &f.displaced_lbfgs;
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
                n_differing(&cov_from(f, &f.displaced, &f.lbfgs), baseline),
                0,
                "round {round}: a concurrent fit moved run_covariance's inner solver"
            );
        }
    });
}
