//! Tier-1 tests for #1613: `fit()` honours `FitOptions::gradient_method` per call.
//!
//! **The gap.** Every gradient decision read `model.gradient_method`, which only the file
//! entry points stamped from the options. `fit` takes `&CompiledModel` (not `Clone`), so it
//! could not stamp, and a direct `fit()` with `FitOptions { gradient_method: Fd, .. }` on a
//! parsed model ran on the analytic `Dual2` gradient — bit-identical to the run without the
//! setting, with no warning. Now `fit` arms the call's flag (`api::pool::FitScope`) and every
//! reader asks `GradientMethod::forced_fd`, a union with the model's own flag.
//!
//! Every case reads **which engine ran** — the `gradient_method_inner` / `_outer` labels,
//! which `fit_inner` computes from the same predicates the loops read — and not only the
//! objective. The R-matrix scope (G4) and the IOV FD reason (G5) are pinned at the reader in
//! `estimation/cov_diagnostics_tests.rs` and `estimation/inner_optimizer_iov_tests.rs`.

use super::*;
use crate::types::{GradientMethod, Optimizer};

const FD: &str = "finite differences";
const ANALYTIC: &str = "analytic (Dual2)";

/// `examples/warfarin.ferx` on `data/warfarin.csv`: closed-form, in analytic scope on both
/// loops, so `Auto` and `Fd` take different engines.
fn warfarin(stamp: GradientMethod) -> (CompiledModel, Population) {
    let mut parsed =
        crate::parser::model_parser::parse_full_model_file(Path::new("examples/warfarin.ferx"))
            .expect("parse warfarin");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("read warfarin");
    parsed.model.gradient_method = stamp;
    (parsed.model, population)
}

/// A short FOCE run with the optimizer pinned: under `optimizer = auto` an FD outer gradient
/// resolves to BOBYQA, whose label is `N/A`, and that would hide the outer half of the fix.
fn opts(gradient: GradientMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Foce,
        interaction: false,
        optimizer: Optimizer::NloptLbfgs,
        gradient_method: gradient,
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        threads: Some(1),
        ..Default::default()
    }
}

fn run(stamp: GradientMethod, gradient: GradientMethod) -> FitResult {
    let (model, population) = warfarin(stamp);
    fit(&model, &population, &model.default_params, &opts(gradient)).expect("fit")
}

/// **G1 + G2 — the direct `fit()` with `options = Fd` on an unstamped model runs FD on both
/// loops and lands on the stamped path's objective bit for bit; the `Auto` control in the same
/// test runs analytic and lands elsewhere.**
///
/// At `d43afca9` the G1 row was bit-identical to the G2 row (inner and outer analytic), which
/// is the reported defect. The straddle is asserted, so the bit-equality with the stamped run
/// cannot pass because `Fd` stopped mattering.
///
/// Mutations, each of which names its side: drop the `fd` member from `FitScope::of` → both
/// labels; revert `analytic_inner_common_bail` to
/// `model.gradient_method` → inner only; revert `analytic_outer_gradient_available` → outer
/// only. A predicate that always says FD fails the G2 half.
#[test]
fn a_direct_fit_with_gradient_fd_in_the_options_runs_fd_on_both_loops() {
    let direct = run(GradientMethod::Auto, GradientMethod::Fd);
    let stamped = run(GradientMethod::Fd, GradientMethod::Fd);
    let control = run(GradientMethod::Auto, GradientMethod::Auto);

    assert_eq!(
        direct.gradient_method_inner, FD,
        "inner: options.gradient_method = Fd must reach the EBE solve"
    );
    assert_eq!(
        direct.gradient_method_outer, FD,
        "outer: options.gradient_method = Fd must reach the outer gradient"
    );
    assert!(direct.ofv.is_finite() && stamped.ofv.is_finite() && control.ofv.is_finite());
    assert_eq!(
        direct.ofv.to_bits(),
        stamped.ofv.to_bits(),
        "direct fit() OFV {:.17e} vs stamped-model OFV {:.17e}",
        direct.ofv,
        stamped.ofv
    );

    // G2, the straddle.
    assert_eq!(control.gradient_method_inner, ANALYTIC, "control inner");
    assert_eq!(control.gradient_method_outer, ANALYTIC, "control outer");
    assert_ne!(
        control.ofv.to_bits(),
        direct.ofv.to_bits(),
        "Fd and Auto must reach different objectives on this fixture, or the equality above \
         holds whether the options are read or not"
    );
}

/// **G3 — the model's own `gradient = fd` still wins over `options = Auto`** (the union;
/// today's behaviour, and the one a caller who stamped by hand relies on).
///
/// Mutation: rewrite `forced_fd` as "the options win" (`scope.unwrap_or(model flag)`).
#[test]
fn a_stamped_model_runs_fd_under_auto_options() {
    let r = run(GradientMethod::Fd, GradientMethod::Auto);
    assert_eq!(r.gradient_method_inner, FD, "inner");
    assert_eq!(r.gradient_method_outer, FD, "outer");
}

/// **G6 — a pool's workers carry the `gradient = fd` of the call they serve, and pools are
/// not reused across a different flag.**
///
/// The labels cannot see this: `fit_inner` computes them on the fit's own thread, which
/// `FitScope::armed` arms whatever pool it runs on. What the pool decides is what the
/// *workers* read, i.e. the per-subject EBE solves the `par_iter` fans out — so this compares
/// objectives. A pinned `threads` leases from one cache keyed by the scope, and an `Auto` fit
/// leases the default-scope pool there; alternating `Auto` → `Fd` → `Auto` at one width leaves
/// each fit an idle pool of the *other* flag to (wrongly) reuse. The `Fd` objective is
/// compared with a stamped model's, whose flag rides on the model and so reaches the workers
/// regardless of the pool.
///
/// Mutation: drop `fd` from `FitScope::same_pool_key` → the `Fd` fit reuses the `Auto` pool
/// (workers analytic) or the second `Auto` fit reuses the `Fd` pool; drop it from
/// `install_on_worker` → the `Fd` fit's workers never see it.
#[test]
fn pool_workers_carry_the_calls_gradient_fd() {
    // An odd width other tests do not pin, so the idle pool this test leaves is the one its
    // next fit finds.
    const WIDTH: usize = 5;
    let pinned = |g| FitOptions {
        threads: Some(WIDTH),
        ..opts(g)
    };
    let fit_with = |stamp, g| {
        let (model, population) = warfarin(stamp);
        fit(&model, &population, &model.default_params, &pinned(g)).expect("fit")
    };
    let auto_before = fit_with(GradientMethod::Auto, GradientMethod::Auto);
    let direct_fd = fit_with(GradientMethod::Auto, GradientMethod::Fd);
    let auto_after = fit_with(GradientMethod::Auto, GradientMethod::Auto);
    let stamped_fd = fit_with(GradientMethod::Fd, GradientMethod::Fd);

    assert_eq!(direct_fd.gradient_method_inner, FD);
    assert_eq!(auto_before.gradient_method_inner, ANALYTIC);
    assert_ne!(
        auto_before.ofv.to_bits(),
        stamped_fd.ofv.to_bits(),
        "premise: Fd and Auto reach different objectives at this width"
    );
    assert_eq!(
        direct_fd.ofv.to_bits(),
        stamped_fd.ofv.to_bits(),
        "the Fd fit's workers: {:.17e} vs stamped {:.17e}",
        direct_fd.ofv,
        stamped_fd.ofv
    );
    assert_eq!(
        auto_after.ofv.to_bits(),
        auto_before.ofv.to_bits(),
        "an Auto fit after an Fd one: {:.17e} vs {:.17e}",
        auto_after.ofv,
        auto_before.ofv
    );
}
