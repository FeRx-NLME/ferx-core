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

/// The post-hoc fixtures: a FOCEI fit (non-interaction FOCE with a prediction-dependent
/// residual declines the analytic R-matrix for its own reason, which would hide the
/// gradient clause) with a covariance matrix, so `run_sir` has a proposal.
fn posthoc_base() -> (CompiledModel, Population, FitResult) {
    let (model, population) = warfarin(GradientMethod::Auto);
    let o = FitOptions {
        run_covariance_step: true,
        ..posthoc_opts(GradientMethod::Auto)
    };
    let base = fit(&model, &population, &model.default_params, &o).expect("base fit");
    assert!(
        base.covariance_matrix.is_some(),
        "premise: the base fit has a covariance"
    );
    (model, population, base)
}

fn posthoc_opts(gradient: GradientMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..opts(gradient)
    }
}

fn cov_bits(r: &FitResult) -> Vec<u64> {
    let m = r.covariance_matrix.as_ref().expect("covariance");
    assert!(
        m.iter().all(|v| v.is_finite()),
        "non-finite covariance entry"
    );
    m.iter().map(|v| v.to_bits()).collect()
}

/// **G4b — a post-hoc `run_covariance` honours the `gradient_method` in its own options**
/// (#1829 review r1 row 2). `Fd` options on an unstamped model must take the FD stencil — bit
/// for bit the matrix a stamped model gives under `Auto` options, whose flag rides on the
/// model — while the `Auto` control takes the analytic R-matrix and lands elsewhere.
///
/// Mutation: hand `with_fit_scope` a gradient-`Auto` copy of the options
/// (`run_covariance.rs`) → the `Fd` call equals the control, not the stamped run.
#[test]
fn a_posthoc_run_covariance_honours_the_options_gradient_fd() {
    let (model, population, base) = posthoc_base();
    let (stamped, _) = warfarin(GradientMethod::Fd);
    let cov = |m: &CompiledModel, g| {
        crate::run_covariance(&base, Some(m), Some(&population), &posthoc_opts(g))
            .expect("run_covariance")
    };
    let fd = cov_bits(&cov(&model, GradientMethod::Fd));
    let reference = cov_bits(&cov(&stamped, GradientMethod::Auto));
    let control = cov_bits(&cov(&model, GradientMethod::Auto));
    assert_ne!(
        control, reference,
        "premise: the analytic R-matrix and the FD stencil differ on this fit"
    );
    assert_eq!(
        fd, reference,
        "run_covariance with options Fd must take the FD stencil, as a stamped model does"
    );
}

/// **G4c — the same for `run_sir`**, whose importance weights re-solve every subject's EBE:
/// the inner η-gradient route is what the options' `Fd` changes there.
///
/// Mutation: hand `run_sir`'s scope a gradient-`Auto` copy of the options (`run_sir.rs`) →
/// the `Fd` call equals the control.
#[test]
fn a_posthoc_run_sir_honours_the_options_gradient_fd() {
    let (model, population, base) = posthoc_base();
    let (stamped, _) = warfarin(GradientMethod::Fd);
    let sir = |m: &CompiledModel, g| {
        let o = FitOptions {
            sir_samples: 40,
            sir_resamples: 20,
            sir_seed: Some(7),
            ..posthoc_opts(g)
        };
        let r = crate::run_sir(&base, Some(m), Some(&population), &o).expect("run_sir");
        let ess = r.sir_ess.expect("ess");
        assert!(ess.is_finite(), "ess {ess}");
        r.sir_ci_theta
            .expect("SIR CIs")
            .into_iter()
            .flat_map(|(lo, hi)| [lo.to_bits(), hi.to_bits()])
            .chain(std::iter::once(ess.to_bits()))
            .collect::<Vec<u64>>()
    };
    let fd = sir(&model, GradientMethod::Fd);
    let reference = sir(&stamped, GradientMethod::Auto);
    let control = sir(&model, GradientMethod::Auto);
    assert_ne!(
        control, reference,
        "premise: the analytic and FD inner routes weight the draws differently"
    );
    assert_eq!(
        fd, reference,
        "run_sir with options Fd must re-solve on FD, as a stamped model does"
    );
}
