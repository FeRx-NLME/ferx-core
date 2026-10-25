//! #2261: the warm starts `fit()` and `run_sir` hand an IOV subject outside the FOCE/FOCEI
//! outer loop carry κ̂ with η̂ — SAEM's and VI's final EBE solves, the in-fit and standalone
//! SIR draws, and the evaluation-only quadrature stage. Each test runs a whole (short) fit
//! under the recorder and checks the fit-wide invariant
//! (`assert_every_warm_pass_is_joint`); a site that hands `η̂` alone fails it, naming the
//! pass.
//!
//! Not every cell of {one-level, nested levels, κ + residual correlation} is reachable:
//! nested levels are FOCE/FOCEI only (`E_NESTED_LEVEL_METHOD_UNSUPPORTED`) and a residual
//! correlation likewise (`E_RESIDUAL_CORRELATION_SCOPE`), so SAEM, VI and the quadrature
//! stage take the one-level fixture alone. IMP and IMPMAP refuse IOV models outright, so
//! their stage warm starts can never carry κ.

use crate::estimation::inner_optimizer::record_inner_solves;
use crate::estimation::inner_optimizer::test_helpers::{
    assert_every_warm_pass_is_joint, kappa_correlation_fixture, nested_levels_fixture,
    warfarin_iov_fixture,
};
use crate::estimation::inner_optimizer::InnerRecording;
use crate::types::{
    CompiledModel, CovarianceFallback, EstimationMethod, FitOptions, FitResult, Population,
};

type Fixture = (CompiledModel, Population, FitOptions);

fn fit_recorded(
    (model, pop, base): Fixture,
    edit: impl FnOnce(FitOptions) -> FitOptions,
) -> (
    CompiledModel,
    Population,
    FitOptions,
    FitResult,
    InnerRecording,
) {
    let opts = edit(FitOptions {
        verbose: false,
        threads: Some(1),
        ..base
    });
    let (fit, rec) = on_fit_thread(&opts, || {
        record_inner_solves(|| {
            crate::api::fit(&model, &pop, &model.default_params, &opts).expect("the fit runs")
        })
    });
    (model, pop, opts, fit, rec)
}

/// `f` on the one-thread pool `fit()` would lease for `opts`, so the fit runs on this thread
/// rather than installing a pool of its own, and the recorder sees its solves.
fn on_fit_thread<R: Send>(opts: &FitOptions, f: impl FnOnce() -> R + Send) -> R {
    crate::api::pool::fit_scope_pool(crate::api::pool::FitScope::of(opts), 1)
        .expect("a one-thread fit pool")
        .install(f)
}

fn saem(o: FitOptions) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        methods: vec![EstimationMethod::Saem],
        saem_n_exploration: 4,
        saem_n_convergence: 4,
        saem_seed: Some(7),
        ..o
    }
}

#[test]
fn saem_final_ebe_solve_starts_at_the_chain_kappa() {
    let (model, _, _, _, rec) = fit_recorded(warfarin_iov_fixture(), saem);
    assert_every_warm_pass_is_joint("SAEM final EBE warfarin_iov", &rec, &model);
}

fn vi(o: FitOptions) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        methods: vec![EstimationMethod::Vi],
        vi_iters: 5,
        vi_seed: Some(7),
        ..o
    }
}

#[test]
fn vi_final_ebe_solve_starts_at_the_variational_kappa_mean() {
    let (model, _, _, _, rec) = fit_recorded(warfarin_iov_fixture(), vi);
    assert_every_warm_pass_is_joint("VI final EBE warfarin_iov", &rec, &model);
}

fn bayes(o: FitOptions) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Bayes,
        methods: vec![EstimationMethod::Bayes],
        bayes_warmup: 10,
        bayes_iters: 10,
        bayes_chains: 2,
        bayes_seed: Some(7),
        ..o
    }
}

/// #2377: Bayes's final EBE pass at the posterior mean starts each occasion at its
/// posterior-mean κ, as SAEM's and VI's do.
#[test]
fn bayes_final_ebe_solve_starts_at_the_posterior_mean_kappa() {
    let (model, _, _, fit, rec) = fit_recorded(warfarin_iov_fixture(), bayes);
    assert!(fit.bayes.is_some(), "fixture: the Bayes sampler ran");
    assert_every_warm_pass_is_joint("Bayes final EBE warfarin_iov", &rec, &model);
    // A κ sum that is never accumulated still hands a joint vector of the right width,
    // all zeros, which the check above accepts. The posterior mean of sampled κ is not.
    let handed_kappa = rec
        .passes
        .iter()
        .filter(|p| !p.kappas.is_empty())
        .filter_map(|p| p.init.as_ref())
        .any(|init| init[model.n_eta..].iter().any(|&k| k != 0.0));
    assert!(
        handed_kappa,
        "every handed κ̂ is 0: the sampled κ were not summed"
    );
}

/// `[focei, laplace]` with `agq_eval_only`: the Laplace stage re-solves the EBEs warm from
/// the FOCEI stage's.
fn focei_then_laplace_eval(o: FitOptions) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        methods: vec![EstimationMethod::FoceI, EstimationMethod::Laplace],
        agq_eval_only: true,
        ..o
    }
}

#[test]
fn quadrature_eval_stage_starts_at_the_previous_stage_mode() {
    let (model, _, _, fit, rec) = fit_recorded(warfarin_iov_fixture(), focei_then_laplace_eval);
    assert_eq!(fit.method_chain.len(), 2, "fixture: both stages ran");
    assert_every_warm_pass_is_joint("Laplace eval stage warfarin_iov", &rec, &model);
}

/// FOCEI with the covariance step and SIR. On a three-subject fixture the Hessian may be
/// non-PD; `covariance_fallback = sir` then runs the fallback SIR, which takes the same
/// warm starts, so either way a SIR run draws.
fn focei_sir(o: FitOptions) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        methods: vec![EstimationMethod::FoceI],
        run_covariance_step: true,
        covariance_fallback: CovarianceFallback::Sir,
        sir: true,
        sir_samples: 20,
        sir_resamples: 10,
        sir_seed: Some(7),
        ..o
    }
}

fn assert_in_fit_sir_draws_are_joint(label: &str, fixture: Fixture) {
    let (model, pop, opts, fit, rec) = fit_recorded(fixture, focei_sir);
    assert!(
        fit.sir_ess.is_some(),
        "{label}: fixture: SIR ran ({:?})",
        fit.warnings
    );
    assert_every_warm_pass_is_joint(label, &rec, &model);
    // `run_sir` on the stored fit draws from `fit.ebe_kappas` the same way.
    if fit.covariance_matrix.is_some() {
        let rec = on_fit_thread(&opts, || {
            record_inner_solves(|| {
                crate::run_sir(&fit, Some(&model), Some(&pop), &opts).expect("run_sir")
            })
            .1
        });
        assert_every_warm_pass_is_joint(&format!("{label} run_sir"), &rec, &model);
    }
}

#[test]
fn sir_draws_start_at_the_fit_mode() {
    assert_in_fit_sir_draws_are_joint("SIR warfarin_iov", warfarin_iov_fixture());
}

#[test]
fn sir_draws_start_at_the_fit_mode_nested_levels() {
    assert_in_fit_sir_draws_are_joint("SIR nested_levels_3", nested_levels_fixture());
}

#[test]
fn sir_draws_start_at_the_fit_mode_kappa_correlation() {
    assert_in_fit_sir_draws_are_joint("SIR warfarin_iov ar1", kappa_correlation_fixture());
}
