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

/// `run_sir` on a fit whose `ebe_kappas` do not line up with the population's cells. A
/// population with other cells than the fit's is refused before this point
/// (`E_POPULATION_MISMATCH`), so the guard's input is a `FitResult` whose κ̂ lost its shape
/// — here subject 0 keeps one cell of its two. `kappas_line_up` must then hand η̂ alone and
/// let every κ restart at 0. Without the guard the joint start is one cell short, and
/// `find_ebe_iov`'s warm-start width `debug_assert` aborts the run — the guard's only
/// observable effect, since `iov_initial_vector` cuts a wrong-width start back to η̂ with
/// κ = 0 anyway. So the kill needs debug assertions (a debug build, or `ci-cov` in CI).
#[test]
fn run_sir_hands_eta_alone_when_the_kappas_do_not_line_up() {
    let (model, pop, opts, mut fit, _) = fit_recorded(warfarin_iov_fixture(), focei_sir);
    assert!(
        fit.covariance_matrix.is_some(),
        "fixture: run_sir needs a covariance matrix ({:?})",
        fit.warnings
    );
    let cells = crate::stats::stacked_layout::group_count(&pop.subjects[0]);
    assert_eq!(
        fit.ebe_kappas[0].len(),
        cells,
        "fixture: the fit's κ̂ line up"
    );
    assert!(cells > 1, "fixture: subject 0 has {cells} cells");
    fit.ebe_kappas[0].pop();

    let (sir, rec) = on_fit_thread(&opts, || {
        record_inner_solves(|| crate::run_sir(&fit, Some(&model), Some(&pop), &opts))
    });
    sir.expect("run_sir with a misaligned κ̂");
    let id = &pop.subjects[0].id;
    let starts: Vec<_> = rec.starts.iter().filter(|s| &s.subject == id).collect();
    assert!(!starts.is_empty(), "fixture: subject {id} was solved");
    for s in &starts {
        assert_eq!(s.kappas.len(), cells, "subject {id}: every cell is solved");
        assert!(
            s.kappas.iter().flatten().all(|&k| k == 0.0),
            "subject {id}: κ did not restart at 0: {:?}",
            s.kappas
        );
    }
    // η̂ is still handed: the starts are not cold.
    assert!(
        starts
            .iter()
            .any(|s| s.psi.iter().zip(&s.mu).any(|(p, m)| p != m)),
        "subject {id}: every start was cold, so η̂ was dropped too"
    );
}
