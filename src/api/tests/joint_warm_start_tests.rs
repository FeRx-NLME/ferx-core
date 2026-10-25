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

/// `run_sir` on a fit whose `ebe_kappas` do not line up with the population's cells, in the
/// three ways that reach `kappas_line_up` (#2379):
///
/// - a `FitResult` whose κ̂ lost a cell (a hand-edited `.fitrx` `ebes_kappa.csv`, or a fit
///   built by hand);
/// - one whose κ̂ cell is a κ too wide (the same sources, an extra column) — the guard's
///   width conjunct;
/// - the issue's own geometry, a population with other cells than the fit's. A fit with a
///   current population fingerprint refuses that first (`E_POPULATION_MISMATCH`), but one
///   without (a `.fitrx` saved before #1685) accepts it, so the guard is what stands
///   between it and a joint start of the wrong width.
///
/// In each, `kappas_line_up` must hand η̂ alone and let every κ restart at 0. Without the
/// guard the joint start has the wrong width, and `find_ebe_iov`'s warm-start width
/// `debug_assert` aborts the run — the guard's only observable effect, since
/// `iov_initial_vector` cuts a wrong-width start back to η̂ with κ = 0 anyway. So the kill
/// needs debug assertions (a debug build, or `ci-cov` in CI).
#[test]
fn run_sir_hands_eta_alone_when_the_kappas_do_not_line_up() {
    let (model, pop, opts, fit, _) = fit_recorded(warfarin_iov_fixture(), focei_sir);
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

    let mut short_cell = fit.clone();
    short_cell.ebe_kappas[0].pop();

    let mut wide_cell = fit.clone();
    let k0 = &mut wide_cell.ebe_kappas[0][0];
    *k0 = nalgebra::DVector::from_iterator(k0.len() + 1, k0.iter().copied().chain([0.1]));

    let mut no_fingerprint = fit.clone();
    no_fingerprint.population_fingerprint = None;
    let mut relabelled = pop.clone();
    let s = &mut relabelled.subjects[0];
    let one = s.occasions[0];
    for occ in s
        .occasions
        .iter_mut()
        .chain(&mut s.dose_occasions)
        .chain(&mut s.reset_occasions)
        .chain(&mut s.pk_only_occasions)
    {
        *occ = one;
    }
    assert_eq!(
        crate::stats::stacked_layout::group_count(&relabelled.subjects[0]),
        1,
        "fixture: the relabelled subject 0 has one cell"
    );

    for (label, fit, pop) in [
        ("a κ̂ cell dropped", &short_cell, &pop),
        ("a κ̂ cell one κ too wide", &wide_cell, &pop),
        ("other cells, no fingerprint", &no_fingerprint, &relabelled),
    ] {
        let (sir, rec) = on_fit_thread(&opts, || {
            record_inner_solves(|| crate::run_sir(fit, Some(&model), Some(pop), &opts))
        });
        sir.unwrap_or_else(|e| panic!("{label}: run_sir: {e}"));
        let id = &pop.subjects[0].id;
        let starts: Vec<_> = rec.starts.iter().filter(|s| &s.subject == id).collect();
        assert!(
            !starts.is_empty(),
            "{label}: fixture: subject {id} was solved"
        );
        for s in &starts {
            assert!(
                s.kappas.iter().flatten().all(|&k| k == 0.0),
                "{label}: subject {id}: κ did not restart at 0: {:?}",
                s.kappas
            );
        }
        // η̂ is still handed: the starts are not cold.
        assert!(
            starts
                .iter()
                .any(|s| s.psi.iter().zip(&s.mu).any(|(p, m)| p != m)),
            "{label}: subject {id}: every start was cold, so η̂ was dropped too"
        );
    }
}
