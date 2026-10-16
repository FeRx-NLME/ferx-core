use super::*;
use crate::estimation::inner_optimizer::{
    run_inner_loop_warm, run_inner_loop_warm_seeded, InnerHessianSeed,
};
use crate::parser::model_parser::parse_model_string;

fn fixture(iov: bool, mixed_fallback: bool) -> (CompiledModel, Population) {
    let kappa = if iov { "kappa KAPPA_CL ~ 0.03" } else { "" };
    let cl = if iov {
        "TVCL * exp(ETA_CL + KAPPA_CL)"
    } else {
        "TVCL * exp(ETA_CL)"
    };
    let model = parse_model_string(&format!(
        "[parameters]\n theta TVCL(5, 0.1, 50)\n theta TVV(50, 5, 500)\n omega ETA_CL ~ 0.04\n {kappa}\n sigma PROP ~ 0.01\n[individual_parameters]\n CL = {cl}\n V = TVV\n F = 0.7\n[structural_model]\n pk one_cpt_iv(cl=CL, v=V, f=F)\n[error_model]\n DV ~ proportional(PROP)"
    )).unwrap();
    let mut pop = make_population(3);
    for (i, s) in pop.subjects.iter_mut().enumerate() {
        s.id = (i + 1).to_string();
        s.observations = s
            .obs_times
            .iter()
            .enumerate()
            .map(|(j, &t)| 1.4 * (-0.1 * t).exp() * (0.92 + 0.04 * (i + j) as f64))
            .collect();
        if iov {
            s.occasions = vec![1, 1, 2];
            s.doses.push(DoseEvent::new(6.0, 100.0, 1, 0.0, false, 0.0));
            s.dose_occasions.push(2);
        }
    }
    if mixed_fallback {
        // The analytic provider deliberately declines rate-defined infusion
        // under F: F reshapes its duration. Other subjects remain analytic.
        pop.subjects[0].doses[0] = DoseEvent::new(0.0, 100.0, 1, 25.0, false, 0.0);
    }
    (model, pop)
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn fused_inner_and_marginal_match_separate_passes() {
    for iov in [false, true] {
        let (model, pop) = fixture(iov, false);
        let params = &model.default_params;
        for interaction in [false, true] {
            let opts = FitOptions {
                interaction,
                inner_maxiter: 4,
                min_obs_for_convergence_check: 4,
                ..Default::default()
            };
            let mu = compute_mu_k(&model, &params.theta, true);
            for width in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(width)
                    .stack_size(crate::FIT_RAYON_STACK_SIZE)
                    .build()
                    .unwrap();
                pool.install(|| {
                    // The fused pass runs the stage's BFGS seed (#1389); the separate
                    // reference must run the same one, so what is compared is fusion alone.
                    let seed = InnerHessianSeed::for_options(&opts);
                    let reference = run_inner_loop_warm_seeded(
                        &model,
                        &pop,
                        params,
                        4,
                        opts.inner_tol,
                        None,
                        Some(&mu),
                        4,
                        opts.inner_restarts,
                        seed,
                        &opts,
                    );
                    for warm in [None, Some(reference.0.as_slice())] {
                        let separate = run_inner_loop_warm_seeded(
                            &model,
                            &pop,
                            params,
                            4,
                            opts.inner_tol,
                            warm,
                            Some(&mu),
                            4,
                            opts.inner_restarts,
                            seed,
                            &opts,
                        );
                        let expected_nll = if iov {
                            crate::stats::likelihood::foce_population_nll_iov(
                                &model,
                                &pop,
                                &params.theta,
                                &separate.0,
                                &separate.1,
                                &separate.3,
                                &params.omega,
                                params.omega_iov.as_ref().unwrap(),
                                &params.sigma.values,
                                interaction,
                            )
                        } else {
                            crate::stats::likelihood::foce_population_nll(
                                &model,
                                &pop,
                                &params.theta,
                                &separate.0,
                                &separate.1,
                                &params.omega,
                                &params.sigma.values,
                                &params.residual_correlations,
                                interaction,
                            )
                        };
                        let fused = run_inner_loop_and_nll(
                            &model,
                            &pop,
                            params,
                            &opts,
                            warm,
                            Some(&mu),
                            None,
                        );
                        assert_eq!(expected_nll.to_bits(), fused.4.to_bits());
                        for (a, b) in separate.0.iter().zip(&fused.0) {
                            assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                        }
                        for (a, b) in separate.1.iter().zip(&fused.1) {
                            assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                        }
                        assert_eq!(separate.3, fused.3);
                        assert_eq!(separate.2.n_unconverged, fused.2.n_unconverged);
                        // The production fused FOCE/FOCEI path may certify an exhausted
                        // analytical BFGS point as stationary and skip the cold fallback.
                        // The strict standalone pass still runs it; the bit-identical EBEs,
                        // Hessians, and NLL above prove that skipping it is transparent.
                        assert!(fused.2.n_fallback <= separate.2.n_fallback);
                        assert_eq!(separate.2.n_start_rejected, fused.2.n_start_rejected);
                    }
                });
            }
        }
    }
}

#[test]
fn fused_dispatch_keeps_laplace_and_agq_objectives() {
    let (model, pop) = fixture(false, false);
    let params = &model.default_params;
    for (method, n_agq) in [(EstimationMethod::Laplace, 1), (EstimationMethod::FoceI, 3)] {
        let opts = FitOptions {
            method,
            n_agq,
            inner_maxiter: 3,
            ..Default::default()
        };
        // Same seed as the fused pass (exact for one-node Laplace, Gauss–Newton for AGQ), so
        // the comparison isolates the fused dispatch — including, for Laplace, the reuse of
        // the retained terminal Hessian against `pop_nll_opts`'s from-scratch anchor.
        let separate = run_inner_loop_warm_seeded(
            &model,
            &pop,
            params,
            3,
            opts.inner_tol,
            None,
            None,
            opts.min_obs_for_convergence_check as usize,
            opts.inner_restarts,
            InnerHessianSeed::for_options(&opts),
            &opts,
        );
        let expected = pop_nll_opts(
            &model,
            &pop,
            params,
            &separate.0,
            &separate.1,
            &separate.3,
            &opts,
        );
        let fused = run_inner_loop_and_nll(&model, &pop, params, &opts, None, None, None);
        assert_eq!(expected.to_bits(), fused.4.to_bits());
    }
}

#[test]
fn fused_gradient_preserves_analytic_and_subject_fallback_results() {
    for iov in [false, true] {
        for mixed in [false, true] {
            let (model, pop) = fixture(iov, mixed);
            let params = &model.default_params;
            let x = pack_params(params);
            let bounds = compute_bounds(params);
            for interaction in [false, true] {
                let opts = FitOptions {
                    interaction,
                    inner_maxiter: 4,
                    ..Default::default()
                };
                let (etas, hms, _, kappas) =
                    run_inner_loop_warm(&model, &pop, params, 4, opts.inner_tol, None, None, 0, 0);
                let per_subject = if iov {
                    crate::estimation::sens_outer_gradient::per_subject_packed_gradients_iov(
                        &model,
                        &pop,
                        params,
                        &x,
                        &etas,
                        &kappas,
                        interaction,
                    )
                } else {
                    crate::estimation::sens_outer_gradient::per_subject_packed_gradients(
                        &model,
                        &pop,
                        params,
                        &x,
                        &etas,
                        interaction,
                        None,
                    )
                };
                if mixed {
                    assert!(
                        per_subject[0].is_none(),
                        "infusion must exercise FD fallback"
                    );
                    assert!(
                        per_subject[1].is_some(),
                        "bolus must keep analytic gradient"
                    );
                }
                let mut expected = vec![0.0; x.len()];
                for (i, g) in per_subject.clone().into_iter().enumerate() {
                    let g = match g {
                        Some(g) if g.iter().all(|v| v.is_finite()) => g,
                        _ if iov => subject_reconverged_fd_gradient_iov(
                            &x,
                            params,
                            &model,
                            &pop.subjects[i],
                            &etas[i],
                            &bounds,
                            &opts,
                        ),
                        _ => held_ebe_salvage(
                            &x, params, &model, &pop, i, &etas[i], &hms[i], &bounds, &opts,
                        ),
                    };
                    for (acc, value) in expected.iter_mut().zip(g) {
                        *acc += 2.0 * value;
                    }
                }
                // #1154: the same assembly records which subjects it salvaged onto the
                // per-subject FD gradient. Asserted here, inside the existing
                // `iov` × `mixed` sweep, because it is the only place both the IOV and
                // the non-IOV `declines.record(i)` arms are exercised — a decline log
                // tested on one engine only is an assertion against a constant on the
                // other (mutating the IOV `record` alone left the whole suite green).
                let declines = OuterFdDeclineLog::new(pop.subjects.len());
                let actual = if iov {
                    population_gradient_sens_iov_mixed(
                        &x,
                        params,
                        &model,
                        &pop,
                        &etas,
                        &kappas,
                        &bounds,
                        &opts,
                        OuterTrial::unknown(),
                        &declines,
                    )
                } else {
                    population_gradient_sens_mixed(
                        &x,
                        params,
                        &model,
                        &pop,
                        &etas,
                        &hms,
                        &bounds,
                        &opts,
                        OuterTrial::unknown(),
                        &declines,
                    )
                };
                assert_eq!(bits(&expected), bits(&actual));
                let expected_declines: Vec<usize> = if mixed { vec![0] } else { vec![] };
                assert_eq!(
                    declines.declined_indices(),
                    expected_declines,
                    "iov = {iov}, mixed = {mixed}, interaction = {interaction}: the decline \
                     log must name exactly the subjects the assembly sent to FD"
                );
                assert_eq!(declines.skipped_salvages(), 0, "no incumbent, no skip");
                if mixed {
                    // #1520: at a blown-up rejected trial the declining subject 0
                    // contributes nothing instead of buying the salvage — on **both**
                    // engines, asserted here for the same reason the decline log is: the
                    // IOV arm is exercised nowhere else. The trial context is synthetic
                    // (the guard reads only it): the population and subject 0 are 1e6
                    // units over an incumbent of 0, far beyond `BLOWN_UP_EXCESS_PER_OBS`.
                    let n = pop.subjects.len();
                    let contribs = vec![0.0; n];
                    let mut best = vec![0.0; n];
                    best[0] = -1e6;
                    let blown = OuterTrial {
                        ofv: 1e6,
                        contribs: &contribs,
                        incumbent: Some((0.0, &best)),
                    };
                    let declines = OuterFdDeclineLog::new(n);
                    let guarded = if iov {
                        population_gradient_sens_iov_mixed(
                            &x, params, &model, &pop, &etas, &kappas, &bounds, &opts, blown,
                            &declines,
                        )
                    } else {
                        population_gradient_sens_mixed(
                            &x, params, &model, &pop, &etas, &hms, &bounds, &opts, blown, &declines,
                        )
                    };
                    // The assembly's own summation order with subject 0's term zero.
                    let mut expected_guarded = vec![0.0; x.len()];
                    for (i, g) in per_subject.iter().enumerate() {
                        let g = if i == 0 {
                            vec![0.0; x.len()]
                        } else {
                            g.clone().expect("subject 1 is analytic")
                        };
                        for (acc, value) in expected_guarded.iter_mut().zip(g) {
                            *acc += 2.0 * value;
                        }
                    }
                    assert_eq!(
                        declines.skipped_salvages(),
                        1,
                        "iov = {iov}, interaction = {interaction}: one salvage skipped"
                    );
                    assert_eq!(declines.declined_indices(), vec![0], "still recorded");
                    assert_eq!(
                        bits(&expected_guarded),
                        bits(&guarded),
                        "iov = {iov}, interaction = {interaction}: subject 0 contributes \
                         nothing"
                    );
                    assert_ne!(
                        bits(&expected),
                        bits(&guarded),
                        "fixture: the salvaged subject's gradient must be non-zero, or \
                         the arms are indistinguishable here"
                    );
                }
            }
        }
    }
}

#[test]
fn inner_map_completes_each_subject_before_returning_its_buffers() {
    let (model, pop) = fixture(false, false);
    let params = &model.default_params;
    let result = run_inner_loop_warm_map(
        &model,
        &pop,
        params,
        3,
        1e-6,
        None,
        None,
        0,
        0,
        InnerSolvePolicy::default(),
        None,
        |subject, ebe| {
            (
                subject.id.clone(),
                ebe.eta.as_ptr() as usize,
                ebe.h_matrix.as_ptr() as usize,
            )
        },
    );
    for (i, (id, eta_ptr, h_ptr)) in result.4.iter().enumerate() {
        assert_eq!(id, &pop.subjects[i].id);
        assert_eq!(
            *eta_ptr,
            result.0[i].as_ptr() as usize,
            "eta buffer was copied after EBE"
        );
        assert_eq!(
            *h_ptr,
            result.1[i].as_ptr() as usize,
            "Jacobian buffer was copied after EBE"
        );
    }
}

// ── #574: the inner pass polls the cancel flag per subject ───────────────────────────
//
// The outer optimizers poll `FitOptions.cancel` only between objective evaluations, and
// one evaluation of a large ODE population is minutes of EBE solves — so an R Ctrl-C used
// to wait out the whole pass. These pin that a cancel landing *inside* a pass stops the
// subjects not yet started.

/// A cancel flipped by the first subject's finish skips every later subject. One worker
/// thread makes "later" deterministic. The same run with the flag present but never
/// flipped is the other side of the gate: every subject is solved and none is rejected.
#[test]
fn a_cancel_during_the_inner_pass_skips_every_subject_not_yet_started() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for iov in [false, true] {
        let (model, pop) = fixture(iov, false);
        let n = pop.subjects.len();
        let params = &model.default_params;
        // A non-zero warm start, so the placeholder's η is distinguishable from zeros.
        let warm: Vec<DVector<f64>> = (0..n)
            .map(|i| DVector::from_element(model.n_eta, 0.01 * (i + 1) as f64))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .stack_size(crate::FIT_RAYON_STACK_SIZE)
            .build()
            .unwrap();
        let run = |flip: bool| {
            let flag = crate::cancel::CancelFlag::new();
            let finished = AtomicUsize::new(0);
            let out = pool.install(|| {
                run_inner_loop_warm_map(
                    &model,
                    &pop,
                    params,
                    20,
                    1e-6,
                    Some(&warm),
                    None,
                    0,
                    0,
                    InnerSolvePolicy::default(),
                    Some(&flag),
                    |_, ebe| {
                        finished.fetch_add(1, Ordering::SeqCst);
                        if flip {
                            flag.cancel();
                        }
                        ebe.nll
                    },
                )
            });
            (out, finished.into_inner())
        };

        let (control, control_finished) = run(false);
        assert_eq!(
            control_finished, n,
            "iov={iov}: control must solve every subject"
        );
        assert_eq!(
            control.2.n_start_rejected, 0,
            "iov={iov}: control rejected a subject"
        );
        assert!(control.4.iter().all(|v| v.is_finite()), "iov={iov}");
        if iov {
            assert!(
                !control.3[1].is_empty(),
                "fixture: IOV subjects must carry κ"
            );
        }

        let (cancelled, cancelled_finished) = run(true);
        assert_eq!(
            cancelled_finished, 1,
            "iov={iov}: subjects after the cancel must not be solved"
        );
        assert_eq!(
            cancelled.2.n_start_rejected,
            n - 1,
            "iov={iov}: every skipped subject must hard-reject the trial"
        );
        // The solved subject is untouched; the skipped ones are placeholders with the
        // shapes of a real result, so nothing downstream indexes out of bounds.
        assert_eq!(
            cancelled.0[0].as_slice(),
            control.0[0].as_slice(),
            "iov={iov}"
        );
        for i in 1..n {
            assert_eq!(
                cancelled.0[i].as_slice(),
                warm[i].as_slice(),
                "iov={iov} subj {i}"
            );
            assert_eq!(
                cancelled.1[i].shape(),
                control.1[i].shape(),
                "iov={iov} subj {i}"
            );
            assert!(
                cancelled.1[i].iter().all(|&v| v == 0.0),
                "iov={iov} subj {i}"
            );
            assert_eq!(
                cancelled.3[i].len(),
                control.3[i].len(),
                "iov={iov} subj {i}"
            );
            assert_eq!(
                cancelled.4[i], 0.0,
                "iov={iov}: skipped subject ran its finish"
            );
        }
    }
}

/// The stage-seeded entry point the trust region, Gauss–Newton and the final inner pass
/// use reads the cancel flag off the stage options it is handed — not only callers that
/// pass a flag explicitly.
#[test]
fn the_stage_seeded_inner_pass_reads_the_stage_cancel_flag() {
    let (model, pop) = fixture(false, false);
    let params = &model.default_params;
    let base = FitOptions {
        inner_maxiter: 20,
        ..Default::default()
    };
    let run = |stage: &FitOptions| {
        run_inner_loop_warm_seeded(
            &model,
            &pop,
            params,
            20,
            1e-6,
            None,
            None,
            0,
            0,
            InnerHessianSeed::for_options(stage),
            stage,
        )
        .2
        .n_start_rejected
    };
    assert_eq!(run(&base), 0, "control rejected a subject");
    let flag = crate::cancel::CancelFlag::new();
    flag.cancel();
    let cancelled = FitOptions {
        cancel: Some(flag),
        ..base
    };
    assert_eq!(run(&cancelled), pop.subjects.len());
}

/// The FOCE/FOCEI pass the outer optimizers score never reports a cancelled pass as an
/// objective: the placeholders' zero contributions would otherwise sum to a finite — and
/// spuriously low — OFV, which the derivative-free and legacy-BFGS callers do not guard
/// against. Both branches of `run_inner_loop_and_nll_prepared`: the FOCE marginal, and
/// the AGQ quadrature, which must not run on placeholders at all.
#[test]
fn a_cancelled_inner_pass_scores_no_objective() {
    let (model, pop) = fixture(false, false);
    let params = &model.default_params;
    for method in [EstimationMethod::FoceI, EstimationMethod::Laplace] {
        let base = FitOptions {
            method,
            inner_maxiter: 20,
            ..Default::default()
        };
        assert_eq!(
            base.agq_nodes().is_some(),
            method == EstimationMethod::Laplace,
            "fixture: one arm per branch"
        );
        let (_, _, stats, _, nll) =
            run_inner_loop_and_nll(&model, &pop, params, &base, None, None, None);
        assert!(
            nll.is_finite(),
            "{method:?}: control objective must be finite"
        );
        assert_eq!(stats.n_start_rejected, 0, "{method:?}");

        let flag = crate::cancel::CancelFlag::new();
        flag.cancel();
        let cancelled = FitOptions {
            cancel: Some(flag),
            ..base
        };
        let (etas, _, stats, _, nll) =
            run_inner_loop_and_nll(&model, &pop, params, &cancelled, None, None, None);
        assert_eq!(
            nll,
            f64::INFINITY,
            "{method:?}: a cancelled pass scored {nll}"
        );
        assert_eq!(stats.n_start_rejected, pop.subjects.len(), "{method:?}");
        assert_eq!(etas.len(), pop.subjects.len(), "{method:?}");
    }
}
