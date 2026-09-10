use super::*;
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
                    let reference = run_inner_loop_warm(
                        &model,
                        &pop,
                        params,
                        4,
                        opts.inner_tol,
                        None,
                        Some(&mu),
                        4,
                        opts.inner_restarts,
                    );
                    for warm in [None, Some(reference.0.as_slice())] {
                        let separate = run_inner_loop_warm(
                            &model,
                            &pop,
                            params,
                            4,
                            opts.inner_tol,
                            warm,
                            Some(&mu),
                            4,
                            opts.inner_restarts,
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
                        let fused =
                            run_inner_loop_and_nll(&model, &pop, params, &opts, warm, Some(&mu));
                        assert_eq!(expected_nll.to_bits(), fused.4.to_bits());
                        for (a, b) in separate.0.iter().zip(&fused.0) {
                            assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                        }
                        for (a, b) in separate.1.iter().zip(&fused.1) {
                            assert_eq!(bits(a.as_slice()), bits(b.as_slice()));
                        }
                        assert_eq!(separate.3, fused.3);
                        assert_eq!(separate.2.n_unconverged, fused.2.n_unconverged);
                        assert_eq!(separate.2.n_fallback, fused.2.n_fallback);
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
        let separate = run_inner_loop_warm(
            &model,
            &pop,
            params,
            3,
            opts.inner_tol,
            None,
            None,
            opts.min_obs_for_convergence_check as usize,
            opts.inner_restarts,
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
        let fused = run_inner_loop_and_nll(&model, &pop, params, &opts, None, None);
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
                let (etas, _, _, kappas) =
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
                for (i, g) in per_subject.into_iter().enumerate() {
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
                            per_subject_noise_abs(opts.outer_fd_noise_abs, pop.subjects.len()),
                        ),
                        _ => subject_reconverged_fd_gradient(
                            &x,
                            params,
                            &model,
                            &pop.subjects[i],
                            &etas[i],
                            &bounds,
                            &opts,
                            per_subject_noise_abs(opts.outer_fd_noise_abs, pop.subjects.len()),
                        ),
                    };
                    for (acc, value) in expected.iter_mut().zip(g) {
                        *acc += 2.0 * value;
                    }
                }
                let actual = if iov {
                    population_gradient_sens_iov_mixed(
                        &x, params, &model, &pop, &etas, &kappas, &bounds, &opts,
                    )
                } else {
                    population_gradient_sens_mixed(&x, params, &model, &pop, &etas, &bounds, &opts)
                };
                assert_eq!(bits(&expected), bits(&actual));
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
        InnerFdConfig::fixed(),
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
