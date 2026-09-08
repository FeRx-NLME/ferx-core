use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::stats::likelihood::{individual_nll_iov, individual_nll_iov_with_scratch};
use std::collections::HashMap;

fn model(ode: bool) -> CompiledModel {
    let structural = if ode {
        "ode(states=[central])\n[odes]\n init(central) = BASE\n d/dt(central) = -(CL/V)*central\n[scaling]\n y = central/V"
    } else {
        "pk one_cpt_iv(cl=CL, v=V)"
    };
    parse_model_string(&format!(
        "[parameters]\n theta TVCL(3, 0.1, 30)\n theta TVV(20, 2, 200)\n omega ETA_CL ~ 0.04\n kappa KAPPA_CL ~ 0.03\n sigma PROP ~ 0.04\n[individual_parameters]\n CL = TVCL * (WT/70) * exp(ETA_CL + KAPPA_CL) * (1 + 0.001*TIME)\n V = TVV\n BASE = WT\n[structural_model]\n {structural}\n[error_model]\n DV ~ proportional(PROP)"
    )).unwrap()
}
fn cov(wt: f64) -> HashMap<String, f64> {
    HashMap::from([("WT".into(), wt)])
}
fn subject() -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![
            DoseEvent::new(0., 100., 1, 0., false, 0.),
            DoseEvent::new(24., 100., 1, 0., false, 0.),
        ],
        dose_occasions: vec![11, 22],
        dose_covariates: vec![cov(75.), cov(105.)],
        obs_times: vec![1., 3., 25., 27.],
        observations: vec![1.; 4],
        obs_cmts: vec![1; 4],
        occasions: vec![11, 11, 22, 22],
        obs_covariates: vec![cov(80.), cov(90.), cov(100.), cov(110.)],
        covariates: cov(70.),
        cens: vec![0; 4],
        pk_only_times: vec![10.],
        pk_only_covariates: vec![cov(95.)],
        reset_times: vec![23.],
        reset_covariates: vec![cov(120.)],
        reset_occasions: vec![22],
        ..Default::default()
    }
}
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn pointers(s: &EventPkParams) -> [usize; 4] {
    [
        s.dose.as_ptr() as usize,
        s.obs.as_ptr() as usize,
        s.pk_only.as_ptr() as usize,
        s.reset.as_ptr() as usize,
    ]
}

#[test]
fn lazy_event_scratch_matches_preallocated_predictions_and_reuses_storage() {
    for ode in [false, true] {
        let m = model(ode);
        let subj = subject();
        let mut lazy = EventPkParams::default();
        let mut eager = EventPkParams::with_capacity_for(&subj);
        let mut storage = None;
        for eta in [0.1, -0.3, 0.1] {
            let a = compute_predictions_with_tv_into(&m, &subj, &[3., 20.], &[eta, 0.], &mut lazy);
            let b = compute_predictions_with_tv_into(&m, &subj, &[3., 20.], &[eta, 0.], &mut eager);
            assert_eq!(bits(&a), bits(&b));
            assert!(a.iter().all(|v| v.is_finite()));
            if let Some(before) = storage {
                assert_eq!(before, pointers(&lazy));
            }
            storage = Some(pointers(&lazy));
        }
    }
}

#[test]
fn static_predictions_leave_event_scratch_unallocated() {
    let m = parse_model_string(
        "[parameters]\n theta TVCL(3, 0.1, 30)\n theta TVV(20, 2, 200)\n sigma PROP ~ 0.04\n[individual_parameters]\n CL = TVCL\n V = TVV\n[structural_model]\n pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n DV ~ proportional(PROP)"
    ).unwrap();
    let subj = Subject {
        doses: vec![DoseEvent::new(0., 100., 1, 0., false, 0.)],
        obs_times: vec![1., 3.],
        observations: vec![1.; 2],
        obs_cmts: vec![1; 2],
        ..Default::default()
    };
    let mut scratch = EventPkParams::default();
    let result = compute_predictions_with_tv_into(&m, &subj, &[3., 20.], &[], &mut scratch);
    assert_eq!(result.len(), 2);
    assert!(result.iter().all(|v| v.is_finite() && *v > 0.));
    assert_eq!(
        [
            scratch.dose.capacity(),
            scratch.obs.capacity(),
            scratch.pk_only.capacity(),
            scratch.reset.capacity()
        ],
        [0; 4]
    );
}

#[test]
fn iov_scratch_reuses_capacity_but_refreshes_every_event_snapshot() {
    let subj = subject();
    for ode in [false, true] {
        let m = model(ode);
        let mut scratch = EventPkParams::default();
        let mut prior = None;
        let mut storage = None;
        for (cl, eta, k1, k2) in [
            (3., 0.1, 0.3, -0.2),
            (4., -0.3, -0.6, 0.5),
            (3., 0.1, 0.3, -0.2),
        ] {
            let theta = [cl, 20.];
            let kappas = vec![vec![k1], vec![k2]];
            let fresh = predict_iov(&m, &subj, &theta, &[eta], &kappas);
            let reused = predict_iov_with_scratch(&m, &subj, &theta, &[eta], &kappas, &mut scratch);
            assert_eq!(bits(&fresh), bits(&reused));
            assert!(reused.iter().all(|v| v.is_finite() && *v > 0.));
            if let Some(before) = prior {
                assert_ne!(before, bits(&reused));
            }
            prior = Some(bits(&reused));
            if let Some(before) = storage {
                assert_eq!(before, pointers(&scratch));
            }
            storage = Some(pointers(&scratch));
            let expected_cl = |wt: f64, time: f64, k: f64| {
                cl * (wt / 70.) * (eta + k).exp() * (1. + 0.001 * time)
            };
            for (j, wt) in [80., 90., 100., 110.].into_iter().enumerate() {
                let k = if j < 2 { k1 } else { k2 };
                approx::assert_relative_eq!(
                    scratch.obs[j].cl(),
                    expected_cl(wt, subj.obs_times[j], k),
                    max_relative = 1e-13
                );
            }
            approx::assert_relative_eq!(
                scratch.dose[1].cl(),
                expected_cl(105., 24., k2),
                max_relative = 1e-13
            );
            approx::assert_relative_eq!(
                scratch.pk_only[0].cl(),
                expected_cl(95., 10., 0.),
                max_relative = 1e-13
            );
            if ode {
                approx::assert_relative_eq!(
                    scratch.reset[0].cl(),
                    expected_cl(120., 23., k2),
                    max_relative = 1e-13
                );
            } else {
                assert!(scratch.reset.is_empty());
            }
        }
    }
}

#[test]
fn iov_scratch_shrinks_and_clears_unused_event_vectors() {
    let ode = model(true);
    let analytical = model(false);
    let mut subj = subject();
    let mut scratch = EventPkParams::default();
    let theta = [3., 20.];
    let kappas = vec![vec![0.2], vec![-0.1]];
    predict_iov_with_scratch(&ode, &subj, &theta, &[0.1], &kappas, &mut scratch);
    assert_eq!(scratch.reset.len(), 1);
    let capacity = scratch.obs.capacity();
    subj.obs_times.truncate(1);
    subj.observations.truncate(1);
    subj.obs_cmts.truncate(1);
    subj.obs_covariates.truncate(1);
    subj.occasions.truncate(1);
    subj.cens.truncate(1);
    subj.pk_only_times.clear();
    subj.pk_only_covariates.clear();
    let actual =
        predict_iov_with_scratch(&analytical, &subj, &theta, &[0.1], &kappas, &mut scratch);
    assert_eq!(
        bits(&actual),
        bits(&predict_iov(&analytical, &subj, &theta, &[0.1], &kappas))
    );
    assert_eq!(scratch.obs.len(), 1);
    assert_eq!(scratch.obs.capacity(), capacity);
    assert!(scratch.pk_only.is_empty() && scratch.reset.is_empty());
    // A Gaussian-only subject with no observations still gets a dose-side
    // snapshot. Its observation, pk-only and unused reset storage must be empty.
    subj.obs_times.clear();
    subj.observations.clear();
    subj.obs_cmts.clear();
    subj.obs_covariates.clear();
    subj.occasions.clear();
    subj.cens.clear();
    subj.doses = vec![DoseEvent::new(0., 100., 1, 0., false, 0.)];
    subj.dose_occasions = vec![11];
    subj.dose_covariates = vec![cov(75.)];
    assert!(
        predict_iov_with_scratch(&analytical, &subj, &theta, &[0.1], &kappas, &mut scratch)
            .is_empty()
    );
    assert_eq!(scratch.dose.len(), 1);
    assert!(scratch.obs.is_empty() && scratch.pk_only.is_empty() && scratch.reset.is_empty());
}

#[test]
fn iov_likelihood_with_scratch_matches_fresh_calls_at_changed_parameters() {
    let subj = subject();
    let m = model(false);
    let params = &m.default_params;
    let mut scratch = EventPkParams::default();
    for shift in [-0.3, 0.1, 0.5] {
        let theta = [3. + shift, 20.];
        let eta = [shift];
        let kappas = vec![vec![shift * 2.], vec![-shift]];
        let expected = individual_nll_iov(
            &m,
            &subj,
            &theta,
            &eta,
            &kappas,
            &params.omega,
            params.omega_iov.as_ref(),
            &params.sigma.values,
        );
        let actual = individual_nll_iov_with_scratch(
            &m,
            &subj,
            &theta,
            &eta,
            &kappas,
            &params.omega,
            params.omega_iov.as_ref(),
            &params.sigma.values,
            &mut scratch,
        );
        assert!(actual.is_finite());
        assert_eq!(expected.to_bits(), actual.to_bits());
    }
}

#[test]
fn reused_iov_predictions_match_dual2_eta_derivatives() {
    let subj = subject();
    let m = model(false);
    let theta = [3., 20.];
    let effects = [0.1, 0.3, -0.2];
    let dual = crate::sens::provider::subject_sensitivities_iov(&m, &subj, &theta, &effects)
        .expect("fixture must exercise Dual2, not FD fallback");
    let mut scratch = EventPkParams::default();
    let mut worst: f64 = 0.;
    for axis in 0..effects.len() {
        let h = 1e-5;
        let mut plus = effects;
        let mut minus = effects;
        plus[axis] += h;
        minus[axis] -= h;
        let mut predict = |e: &[f64; 3]| {
            predict_iov_with_scratch(
                &m,
                &subj,
                &theta,
                &e[..1],
                &[vec![e[1]], vec![e[2]]],
                &mut scratch,
            )
        };
        let a = predict(&plus);
        let b = predict(&minus);
        for (j, obs) in dual.obs.iter().enumerate() {
            let fd = (a[j] - b[j]) / (2. * h);
            let analytic = obs.df_deta[axis];
            assert!(fd.is_finite() && analytic.is_finite());
            let error = (analytic - fd).abs() / (1. + fd.abs());
            worst = worst.max(error);
            // Measured worst scaled error: 1.76e-11 on Windows/nightly.
            // 1e-9 leaves about 57x headroom for platform rounding.
            assert!(
                error < 1e-9,
                "axis {axis}, obs {j}: Dual2={analytic}, FD={fd}"
            );
        }
    }
    eprintln!("scratch Dual2/FD worst scaled error: {worst:e}");
}
