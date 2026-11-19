#[allow(unused_imports)]
use ferx_model::internal::*;
// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::pk;
use ferx_model::stats::residual_error::residual_rd;
use ferx_model::stats::residual_error::residual_rd2;
use ferx_model::stats::residual_error::ResidualCov;
use ferx_model::stats::special::log_normal_cdf;
use ferx_model::types::*;
use nalgebra::DMatrix;
use nalgebra::DVector;

use ferx_model::stats::likelihood::*;

/// The IOV marginal scores non-Gaussian records: with κ on a parameter no record reads
/// (κ̂ = 0) it equals the NONMEM-anchored non-IOV marginal, for TTE and binary, under
/// both interaction settings.
#[cfg(feature = "survival")]
#[test]
fn the_iov_marginal_scores_the_non_gaussian_term() {
    use crate::types::{DoseEvent, EventType, ObsRecord};
    let root = env!("CARGO_MANIFEST_DIR");
    let tte = std::fs::read_to_string(format!("{root}/examples/tte_exponential.ferx")).unwrap();
    let tte_iov = tte
        .replace(
            "  omega ETA_LAMBDA ~ 0.09\n",
            "  omega ETA_LAMBDA ~ 0.09\n  kappa KAPPA_CL ~ 0.04\n",
        )
        .replace("CL     = DUMMY_CL", "CL     = DUMMY_CL * exp(KAPPA_CL)")
        .replace("  method  = focei\n", "  method  = focei\n  iov_column = OCC\n");
    let binary = "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 500.0)\n  \
                  theta TH0(0.3, -10.0, 10.0)\n  omega ETA_I ~ 0.3\n  sigma PROP ~ 0.05\n\
                  [individual_parameters]\n  CL = TVCL\n  V  = TVV\n\
                  [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
                  [error_model]\n  DV ~ proportional(PROP)\n\
                  [binary_model]\n  cmt   = 3\n  logit = TH0 + ETA_I\n\
                  [fit_options]\n  method = focei\n"
        .to_string();
    let binary_iov = binary
        .replace("  omega ETA_I ~ 0.3\n", "  omega ETA_I ~ 0.3\n  kappa KAPPA_CL ~ 0.04\n")
        .replace("CL = TVCL\n", "CL = TVCL * exp(KAPPA_CL)\n")
        .replace("  method = focei\n", "  method = focei\n  iov_column = OCC\n");
    let event = |time: f64, event_type: EventType, entry_time: f64| ObsRecord::Event {
        time,
        event_type,
        entry_time,
        cmt: 2,
    };
    let state =
        |time: f64, state: usize| ObsRecord::DiscreteState { time, raw_time: time, state, cmt: 3 };
    let cases = [
        (
            "tte",
            tte,
            tte_iov,
            vec![event(7.5, EventType::Exact, 0.0), event(30.0, EventType::RightCensored, 7.5)],
        ),
        (
            "binary",
            binary,
            binary_iov,
            vec![state(1.0, 1), state(5.0, 0), state(13.0, 1), state(20.0, 1)],
        ),
    ];
    for (tag, plain_src, iov_src, records) in cases {
        let plain = crate::parser::model_parser::parse_model_string(&plain_src).unwrap();
        let iov = crate::parser::model_parser::parse_model_string(&iov_src).unwrap();
        assert_eq!(iov.n_kappa, 1, "{tag}");
        // Two dose-only occasions on the (unobserved) PK compartment give two κ groups.
        let s = Subject {
            id: "1".into(),
            doses: vec![
                DoseEvent::new(0.0, 10.0, 1, 0.0, false, 0.0),
                DoseEvent::new(12.0, 10.0, 1, 0.0, false, 0.0),
            ],
            dose_occasions: vec![1, 2],
            obs_records: records,
            ..Default::default()
        };
        assert_eq!(crate::stats::stacked_layout::group_count(&s), 2);
        let theta = &plain.default_params.theta;
        let eta = DVector::from_vec(vec![0.12]);
        let h = DMatrix::<f64>::zeros(0, 1);
        let kappas = vec![DVector::zeros(1), DVector::zeros(1)];
        let no_records = Subject { obs_records: Vec::new(), ..s.clone() };
        let iov_nll = |subject: &Subject, interaction: bool| {
            foce_subject_nll_iov(
                &iov,
                subject,
                theta,
                &eta,
                &h,
                &iov.default_params.omega,
                &iov.default_params.sigma.values,
                interaction,
                &kappas,
                iov.default_params.omega_iov.as_ref().unwrap(),
            )
        };
        for interaction in [true, false] {
            let want = foce_subject_nll_cov(
                &plain,
                &s,
                theta,
                &eta,
                &h,
                &plain.default_params.omega,
                &plain.default_params.sigma.values,
                plain.residual_cov_with(&[], theta),
                interaction,
            );
            let got = iov_nll(&s, interaction);
            let dropped = iov_nll(&no_records, interaction);
            assert!(want.is_finite() && got.is_finite(), "{tag}: {want} {got}");
            assert!(
                (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                "{tag} interaction={interaction}: IOV {got} vs non-IOV {want}"
            );
            // The term is live: without it the objective is a different number.
            assert!((dropped - want).abs() > 1e-2, "{tag}: {dropped} vs {want}");
        }
    }
}

/// Under nested levels the IOV linearisation's compact κ columns are exact (the
/// per-cell recording over the compact axes), replacing two solves per compact κ:
/// against the forced central difference they agree to the quotient's accuracy, and
/// they differ in the bits (the exact branch ran). The η columns are the caller's.
#[test]
fn nested_levels_linearize_with_exact_kappa_columns() {
    let (model, pop, _) = crate::estimation::inner_optimizer::test_helpers::nested_levels_fixture();
    assert!(model.has_nested_levels());
    let params = &model.default_params;
    let omega_iov = params.omega_iov.as_ref().expect("an Ω_IOV");
    let mut worst = 0.0f64;
    for s in pop.subjects.iter().take(3) {
        let n_eta = model.n_eta;
        let layout = crate::stats::stacked_layout::StackedLayout::new(n_eta, model.n_kappa, s);
        let kappas: Vec<DVector<f64>> = (0..layout.n_groups())
            .map(|g| DVector::from_element(model.n_kappa, 0.01 * (g as f64 + 1.0)))
            .collect();
        let eta = DVector::from_element(n_eta, 0.04);
        let h = DMatrix::from_element(s.obs_times.len(), n_eta, 0.2);
        let run = |force: bool| {
            NESTED_KAPPA_FD_FORCE.with(|c| c.set(force));
            let l = iov_linearization(
                &model,
                s,
                &params.theta,
                &eta,
                &h,
                &params.omega,
                &kappas,
                omega_iov,
            )
            .expect("linearisation");
            NESTED_KAPPA_FD_FORCE.with(|c| c.set(false));
            l.h_full
        };
        let (he, hf) = (run(false), run(true));
        assert_eq!(he.ncols(), hf.ncols());
        assert!(
            he.iter().zip(hf.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
            "the exact κ branch did not run"
        );
        let scale = hf.iter().fold(1.0f64, |m, &v| m.max(v.abs()));
        for (a, b) in he.iter().zip(hf.iter()) {
            assert!(a.is_finite() && b.is_finite());
            worst = worst.max((a - b).abs() / scale);
        }
    }
    eprintln!("exact nested κ columns vs the central difference: worst {worst:.2e}");
    // Measured 2.2e-10: the quotient's truncation at its `1e-6` step. A column read in
    // the wrong compact order misses by O(1).
    assert!(worst < 1e-6, "{worst:.2e}");
}
use crate::types::{
    BloqMethod, DoseEvent, EndpointError, ErrorModel, ErrorSpec, GradientMethod, PkParams,
    ResidualCorrelation,
};
use approx::assert_relative_eq;
use std::collections::HashMap;

fn make_simple_subject() -> Subject {
    Subject {
        id: "1".to_string(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        observations: vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0],
        obs_cmts: vec![1; 6],
        cens: vec![0; 6],
        occasions: vec![1, 1, 1, 2, 2, 2],
        ..Default::default()
    }
}

fn make_omega(var: f64) -> OmegaMatrix {
    OmegaMatrix::from_diagonal(&[var], vec!["ETA_CL".into()])
}

fn make_model() -> CompiledModel {
    CompiledModel {
        pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
            let mut p = PkParams::default();
            p.values[0] = theta[0] * eta[0].exp(); // CL uses combined eta[0]
            p.values[1] = theta[1]; // V
            p
        }),
        gradient_method: GradientMethod::default(),
        ..crate::types::test_helpers::bare_model(crate::types::ModelParameters {
            theta_lower: vec![0.01, 1.0],
            theta_upper: vec![100.0, 500.0],
            sigma: crate::types::SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            ..crate::types::test_helpers::bare_params(make_omega(0.09))
        })
    }
}

#[test]
fn obs_nll_subject_into_uses_cross_endpoint_residual_covariance() {
    let mut model = make_model();
    model.error_spec = ErrorSpec::PerCmt(HashMap::from([
        (1, EndpointError { error_model: ErrorModel::Proportional, sigma_idx: vec![1] }),
        (2, EndpointError { error_model: ErrorModel::Proportional, sigma_idx: vec![0] }),
    ]));
    model.residual_correlations = vec![ResidualCorrelation { sigma_i: 0, sigma_j: 1, rho: 0.5 }];

    let mut subj = make_simple_subject();
    subj.obs_times = vec![1.0, 1.0];
    subj.obs_raw_times = Vec::new();
    subj.observations = vec![1.8, 2.4];
    subj.obs_cmts = vec![1, 2];
    subj.cens = vec![0, 0];
    subj.occasions = vec![1, 1];
    subj.fremtype = Vec::new();
    let theta = vec![5.0, 50.0];
    let eta = vec![0.0];
    let sigma = vec![0.2, 0.3];
    let mut scratch = pk::EventPkParams::default();

    let corr = model.residual_correlations.clone();
    let dense = obs_nll_subject_into(
        &model,
        &subj,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&corr),
        &eta,
        &mut scratch,
    );
    model.residual_correlations.clear();
    let diagonal = obs_nll_subject_into(
        &model,
        &subj,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &eta,
        &mut scratch,
    );

    assert!(dense.is_finite());
    assert!(diagonal.is_finite());
    assert!(
        (dense - diagonal).abs() > 1e-6,
        "cross-endpoint residual covariance must contribute to SAEM obs NLL"
    );
}

#[test]
fn m3_logcdf_uses_upper_tail_for_negative_cens() {
    let sd = 2.0;
    let f = 12.0;
    let uloq = 10.0;
    let lloq = 14.0;

    let upper = m3_logcdf(uloq, f, sd, -1);
    let lower = m3_logcdf(lloq, f, sd, 1);

    assert!((upper - log_normal_cdf(1.0)).abs() < 1e-12);
    assert!((lower - log_normal_cdf(1.0)).abs() < 1e-12);

    // NONMEM 7.5.1 anchor in tests/nonmem/right_censored_m3.{ctl,csv,lst}:
    // two identical CENS=-1 rows with z=(12-10)/2=1 give OFV
    // 0.69101514210943182. ferx uses the A&S CDF approximation, so compare
    // within a numerical tolerance rather than bit-for-bit.
    let nonmem_ofv = 0.691_015_142_109_431_8;
    let ferx_ofv = -4.0 * upper;
    assert!((ferx_ofv - nonmem_ofv).abs() < 1e-6);
}

/// #1496, `m3_logcdf` (`cens < 0`): an out-of-domain `CENS` scores by its sign (`7` as
/// `1`, `-2` as `-1`); the two tails differ here (z = ∓1).
#[test]
fn m3_logcdf_scores_an_out_of_domain_flag_by_its_sign() {
    let at = |cens: i8| m3_logcdf(10.0, 12.0, 2.0, cens);
    assert!(at(1).is_finite() && at(-1).is_finite());
    assert_ne!(at(1).to_bits(), at(-1).to_bits(), "the two tails differ here");
    for (flag, code) in [(7, 1), (i8::MAX, 1), (-2, -1), (i8::MIN, -1)] {
        assert_eq!(
            at(flag).to_bits(),
            at(code).to_bits(),
            "m3_logcdf: CENS={flag} must score as CENS={code}"
        );
    }
}

/// #2762: an M3 limit so far on the certain side that `|z| ≈ 1.9e154` (`normal_cdf`
/// overflows) scored NaN (`warfarin_bloq.ferx`, `LLOQ = 1e300` → 2e20 sentinel). Its term
/// is `0`; through the conditional, FOCE and FOCEI objectives the far limit must score
/// bit-for-bit as a `±1e6` one.
#[test]
fn a_censored_limit_past_the_normal_cdf_overflow_scores_as_certain() {
    let theta = [5.0, 50.0];
    let eta = [0.1];
    let omega = make_omega(0.09);
    let sigma = [0.2];
    // Straddle, on the fixture row's own z (review r1 #7): the near limit stays below
    // the gate (`(z/√2)²` finite) and the far one reaches it (`(z/√2)²` overflows).
    {
        let model = make_model();
        let subject = make_simple_subject();
        let f = model_predictions(&model, &subject, &theta, &eta)[5];
        let key = model.error_spec.obs_key(&subject, 5);
        let sd = model.error_spec.variance_at(key, f, &sigma).sqrt();
        let sq = |limit: f64| ((limit - f) / sd / std::f64::consts::SQRT_2).powi(2);
        for (far, near) in [(1e300, 1e6), (-1e300, -1e6)] {
            assert!(sq(near).is_finite(), "near {near:e}: f {f}, sd {sd}");
            assert_eq!(sq(far), f64::INFINITY, "far {far:e}: f {f}, sd {sd}");
        }
    }
    let setup = |limit: f64, flag: i8| {
        let mut model = make_model();
        model.bloq_method = BloqMethod::M3;
        let mut subject = make_simple_subject();
        subject.observations[5] = limit;
        subject.cens[5] = flag;
        (model, subject)
    };
    let conditional = |limit: f64, flag: i8| {
        let (model, subject) = setup(limit, flag);
        individual_nll(&model, &subject, &theta, &eta, &omega, &sigma)
    };
    let marginal = |limit: f64, flag: i8, interaction: bool| {
        let (model, subject) = setup(limit, flag);
        let eta_hat = DVector::from_vec(eta.to_vec());
        let h = DMatrix::from_element(subject.obs_times.len(), 1, 4.0);
        foce_subject_nll(&model, &subject, &theta, &eta_hat, &h, &omega, &sigma, &[], interaction)
    };
    let objectives: [(&str, &dyn Fn(f64, i8) -> f64); 3] = [
        ("individual_nll", &conditional),
        ("FOCE", &|l: f64, c: i8| marginal(l, c, false)),
        ("FOCEI", &|l: f64, c: i8| marginal(l, c, true)),
    ];
    for (name, nll) in objectives {
        for (far, near, flag) in [(1e300, 1e6, 1), (-1e300, -1e6, -1)] {
            let (got, want) = (nll(far, flag), nll(near, flag));
            assert!(want.is_finite(), "{name}: control at {near:e}, CENS={flag}");
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name}: limit {far:e} (CENS={flag}) scored {got}, the certain row {want}"
            );
        }
    }
}

/// #1496: `W_CENS_UNEXPECTED`'s two claims checked against the objective: under `m3` an
/// out-of-domain flag scores as its sign's code, under `drop` as a quantified row —
/// through `individual_nll` and the FOCE/FOCEI marginals; `1`, `-1`, `0` all differ.
#[test]
fn an_out_of_domain_cens_flag_scores_by_sign_under_m3_and_as_quantified_under_drop() {
    let theta = [5.0, 50.0];
    let eta = [0.1];
    let omega = make_omega(0.09);
    let sigma = [0.2];
    let setup = |bloq: BloqMethod, flag: i8| {
        let mut model = make_model();
        model.bloq_method = bloq;
        let mut subject = make_simple_subject();
        // The last row, DV 25 against a prediction near 30: both tails are
        // live and differ from the quantified term.
        subject.cens[5] = flag;
        (model, subject)
    };
    let conditional = |bloq: BloqMethod, flag: i8| {
        let (model, subject) = setup(bloq, flag);
        individual_nll(&model, &subject, &theta, &eta, &omega, &sigma)
    };
    let marginal = |bloq: BloqMethod, flag: i8, interaction: bool| {
        let (model, subject) = setup(bloq, flag);
        let eta_hat = DVector::from_vec(eta.to_vec());
        let h = DMatrix::from_element(subject.obs_times.len(), 1, 4.0);
        foce_subject_nll(&model, &subject, &theta, &eta_hat, &h, &omega, &sigma, &[], interaction)
    };
    let objectives: [(&str, &dyn Fn(BloqMethod, i8) -> f64); 3] = [
        ("individual_nll", &conditional),
        ("FOCE", &|b: BloqMethod, c: i8| marginal(b, c, false)),
        ("FOCEI", &|b: BloqMethod, c: i8| marginal(b, c, true)),
    ];
    for (name, nll) in objectives {
        let m3 = |flag: i8| nll(BloqMethod::M3, flag);
        let drop = |flag: i8| nll(BloqMethod::Drop, flag);
        for flag in [0, 1, -1] {
            assert!(m3(flag).is_finite(), "{name}: m3, CENS={flag}");
            assert!(drop(flag).is_finite(), "{name}: drop, CENS={flag}");
        }
        assert_ne!(m3(1).to_bits(), m3(-1).to_bits(), "{name}: the tails differ");
        assert_ne!(m3(1).to_bits(), m3(0).to_bits(), "{name}: the lower tail is live");
        assert_ne!(m3(-1).to_bits(), m3(0).to_bits(), "{name}: the upper tail is live");
        // "under bloq_method = m3 a positive flag is scored on the lower tail, like
        // CENS=1" — and the negative one on the upper tail, like CENS=-1.
        assert_eq!(m3(7).to_bits(), m3(1).to_bits(), "{name}: m3, 7 as 1");
        assert_eq!(m3(-2).to_bits(), m3(-1).to_bits(), "{name}: m3, -2 as -1");
        // "under bloq_method = drop the row is scored as an ordinary observation".
        for flag in [7, -2, 1, -1] {
            assert_eq!(
                drop(flag).to_bits(),
                drop(0).to_bits(),
                "{name}: drop, CENS={flag} scores as the quantified row"
            );
        }
    }
}

#[test]
fn prepared_inner_nll_reuse_is_bit_identical() {
    let model = make_model();
    let subject = make_simple_subject();
    let theta = [5.0, 50.0];
    let eta = [0.17];
    let omega = make_omega(0.09);
    let sigma = [0.05];
    let expected = individual_nll_into_with_schedule(
        &model,
        &subject,
        &theta,
        &eta,
        &omega,
        &sigma,
        &[],
        &mut pk::EventPkParams::default(),
        None,
    );

    let err_keys = model.error_spec.obs_keys(&subject);
    let mult = model.ruv_obs_mult(&subject, &theta);
    let mut event_scratch = pk::EventPkParams::default();
    let mut predictions = vec![-999.0; subject.obs_times.len()];
    let mut eta_work = DVector::from_element(model.n_eta, -999.0);
    let mut prior_work = DVector::from_element(model.n_eta, -999.0);
    let reused = individual_nll_into_prepared_with_schedule(
        &model,
        &subject,
        &theta,
        &eta,
        &omega,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &mut event_scratch,
        None,
        err_keys.as_ref(),
        mult.as_deref(),
        &mut predictions,
        &mut eta_work,
        &mut prior_work,
    );

    assert_eq!(expected.to_bits(), reused.to_bits());
    assert_eq!(predictions.len(), subject.obs_times.len());
    assert!(predictions.iter().all(|v| v.is_finite()));
}

#[test]
fn test_ltbs_individual_nll_matches_additive_on_log_scale() {
    // Under LTBS the inner-loop NLL must score the (already-log-scale)
    // observations against log(prediction) with additive variance σ². This
    // checks the prediction sink's log-wrap flows through `individual_nll`.
    let mut model = make_model();
    model.error_model = ErrorModel::Additive;
    model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
    model.log_transform = true;

    let theta = vec![5.0, 50.0];
    let eta = vec![0.0]; // eta_prior = 0
    let omega = make_omega(0.09);
    let sigma = vec![0.3]; // additive SD on the log scale

    // Observations on the log scale (what `fit()` produces for case 2).
    let mut subj = make_simple_subject();
    for v in &mut subj.observations {
        *v = v.ln();
    }

    // Manual reference: log(natural prediction), additive variance σ².
    let mut natural_model = make_model();
    natural_model.log_transform = false;
    let natural = pk::compute_predictions_with_tv(&natural_model, &subj, &theta, &eta);
    let var = sigma[0] * sigma[0];
    let mut data_ll = 0.0;
    for (j, &f_nat) in natural.iter().enumerate() {
        let log_f = f_nat.max(pk::LTBS_FLOOR).ln();
        let resid = subj.observations[j] - log_f;
        data_ll += resid * resid / var + var.ln();
    }
    let expected = 0.5 * (omega.log_det + data_ll);

    let got = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    approx::assert_relative_eq!(got, expected, epsilon = 1e-9);
}

/// #1029 degenerate oracle: `weight = W` reproduces the hand-built study-as-subject
/// construction (`DV/W`, `PRED/W`, `sigma FIX 1`) up to the data-only Jacobian
/// `Σ_j ln W_j` of `y ↦ y/W`, which cancels from every ΔOFV.
#[test]
fn test_weighted_error_model_matches_the_hand_built_construction() {
    let src = |err: &str| {
        format!(
            "[parameters]\n  theta TVCL(0.2)\n  theta TVV(10.0)\n  omega ETA_CL ~ 0.09\n  \
             sigma ADD_ERR ~ 1.0 (variance) FIX\n[individual_parameters]\n  CL = TVCL * \
             exp(ETA_CL)\n  V  = TVV\n[structural_model]\n  pk one_cpt_iv(cl=CL, \
             v=V)\n[error_model]\n{err}\n[covariates]\n  WPSE continuous\n"
        )
    };
    let weighted = crate::parser::model_parser::parse_model_string(&src(
        "  DV ~ additive(ADD_ERR) weight = WPSE",
    ))
    .unwrap();
    let plain =
        crate::parser::model_parser::parse_model_string(&src("  DV ~ additive(ADD_ERR)")).unwrap();

    // One row per trial arm, each carrying its own reported standard error.
    let w = [0.5_f64, 2.0, 1.25];
    let y = [12.0_f64, 9.0, 15.0];
    let f = [11.0_f64, 10.0, 14.0];

    let mut natural = make_simple_subject();
    natural.obs_times = vec![1.0, 2.0, 3.0];
    natural.observations = y.to_vec();
    natural.obs_cmts = vec![1; 3];
    natural.cens = vec![0; 3];
    natural.occasions = vec![1; 3];
    natural.obs_covariates =
        w.iter().map(|&wi| [("WPSE".to_string(), wi)].into_iter().collect()).collect();

    // The hand-built twin: DV and the prediction both pre-divided by the
    // weight, scored by an unweighted additive model on the same sigma.
    let mut prescaled = natural.clone();
    prescaled.observations = y.iter().zip(w).map(|(&yi, wi)| yi / wi).collect();
    prescaled.obs_covariates = Vec::new();
    let scaled_preds: Vec<f64> = f.iter().zip(w).map(|(&fi, wi)| fi / wi).collect();

    let theta = [0.2, 10.0];
    let sigma = [0.8];
    let eta = [0.0];

    let nll_weighted = obs_nll_subject_from_preds(
        &weighted,
        &natural,
        &f,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &eta,
    );
    let nll_hand = obs_nll_subject_from_preds(
        &plain,
        &prescaled,
        &scaled_preds,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &eta,
    );
    let jacobian: f64 = w.iter().map(|wi| wi.ln()).sum();

    approx::assert_relative_eq!(nll_weighted, nll_hand + jacobian, epsilon = 1e-12);
}

#[test]
fn test_split_obs_by_occasion_two_occ() {
    let subj = make_simple_subject();
    let groups = split_obs_by_occasion(&subj);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].0, 1);
    assert_eq!(groups[0].1, vec![0, 1, 2]);
    assert_eq!(groups[1].0, 2);
    assert_eq!(groups[1].1, vec![3, 4, 5]);
}

#[test]
fn test_split_obs_by_occasion_empty() {
    let mut subj = make_simple_subject();
    subj.occasions = Vec::new();
    subj.obs_times = Vec::new();
    subj.observations = Vec::new();
    subj.obs_cmts = Vec::new();
    subj.cens = Vec::new();
    let groups = split_obs_by_occasion(&subj);
    assert!(groups.is_empty());
}

#[test]
fn test_iov_occasion_groups_appends_dose_only_occ() {
    let mut subj = make_simple_subject();
    subj.dose_occasions = vec![1, 3, 2, 4, 3];
    let groups = iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 4);
    assert_eq!(groups[0], (1, vec![0, 1, 2]));
    assert_eq!(groups[1], (2, vec![3, 4, 5]));
    assert_eq!(groups[2], (3, Vec::new()));
    assert_eq!(groups[3], (4, Vec::new()));
}

/// An occasion only pk-only rows carry gets a group too, after the dose-only ones and
/// in file order; one an observation or a dose already carries does not get a second.
#[test]
fn test_iov_occasion_groups_appends_pk_only_occ() {
    let mut subj = make_simple_subject();
    subj.dose_occasions = vec![1, 3];
    subj.pk_only_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    subj.pk_only_occasions = vec![6, 2, 5, 3, 6, 1];
    let groups = iov_occasion_groups(&subj);
    let ids: Vec<u32> = groups.iter().map(|(occ, _)| *occ).collect();
    assert_eq!(ids, vec![1, 2, 3, 6, 5]);
    assert_eq!(groups[3], (6, Vec::new()));
    assert_eq!(groups[4], (5, Vec::new()));
}

/// #1153: an occasion only reset rows carry gets a group too, after the pk-only ones
/// and in file order; one any other record already carries does not get a second.
#[test]
fn test_iov_occasion_groups_appends_reset_only_occ() {
    let mut subj = make_simple_subject();
    subj.dose_occasions = vec![1, 3];
    subj.pk_only_times = vec![1.0];
    subj.pk_only_occasions = vec![6];
    subj.reset_times = vec![1.5, 2.5, 3.5, 4.5];
    subj.reset_occasions = vec![9, 6, 8, 9];
    let groups = iov_occasion_groups(&subj);
    let ids: Vec<u32> = groups.iter().map(|(occ, _)| *occ).collect();
    assert_eq!(ids, vec![1, 2, 3, 6, 9, 8]);
    assert_eq!(groups[4], (9, Vec::new()));
    assert_eq!(groups[5], (8, Vec::new()));
}

/// #1153: a reset-only group is informed by the same first-record rule — the reset at
/// t = 4 has observations after it (t = 4..6), the one at t = 20 has none — and never
/// under `carries_state = false`. A reset at t = 0.5 labelled 2 changes nothing: OCC 2
/// owns observations.
#[test]
fn test_iov_group_informed_reads_reset_rows() {
    let mut subj = make_simple_subject();
    subj.reset_times = vec![0.5, 4.0, 20.0];
    subj.reset_occasions = vec![2, 8, 9];
    let ids: Vec<u32> = iov_occasion_groups(&subj).iter().map(|g| g.0).collect();
    assert_eq!(ids, vec![1, 2, 8, 9]);
    assert_eq!(iov_group_informed(&subj, true, true), vec![true, true, true, false]);
    assert_eq!(iov_group_informed(&subj, false, true), vec![true, true, false, false]);
    // A reset that re-seeds nothing informs nothing, at any time (review r1 #1).
    assert_eq!(iov_group_informed(&subj, true, false), vec![true, true, false, false]);
}

/// #1153 review r1 #1: `iov_group_informed_for` lets a reset inform its group only on an
/// ODE model with an `init(...)`. A closed-form model's reset reads no parameters and an
/// init-free ODE model's reset zeros the state, so κ of a reset-only occasion moves no
/// prediction there and must stay out of the κ shrinkage.
#[test]
fn test_iov_group_informed_for_reads_the_reset_reseed() {
    let ode = |init: &str| {
        format!(
            "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
             omega ETA_CL ~ 0.09\n  kappa KAPPA ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n\
             [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA)\n  V = TVV\n\
             [structural_model]\n  ode(states=[central])\n\
             [odes]\n{init}  d/dt(central) = -(CL/V) * central\n\
             [scaling]\n  y = central / V\n\
             [error_model]\n  DV ~ proportional(PROP_ERR)\n"
        )
    };
    let closed = "[parameters]\n  theta TVCL(4.0, 0.1, 100.0)\n  theta TVV(30.0, 1.0, 500.0)\n  \
         omega ETA_CL ~ 0.09\n  kappa KAPPA ~ 0.04\n  sigma PROP_ERR ~ 0.1 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL + KAPPA)\n  V = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
         [error_model]\n  DV ~ proportional(PROP_ERR)\n";
    let mut subj = make_simple_subject();
    subj.reset_times = vec![4.0];
    subj.reset_occasions = vec![8];
    for (src, want) in
        [(closed.to_string(), false), (ode(""), false), (ode("  init(central) = CL\n"), true)]
    {
        let m = crate::parser::model_parser::parse_model_string(&src).unwrap();
        assert_eq!(
            iov_group_informed_for(&m, &subj),
            vec![true, true, want],
            "ode {} init {}",
            m.ode_spec.is_some(),
            m.ode_spec.as_ref().is_some_and(|o| o.init_fn.is_some())
        );
    }
}

/// #1911: a group without observations is informed iff an observation sits at or after
/// its first record (obs t = 1..6): dose-only at 2 yes, at 10 no; pk-only at 6 (tie)
/// yes, at 8/12 no; dose 9 + pk-only 1 yes (earliest decides); an own-obs group is
/// always informed. Under `carries_state = false` only observation groups remain.
#[test]
fn test_iov_group_informed_by_first_record_time() {
    let mut subj = make_simple_subject();
    subj.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(9.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(50.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subj.dose_occasions = vec![1, 3, 4, 7, 2];
    subj.pk_only_times = vec![6.0, 12.0, 8.0, 1.0];
    subj.pk_only_occasions = vec![5, 6, 6, 7];
    let ids: Vec<u32> = iov_occasion_groups(&subj).iter().map(|g| g.0).collect();
    assert_eq!(ids, vec![1, 2, 3, 4, 7, 5, 6]);
    assert_eq!(
        iov_group_informed(&subj, true, true),
        vec![true, true, true, false, true, true, false]
    );
    assert_eq!(
        iov_group_informed(&subj, false, true),
        vec![true, true, false, false, false, false, false]
    );
}

/// Without dose-only or pk-only groups every group owns an observation, so the mask
/// is all-true and the shrinkage is the historical one (the #1911 control).
#[test]
fn test_iov_group_informed_all_true_for_observation_groups() {
    let subj = make_simple_subject();
    for carries_state in [true, false] {
        assert_eq!(iov_group_informed(&subj, carries_state, true), vec![true, true]);
    }
}

#[test]
fn test_individual_nll_iov_no_kappa_same_as_base() {
    let model = make_model();
    let subj = make_simple_subject();
    let theta = vec![5.0, 50.0];
    let eta = vec![0.0];
    let omega = make_omega(0.09);
    let sigma = vec![0.05];

    let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    let iov = individual_nll_iov(&model, &subj, &theta, &eta, &[], &omega, None, &sigma);
    approx::assert_relative_eq!(base, iov, epsilon = 1e-10);
}

/// The IOV individual NLL (SAEM E-step, Bayes MH, IOV IS weight) applies the
/// per-observation residual magnitude (#484/#1029) as the non-IOV path does; a weight
/// varying within the subject makes it bite.
#[test]
fn test_individual_nll_iov_applies_the_residual_magnitude() {
    let model = crate::parser::model_parser::parse_model_string(
        "[parameters]\n  theta TVCL(5.0)\n  theta TVV(50.0)\n  omega ETA_CL ~ 0.09\n  \
         sigma PROP_ERR ~ 0.10 (sd)\n  sigma ADD_ERR ~ 0.50 (sd)\n\
         [individual_parameters]\n  CL = TVCL * exp(ETA_CL)\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  \
         DV ~ combined(PROP_ERR * (1.0 + 0.5 * WPSE), ADD_ERR) weight = WPSE\n\
         [covariates]\n  WPSE continuous\n",
    )
    .expect("weighted model parses");
    assert!(model.has_custom_ruv_magnitude());

    let mut subj = make_simple_subject();
    let snap = |w: f64| -> HashMap<String, f64> { [("WPSE".to_string(), w)].into_iter().collect() };
    subj.covariates = snap(0.5);
    subj.obs_covariates =
        (0..subj.observations.len()).map(|j| snap(0.5 + 0.3 * j as f64)).collect();

    let theta = vec![5.0, 50.0];
    let eta = vec![0.0];
    let omega = make_omega(0.09);
    let sigma = vec![0.10, 0.50];

    let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    let iov = individual_nll_iov(&model, &subj, &theta, &eta, &[], &omega, None, &sigma);
    approx::assert_relative_eq!(base, iov, epsilon = 1e-10);

    // Guard against a vacuous pass: the magnitude must actually move the NLL.
    let mut flat = subj.clone();
    flat.obs_covariates = vec![snap(0.5); subj.observations.len()];
    let flat_nll = individual_nll(&model, &flat, &theta, &eta, &omega, &sigma);
    assert!(
        (flat_nll - base).abs() > 1e-6,
        "the per-observation weight must change the NLL, else this test proves nothing"
    );
}

#[test]
fn test_individual_nll_finite_sentinel_on_nonfinite_eta() {
    // #97: a non-finite eta (standing in for an ODE blow-up) must give the finite
    // sentinel. The analytical path scrubs NaN via `.max()`, so it enters through η'Ω⁻¹η.
    let model = make_model();
    let subj = make_simple_subject();
    let omega = make_omega(0.09);
    let nll = individual_nll(&model, &subj, &[5.0, 50.0], &[f64::INFINITY], &omega, &[0.05]);
    assert!(nll.is_finite(), "NLL must stay finite, got {nll}");
    assert_eq!(nll, SUBJECT_SENTINEL_NLL, "a non-finite NLL should map to the 1e20 sentinel");
}

/// #2269: each guard returns exactly `SUBJECT_SENTINEL_NLL`, and its readers
/// (Gauss–Newton / AGQ usability, `W_NONFINITE_OBJECTIVE` "divergence sentinel" label)
/// recognise it.
#[test]
fn subject_sentinel_guards_and_readers_agree() {
    let model = make_model();
    let subj = make_simple_subject();
    let omega = make_omega(0.09);
    let theta = [5.0, 50.0];
    let sigma = [0.05];

    // Guard 1: `individual_nll`'s non-finite-total fold.
    let ind = individual_nll(&model, &subj, &theta, &[f64::INFINITY], &omega, &sigma);
    // Guards 2 and 3: FOCE's and FOCEI's linearized `R̃` that is not positive
    // definite. A NaN `H` makes `R̃` NaN, which the Cholesky refuses.
    let n_obs = subj.obs_times.len();
    let h_nan = DMatrix::from_element(n_obs, 1, f64::NAN);
    let eta0 = DVector::zeros(1);
    let foce = |interaction| {
        foce_subject_nll(
            &model,
            &subj,
            &theta,
            &eta0,
            &h_nan,
            &omega,
            &sigma,
            &model.residual_correlations,
            interaction,
        )
    };
    // Guard 4: an Ω whose log-determinant is not finite (a singular Ω).
    let mut singular = omega.clone();
    singular.log_det = f64::NAN;
    let singular_omega = individual_nll(&model, &subj, &theta, &[0.0], &singular, &sigma);
    // Guard 5: `chol_log_det` on a factor with a non-positive diagonal.
    let chol = chol_log_det(&DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.5, 0.0]));
    for (guard, got) in [
        ("individual_nll", ind),
        ("FOCE R̃", foce(false)),
        ("FOCEI R̃", foce(true)),
        ("individual_nll singular Ω", singular_omega),
        ("chol_log_det", chol),
    ] {
        assert_eq!(got, SUBJECT_SENTINEL_NLL, "{guard} must return the shared sentinel");
    }

    assert!(
        !crate::stats::likelihood::is_usable_subject_nll(SUBJECT_SENTINEL_NLL),
        "the usability test must reject a subject on the rail"
    );
    assert!(
        crate::stats::likelihood::is_usable_subject_nll(SUBJECT_SENTINEL_NLL / 2.0),
        "and accept one below it, else the first assertion proves nothing"
    );
    assert_eq!(
        crate::estimation::outer_optimizer::nonfinite_objective_reason(2.0 * SUBJECT_SENTINEL_NLL),
        "at the divergence sentinel"
    );
    assert_eq!(
        crate::estimation::outer_optimizer::nonfinite_objective_reason(SUBJECT_SENTINEL_NLL / 2.0),
        "too large to be a fit"
    );
}

/// #2269: no per-subject guard spells the sentinel as a bare literal. Scans production
/// code under `src/` (not comments, strings, `*_tests.rs` or `mod tests`).
/// `POPULATION_OBJECTIVE_FILES` are exceptions whose `1e20`s clamp a *population*
/// objective (or, in `io/fitrx.rs`, a formatting value). `saem.rs` uses the constant.
#[test]
fn no_bare_sentinel_literal_in_guard_files() {
    const POPULATION_OBJECTIVE_FILES: [&str; 7] = [
        "estimation/outer_optimizer.rs",
        "estimation/profile.rs",
        "estimation/trust_region.rs",
        "estimation/covariate_mu_ref.rs",
        "estimation/saem_mixture.rs",
        "estimation/impmap.rs",
        "io/fitrx.rs",
    ];
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read src/") {
            let path = e.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }
    // `1e20` as a token: not `1e200`, not `x1e20`.
    fn has_literal(code: &str) -> bool {
        code.match_indices("1e20").any(|(i, _)| {
            let before = code[..i].chars().next_back();
            let after = code[i + 4..].chars().next();
            !before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
                && !after.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
        })
    }
    const DEFINITION: &str = "pub const SUBJECT_SENTINEL_NLL: f64 = 1e20;";
    // The 1-based lines of `src` whose production code holds the literal, and
    // whether `src` holds the definition.
    fn bare_lines(src: &str) -> (Vec<usize>, bool) {
        let (mut lines, mut saw_definition, mut in_test_mod) = (Vec::new(), false, false);
        for (i, line) in src.lines().enumerate() {
            // rustfmt puts a top-level item's opening and closing at column 0.
            if line.starts_with("mod tests") || line.starts_with("pub mod tests") {
                in_test_mod = true;
            }
            if in_test_mod {
                // `\u{7d}` is the closing brace, escaped so this line's braces balance
                // for the brace-counting source scans in `tests/` that blank this module.
                in_test_mod = line != "\u{7d}";
                continue;
            }
            if line.trim() == DEFINITION {
                saw_definition = true;
                continue;
            }
            let code = line.split("//").next().unwrap_or("");
            // Drop string literals, so a message may still say "1e20".
            let code: String = code.split('"').step_by(2).collect();
            if has_literal(&code) {
                lines.push(i + 1);
            }
        }
        (lines, saw_definition)
    }
    // The matcher itself, so a scan that finds nothing is not a matcher that never
    // matches: the token rule, and which parts of a file it reads.
    assert!(has_literal("return 1e20;") && has_literal("x < 1e20"));
    assert!(!has_literal("1e200") && !has_literal("x1e20") && !has_literal("1e20_f"));
    let fixture = [
        "fn f() -> f64 \u{7b}",
        "    1e20 // a guard",
        "\u{7d}",
        "// 1e20 in a comment",
        "fn g() -> &'static str { \"1e20 in a string\" }",
        "mod tests \u{7b}",
        "    fn t() -> f64 { 1e20 }",
        "\u{7d}",
        "fn h() -> f64 { 2.0 * 1e20 }",
    ]
    .join("\n");
    assert_eq!(bare_lines(&fixture), (vec![2, 9], false));

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();
    let mut hits = Vec::new();
    let mut saw_definition = false;
    for path in &files {
        let rel = path.strip_prefix(&root).unwrap().to_string_lossy().replace('\\', "/");
        let is_test_file = rel.ends_with("_tests.rs") || rel.contains("/tests/");
        if is_test_file || POPULATION_OBJECTIVE_FILES.contains(&rel.as_str()) {
            continue;
        }
        let src = std::fs::read_to_string(path).expect("read source");
        let (lines, defines) = bare_lines(&src);
        saw_definition |= defines;
        hits.extend(lines.into_iter().map(|n| format!("src/{rel}:{n}")));
    }
    assert!(
        hits.is_empty(),
        "write SUBJECT_SENTINEL_NLL, not a bare 1e20 (or, for a population objective, \
         add the file to POPULATION_OBJECTIVE_FILES with a reason):\n{}",
        hits.join("\n")
    );
    // The scan must see the definition itself, else a moved file or renamed
    // constant would leave it scanning nothing; and the allowlist must name files
    // that exist, so a rename cannot silently widen it.
    assert!(saw_definition, "scan did not reach the definition");
    for f in POPULATION_OBJECTIVE_FILES {
        assert!(root.join(f).is_file(), "allowlisted {f} does not exist");
    }
}

#[test]
fn test_individual_nll_iov_with_kappa_adds_prior() {
    let model = make_model();
    let subj = make_simple_subject();
    let theta = vec![5.0, 50.0];
    let eta = vec![0.0];
    let omega = make_omega(0.09);
    let omega_iov = make_omega(0.01);
    let sigma = vec![0.05];

    let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    // Non-zero kappas add a kappa prior ≥ 0, so IOV NLL ≥ base NLL.
    let kappas = vec![vec![0.1], vec![-0.1]];
    let iov =
        individual_nll_iov(&model, &subj, &theta, &eta, &kappas, &omega, Some(&omega_iov), &sigma);
    let borrowed: Vec<&[f64]> = kappas.iter().map(Vec::as_slice).collect();
    let reused = individual_nll_iov_with_scratch(
        &model,
        &subj,
        &theta,
        &eta,
        &borrowed,
        &omega,
        Some(&omega_iov),
        &sigma,
        &mut pk::EventPkParams::default(),
    );
    assert_eq!(iov.to_bits(), reused.to_bits());
    // Kappa prior is positive → IOV NLL should differ from base
    assert!((iov - base).abs() > 1e-6, "IOV NLL={}, base NLL={}", iov, base);
}

/// A model whose CL depends on both the BSV eta and the per-occasion
/// kappa (`combined[1]`), so the kappa block genuinely enters the
/// augmented R̃. The kappa read is defensive so the BSV-only
/// `foce_subject_nll` path (which passes a length-1 eta) doesn't panic.
fn make_iov_kappa_model() -> CompiledModel {
    let mut model = make_model();
    model.pk_param_fn =
        Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>, _t: f64| {
            let mut p = PkParams::default();
            let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
            p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL
            p.values[1] = theta[1]; // V
            p
        });
    model
}

/// Issue #101: `foce_subject_nll_iov` must be a proper augmented marginal,
/// not a BSV FOCE term plus an additive kappa MAP penalty.
#[test]
fn test_foce_subject_nll_iov_is_proper_marginal() {
    let model = make_iov_kappa_model();
    let subj = make_simple_subject(); // occasions [1,1,1,2,2,2]
    let theta = vec![5.0, 50.0];
    let eta_hat = DVector::from_vec(vec![0.1]);
    let omega_bsv = make_omega(0.09);
    let sigma = vec![0.05];

    // BSV-only H via central FD of predictions w.r.t. eta[0] at kappa = 0.
    let n_obs = subj.observations.len();
    let mut h_bsv = DMatrix::zeros(n_obs, 1);
    let eps = 1e-6;
    let pp = model_predictions(&model, &subj, &theta, &[0.1 + eps]);
    let pm = model_predictions(&model, &subj, &theta, &[0.1 - eps]);
    for j in 0..n_obs {
        h_bsv[(j, 0)] = (pp[j] - pm[j]) / (2.0 * eps);
    }

    // (1) Reduction: zero kappas + Ω_iov → 0 collapses to the BSV-only
    //     marginal. The OLD code added 0.5·K·log|Ω_iov| = log(1e-12) ≈ -27.6,
    //     so this assertion fails without the proper-marginal fix.
    let base = foce_subject_nll(
        &model,
        &subj,
        &theta,
        &eta_hat,
        &h_bsv,
        &omega_bsv,
        &sigma,
        &model.residual_correlations,
        false,
    );
    let zero_kappas = vec![DVector::zeros(1), DVector::zeros(1)];
    let reduced = foce_subject_nll_iov(
        &model,
        &subj,
        &theta,
        &eta_hat,
        &h_bsv,
        &omega_bsv,
        &sigma,
        false,
        &zero_kappas,
        &make_omega(1e-12),
    );
    // max_relative (not epsilon): these are O(1e5), and the residual κ-block
    // contribution at Ω_iov = 1e-12 is ~1e-11 relative. The old additive
    // penalty would shift by ~27.6 absolute (≈1.5e-4 relative) and fail.
    approx::assert_relative_eq!(reduced, base, max_relative = 1e-9);

    // (2) The marginal responds to Ω_iov through R̃ (the determinant term
    //     the old penalty was missing): with non-zero kappas, two different
    //     Ω_iov give materially different, finite OFVs.
    let kappas = vec![DVector::from_vec(vec![0.08]), DVector::from_vec(vec![-0.05])];
    let nll = |iov_var: f64| {
        foce_subject_nll_iov(
            &model,
            &subj,
            &theta,
            &eta_hat,
            &h_bsv,
            &omega_bsv,
            &sigma,
            false,
            &kappas,
            &make_omega(iov_var),
        )
    };
    let small = nll(0.005);
    let large = nll(0.5);
    assert!(small.is_finite() && large.is_finite());
    assert!(
        (small - large).abs() > 1e-6,
        "Ω_iov must change the marginal OFV (small={small}, large={large})"
    );
}

/// The dense-`R` FOCEI interaction marginal must reduce **exactly** to the
/// diagonal accumulator [`foce_subject_nll_interaction`] when there is no
/// `block_sigma` correlation (ρ = 0), across a multi-η proportional model.
/// This anchors the dense generalisation against the bit-for-bit reference.
#[test]
fn dense_interaction_reduces_to_diagonal_when_uncorrelated() {
    let subject = make_simple_subject(); // 6 obs, single proportional endpoint
    let sigma = vec![0.2];
    let error_spec = ErrorSpec::Single(ErrorModel::Proportional);
    let ipreds = vec![50.0, 40.0, 30.0, 45.0, 35.0, 25.0];
    let eta_hat = DVector::from_vec(vec![0.15, -0.1]);
    let omega = OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["ETA_CL".into(), "ETA_V".into()]);
    let h_matrix = DMatrix::from_row_slice(
        6,
        2,
        &[
            5.0, 1.0, //
            4.0, 1.5, //
            3.0, 2.0, //
            4.5, 1.2, //
            3.5, 1.8, //
            2.5, 2.2,
        ],
    );

    let diag = foce_subject_nll_interaction(
        &subject,
        &ipreds,
        &eta_hat,
        &h_matrix,
        &omega,
        &sigma,
        &error_spec,
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );
    let dense = foce_subject_nll_interaction_dense(
        &subject,
        &ipreds,
        &eta_hat,
        &h_matrix,
        &omega,
        &sigma,
        &error_spec,
        &[], // ρ = 0 → diagonal R, must match the reference exactly
        &[],
        None,
    );
    approx::assert_relative_eq!(dense, diag, max_relative = 1e-10);

    // The #484 magnitude path (Some(mult)) with an all-ones multiplier must
    // reproduce the bare path (modulo the documented ~1 ULP reassociation).
    let ones_mult: Vec<Vec<f64>> = vec![vec![1.0]; 6];
    let dense_mult = foce_subject_nll_interaction_dense(
        &subject,
        &ipreds,
        &eta_hat,
        &h_matrix,
        &omega,
        &sigma,
        &error_spec,
        &[],
        &[],
        Some(&ones_mult),
    );
    approx::assert_relative_eq!(dense_mult, dense, max_relative = 1e-9);
}

/// FOCE+proportional: R is evaluated at a supplied `f(η=0)`, not the SB-linearized
/// `f0`, whose zero crossing collapses `R(f0)` to the floor.
#[test]
fn foce_standard_variance_uses_override_not_zero_crossing_f0() {
    let subject = make_simple_subject(); // 6 obs
    let omega = make_omega(0.09);
    let sigma = vec![0.2]; // proportional SD
    let error_spec = ErrorSpec::Single(ErrorModel::Proportional);

    // ipreds all positive; H·η̂ drives the first f0 component to exactly 0.
    let ipreds = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let eta_hat = DVector::from_vec(vec![1.0]);
    let h_matrix = DMatrix::from_column_slice(6, 1, &[10.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    // f0 = ipred − H·η̂ = [0, 15, 25, 35, 45, 55] → first R(f0) hits the floor.

    let nll_f0 = foce_subject_nll_standard(
        &subject,
        &ipreds,
        &eta_hat,
        &h_matrix,
        &omega,
        &sigma,
        &error_spec,
        &[],
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );
    let nll_override = foce_subject_nll_standard(
        &subject,
        &ipreds,
        &eta_hat,
        &h_matrix,
        &omega,
        &sigma,
        &error_spec,
        &[],
        BloqMethod::Drop,
        &[],
        None,
        Some(&ipreds),
        None,
    );

    assert!(nll_override.is_finite() && nll_f0.is_finite());
    // The f0 path's floored variance inflates the marginal by a clear margin (~56 here);
    // HΩH' cushions it, but it is far above FP noise.
    assert!(
        nll_f0 - nll_override > 20.0,
        "override must change the SB marginal (R evaluated at f(η=0), not f0): \
         nll_f0={nll_f0}, nll_override={nll_override}"
    );
}

/// #646: FOCE M3 censored rows use `R̃ⱼⱼ = Hⱼ Ω Hⱼᵀ + R⁰`, not `R⁰`. With quantified
/// rows' Jacobian zeroed, only the censored row can make the NLL depend on Ω.
#[test]
fn foce_m3_censored_uses_marginal_variance() {
    let mut subject = make_simple_subject();
    subject.cens = vec![0, 0, 0, 0, 0, 1]; // last row below-LLOQ
    subject.observations[5] = 5.0; // LLOQ limit
    let sigma = vec![3.0]; // additive SD → R⁰ = 9, Ω-independent
    let error_spec = ErrorSpec::Single(ErrorModel::Additive);
    let ipreds = vec![50.0, 40.0, 30.0, 45.0, 35.0, 8.0];
    let eta_hat = DVector::from_vec(vec![0.5]);
    // Quant rows H=0 (their R̃ = R is Ω-independent); censored row H = 4 ≠ 0.
    let h_matrix = DMatrix::from_column_slice(6, 1, &[0.0, 0.0, 0.0, 0.0, 0.0, 4.0]);
    let call = |omega: &OmegaMatrix| {
        foce_subject_nll_standard(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            omega,
            &sigma,
            &error_spec,
            &[],
            BloqMethod::M3,
            &[],
            None,
            None,
            None,
        )
    };
    let nll_small = call(&make_omega(0.01));
    let nll_large = call(&make_omega(1.0));
    assert!(nll_small.is_finite() && nll_large.is_finite());
    // Marginal variance grows with Ω (H²·Ω = 16·Ω), widening the censored tail,
    // so the two NLLs must differ well above FP noise. (Old conditional → equal.)
    assert!(
        (nll_small - nll_large).abs() > 1e-3,
        "censored term must depend on Ω via the marginal variance (#646): \
         nll(Ω=0.01)={nll_small}, nll(Ω=1.0)={nll_large}"
    );
}

/// FREM r_diag merge collision: `frem_r_override` must reach the diagonal feeding R̃
/// even though `r_pred_override` re-evaluates R afterward.
#[test]
fn foce_standard_applies_frem_r_override() {
    let subject = make_simple_subject(); // 6 obs
    let omega = make_omega(0.09);
    let sigma = vec![0.2]; // proportional SD
    let error_spec = ErrorSpec::Single(ErrorModel::Proportional);
    let ipreds = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let eta_hat = DVector::from_vec(vec![1.0]);
    let h_matrix = DMatrix::from_column_slice(6, 1, &[10.0, 5.0, 5.0, 5.0, 5.0, 5.0]);

    let call = |frem: Option<&[Option<f64>]>| {
        foce_subject_nll_standard(
            &subject,
            &ipreds,
            &eta_hat,
            &h_matrix,
            &omega,
            &sigma,
            &error_spec,
            &[],
            BloqMethod::Drop,
            &[],
            frem,
            None,
            None,
        )
    };

    // Override the first row's residual variance with a value far from the
    // PK error model's R(f0)[0] = (10·0.2)² = 4.
    let overrides = [Some(250.0), None, None, None, None, None];
    let nll_plain = call(None);
    let nll_frem = call(Some(&overrides));

    assert!(nll_plain.is_finite() && nll_frem.is_finite());
    assert!(
        (nll_plain - nll_frem).abs() > 1e-6,
        "frem_r_override must change the marginal OFV (plain={nll_plain}, frem={nll_frem})"
    );
}

#[test]
fn test_build_block_diag_omega_structure() {
    let bsv = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.09, 0.04]));
    let iov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.01]));
    let combined = build_block_diag_omega(&bsv, &iov, 2);
    // 2 BSV + 2*1 IOV = 4x4
    assert_eq!(combined.nrows(), 4);
    assert_eq!(combined.ncols(), 4);
    assert_eq!(combined[(0, 0)], 0.09);
    assert_eq!(combined[(1, 1)], 0.04);
    assert_eq!(combined[(2, 2)], 0.01); // occ 1 kappa
    assert_eq!(combined[(3, 3)], 0.01); // occ 2 kappa
    assert_eq!(combined[(0, 2)], 0.0); // off-block must be zero
}

/// Under additive error the Almquist `½·c̃'·c̃` correction is zero, so
/// `foce_subject_nll_interaction` equals the hand-computed
/// `a'·diag(1/R)·a + Ω⁻¹` Laplace value bit for bit. (FOCEI ≠ FOCE here: that identity
/// holds only for the Sheiner–Beal form.)
#[test]
fn test_focei_laplace_additive_matches_handcomputed_hessian() {
    let subj = make_simple_subject();
    let mut model = make_model();
    model.error_model = ErrorModel::Additive;
    model.error_spec = ErrorSpec::Single(ErrorModel::Additive);

    let theta = vec![5.0, 50.0];
    let eta_hat = nalgebra::DVector::from_vec(vec![0.05]);
    let omega = make_omega(0.09);
    let sigma = vec![1.0];

    let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
    let n_obs = subj.obs_times.len();
    let eps = 1e-6;
    let mut h = DMatrix::zeros(n_obs, 1);
    let h_step = eps * (1.0 + eta_hat[0].abs());
    let preds_p = pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
    let preds_m = pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
    for i in 0..n_obs {
        h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
    }

    let espec = ErrorSpec::Single(ErrorModel::Additive);
    let focei = foce_subject_nll_interaction(
        &subj,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &espec,
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );

    // Hand-compute the Laplace value with c̃ ≡ 0 (additive R).
    let r = sigma[0] * sigma[0]; // ErrorModel::Additive: R = σ²
    let mut data_ll = 0.0;
    for j in 0..n_obs {
        let res = subj.observations[j] - ipreds[j];
        data_ll += res * res / r + r.ln();
    }
    let eta_prior = eta_hat.dot(&(&omega.inv * &eta_hat));
    let mut htilde_scalar = omega.inv[(0, 0)];
    for j in 0..n_obs {
        htilde_scalar += h[(j, 0)] * h[(j, 0)] / r;
    }
    let log_det_htilde = htilde_scalar.ln(); // 1×1 case
    let expected = 0.5 * (data_ll + eta_prior + omega.log_det + log_det_htilde);

    assert!(
        (focei - expected).abs() < 1e-9,
        "FOCEI Laplace ({}) must equal hand-computed value ({}) under \
         additive error; diff = {}",
        focei,
        expected,
        focei - expected,
    );
}

/// IIV on RUV (#409): a dedicated residual eta with additive error scales R by
/// exp(2·η̂_ruv) and adds `c̃_{j,ruv}=2`, so `H̃ = 2·n_obs + 1/ω`; hand-computed, bit for bit.
#[test]
fn test_focei_iiv_on_ruv_matches_handcomputed() {
    let subj = make_simple_subject();
    let n_obs = subj.observations.len();
    // Residual-error eta only: its prediction-Jacobian column is zero.
    let eta = 0.2_f64;
    let eta_hat = nalgebra::DVector::from_vec(vec![eta]);
    let omega_var = 0.05_f64;
    let omega = OmegaMatrix::from_diagonal(&[omega_var], vec!["ETA_RUV".into()]);
    let sigma = vec![2.0_f64];
    let espec = ErrorSpec::Single(ErrorModel::Additive);
    // Arbitrary predictions; the residual eta does not enter them.
    let ipreds = vec![48.0, 38.0, 32.0, 44.0, 36.0, 26.0];
    let h = DMatrix::<f64>::zeros(n_obs, 1); // ∂f/∂η_ruv ≡ 0

    let focei = foce_subject_nll_interaction(
        &subj,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &espec,
        BloqMethod::Drop,
        &[],
        None,    // no FREM override
        Some(0), // ETA_RUV is eta index 0
        None,    // no custom magnitude
    );

    // Hand-computed marginal.
    let s = (2.0 * eta).exp();
    let r = sigma[0] * sigma[0] * s; // additive R, scaled
    let mut data_ll = 0.0;
    for j in 0..n_obs {
        let res = subj.observations[j] - ipreds[j];
        data_ll += res * res / r + r.ln();
    }
    let eta_prior = eta * eta / omega_var;
    // H̃ = hrh(0) + 0.5·ctc + Ω⁻¹; ctc(ruv,ruv) = Σ_j 2² = 4·n_obs.
    let htilde = 0.5 * (4.0 * n_obs as f64) + 1.0 / omega_var;
    let expected = 0.5 * (data_ll + eta_prior + omega.log_det + htilde.ln());

    assert!(
        (focei - expected).abs() < 1e-9,
        "FOCEI IIV-on-RUV marginal ({focei}) must equal hand-computed ({expected}); \
         diff = {}",
        focei - expected
    );

    // Sanity: passing `None` (no residual eta) drops both the scaling and the
    // c̃ column, so the marginal must differ.
    let focei_none = foce_subject_nll_interaction(
        &subj,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &espec,
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );
    assert!(
        (focei - focei_none).abs() > 1e-6,
        "residual-eta marginal must differ from the no-RUV-eta marginal"
    );
}

#[test]
fn compute_cwres_does_not_scale_frem_rows_by_iiv_on_ruv() {
    let subject = Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 2.0],
        observations: vec![11.0, 21.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        fremtype: vec![0, 100],
        ..Default::default()
    };
    let ipreds = vec![10.0, 20.0];
    let eta_hat = DVector::from_vec(vec![0.5]);
    let h = DMatrix::<f64>::zeros(2, 1);
    let omega = OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_RUV".into()]);
    let sigma = vec![2.0, 0.5];
    let overrides = vec![None, Some(0.25)];

    let cwres = compute_cwres(
        &subject,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &ErrorSpec::Single(ErrorModel::Additive),
        &[],
        Some(&overrides),
        Some(0),
        None,
        None,
        BloqMethod::Drop,
    );

    let pk_expected = 1.0 / (sigma[0] * eta_hat[0].exp());
    let frem_expected = 1.0 / sigma[1];
    assert_relative_eq!(cwres[0], pk_expected, epsilon = 1e-12);
    assert_relative_eq!(cwres[1], frem_expected, epsilon = 1e-12);
}

/// #1182: CWRES is `R̃^{-1/2}(y − f0)` with the symmetric square root, not the marginal
/// standardisation. Oracle: `R̃^{1/2}·z = y − f0` (independent sqrt) and `zᵀz = rᵀR̃⁻¹r`.
#[test]
fn compute_cwres_decorrelates_with_the_symmetric_inverse_square_root() {
    let subject = Subject {
        id: "1".into(),
        obs_times: vec![1.0, 2.0],
        observations: vec![3.0, 5.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        ..Default::default()
    };
    // ipred = f0 + H·η̂ with η̂ = 0.5 and H = [1, 2]ᵀ → f0 = (1, 2).
    let ipreds = vec![1.5, 3.0];
    let eta_hat = DVector::from_vec(vec![0.5]);
    let h = DMatrix::from_column_slice(2, 1, &[1.0, 2.0]);
    let omega = OmegaMatrix::from_diagonal(&[0.25], vec!["ETA".into()]);
    let sigma = vec![0.5];
    let spec = ErrorSpec::Single(ErrorModel::Additive);
    let cwres = compute_cwres(
        &subject,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &spec,
        &[],
        None,
        None,
        None,
        None,
        BloqMethod::Drop,
    );
    // R̃ = H Ω Hᵀ + σ²I = [[0.5, 0.5], [0.5, 1.25]]; resid = (2, 3).
    let r_tilde = DMatrix::from_row_slice(2, 2, &[0.5, 0.5, 0.5, 1.25]);
    let resid = DVector::from_vec(vec![2.0, 3.0]);
    let z = DVector::from_vec(cwres.clone());
    // R̃^{1/2} via the eigendecomposition, independently of the function.
    let eig = nalgebra::SymmetricEigen::new(r_tilde.clone());
    let sqrt_l = DMatrix::from_diagonal(&eig.eigenvalues.map(f64::sqrt));
    let r_half = &eig.eigenvectors * sqrt_l * eig.eigenvectors.transpose();
    let back = r_half * &z;
    assert_relative_eq!(back[0], resid[0], epsilon = 1e-12);
    assert_relative_eq!(back[1], resid[1], epsilon = 1e-12);
    let mahalanobis = (resid.transpose() * r_tilde.try_inverse().unwrap() * &resid)[(0, 0)];
    assert_relative_eq!(z.dot(&z), mahalanobis, epsilon = 1e-12);
    // Neither the marginal standardisation nor the Cholesky factor gives
    // this vector: the marginal second row would be 3/√1.25, and the
    // Cholesky first row 2/√0.5.
    assert!((cwres[1] - 3.0 / 1.25f64.sqrt()).abs() > 0.1);
    assert!((cwres[0] - 2.0 / 0.5f64.sqrt()).abs() > 0.1);

    // A first row censored *for this fit* (`CENS != 0` under M3) is NaN and
    // leaves the second row standardised on its own: 3/√1.25.
    let mut censored = subject.clone();
    censored.cens = vec![1, 0];
    let cw = compute_cwres(
        &censored,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &spec,
        &[],
        None,
        None,
        None,
        None,
        BloqMethod::M3,
    );
    assert!(cw[0].is_nan());
    assert_relative_eq!(cw[1], 3.0 / 1.25f64.sqrt(), epsilon = 1e-12);

    // `r_preds` moves the point R is evaluated at: on a proportional model
    // with a single row, CWRES = resid / √(HΩHᵀ + σ²·pred²).
    let one = Subject {
        id: "1".into(),
        obs_times: vec![1.0],
        observations: vec![3.0],
        obs_cmts: vec![1],
        cens: vec![0],
        ..Default::default()
    };
    let prop = ErrorSpec::Single(ErrorModel::Proportional);
    let h1 = DMatrix::from_column_slice(1, 1, &[1.0]);
    let cw = compute_cwres(
        &one,
        &ipreds[..1],
        &eta_hat,
        &h1,
        &omega,
        &sigma,
        &prop,
        &[],
        None,
        None,
        None,
        Some(&[4.0]),
        BloqMethod::Drop,
    );
    assert_relative_eq!(cw[0], 2.0 / (0.25_f64 + 0.25 * 16.0).sqrt(), epsilon = 1e-12);
    let cw_f0 = compute_cwres(
        &one,
        &ipreds[..1],
        &eta_hat,
        &h1,
        &omega,
        &sigma,
        &prop,
        &[],
        None,
        None,
        None,
        None,
        BloqMethod::Drop,
    );
    assert_relative_eq!(cw_f0[0], 2.0 / (0.25_f64 + 0.25 * 1.0).sqrt(), epsilon = 1e-12);
}

#[test]
fn compute_cwres_applies_custom_magnitude() {
    // #484 review #4: sdtab CWRES must use the per-observation magnitude.
    // With H = 0 the SB prediction f0 = ipred and R_tilde = R (diagonal), so
    // for additive error CWRES_j = (y_j − f_j) / (m_j·σ).
    let subject = Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 2.0],
        observations: vec![12.0, 22.0],
        obs_cmts: vec![1, 1],
        cens: vec![0, 0],
        fremtype: vec![0, 0],
        ..Default::default()
    };
    let ipreds = vec![10.0, 20.0];
    let eta_hat = DVector::from_vec(vec![0.0]);
    let h = DMatrix::<f64>::zeros(2, 1);
    let omega = OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]);
    let sigma = vec![2.0];
    // Row 0 bare (m = 1), row 1 inflated (m = 3).
    let mult = vec![vec![1.0], vec![3.0]];

    let cwres = compute_cwres(
        &subject,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma,
        &ErrorSpec::Single(ErrorModel::Additive),
        &[],
        None,
        None,
        Some(&mult),
        None,
        BloqMethod::Drop,
    );
    assert_relative_eq!(cwres[0], 2.0 / (1.0 * 2.0), epsilon = 1e-12);
    assert_relative_eq!(cwres[1], 2.0 / (3.0 * 2.0), epsilon = 1e-12);
}

/// IIV on residual error (#409): `individual_nll` must multiply the residual
/// variance by exp(2·η_ruv). At η_ruv = 0 the scale is 1 (bit-identical to no
/// IIV-on-RUV); at η_ruv ≠ 0 the value must change, and by exactly the
/// closed-form amount for additive error.
#[test]
fn test_individual_nll_scales_residual_variance() {
    let subj = make_simple_subject();
    let mut model = make_model();
    model.error_model = ErrorModel::Additive;
    model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
    let theta = vec![5.0, 50.0];
    let omega = make_omega(0.09);
    let sigma = vec![2.0];

    // η_ruv = 0 → scale 1 → identical to the no-RUV-eta model.
    let eta0 = vec![0.0];
    let base0 = individual_nll(&model, &subj, &theta, &eta0, &omega, &sigma);
    model.residual_error_eta = Some(0);
    let ruv0 = individual_nll(&model, &subj, &theta, &eta0, &omega, &sigma);
    assert!((base0 - ruv0).abs() < 1e-12, "η_ruv=0 must give scale 1 (base {base0}, ruv {ruv0})");

    // η_ruv = 0.3 → variance ×exp(0.6). Difference vs the unscaled model is
    // 0.5·Σ_j[(1/s − 1)·res²/σ² + ln s] (prior/|Ω| terms cancel).
    let eta = vec![0.3];
    model.residual_error_eta = None;
    let base = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    model.residual_error_eta = Some(0);
    let scaled = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma);
    let s = (2.0_f64 * 0.3).exp();
    let preds = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta);
    let sig2 = sigma[0] * sigma[0];
    let mut delta = 0.0;
    for (j, &f) in preds.iter().enumerate() {
        let res = subj.observations[j] - f;
        delta += (1.0 / s - 1.0) * (res * res / sig2) + s.ln();
    }
    let expected = base + 0.5 * delta;
    assert!(
        (scaled - expected).abs() < 1e-9,
        "individual_nll IIV-on-RUV scaling mismatch: got {scaled}, expected {expected}"
    );
}

/// `obs_nll_subject_into` (the IS/IMPMAP/SAEM data term) must apply the same
/// exp(2·η_ruv) variance scaling.
#[test]
fn test_obs_nll_subject_into_scales_residual_variance() {
    let subj = make_simple_subject();
    let mut model = make_model();
    model.error_model = ErrorModel::Additive;
    model.error_spec = ErrorSpec::Single(ErrorModel::Additive);
    let theta = vec![5.0, 50.0];
    let sigma = vec![2.0];
    let eta = vec![0.4];
    let mut scratch = pk::EventPkParams::with_capacity_for(&subj);

    let base = obs_nll_subject_into(
        &model,
        &subj,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &eta,
        &mut scratch,
    );
    model.residual_error_eta = Some(0);
    let scaled = obs_nll_subject_into(
        &model,
        &subj,
        &theta,
        &sigma,
        ResidualCov::block_sigma(&[]),
        &eta,
        &mut scratch,
    );

    let s = (2.0_f64 * 0.4).exp();
    let preds = pk::compute_predictions_with_tv(&model, &subj, &theta, &eta);
    let sig2 = sigma[0] * sigma[0];
    let mut expected = 0.0;
    for (j, &f) in preds.iter().enumerate() {
        let v = (sig2 * s).max(1e-12);
        expected += 0.5 * (v.ln() + (subj.observations[j] - f).powi(2) / v);
    }
    assert!(
        (scaled - expected).abs() < 1e-9,
        "obs_nll IIV-on-RUV scaling mismatch: got {scaled}, expected {expected}, base {base}"
    );
}

/// The Almquist `½·c̃'·c̃` correction is wired: additive → combined error changes the
/// per-subject OFV non-trivially even when the data term barely shifts.
#[test]
fn test_focei_laplace_combined_uses_inter_correction() {
    let subj = make_simple_subject();
    let mut model = make_model();
    model.error_model = ErrorModel::Combined;
    model.error_spec = ErrorSpec::Single(ErrorModel::Combined);

    let theta = vec![5.0, 50.0];
    let eta_hat = nalgebra::DVector::from_vec(vec![0.1]);
    let omega = make_omega(0.09);
    // Two sigmas for Combined: (prop, add). The proportional component is
    // intentionally small relative to additive so the data-side R hardly
    // changes — any noticeable OFV shift is then due to the c̃'c̃ piece.
    let sigma_combined = vec![0.02, 1.0];
    let sigma_additive_only = vec![1.0]; // matches Combined R when prop ≈ 0

    let ipreds = pk::compute_predictions_with_tv(&model, &subj, &theta, eta_hat.as_slice());
    let n_obs = subj.obs_times.len();
    let eps = 1e-6;
    let mut h = DMatrix::zeros(n_obs, 1);
    let h_step = eps * (1.0 + eta_hat[0].abs());
    let preds_p = pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] + h_step]);
    let preds_m = pk::compute_predictions_with_tv(&model, &subj, &theta, &[eta_hat[0] - h_step]);
    for i in 0..n_obs {
        h[(i, 0)] = (preds_p[i] - preds_m[i]) / (2.0 * h_step);
    }

    let espec_combined = ErrorSpec::Single(ErrorModel::Combined);
    let espec_additive = ErrorSpec::Single(ErrorModel::Additive);
    let focei_combined = foce_subject_nll_interaction(
        &subj,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma_combined,
        &espec_combined,
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );
    let focei_additive = foce_subject_nll_interaction(
        &subj,
        &ipreds,
        &eta_hat,
        &h,
        &omega,
        &sigma_additive_only,
        &espec_additive,
        BloqMethod::Drop,
        &[],
        None,
        None,
        None,
    );
    let gap = focei_combined - focei_additive;
    assert!(
        gap.abs() > 1e-3,
        "FOCEI Laplace must respond to the Almquist `½·c̃'·c̃` INTER \
         correction; combined ({}) and additive ({}) gave gap = {} — \
         too small, c̃'c̃ likely not being accumulated.",
        focei_combined,
        focei_additive,
        gap,
    );
}

/// Joint PK-TTE (ODE-accumulated hazard) must give a finite per-subject NLL on every
/// estimation entry point: the inner EBE NLL, the SAEM M-step, and the FOCEI Laplace
/// objective (which adds the TTE FD-Hessian). Fast — a handful of ODE solves, no outer
/// optimisation (the full fit is the Tier-3 test in `tte_convergence.rs`). This is the
/// per-PR coverage for the ODE-hazard likelihood path (#564).
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_ode_hazard_nll_paths_finite() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    let src = crate::test_models::ODE_9;
    let model = parse_model_string(src).expect("joint PK-TTE model must parse");
    let mut subject = make_simple_subject();
    // make_simple_subject doses CMT 1 (depot) with 6 PK obs; add a TTE event on CMT 2.
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];

    let p = &model.default_params;
    let eta = [0.0_f64];
    let n_obs = subject.observations.len();

    // (1) Inner EBE NLL: individual_nll → tte_endpoint_nll → tte_ode_nll.
    let inner = individual_nll(&model, &subject, &p.theta, &eta, &p.omega, &p.sigma.values);
    assert!(inner.is_finite(), "inner joint NLL must be finite; got {inner}");

    // (2) SAEM M-step: obs_nll_subject_from_preds (Gaussian preds supplied).
    let preds = vec![30.0_f64; n_obs];
    let saem = obs_nll_subject_from_preds(
        &model,
        &subject,
        &preds,
        &p.theta,
        &p.sigma.values,
        model.residual_cov_at(p),
        &eta,
    );
    assert!(saem.is_finite(), "SAEM M-step joint NLL must be finite; got {saem}");

    // (3) FOCEI Laplace objective, incl. the ODE-TTE FD-Hessian term. A zero Gaussian
    // sensitivity matrix is fine here — the prior + TTE Hessian keep H̃ positive-definite.
    let eta_hat = DVector::from_vec(vec![0.0]);
    let h_matrix = DMatrix::<f64>::zeros(n_obs, 1);
    let foce = foce_subject_nll(
        &model,
        &subject,
        &p.theta,
        &eta_hat,
        &h_matrix,
        &p.omega,
        &p.sigma.values,
        &p.residual_correlations,
        true,
    );
    assert!(foce.is_finite(), "FOCEI joint NLL must be finite; got {foce}");

    // The ODE hazard must actually contribute: moving η changes the inner NLL.
    let moved = individual_nll(&model, &subject, &p.theta, &[0.4], &p.omega, &p.sigma.values);
    assert!(
        moved.is_finite() && (moved - inner).abs() > 1e-9,
        "η must move the joint NLL (inner={inner}, moved={moved})"
    );

    // #570: the shared single-solve path must agree with the dedicated two-solve
    // path it replaces — predictions bit-identical, TTE term equal to solver
    // tolerance. Use a non-zero η so the PK trajectory (and thus the hazard) is
    // non-trivial. This is what makes the fast coverage of the share *meaningful*
    // rather than merely "finite".
    let eta_nz = [0.3_f64];
    let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &eta_nz)
        .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
    let mut scratch = pk::EventPkParams::with_capacity_for(&subject);
    let preds_ref = crate::pk::compute_predictions_with_tv_into_with_schedule(
        &model,
        &subject,
        &p.theta,
        &eta_nz,
        &mut scratch,
        None,
    );
    assert_eq!(
        share.preds, preds_ref,
        "shared preds must be bit-identical to the standalone prediction path"
    );
    let (chz_state, hazard) = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: h @ HazardSpec::OdeAccumulated { chz_state },
            ..
        }) => (*chz_state, h),
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };
    let shared_tte = tte_ode_nll_from_shared(
        model.ode_spec.as_ref().unwrap(),
        &share,
        chz_state,
        &subject.obs_records,
    );
    let dedicated_tte = tte_endpoint_nll(
        &model,
        &subject,
        hazard,
        crate::types::TteRecurrence::Single,
        &subject.obs_records,
        &p.theta,
        &eta_nz,
    );
    assert!(
        (shared_tte - dedicated_tte).abs() <= 1e-4 * dedicated_tte.abs().max(1.0),
        "shared TTE NLL {shared_tte} must match dedicated {dedicated_tte} to solver tol"
    );
}

/// The joint PK-TTE harness model of [`joint_pktte_ode_hazard_nll_paths_finite`],
/// as a helper so the #1223 pre-start arms use the same PK block, hazard and error
/// model without editing that test's fixture.
#[cfg(feature = "survival")]
fn joint_pktte_ode_model() -> CompiledModel {
    crate::parser::model_parser::parse_model_string(crate::test_models::ODE_9)
        .expect("joint PK-TTE model must parse")
}

/// [`joint_pktte_ode_model`] without its compiled `[odes]` program: a spec the walks
/// cannot record (`ode_hazard_walk_in_scope` requires `rhs_program`), while the share and
/// the dedicated solve integrate the same `ode.rhs` closure. The fixture for the TTE
/// router's non-walk arms — the one difference from the walk arm's model is the program,
/// so the curves are the plain model's.
#[cfg(feature = "survival")]
fn walk_declined_joint_pktte_model() -> CompiledModel {
    let mut model = joint_pktte_ode_model();
    model.ode_spec.as_mut().expect("ODE model").rhs_program = None;
    model
}

/// The router's walk-arm predicate for `s` on the program-bearing model and on
/// [`walk_declined_joint_pktte_model`]: `(true, false)` is the straddle the non-walk arms'
/// tests stand on — the same subject, the walk taken exactly when the program is there.
#[cfg(feature = "survival")]
fn walk_scope_straddle(s: &Subject) -> (bool, bool) {
    let in_scope = crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope;
    (in_scope(&joint_pktte_ode_model(), s), in_scope(&walk_declined_joint_pktte_model(), s))
}

/// #1885's ODE fixture (the plan's M4): [`joint_pktte_ode_model`]'s subject on oral
/// 100 at 0/12/24, so the later doses land on residual drug, with PK observations at
/// 2/6/14/26 and one exact event at `t_event` on CMT 2. With `reset`, one EVID=3
/// reset after every record, which declines the #570 share and sends the likelihood
/// to the dedicated `tte_ode_nll` solve.
#[cfg(feature = "survival")]
fn ebe_pktte_subject(id: &str, t_event: f64, reset: bool) -> Subject {
    use crate::types::{EventType, ObsRecord};
    let mut s = make_simple_subject();
    s.id = id.to_string();
    s.doses =
        [0.0, 12.0, 24.0].iter().map(|&t| DoseEvent::new(t, 100.0, 1, 0.0, false, 0.0)).collect();
    s.obs_times = vec![2.0, 6.0, 14.0, 26.0];
    s.observations = vec![7.0, 6.0, 9.0, 10.0];
    s.obs_cmts = vec![1; 4];
    s.cens = vec![0; 4];
    s.occasions = vec![1; 4];
    s.obs_records = vec![ObsRecord::Event {
        time: t_event,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 2,
    }];
    if reset {
        s.reset_times = vec![60.0];
    }
    s
}

#[cfg(feature = "survival")]
fn ebe_population(subjects: Vec<Subject>) -> crate::types::Population {
    crate::types::Population {
        subjects,
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// #1885 T2: for a joint PK-TTE subject scored on the **dedicated** solve,
/// `predict_survival_at_etas_diag` on its scored times reads `H`/`h` bit for bit, and
/// the exact-event term rebuilt from the row equals `tte_endpoint_nll`. Two subjects
/// in reverse order catch a zero η or a zip slip; the rebuilt term catches another engine.
#[cfg(feature = "survival")]
#[test]
fn ebe_survival_reads_the_dedicated_ode_solve_bit_for_bit() {
    use crate::types::{EventType, ObsRecord};
    let model = joint_pktte_ode_model();
    let p = &model.default_params;
    let (chz_state, hazard) = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: h @ HazardSpec::OdeAccumulated { chz_state },
            ..
        }) => (*chz_state, h),
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };
    let t = 25.3;
    let cases = [("A", 0.4), ("B", -0.7)];
    let pop = ebe_population(cases.iter().map(|(id, _)| ebe_pktte_subject(id, t, true)).collect());
    let etas = crate::api::SuppliedEtas::from_table(
        vec!["B".into(), "A".into()],
        vec!["ETA_CL".into()],
        vec![vec![-0.7], vec![0.4]],
    )
    .unwrap();
    let out = crate::api::predict_survival_at_etas_diag(&model, &pop, p, &etas, &[t]).unwrap();
    assert_eq!(out.results.len(), 2);
    let tol = crate::survival::MonoTol::from_solver(
        &model.ode_spec.as_ref().unwrap().effective_solver_opts(),
    );
    for ((id, eta), subject) in cases.iter().zip(&pop.subjects) {
        assert!(
            try_joint_pktte_shared_solve(&model, subject, &p.theta, &[*eta]).is_none(),
            "subject {id} must be scored on the dedicated solve, not the share"
        );
        let row = out.results.iter().find(|r| r.id == *id).unwrap();
        assert!(row.cum_hazard.is_finite() && row.hazard.is_finite());
        let (cum, haz) =
            crate::survival::ode_cumhaz_hazard(&model, subject, chz_state, &p.theta, &[*eta], &[t]);
        assert_eq!(
            (row.cum_hazard.to_bits(), row.hazard.to_bits()),
            (cum[0].to_bits(), haz[0].to_bits()),
            "subject {id}: predicted ({}, {}) vs scored ({}, {})",
            row.cum_hazard,
            row.hazard,
            cum[0],
            haz[0]
        );
        let records =
            [ObsRecord::Event { time: t, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];
        let scored = tte_endpoint_nll(
            &model,
            subject,
            hazard,
            crate::types::TteRecurrence::Single,
            &records,
            &p.theta,
            &[*eta],
        );
        assert!(scored.is_finite() && scored < 1e19, "{scored}");
        let rebuilt =
            tte_ode_nll_from_curves(&records, &[t], &[row.cum_hazard], &[row.hazard], tol);
        assert_eq!(rebuilt.to_bits(), scored.to_bits(), "subject {id}");
    }
}

/// #1885 T3: on the #570 **share**, the individual curve (0.5-step grid 0–48 plus Tᵢ)
/// agrees with the share's `H(Tᵢ)` to ODE tolerance, not bitwise.
///
/// Band, measured: worst relative gap 1.300e-4 (Linux aarch64 and macOS arm64); the
/// bound is ~2.3× above it. η̂ moving `H(Tᵢ)` by more than half excludes the η = 0 curve.
#[cfg(feature = "survival")]
#[test]
fn ebe_survival_agrees_with_the_share_to_solver_tolerance() {
    let model = joint_pktte_ode_model();
    let p = &model.default_params;
    let chz_state = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: HazardSpec::OdeAccumulated { chz_state }, ..
        }) => *chz_state,
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };
    const BAND: f64 = 3e-4;
    let mut worst = 0.0_f64;
    for t in [30.0, 25.3] {
        let mut grid: Vec<f64> = (0..=96).map(|k| 0.5 * f64::from(k)).collect();
        grid.push(t);
        for eta in [0.4, -0.7] {
            let subject = ebe_pktte_subject("1", t, false);
            let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[eta])
                .expect("the reset-free subject must be scored on the share");
            let i = share.times.iter().position(|&s| s == t).unwrap();
            let h_share = share.chz_states[i][chz_state];
            let pop = ebe_population(vec![subject]);
            let at = |e: f64| {
                let etas = crate::api::SuppliedEtas::from_table(
                    vec!["1".into()],
                    vec!["ETA_CL".into()],
                    vec![vec![e]],
                )
                .unwrap();
                crate::api::predict_survival_at_etas_diag(&model, &pop, p, &etas, &grid)
                    .unwrap()
                    .results
                    .into_iter()
                    .find(|r| r.time == t)
                    .unwrap()
                    .cum_hazard
            };
            let (h_pred, h_zero) = (at(eta), at(0.0));
            assert!(
                h_share.is_finite() && h_pred.is_finite() && h_zero.is_finite(),
                "T={t} η={eta}: share {h_share}, predict {h_pred}, η=0 {h_zero}"
            );
            let rel = (h_pred - h_share).abs() / h_share;
            worst = worst.max(rel);
            eprintln!("T3 T={t} eta={eta}: share {h_share} predict {h_pred} rel {rel:.3e}");
            assert!(rel <= BAND, "T={t} η={eta}: rel {rel:e} > {BAND:e}");
            let moved = (h_pred - h_zero).abs() / h_pred.max(h_zero);
            assert!(moved >= 0.5, "T={t} η={eta}: η̂ moved H by {moved} only");
        }
    }
    eprintln!("T3 worst share-vs-predict rel gap {worst:.3e}");
}

/// No-TV `[odes]` plus an `OdeAccumulated` hazard reading `TAD` — the shape
/// `try_joint_pktte_shared_solve` admits to the #570 share (its time guard does not
/// inspect the injected `d/dt(__chz_*)` line). Shared by the two tests below.
#[cfg(feature = "survival")]
fn tad_hazard_two_dose_model() -> CompiledModel {
    use crate::parser::model_parser::parse_model_string;
    parse_model_string(
        r"
[parameters]
theta TVCL(1.0, 0.01, 100.0)
theta TVV(10.0, 0.1, 500.0)
theta TVH0(0.1, 0.001, 10.0)
theta TVKT(0.01, 0.0001, 1.0)
sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
CL = TVCL
V = TVV
H0 = TVH0
KT = TVKT
[structural_model]
ode(obs_cmt=central, states=[central])
[odes]
d/dt(central) = -CL / V * central
[event_model]
cmt = 3
hazard = H0 * (1.0 + KT * TAD)
[error_model]
DV ~ proportional(PROP_ERR)
",
    )
    .expect("TAD-reading ODE hazard must parse")
}

/// A joint PK-TTE subject whose first record is a dose at **t = 10**, so a TTE time
/// below it is pre-start; PK obs at 12/16/24 route the pre-start read through the
/// pre-first-break fill, not the `k = 0` boundary visit (#1218).
/// #1261: a TTE hazard readout must receive the dose-time anchors the ODE used (a bare
/// `PkParams::values` made TAD NaN → 1e20 sentinel).
#[cfg(feature = "survival")]
#[test]
fn tte_hazard_readout_uses_the_current_tad_anchor() {
    use crate::types::{EventType, ObsRecord};

    let model = tad_hazard_two_dose_model();
    let (chz_state, hazard) = match model.endpoints.get(&3) {
        Some(EndpointLikelihood::Tte {
            hazard: h @ HazardSpec::OdeAccumulated { chz_state },
            ..
        }) => (*chz_state, h),
        _ => panic!("expected ODE-accumulated endpoint"),
    };
    let mut subject = make_simple_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.obs_times = vec![30.0];
    subject.observations = vec![0.0];
    subject.obs_cmts = vec![1];
    subject.cens = vec![0];
    subject.occasions = vec![1];
    subject.obs_records = vec![ObsRecord::Event {
        time: 30.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let p = &model.default_params;
    let eta = [];
    let nll = tte_endpoint_nll(
        &model,
        &subject,
        hazard,
        crate::types::TteRecurrence::Single,
        &subject.obs_records,
        &p.theta,
        &eta,
    );

    // H(30) = .1 * [integral_0^12(1 + .01t) + integral_12^30(1 + .01(t - 12))] = 3.234.
    // h(30) = .1 * (1 + .01 * 18) = .118. Two doses prevent TAD and TAFD aliasing.
    // The 1e10 guard is ten orders below the old 2e20 sentinel and nine orders above
    // this roughly 5.37 objective, so it rejects the masked NaN without overfitting
    // to solver-level round-off.
    let expected = 3.234_f64 - 0.118_f64.ln();
    assert_relative_eq!(nll, expected, epsilon = 2e-4);
    assert!(
        nll < 1e10,
        "the old NaN path returns the finite 1e20 sentinel, not an infinite NLL; got {nll}"
    );

    let (cum, haz) =
        crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, &p.theta, &eta, &[30.0]);
    assert_relative_eq!(cum[0], 3.234, epsilon = 2e-4);
    assert_relative_eq!(haz[0], 0.118, epsilon = 2e-6);
}

/// #1261 follow-up: through the production dispatcher
/// `individual_nll_into_with_schedule`, which routes the TTE term through
/// `tte_ode_nll_from_shared` / `share.rhs_params[i]`. The TTE contribution is isolated
/// by subtracting a PK-only sibling (same `[odes]`, doses, obs; no share), whose
/// Gaussian term is bit-identical.
#[cfg(feature = "survival")]
#[test]
fn hot_path_individual_nll_reads_the_shared_solve_tad_anchor() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    let model = tad_hazard_two_dose_model();
    let pk_only_model = parse_model_string(
        r"
[parameters]
theta TVCL(1.0, 0.01, 100.0)
theta TVV(10.0, 0.1, 500.0)
sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
CL = TVCL
V = TVV
[structural_model]
ode(obs_cmt=central, states=[central])
[odes]
d/dt(central) = -CL / V * central
[error_model]
DV ~ proportional(PROP_ERR)
",
    )
    .expect("PK-only sibling model must parse");

    let mut subject = make_simple_subject();
    subject.doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
        DoseEvent::new(12.0, 100.0, 1, 0.0, false, 0.0),
    ];
    subject.obs_times = vec![30.0];
    subject.observations = vec![0.0];
    subject.obs_cmts = vec![1];
    subject.cens = vec![0];
    subject.occasions = vec![1];
    subject.obs_records = vec![ObsRecord::Event {
        time: 30.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 3,
    }];
    let mut pk_only_subject = subject.clone();
    pk_only_subject.obs_records = Vec::new();

    let eta = [];

    // The fixture must actually reach the shared solve — otherwise this test would
    // silently exercise the pre-#570 fallback the sibling test already covers,
    // rather than the `tte_ode_nll_from_shared` dispatch under review here.
    assert!(
        try_joint_pktte_shared_solve(&model, &subject, &model.default_params.theta, &eta).is_some(),
        "fixture must qualify for the #570 shared solve"
    );

    let mut scratch = crate::pk::EventPkParams::default();
    let p = &model.default_params;
    let nll_full = individual_nll_into_with_schedule(
        &model,
        &subject,
        &p.theta,
        &eta,
        &p.omega,
        &p.sigma.values,
        &model.residual_correlations,
        &mut scratch,
        None,
    );

    let pk_p = &pk_only_model.default_params;
    let nll_pk_only = individual_nll_into_with_schedule(
        &pk_only_model,
        &pk_only_subject,
        &pk_p.theta,
        &eta,
        &pk_p.omega,
        &pk_p.sigma.values,
        &pk_only_model.residual_correlations,
        &mut scratch,
        None,
    );

    // Same H(30) = 3.234, h(30) = .118 as the sibling test — the isolated TTE term
    // the shared solve must have contributed.
    let expected_tte = 3.234_f64 - 0.118_f64.ln();
    assert_relative_eq!(nll_full - nll_pk_only, expected_tte, epsilon = 2e-4);
    assert!(
        nll_full < 1e10,
        "the old NaN-TAD path returns the finite 1e20 sentinel through this exact \
         dispatcher; got {nll_full}"
    );
}

/// Construct a subject with a positive pre-start TTE time and later PK observations.
#[cfg(feature = "survival")]
fn late_start_joint_subject(records: Vec<ObsRecord>) -> Subject {
    let mut subject = make_simple_subject();
    subject.doses = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, false, 0.0)];
    subject.obs_times = vec![12.0, 16.0, 24.0];
    subject.observations = vec![50.0, 40.0, 30.0];
    subject.obs_cmts = vec![1; 3];
    subject.cens = vec![0; 3];
    subject.occasions = vec![1; 3];
    subject.obs_records = records;
    subject
}

/// #1223: a TTE time **at or before the first record** is scored, not repelled, and
/// identically on all three production callers (the #570 share, the dedicated solve,
/// `predict_survival`). Arms: left-truncation `entry_time`, interval `left`, right
/// censoring time. The seeded state is all zeros, so the discriminating value is
/// `h = H0 = 0.01` (the hand-computed oracle with `H = 0`); the non-zero-state check
/// lives in `ode::predictions::tests::prestart_chz_time_reads_the_seeded_state_on_both_paths`.
#[cfg(feature = "survival")]
#[test]
fn prestart_tte_time_is_scored_on_share_dedicated_and_predict_survival() {
    use crate::types::{EventType, ObsRecord};

    let model = joint_pktte_ode_model();
    let ode = model.ode_spec.as_ref().expect("model has an [odes] block");
    let (chz_state, hazard) = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: h @ HazardSpec::OdeAccumulated { chz_state },
            ..
        }) => (*chz_state, h),
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };
    let p = &model.default_params;
    let theta = &p.theta;
    // `H0` carries no η, and at the seeded state `central = 0`, so `h = H0` for any η.
    // η = 0 also matches what `predict_survival` evaluates at.
    let eta = [0.0_f64];
    let tv_h0 = theta[3];
    assert_eq!(tv_h0, 0.01, "theta[3] must be TVH0 — the arms read h against it");

    for (arm, t_pre, records) in [
        (
            "left truncation: Exact @20, entry 5",
            5.0_f64,
            vec![ObsRecord::Event {
                time: 20.0,
                event_type: EventType::Exact,
                entry_time: 5.0,
                cmt: 2,
            }],
        ),
        (
            "interval censoring: (5, 20]",
            5.0_f64,
            vec![ObsRecord::Event {
                time: 20.0,
                event_type: EventType::IntervalCensored { left: 5.0, right: 20.0 },
                entry_time: 0.0,
                cmt: 2,
            }],
        ),
        (
            "right censoring at 8, before the first record",
            8.0_f64,
            vec![ObsRecord::Event {
                time: 8.0,
                event_type: EventType::RightCensored,
                entry_time: 0.0,
                cmt: 2,
            }],
        ),
        // `left = 0.0` deliberately: `entry_time` is pushed only if `> 0`, `left` always, so
        // this reaches the share; and `predict_survival(&[0.0])` is #1218's zero-maximum grid
        // (PR #1222 coverage). Do not "tidy" it to 5.0/8.0.
        (
            "interval censoring with left = 0 (ungated bound)",
            0.0_f64,
            vec![ObsRecord::Event {
                time: 20.0,
                event_type: EventType::IntervalCensored { left: 0.0, right: 20.0 },
                entry_time: 0.0,
                cmt: 2,
            }],
        ),
    ] {
        let subject = late_start_joint_subject(records);

        // The straddle, asserted rather than assumed: the time under test is before
        // the integration start, and the subject still has an observation at or after
        // it, so the pre-start read goes through the fill (see the helper's note).
        let start = crate::ode::subject_integration_start(&subject);
        assert!(
            t_pre < start,
            "{arm}: fixture must straddle the start — t_pre {t_pre} vs start {start}"
        );
        assert!(
            subject.obs_times.iter().any(|&t| t >= start),
            "{arm}: fixture must keep an observation at or after the start"
        );

        // Engine 1 — the #570 one-solve share.
        let share = try_joint_pktte_shared_solve(&model, &subject, theta, &eta)
            .expect("model and subject must qualify for the #570 shared solve");
        let i = share
            .times
            .iter()
            .position(|&t| t == t_pre)
            .unwrap_or_else(|| panic!("{arm}: {t_pre} must be a shared-solve CHZ time"));
        let st = &share.chz_states[i];
        // Two assertions, not one `&&`: a width mismatch is a different defect from a
        // non-finite value, and a combined message would name the wrong one (printing a
        // perfectly finite `[0.0, 0.0]` under "non-finite") and send the next reader into
        // the wrong engine.
        assert_eq!(
            st.len(),
            ode.n_states,
            "{arm}: share state has {} slots, expected {}",
            st.len(),
            ode.n_states
        );
        assert!(
            st.iter().all(|x| x.is_finite()),
            "{arm}: share left the pre-start state non-finite: {st:?}"
        );
        let seeded = ode.initial_state(&share.pk_values);
        for j in 0..ode.n_states {
            assert_eq!(
                st[j].to_bits(),
                seeded[j].to_bits(),
                "{arm}: share state slot {j} = {} must be the seeded {}",
                st[j],
                seeded[j]
            );
        }
        // `H` and `h` read exactly as `tte_ode_nll_from_shared` reads them.
        let mut du = vec![0.0; ode.n_states];
        (ode.rhs)(st, &share.rhs_params[i], t_pre, &mut du);
        let (share_cum, share_haz) = (st[chz_state], du[chz_state]);

        // Engine 2 — the dedicated two-solve arm.
        let (cum_ded, haz_ded) =
            crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, theta, &eta, &[t_pre]);

        // Caller 3 — the public `predict_survival` (η = 0), on the same dense engine.
        let population = crate::types::Population {
            subjects: vec![subject.clone()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        };
        let surv = crate::api::predict_survival(&model, &population, p, &[t_pre]).unwrap();
        let row = surv
            .iter()
            .find(|r| r.cmt == 2 && r.time == t_pre)
            .unwrap_or_else(|| panic!("{arm}: predict_survival must report CMT 2 @{t_pre}"));

        // Every leg named separately, and finite before any comparison — a fold or a
        // `max` would let a NaN leg ride along on the ones that worked.
        for (leg, cum, haz) in [
            ("share (#570)", share_cum, share_haz),
            ("dedicated (ode_cumhaz_hazard)", cum_ded[0], haz_ded[0]),
            ("predict_survival", row.cum_hazard, row.hazard),
        ] {
            assert!(
                cum.is_finite() && haz.is_finite(),
                "{arm} / {leg}: H = {cum}, h = {haz} — must be finite"
            );
            // Nothing has acted on the system yet: `H = 0`, and `h = H0·exp(BETA·0)`
            // is `H0` bit-exactly (`exp(0.0) == 1.0`). Hand-computed, not read off
            // another engine.
            assert_eq!(cum, 0.0, "{arm} / {leg}: H({t_pre}) must be exactly 0");
            assert_eq!(haz, tv_h0, "{arm} / {leg}: h({t_pre}) must be exactly TVH0");
        }

        // And the per-record NLL agrees between the two engines that actually score.
        let shared_tte = tte_ode_nll_from_shared(ode, &share, chz_state, &subject.obs_records);
        let dedicated_tte = tte_endpoint_nll(
            &model,
            &subject,
            hazard,
            crate::types::TteRecurrence::Single,
            &subject.obs_records,
            theta,
            &eta,
        );
        assert!(
            shared_tte.is_finite() && dedicated_tte.is_finite(),
            "{arm}: TTE NLL must be finite — share {shared_tte}, dedicated {dedicated_tte}"
        );
        assert!(
            (shared_tte - dedicated_tte).abs() <= 1e-4 * dedicated_tte.abs().max(1.0),
            "{arm}: share {shared_tte} must match dedicated {dedicated_tte} to solver tol"
        );

        // FOCEI Laplace leg: the ODE-TTE FD-Hessian perturbs η around the pre-start node, so
        // a sentinel reinstated only inside the perturbation would pass the value checks.
        // Zero Gaussian H is fine: prior + TTE Hessian keep H̃ PD.
        let n_obs = subject.observations.len();
        let foce = foce_subject_nll(
            &model,
            &subject,
            theta,
            &DVector::from_vec(vec![0.0]),
            &DMatrix::<f64>::zeros(n_obs, 1),
            &p.omega,
            &p.sigma.values,
            &p.residual_correlations,
            true,
        );
        // `is_finite()` alone would be useless here: the TTE sentinel is `1e20`, which *is*
        // finite, so a Laplace objective built entirely on repelled records passes a
        // finiteness check. Bound it below the sentinel instead. `1e12` is ~8 orders under
        // `1e20` and ~10 orders over this fixture's actual objective (order 10), so it
        // discriminates "scored" from "repelled" without being a tolerance test.
        assert!(
            foce.is_finite() && foce.abs() < 1e12,
            "{arm}: FOCEI Laplace objective (incl. the TTE FD-Hessian) must be scored, not \
             repelled; got {foce} (the TTE sentinel is {SUBJECT_SENTINEL_NLL:e} and is itself finite)"
        );
    }
}

/// #1223 positive control straddling the integration start: a pre-start `entry_time`
/// contributes nothing (`H(entry) = 0`), a post-start one changes the objective. Rules
/// out both "ignore entry_time" and "start the clock at the earliest TTE time".
#[cfg(feature = "survival")]
#[test]
fn prestart_entry_time_contributes_nothing_post_start_entry_does() {
    use crate::types::{EventType, ObsRecord};

    let model = joint_pktte_ode_model();
    let p = &model.default_params;
    let eta = [0.0_f64];

    let nll_at = |entry_time: f64| -> f64 {
        let subject = late_start_joint_subject(vec![ObsRecord::Event {
            time: 20.0,
            event_type: EventType::Exact,
            entry_time,
            cmt: 2,
        }]);
        individual_nll(&model, &subject, &p.theta, &eta, &p.omega, &p.sigma.values)
    };

    let (none, pre, post) = (nll_at(0.0), nll_at(5.0), nll_at(12.0));
    assert!(
        none.is_finite() && pre.is_finite() && post.is_finite(),
        "every leg must be finite: none {none}, pre {pre}, post {post}"
    );

    // The integration starts at the dose (t = 10), so 5 is before it and 12 after —
    // the pair genuinely straddles the gate rather than sitting on one side of it.
    let subject = late_start_joint_subject(Vec::new());
    let start = crate::ode::subject_integration_start(&subject);
    assert!(5.0 < start && 12.0 > start, "the pair must straddle the integration start ({start})");

    // Bit-exact by construction (measured `306.2186273580891` both ways): the pre-start
    // node is filled before the break walk and `t_last` stays 24, so every segment is
    // identical. A tolerance would be strictly weaker.
    assert_eq!(
        pre.to_bits(),
        none.to_bits(),
        "a pre-start entry must contribute nothing: entry=5 {pre} vs entry=0 {none}"
    );
    // Post-start entry: the objective must move by `H(12) > 0`. Measured separation is
    // 1.363e-3 relative, so the `1e-6` floor is ~1300x below what a real contribution
    // produces — loose enough not to be a tolerance test, tight enough that "the entry
    // time was ignored" (separation 0) cannot pass.
    assert!(
        (post - none).abs() > 1e-6,
        "a post-start entry must change the objective: entry=12 {post} vs entry=0 {none}"
    );
}

/// #1187: the joint PK-TTE hazard's two production routes — the #570 share
/// (`active_infusions`) and the two-solve fallback (`gated_infusions`) — must agree for a
/// `RATE>0` dose into a built-in absorption forcing; before the fix they integrated
/// different exposures. Routing (#1166) can move a model between arms, so A ≡ B matters.
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_hazard_arms_agree_for_infusion_into_absorption() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, EventType, ObsRecord};

    // `first_order(ka=KA)` puts the absorption kernel *in* central (no depot state),
    // so CMT 1 is the input-rate compartment and a `RATE>0` dose there is #719 gap 2.
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(src).expect("joint PK-TTE model must parse");
    let mut subject = make_simple_subject();
    // Replace the bolus with an infusion into the input-rate compartment. This is the
    // one input that matters — with a bolus both arms already agree.
    let amt = subject.doses[0].amt;
    subject.doses = vec![DoseEvent::new(0.0, amt, 1, amt / 2.0, false, 0.0)];
    assert!(
        subject.doses[0].is_infusion(),
        "fixture must carry a real infusion or it tests nothing"
    );
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];

    let p = &model.default_params;
    let eta_nz = [0.3_f64]; // non-zero, so the PK trajectory (hence the hazard) is non-trivial

    let share = try_joint_pktte_shared_solve(&model, &subject, &p.theta, &eta_nz)
        .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
    let (chz_state, hazard) = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: h @ HazardSpec::OdeAccumulated { chz_state },
            ..
        }) => (*chz_state, h),
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };
    let arm_a = tte_ode_nll_from_shared(
        model.ode_spec.as_ref().unwrap(),
        &share,
        chz_state,
        &subject.obs_records,
    );
    let arm_b = tte_endpoint_nll(
        &model,
        &subject,
        hazard,
        crate::types::TteRecurrence::Single,
        &subject.obs_records,
        &p.theta,
        &eta_nz,
    );
    assert!(
        arm_a.is_finite() && arm_b.is_finite(),
        "both hazard arms must be finite (A={arm_a}, B={arm_b})"
    );
    // Non-degeneracy: a hazard that never left its baseline would let the two arms agree
    // for the wrong reason. Both inputs are read back out of the fixture rather than
    // inlined, so editing `TVH0` or the event time above cannot silently leave this
    // guard comparing against a stale baseline.
    let h0 = p.theta[3]; // TVH0 — 4th `theta` declared in `src` above
    assert!(
        (h0 - 0.01).abs() < 1e-12,
        "theta[3] is {h0}, not TVH0 = 0.01 — the `[parameters]` order in this fixture changed"
    );
    let t_event = match &subject.obs_records[0] {
        ObsRecord::Event { time, .. } => *time,
        _ => panic!("fixture must carry an exact event record"),
    };
    // Drug-free exact-event NLL is `H(t) − ln h(t)`, which at `BETA·C = 0` is
    // `H0·t − ln H0`.
    let drug_free = h0 * t_event - h0.ln();
    assert!(
        (arm_b - drug_free).abs() > 1e-2,
        "TTE term {arm_b} sits at its drug-free value {drug_free} — the infusion is not \
         reaching the hazard, so the arms would agree vacuously"
    );
    assert!(
        (arm_a - arm_b).abs() <= 1e-4 * arm_b.abs().max(1.0),
        "joint PK-TTE hazard arms disagree: shared solve {arm_a} vs dedicated {arm_b} \
         (#1187 — the dedicated arm resolves infusions through `gated_infusions`)"
    );
}

/// #1187 external anchor: joint PK-TTE cumulative hazard `A(3)` under a `RATE>0` dose
/// into an absorption compartment vs NONMEM 7.6.0 (`nonmem_anchor/pktte_inf.{ctl,csv}`,
/// `ADVAN13 TOL=9`, all fixed, η = 0). CHZ rather than OFV, which carries tool-specific
/// constants.
///
/// Measured: before the fix the dedicated arm read `H(1.86) = 3.796844e0` vs NONMEM
/// `1.2217e-1` (31×); the shared arm was already correct.
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_cumulative_hazard_matches_nonmem_for_infusion_into_absorption() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, EventType, ObsRecord};

    // Mirrors `nonmem_anchor/pktte_inf.ctl`: CL=1, V=10, KA=1, H0=0.02, BETA=0.5.
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-10
  ode_abstol = 1e-10
";
    let model = parse_model_string(src).expect("joint PK-TTE model must parse");

    // The NONMEM dataset, row for row: AMT=100 RATE=50 into CMT 1 (T = 2 h), PK
    // observations at 0.5…18, and the exact event at t = 1.86.
    let mut subject = make_simple_subject();
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 50.0, false, 0.0)];
    assert!(
        subject.doses[0].is_infusion(),
        "fixture must carry a real infusion or it tests nothing"
    );
    subject.obs_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 18.0];
    subject.observations = vec![4.6271, 6.2626, 7.1063, 8.0446, 6.3602, 4.8718, 3.6207, 2.0972];
    subject.obs_cmts = vec![1; 8];
    subject.cens = vec![0; 8];
    subject.occasions = vec![1; 8];
    subject.obs_records = vec![ObsRecord::Event {
        time: 1.86,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 2,
    }];

    let theta = &model.default_params.theta;
    let eta = [0.0_f64]; // $OMEGA 0 FIX on the NONMEM side

    let chz_state = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: HazardSpec::OdeAccumulated { chz_state }, ..
        }) => *chz_state,
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };

    // NONMEM 7.6.0, `nonmem_anchor/results/pktte_inf.tab`: CHZ at TIME = 1.86, printed at
    // 9 significant figures (`FORMAT=s1PE15.8`): the default 5 would add 4.1e-5 relative
    // rounding of its own, a floor under any bound here.
    const NONMEM_CHZ_AT_EVENT: f64 = 1.22168018e-1;

    // Arm B — the two-solve fallback, i.e. the arm that carried the defect.
    let (cum_dedicated, _haz) =
        crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, theta, &eta, &[1.86]);
    // Arm A — the #570 shared single solve.
    let share = try_joint_pktte_shared_solve(&model, &subject, theta, &eta)
        .expect("joint ODE PK-TTE model must qualify for the #570 shared solve");
    assert_eq!(share.times, vec![1.86], "share must cover the event time");
    let cum_shared = share.chz_states[0][chz_state];

    // Per-arm bounds, measured: dedicated 1.41e-9, shared 3.26e-6 (different timelines,
    // different local error). One loose bound would hide a regression in the tight arm.
    for (arm, got, tol) in [
        ("dedicated (ode_cumhaz_hazard)", cum_dedicated[0], 1e-8),
        ("shared (#570)", cum_shared, 1e-5),
    ] {
        let rel = (got - NONMEM_CHZ_AT_EVENT).abs() / NONMEM_CHZ_AT_EVENT;
        assert!(
            rel < tol,
            "H(1.86) on the {arm} arm: ferx {got:.9e} vs NONMEM \
             {NONMEM_CHZ_AT_EVENT:.9e} (rel {rel:.2e}, bound {tol:.0e})"
        );
    }
}

/// #570 guard: a joint PK-TTE model whose `[individual_parameters]` reads `TIME` must
/// not take the share; its single t=0 snapshot would be ~30–50% off.
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_share_rejects_time_builtin_model() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    // Same joint PK-TTE shape as `joint_pktte_ode_hazard_nll_paths_finite`, but CL
    // switches on TIME, so the prediction path is genuinely time-dependent.
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  STEP = if (TIME > 3) 1 else 0
  CL   = TVCL * (1 + STEP) * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(src).expect("TIME-using joint PK-TTE model must parse");
    // Sanity: the fixture really does trip the TIME built-in detector.
    assert!(
        crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
        "fixture must use the TIME built-in"
    );
    let mut subject = make_simple_subject();
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];
    let p = &model.default_params;

    // The guard must reject the share → established two-solve fallback.
    assert!(
        try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_none(),
        "a TIME-using joint PK-TTE model must not take the #570 shared-solve path"
    );
    // The fallback still yields a finite NLL (and η still moves it).
    let nll = individual_nll(&model, &subject, &p.theta, &[0.0], &p.omega, &p.sigma.values);
    let moved = individual_nll(&model, &subject, &p.theta, &[0.3], &p.omega, &p.sigma.values);
    assert!(
        nll.is_finite() && moved.is_finite() && (nll - moved).abs() > 1e-9,
        "fallback joint NLL must be finite and η-sensitive (nll={nll}, moved={moved})"
    );
}

/// The same rejection for model time read in the **`[odes]` RHS**: only the wide
/// `pk::model_uses_time_anywhere` fires here (the sibling passes under the narrow
/// predicate too). Since #1124 such a model routes to the event-driven walker, so the
/// share would score a different Gaussian term than the reported IPRED.
#[test]
#[cfg(feature = "survival")]
fn joint_pktte_share_rejects_model_time_in_the_odes_rhs() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    // Time dependence only in the PK RHS; the hazard is time-independent so the RHS is the
    // only reason to reject. Since #1166 a `TIME`-reading hazard does not trip the
    // predicate (pinned by `joint_pktte_share_admits_a_time_dependent_hazard`).
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central * (1.0 + 0.3*TAD)
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
";
    let model = parse_model_string(src).expect("TAD-in-RHS joint PK-TTE model must parse");

    // The discriminating preconditions. Without these the test would pass for
    // a model on which the two predicates agree, and would not pin the gate's
    // choice of predicate at all.
    assert!(
        !crate::parser::model_parser::compiled_model_uses_time_builtin(&model),
        "the narrow predicate must NOT fire — the individual parameters are \
         time-independent"
    );
    assert!(
        crate::pk::model_uses_time_anywhere(&model),
        "…while the wide one must, via the `[odes]` RHS"
    );

    let mut subject = make_simple_subject();
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];
    let p = &model.default_params;

    assert!(
        try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_none(),
        "a joint PK-TTE model whose `[odes]` RHS reads model time must not take \
         the #570 shared-solve path"
    );
    // …and the two-solve fallback still produces a finite, η-sensitive NLL.
    let nll = individual_nll(&model, &subject, &p.theta, &[0.0], &p.omega, &p.sigma.values);
    let moved = individual_nll(&model, &subject, &p.theta, &[0.3], &p.omega, &p.sigma.values);
    assert!(
        nll.is_finite() && moved.is_finite() && (nll - moved).abs() > 1e-9,
        "fallback joint NLL must be finite and η-sensitive (nll={nll}, moved={moved})"
    );
}

/// A joint PK-TTE model, `{HAZ}` substituted. Autonomous PK block throughout —
/// the hazard is the only thing that ever reads time here.
#[cfg(feature = "survival")]
fn joint_time_hazard_src(haz: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  theta TVH0(0.01, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA * depot
  d/dt(central) =  KA * depot - (CL/V) * central
[event_model]
  cmt    = 2
  hazard = {haz}
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
"
    )
}

/// #1166: a Gompertz-shaped (`TIME`-reading) hazard on an autonomous PK block **takes**
/// the share; with the sibling, the gate is pinned from both sides.
#[test]
#[cfg(feature = "survival")]
fn joint_pktte_share_admits_a_time_dependent_hazard() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    let model = parse_model_string(&joint_time_hazard_src(
        "H0 * exp(0.05*TIME) * exp(BETA * (central / V))",
    ))
    .expect("Gompertz-in-TIME joint PK-TTE model must parse");

    // The discriminating precondition: the *wide* flag still fires — the
    // augmented program does read model time, and the SS gates must keep
    // seeing that — while the predicate the share asks does not.
    let prog = model.ode_spec.as_ref().and_then(|o| o.rhs_program.as_ref()).expect("rhs program");
    assert!(
        prog.reads_model_time(),
        "the augmented RHS does read model time (the SS gates' predicate)"
    );
    assert!(
        !crate::pk::model_uses_time_anywhere(&model),
        "…but the PK block does not, so the routing predicate must be false"
    );

    let mut subject = make_simple_subject();
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];
    let p = &model.default_params;
    assert!(
        try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_some(),
        "a time-dependent hazard on an autonomous PK block must take the \
         #570 shared-solve path"
    );
}

/// An `EVID=3/4` reset **declines** the share (a reset needs the event-driven walker) —
/// the premise of `tests/pktte_prestart_entry.rs` (A1 via share, A4 via dedicated).
/// Measured: disabling `|| subject.has_resets()` fails only this test.
#[cfg(feature = "survival")]
#[test]
fn joint_pktte_share_rejects_a_subject_with_resets() {
    use crate::types::{EventType, ObsRecord};

    let model = joint_pktte_ode_model();
    let p = &model.default_params;
    let records = vec![ObsRecord::Event {
        time: 20.0,
        event_type: EventType::Exact,
        entry_time: 5.0,
        cmt: 2,
    }];

    // Without the reset the subject qualifies — the straddle that makes the next
    // assertion about the reset rather than about something else in the admission list.
    let subject = late_start_joint_subject(records.clone());
    assert!(!subject.has_resets(), "control subject must carry no resets");
    assert!(
        try_joint_pktte_shared_solve(&model, &subject, &p.theta, &[0.0]).is_some(),
        "the reset-free subject must qualify for the share, or this test proves nothing \
         about resets"
    );

    // With one, it must not.
    let mut reset_subject = late_start_joint_subject(records);
    reset_subject.reset_times = vec![15.0];
    assert!(reset_subject.has_resets(), "fixture must actually carry a reset");
    assert!(
        try_joint_pktte_shared_solve(&model, &reset_subject, &p.theta, &[0.0]).is_none(),
        "an EVID=3/4 reset must decline the #570 share — the share takes a single t=0 PK \
         snapshot, and a reset needs the event-driven walker"
    );
}

/// #1166 routing oracle: two models computing the **same arithmetic**
/// (`* (1.0 + 0.0*TIME)` is an IEEE identity, not folded by the bytecode) must land on
/// the same engine and agree bit-for-bit. Before #1166 the flagged twin took the
/// event-driven walk (0 of 40 subjects bit-identical, worst 3.25e-4).
#[test]
#[cfg(feature = "survival")]
fn identical_arithmetic_takes_the_identical_engine_bit_for_bit() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{EventType, ObsRecord};

    // The pair must straddle the old predicate (found by mutation): the base hazard is
    // autonomous, only the twin's identity factor reads time.
    let plain = parse_model_string(&joint_time_hazard_src("H0 * exp(BETA * (central / V))"))
        .expect("parse");
    let twin = parse_model_string(&joint_time_hazard_src(
        "H0 * exp(BETA * (central / V)) * (1.0 + 0.0*TIME)",
    ))
    .expect("parse");
    // Pin the straddle itself, so a later predicate change that stopped the
    // twin from being *flagged at all* would show up here rather than turning
    // this back into a tautology.
    let flagged = |m: &crate::types::CompiledModel| {
        m.ode_spec
            .as_ref()
            .and_then(|o| o.rhs_program.as_ref())
            .expect("rhs program")
            .reads_model_time()
    };
    assert!(
        !flagged(&plain) && flagged(&twin),
        "the twin must trip the wide flag and the plain model must not — that \
         difference is what the fix has to make invisible"
    );

    let mut subject = make_simple_subject();
    subject.obs_records =
        vec![ObsRecord::Event { time: 5.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];
    let p = &plain.default_params;
    let q = &twin.default_params;

    for eta in [0.0, 0.3, -0.25] {
        let a = individual_nll(&plain, &subject, &p.theta, &[eta], &p.omega, &p.sigma.values);
        let b = individual_nll(&twin, &subject, &q.theta, &[eta], &q.omega, &q.sigma.values);
        assert!(a.is_finite(), "the fixture must produce a real objective (eta={eta}, nll={a})");
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "an arithmetically identical hazard must take the same engine \
             (eta={eta}, plain={a}, identity-twin={b}, delta={})",
            a - b
        );
    }
}

/// #1218 on the fit path: a subject whose only TTE record sits on its first dose builds
/// `times = [0.0]` in `tte_ode_nll`'s dedicated solve and read `NaN` (→ `1e20`). It now
/// scores `H(0) − ln h(0)` with the post-dose hazard. An exact event, since a censor
/// scores `H(0) = 0` under the wrong (seeded-state) fix too. Measured: `1e20` before,
/// `−ln 0.02 = 3.91` under the seeded-state variant vs `−ln 25.6` post-dose.
#[cfg(feature = "survival")]
#[test]
fn tte_ode_nll_scores_an_event_on_the_first_dose_instead_of_the_sentinel() {
    use crate::parser::model_parser::parse_model_string;
    use crate::types::{DoseEvent, EventType, ObsRecord};

    // #1210's drug arm, IV: CL=1, V=10, H0=0.02, BETA=0.5, hazard on central/V.
    let src = r"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVH0(0.02, 1e-5, 10.0)
  theta TVBETA(0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.1 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  H0   = TVH0
  BETA = TVBETA
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[event_model]
  cmt    = 2
  hazard = H0 * exp(BETA * (central / V))
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  method = focei
  ode_reltol = 1e-9
  ode_abstol = 1e-11
";
    let model = parse_model_string(src).expect("joint PK-TTE model must parse");

    // An `SS=1` bolus at t = 0 (so drug is present on the incoming side of the record)
    // and nothing else: no PK rows, one exact event on the dose instant.
    let mut subject = make_simple_subject();
    subject.doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0)];
    subject.obs_times = Vec::new();
    subject.observations = Vec::new();
    subject.obs_cmts = Vec::new();
    subject.cens = Vec::new();
    subject.occasions = Vec::new();
    subject.obs_records =
        vec![ObsRecord::Event { time: 0.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 }];

    let theta = &model.default_params.theta;
    let eta = [0.0_f64];
    let chz_state = match model.endpoints.get(&2) {
        Some(EndpointLikelihood::Tte {
            hazard: HazardSpec::OdeAccumulated { chz_state }, ..
        }) => *chz_state,
        _ => panic!("expected an OdeAccumulated TTE endpoint on CMT 2"),
    };

    let nll = tte_ode_nll(&model, &subject, chz_state, &subject.obs_records, theta, &eta);
    assert!(
        nll.is_finite() && nll < SUBJECT_SENTINEL_NLL,
        "an event on the first dose is scored with the sentinel, not a likelihood: {nll}"
    );

    // The reference: the same H/h read off a grid that also asks for a later point.
    let (cum, haz) =
        crate::survival::ode_cumhaz_hazard(&model, &subject, chz_state, theta, &eta, &[0.0, 1.0]);
    assert!(
        cum[0].is_finite() && haz[0].is_finite(),
        "the multi-point reference row is not finite: {}, {}",
        cum[0],
        haz[0]
    );
    // The straddle: the post-dose hazard must be far from the drug-free `H0 = 0.02`, or
    // this test could not tell the post-dose row from a seeded one.
    assert!(
        haz[0] > 5.0 * 0.02,
        "h(0) = {} is not distinguishable from the drug-free hazard",
        haz[0]
    );
    let want = cum[0] - haz[0].ln();
    assert!(
        (nll - want).abs() < 1e-12,
        "tte_ode_nll = {nll} but H(0) - ln h(0) from the multi-point grid = {want}"
    );
}
/// #2267: the joint (TTE/categorical/CTMM) marginal records the curvature it factors
/// (with the non-Gaussian η-Hessian) and names it when that breaks. Healthy, the parts
/// sum to the contribution under FOCEI and FOCE (measured gap `0`, bound `1e-12·nll`).
#[cfg(feature = "survival")]
#[test]
fn the_joint_marginal_names_a_curvature_the_non_gaussian_hessian_broke() {
    let subject = make_simple_subject();
    let ipreds: Vec<f64> = subject.observations.iter().map(|y| 1.05 * y).collect();
    let h = DMatrix::from_fn(6, 1, |j, _| -0.1 * ipreds[j] * subject.obs_times[j]);
    let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
    let eta = DVector::from_vec(vec![0.1]);
    let tte_nll = 2.0;
    let run = |h: &DMatrix<f64>, tte_h: f64, interaction: bool| {
        let mut parts = Vec::new();
        let nll = foce_subject_nll_interaction_with_tte(
            &subject,
            &ipreds,
            &eta,
            h,
            &omega,
            &[0.1],
            &ErrorSpec::Single(ErrorModel::Proportional),
            BloqMethod::Drop,
            &[],
            tte_nll,
            DMatrix::from_element(1, 1, tte_h),
            interaction,
            None,
            None,
            Some(&mut parts),
        );
        (nll, parts)
    };
    let names =
        |parts: &[(String, f64)]| -> Vec<String> { parts.iter().map(|(n, _)| n.clone()).collect() };

    for interaction in [true, false] {
        let (nll, parts) = run(&h, 3.0, interaction);
        assert_eq!(
            names(&parts),
            ["Gaussian residual term", "random-effect prior", "curvature log-determinant"],
            "interaction = {interaction}"
        );
        let summed = parts.iter().map(|(_, v)| v).sum::<f64>() + tte_nll;
        eprintln!("interaction = {interaction}: gap {:e}", (summed - nll).abs());
        assert!(
            nll.is_finite() && nll < 1e3 && (summed - nll).abs() <= 1e-12 * nll.abs(),
            "interaction = {interaction}: Σ parts + tte = {summed}, marginal = {nll}"
        );
    }

    // The straddle: each case is a sentinel of the same marginal, named for its cause.
    let broken = [
        (
            h.clone(),
            -1e6,
            "curvature (H̃ not positive definite once the non-Gaussian Hessian is added)",
        ),
        (
            h.clone(),
            f64::NAN,
            "non-Gaussian curvature (η-Hessian of the time-to-event, categorical or \
             CTMM terms not finite)",
        ),
        (DMatrix::from_element(6, 1, f64::NAN), 3.0, CURVATURE_FAILED),
    ];
    for (h, tte_h, want) in broken {
        let (nll, parts) = run(&h, tte_h, true);
        assert_eq!(nll, SUBJECT_SENTINEL_NLL, "{want}");
        let last = parts.last().expect("a curvature part");
        assert!(last.0 == want && last.1.is_nan(), "{parts:?}");
    }
}

// ── #901: per-record TTE terms on each ODE arm of the router (T6 / T7) ──

/// The two-record joint PK-TTE subject of T6 / T7 (dose at t = 10): `entry` 5 is outside
/// the dual walk's reach, 12 inside; `ss` leaves the walk's scope; `reset` declines the share.
#[cfg(feature = "survival")]
fn record_terms_subject(reset: bool, ss: bool, entry: f64) -> Subject {
    let mut subject = late_start_joint_subject(vec![
        ObsRecord::Event {
            time: 20.0,
            event_type: crate::types::EventType::Exact,
            entry_time: entry,
            cmt: 2,
        },
        ObsRecord::Event {
            time: 30.0,
            event_type: crate::types::EventType::RightCensored,
            entry_time: entry,
            cmt: 2,
        },
    ]);
    if ss {
        subject.doses = vec![DoseEvent::new(10.0, 100.0, 1, 0.0, true, 12.0)];
    }
    if reset {
        subject.reset_times = vec![15.0];
    }
    subject
}

/// The fixture's `chz_state` and its single TTE endpoint's `(hazard, recurrence)`.
#[cfg(feature = "survival")]
fn ode_tte_endpoint(model: &CompiledModel) -> (usize, &HazardSpec, crate::types::TteRecurrence) {
    let EndpointLikelihood::Tte { hazard, recurrence, .. } = &model.endpoints[&2] else {
        panic!("CMT 2 is the TTE endpoint");
    };
    let HazardSpec::OdeAccumulated { chz_state } = hazard else {
        panic!("the joint fixture's hazard is ODE-accumulated");
    };
    (*chz_state, hazard, *recurrence)
}

/// Fold `terms` the way the survival loops fold them: `dh − log h` per point.
#[cfg(feature = "survival")]
fn fold_terms(terms: &[crate::survival::RecordTerm<f64>]) -> f64 {
    let mut nll = 0.0_f64;
    for t in terms {
        nll = match *t {
            crate::survival::RecordTerm::Point { dh, log_h, .. } => match log_h {
                Some(l) => nll + (dh - l),
                None => nll + dh,
            },
            crate::survival::RecordTerm::Interval { log_prob, .. } => nll - log_prob,
        };
    }
    nll
}

/// The router's terms for `s`, given `share`, with the subject-level entry point
/// (`tte_diagnostics`, which builds its own share) checked to report the same `ΔH`.
#[cfg(feature = "survival")]
fn router_terms(
    model: &CompiledModel, s: &Subject, share: Option<&JointPkTteSolve>, eta: &[f64],
) -> Vec<crate::survival::RecordTerm<f64>> {
    let (_, hazard, recurrence) = ode_tte_endpoint(model);
    let p = &model.default_params;
    let terms = tte_endpoint_record_terms(
        model,
        s,
        hazard,
        recurrence,
        &s.obs_records,
        &p.theta,
        eta,
        share,
    )
    .expect("finite NLL ⟹ terms");
    assert_eq!(terms.len(), 2);
    let rows = tte_diagnostics(model, s, &p.theta, eta);
    let crate::survival::RecordTerm::Point { dh, .. } = terms[1] else {
        panic!("the censor is a point record");
    };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[1].coxsnell.to_bits(), dh.to_bits(), "tte_diagnostics arm");
    terms
}

/// T6a (#901), walk arm: an in-scope subject is scored off the dual walk (#2482) even
/// when a #570 share exists; terms fold **bit-identically** to `survival::ode_tte_nll_g`
/// and differ from the share's NLL. (The walk reproduces the dedicated solve bit for bit
/// here — measured rel 0 at η ∈ {0.2, −0.4, 0, 0.7}.)
#[cfg(feature = "survival")]
#[test]
fn tte_record_terms_fold_to_the_walk_nll() {
    let model = joint_pktte_ode_model();
    let p = &model.default_params;
    let (chz_state, _, _) = ode_tte_endpoint(&model);
    let eta = [0.2];
    let s = record_terms_subject(false, false, 12.0);
    assert!(
        crate::sens::ode_provider::adjoint::ode_hazard_walk_in_scope(&model, &s),
        "the plain subject must be in the walk's scope (straddle)"
    );
    let share = try_joint_pktte_shared_solve(&model, &s, &p.theta, &eta)
        .expect("…and admitted to the share, so the walk's precedence is what is tested");
    let walk = crate::survival::ode_tte_nll_g::<f64>(
        &model,
        &s,
        chz_state,
        &s.obs_records,
        &p.theta,
        &eta,
    )
    .expect("walk value");
    let terms = router_terms(&model, &s, Some(&share), &eta);
    assert_eq!(fold_terms(&terms).to_bits(), walk.to_bits());
    let shared = tte_ode_nll_from_shared(
        model.ode_spec.as_ref().unwrap(),
        &share,
        chz_state,
        &s.obs_records,
    );
    assert_ne!(
        shared.to_bits(),
        walk.to_bits(),
        "the fixture must tell the walk from the share, or the bit test is vacuous"
    );
}

/// T6b (#901), shared arm: without a compiled `[odes]` program the subject leaves the
/// walk's scope while the share admits it; terms fold bit-identically to
/// `tte_ode_nll_from_shared`.
#[cfg(feature = "survival")]
#[test]
fn tte_record_terms_fold_to_the_shared_solve_nll() {
    let model = walk_declined_joint_pktte_model();
    let p = &model.default_params;
    let (chz_state, _, _) = ode_tte_endpoint(&model);
    let eta = [0.2];
    let s = record_terms_subject(false, true, 5.0);
    assert_eq!(
        walk_scope_straddle(&s),
        (true, false),
        "only the missing program may take the subject out of the walks' scope (straddle)"
    );
    let share = try_joint_pktte_shared_solve(&model, &s, &p.theta, &eta)
        .expect("the reset-free subject must take the #570 share (straddle)");
    let nll = tte_ode_nll_from_shared(
        model.ode_spec.as_ref().unwrap(),
        &share,
        chz_state,
        &s.obs_records,
    );
    let terms = router_terms(&model, &s, Some(&share), &eta);
    assert_eq!(fold_terms(&terms).to_bits(), nll.to_bits());
    let dedicated = tte_ode_nll(&model, &s, chz_state, &s.obs_records, &p.theta, &eta);
    assert_ne!(
        dedicated.to_bits(),
        nll.to_bits(),
        "the fixture must tell the two engines apart, or the bit test above is vacuous"
    );
}

/// #901: on the `1e20` rail `tte_endpoint_record_terms` returns `None`, not the partial
/// terms visited before the failing record (an RTTE interval record); the control
/// without it returns its one term.
#[cfg(feature = "survival")]
#[test]
fn tte_record_terms_are_none_on_the_sentinel_rail() {
    use crate::types::{EventType, HazardFamily, RtteClock, TteRecurrence};
    let model = joint_pktte_ode_model(); // only read on the ODE arms
    let subject = make_simple_subject();
    let hazard = HazardSpec::Analytic {
        family: HazardFamily::Exponential,
        param_fn: Box::new(|_: &[f64], _: &[f64], _: &HashMap<String, f64>| vec![0.1]),
    };
    let forward = TteRecurrence::Repeated { clock: RtteClock::Forward };
    let event =
        ObsRecord::Event { time: 4.0, event_type: EventType::Exact, entry_time: 0.0, cmt: 2 };
    let interval = ObsRecord::Event {
        time: 8.0,
        event_type: EventType::IntervalCensored { left: 6.0, right: 8.0 },
        entry_time: 0.0,
        cmt: 2,
    };
    let terms = |records: &[ObsRecord]| {
        tte_endpoint_record_terms(&model, &subject, &hazard, forward, records, &[], &[0.0], None)
    };
    assert_eq!(terms(std::slice::from_ref(&event)).map(|t| t.len()), Some(1));
    assert!(terms(&[event, interval]).is_none());
}

/// #901, competing risks: one row per cause's record, in ascending CMT (records given
/// descending), each from its cause-specific `H_k`, never the total hazard.
#[cfg(feature = "survival")]
#[test]
fn tte_diagnostics_give_one_cause_specific_row_per_cause_record() {
    use crate::types::EventType;
    let model = crate::parser::model_parser::parse_model_string(
        "
[parameters]
  theta TVLAMBDA_A(0.10, 0.001, 10.0)
  theta TVLAMBDA_B(0.06, 0.001, 10.0)
  omega ETA_F ~ 0.25
[event_model cause_a]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA_A * exp(ETA_F)
[event_model cause_b]
  cmt    = 3
  family = exponential
  scale  = TVLAMBDA_B * exp(ETA_F)
",
    )
    .expect("competing-risks model must parse");
    let mut subject = make_simple_subject();
    subject.obs_records = [(3, EventType::RightCensored), (2, EventType::Exact)]
        .into_iter()
        .map(|(cmt, event_type)| ObsRecord::Event { time: 4.0, event_type, entry_time: 0.0, cmt })
        .collect();
    let eta = [0.3_f64];
    let rows = tte_diagnostics(&model, &subject, &model.default_params.theta, &eta);
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].cmt, rows[0].dv), (2, 1.0));
    assert_eq!((rows[1].cmt, rows[1].dv), (3, 0.0));
    let e = eta[0].exp();
    for (row, lambda) in rows.iter().zip([0.10, 0.06]) {
        let want = lambda * e * 4.0;
        assert!(
            (row.coxsnell - want).abs() < 1e-14,
            "CMT {}: {} vs cause-specific {want}",
            row.cmt,
            row.coxsnell
        );
        assert!((row.martingale - (row.dv - want)).abs() < 1e-14);
    }
}

/// #901: a TTE row's `TIME` is the time **as written** (`Subject::param_time_at`) in a
/// reset period: written 4 at engine 104 must report 4, not 104.
#[cfg(feature = "survival")]
#[test]
fn tte_diagnostics_report_the_written_time_in_a_restarted_period() {
    use crate::types::EventType;
    let model = crate::parser::model_parser::parse_model_string(
        "
[parameters]
  theta TVLAMBDA(0.10, 0.001, 10.0)
  omega ETA ~ 0.25
[event_model]
  cmt    = 2
  family = exponential
  scale  = TVLAMBDA * exp(ETA)
",
    )
    .expect("TTE model must parse");
    let mut subject = make_simple_subject();
    subject.reset_period_times = vec![(100.0, 0.0)];
    subject.obs_records = vec![ObsRecord::Event {
        time: 104.0,
        event_type: EventType::Exact,
        entry_time: 0.0,
        cmt: 2,
    }];
    assert_eq!(subject.param_time_at(104.0), 4.0, "fixture: written time is 4");
    let eta = [0.0_f64];
    let rows = tte_diagnostics(&model, &subject, &model.default_params.theta, &eta);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].time, 4.0, "TIME must be as written, not the engine 104");
    // …while ΔH stays on the engine clock the likelihood scored.
    assert!((rows[0].coxsnell - 0.10 * 104.0).abs() < 1e-12);
}

/// T7 (#901), dedicated arm: no `[odes]` program and an EVID-3 reset (both declines
/// asserted), terms fold bit-identically to `tte_ode_nll`.
#[cfg(feature = "survival")]
#[test]
fn tte_record_terms_fold_to_the_dedicated_solve_nll() {
    let model = walk_declined_joint_pktte_model();
    let p = &model.default_params;
    let (chz_state, _, _) = ode_tte_endpoint(&model);
    let eta = [0.2];
    let r = record_terms_subject(true, true, 5.0);
    assert_eq!(
        walk_scope_straddle(&r),
        (true, false),
        "only the missing program may take the subject out of the walks' scope (straddle)"
    );
    assert!(
        try_joint_pktte_shared_solve(&model, &r, &p.theta, &eta).is_none(),
        "the reset subject must decline the share (straddle)"
    );
    let nll = tte_ode_nll(&model, &r, chz_state, &r.obs_records, &p.theta, &eta);
    let terms = router_terms(&model, &r, None, &eta);
    assert_eq!(fold_terms(&terms).to_bits(), nll.to_bits());
}
