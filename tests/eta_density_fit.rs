//! Non-Gaussian random-effect distributions (#1716) through the public `fit()` /
//! `simulate()` entry points.
//!
//! Tier 2: every estimator that scores a density runs a few iterations on one and returns
//! a finite objective, and the ones that do not yet refuse with `E_DENSITY_UNSUPPORTED`.
//! Tier 3 (`slow-tests`): simulate a population with a Gamma clearance with ferx's own
//! sampler, refit it, and recover the generating parameters, which is the only check that
//! the sampler and the likelihood describe the same distribution.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{EstimationMethod, FitOptions, Population, SimOutcome};

/// The warfarin model with a Gamma clearance (mean `TVCL`, variance ω) and a t volume.
const DENSITY_WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta NU(10.0, 2.5, 100.0)
  omega ETA_CL ~ 0.09 density gamma
  omega ETA_V  ~ 0.04 density t(NU)
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * ETA_CL
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn model_and_data() -> (ferx_core::types::CompiledModel, Population) {
    let parsed = parse_full_model(DENSITY_WARFARIN).expect("density model parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("warfarin data reads");
    (parsed.model, pop)
}

fn short(method: EstimationMethod, interaction: bool) -> FitOptions {
    FitOptions {
        method,
        interaction,
        outer_maxiter: 3,
        saem_n_exploration: 5,
        saem_n_convergence: 5,
        imp_iterations: 2,
        run_covariance_step: false,
        ..FitOptions::default()
    }
}

/// Each estimator that scores a density returns a finite objective after a few
/// iterations, and SAEM's density M-step moves the shape θ ν off its initial value
/// (it is owned by that step alone: the data M-step pins it).
#[test]
fn every_density_estimator_runs() {
    let (model, pop) = model_and_data();
    for (method, interaction) in [
        (EstimationMethod::FoceI, true),
        (EstimationMethod::Foce, false),
        (EstimationMethod::Laplace, true),
        (EstimationMethod::Saem, true),
        (EstimationMethod::Imp, true),
    ] {
        let r = ferx_core::fit(
            &model,
            &pop,
            &model.default_params,
            &short(method, interaction),
        )
        .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(
            r.ofv.is_finite() && r.ofv.abs() < 1e10,
            "{method:?}: OFV {}",
            r.ofv
        );
        if method == EstimationMethod::Saem {
            assert_ne!(r.theta[3], model.default_params.theta[3], "SAEM moves ν");
        }
    }
}

/// The estimators that do not score a density yet refuse it by code.
#[test]
fn bayes_and_vi_refuse_a_density() {
    let (model, pop) = model_and_data();
    for method in [EstimationMethod::Bayes, EstimationMethod::Vi] {
        let e = ferx_core::fit(&model, &pop, &model.default_params, &short(method, true))
            .expect_err("refused");
        let code = e.diagnostic().map(|d| d.code.clone());
        assert_eq!(
            code.as_deref(),
            Some("E_DENSITY_UNSUPPORTED"),
            "{method:?}: {e}"
        );
    }
}

/// Simulate 1000 subjects from known parameters with a Gamma clearance (ω = 0.16, so a
/// CV of 40% on a right-skewed CL) and refit with FOCEI: the generating parameters come
/// back. Every fit-side test scores the density on data it did not draw, so only this
/// round trip sees a sampler that draws another distribution than the one scored.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_simulated_gamma_clearance_is_recovered() {
    let (model, template) = model_and_data();
    let mut truth = model.default_params.clone();
    truth.theta = vec![0.15, 9.0, 1.2, 8.0];
    truth.omega = ferx_core::types::OmegaMatrix::from_matrix(
        nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.16, 0.03, 0.2])),
        truth.omega.eta_names.clone(),
        true,
    );
    // 100 replicates of warfarin's 10 subjects: 1000 subjects, so ω_CL's sampling SE is
    // about √(2/1000) = 4.5%. At 210 subjects it was 9.8%, too wide to see a sampler that
    // draws another distribution.
    let n_rep = 100;
    let sims = ferx_core::simulate_with_seed(&model, &template, &truth, n_rep, 1716).unwrap();
    let mut pop = template.clone();
    pop.subjects.clear();
    for k in 0..n_rep {
        let mut rows = sims.iter().filter(|r| r.sim == k + 1);
        for subj in &template.subjects {
            let mut s = subj.clone();
            s.id = format!("{}_{}", subj.id, k + 1);
            for y in s.observations.iter_mut() {
                match rows.next().expect("one row per observation").outcome {
                    SimOutcome::Continuous { value } => *y = value,
                    ref o => panic!("expected a continuous outcome, got {o:?}"),
                }
            }
            pop.subjects.push(s);
        }
    }
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let r = ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("FOCEI fit");
    let rel = |got: f64, want: f64| (got - want).abs() / want;
    let gaps = [
        ("TVCL", rel(r.theta[0], 0.15)),
        ("TVV", rel(r.theta[1], 9.0)),
        ("ω_CL", rel(r.omega[(0, 0)], 0.16)),
    ];
    eprintln!("recovery: {gaps:?}, ν = {}", r.theta[3]);
    // Measured (seed 1716): TVCL 0.9%, TVV 0.06%, ω_CL 4.0%, about one SE each; bounds
    // about 3 SE. ν (measured 12.6 against 8) is weakly identified by 1000 subjects and
    // is not asserted.
    let bounds = [0.03, 0.03, 0.15];
    for ((name, g), b) in gaps.into_iter().zip(bounds) {
        assert!(g.is_finite() && g < b, "{name}: {g}");
    }
}
