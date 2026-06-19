//! Slow regression test for issue #267: SAEM must not collapse the additive
//! component of a combined residual-error model.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{DoseEvent, OmegaMatrix, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};

mod common;

const MODEL: &str = r#"
[parameters]
  theta TVCL(3.0, 0.1, 20.0)
  theta TVV(30.0, 1.0, 200.0)
  omega ETA_CL ~ 0.08
  omega ETA_V  ~ 0.08
  sigma PROP_ERR ~ 0.05 (sd)
  sigma ADD_ERR  ~ 2.00 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

fn template_population(n: usize) -> Population {
    let times = [0.5_f64, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0];
    let subjects = (1..=n)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                times.to_vec(),
                vec![0.0; times.len()],
                vec![1; times.len()],
            )
        })
        .collect();

    Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn simulated_population(model: &ferx_core::types::CompiledModel) -> Population {
    let template = template_population(32);
    let mut truth = model.default_params.clone();
    truth.theta = vec![3.0, 30.0];
    truth.omega = OmegaMatrix::from_diagonal(&[0.08, 0.08], vec!["ETA_CL".into(), "ETA_V".into()]);
    truth.sigma.values = vec![0.05, 3.00];

    let sim = simulate_with_seed(model, &template, &truth, 1, 20260619);
    let mut pop = template;
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|row| row.id == subj.id)
            .map(|row| row.outcome.continuous_value())
            .collect();
    }
    pop
}

fn fit_with(
    method: EstimationMethod,
    model: &ferx_core::types::CompiledModel,
    pop: &Population,
) -> f64 {
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.run_covariance_step = false;
    opts.verbose = false;
    opts.outer_maxiter = 300;
    opts.saem_n_exploration = 80;
    opts.saem_n_convergence = 80;
    opts.saem_seed = Some(267);

    let result = fit(model, pop, &model.default_params, &opts).expect("fit must succeed");
    result.sigma[1]
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_combined_error_additive_sigma_matches_focei() {
    let model = parse_model_string(MODEL).expect("combined model parses");
    let pop = simulated_population(&model);

    let focei_add = fit_with(EstimationMethod::FoceI, &model, &pop);
    let saem_add = fit_with(EstimationMethod::Saem, &model, &pop);

    assert!(
        focei_add > 0.75,
        "fixture must identify a non-trivial additive sigma; FOCEI ADD={focei_add}"
    );
    let rel = (saem_add - focei_add).abs() / focei_add;
    assert!(
        rel < 0.35,
        "SAEM ADD should stay close to FOCEI, not collapse: SAEM={saem_add}, FOCEI={focei_add}, rel={rel:.3}"
    );
}
