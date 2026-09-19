//! Tier-2 integration tests for the automatic SAEM block-proposal count (#1459).
//!
//! `FitOptions::saem_n_mh_steps` defaults to `0` — the `auto` sentinel — and is
//! resolved from the dataset by `estimation::saem::auto_n_mh_steps`. The
//! arithmetic itself is unit-tested next to that function; what these tests
//! guard is the **wiring**, which the unit tests cannot see: three separate
//! call sites read the option, and any one of them left reading it raw would
//! hand its kernel a literal zero-proposal count — a silent, green regression
//! (SAEM would still "converge", the Bayes η block would fall back to its
//! `.max(1)` single proposal).
//!
//! Each test is a differential pair:
//!
//! * the default (`auto`) must be **bit-identical** to the explicit count the
//!   rule resolves to on that fixture, and
//! * that count must be **distinguishable** from a different explicit count on
//!   the same fixture — otherwise the equality above holds for any wiring at
//!   all and the test cannot fail.
//!
//! The fixture is sparse on purpose: 2 observations per subject against 2 η is
//! `2/(1·2) = 1.0` observation per η, so `2.5 · 1.0 = 2.5` rounds to 3 and the
//! floor of 6 applies — the arm of the rule that a sparse PK dataset lands on.
//!
//! Data are fully synthetic (simulated from the model's own parameters).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{CompiledModel, DoseEvent, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};

mod common;

/// The count `auto` resolves to on [`sparse_population`]: `n_obs = 2·N`,
/// `n_subjects = N`, `n_eta = 2`, so `2.5 · 2/(N·2/N·2)`… concretely
/// `2.5 · (2N)/(N·2) = 2.5`, which rounds to 3 and is lifted to the floor, 6.
const RESOLVED_ON_SPARSE: usize = 6;

/// A count the fixture must be able to tell apart from [`RESOLVED_ON_SPARSE`],
/// so "auto == 6" is not satisfied by every possible wiring. This is the
/// pre-#1459 fixed default.
const OTHER_COUNT: usize = 20;

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0,  0.1, 100.0)
  theta TVV(50.0,  1.0, 500.0)
  omega ETA_CL ~ 0.10
  omega ETA_V  ~ 0.10
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// 2 observations per subject against 2 η — 1.0 observation per η.
fn sparse_population(model: &CompiledModel, n: usize) -> Population {
    let times = [1.0_f64, 8.0];
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
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };

    let sim = simulate_with_seed(model, &template, &model.default_params, 1, 20260919);
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        subj.observations = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value().max(1e-6))
            .collect();
    }
    pop
}

fn short_saem_opts(n_mh_steps: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 8,
        saem_n_convergence: 8,
        saem_omega_burnin: 4,
        saem_seed: Some(4242),
        saem_n_mh_steps: n_mh_steps,
        run_covariance_step: false,
        threads: Some(1),
        ..FitOptions::default()
    }
}

/// `(theta, omega diagonal, sigma)` of a fit, as the comparison key. A change
/// in the number of MH proposals moves every one of them.
fn fingerprint(res: &ferx_core::types::FitResult) -> Vec<f64> {
    let mut v = res.theta.clone();
    for i in 0..res.omega.nrows() {
        v.push(res.omega[(i, i)]);
    }
    v.extend(res.sigma.iter().copied());
    assert!(
        v.iter().all(|x| x.is_finite()),
        "non-finite estimate in the fingerprint: {v:?}"
    );
    v
}

#[test]
fn saem_default_resolves_to_the_rules_count() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = sparse_population(&model, 20);

    // The default must BE the sentinel; if it is ever changed to a literal
    // count this test would otherwise still pass while testing nothing.
    assert_eq!(
        FitOptions::default().saem_n_mh_steps,
        0,
        "the default is the `auto` sentinel — see FitOptions::saem_n_mh_steps"
    );

    let auto = fit(&model, &pop, &model.default_params, &short_saem_opts(0)).expect("auto fit Ok");
    let explicit = fit(
        &model,
        &pop,
        &model.default_params,
        &short_saem_opts(RESOLVED_ON_SPARSE),
    )
    .expect("explicit fit Ok");
    let other = fit(
        &model,
        &pop,
        &model.default_params,
        &short_saem_opts(OTHER_COUNT),
    )
    .expect("other fit Ok");

    assert_eq!(
        fingerprint(&auto),
        fingerprint(&explicit),
        "`auto` must run exactly {RESOLVED_ON_SPARSE} block proposals on this fixture"
    );
    assert_ne!(
        fingerprint(&auto),
        fingerprint(&other),
        "fixture cannot distinguish {RESOLVED_ON_SPARSE} proposals from {OTHER_COUNT} — the \
         equality above would then hold for any wiring"
    );
}

#[test]
fn conddist_pass_resolves_the_same_count() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = sparse_population(&model, 20);

    let with_conddist = |n_mh_steps: usize| {
        let mut opts = short_saem_opts(n_mh_steps);
        opts.saem_conddist = true;
        opts.saem_conddist_nsamp = 40;
        opts.saem_conddist_burnin = 10;
        let res = fit(&model, &pop, &model.default_params, &opts).expect("fit Ok");
        let cd = res
            .cond_dist
            .expect("cond_dist is populated when saem_conddist = true");
        // The conditional means are what the pass exists to produce, and they
        // are downstream of its MH kernels.
        let means: Vec<f64> = cd
            .cond_mean
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        assert!(
            means.iter().all(|x| x.is_finite()),
            "non-finite conditional mean: {means:?}"
        );
        means
    };

    assert_eq!(
        with_conddist(0),
        with_conddist(RESOLVED_ON_SPARSE),
        "the conditional-distribution pass must resolve `auto` the same way the main loop does"
    );
    assert_ne!(
        with_conddist(0),
        with_conddist(OTHER_COUNT),
        "fixture cannot distinguish the two counts in the conddist pass"
    );
}

#[test]
fn bayes_eta_block_resolves_the_same_count() {
    let model = parse_model_string(MODEL).expect("model parses");
    let pop = sparse_population(&model, 20);

    let run = |n_mh_steps: usize| {
        let opts = FitOptions {
            method: EstimationMethod::Bayes,
            bayes_warmup: 20,
            bayes_iters: 40,
            bayes_chains: 1,
            bayes_seed: Some(7),
            saem_n_mh_steps: n_mh_steps,
            run_covariance_step: false,
            threads: Some(1),
            ..FitOptions::default()
        };
        let res = fit(&model, &pop, &model.default_params, &opts).expect("bayes fit Ok");
        fingerprint(&res)
    };

    assert_eq!(
        run(0),
        run(RESOLVED_ON_SPARSE),
        "the Bayes η block must resolve `auto`; unresolved it would take its `.max(1)` \
         single-proposal path"
    );
    assert_ne!(
        run(0),
        run(1),
        "fixture cannot distinguish {RESOLVED_ON_SPARSE} η proposals from 1 — the equality \
         above would then also hold for the unresolved sentinel"
    );
}
