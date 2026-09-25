//! Tier-2 integration tests for the automatic SAEM block-proposal count (#1459).
//!
//! `FitOptions::saem_n_mh_steps` defaults to `0` — the `auto` sentinel — and the
//! SAEM E-step and the conditional-distribution pass resolve it from the dataset
//! with `estimation::saem::auto_n_mh_steps`. The arithmetic is unit-tested next
//! to that function; what these tests guard is the **wiring**, which the unit
//! tests cannot see: each consumer reads the option separately, and one left
//! reading it raw would hand its kernel a literal zero-proposal count — a
//! silent, green regression.
//!
//! Two properties per consumer, on **three fixtures that land on three
//! different arms of the rule**:
//!
//! | fixture | obs/subject | obs/η | rule | arm |
//! |---|---|---|---|---|
//! | [`sparse_population`] | 2 | 1.0 | 6 | the floor |
//! | [`middling_population`] | 6 | 3.0 | 8 | the density term |
//! | [`dense_population`] | 16 | 8.0 | 20 | the cap |
//!
//! * the default must be **bit-identical** to the explicit count the rule
//!   resolves to on that fixture, and
//! * that count must be **distinguishable** from a different explicit count on
//!   the same fixture — otherwise the equality holds for any wiring at all.
//!
//! Three arms rather than one because a single fixture cannot tell the rule from
//! a constant: with only the sparse case, replacing a resolver call by
//! `if requested == 0 { 6 } else { requested }` passes everything (raised in
//! review of PR #1468).
//!
//! The Bayes η block deliberately does **not** follow the rule — it keeps the
//! historical fixed count under `auto`, because the rule was calibrated on SAEM
//! quantities. `bayes_eta_block_keeps_the_historical_count` pins that, so a
//! later change to it has to be a decision rather than a side effect.
//!
//! Data are fully synthetic (simulated from the model's own parameters).

use ferx_core::estimation::saem_conddist::run_conditional_distribution;
use ferx_core::types::{CompiledModel, DoseEvent, Population};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};
use nalgebra::DVector;

mod common;

/// What `auto` resolves to on each fixture. Written out rather than computed, so
/// a change to the rule has to be reflected here deliberately.
const RESOLVED_SPARSE: usize = 6; // floor:   2.5 · 1.0 = 2.5  → 3, lifted to 6
const RESOLVED_MIDDLING: usize = 8; // density: 2.5 · 3.0 = 7.5  → 8
const RESOLVED_DENSE: usize = 20; // cap:     2.5 · 8.0 = 20.0 → 20

/// The pre-#1459 fixed default, and the count the Bayes η block still uses.
const HISTORICAL_COUNT: usize = 20;

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

/// A population of `n` subjects with `obs_times.len()` observations each,
/// simulated from the model's own parameters.
fn population_with(model: &CompiledModel, n: usize, obs_times: &[f64]) -> Population {
    let subjects = (1..=n)
        .map(|i| {
            common::subject(
                &format!("{i}"),
                vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times.to_vec(),
                vec![0.0; obs_times.len()],
                vec![1; obs_times.len()],
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

    let sim = simulate_with_seed(model, &template, &model.default_params, 1, 20260919).unwrap();
    let mut pop = template.clone();
    for subj in pop.subjects.iter_mut() {
        let dv: Vec<f64> = sim
            .iter()
            .filter(|s| s.id == subj.id)
            .map(|s| s.outcome.continuous_value())
            .collect();
        // A non-finite simulated concentration would otherwise be floored to a
        // valid-looking number below and the fixture would go on to "pass" on a
        // dataset that never existed.
        assert!(
            dv.len() == subj.obs_times.len() && dv.iter().all(|x| x.is_finite()),
            "simulation produced {} non-finite or missing values for subject {}: {dv:?}",
            dv.iter().filter(|x| !x.is_finite()).count(),
            subj.id
        );
        subj.observations = dv.into_iter().map(|x| x.max(1e-6)).collect();
    }
    pop
}

/// 2 observations per subject against 2 η — 1.0 per η, the floor arm.
fn sparse_population(model: &CompiledModel) -> Population {
    population_with(model, 20, &[1.0, 8.0])
}

/// 6 observations per subject against 2 η — 3.0 per η, the density arm.
fn middling_population(model: &CompiledModel) -> Population {
    population_with(model, 20, &[0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
}

/// 16 observations per subject against 2 η — 8.0 per η, the cap arm (the shape
/// of the Emax PKPD benchmark the historical 20 was calibrated on).
fn dense_population(model: &CompiledModel) -> Population {
    population_with(
        model,
        20,
        &[
            0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0,
        ],
    )
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

/// `(theta, omega diagonal, sigma)` of a fit, as the comparison key. A change in
/// the number of MH proposals moves every one of them.
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
fn saem_default_resolves_to_the_rules_count_on_every_arm() {
    let model = parse();

    // The default must BE the sentinel; if it is ever changed to a literal count
    // this test would otherwise still pass while testing nothing.
    assert_eq!(
        FitOptions::default().saem_n_mh_steps,
        0,
        "the default is the `auto` sentinel — see FitOptions::saem_n_mh_steps"
    );

    let run = |pop: &Population, n: usize| {
        fingerprint(&fit(&model, pop, &model.default_params, &short_saem_opts(n)).expect("fit Ok"))
    };

    for (what, pop, resolved, other) in [
        ("sparse", sparse_population(&model), RESOLVED_SPARSE, 20),
        (
            "middling",
            middling_population(&model),
            RESOLVED_MIDDLING,
            6,
        ),
        ("dense", dense_population(&model), RESOLVED_DENSE, 6),
    ] {
        assert_eq!(
            run(&pop, 0),
            run(&pop, resolved),
            "{what}: `auto` must run exactly {resolved} block proposals"
        );
        assert_ne!(
            run(&pop, resolved),
            run(&pop, other),
            "{what}: fixture cannot distinguish {resolved} proposals from {other}, so the \
             equality above would hold for any wiring"
        );
    }
}

/// The conditional-distribution pass, called **directly** so that nothing
/// upstream of it varies: same model, same parameters, same warm-start ETAs, and
/// the only difference between the two calls is `saem_n_mh_steps`.
///
/// Going through `fit()` would not isolate it — the same option also sizes the
/// SAEM E-step that produces the parameters the pass then samples at, so a
/// difference could come from the fit rather than from this pass (raised in
/// review of PR #1468).
#[test]
fn conddist_pass_resolves_the_same_count_on_every_arm() {
    let model = parse();

    let conddist = |pop: &Population, n_mh_steps: usize| {
        let opts = FitOptions {
            saem_conddist_nsamp: 40,
            saem_conddist_burnin: 10,
            saem_n_mh_steps: n_mh_steps,
            saem_seed: Some(99),
            ..FitOptions::default()
        };
        let warm: Vec<DVector<f64>> = pop
            .subjects
            .iter()
            .map(|_| DVector::zeros(model.n_eta))
            .collect();
        let warm_kappas: Vec<Vec<DVector<f64>>> = pop.subjects.iter().map(|_| Vec::new()).collect();
        let cd = run_conditional_distribution(
            &model,
            pop,
            &model.default_params,
            &warm,
            &warm_kappas,
            &opts,
        );
        let means: Vec<f64> = cd
            .cond_mean
            .iter()
            .flat_map(|m| m.iter().copied())
            .collect();
        assert!(
            !means.is_empty() && means.iter().all(|x| x.is_finite()),
            "non-finite or empty conditional mean: {means:?}"
        );
        means
    };

    for (what, pop, resolved, other) in [
        ("sparse", sparse_population(&model), RESOLVED_SPARSE, 20),
        (
            "middling",
            middling_population(&model),
            RESOLVED_MIDDLING,
            6,
        ),
        ("dense", dense_population(&model), RESOLVED_DENSE, 6),
    ] {
        assert_eq!(
            conddist(&pop, 0),
            conddist(&pop, resolved),
            "{what}: the conditional-distribution pass must resolve `auto` to {resolved}, the \
             same count the main loop resolves"
        );
        assert_ne!(
            conddist(&pop, resolved),
            conddist(&pop, other),
            "{what}: fixture cannot distinguish {resolved} proposals from {other} in the \
             conddist pass"
        );
    }
}

/// The Bayes η block keeps the historical count under `auto` — on **both** a
/// sparse and a dense dataset, so the property is "does not read the rule",
/// not "agrees with the rule on this one shape".
///
/// The rule is calibrated on SAEM quantities and the evidence that a low count
/// is safe rests on SAEM's componentwise kernel (#1466); this sampler has only
/// the block kernel and is judged on posterior mixing. Lowering it here needs
/// its own benchmark — if you add one, change this test deliberately.
#[test]
fn bayes_eta_block_keeps_the_historical_count() {
    let model = parse();

    let run = |pop: &Population, n_mh_steps: usize| {
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
        fingerprint(&fit(&model, pop, &model.default_params, &opts).expect("bayes fit Ok"))
    };

    for (what, pop) in [
        ("sparse", sparse_population(&model)),
        ("dense", dense_population(&model)),
    ] {
        assert_eq!(
            run(&pop, 0),
            run(&pop, HISTORICAL_COUNT),
            "{what}: the Bayes η block must keep {HISTORICAL_COUNT} proposals under `auto`"
        );
        assert_ne!(
            run(&pop, HISTORICAL_COUNT),
            run(&pop, RESOLVED_SPARSE),
            "{what}: fixture cannot distinguish {HISTORICAL_COUNT} η proposals from \
             {RESOLVED_SPARSE}, so the equality above would also hold if it read the rule"
        );
    }
}

fn parse() -> CompiledModel {
    ferx_core::parser::model_parser::parse_model_string(MODEL).expect("model parses")
}
