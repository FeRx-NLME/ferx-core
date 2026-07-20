//! Tier-3 convergence anchor for the Phase 4 Binary / logistic endpoint (#760).
//!
//! Gated behind BOTH `survival` (the feature) and `slow-tests` (a full fit to
//! convergence), so it is skipped on the per-PR job and runs nightly — see the
//! test-tier rules in CLAUDE.md.
//!
//! **Exact anchor.** A fixed-effects (`n_eta = 0`) logistic fit *is* ordinary
//! logistic regression, so ferx must reproduce base-R
//! `glm(DV ~ X + TIME, family = binomial)` on the same committed data. The reference,
//! from `data/binary_logistic.csv` (generator seeded, 60 subjects × 3 times):
//!
//! ```text
//! (Intercept) = -0.775172   X = 0.870140   TIME = 0.827029   deviance (-2logL) = 213.5955
//! ```
//!
//! ferx's OFV is on the `-2logL` (deviance) scale, so it must equal the glm deviance —
//! the strongest correctness signal, pinning the Bernoulli likelihood constant *and*
//! form against an independent tool. NONMEM `F_FLAG=1` was also run and reproduces the
//! same fit to 6 sig figs (OFV 213.59548; control file + numbers in
//! `tests/reference/binary_logistic/`), so this test asserts against the shared glm /
//! NONMEM values.

#![cfg(all(feature = "survival", feature = "slow-tests"))]

use ferx_core::{fit_from_files, FitOptions};

#[test]
fn binary_fixed_effects_matches_r_glm() {
    let r = fit_from_files(
        "examples/binary_logistic.ferx",
        Some("data/binary_logistic.csv"),
        Some(&["X"]),
        Some(FitOptions::default()),
    )
    .expect("fixed-effects binary fit must converge");

    // base-R glm(DV ~ X + TIME, family = binomial) on data/binary_logistic.csv.
    let (r_intercept, r_x, r_time, r_deviance) = (-0.775172, 0.870140, 0.827029, 213.595477);
    let (th0, thx, tht) = (r.theta[0], r.theta[1], r.theta[2]);

    // OFV == glm deviance pins the likelihood constant + form (exact to optimizer tol).
    assert!(
        (r.ofv - r_deviance).abs() < 0.05,
        "ferx OFV {} vs glm deviance {r_deviance}",
        r.ofv
    );
    // θ estimates match the glm MLE within the derivative-free (BOBYQA) outer tolerance.
    assert!(
        (th0 - r_intercept).abs() < 0.01,
        "TH0 {th0} vs glm {r_intercept}"
    );
    assert!((thx - r_x).abs() < 0.01, "THX {thx} vs glm {r_x}");
    assert!((tht - r_time).abs() < 0.01, "THT {tht} vs glm {r_time}");
}

/// **Simulation-estimation (SSE) round-trip — Slice 1b (§8.8.8).**
///
/// Simulate binary outcomes from a *known* θ with ferx, refit them with ferx, and
/// require the generating θ back. This is the only test that can catch a **wrong
/// sampler**: every fit-side test (including the glm anchor above) scores observed
/// data and would pass unchanged if `simulate()` drew from the wrong `p` — or, as
/// before this slice, drew nothing at all.
///
/// Design: reuse the committed fixed-effects model/data for its covariate structure,
/// overwrite the outcomes with draws at a known θ, then refit. `n_eta = 0` keeps the
/// recovery deterministic (no EBE / Ω noise) so the tolerance can be tight and the
/// test cannot flake on a random-effects draw.
#[test]
fn binary_simulate_then_fit_recovers_theta() {
    use ferx_core::api::read_population_for;
    use ferx_core::parser::model_parser::parse_full_model_file;
    use ferx_core::types::{ObsRecord, SimOutcome};
    use std::path::Path;

    let parsed = parse_full_model_file(Path::new("examples/binary_logistic.ferx"))
        .expect("example model must parse");
    let (template, _cov) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/binary_logistic.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("example data must read");

    // Known generating θ = (TH0, THX, THT) — deliberately not the glm optimum above,
    // so recovering it proves the sampler followed *these* parameters and not some
    // residue of the fitted data.
    let truth = [-0.6_f64, 1.1, 0.4];
    let mut gen_params = parsed.model.default_params.clone();
    gen_params.theta = truth.to_vec();

    // 20 replicate draws over 180 records ⇒ 3600 Bernoulli outcomes, enough that the
    // MLE sits within ~0.1 of truth while keeping the fit fast.
    let n_sim = 20;
    let sims =
        ferx_core::simulate_with_seed(&parsed.model, &template, &gen_params, n_sim, 20260720);
    assert_eq!(
        sims.len(),
        template
            .subjects
            .iter()
            .map(|s| s.obs_records.len())
            .sum::<usize>()
            * n_sim,
        "simulate() must emit one row per (record × replicate)"
    );

    // Rebuild a population whose outcomes are the simulated ones: each replicate
    // becomes its own subject (ids stay unique), so the refit sees 3600 observations.
    let mut pop = template.clone();
    pop.subjects.clear();
    for k in 0..n_sim {
        let replicate: Vec<&ferx_core::SimulationResult> =
            sims.iter().filter(|r| r.sim == k + 1).collect();
        let mut cursor = 0usize;
        for subj in &template.subjects {
            let mut s = subj.clone();
            s.id = format!("{}_{}", subj.id, k + 1);
            for rec in &mut s.obs_records {
                if let ObsRecord::DiscreteState { state, .. } = rec {
                    match replicate[cursor].outcome {
                        SimOutcome::Category { state: drawn } => *state = drawn,
                        ref o => panic!("expected a Category outcome, got {o:?}"),
                    }
                    cursor += 1;
                }
            }
            pop.subjects.push(s);
        }
        assert_eq!(
            cursor,
            replicate.len(),
            "every simulated row must be consumed"
        );
    }

    let refit = ferx_core::fit(
        &parsed.model,
        &pop,
        &parsed.model.default_params,
        &FitOptions::default(),
    )
    .expect("refit of simulated binary data must converge");

    // Recovery: with 3600 draws the per-θ SE is ≈0.05, so 0.15 is ~3 SE — tight enough
    // that a sampler drawing from the wrong p (e.g. an inverted link, or p from the
    // wrong η) fails, loose enough to absorb Monte-Carlo noise at this seed.
    for (i, (&got, &want)) in refit.theta.iter().zip(truth.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.15,
            "theta[{i}] recovered {got}, generated from {want} (diff {})",
            (got - want).abs()
        );
    }
}
