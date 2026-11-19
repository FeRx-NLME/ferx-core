//! Simulate → refit round trip for the `[count_model]` endpoint (#2877), Tier 3.
//!
//! The NONMEM / R anchors score *observed* counts, so they cannot see a wrong sampler:
//! `simulate()` could draw from the wrong mean, the wrong family or the wrong η, and every
//! fit-side test would pass. Here counts are drawn from a known `(θ, r, ω)` on the
//! `count_nb` design (WT between subjects, TRT within), refit, and the generating values
//! must come back.

#![cfg(feature = "survival")]

use ferx_core::types::{FitOptions, ObsRecord, OmegaMatrix, SimOutcome};

/// Draw `n_sim` replicates of the fixture's design at `truth` (`[BASE, BETA, GAM, R?]`) and
/// ω, rebuild one population with each replicate as its own subjects, and refit it from the
/// model file's start under Laplace. Returns the refit's θ and ω.
fn round_trip(fam: &str, truth: &[f64], omega: f64, n_sim: usize, seed: u64) -> (Vec<f64>, f64) {
    let root = env!("CARGO_MANIFEST_DIR");
    let p = ferx_core::prepare_run(
        &format!("{root}/nonmem_anchor/count_{fam}.ferx"),
        Some(&format!("{root}/nonmem_anchor/count_{fam}.csv")),
    )
    .unwrap();
    let model = &p.parsed.model;
    let template = &p.population;
    let mut gen = p.init_params.clone();
    gen.theta = truth.to_vec();
    gen.omega = OmegaMatrix::from_diagonal(&[omega], vec!["ETA_C".into()]);
    let sims = ferx_core::simulate_with_seed(model, template, &gen, n_sim, seed).unwrap();
    let n_rec: usize = template.subjects.iter().map(|s| s.obs_records.len()).sum();
    assert_eq!(
        sims.len(),
        n_rec * n_sim,
        "one row per (record × replicate)"
    );

    let mut pop = template.clone();
    pop.subjects.clear();
    for k in 0..n_sim {
        let rep: Vec<_> = sims.iter().filter(|r| r.sim == k + 1).collect();
        let mut cursor = 0usize;
        for subj in &template.subjects {
            let mut s = subj.clone();
            s.id = format!("{}_{}", subj.id, k + 1);
            for rec in &mut s.obs_records {
                if let ObsRecord::Count { count, .. } = rec {
                    match rep[cursor].outcome {
                        SimOutcome::Count { count: drawn } => *count = drawn,
                        ref o => panic!("expected a Count outcome, got {o:?}"),
                    }
                    cursor += 1;
                }
            }
            pop.subjects.push(s);
        }
        assert_eq!(cursor, rep.len(), "every simulated row is consumed");
    }
    let options = FitOptions {
        verbose: false,
        run_covariance_step: false,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(model, &pop, &p.init_params, &options).unwrap();
    (f.theta.clone(), f.omega[(0, 0)])
}

/// 10 replicates × 360 records (600 subjects). Measured recovery on macOS arm64 at these
/// seeds: Poisson BASE 3.406 (log −0.027), BETA 0.603, GAM −0.473, ω 0.261; NB BASE 3.765
/// (log +0.073), BETA 0.598, GAM −0.532, r 1.995, ω 0.329. With 600 subjects at ω = 0.3 the
/// standard error of log BASE is about √(ω/600) ≈ 0.022, so the 0.12 band is ~5 SE; the
/// other bands are set the same way from the measured gaps with ≥ 1.6× headroom. A sampler
/// drawing from the wrong mean (a sign flip on GAM moves it by 1.0, the population η
/// instead of the individual one sends ω to 0, the NB as a Poisson sends r to its bound)
/// misses them by far more.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn simulated_counts_refit_to_the_generating_parameters() {
    for (fam, truth, omega, seed) in [
        ("pois", vec![3.5, 0.6, -0.5], 0.3, 2877u64),
        ("nb", vec![3.5, 0.6, -0.5, 2.0], 0.3, 2878u64),
    ] {
        let (th, om) = round_trip(fam, &truth, omega, 10, seed);
        eprintln!("count_{fam}: truth {truth:?} ω {omega} → θ {th:?} ω {om:.4}");
        assert!(
            (th[0].ln() - truth[0].ln()).abs() < 0.12,
            "{fam} BASE {}",
            th[0]
        );
        assert!((th[1] - truth[1]).abs() < 0.25, "{fam} BETA {}", th[1]);
        assert!((th[2] - truth[2]).abs() < 0.09, "{fam} GAM {}", th[2]);
        assert!((om - omega).abs() < 0.08, "{fam} ω {om}");
        if fam == "nb" {
            assert!(
                (th[3].ln() - truth[3].ln()).abs() < 0.2,
                "{fam} r {}",
                th[3]
            );
        }
    }
}
