//! Integration test: `simulate()` samples inter-occasion variability (kappa).
//!
//! Before the fix, `simulate()` zeroed every occasion kappa (the drawn
//! `eta_slice` was resized with `0.0` into the kappa slots and `omega_iov` was
//! never sampled), so a simulated / VPC dataset carried **no** between-occasion
//! variability regardless of the fitted `Omega_IOV` — silently under-dispersing
//! relative to the fitted model and to NONMEM `$SIM`.
//!
//! This pins that a nonzero `Omega_IOV` actually moves the simulated
//! predictions. It compares simulating with the fitted `Omega_IOV` against
//! simulating with `Omega_IOV` forced to zero, under the *same* seed: the kappa
//! draws still happen (so the RNG stream stays aligned), and the only difference
//! is the kappa magnitude. On the old code both are byte-identical (kappa is
//! always zero), so the difference assertion below fails; with the fix they
//! differ by the sampled inter-occasion variability.

use ferx_core::{parse_model_file, read_nonmem_csv, simulate_with_seed, SimulationResult};
use std::path::Path;

fn ipreds(rows: &[SimulationResult]) -> Vec<f64> {
    rows.iter().map(|r| r.ipred).collect()
}

#[test]
fn simulate_samples_inter_occasion_kappa() {
    let model =
        parse_model_file(Path::new("examples/warfarin_iov.ferx")).expect("warfarin_iov parses");
    assert!(model.n_kappa > 0, "fixture must declare IOV (kappa)");

    // `iov_column = Some("OCC")` so the per-record occasion labels are read;
    // without it the subjects carry no occasions and kappa has nothing to vary.
    let pop = read_nonmem_csv(Path::new("data/warfarin_iov.csv"), None, Some("OCC"))
        .expect("warfarin_iov data loads");
    assert!(
        pop.subjects.iter().any(|s| !s.occasions.is_empty()),
        "fixture data must carry occasion labels"
    );

    let seed = 20_260_707;
    let n_sim = 3;

    // (a) Simulate with the fitted Omega_IOV.
    let with_iov = simulate_with_seed(&model, &pop, &model.default_params, n_sim, seed);

    // (b) Same everything, but Omega_IOV forced to zero — the old (buggy)
    // behaviour. The per-occasion kappa draws still happen (RNG stays aligned),
    // but scale to zero, so any difference from (a) is purely the sampled
    // inter-occasion variability.
    let mut zero_iov = model.default_params.clone();
    {
        let om = zero_iov
            .omega_iov
            .as_mut()
            .expect("omega_iov present for an IOV model");
        om.chol.fill(0.0);
        om.matrix.fill(0.0);
    }
    let without_iov = simulate_with_seed(&model, &pop, &zero_iov, n_sim, seed);

    assert_eq!(
        with_iov.len(),
        without_iov.len(),
        "same design ⇒ same number of simulated rows"
    );
    assert!(
        ipreds(&with_iov).iter().all(|v| v.is_finite()),
        "simulated ipreds must be finite"
    );

    // The fitted Omega_IOV must actually move the predictions. Old code: kappa is
    // always zero, so the two runs are byte-identical and this fails.
    let max_abs_diff = ipreds(&with_iov)
        .iter()
        .zip(ipreds(&without_iov).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs_diff > 1e-6,
        "simulate() must sample Omega_IOV: with-IOV and zero-IOV ipreds are identical \
         (max abs diff {max_abs_diff:.3e}) — occasion kappa is being dropped"
    );

    // Reproducibility: same seed + params ⇒ identical draws.
    let with_iov_again = simulate_with_seed(&model, &pop, &model.default_params, n_sim, seed);
    assert_eq!(
        ipreds(&with_iov),
        ipreds(&with_iov_again),
        "simulate() must be reproducible under a fixed seed"
    );
}
