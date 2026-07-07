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
use std::collections::BTreeMap;
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

/// NONMEM `$SIM` variance-component anchor for the IOV draw.
///
/// The RNG-alignment test above proves kappa is *sampled*; this pins that it is
/// sampled with the right *magnitude* and per-occasion independence, cross-checked
/// against NONMEM.
///
/// Design (`tests/fixtures/iov_anchor.{ferx,csv}`, shared bit-for-bit with the
/// NONMEM kit `nonmem_anchor/iov_anchor.{ctl,csv}`): a 1-cpt IV-bolus model with
/// IOV on `V`, `EVID=4`
/// (reset + bolus) at the start of each of 3 occasions, and one observation at
/// `t=0`. Because the compartment is reset and no time elapses,
/// `IPRED = AMT / V` exactly, so
///   `log IPRED = log(AMT) − log(TVV) − ETA_V − KAPPA_occ`.
/// Within a subject only `KAPPA_occ` varies between occasions, so the pooled
/// within-subject between-occasion variance of `log IPRED` is an unbiased
/// estimator of `omega^2_IOV` (here 0.04) — no back-calculation needed.
///
/// Cross-tool anchor on the identical 300×3 design (truth `omega^2_IOV` = 0.04000):
///   NONMEM 7.6.0 `$SIMULATION` (1 subproblem, df 600) = 0.04151  (SE ~0.0024)
///   ferx `simulate_with_seed` (20 replicates, df 12000) = 0.03939  (this test)
///   ferx high-N centring (100 replicates, df 60000)     = 0.03988  (0.5 SE from truth)
/// All three agree with the 0.04 truth within Monte-Carlo error, and the high-N
/// ferx run rules out any systematic under-dispersion. Old code (occasion kappa
/// dropped) collapses this to ~0.0.
#[test]
fn simulate_iov_recovers_omega_iov_variance() {
    let model = parse_model_file(Path::new("tests/fixtures/iov_anchor.ferx"))
        .expect("iov_anchor model parses");
    assert!(model.n_kappa > 0, "anchor must declare IOV (kappa)");
    let pop = read_nonmem_csv(
        Path::new("tests/fixtures/iov_anchor.csv"),
        None,
        Some("OCC"),
    )
    .expect("iov_anchor data loads");

    // 20 replicates, fixed seed ⇒ deterministic; df≈12000 so the estimate's
    // SE (~5e-4) is far below the ferx↔NONMEM gap, turning this into a tight
    // recovery check rather than a noisy one.
    let n_rep = 20;
    let rows = simulate_with_seed(&model, &pop, &model.default_params, n_rep, 20_260_707);

    // Group log(IPRED) by (replicate, subject, occasion). Each replicate redraws
    // independent kappa, so the key must include `sim` — pooling across replicates
    // by `id` alone would fold in between-replicate spread. Occasion ↔ time {0,24,48}.
    let mut by_subj: BTreeMap<(usize, String), BTreeMap<u64, f64>> = BTreeMap::new();
    for r in &rows {
        assert!(
            r.ipred.is_finite() && r.ipred > 0.0,
            "ipred must be positive"
        );
        let occ = (r.time / 24.0).round() as u64;
        by_subj
            .entry((r.sim, r.id.clone()))
            .or_default()
            .insert(occ, r.ipred.ln());
    }

    // Pooled within-subject between-occasion variance: Σ Σ (x − mean_i)² / Σ (n_i − 1).
    let (mut ss, mut df) = (0.0_f64, 0usize);
    for occs in by_subj.values() {
        let v: Vec<f64> = occs.values().copied().collect();
        if v.len() >= 2 {
            let mean = v.iter().sum::<f64>() / v.len() as f64;
            ss += v.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
            df += v.len() - 1;
        }
    }
    let var = ss / df as f64;
    eprintln!("ferx between-occasion Var(log IPRED) = {var:.5} (df={df}); target omega^2_IOV=0.04, NONMEM=0.04151");

    // Recovers omega^2_IOV=0.04. The estimate here (df≈12000, SE ~5e-4) sits at
    // 0.0394; the 0.008 band is deliberately loose — wide enough to survive a
    // benign RNG-stream reordering, tight enough to still catch the regressions
    // this guards: occasion kappa dropped (→ ~0.0) or a wrong-scale draw.
    assert!(
        (var - 0.04).abs() < 0.008,
        "between-occasion Var(log IPRED) = {var:.5}, expected ~0.04 (omega^2_IOV); \
         NONMEM $SIM on the same design gives 0.0415"
    );
}
