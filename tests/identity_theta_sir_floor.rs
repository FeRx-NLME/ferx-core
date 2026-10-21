//! SIR's proposal conditioning does not see the units of an identity-packed θ (#2179).
//!
//! A θ whose lower bound is negative is packed as itself, so its variance in the covariance
//! matrix SIR samples from scales as `k²` when the θ is written `k` times larger.
//! `condition_free_proposal` floored that matrix's eigenvalues at `max(1e-10·λ_max, 1e-12)`, so
//! at `k = 1e5` the θ's own variance became `λ_max` and every other direction fell under the
//! floor, and at `k = 3e-6` the θ's variance fell under the absolute `1e-12`. The covariance
//! step itself is units-free since #2145 (`tests/cov_floor_identity_theta_units.rs`), so what is
//! measured here is SIR's own floor. These fits evaluate once (`outer_maxiter = 0`) and compare
//! each rescaled file with the same model at `k = 1`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

/// Warfarin at its FOCEI estimates, with `V` written as the identity-packed `TVV_S = k·V`
/// (bounds `±100·TVV_S`, so the lower bound is negative and the θ is packed as itself).
fn warfarin_scaled_v(k: f64) -> String {
    let tv = 7.737706 * k;
    format!(
        "
[parameters]
  theta TVCL(0.132695, 0.001, 10.0)
  theta TVV_S({tv:e}, {lo:e}, {hi:e})
  theta TVKA(0.810796, 0.01, 50.0)
  omega ETA_CL ~ 0.028590
  omega ETA_V  ~ 0.009592
  omega ETA_KA ~ 0.335871
  sigma PROP_ERR ~ 0.010565 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV_S / {k:e} * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = -100.0 * tv,
        hi = 100.0 * tv,
    )
}

fn fit_scaled(k: f64) -> FitResult {
    let model = parse_model_string(&warfarin_scaled_v(k)).expect("scaled-V model parses");
    let pop = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("data loads");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::FoceI;
    opts.interaction = true;
    opts.outer_maxiter = 0;
    opts.run_covariance_step = true;
    opts.covariance_method = CovarianceMethod::Hessian;
    opts.sir = true;
    opts.sir_samples = 1000;
    opts.sir_resamples = 250;
    opts.sir_seed = Some(7);
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("pinned fit runs")
}

const NAMES: [&str; 7] = [
    "TVCL", "TVV", "TVKA", "ETA_CL", "ETA_V", "ETA_KA", "PROP_ERR",
];

/// Every SIR interval on the natural scale, `TVV_S`'s divided by `k`, in [`NAMES`] order, and
/// the ESS; `Err` with the fit's warnings when SIR produced no intervals.
fn natural_sir(r: &FitResult, k: f64) -> Result<(Vec<(f64, f64)>, f64), String> {
    let Some(th) = r.sir_ci_theta.as_ref() else {
        return Err(format!("SIR produced no theta CIs: {:?}", r.warnings));
    };
    let mut ci = vec![th[0], (th[1].0 / k, th[1].1 / k), th[2]];
    ci.extend(r.sir_ci_omega.as_ref().expect("omega CIs").iter().copied());
    ci.extend(r.sir_ci_sigma.as_ref().expect("sigma CIs").iter().copied());
    assert_eq!(ci.len(), NAMES.len(), "k = {k:e}: one CI per parameter");
    Ok((ci, r.sir_ess.expect("SIR ESS")))
}

/// The SIR warnings that say the proposal was floored or capped.
fn conditioning_warnings(r: &FitResult) -> Vec<&String> {
    r.warnings
        .iter()
        .filter(|w| w.contains("carry no uncertainty") || w.contains("proposal was shrunk"))
        .collect()
}

#[test]
fn rescaling_an_identity_packed_theta_leaves_every_sir_interval_where_it_was() {
    let r1 = fit_scaled(1.0);
    assert!(
        conditioning_warnings(&r1).is_empty(),
        "k = 1: the reference proposal was conditioned: {:?}",
        r1.warnings
    );
    let (reference, ess_ref) = natural_sir(&r1, 1.0).expect("k = 1 SIR runs");
    // Every k is measured before anything is asserted, so one failing row does not hide the
    // others.
    let mut failures = Vec::new();
    for k in [1e3, 1e5, 1e6, 1e8, 3e-6] {
        let label = format!("k = {k:e}");
        let r = fit_scaled(k);
        let conditioning = conditioning_warnings(&r);
        if !conditioning.is_empty() {
            failures.push(format!(
                "{label}: the proposal was conditioned: {conditioning:?}"
            ));
        }
        let (got, ess) = match natural_sir(&r, k) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{label}: {e}"));
                continue;
            }
        };
        let mut worst: f64 = 0.0;
        for (i, (g, w)) in got.iter().zip(&reference).enumerate() {
            for (end, gv, wv) in [("lo", g.0, w.0), ("hi", g.1, w.1)] {
                // `f64::max` would discard a NaN, so a non-finite endpoint is a failure of its own.
                if !gv.is_finite() {
                    failures.push(format!("{label}: CI({}) {end} = {gv}", NAMES[i]));
                }
                let rel = ((gv - wv) / wv).abs();
                eprintln!(
                    "{label}: CI({}) {end} {gv:.9e} vs {wv:.9e}, rel {rel:.3e}",
                    NAMES[i]
                );
                worst = worst.max(rel);
            }
        }
        let ess_rel = ((ess - ess_ref) / ess_ref).abs();
        eprintln!(
            "WORST {label}: CI {worst:.3e}, ESS {ess:.6} vs {ess_ref:.6} (rel {ess_rel:.3e})"
        );
        // Before #2179, on macOS arm64 at 594d1e75: k = 1e5 moved a CI by 25 % and the ESS
        // from 411.3 to 66.8, k = 1e6 by 26× with an ESS of 1.0, k = 3e-6 by 16 % with 212.9,
        // and at k = 1e8 every one of the 1000 draws had an invalid weight. Since, the draws
        // are `ŷ + L·z` with `L` exactly `K·L₁`, so what is left is rounding: worst CI
        // difference 2.3e-15 (k = 1e6) and ESS 1.6e-10 (k = 1e8) on macOS arm64. The ESS
        // moves more than the CIs because the weights carry the objective's own rounding at
        // each draw. The bounds are 430× and 60× the macOS worst.
        if !(worst < 1e-12) {
            failures.push(format!("{label}: worst relative CI difference {worst:e}"));
        }
        if !(ess_rel < 1e-8) {
            failures.push(format!("{label}: ESS {ess} vs {ess_ref}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
