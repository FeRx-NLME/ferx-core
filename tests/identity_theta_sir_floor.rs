//! SIR's proposal conditioning does not see the units of an identity-packed θ (#2179).
//!
//! A θ whose lower bound is negative is packed as itself, so its variance in the covariance
//! matrix SIR samples from scales as `k²` when the θ is written `k` times larger.
//! `condition_free_proposal` floored that matrix's eigenvalues at `max(1e-10·λ_max, 1e-12)`, so
//! at `k = 1e5` the θ's own variance became `λ_max` and every other direction fell under the
//! floor, and at `k = 3e-6` the θ's variance fell under the absolute `1e-12`. The covariance
//! step itself is units-free since #2145 (`tests/cov_floor_identity_theta_units.rs`), so what is
//! measured here is SIR's own floor. The same holds for an Ω Cholesky off-diagonal whose row's
//! η is written in other units (#2274): SIR divides it by `L̂_ii` as the covariance step does.
//! These fits evaluate once (`outer_maxiter = 0`) and compare each rescaled file with the same
//! model at `k = 1`.

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
        // each draw. On Linux x86_64 (glibc 2.28) the worst are 3.4e-15 (k = 1e6) and 7.7e-11
        // (k = 3e-6). The bounds are 430× and 60× the macOS worst, 300× and 130× the Linux.
        if !(worst < 1e-12) {
            failures.push(format!("{label}: worst relative CI difference {worst:e}"));
        }
        if !(ess_rel < 1e-8) {
            failures.push(format!("{label}: ESS {ess} vs {ess_ref}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// #2289's block-Ω fixture (`tests/cov_floor_omega_offdiag_units.rs`): the volume carries an
/// additive `ETA_VS` in units `k` times smaller, blocked with the exponential `ETA_CL`, so the
/// Cholesky off-diagonal `L₂₁ ∝ k` and its variance in `Σ` scales as `k²`, while every θ here is
/// log-packed. SIR conditions its proposal in the covariance step's basis, so the off-diagonal is
/// divided by its row's `L̂₂₂` there too (#2274).
fn covmuref_offdiag(k: f64) -> String {
    let tvv_s = 50.670449 * k;
    let w_vs = 116.938296 * k * k;
    let c = -0.147595 * k;
    format!(
        "[parameters]
  theta TVCL(4.820644, 0.01, 100.0)
  theta TH_WEAK(1.000317, 0.01, 100.0)
  theta TVV_S({tvv_s:e}, {lo:e}, {hi:e})
  block_omega (ETA_CL, ETA_VS) = [0.068897, {c:e}, {w_vs:e}]
  sigma PROP_ERR ~ 0.098500 (sd)
[individual_parameters]
  CL = TVCL * (WT / 70.0) ^ (0.9124 + 1e-3 * (TH_WEAK - 1.0)) * exp(ETA_CL)
  V  = (TVV_S + ETA_VS) / {k:e}
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
",
        lo = 0.01 * tvv_s,
        hi = 100.0 * tvv_s,
    )
}

fn fit_offdiag(k: f64) -> FitResult {
    let model = parse_model_string(&covmuref_offdiag(k)).expect("block-Ω model parses");
    let pop =
        read_nonmem_csv(Path::new("data/covmuref_power.csv"), None, None).expect("data loads");
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

const OFFDIAG_NAMES: [&str; 6] = [
    "TVCL",
    "TH_WEAK",
    "TVV_S/k",
    "omega_CL",
    "omega_VS/k^2",
    "PROP_ERR",
];

/// SIR intervals on the `k = 1` scale in [`OFFDIAG_NAMES`] order (SIR reports Ω's diagonal
/// only), and the ESS.
fn offdiag_sir(r: &FitResult, k: f64) -> Result<(Vec<(f64, f64)>, f64), String> {
    let Some(th) = r.sir_ci_theta.as_ref() else {
        return Err(format!("SIR produced no theta CIs: {:?}", r.warnings));
    };
    let om = r.sir_ci_omega.as_ref().expect("omega CIs");
    let mut ci = vec![th[0], th[1], (th[2].0 / k, th[2].1 / k)];
    ci.push(om[0]);
    ci.push((om[1].0 / (k * k), om[1].1 / (k * k)));
    ci.extend(r.sir_ci_sigma.as_ref().expect("sigma CIs").iter().copied());
    assert_eq!(
        ci.len(),
        OFFDIAG_NAMES.len(),
        "k = {k:e}: one CI per parameter"
    );
    Ok((ci, r.sir_ess.expect("SIR ESS")))
}

#[test]
fn rescaling_an_additive_eta_leaves_every_sir_interval_where_it_was() {
    let r1 = fit_offdiag(1.0);
    let cond_ref = conditioning_warnings(&r1).len();
    let (reference, ess_ref) = offdiag_sir(&r1, 1.0).expect("k = 1 SIR runs");
    let mut failures = Vec::new();
    // Only units smaller than the fixture's own. The packing rails `|L_ij| ≤ 10` and
    // `log L_ii ∈ [−6, 6]` act on the Cholesky entries in the file's units, so they cap and reject
    // draws differently at another `k`: `L̂₂₂ ≈ 10.8·k` and `L̂₂₁ ≈ −0.56·k`, and at k = 2 the rails
    // already move a CI by 5.7e-2 (19 % at k = 5) whatever the proposal's basis is. Below
    // k ≈ 2.3e-4 `log L̂₂₂` itself leaves the rail. The k here keep every draw inside both.
    for k in [0.5, 0.1, 0.01, 3e-3] {
        let label = format!("k = {k:e}");
        let r = fit_offdiag(k);
        let cond = conditioning_warnings(&r);
        if cond.len() != cond_ref {
            failures.push(format!(
                "{label}: {} conditioning warnings, k = 1 had {cond_ref}: {cond:?}",
                cond.len()
            ));
        }
        let (got, ess) = match offdiag_sir(&r, k) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{label}: {e}"));
                continue;
            }
        };
        let mut worst: f64 = 0.0;
        for (i, (g, w)) in got.iter().zip(&reference).enumerate() {
            for (end, gv, wv) in [("lo", g.0, w.0), ("hi", g.1, w.1)] {
                if !gv.is_finite() {
                    failures.push(format!("{label}: CI({}) {end} = {gv}", OFFDIAG_NAMES[i]));
                }
                let rel = ((gv - wv) / wv).abs();
                eprintln!(
                    "{label}: CI({}) {end} {gv:.9e} vs {wv:.9e}, rel {rel:.3e}",
                    OFFDIAG_NAMES[i]
                );
                worst = worst.max(rel);
            }
        }
        let ess_rel = ((ess - ess_ref) / ess_ref).abs();
        eprintln!(
            "WORST {label}: CI {worst:.3e}, ESS {ess:.6} vs {ess_ref:.6} (rel {ess_rel:.3e})"
        );
        // On macOS arm64, with the off-diagonal left `Dimensionless` in SIR's proposal (the
        // covariance step's basis since #2289, SIR's before it) every k moved a CI by 2.0e-4
        // and the ESS by 2.6e-4 to 3.4e-4. With it `Scaled(L̂₂₂)` the worst CI difference is
        // 4.7e-10 (k = 0.1) and the ESS 4.0e-7 (k = 0.01). k = 0.5, a power of two that
        // rescales every float exactly, agrees to 1.1e-14; at the other k the covariance step's
        // finite-difference Hessian rounds differently, so `Σ` itself moves at that level.
        // Linux x86_64 (glibc 2.28) gives the same worst values to every printed digit (k = 0.5:
        // 3.7e-14). The bounds are 21× and 25× that worst, and 20,000× and 26× under the
        // unscaled proposal.
        if !(worst < 1e-8) {
            failures.push(format!("{label}: worst relative CI difference {worst:e}"));
        }
        if !(ess_rel < 1e-5) {
            failures.push(format!("{label}: ESS {ess} vs {ess_ref}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
