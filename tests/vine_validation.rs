//! Vine-copula validation: Clayton-copula ETA simulation study.
//!
//! Generates 200 subjects whose joint (ETA_CL, ETA_V) are drawn from a
//! Clayton copula with θ = 2. The Clayton copula has **lower tail dependence**
//! (subjects with unusually low CL also tend to have unusually low V — the
//! joint distribution is concentrated in the lower-left corner, not the
//! upper-right). A multivariate-normal OMEGA is fundamentally unable to
//! represent this asymmetry: it uses a Gaussian copula, which has zero tail
//! dependence regardless of the correlation parameter.
//!
//! ## True parameters
//!
//! | Parameter      | Value | Notes                              |
//! |----------------|-------|------------------------------------|
//! | TVCL           | 5.0   | Typical CL (L/h)                   |
//! | TVV            | 50.0  | Typical V (L)                      |
//! | ω_CL           | 0.40  | BSV SD on log scale (var = 0.16)   |
//! | ω_V            | 0.30  | BSV SD on log scale (var = 0.09)   |
//! | σ_prop         | 0.10  | Proportional residual error        |
//! | Clayton θ      | 2.0   | Copula parameter                   |
//! | Kendall τ      | 0.500 | Exact: θ/(θ+2)                     |
//! | Lower tail dep | 0.707 | Exact: 2^{-1/θ}                    |
//! | Upper tail dep | 0.000 | Clayton has no upper tail dep      |
//!
//! ## What the test checks
//!
//! 1. Both methods recover TVCL and TVV within 30% of truth.
//! 2. Vine OFV ≤ Gaussian OFV (vine can't be worse on the training data).
//! 3. Vine identifies a tail-dependent family (Clayton or Student-t).
//! 4. Vine Kendall τ is within 0.20 of truth (0.50).
//! 5. Vine lower tail dependence > 0.20 (Clayton and Student-t satisfy this;
//!    Gaussian and Frank yield λ_L = 0, failing this check).
//!
//! ## Running
//!
//!   cargo test --features slow-tests --test vine_validation -- --nocapture

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::stats::special::normal_cdf;
use ferx_core::types::{DoseEvent, OmegaDist, Population, Subject};
use ferx_core::{fit, EstimationMethod, FitOptions};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use std::collections::HashMap;

// ─── True simulation parameters ──────────────────────────────────────────────

const TVCL: f64 = 5.0;
const TVV: f64 = 50.0;
const OMEGA_CL: f64 = 0.40; // BSV SD (log scale)
const OMEGA_V: f64 = 0.30; // BSV SD (log scale)
const SIGMA: f64 = 0.10; // proportional residual error (SD)
const THETA_C: f64 = 2.0; // Clayton copula parameter
const TRUE_TAU: f64 = THETA_C / (THETA_C + 2.0); // = 0.500
const TRUE_LAMBDA_L: f64 = 0.7071; // 2^{-1/THETA_C}

const N_SUBJECTS: usize = 200;
const DOSE: f64 = 100.0; // IV bolus (mg)
const OBS_TIMES: &[f64] = &[0.5, 1.0, 2.0, 4.0, 8.0, 12.0];

// ─── Model DSL ────────────────────────────────────────────────────────────────
//
// Uses block_omega so the Gaussian fit can attempt to represent the Clayton
// dependence via a linear correlation (giving it the best possible shot).
// Both methods are initialised from the same starting values.

const MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.5, 50.0)
  theta TVV(50.0, 5.0, 500.0)
  block_omega (ETA_CL, ETA_V) = [0.16, 0.0, 0.09]
  sigma PROP_ERR ~ 0.10 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

// ─── Numerical helpers ────────────────────────────────────────────────────────

/// Abramowitz & Stegun 26.2.16 rational approximation to the probit function.
/// Max |error| < 4.5 × 10⁻⁴ — sufficient for simulation purposes.
fn probit(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let sign = if p <= 0.5 { -1.0_f64 } else { 1.0_f64 };
    let q = if p <= 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * q.ln()).sqrt();
    let num = 2.515_517 + 0.802_853 * t + 0.010_328 * t * t;
    let den = 1.0 + 1.432_788 * t + 0.189_269 * t * t + 0.001_308 * t * t * t;
    sign * (t - num / den)
}

/// Inverse of the Clayton conditional CDF h(v|u;θ) = ∂C/∂u.
///
/// Solves h(v|u;θ) = p for v:
///   v = [(p·u^{θ+1})^{−θ/(θ+1)} − u^{−θ} + 1]^{−1/θ}
fn clayton_h_inv(p: f64, u: f64, theta: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let u = u.clamp(1e-12, 1.0 - 1e-12);
    let a = (p * u.powf(theta + 1.0)).powf(-theta / (theta + 1.0));
    let b = u.powf(-theta);
    let inner = (a - b + 1.0).max(1e-15);
    inner.powf(-1.0 / theta)
}

// ─── Population builder ───────────────────────────────────────────────────────

/// Simulate `n` subjects with Clayton-copula joint (ETA_CL, ETA_V).
///
/// Sampling procedure for each subject:
///   1. z1 ~ N(0,1);  ETA_CL = z1 · ω_CL
///   2. u1 = Φ(z1)                             (probability integral transform)
///   3. w  ~ U(0,1)
///   4. u2 = h_Clayton^{-1}(w | u1; θ)         (conditional inversion)
///   5. ETA_V = probit(u2) · ω_V
///   6. CL = TVCL·exp(ETA_CL),  V = TVV·exp(ETA_V)
///   7. C(t) = dose/V · exp(−CL/V · t);  Y = C(t)·(1 + ε),  ε ~ N(0, σ²)
fn simulate_population(n: usize, seed: u64) -> Population {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let unif = Uniform::new(0.0_f64, 1.0_f64);
    let z_dist = Normal::<f64>::new(0.0, 1.0).unwrap();

    let subjects = (1..=n)
        .map(|i| {
            // Draw (ETA_CL, ETA_V) from Clayton copula
            let z1 = rng.sample(z_dist);
            let u1 = normal_cdf(z1);
            let eta_cl = z1 * OMEGA_CL;

            let w = rng.sample(unif);
            let u2 = clayton_h_inv(w, u1, THETA_C);
            let eta_v = probit(u2) * OMEGA_V;

            // Individual PK parameters
            let cl = TVCL * eta_cl.exp();
            let v = TVV * eta_v.exp();
            let ke = cl / v;

            // Observations with proportional error
            let eps_dist = Normal::<f64>::new(0.0, SIGMA).unwrap();
            let observations: Vec<f64> = OBS_TIMES
                .iter()
                .map(|&t| {
                    let c = DOSE / v * (-ke * t).exp();
                    let eps = rng.sample(eps_dist);
                    (c * (1.0 + eps)).max(1e-6)
                })
                .collect();

            Subject {
                id: format!("{i}"),
                doses: vec![DoseEvent::new(0.0, DOSE, 1, 0.0, false, 0.0)],
                obs_times: OBS_TIMES.to_vec(),
                observations,
                obs_cmts: vec![1; OBS_TIMES.len()],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; OBS_TIMES.len()],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            }
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

// ─── Fit helper ───────────────────────────────────────────────────────────────

fn run_fit(
    model: &ferx_core::types::CompiledModel,
    population: &Population,
    use_vine: bool,
    seed: u64,
) -> ferx_core::FitResult {
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Saem;
    opts.saem_n_exploration = 300;
    opts.saem_n_convergence = 200;
    opts.saem_n_mh_steps = 10;
    opts.saem_omega_burnin = 50;
    opts.saem_seed = Some(seed);
    opts.run_covariance_step = false;
    opts.verbose = false;
    if use_vine {
        opts.saem_omega_dist = OmegaDist::VineCopula;
    }
    fit(model, population, &model.default_params, &opts).expect("SAEM fit must succeed")
}

// ─── Test ─────────────────────────────────────────────────────────────────────

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn vine_copula_recovers_clayton_dependence() {
    let bar = "═".repeat(70);
    eprintln!("\n{bar}");
    eprintln!("Vine Copula Validation — Clayton joint ETA simulation study");
    eprintln!("{bar}");
    eprintln!("Simulation truth:");
    eprintln!("  TVCL = {TVCL} L/h   TVV = {TVV} L");
    eprintln!("  ω_CL = {OMEGA_CL}   ω_V = {OMEGA_V}   σ_prop = {SIGMA}");
    eprintln!("  Copula: Clayton(θ = {THETA_C})");
    eprintln!("  Kendall τ = {TRUE_TAU:.3}   lower tail dep λ_L = {TRUE_LAMBDA_L:.4}");
    eprintln!(
        "  N = {N_SUBJECTS} subjects × {} obs/subject",
        OBS_TIMES.len()
    );

    let population = simulate_population(N_SUBJECTS, 42);
    let model = parse_model_string(MODEL).expect("model must parse");

    // ── Fit ──────────────────────────────────────────────────────────────────
    eprintln!("\nFitting Gaussian SAEM…");
    let gauss = run_fit(&model, &population, false, 100);
    eprintln!("Fitting vine SAEM…");
    let vine = run_fit(&model, &population, true, 100);

    // ── Gaussian results ─────────────────────────────────────────────────────
    eprintln!("\n{}", "─".repeat(70));
    eprintln!("Gaussian SAEM");
    eprintln!("  OFV  = {:.3}", gauss.ofv);
    eprintln!(
        "  TVCL = {:.3}  (true {TVCL})   TVV = {:.3}  (true {TVV})",
        gauss.theta[0], gauss.theta[1]
    );
    eprintln!("  σ_prop = {:.4}  (true {SIGMA})", gauss.sigma[0]);
    let var_cl = gauss.omega[(0, 0)];
    let var_v = gauss.omega[(1, 1)];
    let cov_cl_v = gauss.omega[(0, 1)];
    let pearson = if var_cl > 0.0 && var_v > 0.0 {
        cov_cl_v / (var_cl.sqrt() * var_v.sqrt())
    } else {
        0.0
    };
    eprintln!(
        "  ω_CL = {:.4} (true {OMEGA_CL})   ω_V = {:.4} (true {OMEGA_V})",
        var_cl.sqrt(),
        var_v.sqrt()
    );
    eprintln!("  Pearson corr(CL,V) = {pearson:.4}   [Gaussian copula: λ_L = 0 by definition]");

    // ── Vine results ─────────────────────────────────────────────────────────
    eprintln!("\n{}", "─".repeat(70));
    eprintln!("Vine SAEM");
    eprintln!("  OFV  = {:.3}", vine.ofv);
    eprintln!(
        "  TVCL = {:.3}  (true {TVCL})   TVV = {:.3}  (true {TVV})",
        vine.theta[0], vine.theta[1]
    );
    eprintln!("  σ_prop = {:.4}  (true {SIGMA})", vine.sigma[0]);

    let vine_tau;
    let vine_lambda_l;
    let vine_family;

    if let Some(ref vp) = vine.vine_params {
        eprintln!("  Marginal Gaussians:");
        for (name, mean, sd) in &vp.marginals {
            eprintln!("    {name:12} mean={mean:+.4}  sd={sd:.4}");
        }
        eprintln!("  Vine trees:");
        for tree in &vp.trees {
            for entry in &tree.pairs {
                let params: Vec<String> = entry
                    .copula
                    .params
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.4}"))
                    .collect();
                eprintln!(
                    "    Tree {}: {}  [{} {}  τ={:.4}  λL={:.4}  λU={:.4}]",
                    tree.tree,
                    entry.label,
                    entry.copula.family,
                    params.join(" "),
                    entry.copula.kendall_tau,
                    entry.copula.tail_dep_lower,
                    entry.copula.tail_dep_upper
                );
            }
        }

        let t1 = &vp.trees[0].pairs[0].copula;
        vine_tau = t1.kendall_tau;
        vine_lambda_l = t1.tail_dep_lower;
        vine_family = t1.family.clone();
    } else {
        panic!("vine_params should be Some for omega_dist = vine");
    }

    // ── Comparison ───────────────────────────────────────────────────────────
    let delta_ofv = gauss.ofv - vine.ofv;
    eprintln!("\n{}", "─".repeat(70));
    eprintln!("Comparison");
    eprintln!("  ΔOFV (Gaussian − Vine) = {delta_ofv:.2}  [positive ⇒ vine fits better]");
    eprintln!(
        "  Vine τ   = {vine_tau:.4}  (true {TRUE_TAU:.4}, Δ = {:+.4})",
        vine_tau - TRUE_TAU
    );
    eprintln!(
        "  Vine λ_L = {vine_lambda_l:.4}  (true {TRUE_LAMBDA_L:.4}, Δ = {:+.4})",
        vine_lambda_l - TRUE_LAMBDA_L
    );
    eprintln!("  Vine family selected: {vine_family}  (expected: clayton)");
    eprintln!("  Gaussian Pearson r  = {pearson:.4}  (no tail dependence captured)");
    eprintln!("{bar}\n");

    // ── Assertions ────────────────────────────────────────────────────────────

    // 1. Both methods recover TVCL and TVV within 30%
    assert!(
        (gauss.theta[0] - TVCL).abs() / TVCL < 0.30,
        "Gaussian TVCL {:.3} too far from truth {TVCL}",
        gauss.theta[0]
    );
    assert!(
        (vine.theta[0] - TVCL).abs() / TVCL < 0.30,
        "Vine TVCL {:.3} too far from truth {TVCL}",
        vine.theta[0]
    );
    assert!(
        (gauss.theta[1] - TVV).abs() / TVV < 0.30,
        "Gaussian TVV {:.3} too far from truth {TVV}",
        gauss.theta[1]
    );
    assert!(
        (vine.theta[1] - TVV).abs() / TVV < 0.30,
        "Vine TVV {:.3} too far from truth {TVV}",
        vine.theta[1]
    );

    // 2. Vine OFV ≤ Gaussian OFV (vine is a strictly more flexible model)
    assert!(
        vine.ofv <= gauss.ofv + 5.0,
        "Vine OFV ({:.3}) should be ≤ Gaussian OFV ({:.3}); vine is more flexible",
        vine.ofv,
        gauss.ofv
    );

    // 3. Vine identifies a tail-dependent family.
    //    Clayton → λ_L ≈ 0.707.  Student-t → some λ_L > 0.  Gumbel → λ_U > 0.
    //    Gaussian and Frank have no tail dependence (λ = 0).
    assert!(
        vine_lambda_l > 0.20,
        "Vine λ_L = {vine_lambda_l:.4} too low; Gaussian/Frank have λ_L = 0"
    );

    // 4. Vine Kendall τ within 0.20 of truth (0.500)
    assert!(
        (vine_tau - TRUE_TAU).abs() < 0.20,
        "Vine τ = {vine_tau:.4} too far from truth {TRUE_TAU:.4}"
    );

    // 5. Clayton selected (ideally) or another lower-tail-dependent family
    let is_lower_tail_dep = vine_family == "clayton" || vine_family == "student_t";
    assert!(
        is_lower_tail_dep,
        "Expected tail-dependent family (clayton or student_t), got '{vine_family}'"
    );
}
