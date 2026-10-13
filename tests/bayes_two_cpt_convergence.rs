//! `method = bayes` converges on the two-compartment oral examples (#1626).
//!
//! Before #1626 neither converged at any chain length: on the base model one
//! subject's η fell into a negligible-mass KA ↔ distribution flip-flop mode in
//! warmup and the random-walk η kernel never left it (max R̂ ≈ 1.19 at 1,000,
//! 6,000 and 20,000 draws per chain), and on the covariate model the
//! random-walk θ block drew the exponents below their bounds while barely
//! moving them (max R̂ 2.7 → 7.2 as the chains lengthened).
//!
//! The models are the ferx-r examples `two_cpt_oral_base` and
//! `two_cpt_oral_cov` (`ferx-r/inst/examples/models/`), inlined so this file
//! pins the exact text the issue was reported on; both read
//! `data/two_cpt_oral_cov.csv`, which is byte-identical to ferx-r's copy.
//!
//! There is no NONMEM run for a Bayes sampler's mixing, so the external
//! reference is the FOCEI optimum on the same model and data (Linux, this
//! branch, `FitOptions::default()` with `method = focei`, no covariance step):
//! the posterior means of θ and σ must sit within a quarter of a posterior SD
//! of it, Ω within half (its posterior mean runs above the MLE at N = 30 — the
//! inverse-Wishart posterior-mean bias `docs/estimation/bayes.qmd` describes).
//!
//! Gated: a 4-chain, 2,000 + 2,000 sweep fit per model (6–8 s on 16 cores).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const BASE_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 1.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 1.0, 500.0)
  theta TVKA(1.2, 0.01, 10.0)

  omega ETA_CL ~ 0.10
  omega ETA_V1 ~ 0.10
  omega ETA_Q  ~ 0.05
  omega ETA_V2 ~ 0.05
  omega ETA_KA ~ 0.15

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

const COV_MODEL: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 100.0)
  theta TVV1(50.0, 1.0, 500.0)
  theta TVQ(10.0, 0.1, 100.0)
  theta TVV2(100.0, 1.0, 500.0)
  theta TVKA(1.2, 0.01, 10.0)
  theta THETA_WT(0.75, 0.01, 5.0)
  theta THETA_CRCL(0.50, 0.01, 5.0)

  omega ETA_CL ~ 0.10
  omega ETA_V1 ~ 0.10
  omega ETA_Q  ~ 0.05
  omega ETA_V2 ~ 0.05
  omega ETA_KA ~ 0.15

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * (WT / 70)^THETA_WT * (CRCL / 100)^THETA_CRCL * exp(ETA_CL)
  V1 = TVV1 * (WT / 70)^THETA_WT * exp(ETA_V1)
  Q  = TVQ  * exp(ETA_Q)
  V2 = TVV2 * exp(ETA_V2)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// FOCEI optimum, base model (OFV −1185.462).
const FOCEI_BASE: &[(&str, f64)] = &[
    ("TVCL", 4.4219),
    ("TVV1", 48.3170),
    ("TVQ", 9.5698),
    ("TVV2", 94.2201),
    ("TVKA", 1.2572),
    ("OMEGA(1,1)", 0.1563),
    ("OMEGA(2,2)", 0.1173),
    ("OMEGA(3,3)", 0.0518),
    ("OMEGA(4,4)", 0.0382),
    ("OMEGA(5,5)", 0.1731),
    ("PROP_ERR", 0.019597),
];

/// FOCEI optimum, covariate model (OFV −1199.326).
const FOCEI_COV: &[(&str, f64)] = &[
    ("TVCL", 4.9478),
    ("TVV1", 48.5875),
    ("TVQ", 9.5769),
    ("TVV2", 94.1733),
    ("TVKA", 1.2559),
    ("THETA_WT", 0.6535),
    ("THETA_CRCL", 0.5646),
    ("OMEGA(1,1)", 0.1075),
    ("OMEGA(2,2)", 0.1063),
    ("OMEGA(3,3)", 0.0512),
    ("OMEGA(4,4)", 0.0379),
    ("OMEGA(5,5)", 0.1754),
    ("PROP_ERR", 0.019619),
];

fn run(src: &str, focei: &[(&str, f64)]) -> ferx_core::types::BayesResult {
    let model = parse_model_string(src).expect("model parses");
    let pop =
        read_nonmem_csv(Path::new("data/two_cpt_oral_cov.csv"), None, None).expect("data loads");
    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Bayes;
    opts.run_covariance_step = false;
    opts.bayes_warmup = 2000;
    opts.bayes_iters = 2000;
    // Seed 3 is one on which, without the EBE start, a base-model chain
    // falls into the flip-flop mode (TVV1 0.59 posterior SD from FOCEI,
    // measured), so this fixture sees that regression; the default seed does
    // not happen to.
    opts.bayes_seed = Some(3);
    let res = fit(&model, &pop, &model.default_params, &opts).expect("bayes runs");
    let b = res.bayes.clone().expect("BayesResult present");
    let (mut worst_theta, mut worst_omega) = (0.0_f64, 0.0_f64);
    for &(name, want) in focei {
        let s = b
            .summaries
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name} summarised"));
        assert!(
            s.mean.is_finite() && s.sd.is_finite() && s.sd > 0.0,
            "{name}"
        );
        let in_sd = (s.mean - want).abs() / s.sd;
        // Measured (Linux, seed 3): θ and σ ≤ 0.106 (base) / 0.102 (cov)
        // posterior SD from FOCEI, Ω ≤ 0.25 / 0.30 (above the MLE, as
        // expected). The pre-#1626 sampler put TVV1 0.37 SD off on the base
        // model and THETA_WT 1.86 SD off on the covariate model.
        let band = if name.starts_with("OMEGA") { 0.5 } else { 0.25 };
        assert!(
            in_sd < band,
            "{name}: posterior mean {} is {in_sd:.2} posterior SD from FOCEI {want}",
            s.mean
        );
        if name.starts_with("OMEGA") {
            worst_omega = worst_omega.max(in_sd);
        } else {
            worst_theta = worst_theta.max(in_sd);
        }
    }
    eprintln!(
        "max_rhat {:.4}  min ess_bulk {:.0}  |mean − FOCEI| / sd: θ,σ {:.3}  Ω {:.3}",
        b.max_rhat,
        b.summaries
            .iter()
            .map(|s| s.ess_bulk)
            .fold(f64::INFINITY, f64::min),
        worst_theta,
        worst_omega
    );
    b
}

fn assert_converged(b: &ferx_core::types::BayesResult) {
    assert!(b.max_rhat.is_finite(), "max_rhat {}", b.max_rhat);
    // Measured (Linux, seed 3): 1.003 (base), 1.006 (cov); over the default
    // seed and seeds 1–3, ≤ 1.007. Pre-#1626: 1.19 (base), 2.7–7.2 (cov), at
    // any length.
    assert!(b.max_rhat < 1.01, "max split-R-hat {}", b.max_rhat);
    for s in &b.summaries {
        assert!(s.ess_bulk.is_finite(), "{} ess", s.name);
        // Measured minimum (seed 3): 1,029 (base), 919 (cov) of 8,000 draws,
        // both PROP_ERR; 646 on the default seed. Pre-#1626: 7 and 3.
        assert!(s.ess_bulk > 400.0, "{}: bulk ESS {}", s.name, s.ess_bulk);
    }
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bayes_converges_on_two_cpt_oral_base() {
    let b = run(BASE_MODEL, FOCEI_BASE);
    assert_converged(&b);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bayes_converges_on_two_cpt_oral_cov() {
    let b = run(COV_MODEL, FOCEI_COV);
    assert_converged(&b);
    // The exponents stay inside their declared bounds (0.01, 5.0); before
    // #1626 their 2.5% quantiles reached 5e-9 and 1e-8.
    for name in ["THETA_WT", "THETA_CRCL"] {
        let s = b.summaries.iter().find(|s| s.name == name).expect(name);
        assert!(
            s.q025 >= 0.01 && s.q975 <= 5.0,
            "{name}: [{}, {}]",
            s.q025,
            s.q975
        );
    }
}
