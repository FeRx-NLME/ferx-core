//! Stan anchor for `method = bayes` on a non-Gaussian random effect (#1716, review of
//! #1721, finding 4).
//!
//! `stan_anchor/warfarin_gamma_cl.stan` writes, prior for prior, the posterior ferx's
//! Bayes sampler targets for the model below: a Gamma-distributed clearance
//! (`density gamma`), Gaussian V and KA, a proportional error, ferx's default priors (log θ
//! ~ N(log θ₀, 10²) within the bounds, log σ ~ N(log σ₀, 10²), each ω ~ IG(1.5, ω₀/2)).
//! Stan's NUTS and ferx's Gibbs-within-Metropolis are two samplers of one distribution, so
//! their posterior means agree within the combined Monte Carlo error. Nothing inside ferx
//! sees the density block's Metropolis step as a whole — its log-Cholesky proposal, the
//! Jacobian, the IG marginal and the density's own score together — and this does.
//!
//! The Gaussian twin (`CL = TVCL·exp(η)`, also in the JSON) has a different posterior, and
//! the test asserts ferx lands measurably far from it: a sampler that ignored the density
//! and scored a Gaussian prior would pass the first comparison only if the two posteriors
//! coincided, and they do not.
//!
//! TVV ≥ 1 and TVKA ≥ 0.1 exclude the one-compartment oral flip-flop mode (KA ↔ CL/V),
//! where Stan's chains split between the two modes at wider bounds (R̂ 1.8); both samplers
//! target the same truncated posterior.
//!
//! The references are committed (`stan_anchor/results/warfarin_gamma_cl.json`, regenerated
//! by `stan_anchor/run_stan.py`). Tier 3: long chains, so `slow-tests`.

use std::collections::HashMap;
use std::path::Path;

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{EstimationMethod, FitOptions};

const MODEL: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 1.0, 500.0)
  theta TVKA(1.5, 0.1, 50.0)
  omega ETA_CL ~ 0.09 density gamma
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * ETA_CL
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

/// ferx's summary name → the Stan generated quantity.
const PARAMS: [(&str, &str); 7] = [
    ("TVCL", "TVCL"),
    ("TVV", "TVV"),
    ("TVKA", "TVKA"),
    ("OMEGA(1,1)", "OMEGA_CL"),
    ("OMEGA(2,2)", "OMEGA_V"),
    ("OMEGA(3,3)", "OMEGA_KA"),
    ("PROP_ERR", "PROP_ERR"),
];

#[derive(serde::Deserialize)]
struct Ref {
    mean: f64,
    mcse: f64,
    rhat: f64,
}

#[derive(serde::Deserialize)]
struct Refs {
    density: HashMap<String, Ref>,
    gaussian_twin: HashMap<String, Ref>,
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn bayes_on_a_gamma_clearance_matches_stan() {
    let refs: Refs = serde_json::from_str(
        &std::fs::read_to_string(Path::new("stan_anchor/results/warfarin_gamma_cl.json"))
            .expect("the Stan reference is committed"),
    )
    .expect("the Stan reference parses");
    let parsed = parse_full_model(MODEL).expect("the model parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &[],
    )
    .expect("warfarin data reads");
    let opts = FitOptions {
        method: EstimationMethod::Bayes,
        bayes_chains: 4,
        bayes_warmup: BAYES_WARMUP,
        bayes_iters: BAYES_ITERS,
        bayes_seed: Some(1716),
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let r = ferx_core::fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
        .expect("Bayes runs");
    let bayes = r.bayes.as_ref().expect("a Bayes result");
    let mut worst = (0.0f64, "");
    let mut worst_rhat = 0.0f64;
    let mut farthest_from_twin = (0.0f64, "");
    for (ferx_name, stan_name) in PARAMS {
        let s = bayes
            .summaries
            .iter()
            .find(|s| s.name == ferx_name)
            .unwrap_or_else(|| panic!("ferx reports {ferx_name}"));
        let stan = &refs.density[stan_name];
        let twin = &refs.gaussian_twin[stan_name];
        assert!(
            s.mean.is_finite()
                && s.mcse.is_finite()
                && s.rhat.is_finite()
                && stan.mean.is_finite()
                && stan.mcse.is_finite(),
            "{ferx_name}: ferx {} ± {}, Stan {} ± {}",
            s.mean,
            s.mcse,
            stan.mean,
            stan.mcse
        );
        worst_rhat = worst_rhat.max(s.rhat);
        let se = (s.mcse * s.mcse + stan.mcse * stan.mcse).sqrt();
        let z = (s.mean - stan.mean).abs() / se;
        assert!(
            stan.rhat < 1.01,
            "the Stan reference for {stan_name} has not converged"
        );
        // The twin's own V/KA posterior is bimodal (R̂ 1.4–1.5 at these bounds); only its
        // converged parameters serve as the "must differ" reference.
        let zt = if twin.rhat < 1.01 {
            (s.mean - twin.mean).abs() / (s.mcse * s.mcse + twin.mcse * twin.mcse).sqrt()
        } else {
            0.0
        };
        eprintln!(
            "{ferx_name}: ferx {:.5} ± {:.5} (R̂ {:.3}, ESS {:.0}) | Stan {:.5} ± {:.5} (z {z:.2}) | Gaussian twin {:.5} (z {zt:.1})",
            s.mean, s.mcse, s.rhat, s.ess_bulk, stan.mean, stan.mcse, twin.mean
        );
        if z > worst.0 {
            worst = (z, ferx_name);
        }
        if zt > farthest_from_twin.0 {
            farthest_from_twin = (zt, ferx_name);
        }
    }
    assert!(
        worst_rhat < 1.01,
        "ferx's chains disagree: worst R̂ {worst_rhat:.3}"
    );
    assert!(worst.0 < Z_MAX, "worst {:.2} SE ({})", worst.0, worst.1);
    assert!(
        farthest_from_twin.0 > TWIN_Z_MIN,
        "ferx is within {:.1} SE of the Gaussian twin everywhere: the anchor does not see \
         the density",
        farthest_from_twin.0
    );
}
// Measured (seed 1716, 4 × 10 000 draws): every R̂ ≤ 1.001, ESS 4 177–27 829, ferx vs
// Stan within 2.05 SE at worst (ω_V; 0.18–1.74 elsewhere: the spread of 7 draws of
// |N(0, 1)|), so `Z_MAX` = 4 is a 4σ test per parameter. ferx lands 11.3 SE from the
// Gaussian twin's TVCL (the gamma pins the mean, the twin the median), so `TWIN_Z_MIN` = 6
// leaves room for the z's own Monte Carlo spread. Before the θ–z shift move, the same run
// gave R̂ 2.4–3.3 and ESS 3–11: TVCL stuck at its 0.2 start against Stan's 0.135.
const BAYES_WARMUP: usize = 2000;
const BAYES_ITERS: usize = 10000;
const Z_MAX: f64 = 4.0;
const TWIN_Z_MIN: f64 = 6.0;
