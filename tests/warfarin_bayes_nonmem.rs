//! NONMEM 7.5.1 `METHOD=BAYES` cross-check for ferx's Bayesian estimator (#380).
//!
//! Runs ferx `method = bayes` (Gibbs-within-HMC) standalone on warfarin and
//! asserts the posterior means match NONMEM's `METHOD=BAYES` reference on the
//! same model/data. Bayes is parity-targeted at NONMEM BAYES, so the anchor is
//! NONMEM itself (CLAUDE.md cross-engine validation for a new estimator).
//!
//! Gated behind `slow-tests` (runs a multi-thousand-sweep MCMC); skipped in the
//! default PR job, run nightly via `slow-tests.yml`.
//!
//! ## Reference values
//!
//! From `tests/nonmem/warfarin_bayes.ctl` run on `data/warfarin.csv` with
//! NONMEM 7.5.1 (`METHOD=BAYES INTERACTION NBURN=1000 NITER=2000 SEED=1`,
//! MU-referenced, diffuse NWPRI priors), posterior means from the `.ext`
//! `-1000000000` row:
//!
//! | Parameter | NONMEM BAYES | ferx Bayes |
//! |-----------|-------------:|-----------:|
//! | TVCL      | 0.1345       | 0.1329     |
//! | TVV       | 7.759        | 7.744      |
//! | TVKA      | 0.859        | 0.828      |
//! | σ (SD)    | 0.01106      | 0.0107     |
//! | ω²(CL)    | 0.0524       | 0.0377     |
//! | ω²(V)     | 0.0197       | 0.0137     |
//! | ω²(KA)    | 0.439        | 0.369      |
//!
//! The **population means (θ) and residual error agree to a few percent** — the
//! primary validation. NONMEM reports `SIGMA` as the variance of `EPS`; ferx
//! parameterises proportional error as `Var = (f·σ)²` and reports σ as the SD,
//! so the comparison is `ferx σ ≈ √(NONMEM SIGMA)`.
//!
//! The Ω posterior means are intentionally loosely banded: both engines' Bayes
//! Ω means sit above the FOCEI MLE (inverse-Wishart posterior-mean bias at the
//! N=10 extract), and the residual gap between the two reflects differing
//! inverse-Wishart prior specs (ferx `IW(n_eta+2, Ω₀)` vs the NWPRI
//! `$OMEGAPD`/`$OMEGAP` used in the .ctl), not a sampler discrepancy.

use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 METHOD=BAYES posterior means (see module docstring / .ctl).
const NM_TVCL: f64 = 0.13454;
const NM_TVV: f64 = 7.75865;
const NM_TVKA: f64 = 0.85908;
const NM_SIGMA_SD: f64 = 0.011062; // sqrt(SIGMA(1,1) = 1.22364e-4)
const NM_OMEGA_CL: f64 = 0.05243;
const NM_OMEGA_V: f64 = 0.01974;
const NM_OMEGA_KA: f64 = 0.43903;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-8)
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ferx_bayes_matches_nonmem_bayes_on_warfarin() {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin model must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");

    let mut opts = FitOptions::default();
    opts.method = EstimationMethod::Bayes;
    opts.run_covariance_step = false;
    opts.bayes_warmup = 1000;
    opts.bayes_iters = 2000;
    opts.bayes_chains = 2;
    opts.bayes_seed = Some(1);
    opts.saem_n_mh_steps = 10;
    let r = fit(&model, &population, &model.default_params, &opts).expect("Bayes fit must succeed");

    // Chains must have mixed.
    let bayes = r.bayes.as_ref().expect("BayesResult present");
    assert!(
        bayes.max_rhat < 1.05,
        "chains did not mix: max R-hat = {}",
        bayes.max_rhat
    );

    // Population means: CL/V well-identified (8%), KA poorly-identified (15%).
    assert!(
        rel(r.theta[0], NM_TVCL) < 0.08,
        "TVCL {} vs NM {NM_TVCL}",
        r.theta[0]
    );
    assert!(
        rel(r.theta[1], NM_TVV) < 0.08,
        "TVV {} vs NM {NM_TVV}",
        r.theta[1]
    );
    assert!(
        rel(r.theta[2], NM_TVKA) < 0.15,
        "TVKA {} vs NM {NM_TVKA}",
        r.theta[2]
    );

    // Residual error: ferx σ (SD) ≈ √(NONMEM SIGMA variance).
    assert!(
        rel(r.sigma[0], NM_SIGMA_SD) < 0.15,
        "sigma_SD {} vs NM {NM_SIGMA_SD}",
        r.sigma[0]
    );

    // Ω posterior means: loosely banded (prior-spec sensitive; see docstring).
    // Assert same order of magnitude (within a factor of ~2) of the NONMEM means.
    assert!(
        rel(r.omega[(0, 0)], NM_OMEGA_CL) < 0.6,
        "omega_CL {}",
        r.omega[(0, 0)]
    );
    assert!(
        rel(r.omega[(1, 1)], NM_OMEGA_V) < 0.6,
        "omega_V {}",
        r.omega[(1, 1)]
    );
    assert!(
        rel(r.omega[(2, 2)], NM_OMEGA_KA) < 0.6,
        "omega_KA {}",
        r.omega[(2, 2)]
    );
}
