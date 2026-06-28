//! Tier 3 (slow) regression for issue #528: an analytical `[initial_conditions]`
//! model must fit through `method = [saem, imp]` without the importance-sampling
//! phase walking the population parameters to nonsense.
//!
//! The mechanism: for a baseline subject the closed-form initial concentration is
//! `A₀/V · e^{−kt} = CONC0·V/V · e^{−kt} = CONC0·e^{−kt}` — **V cancels in the
//! amplitude**, so the baseline data constrain `ETA_V` only weakly (through the
//! decay rate `k = CL/V`). The single-proposal importance sampler then collapses
//! its weights on those subjects, and the importance-weighted M-step is hijacked
//! by the surviving samples — pushing θ far from the truth (in the NONMEM `run14`
//! report all the way to the bounds with `OFV ≈ 1e35`).
//!
//! The defensive-mixture proposal (`imp_defensive_alpha`, opt-in; this test sets
//! it explicitly) draws a
//! fraction of samples from the prior `N(0, Ω)` and scores every sample under the
//! resulting mixture density. Because the prior covers the conditional posterior,
//! this **bounds the importance weights**, so no single collapsed-weight subject
//! can dominate the M-step. It does not necessarily restore a high raw ESS, but
//! it keeps the population estimates identifiable — which is what this test pins.
//!
//! The contrast below is decisive and deterministic: with the legacy sampler
//! (`alpha = 0`) the recovered V/CL run away to ~16× their true values; with the
//! default mixture they land within a few percent. Data is simulated from the
//! model (fixed seed) so the test is self-contained; the quantitative NONMEM
//! comparison (`run14`, SAEM→IMP, OFV −249.23) lives with the cross-repo
//! `ferx-testdata/thioguanine_mmc` model and is reported in the PR.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{DoseEvent, Population, SimOutcome};
use ferx_core::{fit, simulate_with_seed, EstimationMethod, FitOptions};

mod common;

// True values: TVCL = 3, TVV = 20, TVKA = 1.
const TRUE_CL: f64 = 3.0;
const TRUE_V: f64 = 20.0;

// Mirrors the NONMEM `run14` structure that triggered the divergence: 1-cpt oral
// with an analytical baseline, log-additive (LTBS) residual error, and IIV on the
// residual error. `ETA_V` enters only through V, which cancels in the baseline
// amplitude — the weakly-identified ridge the IMP sampler chokes on.
const MODEL_SRC: &str = r"
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL  ~ 0.09
  omega ETA_V   ~ 0.09
  omega ETA_RUV ~ 0.05

  sigma ADD_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[initial_conditions]
  # Baseline already present in central at t=0: amount = CONC0 * V (NONMEM A_0(2)=CONC0*S2).
  init(central) = CONC0 * V

[error_model]
  DV ~ log_additive(ADD_ERR)
  iiv_on_ruv = ETA_RUV
";

/// Template population: half the subjects carry a pre-dose baseline with only a
/// trace dose (`CONC0 > 0`, observed on the decay tail — the weakly-identified-V
/// case), the rest are ordinary oral-dose subjects with informative absorption.
fn template() -> Population {
    let mut subjects = Vec::new();
    for i in 0..24 {
        let baseline = i % 2 == 0;
        let (conc0, amt, obs_times) = if baseline {
            (20.0, 0.01, vec![2.0, 8.0, 24.0])
        } else {
            (0.0, 100.0, vec![0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
        };
        let n = obs_times.len();
        let doses = vec![DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)];
        let mut s = common::subject(
            &format!("{}", i + 1),
            doses,
            obs_times,
            vec![0.0; n],
            vec![2; n],
        );
        s.covariates.insert("CONC0".to_string(), conc0);
        subjects.push(s);
    }
    Population {
        covariate_names: vec!["CONC0".to_string()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
        subjects,
    }
}

/// Replace each subject's observations with one simulated replicate (fixed seed).
fn simulate_dv(
    model: &ferx_core::types::CompiledModel,
    template: &Population,
    params: &ferx_core::types::ModelParameters,
) -> Population {
    let sims = simulate_with_seed(model, template, params, 1, 528);
    let mut pop = template.clone();
    // Sims are emitted in (subject, observation) order; refill each subject's
    // observation vector in that order.
    let mut iter = sims.into_iter();
    for subj in pop.subjects.iter_mut() {
        for obs in subj.observations.iter_mut() {
            let row = iter.next().expect("one sim row per scheduled observation");
            match row.outcome {
                SimOutcome::Continuous { value } => *obs = value,
                #[cfg(feature = "survival")]
                _ => {}
            }
        }
    }
    pop
}

fn saem_imp_opts(defensive_alpha: f64) -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Imp];
    opts.saem_n_exploration = 150;
    opts.saem_n_convergence = 150;
    opts.imp_iterations = 30;
    opts.imp_samples = 500;
    opts.imp_seed = Some(7);
    opts.saem_seed = Some(7);
    opts.imp_defensive_alpha = defensive_alpha;
    opts
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn saem_imp_on_analytical_init_recovers_parameters_with_defensive_mixture() {
    let parsed = parse_full_model(MODEL_SRC).expect("init model parses");
    let model = parsed.model;
    assert_eq!(
        model.analytical_init.len(),
        1,
        "[initial_conditions] must populate analytical_init"
    );
    let pop = simulate_dv(&model, &template(), &model.default_params);

    // Defensive mixture enabled (alpha = 0.1): θ is recovered near the truth.
    let with_mix = fit(&model, &pop, &model.default_params, &saem_imp_opts(0.1))
        .expect("saem → imp on the init model must produce a fit");
    assert!(with_mix.converged, "defensive-mixture fit must converge");
    assert!(
        with_mix.ofv.is_finite() && with_mix.ofv.abs() < 1e6,
        "OFV must be sane with the mixture, got {}",
        with_mix.ofv
    );
    let v = with_mix.theta[1];
    let cl = with_mix.theta[0];
    assert!(
        (v - TRUE_V).abs() / TRUE_V < 0.25,
        "TVV should be recovered near {TRUE_V}, got {v}"
    );
    assert!(
        (cl - TRUE_CL).abs() / TRUE_CL < 0.25,
        "TVCL should be recovered near {TRUE_CL}, got {cl}"
    );
    let imp = with_mix
        .importance_sampling
        .as_ref()
        .expect("importance_sampling field populated for a [.., imp] chain");
    assert!(
        imp.minus2_log_likelihood.is_finite(),
        "IS −2logL must be finite, got {}",
        imp.minus2_log_likelihood
    );

    // Legacy single-proposal sampler (alpha = 0) on identical data: the M-step is
    // hijacked by the collapsed-weight baseline subjects and V/CL run away. This
    // is the behaviour the fix removes; asserting it makes the test a genuine
    // regression rather than a smoke test.
    let no_mix = fit(&model, &pop, &model.default_params, &saem_imp_opts(0.0))
        .expect("legacy imp still returns a (bad) fit");
    let v0 = no_mix.theta[1];
    assert!(
        v0 > 2.0 * TRUE_V,
        "legacy sampler is expected to blow V up (>2× truth); got {v0} — if this \
         fails the synthetic data no longer reproduces the collapse and the test \
         above is no longer guarding the fix"
    );
    // The mixture must be a large, unambiguous improvement on the recovered V.
    assert!(
        (v - TRUE_V).abs() < (v0 - TRUE_V).abs() / 3.0,
        "defensive mixture should recover V far better than legacy: mix {v} vs legacy {v0}"
    );
}
