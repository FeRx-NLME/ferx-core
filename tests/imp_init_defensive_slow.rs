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
//! This test pins two things:
//!
//! 1. **Default-config recovery (the #528 core).** Under the realistic default
//!    options (ISCALE pilot search on), the mixture fit recovers TVV/TVCL near the
//!    truth with a finite OFV and IS −2logL — the IMP phase does not walk the
//!    population parameters to nonsense. The quantitative NONMEM comparison
//!    (`run14`, SAEM→IMP, OFV −249.23) lives with the cross-repo
//!    `ferx-testdata/thioguanine_mmc` model and is reported in the PR.
//!
//! 2. **Mixture-vs-legacy discrimination (#961).** A subtlety surfaced after #528:
//!    on this fixture the *default* legacy (`alpha = 0`) sampler no longer collapses
//!    (V ≈ 23.4, essentially the mixture's estimate). The runaway is not gone — it
//!    is masked by a *second, general* safeguard, the per-subject **ISCALE scalar
//!    pilot search**, which isotropically rebroadens a too-narrow proposal until
//!    the ESS recovers. (Two upstream changes conspired: the SAEM `iiv_on_ruv`
//!    fixes #895/#904 now hand the IMP phase a saner start, and the ISCALE search
//!    then finishes the rescue.) ISCALE and the defensive mixture overlap, so with
//!    ISCALE on the mixture reads as redundant *here* — but it is not redundant in
//!    general: ISCALE is a single isotropic scale, whereas the `N(0, Ω)` mixture
//!    component bounds every weight regardless of how badly the narrow proposal is
//!    centred or shaped. To restore genuine discrimination we **pin ISCALE to 1.0**
//!    (its documented "disabled" setting), removing the overlapping rescue. With it
//!    off the weakly-identified-V collapse returns: legacy runs V/CL away
//!    (V ≳ 14× truth), the mixture still recovers them. That is the contrast this
//!    test asserts. The fast branch-coverage smoke test is
//!    `importance_sampling_api::imp_defensive_mixture_runs_for_both_alpha_branches`.
//!
//! Data is simulated from the model (fixed seed) so the test is self-contained.

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
                // Non-Continuous outcomes (Event under `survival`, plus the
                // unconditional Category/Count) never occur for this Gaussian
                // model; ignore them.
                _ => {}
            }
        }
    }
    pop
}

fn saem_imp_opts(defensive_alpha: f64, pin_iscale: bool) -> FitOptions {
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
    if pin_iscale {
        // Disable the per-subject ISCALE scalar pilot search (`iscale_min ==
        // iscale_max == 1.0` is its documented "off" setting). That search is a
        // *separate*, general safeguard: it isotropically rebroadens a too-narrow
        // proposal until the ESS recovers, and on this fixture it alone rescues the
        // weakly-identified-V collapse — making the defensive mixture look
        // redundant. Pinning it isolates what `imp_defensive_alpha` uniquely
        // contributes: bounding the importance weights via the broad `N(0, Ω)`
        // component when the narrow proposal is badly matched (#961).
        opts.iscale_min = 1.0;
        opts.iscale_max = 1.0;
    }
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

    // --- Part 1: default config (ISCALE pilot search on). The mixture recovers
    // θ near the truth — the #528 core: the IMP phase stays identifiable.
    let with_mix = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_imp_opts(0.1, false),
    )
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

    // Default-config legacy (alpha = 0, ISCALE pilot search on) — the realistic
    // shipped path. Post-#528 the ISCALE search alone keeps this bounded; assert it
    // stays finite and near truth so an ISCALE-rescue regression trips here (the
    // guard the pinned-ISCALE Part 2 below deliberately removes).
    let legacy_default = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_imp_opts(0.0, false),
    )
    .expect("legacy imp (default ISCALE) must produce a fit");
    let v_legacy_default = legacy_default.theta[1];
    assert!(
        legacy_default.ofv.is_finite()
            && v_legacy_default.is_finite()
            && v_legacy_default < 2.0 * TRUE_V,
        "default-config legacy IMP must stay bounded (ISCALE rescue intact): V = \
         {v_legacy_default}, OFV = {}",
        legacy_default.ofv
    );

    // --- Part 2: mixture-vs-legacy discrimination with the ISCALE scalar pilot
    // search pinned off (#961). ISCALE is a *separate*, general safeguard that on
    // this fixture alone rescues the collapse and masks the mixture's value; pin it
    // to 1.0 to remove that overlapping rescue and re-expose the weakly-identified-V
    // weight collapse the defensive mixture was built to fix (#528).
    //
    //   * legacy single-proposal (alpha = 0): weights collapse on the baseline
    //     subjects, the importance-weighted M-step is hijacked, and V/CL run away.
    //   * defensive mixture (alpha = 0.1): the broad `N(0, Ω)` component bounds
    //     every weight, so no single collapsed-weight subject can dominate and θ
    //     stays identifiable.
    let no_mix = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_imp_opts(0.0, true),
    )
    .expect("legacy imp still returns a (bad) fit");
    let with_mix_pinned = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_imp_opts(0.1, true),
    )
    .expect("mixture imp (ISCALE pinned) must produce a fit");
    let v0 = no_mix.theta[1];
    let vp = with_mix_pinned.theta[1];
    // Legacy blows V up well past 2× truth (empirically ≈ 14×) once ISCALE cannot
    // rebroaden the collapsed proposal. If this fails, the collapse is no longer
    // reproducible even with ISCALE off and the discrimination guard is void —
    // reopen #961.
    assert!(
        v0 > 2.0 * TRUE_V,
        "legacy sampler (ISCALE pinned) is expected to blow V up (>2× truth); got \
         {v0} — if this fails the fixture no longer reproduces the collapse"
    );
    // The mixture (same pinned config) still recovers V near the truth. Combined
    // with the `v0 > 2·TRUE_V` legacy blow-up above, this *is* the discrimination:
    // |vp−20| < 5 while |v0−20| > 20, an unambiguous mixture-over-legacy win. (No
    // separate ratio assert — it would be implied by these two and never fail
    // independently.)
    assert!(
        (vp - TRUE_V).abs() / TRUE_V < 0.25,
        "defensive mixture (ISCALE pinned) should still recover TVV near {TRUE_V}, \
         got {vp} (legacy blew up to {v0})"
    );
}
