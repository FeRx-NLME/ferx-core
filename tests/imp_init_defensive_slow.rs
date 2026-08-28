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
//! 2. **Mixture-vs-legacy discrimination (#961, re-based in #1113).** The
//!    discrimination is the *negative* half of the test: legacy
//!    (`imp_defensive_alpha = 0`) must run the population parameters away on a
//!    fixture the mixture recovers, or the fixture has stopped reproducing the
//!    defect the option exists for.
//!
//!    Two safeguards have masked that contrast in turn, and the fixture has been
//!    re-based once for each:
//!
//!    * #961 — the per-subject **ISCALE scalar pilot search** (a *separate*,
//!      general rescue that isotropically rebroadens a too-narrow proposal until
//!      the ESS recovers) alone kept legacy bounded on the original fixture. That
//!      was worked around by **pinning ISCALE to 1.0** (its documented "disabled"
//!      setting) for the discrimination arm.
//!    * #1113 — #1017's LTBS prediction floor (`CompiledModel::floor_prediction`)
//!      then removed the *fabricated* residuals that were doing much of the work:
//!      the old `f.max(1e-12)` clamp was replacing a legitimately negative
//!      log-prediction (this fixture's error model is `log_additive`, so `f` is
//!      `log(c)` and is negative for any concentration below one unit) with ~0.
//!      With that fixed, legacy landed `V = 19.84` — 1 % off truth — and the
//!      contrast was gone even with ISCALE pinned. The floor fix is
//!      unconditionally correct and stays.
//!
//!    So the fixture itself was re-based on the mechanism rather than on either
//!    safeguard: **three quarters** of the subjects are now the weakly-identified
//!    baseline case (was one half) and `ETA_V`'s prior is wider (variance 0.25,
//!    was 0.09), so the single-proposal sampler faces the collapse #528 reported
//!    without help from a clamp artefact. On that fixture the runaway returns in
//!    the **shipped default config** — ISCALE on — which is a strictly stronger
//!    statement than #961's pinned-only contrast: the defensive mixture is not
//!    redundant with the ISCALE search, it is what keeps this fit identifiable.
//!    The pinned-ISCALE arm is kept as a second, independent reading.
//!
//!    The mixture's answer is anchored *within the test* against a deterministic
//!    **FOCEI** fit of the same model and data (`CL 2.618, V 16.963`): the
//!    mixture must land on that optimum, not merely inside a band around the
//!    simulation truth — one noisy realization's MLE sits ~15% below truth here,
//!    and "near truth" alone would not distinguish a real fit from a fit stuck at
//!    its starting values.
//!
//!    The failure guard is deliberately **relative** and taken over *both* `CL`
//!    and `V`: the runaway's magnitude and which coordinate absorbs it are
//!    chaotic and libm-dependent (a seed sweep put legacy anywhere from
//!    `V +160%` with `CL +684%` to `V +9%` with `CL +1903%`), so an absolute
//!    multiple-of-truth threshold on one coordinate reads as a platform-dependent
//!    failure rather than a lost contrast (#751). Across the five seeds swept,
//!    legacy's worse coordinate was off by at least 90% in every default-config
//!    run and 160% in every pinned run, while the mixture stayed at 13-15% (the
//!    FOCEI optimum) throughout. The fast branch-coverage smoke test is
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
  omega ETA_V   ~ 0.25
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

/// Template population: **three quarters** of the subjects carry a pre-dose
/// baseline with only a trace dose (`CONC0 > 0`, observed on the decay tail — the
/// weakly-identified-V case), the rest are ordinary oral-dose subjects with
/// informative absorption. The 3:1 split (was 1:1) is what re-exposes the #528
/// weight collapse now that #1017's LTBS prediction floor no longer fabricates
/// residuals on this log-scale model — see the module docs.
fn template() -> Population {
    let mut subjects = Vec::new();
    for i in 0..24 {
        let baseline = i % 4 != 0;
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

/// Worst relative error over the two identifiable typical values (`TVCL`, `TVV`)
/// against a reference pair. Which of the two absorbs a collapsed-weight runaway
/// is chaotic — a seed sweep produced legacy fits that ran `V` away with `CL`
/// almost intact and fits that did the reverse — so every guard below reads the
/// worse coordinate rather than `V` alone (#1113).
fn max_rel_err(theta: &[f64], ref_cl: f64, ref_v: f64) -> f64 {
    ((theta[0] - ref_cl).abs() / ref_cl).max((theta[1] - ref_v).abs() / ref_v)
}

/// Deterministic FOCEI reference fit of the same model and data: the optimum the
/// stochastic SAEM→IMP chain is expected to land on. Anchoring the mixture here
/// rather than on the simulation truth alone matters because this realization's
/// MLE sits ~15% below truth (`CL 2.618, V 16.963` vs `3, 20`) — a band around
/// truth wide enough to admit the MLE would also admit a fit that never moved
/// from its starting values.
fn focei_reference(
    model: &ferx_core::types::CompiledModel,
    pop: &Population,
) -> ferx_core::types::FitResult {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.interaction = true;
    fit(model, pop, &model.default_params, &opts).expect("FOCEI reference fit must produce a fit")
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

    // Deterministic optimum of this realization — the target for every recovering
    // arm below, and a sanity check that the fixture is still an identifiable
    // problem for a non-sampling estimator.
    let focei = focei_reference(&model, &pop);
    let (ref_cl, ref_v) = (focei.theta[0], focei.theta[1]);
    assert!(
        focei.ofv.is_finite() && max_rel_err(&focei.theta, TRUE_CL, TRUE_V) < 0.25,
        "FOCEI reference must itself sit near the truth (CL {ref_cl}, V {ref_v}, OFV {}) — \
         otherwise the fixture no longer poses an identifiable problem",
        focei.ofv
    );

    // --- Part 1: default config (ISCALE pilot search on) — the shipped path. The
    // mixture recovers the FOCEI optimum: the #528 core, the IMP phase stays
    // identifiable.
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
    let err_mix_default = max_rel_err(&with_mix.theta, ref_cl, ref_v);
    assert!(
        err_mix_default < 0.10,
        "defensive mixture must land on the FOCEI optimum (CL {ref_cl:.3}, V {ref_v:.3}), got \
         CL {} / V {} ({:.0}% off)",
        with_mix.theta[0],
        with_mix.theta[1],
        err_mix_default * 100.0
    );
    assert!(
        max_rel_err(&with_mix.theta, TRUE_CL, TRUE_V) < 0.25,
        "defensive mixture must also stay near the simulation truth (TVCL {TRUE_CL}, \
         TVV {TRUE_V}), got CL {} / V {}",
        with_mix.theta[0],
        with_mix.theta[1]
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

    // --- Part 2: mixture-vs-legacy discrimination in the **default** config
    // (#1113). On the re-based fixture the ISCALE scalar pilot search no longer
    // rescues the legacy single-proposal sampler, so the contrast the defensive
    // mixture exists for is visible on the shipped path — no knob has to be turned
    // off to see it:
    //
    //   * legacy single-proposal (alpha = 0): weights collapse on the baseline
    //     subjects, the importance-weighted M-step is hijacked, and CL/V run away.
    //   * defensive mixture (alpha = 0.1): the broad `N(0, Ω)` component bounds
    //     every weight, so no single collapsed-weight subject can dominate and θ
    //     stays identifiable.
    let legacy_default = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_imp_opts(0.0, false),
    )
    .expect("legacy imp (default ISCALE) still returns a (bad) fit");
    let err_legacy_default = max_rel_err(&legacy_default.theta, TRUE_CL, TRUE_V);
    assert!(
        err_legacy_default > 0.5,
        "legacy sampler (default config) is expected to run CL/V far off truth (>50% error); \
         got CL {} / V {} ({:.0}% off) — if this fails the fixture no longer reproduces the \
         collapse and the negative control must be re-based, not weakened (#961 / #1113)",
        legacy_default.theta[0],
        legacy_default.theta[1],
        err_legacy_default * 100.0
    );
    let err_mix_vs_truth = max_rel_err(&with_mix.theta, TRUE_CL, TRUE_V);
    assert!(
        err_legacy_default > 3.0 * err_mix_vs_truth,
        "defensive mixture must beat legacy by a wide margin in the default config: mixture \
         CL {} / V {} ({:.0}% off truth) vs legacy CL {} / V {} ({:.0}% off) — only {:.1}× better",
        with_mix.theta[0],
        with_mix.theta[1],
        err_mix_vs_truth * 100.0,
        legacy_default.theta[0],
        legacy_default.theta[1],
        err_legacy_default * 100.0,
        err_legacy_default / err_mix_vs_truth
    );

    // --- Part 3: the same contrast with the ISCALE scalar pilot search pinned off
    // (#961). ISCALE is a *separate*, general safeguard; pinning it to 1.0 (its
    // documented "disabled" setting) reads the discrimination with that overlapping
    // rescue removed, so a future ISCALE change cannot silently become the reason
    // Part 2 passes.
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
    // The pinned mixture must be a genuine, healthy recovery — not just a θ that
    // happens to land near truth. Guard convergence and OFV too, so a partial
    // regression (the fit fails to converge, or the objective blows up) still trips.
    assert!(
        with_mix_pinned.converged,
        "defensive mixture (ISCALE pinned) must converge"
    );
    assert!(
        with_mix_pinned.ofv.is_finite() && with_mix_pinned.ofv.abs() < 1e6,
        "defensive mixture (ISCALE pinned) OFV must be finite/sane: {}",
        with_mix_pinned.ofv
    );
    let err_mixture = max_rel_err(&with_mix_pinned.theta, ref_cl, ref_v);
    assert!(
        err_mixture < 0.10,
        "defensive mixture (ISCALE pinned) must still land on the FOCEI optimum \
         (CL {ref_cl:.3}, V {ref_v:.3}), got CL {} / V {} ({:.0}% off)",
        with_mix_pinned.theta[0],
        with_mix_pinned.theta[1],
        err_mixture * 100.0
    );
    // ...and the legacy sampler does not: with ISCALE unable to rebroaden the
    // collapsed proposal, its worse coordinate runs far outside the band the
    // mixture stays inside. Stated as *relative* discrimination rather than an
    // absolute multiple of truth on purpose — how far the runaway travels, and in
    // which coordinate, is a chaotic property of the collapsed weights and differs
    // by libm; pinning the absolute magnitude turned a real, reproducible contrast
    // into a platform-dependent red (#751 / #961). What the test needs is that
    // legacy is *badly* wrong and the mixture is not.
    let err_legacy = max_rel_err(&no_mix.theta, TRUE_CL, TRUE_V);
    let err_mixture_vs_truth = max_rel_err(&with_mix_pinned.theta, TRUE_CL, TRUE_V);
    assert!(
        err_legacy > 0.5,
        "legacy sampler (ISCALE pinned) is expected to run CL/V far off truth (>50% error); \
         got CL {} / V {} ({:.0}% off) — if this fails the fixture no longer reproduces the \
         collapse",
        no_mix.theta[0],
        no_mix.theta[1],
        err_legacy * 100.0
    );
    assert!(
        err_legacy > 3.0 * err_mixture_vs_truth,
        "defensive mixture must beat legacy by a wide margin: mixture CL {} / V {} \
         ({:.0}% off truth) vs legacy CL {} / V {} ({:.0}% off) — only {:.1}× better",
        with_mix_pinned.theta[0],
        with_mix_pinned.theta[1],
        err_mixture_vs_truth * 100.0,
        no_mix.theta[0],
        no_mix.theta[1],
        err_legacy * 100.0,
        err_legacy / err_mixture_vs_truth
    );
}
