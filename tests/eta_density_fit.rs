//! Non-Gaussian random-effect distributions (#1716) through the public `fit()` /
//! `simulate()` entry points.
//!
//! Tier 2: every estimator that scores a density runs a few iterations on one and returns
//! a finite objective, on every omega-line form (copula, multivariate log-normal,
//! truncation, a `[mixture]` override, an IOV κ) and under each Bayes Ω prior.
//! Tier 3 (`slow-tests`): simulate a population with a Gamma clearance with ferx's own
//! sampler, refit it, and recover the generating parameters, which is the only check that
//! the sampler and the likelihood describe the same distribution.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{EstimationMethod, FitOptions, Population, SimOutcome};

/// The warfarin model with a Gamma clearance (mean `TVCL`, variance ω) and a t volume.
const DENSITY_WARFARIN: &str = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  theta NU(10.0, 2.5, 100.0)
  omega ETA_CL ~ 0.09 density gamma
  omega ETA_V  ~ 0.04 density t(NU)
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * ETA_CL
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;

fn model_and_data() -> (ferx_core::types::CompiledModel, Population) {
    let parsed = parse_full_model(DENSITY_WARFARIN).expect("density model parses");
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
    (parsed.model, pop)
}

fn short(method: EstimationMethod, interaction: bool) -> FitOptions {
    FitOptions {
        method,
        interaction,
        outer_maxiter: 3,
        saem_n_exploration: 5,
        saem_n_convergence: 5,
        imp_iterations: 2,
        bayes_warmup: 40,
        bayes_iters: 40,
        bayes_chains: 1,
        vi_iters: 20,
        // VI's own `ofv` is NaN by design (the ELBO is a bound): report the Laplace
        // objective at its estimate, which scores the density too.
        vi_final_ofv: ferx_core::types::ViFinalOfv::Laplace,
        run_covariance_step: false,
        ..FitOptions::default()
    }
}

/// Each estimator that scores a density returns a finite objective after a few
/// iterations. SAEM's density M-step and Bayes's Metropolis moves shift ν and the
/// gamma's ω off their initial values (under SAEM the data M-step pins ν, so only the
/// density step can move it).
#[test]
fn every_density_estimator_runs() {
    let (model, pop) = model_and_data();
    for (method, interaction) in [
        (EstimationMethod::FoceI, true),
        (EstimationMethod::Foce, false),
        (EstimationMethod::Laplace, true),
        (EstimationMethod::Saem, true),
        (EstimationMethod::Imp, true),
        (EstimationMethod::Bayes, true),
        (EstimationMethod::Vi, true),
    ] {
        let r = ferx_core::fit(&model, &pop, &model.default_params, &short(method, interaction))
            .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(r.ofv.is_finite() && r.ofv.abs() < 1e10, "{method:?}: OFV {}", r.ofv);
        if matches!(method, EstimationMethod::Saem | EstimationMethod::Bayes | EstimationMethod::Vi)
        {
            assert_ne!(r.theta[3], model.default_params.theta[3], "{method:?} moves ν");
            assert_ne!(r.omega[(0, 0)], 0.09, "{method:?} moves the gamma's ω");
        }
    }
}

/// VI's Ω step under `vi_omega_update = adam` follows the Gaussian prior's gradient; a
/// density group's ω and shape θ are the density M-step's instead, so Adam runs on a
/// density model and the gamma's ω still moves.
#[test]
fn vi_adam_omega_runs_a_density() {
    let (model, pop) = model_and_data();
    let opts = FitOptions {
        vi_omega_update: ferx_core::types::ViOmegaUpdate::Adam,
        ..short(EstimationMethod::Vi, true)
    };
    let r = ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("VI + Adam");
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    assert_ne!(r.omega[(0, 0)], 0.09, "the gamma's ω moves");
}

/// `DENSITY_WARFARIN` with its omega lines and individual parameters replaced.
fn variant(omegas: &str, indiv: &str, extra: &str) -> ferx_core::types::CompiledModel {
    let src = DENSITY_WARFARIN
        .replace(
            "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V  ~ 0.04 density t(NU)\n  omega ETA_KA ~ 0.30\n",
            omegas,
        )
        .replace(
            "  CL = TVCL * ETA_CL\n  V  = TVV  * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n",
            indiv,
        )
        + extra;
    parse_full_model(&src).unwrap_or_else(|e| panic!("{e}\n{src}")).model
}

/// The omega-line forms beyond a single kind — a Gaussian copula of two margins, a
/// multivariate log-normal block, a truncated gamma, and a `[mixture]` class overriding a
/// density ETA's ω — each fit under FOCEI and SAEM to a finite objective, SAEM moving the
/// density's ω (only its density M-step can).
#[test]
fn every_omega_line_form_fits() {
    let (_, pop) = model_and_data();
    let plain = "  CL = TVCL * ETA_CL\n  V  = TVV  * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n";
    let cases = [
        (
            "copula",
            variant(
                "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] density copula(ETA_CL = gamma, ETA_V = t(NU))\n  omega ETA_KA ~ 0.30\n",
                plain,
                "",
            ),
        ),
        (
            "mvlognormal",
            variant(
                "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] density lognormal_mean\n  omega ETA_KA ~ 0.30 density t(NU)\n",
                "  CL = TVCL * ETA_CL\n  V  = TVV  * ETA_V\n  KA = TVKA * exp(ETA_KA)\n",
                "",
            ),
        ),
        (
            "truncated",
            variant(
                "  omega ETA_CL ~ 0.09 density truncated(gamma, 0.2, 5)\n  omega ETA_V  ~ 0.04 density t(NU)\n  omega ETA_KA ~ 0.30\n",
                plain,
                "",
            ),
        ),
        (
            "mixture",
            variant(
                "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V  ~ 0.04 density t(NU)\n  omega ETA_KA ~ 0.30\n",
                "  CL = if (MIXNUM == 1) TVCL * ETA_CL else 2 * TVCL * ETA_CL\n  V  = TVV  * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n",
                "\n[mixture]\n  nsub = 2\n  logit(1) = 0\n  omega(2) ETA_CL ~ 0.2\n",
            ),
        ),
    ];
    for (name, model) in &cases {
        for method in [EstimationMethod::FoceI, EstimationMethod::Saem] {
            let r = ferx_core::fit(model, &pop, &model.default_params, &short(method, true))
                .unwrap_or_else(|e| panic!("{name} {method:?}: {e}"));
            assert!(r.ofv.is_finite() && r.ofv.abs() < 1e10, "{name} {method:?}: OFV {}", r.ofv);
            if method == EstimationMethod::Saem {
                assert_ne!(r.omega[(0, 0)], 0.09, "{name}: SAEM moves the density's ω");
            }
        }
    }
}

/// An IOV `kappa` with a density: FOCEI and SAEM fit it on warfarin's occasions, and SAEM's
/// κ density M-step moves its ω.
#[test]
fn a_kappa_density_fits() {
    let src = DENSITY_WARFARIN
        .replace(
            "  omega ETA_KA ~ 0.30\n",
            "  omega ETA_KA ~ 0.30\n  kappa K_CL ~ 0.04 density gamma\n",
        )
        .replace("  CL = TVCL * ETA_CL\n", "  CL = TVCL * ETA_CL * K_CL\n")
        + "\n[fit_options]\n  iov_column = OCC\n";
    let parsed = parse_full_model(&src).expect("kappa density model parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin_iov.csv",
        None,
        Some("OCC"),
        None,
        &[],
    )
    .expect("warfarin IOV data reads");
    let model = parsed.model;
    assert_eq!(model.kappa_densities.len(), 1);
    for method in [EstimationMethod::FoceI, EstimationMethod::Saem] {
        let opts = FitOptions { iov_column: Some("OCC".into()), ..short(method, true) };
        let r = ferx_core::fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(r.ofv.is_finite() && r.ofv.abs() < 1e10, "{method:?}: OFV {}", r.ofv);
        if method == EstimationMethod::Saem {
            let k = r.omega_iov.as_ref().expect("an IOV Ω")[(0, 0)];
            assert_ne!(k, 0.04, "SAEM moves the κ density's ω");
        }
        // The KAPPA row carries the κ's distribution note.
        let note = ferx_core::io::output::kappa_distribution_note(&r, 0).expect("K_CL's note");
        assert!(note.starts_with("density gamma (mean 1, CV% = "), "{note}");
        let rows = ferx_core::io::output::format_kappa_rows_from(
            &ferx_core::io::output::KappaRowsInput::from_result(&r).expect("IOV"),
            Default::default(),
        );
        assert!(rows.contains(&note), "{rows}");
    }
}

/// `bayes_omega_prior = wishart | lkj(η)`: Bayes runs under each, every block moving by
/// the Metropolis step, so the Gaussian ETA_KA's ω moves too.
#[test]
fn bayes_omega_priors_run() {
    let (model, pop) = model_and_data();
    for prior in
        [ferx_core::types::OmegaPrior::Wishart, ferx_core::types::OmegaPrior::Lkj { eta: 2.0 }]
    {
        let opts = FitOptions { bayes_omega_prior: prior, ..short(EstimationMethod::Bayes, true) };
        let r = ferx_core::fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{prior:?}: {e}"));
        assert!(r.ofv.is_finite(), "{prior:?}: OFV {}", r.ofv);
        assert_ne!(r.omega[(2, 2)], 0.30, "{prior:?} moves ETA_KA's ω");
    }
}

/// A density ETA's omega row and YAML entry report its distribution and the percentiles
/// of its effect; a Gaussian ETA's do not. The gamma's percentiles are those of a mean-1
/// effect, so they straddle 1.
#[test]
fn a_density_eta_reports_its_distribution() {
    let (model, pop) = model_and_data();
    let r =
        ferx_core::fit(&model, &pop, &model.default_params, &short(EstimationMethod::FoceI, true))
            .expect("FOCEI fit");
    let note = ferx_core::io::output::eta_distribution_note(&r, 0).expect("ETA_CL's note");
    // The gamma's effect has mean 1 and variance ω exactly, so its CV is 100·√ω.
    let cv = 100.0 * r.omega[(0, 0)].sqrt();
    assert!(
        note.starts_with(&format!("density gamma (mean 1, CV% = {cv:.1}): 5/50/95% = ")),
        "{note} (CV {cv})"
    );
    let [lo, mid, hi] = ferx_core::io::output::eta_effect_quantiles(&r, 0).expect("quantiles");
    assert!(lo < mid && mid < 1.0 && 1.0 < hi, "{lo} {mid} {hi}");
    // The t's effect: mean 0, variance ω ν/(ν − 2) (ω is its scale²).
    let nu = r.theta[3];
    let sd = (r.omega[(1, 1)] * nu / (nu - 2.0)).sqrt();
    let t_note = ferx_core::io::output::eta_distribution_note(&r, 1).expect("ETA_V's note");
    assert!(
        t_note.starts_with(&format!("density t(NU), ω = scale² (mean 0, SD = {sd:.4}): ")),
        "{t_note} (SD {sd})"
    );
    // The EBE table on the effect's scale: `exp(ẑ)` for the gamma, the EBE itself for
    // the t (identity map) and the Gaussian ETA_KA.
    let effects = ferx_core::io::output::eta_effects(&r);
    for (s, e) in r.subjects.iter().zip(&effects) {
        assert!(s.eta.iter().chain(e).all(|v| v.is_finite()), "{e:?}");
        assert_eq!(e[0], s.eta[0].exp());
        assert_eq!((e[1], e[2]), (s.eta[1], s.eta[2]));
    }
    // Effect-scale shrinkage for the gamma: `1 − RMS(exp(ẑ) − 1)/√ω`, not the Gaussian
    // `1 − RMS(ẑ)/√ω`; the two differ because the latent is not centred.
    let n = r.subjects.len() as f64;
    let rms = (effects.iter().map(|e| (e[0] - 1.0).powi(2)).sum::<f64>() / n).sqrt();
    let want = 1.0 - rms / r.omega[(0, 0)].sqrt();
    let latent = (r.subjects.iter().map(|s| s.eta[0].powi(2)).sum::<f64>() / n).sqrt();
    let gaussian = 1.0 - latent / r.omega[(0, 0)].sqrt();
    assert!(r.shrinkage_eta[0].is_finite() && want.is_finite() && gaussian.is_finite());
    assert!((r.shrinkage_eta[0] - want).abs() < 1e-12, "{} vs {want}", r.shrinkage_eta[0]);
    assert!((r.shrinkage_eta[0] - gaussian).abs() > 1e-6, "the scale matters here");
    assert_eq!(ferx_core::io::output::eta_distribution_note(&r, 2), None);
    // Which summary each typical value is (#2669): the gamma's TVCL the mean of CL, the
    // symmetric t's TVV and the Gaussian TVKA medians.
    let tv: Vec<(String, String, &str)> = r
        .typical_values
        .iter()
        .map(|v| (v.theta.clone(), v.parameter.clone(), v.statistic.label()))
        .collect();
    for want in [("TVCL", "CL", "mean"), ("TVV", "V", "median"), ("TVKA", "KA", "median")] {
        assert!(
            tv.iter().any(|(t, p, st)| (t.as_str(), p.as_str(), *st) == want),
            "{want:?} in {tv:?}"
        );
    }
    let rows = ferx_core::io::output::format_omega_rows_from(
        &ferx_core::io::output::OmegaRowsInput::from_result(&r),
        Default::default(),
    );
    assert!(rows.contains(&note), "{rows}");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fit.yaml");
    ferx_core::io::output::write_estimates_yaml(&r, path.to_str().unwrap()).unwrap();
    let yaml = std::fs::read_to_string(&path).unwrap();
    assert!(yaml.contains("    distribution: gamma\n"), "{yaml}");
    assert!(yaml.contains("    effect_quantiles_5_50_95: ["), "{yaml}");
    assert!(yaml.contains("    effect_mean: 1.000000\n"), "{yaml}");
    assert!(yaml.contains(&format!("    effect_cv_pct: {cv:.2}\n")), "{yaml}");
    eprintln!("{rows}");
}

/// Simulate 1000 subjects from known parameters with a Gamma clearance (ω = 0.16, so a
/// CV of 40% on a right-skewed CL) and refit with FOCEI: the generating parameters come
/// back. Every fit-side test scores the density on data it did not draw, so only this
/// round trip sees a sampler that draws another distribution than the one scored.
#[test]
#[cfg_attr(not(feature = "slow-tests"), ignore = "slow: opt in with --features slow-tests")]
fn a_simulated_gamma_clearance_is_recovered() {
    let (model, template) = model_and_data();
    let mut truth = model.default_params.clone();
    truth.theta = vec![0.15, 9.0, 1.2, 8.0];
    truth.omega = ferx_core::types::OmegaMatrix::from_matrix(
        nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.16, 0.03, 0.2])),
        truth.omega.eta_names.clone(),
        true,
    );
    // 100 replicates of warfarin's 10 subjects: 1000 subjects, so ω_CL's sampling SE is
    // about √(2/1000) = 4.5%. At 210 subjects it was 9.8%, too wide to see a sampler that
    // draws another distribution.
    let n_rep = 100;
    let sims = ferx_core::simulate_with_seed(&model, &template, &truth, n_rep, 1716).unwrap();
    let mut pop = template.clone();
    pop.subjects.clear();
    for k in 0..n_rep {
        let mut rows = sims.iter().filter(|r| r.sim == k + 1);
        for subj in &template.subjects {
            let mut s = subj.clone();
            s.id = format!("{}_{}", subj.id, k + 1);
            for y in s.observations.iter_mut() {
                match rows.next().expect("one row per observation").outcome {
                    SimOutcome::Continuous { value } => *y = value,
                    ref o => panic!("expected a continuous outcome, got {o:?}"),
                }
            }
            pop.subjects.push(s);
        }
    }
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let r = ferx_core::fit(&model, &pop, &model.default_params, &opts).expect("FOCEI fit");
    let rel = |got: f64, want: f64| (got - want).abs() / want;
    let gaps = [
        ("TVCL", rel(r.theta[0], 0.15)),
        ("TVV", rel(r.theta[1], 9.0)),
        ("ω_CL", rel(r.omega[(0, 0)], 0.16)),
    ];
    eprintln!("recovery: {gaps:?}, ν = {}", r.theta[3]);
    // Measured (seed 1716): TVCL 0.9%, TVV 0.06%, ω_CL 4.0%, about one SE each; bounds
    // about 3 SE. ν (measured 12.6 against 8) is weakly identified by 1000 subjects and
    // is not asserted.
    let bounds = [0.03, 0.03, 0.15];
    for ((name, g), b) in gaps.into_iter().zip(bounds) {
        assert!(g.is_finite() && g < b, "{name}: {g}");
    }
}

/// A kind with a kink or a positive-density edge (`laplace`, `uniform`, `pareto`, …) is
/// refused under every method that reads the prior's curvature at the EBE — FOCE, FOCEI,
/// the GN pair, Laplace, VI's Laplace objective and SAEM's `saem_ofv = focei` — with
/// `E_DENSITY_UNSUPPORTED` naming IMP / SAEM / Bayes, which run it and report a number that
/// scores its prior. A smooth kind (`gamma`) is not
/// refused under FOCEI: the gate is the kind, not the presence of a density.
#[test]
fn a_kink_kind_is_refused_where_curvature_is_read() {
    let (_, pop) = model_and_data();
    let plain = "  CL = TVCL * ETA_CL\n  V  = TVV  * exp(ETA_V)\n  KA = TVKA * exp(ETA_KA)\n";
    let kink = variant(
        "  omega ETA_CL ~ 0.09 density gamma\n  omega ETA_V  ~ 0.04 density laplace\n  omega ETA_KA ~ 0.30\n",
        plain,
        "",
    );
    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
        EstimationMethod::Laplace,
        EstimationMethod::Vi,
    ] {
        let e = ferx_core::fit(&kink, &pop, &kink.default_params, &short(method, true))
            .expect_err(&format!("{method:?} refuses a kink kind"));
        let code = e.diagnostic().map(|d| d.code.clone());
        assert_eq!(code.as_deref(), Some("E_DENSITY_UNSUPPORTED"), "{method:?}: {e}");
        let msg = e.to_string();
        assert!(msg.contains("density laplace") && msg.contains("method = imp"), "{msg}");
    }
    // Where a kink kind runs, the reported number is one that scores its prior (#2669
    // review, finding 1). IMP: `ofv` is the IS marginal itself, labelled so; SAEM: its own
    // objective, with no FOCE/Laplace diagnostic; and the covariance step, which reads the
    // FOCE curvature, is skipped with `W_DENSITY_NO_COVARIANCE` rather than reported.
    for method in [EstimationMethod::Imp, EstimationMethod::Saem, EstimationMethod::Bayes] {
        let o = FitOptions { run_covariance_step: true, ..short(method, true) };
        let r = ferx_core::fit(&kink, &pop, &kink.default_params, &o)
            .unwrap_or_else(|e| panic!("{method:?} runs a kink kind: {e}"));
        assert!(r.ofv.is_finite(), "{method:?}: OFV {}", r.ofv);
        assert!(r.covariance_matrix.is_none(), "{method:?}: a covariance matrix");
        match method {
            EstimationMethod::Imp => {
                let is = r.importance_sampling.as_ref().expect("the IS marginal");
                assert!(r.ofv_is_importance_sampling);
                assert_eq!(r.ofv_data.to_bits(), is.minus2_log_likelihood.to_bits());
                assert!(
                    r.warnings.iter().any(|w| w.starts_with("W_DENSITY_NO_COVARIANCE")),
                    "{:?}",
                    r.warnings
                );
            }
            EstimationMethod::Saem => {
                let so = r.saem_objective.as_ref().expect("SAEM's own objective");
                assert!(so.foce_ofv.is_nan(), "FOCE diagnostic {}", so.foce_ofv);
                assert!(
                    r.warnings.iter().any(|w| w.starts_with("W_DENSITY_NO_COVARIANCE")),
                    "{:?}",
                    r.warnings
                );
            }
            _ => assert!(!r.ofv_is_importance_sampling),
        }
    }
    // SAEM reporting the FOCE objective reads the curvature: refused.
    let o = FitOptions {
        saem_ofv: ferx_core::types::SaemOfv::Focei,
        ..short(EstimationMethod::Saem, true)
    };
    let e = ferx_core::fit(&kink, &pop, &kink.default_params, &o).expect_err("saem_ofv = focei");
    assert!(e.to_string().contains("saem_ofv = focei"), "{e}");
    // The other side of the gate: a smooth kind under FOCEI runs (`every_density_
    // estimator_runs` uses gamma and t).
    let (smooth, _) = model_and_data();
    assert!(ferx_core::fit(
        &smooth,
        &pop,
        &smooth.default_params,
        &short(EstimationMethod::FoceI, true)
    )
    .is_ok());
}

/// Adaptive quadrature integrates a kink kind's prior (#2669 review, finding 1): its
/// Hessian only places the nodes. So an AGQ evaluation of a one-ETA `density laplace`
/// model is accepted, and at 15 nodes equals the importance-sampling −2 log L. Measured on
/// warfarin: IS 284.69396 ± 0.0056 (K = 200 000); AGQ 284.6949 / 284.6881 / 284.6893 /
/// 284.6887 at 5 / 9 / 15 / 21 nodes; 3 nodes 284.80. The bound is 4 Monte Carlo SEs at
/// K = 20 000.
/// Dies under: AGQ refused for a kink kind, or a quadrature that misses the kink.
#[test]
fn agq_integrates_a_kink_kind_to_the_is_marginal() {
    let src = r#"
[parameters]
  theta TVCL(0.15, 0.001, 10.0)
  theta TVV(8.0, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)
  omega ETA_CL ~ 0.09 density laplace
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let parsed = parse_full_model(src).unwrap();
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &[],
    )
    .unwrap();
    let m = &parsed.model;
    let o = FitOptions {
        method: EstimationMethod::Imp,
        imp_eval_only: true,
        imp_samples: 20_000,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let r = ferx_core::fit(m, &pop, &m.default_params, &o).unwrap();
    let is = r.importance_sampling.as_ref().unwrap();
    assert!(r.ofv_is_importance_sampling && r.ofv == is.minus2_log_likelihood);
    let o = FitOptions {
        method: EstimationMethod::Laplace,
        agq_eval_only: true,
        n_agq: 15,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    let agq = ferx_core::fit(m, &pop, &m.default_params, &o).expect("AGQ runs a kink kind").ofv;
    let gap = (agq - is.minus2_log_likelihood).abs();
    eprintln!(
        "AGQ(15) {agq:.5} vs IS {:.5} ± {:.5}",
        is.minus2_log_likelihood, is.mc_standard_error
    );
    assert!(gap < 4.0 * is.mc_standard_error, "gap {gap:.5}, SE {:.5}", is.mc_standard_error);
    // One node is Laplace, which reads the curvature: refused.
    let o = FitOptions { n_agq: 1, ..o };
    let e = ferx_core::fit(m, &pop, &m.default_params, &o).expect_err("Laplace refused");
    assert!(e.to_string().contains("follows `density laplace`"), "{e}");
}

/// VI's density M-step writes only its own slots (#2669 review, finding 8): a FIXed θ keeps
/// its declared value bit for bit over a run (a full `unpack → pack` each iteration drifted
/// it by an ULP at a time), under both `vi_omega_update` modes, and the Adam mode runs with
/// the density's slots owned by the M-step.
/// Dies under: the repack restored, or the density slots left to Adam as well.
#[test]
fn vi_density_mstep_leaves_other_slots_bit_exact() {
    let src = DENSITY_WARFARIN
        .replace("theta TVKA(1.5, 0.01, 50.0)", "theta TVKA(1.4537, 0.01, 50.0) FIX");
    let parsed = parse_full_model(&src).unwrap();
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &[],
    )
    .unwrap();
    let m = &parsed.model;
    let ka = m.theta_names.iter().position(|n| n == "TVKA").unwrap();
    for update in
        [ferx_core::types::ViOmegaUpdate::ClosedForm, ferx_core::types::ViOmegaUpdate::Adam]
    {
        let o = FitOptions {
            vi_iters: 60,
            vi_omega_update: update,
            ..short(EstimationMethod::Vi, true)
        };
        let r = ferx_core::fit(m, &pop, &m.default_params, &o).unwrap();
        assert_eq!(
            r.theta[ka].to_bits(),
            m.default_params.theta[ka].to_bits(),
            "{update:?}: {}",
            r.theta[ka]
        );
        assert!(r.omega[(0, 0)].is_finite() && r.omega[(0, 0)] > 0.0, "{update:?}");
    }
}

/// The covariance step runs on a density model (#2669 review, finding 14): FOCEI on the
/// gamma CL + t(ν) V model returns a finite, positive SE for the density ω's and for the
/// shape θ ν, which enters the prior only. Every other density test runs with the step off.
/// Dies under: the step declining density subjects, or a shape θ left out of its matrix.
#[test]
fn the_covariance_step_gives_density_parameters_an_se() {
    let (model, pop) = model_and_data();
    let o = FitOptions {
        outer_maxiter: 40,
        run_covariance_step: true,
        ..short(EstimationMethod::FoceI, true)
    };
    let r = ferx_core::fit(&model, &pop, &model.default_params, &o).unwrap();
    let se_theta = r.se_theta.as_ref().expect("θ SEs");
    let nu = r.theta_names.iter().position(|n| n == "NU").unwrap();
    assert!(se_theta[nu].is_finite() && se_theta[nu] > 0.0, "SE(ν) = {}", se_theta[nu]);
    assert!(r.se_omega.is_some(), "ω SEs");
    for i in 0..2 {
        let se =
            ferx_core::types::omega_se_at(&r.se_omega, r.omega.nrows(), i, i).unwrap_or(f64::NAN);
        assert!(se.is_finite() && se > 0.0, "SE(ω_{i}) = {se}");
    }
}
