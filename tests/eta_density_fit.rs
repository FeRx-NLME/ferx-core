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
        let r = ferx_core::fit(
            &model,
            &pop,
            &model.default_params,
            &short(method, interaction),
        )
        .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(
            r.ofv.is_finite() && r.ofv.abs() < 1e10,
            "{method:?}: OFV {}",
            r.ofv
        );
        if matches!(
            method,
            EstimationMethod::Saem | EstimationMethod::Bayes | EstimationMethod::Vi
        ) {
            assert_ne!(
                r.theta[3], model.default_params.theta[3],
                "{method:?} moves ν"
            );
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
    parse_full_model(&src)
        .unwrap_or_else(|e| panic!("{e}\n{src}"))
        .model
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
                "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] density copula(gamma, t(NU))\n  omega ETA_KA ~ 0.30\n",
                plain,
                "",
            ),
        ),
        (
            "mvlognormal",
            variant(
                "  block_omega (ETA_CL, ETA_V) = [0.09, 0.01, 0.04] density lognormal\n  omega ETA_KA ~ 0.30 density t(NU)\n",
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
            assert!(
                r.ofv.is_finite() && r.ofv.abs() < 1e10,
                "{name} {method:?}: OFV {}",
                r.ofv
            );
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
        let opts = FitOptions {
            iov_column: Some("OCC".into()),
            ..short(method, true)
        };
        let r = ferx_core::fit(&model, &pop, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?}: {e}"));
        assert!(
            r.ofv.is_finite() && r.ofv.abs() < 1e10,
            "{method:?}: OFV {}",
            r.ofv
        );
        if method == EstimationMethod::Saem {
            let k = r.omega_iov.as_ref().expect("an IOV Ω")[(0, 0)];
            assert_ne!(k, 0.04, "SAEM moves the κ density's ω");
        }
    }
}

/// `bayes_omega_prior = wishart | lkj(η)`: Bayes runs under each, every block moving by
/// the Metropolis step, so the Gaussian ETA_KA's ω moves too.
#[test]
fn bayes_omega_priors_run() {
    let (model, pop) = model_and_data();
    for prior in [
        ferx_core::types::OmegaPrior::Wishart,
        ferx_core::types::OmegaPrior::Lkj { eta: 2.0 },
    ] {
        let opts = FitOptions {
            bayes_omega_prior: prior,
            ..short(EstimationMethod::Bayes, true)
        };
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
    let r = ferx_core::fit(
        &model,
        &pop,
        &model.default_params,
        &short(EstimationMethod::FoceI, true),
    )
    .expect("FOCEI fit");
    let note = ferx_core::io::output::eta_distribution_note(&r, 0).expect("ETA_CL's note");
    assert!(note.starts_with("density gamma: 5/50/95% = "), "{note}");
    let [lo, mid, hi] = ferx_core::io::output::eta_effect_quantiles(&r, 0).expect("quantiles");
    assert!(lo < mid && mid < 1.0 && 1.0 < hi, "{lo} {mid} {hi}");
    assert!(ferx_core::io::output::eta_distribution_note(&r, 1)
        .expect("ETA_V's note")
        .starts_with("density t(NU): "));
    assert_eq!(ferx_core::io::output::eta_distribution_note(&r, 2), None);
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
    eprintln!("{rows}");
}

/// Simulate 1000 subjects from known parameters with a Gamma clearance (ω = 0.16, so a
/// CV of 40% on a right-skewed CL) and refit with FOCEI: the generating parameters come
/// back. Every fit-side test scores the density on data it did not draw, so only this
/// round trip sees a sampler that draws another distribution than the one scored.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
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
