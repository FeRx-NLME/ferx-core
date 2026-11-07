//! `saem_accumulation = polyak` (#2721): SAEM's accumulation phase as stochastic
//! approximation on the marginal score, with Polyak–Ruppert averaging and the
//! zero-variance control variate.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::{parse_full_model, parse_full_model_file};
use ferx_core::types::{OmegaMatrix, SaemAccumulation, SaemMstepSolver};
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::path::Path;

/// #1445's sparse combined-error fixture (`data/saem_sparse_combined.csv`, 300 subjects,
/// 1–3 samples each), the design on which σ and most of Ω miss 70–99 % of their
/// complete-data information.
const SPARSE_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.2, 40.0)
  theta TVV1(22.0, 2.0, 200.0)
  theta TVV2(13.0, 1.0, 200.0)
  block_omega (ETA_CL, ETA_V1, ETA_V2) = [0.17, 0.02, 0.15, 0.02, 0.02, 0.69]
  sigma PROP_ERR ~ 0.13 (sd)
  sigma ADD_ERR  ~ 1.80 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = 12.0
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

fn load(
    src: &str,
    data: &str,
) -> (
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
) {
    let parsed = parse_full_model(src).expect("model parses");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        None,
        None,
        &[],
    )
    .expect("data reads");
    (parsed.model, pop)
}

fn saem_opts(seed: u64, acc: SaemAccumulation, explore: usize, converge: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Saem,
        methods: vec![],
        run_covariance_step: false,
        verbose: false,
        saem_n_exploration: explore,
        saem_n_convergence: converge,
        saem_seed: Some(seed),
        saem_accumulation: acc,
        ..FitOptions::default()
    }
}

/// Tier 2: the option parses, runs in scope and reports nothing went wrong; and out of
/// scope (IOV) it says so and keeps the complete-data gain.
#[test]
fn polyak_accumulation_runs_in_scope_and_names_its_fallback() {
    let src = std::fs::read_to_string("examples/warfarin_saem.ferx").unwrap();
    let src = src.replace(
        "method        = saem",
        "method        = saem\n  saem_accumulation = polyak",
    );
    let parsed = parse_full_model(&src).expect("model parses");
    assert_eq!(
        parsed.fit_options.saem_accumulation,
        SaemAccumulation::Polyak
    );
    let (model, pop) = load(&src, "data/warfarin.csv");
    let r = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_opts(7, SaemAccumulation::Polyak, 10, 10),
    )
    .expect("SAEM with the Polyak accumulation fits");
    assert!(r.ofv.is_finite(), "OFV {}", r.ofv);
    assert!(
        !r.warnings.iter().any(|w| w.contains("saem_accumulation")),
        "in scope, no fallback: {:?}",
        r.warnings
    );
    assert!(r.theta.iter().chain(r.sigma.iter()).all(|v| v.is_finite()));
    // It actually ran: the same seed under the classic accumulation lands elsewhere.
    // (A gate that silently never engaged would return the classic estimate exactly.)
    let c = fit(
        &model,
        &pop,
        &model.default_params,
        &saem_opts(7, SaemAccumulation::Classic, 10, 10),
    )
    .expect("classic SAEM fits");
    assert!(
        r.theta != c.theta || r.omega != c.omega,
        "polyak and classic returned the same estimate: {:?}",
        r.theta
    );

    let iov = parse_full_model_file(Path::new("examples/warfarin_iov_saem.ferx")).unwrap();
    let (pop_iov, _) = read_population_for(
        &iov.model,
        &iov.covariate_decls,
        "data/warfarin_iov_saem.csv",
        None,
        iov.fit_options.iov_column.as_deref(),
        None,
        &[],
    )
    .expect("IOV data reads");
    let r = fit(
        &iov.model,
        &pop_iov,
        &iov.model.default_params,
        &saem_opts(7, SaemAccumulation::Polyak, 5, 5),
    )
    .expect("SAEM falls back and fits");
    assert!(
        r.warnings
            .iter()
            .any(|w| w.contains("saem_accumulation = polyak") && w.contains("inter-occasion")),
        "{:?}",
        r.warnings
    );
}

/// Importance-sampled −2 log L at a fit's estimates: the quality metric, independent of
/// the estimator's own objective.
fn is_m2ll(
    model: &ferx_core::types::CompiledModel,
    pop: &ferx_core::types::Population,
    r: &ferx_core::types::FitResult,
) -> f64 {
    let mut p = model.default_params.clone();
    p.theta = r.theta.clone();
    p.omega = OmegaMatrix::from_matrix_with_mask(
        r.omega.clone(),
        p.omega.eta_names.clone(),
        p.omega.diagonal,
        p.omega.free_mask.clone(),
    );
    p.sigma.values = r.sigma.clone();
    let o = FitOptions {
        method: EstimationMethod::Imp,
        methods: vec![],
        interaction: true,
        imp_eval_only: true,
        imp_samples: 3000,
        imp_seed: Some(11),
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    fit(model, pop, &p, &o)
        .expect("IS evaluation")
        .importance_sampling
        .expect("IS result")
        .minus2_log_likelihood
}

/// Benchmark (not a regression test): arms × seeds on a dataset, printing σ, Ω's diagonal,
/// the importance-sampled −2 log L and the wall time. `POLYAK_BENCH` selects the dataset
/// (`sparse`, `warfarin`), `POLYAK_SEEDS` the number of seeds.
#[test]
#[ignore = "benchmark for #2721: run with --ignored --nocapture"]
fn polyak_bench() {
    let which = std::env::var("POLYAK_BENCH").unwrap_or_else(|_| "sparse".into());
    let n_seeds: u64 = std::env::var("POLYAK_SEEDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let arms = std::env::var("POLYAK_ARMS")
        .unwrap_or_else(|_| "bobyqa,score_sa,polyak,polyak_nocv".into());
    let (model, pop) = match which.as_str() {
        "warfarin" => {
            let src = std::fs::read_to_string("examples/warfarin_saem.ferx").unwrap();
            load(&src, "data/warfarin.csv")
        }
        "sparse" => load(SPARSE_MODEL, "data/saem_sparse_combined.csv"),
        other => {
            // `model.ferx:data.csv`
            let (m, d) = other
                .split_once(':')
                .expect("POLYAK_BENCH = sparse | warfarin | model:data");
            load(&std::fs::read_to_string(m).unwrap(), d)
        }
    };
    for arm in arms.split(',') {
        for seed in 1..=n_seeds {
            let mut o = saem_opts(seed, SaemAccumulation::Classic, 150, 250);
            match arm {
                "score_sa" => o.saem_mstep_solver = SaemMstepSolver::ScoreSa,
                "polyak" => o.saem_accumulation = SaemAccumulation::Polyak,
                "polyak_nocv" => {
                    o.saem_accumulation = SaemAccumulation::Polyak;
                    o.saem_control_variate = false;
                }
                _ => {}
            }
            o.verbose = std::env::var("POLYAK_VERBOSE").is_ok();
            let t = std::time::Instant::now();
            let r = fit(&model, &pop, &model.default_params, &o).expect("fit");
            let secs = t.elapsed().as_secs_f64();
            let m2 = is_m2ll(&model, &pop, &r);
            let om: Vec<String> = (0..r.omega.nrows())
                .map(|j| format!("{:.4}", r.omega[(j, j)]))
                .collect();
            eprintln!(
                "BENCH {which} {arm} seed {seed}: theta {:?} sigma {:?} omega_diag [{}] IS-2LL {m2:.2} time {secs:.1}s warn {}",
                r.theta.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
                r.sigma.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>(),
                om.join(", "),
                r.warnings.iter().filter(|w| w.contains("saem_accumulation")).count()
            );
        }
    }
}

/// Tier 3, #2721: on #1445's sparse combined-error design, where σ and most of Ω miss
/// 70–99 % of their information, the Polyak accumulation reaches a lower
/// importance-sampled −2 log L than the classic one, and keeps the minority additive σ
/// near its simulation truth.
///
/// **Measured** (ci-fast profile, seeds 1/2/3, 150/250): `polyak` 3181.45 / 3181.49 /
/// 3181.81 (σ_add 1.70 / 2.04 / 1.82); classic `bobyqa` 3182.46 / 3182.50 / 3184.22. The
/// 3-seed mean gap is 1.48; the gate asks for 0.5, a third of it. Over 10 seeds the means
/// are 3181.96 ± 0.51 against 3183.46 ± 0.68. σ_add's gate is #1445's own, a factor of two
/// either side of the truth 1.8, which the collapsed answer (≈ 0.01) misses by two orders
/// of magnitude.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn polyak_accumulation_beats_the_classic_one_on_the_1445_design() {
    let (model, pop) = load(SPARSE_MODEL, "data/saem_sparse_combined.csv");
    let (mut classic, mut polyak) = (0.0, 0.0);
    for seed in 1..=3u64 {
        let r = fit(
            &model,
            &pop,
            &model.default_params,
            &saem_opts(seed, SaemAccumulation::Classic, 150, 250),
        )
        .expect("classic SAEM fits");
        let c = is_m2ll(&model, &pop, &r);
        let r = fit(
            &model,
            &pop,
            &model.default_params,
            &saem_opts(seed, SaemAccumulation::Polyak, 150, 250),
        )
        .expect("Polyak SAEM fits");
        assert!(
            !r.warnings.iter().any(|w| w.contains("saem_accumulation")),
            "seed {seed}: the Polyak accumulation fell back: {:?}",
            r.warnings
        );
        let p = is_m2ll(&model, &pop, &r);
        assert!(c.is_finite() && p.is_finite(), "seed {seed}: {c} / {p}");
        let add = r.sigma[1];
        assert!(
            add.is_finite() && add > 0.9 && add < 3.6,
            "seed {seed}: σ_add {add} is not within a factor of two of 1.8"
        );
        eprintln!("seed {seed}: classic {c:.2}, polyak {p:.2}, σ_add {add:.3}");
        classic += c / 3.0;
        polyak += p / 3.0;
    }
    assert!(
        polyak < classic - 0.5,
        "polyak's mean IS −2 log L {polyak:.2} is not 0.5 below classic's {classic:.2}"
    );
}
