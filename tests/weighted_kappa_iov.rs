//! Tier-2: sample-size-weighted IOV (#1031) — `kappa K ~ γ² weight = W` declares
//! `κ_ik ~ N(0, Ω_IOV / W_ik)`, the between-treatment-arm variability of a
//! longitudinal MBMA.
//!
//! The engine applies it by rewriting `K` into `K / sqrt(W)` inside
//! `[individual_parameters]` — exactly the hand-written form the modifier
//! replaces. These tests pin that equivalence *at the objective*, under every
//! estimator: the declared model and the hand-written one must score the same
//! data identically, and a model that ignored the weight would score the
//! unweighted κ instead (a plausible wrong answer, which is the whole reason
//! the declaration exists).
//!
//! Per-kernel gates — the analytic `∂f/∂κ` against finite differences of
//! `predict_iov`, the parser rewrite, the non-positive-weight data check — are
//! Tier-1 unit tests in `sens::provider`, `parser::model_parser` and
//! `api::validation`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{EstimationMethod, Population};
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::io::Write;

/// Arm-level IOV on CL, weighted by the arm size `NARM`. Two occasions per
/// "study" (subject) with *different* arm sizes, so a scaling that were applied
/// uniformly — or not at all — changes the objective.
const WEIGHTED_KAPPA: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.5 (sd) weight = NARM
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  NARM continuous

[fit_options]
  iov_column = OCC
";

/// The same model written by hand — the form #1031 replaces. Identical Ω_IOV
/// declaration, scaling moved into the structural expression.
const HAND_WRITTEN_KAPPA: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.5 (sd)
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL / sqrt(NARM))
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  NARM continuous

[fit_options]
  iov_column = OCC
";

/// The unweighted model — what a dropped `weight =` would silently fit.
const UNWEIGHTED_KAPPA: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.5 (sd)
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[covariates]
  NARM continuous

[fit_options]
  iov_column = OCC
";

/// Four studies × two arms; each arm is one occasion with its own size.
fn arm_population() -> Population {
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,OCC,NARM\n");
    for sid in 1..=4u32 {
        let cl = 0.9 + 0.1 * sid as f64;
        for (occ, narm) in [(1u32, 200.0_f64), (2, 25.0)] {
            let t0 = 24.0 * (occ - 1) as f64;
            csv.push_str(&format!("{sid},{t0},.,100,1,1,1,{occ},{narm}\n"));
            for (k, dt) in [1.0_f64, 4.0, 8.0].iter().enumerate() {
                let t = t0 + dt;
                // A small arm-specific shift, larger for the smaller arm —
                // the shape the weighted variance is meant to describe.
                let shift = if occ == 1 { 0.02 } else { 0.15 };
                let conc = (100.0 / 10.0) * (-(cl * (1.0 + shift) / 10.0) * dt).exp();
                let dv = conc * (1.0 + 0.02 * ((sid as usize + k) as f64).sin());
                csv.push_str(&format!("{sid},{t},{dv:.5},.,0,1,0,{occ},{narm}\n"));
            }
        }
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("arms.csv");
    let mut f = std::fs::File::create(&path).expect("create csv");
    f.write_all(csv.as_bytes()).expect("write csv");
    drop(f);
    let pop = read_nonmem_csv(&path, None, Some("OCC")).expect("arm data must load");
    drop(dir);
    pop
}

fn smoke_opts(method: EstimationMethod) -> FitOptions {
    let mut o = FitOptions {
        outer_maxiter: 2,
        ..Default::default()
    };
    o.method = method;
    o.iov_column = Some("OCC".to_string());
    o.saem_n_exploration = 2;
    o.saem_n_convergence = 2;
    o.imp_iterations = 2;
    o.imp_samples = 64;
    o.saem_seed = Some(1);
    o.imp_seed = Some(1);
    o
}

/// The declared form and the hand-written one are the same model, so every
/// estimator must score them identically — that equivalence is the entire
/// implementation, and it is what lets the rest of the engine stay unaware of
/// the feature.
#[test]
fn the_declared_weight_matches_the_hand_written_scaling_under_every_estimator() {
    let declared = parse_model_string(WEIGHTED_KAPPA).expect("weighted model must parse");
    let by_hand = parse_model_string(HAND_WRITTEN_KAPPA).expect("hand-written model must parse");
    assert!(declared.has_weighted_kappa());
    assert!(!by_hand.has_weighted_kappa());
    let pop = arm_population();

    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::Laplace,
        EstimationMethod::Saem,
        // IMP/IMPMAP reject IOV outright (their κ M-step is a planned
        // follow-up), so the estimator list stops here.
    ] {
        let opts = smoke_opts(method);
        let a = fit(&declared, &pop, &declared.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?} declared-weight fit: {e}"));
        let b = fit(&by_hand, &pop, &by_hand.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?} hand-written fit: {e}"));
        assert!(
            a.ofv.is_finite(),
            "{method:?} returned a non-finite OFV: {}",
            a.ofv
        );
        assert!(
            (a.ofv - b.ofv).abs() <= 1e-8 * a.ofv.abs().max(1.0),
            "{method:?}: `weight = NARM` scored {} but the hand-written \
             `KAPPA_CL / sqrt(NARM)` scored {} — they are the same model",
            a.ofv,
            b.ofv
        );
    }
}

/// ... and the weight is genuinely applied: the unweighted model — what a
/// silently dropped modifier would fit — must score the same data differently.
#[test]
fn the_weight_changes_the_objective() {
    let declared = parse_model_string(WEIGHTED_KAPPA).expect("weighted model must parse");
    let plain = parse_model_string(UNWEIGHTED_KAPPA).expect("unweighted model must parse");
    let pop = arm_population();
    let opts = smoke_opts(EstimationMethod::Foce);
    let a = fit(&declared, &pop, &declared.default_params, &opts).expect("weighted fit");
    let b = fit(&plain, &pop, &plain.default_params, &opts).expect("unweighted fit");
    assert!(
        (a.ofv - b.ofv).abs() > 1e-6,
        "weighted and unweighted κ scored identically ({} vs {}) — the weight is \
         not reaching the likelihood",
        a.ofv,
        b.ofv
    );
}

/// The fit reports the weight and the arm size it should be read against: the
/// estimate stays the unweighted γ², and `γ/√median(N)` is the between-arm SD
/// a reader needs.
#[test]
fn the_fit_reports_the_weight_and_the_typical_arm_size() {
    let declared = parse_model_string(WEIGHTED_KAPPA).expect("weighted model must parse");
    let pop = arm_population();
    let res = fit(
        &declared,
        &pop,
        &declared.default_params,
        &smoke_opts(EstimationMethod::Foce),
    )
    .expect("weighted fit");
    assert_eq!(res.kappa_weights, vec![Some("NARM".to_string())]);
    // Eight subject-occasions: four at 200, four at 25 — the median lands on
    // one of the two, and either is a legitimate "typical arm".
    let typical = res.kappa_weight_typical[0].expect("a typical arm size");
    assert!(
        typical == 200.0 || typical == 25.0,
        "unexpected typical arm size {typical}"
    );
}
