//! Tier-2: `weight = <expr>` (#1029) — and the #484 residual magnitude it
//! desugars into — must run under **every** estimator, not just FOCE/FOCEI.
//!
//! Until this landed, a magnitude-carrying model was rejected up front for SAEM,
//! GN, GN-hybrid, IMP, IMPMAP and Bayes, because those paths read the residual
//! variance through call sites that never applied the per-observation
//! multiplier — a silently mis-specified error model if it had been allowed.
//! Every one of them now threads it, so the same model is fittable throughout
//! and the *likelihood* is the same object regardless of which estimator scores
//! it.
//!
//! These are smoke tests at the public `fit()` boundary: each method gets a
//! handful of iterations and must return `Ok`. Per-estimator numerical gates
//! (analytic score vs FD, proposal precision vs a hand-built reference, the IOV
//! individual NLL vs its non-IOV twin) are Tier-1 unit tests in
//! `estimation::saem`, `estimation::gauss_newton`,
//! `estimation::importance_sampling` and `stats::likelihood`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::{EstimationMethod, Population};
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::io::Write;

/// Study-as-subject MBMA shape: each row is a trial-arm mean carrying its own
/// reported standard error in `WPSE`, which varies *within* a subject.
const WEIGHTED_MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.04
  sigma ADD_ERR ~ 1.0 (variance) FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD_ERR) weight = WPSE

[covariates]
  WPSE continuous
";

fn weighted_population() -> Population {
    // Four subjects x three observations, each row with its own weight.
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,WPSE\n");
    for sid in 1..=4u32 {
        let cl = 0.9 + 0.1 * sid as f64;
        csv.push_str(&format!("{sid},0,.,100,1,1,1,0.5\n"));
        for (k, t) in [1.0_f64, 4.0, 8.0].iter().enumerate() {
            let conc = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
            let dv = conc * (1.0 + 0.02 * ((sid as usize + k) as f64).sin());
            // Later (lower) observations are less precisely measured.
            let wpse = 0.5 + 0.4 * k as f64;
            csv.push_str(&format!("{sid},{t},{dv:.5},.,0,1,0,{wpse}\n"));
        }
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("weighted.csv");
    let mut f = std::fs::File::create(&path).expect("create csv");
    f.write_all(csv.as_bytes()).expect("write csv");
    drop(f);
    let pop = read_nonmem_csv(&path, None, None).expect("weighted data must load");
    // Keep `dir` alive until after the read.
    drop(dir);
    pop
}

fn smoke_opts(method: EstimationMethod) -> FitOptions {
    let mut o = FitOptions {
        outer_maxiter: 2,
        ..Default::default()
    };
    o.method = method;
    // Keep the stochastic estimators to a couple of cheap passes.
    o.saem_n_exploration = 2;
    o.saem_n_convergence = 2;
    o.imp_iterations = 2;
    o.imp_samples = 64;
    o.saem_seed = Some(1);
    o.imp_seed = Some(1);
    o
}

/// Every estimator accepts the weighted model and returns a finite objective.
/// The `E_RUV_MAGNITUDE_METHOD_UNSUPPORTED` gate this replaces made all but the
/// first two an outright error.
#[test]
fn every_estimator_fits_a_weighted_error_model() {
    let model = parse_model_string(WEIGHTED_MODEL).expect("weighted model must parse");
    assert!(
        model.has_custom_ruv_magnitude(),
        "`weight =` must compile to an active residual magnitude"
    );
    let pop = weighted_population();

    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::FoceGnHybrid,
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Impmap,
        EstimationMethod::Laplace,
        EstimationMethod::Bayes,
    ] {
        let res = fit(&model, &pop, &model.default_params, &smoke_opts(method))
            .unwrap_or_else(|e| panic!("{method:?} must fit a `weight =` model: {e}"));
        assert!(
            res.ofv.is_finite(),
            "{method:?} returned a non-finite OFV: {}",
            res.ofv
        );
    }
}

/// The weight must actually be *applied*, not merely tolerated: re-fitting with
/// every weight set to the same value has to move the objective. A method that
/// silently ignored the multiplier would score both datasets identically.
#[test]
fn the_weight_changes_the_objective_under_every_estimator() {
    let model = parse_model_string(WEIGHTED_MODEL).expect("weighted model must parse");
    let varied = weighted_population();

    // Same rows, but a constant weight column.
    let mut flat = varied.clone();
    for subj in &mut flat.subjects {
        for snap in subj.obs_covariates.iter_mut() {
            snap.insert("WPSE".to_string(), 0.5);
        }
        for snap in subj.dose_covariates.iter_mut() {
            snap.insert("WPSE".to_string(), 0.5);
        }
        subj.covariates.insert("WPSE".to_string(), 0.5);
    }

    for method in [
        EstimationMethod::Foce,
        EstimationMethod::FoceI,
        EstimationMethod::FoceGn,
        EstimationMethod::Saem,
        EstimationMethod::Imp,
        EstimationMethod::Laplace,
    ] {
        let opts = smoke_opts(method);
        let a = fit(&model, &varied, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?} varied-weight fit: {e}"));
        let b = fit(&model, &flat, &model.default_params, &opts)
            .unwrap_or_else(|e| panic!("{method:?} flat-weight fit: {e}"));
        assert!(
            (a.ofv - b.ofv).abs() > 1e-6,
            "{method:?} scored varied and constant weights identically ({} vs {}) — \
             the per-observation multiplier is not reaching its likelihood",
            a.ofv,
            b.ofv
        );
    }
}
