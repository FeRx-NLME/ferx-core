//! `--simulate` must not report the same model/data finding twice (#1280).
//!
//! `run_model_simulate` simulates a design template and then **fits** it, appending the
//! simulation's warnings to the fit's. Since #1280 the simulation half also carries the
//! model/data bundle — which `fit()` then produces a second time, in identical wording, from
//! the same model. Without the exact-duplicate filter in `api::run`, every finding a user sees
//! on a `--simulate` run appears twice.
//!
//! # Tiering
//!
//! Tier 2: it calls the public `run_model_simulate` entry point (the only way to reach the
//! append site) and returns after one outer iteration on three subjects — no convergence loop.

use std::io::Write;

/// The finding has to be one **both** halves produce, and the two halves see different data:
/// the simulation runs on the `DV = .` design template, the fit on the population it wrote
/// back. `W_NEGATIVE_LAG_SCALE` is therefore the wrong choice — it reads `|DV|`, so it is
/// inert on the template and fires only on the fit half (measured: the unfiltered-append
/// mutation stayed green on it). `W_NEGATIVE_LAGTIME` reads the model at its typical values
/// and nothing else, so both halves raise it in identical wording.
const MODEL: &str = r#"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  theta TVKA(1.0, 0.001, 50.0)
  theta TVLAG(-0.5, -10.0, 10.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1

[individual_parameters]
  CL      = TVCL * exp(ETA_CL)
  V       = TVV
  KA      = TVKA
  lagtime = TVLAG

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=lagtime)

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  method = focei
  maxiter = 1
  covariance = false

[simulation]
  n_subjects = 3
  dose_amt   = 100
  dose_cmt   = 1
  seed       = 7
  times      = [1, 4, 12]
"#;

/// The bundle reaches a `--simulate` run, and each finding appears **once**.
///
/// Regression this catches: appending the simulation's warnings unfiltered, which duplicates
/// every model/data finding. Mutation, run: replace the filtered loop in `api::run` with a
/// plain `result.warnings.extend(sim_warnings)` and this fails with `left: 2, right: 1`.
///
/// The first assertion is what makes the second non-vacuous: a count of 1 is also what an
/// implementation that reports the finding *zero* times on the simulation half would give, so
/// the presence check has to come first and the message has to be one the bundle owns.
#[test]
fn a_simulate_run_reports_each_model_data_finding_exactly_once() {
    const NEGATIVE_LAG: &str = "Negative lagtimes are physically nonsensical";

    let dir = tempfile::tempdir().expect("tempdir");
    let model_path = dir.path().join("sim.ferx");
    write!(std::fs::File::create(&model_path).unwrap(), "{MODEL}").unwrap();

    let (result, _population) =
        ferx_core::run_model_simulate(model_path.to_str().unwrap()).expect("simulate");

    let n = result
        .warnings
        .iter()
        .filter(|w| w.contains(NEGATIVE_LAG))
        .count();
    assert!(
        n > 0,
        "the model/data bundle must reach a --simulate run at all: {:?}",
        result.warnings
    );
    assert_eq!(
        n, 1,
        "…and exactly once — the simulation half and the fit half produce it in identical \
         wording, so an unfiltered append shows every finding twice: {:?}",
        result.warnings
    );

    // The structured channel is rebuilt from `warnings`, so it inherits the count.
    let structured = result
        .warnings_structured
        .iter()
        .filter(|e| e.message.contains(NEGATIVE_LAG))
        .count();
    assert_eq!(
        structured, 1,
        "the typed / JSON surface is rebuilt from `warnings` and must agree with it"
    );
}
