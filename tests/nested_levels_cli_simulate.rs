//! `ferx --simulate` (`run_model_simulate`) on a model with nested levels (#2084, #1082
//! follow-up).
//!
//! A `[simulation]` design states one value per subject for each covariate, so it cannot
//! hold the groups below a subject that a nested level models. The three outcomes:
//!
//! - **no `covariate STUDY/ARM` lines**: refused before anything is simulated, with
//!   advice specific to nested levels rather than the generic "add `covariate ARM =
//!   <value>`", since following that advice gives one group per level per subject;
//! - **with them, and a kappa whose readers match a random effect of the level above
//!   it**: simulated, then the fit refuses with `E_LEVEL_SINGLE_MEMBER`;
//! - **with them, and no such twin**: runs end to end, one group per level per subject.
//!
//! The twin and the no-twin fixtures differ only in which parameter `K_ST` reads, so the
//! pair straddles the single-member gate.
//!
//! # Tiering
//!
//! Tier 2: the public CLI entry point, four subjects, `maxiter = 1`, no covariance step.

use std::io::Write;

/// `K_ST` reads `k_st_param` (`EMAX` makes it a twin of `ETA_SP`, which reads `EMAX`
/// alone); `K_ARM` reads `ET50`, which nothing at the `STUDY` level reads.
fn model(k_st_param: &str, design: &str) -> String {
    let (emax_k, e0_k) = match k_st_param {
        "EMAX" => (" + K_ST", ""),
        "E0" => ("", " * exp(K_ST)"),
        other => panic!("no fixture for {other}"),
    };
    format!(
        "[parameters]\n  theta TVE0(10.0)\n  theta TVEMAX(20.0)\n  theta TVET50(3.0)\n  \
         omega ETA_SP ~ 0.09\n  level STUDY by STUDY in ID\n  level ARM by ARM in STUDY\n  \
         kappa K_ST ~ 0.04 at STUDY\n  kappa K_ARM ~ 0.04 at ARM\n  sigma ADD ~ 0.25\n\
         [individual_parameters]\n  EMAX = TVEMAX * exp(ETA_SP{emax_k})\n  \
         E0 = TVE0{e0_k}\n  ET50 = TVET50 * exp(K_ARM)\n\
         [structural_model]\n  y = E0 + EMAX * TIME / (TIME + ET50)\n\
         [error_model]\n  DV ~ additive(ADD)\n\
         [fit_options]\n  maxiter = 1\n  covariance = false\n\
         [simulation]\n  n_subjects = 4\n  dose_amt = 100\n  dose_cmt = 1\n  seed = 7\n  \
         times = [1, 2, 4, 8]\n{design}"
    )
}

const BOTH_LEVELS: &str = "  covariate STUDY = [1, 1, 2, 2]\n  covariate ARM = [1, 2, 1, 2]\n";

fn simulate(
    text: &str,
) -> Result<(ferx_core::FitResult, ferx_core::Population), ferx_core::EngineError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("nested.ferx");
    write!(std::fs::File::create(&path).unwrap(), "{text}").unwrap();
    ferx_core::run_model_simulate(path.to_str().unwrap())
}

/// The whole message is asserted, so deleting any one of its sentences fails here.
/// Before #2084 this was the generic weight/body-weight advice ending in "add
/// `covariate ARM = <value>` … to [simulation]".
#[test]
fn a_design_without_the_level_columns_is_refused_with_nested_advice() {
    let err = simulate(&model("E0", "")).expect_err("refused");
    assert_eq!(err.code(), Some("E_MISSING_COVARIATE"), "{err}");
    assert_eq!(err.block(), Some("simulation"), "{err}");
    assert_eq!(
        err.to_string(),
        "[simulation]: the model's nested level(s) `STUDY` (column `STUDY`), `ARM` (column \
         `ARM`) read their groups from column(s) that the simulation design does not supply. \
         A [simulation] design gives each subject one value per covariate, so it cannot \
         express several groups of a level within a subject. Stating `covariate STUDY = \
         <value>` gives every subject a single group at each level, which the fit refuses \
         with `E_LEVEL_SINGLE_MEMBER` whenever a kappa reads the same parameters as a random \
         effect of the level above it. To simulate groups below a subject, use `simulate()` \
         on a dataset that carries the level columns row by row."
    );

    // Only the level the design leaves out is named, and its column is the one advised.
    let err = simulate(&model("E0", "  covariate STUDY = 1\n")).expect_err("refused");
    assert_eq!(err.code(), Some("E_MISSING_COVARIATE"), "{err}");
    let msg = err.to_string();
    assert!(
        msg.starts_with(
            "[simulation]: the model's nested level(s) `ARM` (column `ARM`) read their groups"
        ),
        "{msg}"
    );
    assert!(msg.contains("Stating `covariate ARM = <value>`"), "{msg}");
}

/// One study per subject makes `K_ST` move the same records as `ETA_SP`. With `K_ST` on
/// `EMAX`, which `ETA_SP` reads, the fit of the simulated data refuses; the run never
/// reaches an estimate.
#[test]
fn a_design_with_the_level_columns_and_a_twin_kappa_is_refused_by_the_fit() {
    let err = simulate(&model("EMAX", BOTH_LEVELS)).expect_err("refused");
    assert_eq!(err.code(), Some("E_LEVEL_SINGLE_MEMBER"), "{err}");
    assert_eq!(err.block(), Some("parameters"), "{err}");
    assert_eq!(
        err.to_string(),
        "level `STUDY` has a single group inside every subject, so kappa `K_ST` moves the \
         same records as `ETA_SP` does, through the same parameters (EMAX): the two \
         variances cannot be told apart. Drop one of the two random effects, or drop the \
         level."
    );
}

/// The twin fixture with `K_ST` moved onto `E0`: no kappa shares its readers with a random
/// effect of the level above it, so the run goes end to end, and every subject carries a
/// single group at each level (the design's one value per subject).
#[test]
fn a_design_with_the_level_columns_and_no_twin_runs_end_to_end() {
    let (result, population) = simulate(&model("E0", BOTH_LEVELS)).expect("runs");
    assert!(result.ofv.is_finite(), "a finite objective: {}", result.ofv);
    assert_eq!(population.subjects.len(), 4);
    for (subject, (study, arm)) in
        population
            .subjects
            .iter()
            .zip([(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0)])
    {
        assert_eq!(subject.observations.len(), 4);
        assert!(subject.observations.iter().all(|v| v.is_finite()));
        for j in 0..subject.observations.len() {
            let cov = subject.obs_cov(j);
            assert_eq!(
                (cov["STUDY"], cov["ARM"]),
                (study, arm),
                "subject {} row {j}: one group per level",
                subject.id
            );
        }
    }
}
