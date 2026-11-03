//! `ferx model.ferx --simulate` on a `[mixture]` model writes the class each subject was
//! simulated under to the sdtab as `MIXNUM` (#2372), the column NONMEM's `$TABLE` carries
//! under `$SIMULATION`. Without it the generating class is lost at the write-back into the
//! fitted population, and the table has only the *fitted* `MIXEST` / `PMIX_*`.
//!
//! # Tiering
//!
//! Tier 2: it calls the public `run_model_simulate` entry point (the only way to reach the
//! write-back) and returns after one outer iteration on four subjects — no convergence loop.

use std::io::Write;

/// Two classes, class 2 clearing 4× faster. `logit(1) = MIXL` with `MIXL` fixed, so
/// `±1000` makes one class certain: `P(class 1) = exp(-1000) = 0` in `f64` at `-1000`.
fn model(mixl: f64, mixture: bool) -> String {
    let (block, cl) = if mixture {
        (
            "[mixture]\n  nsub = 2\n  logit(1) = MIXL\n",
            "if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)",
        )
    } else {
        ("", "TVCL * exp(ETA_CL) + 0 * MIXL")
    };
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta MIXL({mixl}, -2000.0, 2000.0, FIX)\n  omega ETA_CL ~ 0.09\n  \
         sigma EPS ~ 0.04\n{block}[individual_parameters]\n  CL = {cl}\n  V  = TVV\n\
         [structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
         [error_model]\n  DV ~ proportional(EPS)\n\
         [fit_options]\n  method = foce\n  maxiter = 1\n  covariance = false\n\
         [simulation]\n  n_subjects = 4\n  dose_amt = 100\n  dose_cmt = 1\n  seed = 7\n  \
         times = [0.5, 2, 8]\n"
    )
}

/// The sdtab's `MIXNUM` column of a `--simulate` run of `text`, `None` when absent.
fn mixnum_column(text: &str) -> Option<Vec<f64>> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("sim.ferx");
    write!(std::fs::File::create(&path).unwrap(), "{text}").unwrap();
    let (result, population) =
        ferx_core::run_model_simulate(path.to_str().unwrap()).expect("simulate");
    ferx_core::io::output::sdtab(&result, &population)
        .into_iter()
        .find(|(name, _)| name == "MIXNUM")
        .map(|(_, v)| v)
}

/// Both certain classes, so a column stuck on either class (or on the fitted `MIXEST`,
/// which at one outer iteration need not agree) fails one side. 4 subjects × 3 times.
///
/// Mutation, run: dropping the `simulated_mixnum` write-back in `api::run` removes the
/// column (`None`), failing both certain-class arms.
#[test]
fn a_simulate_run_writes_the_generating_class_as_mixnum() {
    assert_eq!(
        mixnum_column(&model(-1000.0, true)),
        Some(vec![2.0; 12]),
        "every subject is certain to be class 2"
    );
    assert_eq!(
        mixnum_column(&model(1000.0, true)),
        Some(vec![1.0; 12]),
        "every subject is certain to be class 1"
    );
}

/// A model without `[mixture]` has no class to report, so the column is absent.
#[test]
fn a_simulate_run_without_a_mixture_writes_no_mixnum() {
    assert_eq!(mixnum_column(&model(0.0, false)), None);
}
