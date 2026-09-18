//! Tier-2: an explicit front-end `threads` beats `[fit_options] threads` all the
//! way through to the thread count the fit actually ran on (#1416).
//!
//! The unit tests in `src/api/tests/threads_override_tests.rs` pin the merge
//! rule; this pins that the merged value is what `fit()` sizes its pool from,
//! for **both** production callers — the data path and `--simulate`. The
//! quantity asserted is `FitResult::n_threads_used`, which `fit()` reads off the
//! live Rayon pool, so it is the same number the CLI prints and writes to
//! `n_threads_used` in the fit YAML — the two readouts the bug report caught the
//! defect with.
//!
//! Each test is a differential pair: the same model file is run once with no
//! override and once with one. Without the control arm a fixture that never
//! honoured `[fit_options] threads` in the first place would pass the override
//! arm for the wrong reason.
//!
//! `maxiter = 1` keeps these in the PR job (Tier 2: a handful of outer
//! iterations, no convergence loop), and `checkpoint = false` keeps them from
//! writing a `.tmp` beside the test binary.

use ferx_core::{run_model_simulate_with_overrides, run_model_with_overrides, RunOverrides};

const DATA: &str = "data/warfarin.csv";

/// The warfarin example, trimmed to one outer iteration and pinned to
/// `threads = FILE_THREADS` in its `[fit_options]`.
const FILE_THREADS: usize = 2;

fn model_text() -> String {
    format!(
        "[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 1
  covariance = false
  checkpoint = false
  threads    = {FILE_THREADS}

[simulation]
  n_subjects = 10
  dose_amt   = 100.0
  dose_cmt   = 1
  times      = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
  seed       = 1
"
    )
}

/// Write the model into a fresh temp dir and return (dir, path). The dir is
/// returned so the caller keeps it alive for the duration of the run.
fn model_file(stem: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(format!("{stem}.ferx"));
    std::fs::write(&path, model_text()).expect("write model");
    let path = path.to_str().expect("utf-8 path").to_string();
    (dir, path)
}

fn override_warning(warnings: &[String]) -> Option<&String> {
    warnings
        .iter()
        .find(|w| w.contains("thread count overridden"))
}

#[test]
fn the_data_path_honours_an_explicit_thread_count_over_the_model_file() {
    let (_dir, path) = model_file("threads_data");

    // Control: no override, so the model file's count is what runs. This is also
    // the arm that would have gone green under the old behaviour.
    let (control, _) =
        run_model_with_overrides(&path, Some(DATA), &RunOverrides::default()).expect("control fit");
    assert_eq!(
        control.n_threads_used, FILE_THREADS,
        "the model file's `[fit_options] threads` must still be honoured when \
         nothing overrides it — otherwise the override arm below proves nothing"
    );
    assert!(
        override_warning(&control.warnings).is_none(),
        "no override, no warning: {:?}",
        control.warnings
    );

    // The bug: `--threads 1` against a file that says 2.
    let overrides = RunOverrides {
        threads: Some(1),
        ..Default::default()
    };
    let (pinned, _) =
        run_model_with_overrides(&path, Some(DATA), &overrides).expect("overridden fit");
    assert_eq!(
        pinned.n_threads_used, 1,
        "an explicit thread count must reach the pool the fit runs on"
    );

    let warning = override_warning(&pinned.warnings).expect("the override must announce itself");
    assert!(
        warning.contains(&format!("threads = {FILE_THREADS}")),
        "the warning must name the count it dropped: {warning}"
    );
    // And it must reach the typed surface the YAML/JSON output is built from,
    // not only the raw string list.
    assert!(
        pinned
            .warnings_structured
            .iter()
            .any(|w| w.message.contains("thread count overridden")),
        "the appended warning must be in `warnings_structured` too"
    );
}

/// The other half of the merge rule, per caller: `Some(0)` is `--threads 0` /
/// `auto`, a caller naming the engine's own worker count. A caller that filtered
/// it to `None` on the way in — the obvious "simplification", since `Some(0)`
/// and `None` mean the same thing to `FitOptions` — would silently let the model
/// file's pinned count stand, and every `Some(1)` assertion above would stay
/// green. Asserted on the warning rather than on `n_threads_used`: the engine
/// default is a property of the host's core count and could coincide with the
/// file's 2 on a small machine, whereas the warning fires iff the override
/// landed.
#[test]
fn an_explicitly_requested_default_overrides_the_model_file_on_both_callers() {
    let overrides = RunOverrides {
        threads: Some(0),
        ..Default::default()
    };

    let (_dir, path) = model_file("threads_zero_data");
    let (fitted, _) =
        run_model_with_overrides(&path, Some(DATA), &overrides).expect("data-path fit");
    let warning = override_warning(&fitted.warnings)
        .expect("`Some(0)` must override the data path's model file too");
    assert!(
        warning.contains("the default worker count")
            && warning.contains(&format!("threads = {FILE_THREADS}")),
        "{warning}"
    );

    let (_dir, path) = model_file("threads_zero_sim");
    let (simulated, _) =
        run_model_simulate_with_overrides(&path, &overrides).expect("simulate fit");
    assert!(
        override_warning(&simulated.warnings).is_some(),
        "`Some(0)` must override the simulate path's model file too: {:?}",
        simulated.warnings
    );
}

#[test]
fn simulate_honours_an_explicit_thread_count_over_the_model_file() {
    // `--simulate` reaches `fit()` through a different entry point, which built
    // its own `FitOptions` and so needed the merge applied separately. Mutating
    // one caller's merge must redden this test and not the one above.
    let (_dir, path) = model_file("threads_sim");

    let (control, _) =
        run_model_simulate_with_overrides(&path, &RunOverrides::default()).expect("control sim");
    assert_eq!(
        control.n_threads_used, FILE_THREADS,
        "the simulate path must honour the model file when nothing overrides it"
    );

    let overrides = RunOverrides {
        threads: Some(1),
        ..Default::default()
    };
    let (pinned, _) = run_model_simulate_with_overrides(&path, &overrides).expect("overridden sim");
    assert_eq!(pinned.n_threads_used, 1);
    assert!(
        override_warning(&pinned.warnings).is_some(),
        "the simulate path must announce the override as well: {:?}",
        pinned.warnings
    );
}
