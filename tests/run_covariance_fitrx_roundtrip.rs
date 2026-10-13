//! #1815 T3 / T3b: `run_covariance` on a fit reloaded from `.fitrx` repeats the fit's
//! inline covariance step bit for bit, because the bundle carries the optimizer's packed
//! vector and the reloaded fit unpacks the very Cholesky factor the inline step used.
//!
//! **Claim**, asserted on every row: the reloaded fit repeats the inline step bit for bit.
//!
//! **Premise**, asserted over the set: the same reloaded fit with `packed_estimate`
//! cleared — what every bundle was before #1815 — gives *different* bits, so the claim
//! cannot pass only because re-decomposing `omega` happens to land on the same factor.
//! A row whose premise fails still checks its claim; it just cannot tell the #1815 fix
//! from the pre-#1815 fallback.
//!
//! The premise is asserted over the set, not per row, because no single row's premise
//! is platform-stable (#1929). The legacy path re-decomposes `omega`, and the last bits
//! of that factor go through the OS math library, so whether it lands exactly on the
//! packed factor is a property of the libm, not of the fixture. Measured: the row that
//! reloads bit-identically without the vector moves with the platform — warfarin FOCE
//! on Linux x86_64 / glibc 2.28, warfarin_iov FOCEI on macOS arm64 — and the gaps that
//! do exist move between commits on one host (warfarin FOCE on macOS arm64: 9.7e-7 at
//! `160cc9a3`, 1.4e-11 at `afe7f52b`). A per-row table goes stale, so the test prints
//! the set instead: one `premise` line per row with its max |Δ| over the covariance
//! matrix and `se_theta`, then the rows that discriminate. It requires at least
//! `MIN_DISCRIMINATING` of them: one is enough to kill the mutation below, two leave
//! headroom for one more row landing bit-identical on some platform.
//!
//! Mutations (#1929, macOS arm64) — route `PackedEstimate::Usable` to the fallback arm
//! in `run_covariance`: the claim dies on the first row whose premise holds (warfarin
//! FOCE there). `load_fit` reads `packed_estimate: None`: the bundle assert dies first,
//! on row one; with that assert removed, the claim dies on the same row as above.
//!
//! Engines: every fixture runs analytic (`Dual2`) inner gradients, asserted per fit
//! through `gradient_method_inner`; the outer optimizer is the `auto` default.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::{
    fit, prepare_run, run_covariance, CompiledModel, EstimationMethod, FitOptions, FitResult,
    Population,
};

/// Rows whose premise must hold, out of `ROWS` (module doc).
const MIN_DISCRIMINATING: usize = 2;

const ROWS: [(&str, &str, EstimationMethod); 4] = [
    (
        "examples/warfarin.ferx",
        "data/warfarin.csv",
        EstimationMethod::Foce,
    ),
    (
        "examples/two_cpt_oral_cov.ferx",
        "data/two_cpt_oral_cov.csv",
        EstimationMethod::FoceI,
    ),
    // Review r1 #3: the docs' Laplace claim.
    (
        "examples/warfarin.ferx",
        "data/warfarin.csv",
        EstimationMethod::Laplace,
    ),
    // The issue's "done when" fixture.
    (
        "examples/warfarin_iov.ferx",
        "data/warfarin_iov.csv",
        EstimationMethod::FoceI,
    ),
];

struct Fitted {
    model: CompiledModel,
    population: Population,
    fit: FitResult,
}

fn fit_inline(model_path: &str, data_path: &str, method: EstimationMethod) -> Fitted {
    let prep = prepare_run(model_path, Some(data_path)).expect("prepare");
    let (model, population, init) = (prep.parsed.model, prep.population, prep.init_params);
    let options = FitOptions {
        verbose: false,
        method,
        interaction: method != EstimationMethod::Foce,
        run_covariance_step: true,
        sir: false,
        ..FitOptions::default()
    };
    let fit = fit(&model, &population, &init, &options)
        .unwrap_or_else(|e| panic!("{model_path}: fit: {e}"));
    assert!(
        fit.gradient_method_inner.starts_with("analytic"),
        "{model_path}: the engine this file names moved: {}",
        fit.gradient_method_inner
    );
    assert!(
        fit.packed_estimate.is_some(),
        "{model_path}: a packed-space fit must carry packed_estimate"
    );
    Fitted {
        model,
        population,
        fit,
    }
}

fn reload(f: &Fitted) -> FitResult {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fit.fitrx");
    save_fit(
        &f.fit,
        &f.population,
        "model\n",
        &path,
        SaveFitOptions::default(),
    )
    .expect("save_fit");
    load_fit(&path).expect("load_fit").fit
}

fn covariance_bits(f: &Fitted, fit: &FitResult, what: &str) -> (Vec<u64>, Vec<u64>) {
    let options = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let out = run_covariance(fit, Some(&f.model), Some(&f.population), &options)
        .unwrap_or_else(|e| panic!("{what}: run_covariance: {e}"));
    bits_of(&out, what)
}

/// The covariance matrix and `se_theta`, as bits. Both must be finite: a `NaN` would
/// compare unequal and could fake a premise.
fn bits_of(fit: &FitResult, what: &str) -> (Vec<u64>, Vec<u64>) {
    let cov = fit
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: covariance step produced no matrix"));
    assert!(cov.iter().all(|x| x.is_finite()), "{what}: non-finite cov");
    let se = fit
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: no se_theta"));
    assert!(se.iter().all(|x| x.is_finite()), "{what}: non-finite se");
    (
        cov.iter().map(|x| x.to_bits()).collect(),
        se.iter().map(|x| x.to_bits()).collect(),
    )
}

/// Max |Δ| of two bit vectors read back as `f64`. Both sides are finite (`bits_of`),
/// so `f64::max` has no `NaN` to swallow.
fn max_gap(a: &[u64], b: &[u64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from_bits(*x) - f64::from_bits(*y)).abs())
        .fold(0.0, f64::max)
}

/// Asserts the row's claim, then returns its premise: `Some(max |Δ|)` between the legacy
/// reload (no packed vector) and the inline step when their bits differ, `None` when
/// they are bit-identical.
fn roundtrip_row(
    row: &str,
    model_path: &str,
    data_path: &str,
    method: EstimationMethod,
) -> Option<f64> {
    let f = fit_inline(model_path, data_path, method);
    let inline = bits_of(&f.fit, "inline");
    let reloaded = reload(&f);
    assert_eq!(
        reloaded
            .packed_estimate
            .as_deref()
            .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
        f.fit
            .packed_estimate
            .as_deref()
            .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
        "{row}: the bundle must carry the packed vector bit for bit"
    );
    assert!(
        covariance_bits(&f, &reloaded, "reloaded") == inline,
        "{row}: claim — run_covariance on the reloaded fit must repeat the inline step"
    );
    let mut legacy = reloaded;
    legacy.packed_estimate = None;
    let legacy = covariance_bits(&f, &legacy, "legacy");
    (legacy != inline).then(|| max_gap(&legacy.0, &inline.0).max(max_gap(&legacy.1, &inline.1)))
}

#[test]
fn reload_repeats_inline_covariance() {
    let mut discriminating = Vec::new();
    let mut identical = Vec::new();
    for (model_path, data_path, method) in ROWS {
        let row = format!("{model_path} {method:?}");
        match roundtrip_row(&row, model_path, data_path, method) {
            Some(gap) => {
                eprintln!("premise {row}: max |legacy - inline| = {gap:e}");
                discriminating.push(row);
            }
            None => {
                eprintln!("premise {row}: bit-identical without the packed vector");
                identical.push(row);
            }
        }
    }
    eprintln!("premise held on {discriminating:?}; failed on {identical:?}");
    assert!(
        discriminating.len() >= MIN_DISCRIMINATING,
        "premise — at least {MIN_DISCRIMINATING} of {} rows must give different covariance \
         bits without the packed vector, or the claim cannot tell the #1815 fix from the \
         pre-#1815 fallback; held on {discriminating:?}, failed on {identical:?}",
        ROWS.len()
    );
}
