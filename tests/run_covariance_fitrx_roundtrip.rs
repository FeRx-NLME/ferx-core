//! #1815 T3 / T3b: `run_covariance` on a fit reloaded from `.fitrx` repeats the fit's
//! inline covariance step bit for bit, because the bundle carries the optimizer's packed
//! vector and the reloaded fit unpacks the very Cholesky factor the inline step used.
//!
//! Each discriminating row first asserts its **premise**: the same reloaded fit with
//! `packed_estimate` cleared — what every bundle was before #1815 — gives *different*
//! bits, so the row cannot pass on a fixture where re-decomposing `omega` happens to
//! land on the same factor. Which fixture discriminates is platform-dependent (the
//! re-decomposition's last bits go through the OS math library): measured at `160cc9a3`,
//! warfarin FOCE diverges by 9.7e-7 on macOS arm64 and 6.9e-5 on Linux/aarch64, and
//! two_cpt_oral_cov FOCEI by 1.5e-6 / 2.8e-7, so both rows discriminate on both
//! platforms. warfarin_iov FOCEI (the issue's own fixture) reloads bit-identically on
//! macOS even without the vector and diverges by 2.6e-8 on Linux, so its row asserts the
//! claim only.
//!
//! Mutations — `load_fit` reads `packed_estimate: None`: every claim dies; route
//! `PackedEstimate::Usable` to the fallback arm in `run_covariance`: every claim dies.
//!
//! Engines: every fixture runs analytic (`Dual2`) inner gradients, asserted per fit
//! through `gradient_method_inner`; the outer optimizer is the `auto` default.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::{
    fit, prepare_run, run_covariance, CompiledModel, EstimationMethod, FitOptions, FitResult,
    Population,
};

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

fn roundtrip_row(model_path: &str, data_path: &str, method: EstimationMethod, premise: bool) {
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
        "{model_path}: the bundle must carry the packed vector bit for bit"
    );
    if premise {
        let mut legacy = reloaded.clone();
        legacy.packed_estimate = None;
        assert!(
            covariance_bits(&f, &legacy, "legacy") != inline,
            "{model_path}: premise — without the packed vector the reloaded covariance \
             must differ from the inline step on this fixture"
        );
    }
    assert!(
        covariance_bits(&f, &reloaded, "reloaded") == inline,
        "{model_path}: run_covariance on the reloaded fit must repeat the inline step"
    );
}

#[test]
fn warfarin_foce_reload_repeats_inline_covariance() {
    roundtrip_row(
        "examples/warfarin.ferx",
        "data/warfarin.csv",
        EstimationMethod::Foce,
        true,
    );
}

#[test]
fn two_cpt_oral_cov_focei_reload_repeats_inline_covariance() {
    roundtrip_row(
        "examples/two_cpt_oral_cov.ferx",
        "data/two_cpt_oral_cov.csv",
        EstimationMethod::FoceI,
        true,
    );
}

#[test]
fn warfarin_laplace_reload_repeats_inline_covariance() {
    roundtrip_row(
        "examples/warfarin.ferx",
        "data/warfarin.csv",
        EstimationMethod::Laplace,
        true,
    );
}

/// The issue's "done when" fixture. Claim only: its premise holds on Linux and not on
/// macOS (module doc).
#[test]
fn warfarin_iov_focei_reload_repeats_inline_covariance() {
    roundtrip_row(
        "examples/warfarin_iov.ferx",
        "data/warfarin_iov.csv",
        EstimationMethod::FoceI,
        false,
    );
}
