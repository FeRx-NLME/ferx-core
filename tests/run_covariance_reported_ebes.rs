//! #2026: `run_covariance` on the fit it was handed repeats the inline covariance step bit for
//! bit, whichever EBE candidate the fit's final step reported.
//!
//! The final inner loop solves cold, and when the cold solve misses the objective the
//! optimizer measured it also scores the incumbent EBEs and a re-solve seeded from them,
//! reporting the lowest (#833). The inline covariance step runs from the reported EBEs;
//! `run_covariance` used to re-solve cold, so on a fit that reported a non-cold candidate the
//! two disagreed. The rows straddle that fork, each asserting its own arm through the
//! `W_EBE_START_DEPENDENT` warning the final step writes (or does not):
//!
//! - **cold reported** (default `inner_maxiter`): the cold solve reproduces the objective, no
//!   warning is written, and the reported EBEs are the cold ones. Also reloaded from `.fitrx`,
//!   whose six-decimal EBEs must still be recognised as the cold solve — the claim there is
//!   bit identity too.
//! - **non-cold reported** (`inner_maxiter = 7`): the warm-started trajectory reaches
//!   warfarin's FOCEI optimum, but seven inner iterations from η = 0 do not, so the cold
//!   solve scores *worse* and a non-cold candidate is reported. Both covariance methods: the
//!   default `r` reads the EBEs only as its warm start, the score cross-product (`s`) reads
//!   their `H` as well.
//!
//! The split is a budget, not a trajectory: both rows stop at the same OFV at `outer_maxiter`
//! 30 and 200 (macOS arm64; the starved row reports `converged = false` there), and the cold
//! solve is then a fixed number of steps from a fixed start at that point. On macOS arm64 and
//! Linux x86_64 both rows reach OFV −286.0042 (to the premise's 1e-3) and the
//! `inner_maxiter = 7` row reports the cold solve worse — by 20.58 OFV units on macOS arm64.
//! An earlier fixture split cold from warm with an ODE step cap, and its arm flipped between
//! the two platforms.
//!
//! `run_covariance` is handed the fit's own options, as a caller repeating the step would: a
//! different `inner_maxiter` changes the cold solve it compares the reported EBEs against.
//!
//! Mutations: return the cold solve unconditionally from `reported_ebe_inputs` → both
//! `non_cold` rows die; drop `REPORTED_EBE_TOL` to zero → the reload row dies; rebuild `H` at
//! the cold EBEs instead of the reported ones → the `s` row dies.
//!
//! Engines: analytic (`Dual2`) inner gradients, asserted per fit through
//! `gradient_method_inner`; the outer optimizer is the `auto` default.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::{
    fit, prepare_run, run_covariance, CompiledModel, CovarianceMethod, EstimationMethod,
    FitOptions, FitResult, Population,
};

struct Fitted {
    model: CompiledModel,
    population: Population,
    options: FitOptions,
    fit: FitResult,
}

/// The inner budget at which the cold re-solve falls short of the warm trajectory.
const STARVED_INNER: usize = 7;

/// Warfarin under FOCEI with `inner_maxiter` (`None` = the default). `arm` is a phrase of
/// the `W_EBE_START_DEPENDENT` warning that names the row's arm, or `None` for no warning.
fn fitted(inner_maxiter: Option<usize>, method: CovarianceMethod, arm: Option<&str>) -> Fitted {
    let what = format!("inner_maxiter = {inner_maxiter:?}, {method:?}");
    let prep = prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
        .unwrap_or_else(|e| panic!("{what}: prepare: {e}"));
    let (model, population, init) = (prep.parsed.model, prep.population, prep.init_params);
    let defaults = FitOptions::default();
    let options = FitOptions {
        verbose: false,
        method: EstimationMethod::FoceI,
        run_covariance_step: true,
        sir: false,
        inner_maxiter: inner_maxiter.unwrap_or(defaults.inner_maxiter),
        covariance_method: method,
        ..defaults
    };
    let fit =
        fit(&model, &population, &init, &options).unwrap_or_else(|e| panic!("{what}: fit: {e}"));
    assert!(
        fit.gradient_method_inner.starts_with("analytic"),
        "{what}: the engine this file names moved: {}",
        fit.gradient_method_inner
    );
    assert!(
        (fit.ofv - (-286.0042)).abs() < 1e-3,
        "{what}: premise — the fit reaches warfarin's FOCEI optimum (−286.0042), got {}",
        fit.ofv
    );
    let warning = fit
        .warnings
        .iter()
        .find(|w| w.starts_with("W_EBE_START_DEPENDENT"));
    match (arm, warning) {
        (None, None) => {}
        (Some(arm), Some(w)) if w.contains(arm) => {}
        _ => panic!("{what}: premise — the row's arm moved; expected {arm:?}, got {warning:?}"),
    }
    Fitted {
        model,
        population,
        options,
        fit,
    }
}

/// The covariance matrix, as bits. It must be finite: a `NaN` compares unequal.
fn bits(fit: &FitResult, what: &str) -> Vec<u64> {
    let cov = fit
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: covariance step produced no matrix"));
    assert!(
        cov.iter().all(|x| x.is_finite()),
        "{what}: non-finite covariance"
    );
    cov.iter().map(|x| x.to_bits()).collect()
}

/// `run_covariance` on `fit` under the options the fit ran with.
fn rerun(f: &Fitted, fit: &FitResult, what: &str) -> Vec<u64> {
    let out = run_covariance(fit, Some(&f.model), Some(&f.population), &f.options)
        .unwrap_or_else(|e| panic!("{what}: run_covariance: {e}"));
    bits(&out, what)
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

const COLD_WORSE: &str = "worse than the EBEs the optimizer minimised against";

#[test]
fn cold_reported_repeats_inline_covariance_in_memory_and_after_reload() {
    let f = fitted(None, CovarianceMethod::Hessian, None);
    let inline = bits(&f.fit, "cold reported: inline");
    assert!(
        rerun(&f, &f.fit, "cold reported") == inline,
        "cold reported: run_covariance must repeat the inline step"
    );
    let back = reload(&f);
    assert!(
        back.subjects
            .iter()
            .zip(&f.fit.subjects)
            .any(|(b, a)| b.eta != a.eta),
        "premise: the .fitrx round trip must round the EBEs"
    );
    assert!(
        rerun(&f, &back, "cold reported, reloaded") == inline,
        "cold reported, reloaded: the rounded EBEs must still be taken as the cold solve"
    );
}

#[test]
fn non_cold_reported_repeats_inline_covariance_r() {
    let f = fitted(
        Some(STARVED_INNER),
        CovarianceMethod::Hessian,
        Some(COLD_WORSE),
    );
    assert!(
        rerun(&f, &f.fit, "non-cold reported, r") == bits(&f.fit, "inline"),
        "non-cold reported, r: run_covariance must start from the reported EBEs"
    );
}

#[test]
fn non_cold_reported_repeats_inline_covariance_s() {
    let f = fitted(
        Some(STARVED_INNER),
        CovarianceMethod::CrossProduct,
        Some(COLD_WORSE),
    );
    assert!(
        rerun(&f, &f.fit, "non-cold reported, s") == bits(&f.fit, "inline"),
        "non-cold reported, s: run_covariance must score at the reported EBEs and their H"
    );
}
