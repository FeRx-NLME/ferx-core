//! #1847: `run_covariance` on the fit it was handed repeats the fit's inline covariance step
//! bit for bit, on every engine that has one, in memory and reloaded from `.fitrx`.
//!
//! Gauss–Newton, SAEM, IMP, IMPMAP and VI report the EBEs of their own warm-started mode
//! solves, and their inline step starts from those. They agree with a cold inner solve to
//! about 1e-7 but not bit for bit, and before #1847 `run_covariance` took anything within
//! 1e-6 of the cold solve to *be* the cold solve, so it started these five from a different
//! point and missed by up to 3.5e-7 relative SE. Now the cold solve is used only when the
//! reported EBEs are bit-equal to it, and `.fitrx` writes them exactly, so the reload keeps
//! those bits.
//!
//! **Claims**, per row: in memory and reloaded, the covariance matrix and every SE vector
//! are bit-identical to the inline step's, and `cov_fixed` is the same mask.
//!
//! Mutations (measured, see the PR):
//! - compare within the legacy 1e-6 band always (the pre-#1847 start rule): the five
//!   warm-started rows' in-memory claims die;
//! - write the bundle's CSV doubles at six decimals again: every reload claim of the
//!   warm-started rows dies.
//!
//! Engines: every fixture runs analytic (`Dual2`) inner gradients, asserted per fit through
//! `gradient_method_inner`. Fixtures: warfarin (and warfarin_iov for κ̂) on every PR;
//! two_cpt_oral_cov and standalone IMP behind `slow-tests`.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::{
    fit, prepare_run, run_covariance, CompiledModel, EstimationMethod, FitOptions, FitResult,
    Population,
};
use std::path::Path;

use EstimationMethod::*;

struct Fitted {
    row: String,
    model: CompiledModel,
    population: Population,
    fit: FitResult,
}

fn fit_inline(model_path: &str, data_path: &str, methods: &[EstimationMethod]) -> Fitted {
    let row = format!("{model_path} {methods:?}");
    let prep =
        prepare_run(model_path, Some(data_path)).unwrap_or_else(|e| panic!("{row}: prepare: {e}"));
    let (model, population, init) = (prep.parsed.model, prep.population, prep.init_params);
    let last = *methods.last().unwrap();
    let options = FitOptions {
        verbose: false,
        method: last,
        methods: if methods.len() > 1 {
            methods.to_vec()
        } else {
            Vec::new()
        },
        interaction: last != Foce,
        run_covariance_step: true,
        sir: false,
        ..FitOptions::default()
    };
    let fit =
        fit(&model, &population, &init, &options).unwrap_or_else(|e| panic!("{row}: fit: {e}"));
    assert!(
        fit.gradient_method_inner.starts_with("analytic"),
        "{row}: the engine this file names moved: {}",
        fit.gradient_method_inner
    );
    Fitted {
        row,
        model,
        population,
        fit,
    }
}

/// The covariance matrix and every SE vector, as bits, in that order. All finite: a `NaN`
/// compares unequal and could fake a premise.
fn bits_of(fit: &FitResult, what: &str) -> Vec<Vec<u64>> {
    let cov = fit
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: covariance step produced no matrix"));
    let mut out = vec![cov.iter().copied().collect::<Vec<f64>>()];
    for (name, se) in [
        (
            "se_theta",
            fit.se_theta.as_ref().map(|v| v.as_slice().to_vec()),
        ),
        ("se_omega", fit.se_omega.clone()),
        ("se_sigma", fit.se_sigma.clone()),
        ("se_kappa", fit.se_kappa.clone()),
    ] {
        if name != "se_kappa" || se.is_some() {
            out.push(se.unwrap_or_else(|| panic!("{what}: no {name}")));
        }
    }
    for v in &out {
        assert!(v.iter().all(|x| x.is_finite()), "{what}: non-finite entry");
    }
    out.into_iter()
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn rerun(f: &Fitted, fit: &FitResult, what: &str) -> Vec<Vec<u64>> {
    let options = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let out = run_covariance(fit, Some(&f.model), Some(&f.population), &options)
        .unwrap_or_else(|e| panic!("{what}: run_covariance: {e}"));
    assert!(!out.cov_fixed.is_empty(), "{what}: a matrix has a mask");
    assert_eq!(
        out.cov_fixed, f.fit.cov_fixed,
        "{what}: the inline and post-hoc steps held the same coordinates (#2695)"
    );
    bits_of(&out, what)
}

fn save(f: &Fitted, path: &Path) {
    save_fit(
        &f.fit,
        &f.population,
        "model\n",
        path,
        SaveFitOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{}: save_fit: {e}", f.row));
}

/// One row: the in-memory and reload claims (module doc).
fn check_row(model_path: &str, data_path: &str, methods: &[EstimationMethod]) {
    let f = fit_inline(model_path, data_path, methods);
    let row = f.row.clone();
    let inline = bits_of(&f.fit, &format!("{row} inline"));

    assert!(
        rerun(&f, &f.fit, &format!("{row} in memory")) == inline,
        "{row}: run_covariance on the fit in memory must repeat the inline step"
    );

    let dir = tempfile::tempdir().unwrap();
    let exact = dir.path().join("fit.fitrx");
    save(&f, &exact);
    let reloaded = load_fit(&exact)
        .unwrap_or_else(|e| panic!("{row}: load_fit: {e}"))
        .fit;
    for (a, b) in reloaded.subjects.iter().zip(&f.fit.subjects) {
        assert!(
            a.eta
                .iter()
                .zip(b.eta.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{row}: the bundle must carry the EBEs bit for bit (subject {})",
            b.id
        );
    }
    // Flattened: a model without κ holds one empty list per subject in memory and none
    // after a reload, and both are "no κ̂".
    let kappa_bits = |r: &FitResult| {
        r.ebe_kappas
            .iter()
            .flatten()
            .flat_map(|k| k.iter().map(|x| x.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        kappa_bits(&reloaded),
        kappa_bits(&f.fit),
        "{row}: the bundle must carry κ̂ bit for bit"
    );
    assert_eq!(
        f.model.n_kappa > 0,
        !kappa_bits(&f.fit).is_empty(),
        "{row}: premise — an IOV row has κ̂ to carry, the others none"
    );
    assert!(
        rerun(&f, &reloaded, &format!("{row} reloaded")) == inline,
        "{row}: run_covariance on the reloaded fit must repeat the inline step"
    );
}

const WARFARIN: (&str, &str) = ("examples/warfarin.ferx", "data/warfarin.csv");
const TWO_CPT: (&str, &str) = (
    "examples/two_cpt_oral_cov.ferx",
    "data/two_cpt_oral_cov.csv",
);

#[test]
fn warfarin_foce() {
    check_row(WARFARIN.0, WARFARIN.1, &[Foce]);
}

#[test]
fn warfarin_focei() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceI]);
}

#[test]
fn warfarin_gn_hybrid() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceGnHybrid]);
}

/// κ̂ through `ebes_kappa.csv`.
#[test]
fn warfarin_iov_focei() {
    check_row(
        "examples/warfarin_iov.ferx",
        "data/warfarin_iov.csv",
        &[FoceI],
    );
}

/// κ̂ per group of each nested level, which the optimizer's warm caches carry since #2261.
#[test]
fn nested_levels_3_focei() {
    check_row(
        "nonmem_anchor/nested_levels_3.ferx",
        "nonmem_anchor/nested_levels_3.csv",
        &[FoceI],
    );
}

#[test]
fn warfarin_gn() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceGn]);
}

#[test]
fn warfarin_saem() {
    check_row(WARFARIN.0, WARFARIN.1, &[Saem]);
}

#[test]
fn warfarin_imp_after_focei() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceI, Imp]);
}

#[test]
fn warfarin_impmap() {
    check_row(WARFARIN.0, WARFARIN.1, &[Impmap]);
}

#[test]
fn warfarin_vi() {
    check_row(WARFARIN.0, WARFARIN.1, &[Vi]);
}

const WARFARIN_IOV: (&str, &str) = ("examples/warfarin_iov.ferx", "data/warfarin_iov.csv");

/// κ̂ on the two warm-started engines whose final EBE solve starts at the joint mode since
/// #2261 (SAEM at the chain's last κ, VI at the variational κ means). IMP and IMPMAP refuse
/// IOV models, so they have no such row.
#[test]
fn warfarin_iov_saem() {
    check_row(WARFARIN_IOV.0, WARFARIN_IOV.1, &[Saem]);
}

#[test]
fn warfarin_iov_vi() {
    check_row(WARFARIN_IOV.0, WARFARIN_IOV.1, &[Vi]);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn two_cpt_oral_cov_warm_started_engines() {
    for methods in [&[FoceGn][..], &[Saem], &[FoceI, Imp], &[Impmap], &[Vi]] {
        check_row(TWO_CPT.0, TWO_CPT.1, methods);
    }
}

/// Standalone IMP from a cold start. Before #2331 its proposal collapsed on warfarin (ETA
/// shrinkage near 100%) and the inline step failed with `W_COV_NON_STATIONARY_ESTIMATE`, so
/// this row only pinned that `run_covariance` failed the same way (#1847). Since #2331
/// recentres the proposal, the fit converges and has a successful covariance step to
/// repeat, so it is an ordinary warm-started row. Slow-gated as a fit to convergence, though
/// no longer for its cost: 1.6 s on Linux aarch64 and 2.1 s on macOS arm64 since #2331,
/// against 183 s before it (#2183).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn warfarin_standalone_imp() {
    check_row(WARFARIN.0, WARFARIN.1, &[Imp]);
}
