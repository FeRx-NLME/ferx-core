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
//! are bit-identical to the inline step's.
//!
//! **Legacy bundle**, per row: the same bundle with its CSVs rewritten at six decimals —
//! byte for byte what `save_fit` wrote before #1847 — still loads, and `run_covariance`
//! runs it from the cold solve, as it always did. On FOCE / FOCEI / `gn_hybrid` the cold
//! solve *is* the reported one, so that is bit-identical to the inline step; starting from
//! the rounded EBEs instead would not be (measured 5.4e-8 to 2.8e-7 on #1847). On the five
//! warm-started engines it is not bit-identical, which is the premise that the claim
//! above discriminates: the same fit can miss. The old bundle's fit is then saved again by
//! today's writer and reloaded: it must keep its EBEs bit for bit and give the same
//! covariance, so a re-saved old fit is still recognised.
//!
//! Mutations (measured, see the PR):
//! - compare within the legacy 1e-6 band always (the pre-#1847 start rule): the five
//!   warm-started rows' in-memory claims die;
//! - write the bundle's CSV doubles at six decimals again: every reload claim of the five
//!   dies, and the control rows' reloads stay green (they run cold either way);
//! - compare bits always, with no legacy band: the control rows' legacy claims die.
//!
//! Engines: every fixture runs analytic (`Dual2`) inner gradients, asserted per fit through
//! `gradient_method_inner`. Fixtures: warfarin (and warfarin_iov for κ̂) on every PR;
//! two_cpt_oral_cov and standalone IMP behind `slow-tests`.

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::{
    fit, prepare_run, run_covariance, CompiledModel, EstimationMethod, FitOptions, FitResult,
    Population,
};
use std::io::{Read, Write};
use std::path::Path;

use EstimationMethod::*;

/// Whether the engine reports a cold solve's EBEs (the control rows) or its own
/// warm-started ones (the rows #1847 fixed).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Ebes {
    Cold,
    WarmStarted,
}

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

/// Worst relative gap over the flattened bit vectors; both sides are finite (`bits_of`), so
/// `f64::max` has no `NaN` to swallow.
fn max_rel_gap(a: &[Vec<u64>], b: &[Vec<u64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| {
            let (x, y) = (f64::from_bits(*x), f64::from_bits(*y));
            (x - y).abs() / y.abs().max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max)
}

fn rerun(f: &Fitted, fit: &FitResult, what: &str) -> Vec<Vec<u64>> {
    let options = FitOptions {
        verbose: false,
        ..FitOptions::default()
    };
    let out = run_covariance(fit, Some(&f.model), Some(&f.population), &options)
        .unwrap_or_else(|e| panic!("{what}: run_covariance: {e}"));
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

/// A CSV cell as a bundle written before #1847 spelled it: a non-integer number at six
/// decimals (`io::output::fmt_num`); integers (`ID`, `n_obs`, `OCC`), text and empty cells
/// as they are.
fn six_decimal_cell(cell: &str) -> String {
    if cell.parse::<i64>().is_ok() {
        return cell.to_string();
    }
    match cell.parse::<f64>() {
        Ok(x) if x.is_finite() => format!("{x:.6}"),
        _ => cell.to_string(),
    }
}

/// Rewrites `src` to `dst` as a bundle written before #1847: every CSV entry's numbers at
/// six decimals, every other entry byte for byte. The data rows of these bundles carry no
/// quoted cells, so a plain comma split is the row.
fn write_legacy_bundle(src: &Path, dst: &Path) {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let mut out = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    let opts = zip::write::SimpleFileOptions::default();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).unwrap();
        let name = entry.name().unwrap().to_string();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).unwrap();
        if name.ends_with(".csv") {
            let text = String::from_utf8(bytes).unwrap();
            let mut lines = text.lines();
            let mut rewritten = format!("{}\n", lines.next().unwrap());
            for line in lines {
                assert!(!line.contains('"'), "{name}: a quoted cell");
                let cells: Vec<String> = line.split(',').map(six_decimal_cell).collect();
                rewritten.push_str(&cells.join(","));
                rewritten.push('\n');
            }
            bytes = rewritten.into_bytes();
        }
        out.start_file(name, opts).unwrap();
        out.write_all(&bytes).unwrap();
    }
    out.finish().unwrap();
}

/// One row: the in-memory and reload claims, then the legacy bundle (module doc).
fn check_row(model_path: &str, data_path: &str, methods: &[EstimationMethod], ebes: Ebes) {
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

    let old = dir.path().join("legacy.fitrx");
    write_legacy_bundle(&exact, &old);
    let legacy = load_fit(&old)
        .unwrap_or_else(|e| panic!("{row}: a pre-#1847 bundle must still load: {e}"))
        .fit;
    assert!(
        legacy
            .subjects
            .iter()
            .zip(&f.fit.subjects)
            .any(|(a, b)| a.eta != b.eta),
        "{row}: premise — the legacy bundle must round the EBEs"
    );
    let from_legacy = rerun(&f, &legacy, &format!("{row} legacy"));
    match ebes {
        Ebes::Cold => assert!(
            from_legacy == inline,
            "{row}: a pre-#1847 bundle must run from the cold solve, which repeats the \
             inline step here; off by {:e}",
            max_rel_gap(&from_legacy, &inline)
        ),
        Ebes::WarmStarted => {
            assert!(
                from_legacy != inline,
                "{row}: premise — the cold start a pre-#1847 bundle takes must miss the \
                 inline step, or the claims above cannot tell the start rule apart"
            );
            eprintln!(
                "{row}: pre-#1847 bundle (cold start) vs inline: max rel {:e}",
                max_rel_gap(&from_legacy, &inline)
            );
        }
    }

    // Re-saved (#2092 review): a pre-#1847 fit saved again by today's writer keeps its
    // six-decimal EBEs bit for bit, so it is still recognised as one and still runs from the
    // cold solve. Losing that would start it from its rounded EBEs instead — off the
    // controls' inline step, and off the cold start on the warm-started rows.
    let resaved_path = dir.path().join("resaved.fitrx");
    save_fit(
        &legacy,
        &f.population,
        "model\n",
        &resaved_path,
        SaveFitOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{row}: re-saving a pre-#1847 fit: {e}"));
    let resaved = load_fit(&resaved_path)
        .unwrap_or_else(|e| panic!("{row}: reloading a re-saved pre-#1847 fit: {e}"))
        .fit;
    let eta_bits = |r: &FitResult| {
        r.subjects
            .iter()
            .flat_map(|s| s.eta.iter().map(|x| x.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        (eta_bits(&resaved), kappa_bits(&resaved)),
        (eta_bits(&legacy), kappa_bits(&legacy)),
        "{row}: re-saving a pre-#1847 fit must keep its six-decimal EBEs and κ̂"
    );
    assert!(
        rerun(&f, &resaved, &format!("{row} re-saved legacy")) == from_legacy,
        "{row}: a re-saved pre-#1847 fit must still be recognised and run from the cold solve"
    );
}

const WARFARIN: (&str, &str) = ("examples/warfarin.ferx", "data/warfarin.csv");
const TWO_CPT: (&str, &str) = (
    "examples/two_cpt_oral_cov.ferx",
    "data/two_cpt_oral_cov.csv",
);

#[test]
fn warfarin_foce() {
    check_row(WARFARIN.0, WARFARIN.1, &[Foce], Ebes::Cold);
}

#[test]
fn warfarin_focei() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceI], Ebes::Cold);
}

#[test]
fn warfarin_gn_hybrid() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceGnHybrid], Ebes::Cold);
}

/// κ̂ through `ebes_kappa.csv`.
#[test]
fn warfarin_iov_focei() {
    check_row(
        "examples/warfarin_iov.ferx",
        "data/warfarin_iov.csv",
        &[FoceI],
        Ebes::Cold,
    );
}

#[test]
fn warfarin_gn() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceGn], Ebes::WarmStarted);
}

#[test]
fn warfarin_saem() {
    check_row(WARFARIN.0, WARFARIN.1, &[Saem], Ebes::WarmStarted);
}

#[test]
fn warfarin_imp_after_focei() {
    check_row(WARFARIN.0, WARFARIN.1, &[FoceI, Imp], Ebes::WarmStarted);
}

#[test]
fn warfarin_impmap() {
    check_row(WARFARIN.0, WARFARIN.1, &[Impmap], Ebes::WarmStarted);
}

#[test]
fn warfarin_vi() {
    check_row(WARFARIN.0, WARFARIN.1, &[Vi], Ebes::WarmStarted);
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn two_cpt_oral_cov_warm_started_engines() {
    for methods in [&[FoceGn][..], &[Saem], &[FoceI, Imp], &[Impmap], &[Vi]] {
        check_row(TWO_CPT.0, TWO_CPT.1, methods, Ebes::WarmStarted);
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
    check_row(WARFARIN.0, WARFARIN.1, &[Imp], Ebes::WarmStarted);
}
