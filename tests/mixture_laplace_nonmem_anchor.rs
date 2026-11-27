//! #3025's NONMEM anchor: a two-class `[mixture]` model under Laplace, against
//! `nonmem_anchor/mixture_laplace.ctl` (`$MIX NSPOP=2`, `$EST METHOD=COND LAPLACIAN INTER`,
//! `$COV MATRIX=R`; NONMEM 7.5.1 on klebsiella, `nonmem_anchor/results/mixture_laplace.{nm.json,ext}`).
//!
//! The ferx twin is `nonmem_anchor/mixture_laplace.ferx` on `nonmem_anchor/mixture_iv.csv`.
//! NONMEM writes the mixing fraction as `P(1) = THETA(4)`, ferx as `logit(1) = MIXL`, so
//! the fourth θ is mapped through `logit` going in and its SE through the delta method
//! `SE(p) = p(1−p)·SE(MIXL)` coming out.
//!
//! **Tier 2** (every PR): one evaluation at NONMEM's optimum (`maxiter = 0`, OFV and the
//! covariance step) and one fit from the model file's initial estimates (about 50
//! evaluations on 30 subjects).

use ferx_core::run_model_with_data;
use ferx_core::types::FitResult;
use std::path::PathBuf;

const MODEL: &str = "nonmem_anchor/mixture_laplace.ferx";
const DATA: &str = "nonmem_anchor/mixture_iv.csv";
const EXT: &str = "nonmem_anchor/results/mixture_laplace.ext";

/// The `-1000000000` (estimate) and `-1000000001` (SE) rows of the `.ext`: θ1..θ4, OBJ.
fn nonmem() -> ([f64; 4], [f64; 4], f64) {
    let text = std::fs::read_to_string(EXT).unwrap();
    let row = |tag: &str| -> Vec<f64> {
        text.lines()
            .map(|l| l.split_whitespace().collect::<Vec<_>>())
            .find(|f| f.first() == Some(&tag))
            .unwrap_or_else(|| panic!("{EXT}: no {tag} row"))[1..]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect()
    };
    let (est, se) = (row("-1000000000"), row("-1000000001"));
    // Columns: THETA1..4, SIGMA(1,1), OMEGA(1,1), OBJ.
    ([est[0], est[1], est[2], est[3]], [se[0], se[1], se[2], se[3]], est[6])
}

/// The model file with `edit` applied, run on the anchor data.
fn run(name: &str, edit: impl Fn(String) -> String) -> FitResult {
    let text = edit(std::fs::read_to_string(MODEL).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join(format!("{name}.ferx"));
    std::fs::write(&path, text).unwrap();
    let (fit, _) = run_model_with_data(path.to_str().unwrap(), Some(DATA))
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    fit
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn inv_logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// ferx's θ and SEs on NONMEM's scale (`P(1)` for the mixing coefficient).
fn on_nonmem_scale(f: &FitResult) -> ([f64; 4], [f64; 4]) {
    let se = f.se_theta.as_ref().expect("covariance step ran");
    let p = inv_logit(f.theta[3]);
    ([f.theta[0], f.theta[1], f.theta[2], p], [se[0], se[1], se[2], p * (1.0 - p) * se[3]])
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// |ΔOFV| bound. Measured on macOS arm64: −3.8e-7 at NONMEM's optimum, −4.8e-7 after the
/// free fit. Scoring the same point with the FOCEI mixture objective instead moves it by
/// ~0.19 (FOCEI's own optimum on this data is 301.582), so the bound tells the two apart.
const OFV_AT_NONMEM: f64 = 1e-5;
/// Per-SE relative bounds, TVCL1 / TVCL2 / TVV / P(1). Measured: 2.7e-3, 7.0e-3, 4.1e-4,
/// 1.7e-4. ferx's R is exact; NONMEM's is a finite difference, and TVCL2 shows the same
/// ~0.7 % gap on the FOCEI anchor (`tests/mixture_nonmem.rs`, `mixture_iv_cov.ctl`).
const SE_REL: [f64; 4] = [5e-3, 1.2e-2, 1e-3, 1e-3];
/// Relative θ bound for the free fit. Measured worst 2.4e-5 (TVCL2).
const THETA_REL: f64 = 1e-4;

/// At NONMEM's optimum ferx's Laplace mixture objective reproduces NONMEM's, and its exact
/// R-matrix SEs match `$COV MATRIX=R`.
#[test]
fn laplace_mixture_reproduces_nonmem_at_its_optimum() {
    let (theta, nm_se, nm_ofv) = nonmem();
    let fit = run("at_nonmem", |t| {
        let t = t
            .replace("TVCL1(1.2,", &format!("TVCL1({},", theta[0]))
            .replace("TVCL2(2.5,", &format!("TVCL2({},", theta[1]))
            .replace("TVV(10.0,", &format!("TVV({},", theta[2]))
            .replace("MIXL(0.0,", &format!("MIXL({},", logit(theta[3])))
            .replace("  covariance = true\n", "  covariance = true\n  maxiter = 0\n");
        assert!(t.contains("maxiter = 0") && !t.contains("TVCL1(1.2,"), "fixture moved");
        t
    });
    let d_ofv = fit.ofv - nm_ofv;
    eprintln!("ΔOFV (ferx − NONMEM) at NONMEM's optimum: {d_ofv:+.3e}");
    assert!(d_ofv.abs() < OFV_AT_NONMEM, "ferx {} vs NONMEM {nm_ofv}", fit.ofv);
    let (_, se) = on_nonmem_scale(&fit);
    for (k, name) in ["TVCL1", "TVCL2", "TVV", "P(1)"].iter().enumerate() {
        let r = rel(se[k], nm_se[k]);
        eprintln!("SE {name}: ferx {:.6} NONMEM {:.6} rel {r:.2e}", se[k], nm_se[k]);
        assert!(r < SE_REL[k], "SE {name}: ferx {} vs NONMEM {}", se[k], nm_se[k]);
    }
}

/// From the model file's starts, ferx's Laplace mixture fit lands on NONMEM's optimum.
#[test]
fn laplace_mixture_fit_lands_on_nonmem_optimum() {
    let (theta, _, nm_ofv) = nonmem();
    let fit = run("free", |t| t);
    eprintln!("ΔOFV (ferx − NONMEM), free fit: {:+.3e}", fit.ofv - nm_ofv);
    assert!((fit.ofv - nm_ofv).abs() < OFV_AT_NONMEM, "ferx {} vs NONMEM {nm_ofv}", fit.ofv);
    let (est, _) = on_nonmem_scale(&fit);
    for (k, name) in ["TVCL1", "TVCL2", "TVV", "P(1)"].iter().enumerate() {
        let r = rel(est[k], theta[k]);
        eprintln!("{name}: ferx {:.6} NONMEM {:.6} rel {r:.2e}", est[k], theta[k]);
        assert!(r < THETA_REL, "{name}: ferx {} vs NONMEM {}", est[k], theta[k]);
    }
}
