//! A profiled θ level block read through a logit link (#2111), through the public `fit()`:
//! the NONMEM anchors A-B2 under FOCEI and LAPLACE, and the profiled covariance against the
//! unprofiled one at the same point (C1-link). Every fit is an evaluation
//! (`outer_maxiter = 0`), so the file is Tier 2.
//!
//! **A-B2.** `nonmem_anchor/profile_logit_b2_{focei,laplace}.ctl` code B2 (an observed
//! proportion, `y = inv_logit(TVL + PLACEBO[study, visit] + η + EMAX·TRT)`, additive error)
//! with every level a `THETA`, each study's last visit dependent as ferx lays it out. ferx
//! is handed NONMEM's structural optimum (TVL, EMAX, Ω, σ) and **solves the 30 free levels
//! itself**, through the link, on the adjoint's exact Newton matrix.

use ferx_core::types::{FitOptions, FitResult, OmegaMatrix};
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const B2: &str = "nonmem_anchor/profile_logit_b2.ferx";
const B2_DATA: &str = "nonmem_anchor/profile_logit_b2.csv";
const PROFILED: &str = "](0.0) profile";
const UNPROFILED: &str = "](0.0, -10.0, 10.0)";
const FOCEI: &str = "  method     = focei\n";

fn unique_temp(stem: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}.ferx", std::process::id()))
}

/// B2 under `method`, profiled or not.
fn prep(method: &str, profiled: bool) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join(B2)).unwrap();
    assert!(
        text.contains(PROFILED) && text.contains(FOCEI),
        "fixture moved"
    );
    let text = text.replace(FOCEI, &format!("  method     = {method}\n"));
    let text = if profiled {
        text
    } else {
        text.replace(PROFILED, UNPROFILED)
    };
    let path = unique_temp("ferx_profiled_logit");
    std::fs::write(&path, text).unwrap();
    ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(B2_DATA).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

/// The final estimates and standard errors of a NONMEM `.ext`, by column name.
fn ext(name: &str) -> (HashMap<String, f64>, HashMap<String, f64>) {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut names: Vec<String> = Vec::new();
    let (mut est, mut se) = (HashMap::new(), HashMap::new());
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields.first() {
            Some(&"TABLE") | None => continue,
            Some(&"ITERATION") => names = fields.iter().map(|s| s.to_string()).collect(),
            Some(it) => {
                let row: HashMap<String, f64> = names
                    .iter()
                    .cloned()
                    .zip(fields.iter().map(|v| v.parse::<f64>().unwrap()))
                    .collect();
                match *it {
                    "-1000000000" => est = row,
                    "-1000000001" => se = row,
                    _ => {}
                }
            }
        }
    }
    assert!(!est.is_empty() && !se.is_empty(), "{name}: no estimates");
    (est, se)
}

/// The evaluation options: one point, the covariance step on.
fn eval_options(p: &ferx_core::PreparedRun) -> FitOptions {
    FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    }
}

/// B2 under `method`, profiled, evaluated at the structural optimum of `ext_name` (levels
/// left at their initial 0, for the profile solve to find).
fn profiled_at(ext_name: &str, method: &str) -> (ferx_core::PreparedRun, FitResult) {
    let (est, _) = ext(ext_name);
    let p = prep(method, true);
    let mut params = p.init_params.clone();
    params.theta[0] = est["THETA1"];
    params.theta[1] = est["THETA2"];
    params.omega = OmegaMatrix::from_diagonal(&[est["OMEGA(1,1)"]], vec!["ETA_S".into()]);
    // ferx carries σ as a standard deviation, NONMEM as a variance.
    params.sigma.values[0] = est["SIGMA(1,1)"].sqrt();
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &eval_options(&p))
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// |ΔOFV| bound. Measured −7.1e-8 (FOCEI) / −2.0e-7 (LAPLACE), ferx below NONMEM both
/// times: NONMEM's levels are the end of a 4.1 / 4.5 significant-digit fit, ferx solves
/// them to `PROFILE_TOL` at that structural point. macOS arm64.
const A_B2_OFV: f64 = 1e-6;
/// Levels, in NONMEM SEs. Measured 9.5e-5 / 1.8e-4.
const A_B2_LEVEL: f64 = 1e-3;
/// Free-level and structural SEs against NONMEM's `$COV`, relative. Measured 4.1e-5 /
/// 2.4e-5 (levels / structural, the worse of the two methods); `MATRIX=R SIGL=12` is
/// precise here, so the structural SEs are asserted against it, not only printed.
const A_B2_SE: f64 = 5e-4;

/// The levels, the objective and the SEs against NONMEM with every level a `THETA`, each SE
/// against NONMEM's `$COV` (`MATRIX=R SIGL=12`). No level gradient or matrix may come from
/// finite differences: the fixture is in the adjoint's scope, so a warning naming them
/// would mean the link's route never ran.
fn assert_reproduces(ext_name: &str, method: &str) {
    let (est, se) = ext(ext_name);
    let (p, f) = profiled_at(ext_name, method);
    assert_eq!(
        p.parsed.model.n_theta, 32,
        "2 structural θ and 30 free levels"
    );
    assert_eq!(f.profiled_blocks, ["PLACEBO"]);
    for w in &f.warnings {
        assert!(
            !w.contains("finite difference") && !w.contains("finite-difference"),
            "{method}: {w}"
        );
    }
    let d_ofv = f.ofv - est["OBJ"];
    assert!(d_ofv.is_finite(), "{method}: {}", f.ofv);
    let se_theta = f
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("{method}: covariance step: {:?}", f.warnings));
    let (mut worst_level, mut worst_se) = (0.0_f64, 0.0_f64);
    for k in 2..32 {
        let name = format!("THETA{}", k + 1);
        let gap = ((f.theta[k] - est[&name]) / se[&name]).abs();
        let rel = ((se_theta[k] - se[&name]) / se[&name]).abs();
        assert!(gap.is_finite() && rel.is_finite(), "{method} {name}");
        worst_level = worst_level.max(gap);
        worst_se = worst_se.max(rel);
    }
    let struct_se: Vec<f64> = (0..2)
        .map(|k| {
            let name = format!("THETA{}", k + 1);
            let rel = ((se_theta[k] - se[&name]) / se[&name]).abs();
            assert!(rel.is_finite(), "{method} {name}");
            rel
        })
        .collect();
    eprintln!(
        "A-B2 {method}: ΔOFV {d_ofv:.3e}, worst level {worst_level:.3e} NONMEM SE, free-level SE \
         {worst_se:.3e}, structural SE {:.3e} / {:.3e}",
        struct_se[0], struct_se[1]
    );
    assert!(
        d_ofv.abs() < A_B2_OFV,
        "{method}: ferx {} vs NONMEM {}",
        f.ofv,
        est["OBJ"]
    );
    assert!(worst_level < A_B2_LEVEL, "{method}: {worst_level:.3e}");
    assert!(worst_se < A_B2_SE, "{method}: {worst_se:.3e}");
    assert!(
        struct_se.iter().all(|&r| r < A_B2_SE),
        "{method}: {struct_se:?}"
    );
}

// ── A-B2: NONMEM, every level a THETA ───────────────────────────────────────────

/// FOCEI (`METHOD=1 INTER`). NONMEM OBJ −344.413016237 (anchor image; arm64 −344.413015951).
#[test]
fn a_profiled_logit_evaluation_reproduces_nonmem_focei() {
    assert_reproduces("profile_logit_b2_focei.ext", "focei");
}

/// LAPLACE (`METHOD=1 LAPLACE INTER`), 0.028 above FOCEI's optimum, so the two objectives
/// differ on this data. NONMEM OBJ −344.385100318 (anchor image; arm64 −344.385100494).
#[test]
fn a_profiled_logit_evaluation_reproduces_nonmem_laplace() {
    assert_reproduces("profile_logit_b2_laplace.ext", "laplace");
}

// ── D1-link: fits to convergence (`slow-tests`) ─────────────────────────────────

/// The profiled fit from the model file's inits lands on NONMEM's OBJ under FOCEI and
/// LAPLACE, and the unprofiled fit started at the profiled optimum does not improve on it:
/// the profile is the minimum over the levels at every structural point, so the two
/// objectives have one optimum. The unprofiled side starts there because its own stop is
/// the looser one (it stops 1.7e-5 above NONMEM under Laplace from the inits, #2462).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_profiled_logit_fit_lands_on_nonmems_optimum() {
    for (method, ext_name) in [
        ("focei", "profile_logit_b2_focei.ext"),
        ("laplace", "profile_logit_b2_laplace.ext"),
    ] {
        let (est, _) = ext(ext_name);
        let p = prep(method, true);
        let o = FitOptions {
            verbose: false,
            ..p.parsed.fit_options.clone()
        };
        let fp = ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &o)
            .unwrap_or_else(|e| panic!("{e}"));
        let pu = prep(method, false);
        let mut start = pu.init_params.clone();
        start.theta.clone_from(&fp.theta);
        let ou = FitOptions {
            verbose: false,
            ..pu.parsed.fit_options.clone()
        };
        let fu = ferx_core::fit(&pu.parsed.model, &pu.population, &start, &ou)
            .unwrap_or_else(|e| panic!("{e}"));
        eprintln!(
            "D1-link {method}: profiled {:.9} ({}), vs NONMEM {:.3e}, unprofiled from there {:.3e}",
            fp.ofv,
            fp.converged,
            fp.ofv - est["OBJ"],
            fu.ofv - fp.ofv
        );
        assert!(fp.converged, "{method}: {:?}", fp.warnings);
        assert!((fp.ofv - est["OBJ"]).abs() < 1e-6, "{method}: {}", fp.ofv);
        assert!(fu.ofv - fp.ofv > -1e-6, "{method}: {}", fu.ofv);
    }
}

// ── C1-link: the profiled covariance is the unprofiled one at the same point ────

/// The profiled `R` (Schur complement over the structural parameters, the level blocks
/// from the adjoint) against the unprofiled step at the profiled point, with every level a
/// free θ, under FOCEI and Laplace. The unprofiled side is the analytic R-matrix, which
/// reads no step, so a gap is the profile's. Measured: worst SE 2.4e-11 / 8.2e-14 relative,
/// ΔOFV −1.5e-8 / 3.8e-12 (the solve's stop under FOCEI), macOS arm64.
#[test]
fn the_profiled_logit_covariance_is_the_unprofiled_one_at_the_same_point() {
    for (method, ext_name) in [
        ("focei", "profile_logit_b2_focei.ext"),
        ("laplace", "profile_logit_b2_laplace.ext"),
    ] {
        let (est, _) = ext(ext_name);
        let (_, fp) = profiled_at(ext_name, method);
        let pu = prep(method, false);
        let mut params = pu.init_params.clone();
        params.theta.clone_from(&fp.theta);
        params.omega = OmegaMatrix::from_diagonal(&[est["OMEGA(1,1)"]], vec!["ETA_S".into()]);
        params.sigma.values[0] = est["SIGMA(1,1)"].sqrt();
        let fu = ferx_core::fit(
            &pu.parsed.model,
            &pu.population,
            &params,
            &eval_options(&pu),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let (sp, su) = (fp.se_theta.as_ref().unwrap(), fu.se_theta.as_ref().unwrap());
        let mut worst = 0.0_f64;
        for (a, b) in sp.iter().zip(su) {
            let rel = ((a - b) / b).abs();
            assert!(rel.is_finite(), "{method}: {a} vs {b}");
            worst = worst.max(rel);
        }
        eprintln!(
            "C1-link {method}: ΔOFV {:.3e}, worst SE {worst:.3e}",
            fu.ofv - fp.ofv
        );
        assert!((fu.ofv - fp.ofv).abs() < 1e-7, "{method}");
        assert!(worst < 1e-9, "{method}: {worst:.3e}");
    }
}
