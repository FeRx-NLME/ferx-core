//! `profile`d θ level blocks (#1065), through the public `fit()`: the NONMEM anchor A2,
//! the profiled covariance against the unprofiled one at the same point (C1), and the
//! parameter count (P1). Every fit here is an evaluation (`outer_maxiter = 0`, one outer
//! step for P1), so the file is Tier 2: the convergence fits are in
//! `profiled_levels_fits.rs`, gated `slow-tests`.
//!
//! **A2.** `nonmem_anchor/profile_mbma.ctl` codes the same model with every level a
//! `THETA` (each study's last visit dependent, as ferx lays it out), the between-arm
//! kappa as an occasion `ETA` with `SAME`, and `$COV MATRIX=R`. Synthetic data
//! (`simulate_profile_mbma.py`): 12 studies × 2 arms × 4 visits, the drug effect
//! `EMAX·exp(ETA_EM)`, nonlinear in η with a sensitivity that varies by visit. NONMEM
//! estimates the levels with the structural parameters; ferx is given NONMEM's
//! structural optimum only, and **solves the levels itself**. So the comparison sees
//! the profile solve, the objective, the profiled `R` and the level SEs at once, against
//! a reference that profiles nothing.

use ferx_core::types::{FitOptions, FitResult, OmegaMatrix};
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

const F2: &str = "nonmem_anchor/profile_mbma_fit.ferx";
const F2_DATA: &str = "nonmem_anchor/profile_mbma.csv";
const F1: &str = "tests/data/profile_mbma/f1_profile.ferx";
const F1_DATA: &str = "tests/data/mbma_placebo/mbma_placebo.csv";
const PROFILED: &str = "](0.0) profile";
const UNPROFILED: &str = "](0.0, -50.0, 50.0)";

/// `model` on `data`, profiled or not (the block line rewritten).
fn prep(model: &str, data: &str, profiled: bool) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    assert!(text.contains(PROFILED), "fixture moved");
    let path = std::env::temp_dir().join(format!(
        "ferx_profiled_levels_{}_{}_{}.ferx",
        std::process::id(),
        profiled,
        model.replace('/', "_")
    ));
    let text = if profiled {
        text
    } else {
        text.replace(PROFILED, UNPROFILED)
    };
    std::fs::write(&path, text).unwrap();
    ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(data).to_str().unwrap()),
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

/// The `THETA` block of a NONMEM `.cov`, `THETAk` at index `k - 1`.
fn cov_thetas(name: &str, n: usize) -> DMatrix<f64> {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/results").join(name)).unwrap();
    let mut cols: Vec<String> = Vec::new();
    let mut m = DMatrix::from_element(n, n, f64::NAN);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first() {
            Some(&"NAME") => cols = f[1..].iter().map(|s| s.to_string()).collect(),
            Some(r) if r.starts_with("THETA") => {
                let i: usize = r[5..].parse().unwrap();
                for (c, v) in cols.iter().zip(&f[1..]) {
                    if let Some(j) = c.strip_prefix("THETA") {
                        let j: usize = j.parse().unwrap();
                        if i <= n && j <= n {
                            m[(i - 1, j - 1)] = v.parse().unwrap();
                        }
                    }
                }
            }
            _ => {}
        }
    }
    assert!(
        m.iter().all(|v| v.is_finite()),
        "{name}: THETA block incomplete"
    );
    m
}

/// F2 evaluated at NONMEM's structural optimum (levels left at their initial 0, for the
/// profile solve to find), with the covariance step.
fn at_nonmem_optimum(profiled: bool, theta: Option<&[f64]>) -> (ferx_core::PreparedRun, FitResult) {
    let (est, _) = ext("profile_mbma.ext");
    let p = prep(F2, F2_DATA, profiled);
    let mut params = p.init_params.clone();
    match theta {
        Some(t) => params.theta = t.to_vec(),
        None => {
            for k in 0..3 {
                params.theta[k] = est[&format!("THETA{}", k + 1)];
            }
        }
    }
    params.omega = OmegaMatrix::from_diagonal(&[est["OMEGA(1,1)"]], vec!["ETA_EM".into()]);
    params.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[est["OMEGA(2,2)"]],
        vec!["KAPPA_ARM".into()],
    ));
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

// ── A2: NONMEM, every level a THETA ─────────────────────────────────────────────

#[test]
fn a_profiled_evaluation_reproduces_nonmem_with_every_level_a_theta() {
    let (est, se) = ext("profile_mbma.ext");
    let cov = cov_thetas("profile_mbma.cov", 39);
    let (p, f) = at_nonmem_optimum(true, None);
    let model = &p.parsed.model;
    assert_eq!(model.n_theta, 39, "3 structural θ and 36 free levels");
    assert_eq!(f.profiled_blocks, ["PLACEBO"]);
    assert!(f.ofv.is_finite());

    // The objective, with the levels ferx solved for at NONMEM's structural point.
    let d_ofv = f.ofv - est["OBJ"];

    // The levels: ferx's profile against NONMEM's estimates, in NONMEM SEs.
    let mut worst_level = 0.0_f64;
    for k in 3..39 {
        let name = format!("THETA{}", k + 1);
        let gap = ((f.theta[k] - est[&name]) / se[&name]).abs();
        assert!(gap.is_finite(), "{name}");
        worst_level = worst_level.max(gap);
    }

    // Structural SEs, from the profiled R, against NONMEM's own profile likelihood
    // (`profile_curvature_se`): NONMEM's `$COV` R is not the reference here.
    let se_theta = f
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("covariance step: {:?}", f.warnings));
    let mut worst_struct = 0.0_f64;
    for (k, name) in ["TVE0", "EMAX", "ET50"].iter().enumerate() {
        let want = profile_curvature_se(name);
        let rel = (se_theta[k] - want).abs() / want;
        assert!(rel.is_finite(), "{name}");
        worst_struct = worst_struct.max(rel);
    }

    // Level SEs: a free level's is its THETA's; a dependent level's is the delta method
    // `sqrt(1ᵀ C 1)` over its study's three THETAs in NONMEM's own `.cov`.
    let c = f.covariance_matrix.as_ref().unwrap();
    let levels = &ferx_core::theta_level_values(model, &f.theta, Some(c), None).unwrap()["PLACEBO"];
    assert_eq!(levels.len(), 48);
    let (mut worst_free, mut worst_dep, mut n_dep) = (0.0_f64, 0.0_f64, 0);
    for (i, l) in levels.iter().enumerate() {
        let got = l.se.unwrap_or_else(|| panic!("{}: no SE", l.label));
        let want = match l.theta_index {
            Some(k) => se[&format!("THETA{}", k + 1)],
            None => {
                n_dep += 1;
                let s = i / 4;
                let r = 3 + 3 * s..6 + 3 * s;
                r.clone()
                    .flat_map(|a| r.clone().map(move |b| (a, b)))
                    .map(|(a, b)| cov[(a, b)])
                    .sum::<f64>()
                    .sqrt()
            }
        };
        let rel = (got - want).abs() / want;
        assert!(rel.is_finite(), "{}: {got} vs {want}", l.label);
        if l.theta_index.is_some() {
            worst_free = worst_free.max(rel);
        } else {
            worst_dep = worst_dep.max(rel);
        }
    }
    assert_eq!(n_dep, 12, "each study's last visit is dependent");
    eprintln!(
        "A2: ΔOFV {d_ofv:.3e}, worst level {worst_level:.3e} SE, structural SE {worst_struct:.3e}, \
         free-level SE {worst_free:.3e}, dependent-level SE {worst_dep:.3e}"
    );
    assert!(
        d_ofv.abs() < A2_OFV,
        "OFV ferx {} vs NONMEM {}",
        f.ofv,
        est["OBJ"]
    );
    assert!(worst_level < A2_LEVEL, "{worst_level}");
    assert!(worst_struct < A2_STRUCT_SE, "{worst_struct}");
    assert!(worst_free < A2_LEVEL_SE, "{worst_free}");
    assert!(worst_dep < A2_LEVEL_SE, "{worst_dep}");
}

/// The standard error NONMEM's own profile likelihood gives `name`: the curvature of the
/// objective with that THETA FIXed at its estimate ±δ and ±2δ and everything else
/// re-estimated (`results/profile_mbma_profiles.txt`), Richardson-extrapolated to δ → 0,
/// as `sqrt(2 / c)` since the OFV is −2 log L.
///
/// Why not NONMEM's `$COV` SE for these three: it is a finite-difference `R` whose
/// accuracy follows the estimation's precision, and ET50 is weakly identified (34% RSE).
/// Measured on this fixture (#2130 review #4): at `SIGDIGITS=4` `$COV` gave ET50 0.234,
/// 59% below the profile's 0.576; at `SIGL=12 NSIG=4` (the committed run) it gives 0.523,
/// still 9% below, and EMAX moves from 8% to 2% below. The profile is the stable
/// reference; ferx's `R` (profiled and unprofiled, which agree to 4.7e-3, C1) matches it.
fn profile_curvature_se(name: &str) -> f64 {
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/results/profile_mbma_profiles.txt"))
            .unwrap();
    let rows: Vec<(f64, f64)> = text
        .lines()
        .filter(|l| l.starts_with(name))
        .map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .skip(1)
                .map(|v| v.parse().unwrap())
                .collect();
            (f[0], f[1])
        })
        .collect();
    assert_eq!(rows.len(), 5, "{name}: ±2δ, ±δ and the centre");
    let d = rows[3].0 - rows[2].0;
    let o = |i: usize| rows[i].1;
    let c1 = (o(3) + o(1) - 2.0 * o(2)) / (d * d);
    let c2 = (o(4) + o(0) - 2.0 * o(2)) / (4.0 * d * d);
    let c = (4.0 * c1 - c2) / 3.0;
    assert!(c.is_finite() && c > 0.0, "{name}: {c}");
    (2.0 / c).sqrt()
}

/// ΔOFV realised 4.9e-8 (macOS arm64) against the `SIGL=12` run's OBJ: two engines'
/// objectives at one point, the levels solved by ferx and estimated by NONMEM. NONMEM's
/// own two images differ by 1.5e-8 here. 4× headroom.
const A2_OFV: f64 = 2e-7;
/// Worst level gap realised 8.2e-6 NONMEM SEs; 12× headroom.
const A2_LEVEL: f64 = 1e-4;
/// Structural SE against NONMEM's profile likelihood: worst realised 7.5e-3 (EMAX); the
/// reference's own finite-δ extrapolation moves it by ~5e-3. 2.7× headroom.
const A2_STRUCT_SE: f64 = 2e-2;
/// Level SEs against NONMEM's `$COV` at `SIGL=12`: worst realised 4.0e-3 (free) and
/// 6.4e-3 (dependent), macOS arm64. (Against the earlier `SIGDIGITS=4` run they were
/// 2.9e-2 / 3.5e-2: that `R` was the imprecise side.) 3.1× headroom on the worst.
const A2_LEVEL_SE: f64 = 2e-2;

// ── C1: the profiled covariance is the unprofiled one ────────────────────────────

/// At one joint point — NONMEM's structural optimum and the levels ferx profiled there —
/// the profiled covariance step and the unprofiled one (the FD `R` over all 41 packed
/// coordinates) give the same standard errors for every parameter, structural and
/// level. Dies when the stencil holds the levels instead of re-profiling them (`A⁻¹`,
/// too small), when the `D⁻¹BᵀCov(θ)BD⁻¹` term is dropped, and when `D` is the Newton
/// matrix rather than the exact Hessian.
#[test]
fn the_profiled_covariance_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = at_nonmem_optimum(true, None);
    let (_, plain) = at_nonmem_optimum(false, Some(&prof.theta));
    assert!(
        (prof.ofv - plain.ofv).abs() < 1e-6,
        "the same point: {} vs {}",
        prof.ofv,
        plain.ofv
    );
    let (a, b) = (
        prof.se_theta.as_ref().expect("profiled SE"),
        plain.se_theta.as_ref().expect("unprofiled SE"),
    );
    let (mut worst_struct, mut worst_level) = (0.0_f64, 0.0_f64);
    for k in 0..a.len() {
        let rel = (a[k] - b[k]).abs() / b[k];
        assert!(rel.is_finite(), "θ{k}: {} vs {}", a[k], b[k]);
        if k < 3 {
            worst_struct = worst_struct.max(rel);
        } else {
            worst_level = worst_level.max(rel);
        }
    }
    let om = |f: &FitResult| f.se_omega.as_ref().unwrap()[0];
    let ka = |f: &FitResult| f.se_kappa.as_ref().unwrap()[0];
    let worst_var = ((om(&prof) - om(&plain)).abs() / om(&plain))
        .max((ka(&prof) - ka(&plain)).abs() / ka(&plain));
    assert!(worst_var.is_finite());

    // The off-diagonal blocks the SEs cannot see (#2130 review #3): `Cov(θ, β)` and the
    // level × level covariances, in the published packed matrix, as a correlation gap
    // `|C_p − C_u| / sqrt(C_u,ii · C_u,jj)` against the unprofiled one. Over every θ
    // pair — structural × level and level × level — each off its own diagonal.
    let (cp, cu) = (
        prof.covariance_matrix.as_ref().unwrap(),
        plain.covariance_matrix.as_ref().unwrap(),
    );
    assert_eq!(cp.shape(), cu.shape());
    let (mut worst_cross, mut worst_bb) = (0.0_f64, 0.0_f64);
    for i in 0..a.len() {
        for j in (0..a.len()).filter(|&j| j != i && j >= 3) {
            let gap = (cp[(i, j)] - cu[(i, j)]).abs() / (cu[(i, i)] * cu[(j, j)]).sqrt();
            assert!(gap.is_finite(), "({i}, {j})");
            if i < 3 {
                worst_cross = worst_cross.max(gap);
            } else {
                worst_bb = worst_bb.max(gap);
            }
        }
    }
    eprintln!(
        "C1: structural SE {worst_struct:.3e}, level SE {worst_level:.3e}, variance SE \
         {worst_var:.3e}; correlation gap Cov(θ,β) {worst_cross:.3e}, Cov(β,β') {worst_bb:.3e}"
    );
    assert!(worst_struct < C1_BOUND, "{worst_struct}");
    assert!(worst_level < C1_BOUND, "{worst_level}");
    assert!(worst_var < C1_BOUND, "{worst_var}");
    assert!(worst_cross < C1_CORR_BOUND, "{worst_cross}");
    assert!(worst_bb < C1_CORR_BOUND, "{worst_bb}");
}

/// Worst realised 4.7e-3 (structural), 2.3e-4 (levels), 3.0e-3 (Ω, Ω_IOV), Linux aarch64
/// and macOS arm64 alike:
/// two FD Hessians of one objective, at the same step rule but over 5 coordinates
/// re-profiled against 41 held; 2.1× headroom on the worst.
const C1_BOUND: f64 = 1e-2;
/// Correlation-scale gap of the off-diagonal blocks: worst realised 9.0e-3 (`Cov(θ, β)`)
/// and 5.7e-4 (`Cov(β, β')`), macOS arm64. The `Cov(θ, β)` gap is the structural SEs'
/// own 4.7e-3 FD difference carried through `−Cov(θ)·B·D⁻¹`; 2.8× headroom on it.
const C1_CORR_BOUND: f64 = 2.5e-2;

/// `run_covariance` on a profiled fit replays the profiled step the fit ran: the same
/// standard errors, levels included. The unprofiled step it would otherwise take is the
/// 41-coordinate stencil the profile avoids (at 800 levels, routed to `s`).
#[test]
fn run_covariance_replays_the_profiled_step() {
    let (p, f) = at_nonmem_optimum(true, None);
    let opts = FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    };
    let r = ferx_core::run_covariance(&f, Some(&p.parsed.model), Some(&p.population), &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    let (a, b) = (
        f.se_theta.as_ref().unwrap(),
        r.se_theta.as_ref().expect("replayed SE"),
    );
    let worst = (0..a.len())
        .map(|k| ((a[k] - b[k]) / a[k]).abs())
        .fold(0.0_f64, |m, v| {
            assert!(v.is_finite());
            m.max(v)
        });
    eprintln!("replay: worst SE gap {worst:.3e}");
    assert!(worst < REPLAY_BOUND, "{worst}");
}

/// Worst realised 5.5e-6: the replay re-profiles from the fit's levels, already at the
/// optimum, so only the re-solved EBEs differ; 18× headroom.
const REPLAY_BOUND: f64 = 1e-4;

// ── A1: the existing level-SE anchor, profiled (#2130 review #8) ───────────────────

/// The plan's A1: `nonmem_anchor/level_se_mbma_fit.ferx` (#1890's anchor, linear in its
/// one η, no κ) with its block declared `profile`, evaluated at NONMEM (a)'s structural
/// optimum only, the 18 free levels solved by ferx. Checked against the same NONMEM runs
/// and bounds as the unprofiled `tests/level_se_nonmem_anchor.rs`: the OFV, the free-level
/// SEs against (a)'s `THETA`s, and the 6 dependent levels against (b)'s, where each is a
/// `THETA` of its own.
#[test]
fn a_profiled_evaluation_reproduces_the_level_se_anchor() {
    let (est_a, se_a) = ext("level_se_a.ext");
    let (_, se_b) = ext("level_se_b.ext");
    let text =
        std::fs::read_to_string(root().join("nonmem_anchor/level_se_mbma_fit.ferx")).unwrap();
    assert!(text.contains(UNPROFILED), "fixture moved");
    let path = std::env::temp_dir().join(format!(
        "ferx_profiled_levels_{}_a1.ferx",
        std::process::id()
    ));
    std::fs::write(&path, text.replace(UNPROFILED, PROFILED)).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(
            root()
                .join("tests/data/mbma_placebo/mbma_placebo.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let model = &p.parsed.model;
    assert_eq!(model.n_theta, 22, "4 structural θ and 18 free levels");
    let mut params = p.init_params.clone();
    for k in 0..4 {
        params.theta[k] = est_a[&format!("THETA{}", k + 1)];
    }
    params.omega = OmegaMatrix::from_diagonal(&[est_a["OMEGA(1,1)"]], vec!["ETA_E0".into()]);
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(model, &p.population, &params, &opts).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(f.profiled_blocks, ["PLACEBO"]);
    let d_ofv = f.ofv - est_a["OBJ"];
    let cov = f
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("covariance step: {:?}", f.warnings));
    let levels =
        &ferx_core::theta_level_values(model, &f.theta, Some(cov), None).unwrap()["PLACEBO"];
    assert_eq!(levels.len(), 24);
    let (mut worst_free, mut worst_dep, mut n_dep) = (0.0_f64, 0.0_f64, 0);
    for (i, l) in levels.iter().enumerate() {
        let got = l.se.unwrap_or_else(|| panic!("{}: no SE", l.label));
        let want = match l.theta_index {
            Some(k) => se_a[&format!("THETA{}", k + 1)],
            None => {
                n_dep += 1;
                se_b[&format!("THETA{}", 7 + 3 * (i / 4))]
            }
        };
        let rel = (got - want).abs() / want;
        assert!(rel.is_finite(), "{}: {got} vs {want}", l.label);
        if l.theta_index.is_some() {
            worst_free = worst_free.max(rel);
        } else {
            worst_dep = worst_dep.max(rel);
        }
    }
    assert_eq!(n_dep, 6);
    eprintln!(
        "A1: ΔOFV {d_ofv:.3e}, free-level SE {worst_free:.3e}, dependent-level SE {worst_dep:.3e}"
    );
    assert!(
        d_ofv.abs() < A1_OFV,
        "ferx {} vs NONMEM {}",
        f.ofv,
        est_a["OBJ"]
    );
    assert!(worst_free < A1_SE, "{worst_free}");
    assert!(worst_dep < A1_SE, "{worst_dep}");
}

/// Realised (macOS arm64): ΔOFV −1.1e-6 — ferx below NONMEM, its levels solved exactly
/// where NONMEM's are the end of a three-significant-digit fit — and level SEs within
/// 2.2e-3 (free) / 2.8e-3 (dependent), against 3.4e-3 / 4.2e-3 for the unprofiled anchor
/// at the same point. 9× and 3.6× headroom.
const A1_OFV: f64 = 1e-5;
const A1_SE: f64 = 1e-2;

// ── EBE convergence: counted and guarded, as unprofiled (#2130 review #2) ──────────

/// A starved inner solve (`inner_maxiter = 12` on F1: 4 subjects unconverged, 11 Nelder–Mead
/// fallbacks, measured; fewer iterations fail the profile solve outright) leaves EBEs
/// unconverged. The profiled fit
/// publishes that count, and the unprofiled loop's EBE guard applies to it: under
/// `max_unconverged_frac = 0` the evaluation is rejected and no objective is reported,
/// while under `1.0` (the guard off) the same evaluation is accepted. The pair straddles
/// the gate.
#[test]
fn a_profiled_fit_counts_and_guards_unconverged_ebes() {
    let run = |frac: f64| {
        let p = prep(F1, F1_DATA, true);
        let opts = FitOptions {
            verbose: false,
            outer_maxiter: 0,
            run_covariance_step: false,
            inner_maxiter: 12,
            max_unconverged_frac: frac,
            ..p.parsed.fit_options.clone()
        };
        ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &opts)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let open = run(1.0);
    assert!(open.ofv.is_finite(), "the guard is off: {}", open.ofv);
    assert!(
        open.max_unconverged_subjects > 0,
        "a starved inner solve leaves EBEs unconverged, and the fit says so"
    );
    assert_eq!(open.ebe_convergence_warnings, 0);
    let shut = run(0.0);
    assert!(
        !shut.ofv.is_finite(),
        "the guard rejects the evaluation: {}",
        shut.ofv
    );
    assert!(shut.ebe_convergence_warnings >= 1);
    assert!(!shut.converged);
}

// ── P1: the levels count, through `fit()` ────────────────────────────────────────

/// A profiled fit reports the unprofiled fit's `n_parameters` and BIC tally and names
/// its block, through `fit()` itself: the optimizer holds the levels, the count does
/// not. One outer step, so it returns at once.
#[test]
fn a_profiled_fit_reports_the_unprofiled_parameter_count() {
    let run = |profiled: bool| {
        let p = prep(F1, F1_DATA, profiled);
        let opts = FitOptions {
            verbose: false,
            outer_maxiter: if profiled { 1 } else { 0 },
            run_covariance_step: false,
            ..p.parsed.fit_options.clone()
        };
        ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &opts)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let (prof, plain) = (run(true), run(false));
    assert_eq!(prof.n_parameters, plain.n_parameters);
    assert_eq!(prof.n_parameters, 24, "4 θ, 18 free levels, ω, κ");
    assert_eq!(prof.bic_inputs, plain.bic_inputs);
    assert_eq!(prof.profiled_blocks, ["PLACEBO"]);
    assert!(plain.profiled_blocks.is_empty());
    assert!(
        prof.theta_fixed.iter().all(|f| !f),
        "the levels are estimated, not FIX: {:?}",
        prof.theta_fixed
    );
}
