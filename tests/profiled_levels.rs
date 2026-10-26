//! `profile`d θ level blocks (#1065), through the public `fit()`: the NONMEM anchor A2,
//! the profiled covariance against the unprofiled one at the same point (C1), and the
//! parameter count (P1). Every fit here is an evaluation (`outer_maxiter = 0`, one outer
//! step for P1), so the file is Tier 2: the convergence fits are in
//! `profiled_levels_fits.rs`, gated `slow-tests`. One evaluation here is gated for its
//! cost alone, the nested C4 (#2108).
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

use ferx_core::io::fitrx::{load_fit, save_fit, SaveFitOptions};
use ferx_core::types::{CovarianceMethod, FitOptions, FitResult, OmegaMatrix};
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

/// A temp path no other call in this process gets: tests run on parallel threads, and two
/// that prep one fixture under one name raced — one read the other's half-written file
/// ("Missing [parameters] block", slow-tests run 39170264924; #2090 is the same bug).
fn unique_temp(stem: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!("{stem}_{}_{n}.ferx", std::process::id()))
}

/// `model` on `data`, profiled or not (the block line rewritten).
fn prep(model: &str, data: &str, profiled: bool) -> ferx_core::PreparedRun {
    let text = std::fs::read_to_string(root().join(model)).unwrap();
    assert!(text.contains(PROFILED), "fixture moved");
    let path = unique_temp("ferx_profiled_levels");
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

/// A covariance step, as an edit of a fixture's options.
type Step = fn(FitOptions) -> FitOptions;

/// `covariance_method = r`, the fixture's own settings.
fn r(o: FitOptions) -> FitOptions {
    FitOptions {
        covariance_method: CovarianceMethod::Hessian,
        covariance_method_set: true,
        ..o
    }
}

/// `covariance_method = s`, EBEs to [`EVAL_INNER_TOL`]: `S` is scored at the modes each side
/// holds, so the two sides' modes must agree past the comparison.
fn s(o: FitOptions) -> FitOptions {
    FitOptions {
        covariance_method: CovarianceMethod::CrossProduct,
        covariance_method_set: true,
        inner_tol: EVAL_INNER_TOL,
        ..o
    }
}

/// `covariance_method = rsr` at [`RSR_FD_STEP`], every inner solve to [`EVAL_INNER_TOL`].
/// The unprofiled reference is the **analytic** R-matrix on these fixtures, exact and blind to
/// `fd_hessian_step`: on F2 its SE(TVE0) is 3.3115770e-1 at both 1e-2 and 1e-3. The profiled
/// side's `A_p` is an FD stencil, and its truncation error is the whole gap: SE(TVE0)
/// 3.1760e-1 (−4.1%) at the default 1e-2, 3.3098e-1 (−5.3e-4) at 1e-3, `O(h²)` in between
/// (macOS arm64; review row 1 on #2403). At the default the profiled stencil's own error
/// would sit above the `sᵢθ` mutant's gap, so C4 runs where it does not.
fn rsr(o: FitOptions) -> FitOptions {
    FitOptions {
        covariance_method: CovarianceMethod::Sandwich,
        covariance_method_set: true,
        fd_hessian_step: RSR_FD_STEP,
        fd_hessian_step_set: true,
        inner_tol: EVAL_INNER_TOL,
        cov_inner_tol: Some(EVAL_INNER_TOL),
        ..o
    }
}

const RSR_FD_STEP: f64 = 1e-3;
/// The inner tolerance of an evaluation compared across the two sides.
const EVAL_INNER_TOL: f64 = 1e-8;

/// F2 evaluated at NONMEM's structural optimum (levels left at their initial 0, for the
/// profile solve to find), with the covariance step `step`.
fn at_nonmem_optimum(
    profiled: bool,
    theta: Option<&[f64]>,
    step: Step,
) -> (ferx_core::PreparedRun, FitResult) {
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
    let opts = step(opts);
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// The profiled `at_nonmem_optimum(true, None)`, evaluated once for the three tests that
/// read it. Each evaluation carries the covariance step, and three identical copies running
/// on parallel test threads made this the slowest binary in `Tests + coverage (core)`
/// (661 s, #2167).
fn profiled_at_nonmem_optimum() -> &'static (ferx_core::PreparedRun, FitResult) {
    static F: std::sync::OnceLock<(ferx_core::PreparedRun, FitResult)> = std::sync::OnceLock::new();
    F.get_or_init(|| at_nonmem_optimum(true, None, r))
}

// ── A2: NONMEM, every level a THETA ─────────────────────────────────────────────

#[test]
fn a_profiled_evaluation_reproduces_nonmem_with_every_level_a_theta() {
    let (est, se) = ext("profile_mbma.ext");
    let cov = cov_thetas("profile_mbma.cov", 39);
    let (p, f) = profiled_at_nonmem_optimum();
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
/// Structural SE against NONMEM's profile likelihood: worst realised 7.5e-4 at the profiled
/// default step, 7.5e-3 (EMAX) at the global `1e-2` it used before. The bound stays at
/// 2e-2 because the reference's own finite-δ extrapolation moves it by ~5e-3, so the
/// smaller gap is not evidence of more than that.
const A2_STRUCT_SE: f64 = 2e-2;
/// Level SEs against NONMEM's `$COV` at `SIGL=12`: worst realised 4.0e-3 (free) and
/// 6.4e-3 (dependent), macOS arm64. (Against the earlier `SIGDIGITS=4` run they were
/// 2.9e-2 / 3.5e-2: that `R` was the imprecise side.) 3.1× headroom on the worst.
const A2_LEVEL_SE: f64 = 2e-2;

// ── A3: NONMEM, one level FIXed at the bound a profiled box holds it on (#2112) ──

/// The F2 block declared with the box `(0.0, -10.0, 2.5)`, which holds
/// `PLACEBO[STUDY=7,TIME=4]` (THETA24) on its upper bound at the optimum.
const A3_BOXED: &str = "](0.0, -10.0, 2.5) profile";

/// A3: `nonmem_anchor/profile_mbma_bound.ctl` is A2's stream with THETA24 FIXed at 2.5,
/// the exact object a bounded profile solve holds there. ferx is handed NONMEM's structural
/// optimum only and solves the 35 other free levels **under the box**; the bounded solve
/// must pin THETA24 at exactly 2.5 itself and land on NONMEM's objective and levels. Dies
/// when the box is not applied (the level goes to its unbounded 2.83 and the OFV drops by
/// ~0.57) or is applied by clamping the Newton step (off the constrained minimum).
#[test]
fn a_bounded_profiled_evaluation_reproduces_nonmem_with_the_level_fixed() {
    let (est, se) = ext("profile_mbma_bound.ext");
    let text = std::fs::read_to_string(root().join(F2)).unwrap();
    assert!(text.contains(PROFILED), "fixture moved");
    let path = unique_temp("ferx_profiled_levels_a3");
    std::fs::write(&path, text.replace(PROFILED, A3_BOXED)).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(F2_DATA).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut params = p.init_params.clone();
    for k in 0..3 {
        params.theta[k] = est[&format!("THETA{}", k + 1)];
    }
    params.omega = OmegaMatrix::from_diagonal(&[est["OMEGA(1,1)"]], vec!["ETA_EM".into()]);
    params.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[est["OMEGA(2,2)"]],
        vec!["KAPPA_ARM".into()],
    ));
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..p.parsed.fit_options.clone()
    };
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    let pinned = 23;
    assert_eq!(p.init_params.theta_names[pinned], "PLACEBO[STUDY=7,TIME=4]");
    assert_eq!(est["THETA24"], 2.5);
    assert_eq!(
        f.theta[pinned].to_bits(),
        2.5_f64.to_bits(),
        "held on its bound"
    );
    assert!(
        f.warnings.iter().any(|w| w.starts_with(
            "Profiled level(s) on a declared bound: PLACEBO[STUDY=7,TIME=4] (2.5000 at upper bound)."
        )),
        "{:?}",
        f.warnings
    );
    let d_ofv = f.ofv - est["OBJ"];
    let mut worst_level = 0.0_f64;
    for k in (3..39).filter(|&k| k != pinned) {
        let name = format!("THETA{}", k + 1);
        let gap = ((f.theta[k] - est[&name]) / se[&name]).abs();
        assert!(gap.is_finite(), "{name}");
        worst_level = worst_level.max(gap);
    }
    eprintln!("A3: ΔOFV {d_ofv:.3e}, worst free level {worst_level:.3e} NONMEM SE");
    assert!(
        d_ofv.is_finite() && d_ofv.abs() < A3_OFV,
        "OFV ferx {} vs NONMEM {}",
        f.ofv,
        est["OBJ"]
    );
    assert!(worst_level < A3_LEVEL, "{worst_level}");
}

/// ΔOFV against `profile_mbma_bound.ext`'s OBJ 102.484421356 (`anchor`; arm64
/// 102.484421357, 1.2e-9 apart). Realised 3.36e-8 (Linux aarch64; 3.34e-8 macOS arm64);
/// A2's band, 6× headroom.
const A3_OFV: f64 = A2_OFV;
/// Worst free-level gap in NONMEM SEs: realised 2.61e-5 (Linux aarch64 and macOS arm64);
/// A2's band, 3.8× headroom.
const A3_LEVEL: f64 = A2_LEVEL;

// ── C1: the profiled covariance is the unprofiled one ────────────────────────────

/// At one joint point — NONMEM's structural optimum and the levels ferx profiled there —
/// the profiled covariance step and the unprofiled one (the FD `R` over all 41 packed
/// coordinates) give the same standard errors for every parameter, structural and
/// level. Dies when the stencil holds the levels instead of re-profiling them (`A⁻¹`,
/// too small), when the `D⁻¹BᵀCov(θ)BD⁻¹` term is dropped, and when `D` is the Newton
/// matrix rather than the exact Hessian.
#[test]
fn the_profiled_covariance_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = profiled_at_nonmem_optimum();
    let (_, plain) = at_nonmem_optimum(false, Some(&prof.theta), r);
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

/// The unprofiled side is the **analytic** R-matrix, exact and blind to `fd_hessian_step`.
/// The profiled side is the FD stencil of the profile objective, at its own default step
/// (`PROFILE_FD_HESSIAN_STEP = 3e-3`), so the whole gap is that stencil's truncation error.
/// Worst realised 7.4e-4 (structural), 3.6e-5 (levels), 9.6e-4 (Ω, Ω_IOV), macOS arm64.
/// 3.1× headroom on the worst. At the global default `1e-2`, which the profiled stencil
/// used before it had its own, the structural gap is 7.5e-3, past the bound. So a routing
/// that ignores the profile default fails here.
const C1_BOUND: f64 = 3e-3;
/// Correlation-scale gap of the off-diagonal blocks: worst realised 9.3e-4 (`Cov(θ, β)`)
/// and 8.1e-5 (`Cov(β, β')`), macOS arm64. The `Cov(θ, β)` gap is the structural SEs' FD
/// error carried through `−Cov(θ)·B·D⁻¹`; 5.4× headroom on it. With `A_p` at `1e-2` it is
/// 1.8e-2.
const C1_CORR_BOUND: f64 = 5e-3;

/// `run_covariance` on a profiled fit replays the profiled step the fit ran: the same
/// covariance matrix, levels included, **bit for bit** (#2173) — in memory and after a
/// `.fitrx` round trip. The unprofiled step it would otherwise take is the 41-coordinate
/// stencil the profile avoids (at 800 levels, routed to `s`).
///
/// Bit-equality holds because the inline step and the replay share one implementation
/// (`ProfileCtx::covariance_at_estimate`): each re-profiles from the reported levels and
/// EBEs before the stencil. Before #2173 the inline step used the optimizer's own solve and
/// the replay a fresh one, and this fixture measured a worst SE gap of 7.6e-7 (3.9e-3 on
/// the worst covariance entry), macOS arm64.
#[test]
fn run_covariance_replays_the_profiled_step() {
    let (p, f) = profiled_at_nonmem_optimum();
    let model_text = std::fs::read_to_string(root().join(F2)).unwrap();
    assert_replays_bit_for_bit(p, f, &model_text, r);
}

/// `run_covariance` under `step` on the profiled fit `f` (whose inline step was `step`)
/// repeats that step bit for bit, in memory and after a `.fitrx` round trip of `f` saved with
/// `model_text`: the same covariance matrix, levels included, the same SEs and label.
fn assert_replays_bit_for_bit(
    p: &ferx_core::PreparedRun,
    f: &FitResult,
    model_text: &str,
    step: Step,
) {
    let opts = step(FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    });
    let method = opts.covariance_method;
    assert_eq!(f.covariance_method, Some(method), "{:?}", f.warnings);
    let want = f.covariance_matrix.as_ref().expect("inline covariance");
    assert!(want.iter().all(|v| v.is_finite()));
    let replay = |fit: &FitResult, how: &str| {
        let r = ferx_core::run_covariance(fit, Some(&p.parsed.model), Some(&p.population), &opts)
            .unwrap_or_else(|e| panic!("{how}: {e}"));
        let got = r
            .covariance_matrix
            .as_ref()
            .unwrap_or_else(|| panic!("{how}: no replayed covariance: {:?}", r.warnings));
        assert_eq!(got.shape(), want.shape(), "{how}");
        let differing = got
            .iter()
            .zip(want.iter())
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(differing, 0, "{how}: {differing} covariance entries differ");
        assert_eq!(r.se_theta, f.se_theta, "{how}");
        assert_eq!(r.covariance_method, Some(method), "{how}");
    };
    replay(f, "in memory");

    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ferx_profiled_replay_{}_{n}.fitrx",
        std::process::id()
    ));
    save_fit(
        f,
        &p.population,
        model_text,
        &path,
        SaveFitOptions::default(),
    )
    .expect("saves");
    let loaded = load_fit(&path).expect("loads");
    let _ = std::fs::remove_file(&path);
    replay(&loaded.fit, "after .fitrx reload");
}

/// `run_covariance` handed levels other than the fit's β̂ re-profiles them before the
/// stencil and runs it at the levels it solved, not the ones it was handed (#2213). One
/// level of the shared F2 evaluation moved by 1.0 (its NONMEM SE is 0.71) is a start where
/// the re-solve moves the levels, which a replay of the fit's own estimates never is:
/// `solve_group` takes no Newton step from a converged start. The replayed covariance is the
/// inline one to the re-solve's tolerance, worst correlation-scale gap 1.55e-7 (Linux
/// aarch64 and macOS arm64 alike), 6.5× under the bound. Run at the handed levels instead
/// (`covariance_at_estimate` without its coordinate overwrite) the gap is 4.5e-2; moved by
/// 0.1, 4.4e-3 against 4.0e-8 (macOS arm64).
#[test]
fn run_covariance_on_moved_levels_re_profiles_them() {
    let (p, f) = profiled_at_nonmem_optimum();
    let opts = FitOptions {
        verbose: false,
        ..p.parsed.fit_options.clone()
    };
    let want = f.covariance_matrix.as_ref().expect("inline covariance");
    let mut moved = f.clone();
    moved.theta[3] += 1.0;
    let r = ferx_core::run_covariance(&moved, Some(&p.parsed.model), Some(&p.population), &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    let got = r
        .covariance_matrix
        .as_ref()
        .unwrap_or_else(|| panic!("{:?}", r.warnings));
    assert_eq!(got.shape(), want.shape());
    // The straddle (#2388 review row 4): the move reached the replay. Were the edited fit
    // taken as carrying its own packed point, the replay would unpack the unmoved `x̂` and
    // repeat the inline step bit for bit, with or without the overwrite. Measured: 1681
    // entries differ, every one outside σ's zero row and column (macOS arm64).
    let differing = got
        .iter()
        .zip(want.iter())
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert!(differing > 0, "the replay ran at the unmoved levels");
    let mut worst = 0.0_f64;
    for i in 0..want.nrows() {
        for j in 0..want.ncols() {
            // σ is `FIX`: its row and column are zero on both sides.
            let s = (want[(i, i)] * want[(j, j)]).sqrt();
            if s == 0.0 {
                assert_eq!(got[(i, j)], 0.0, "({i}, {j})");
                continue;
            }
            let g = ((got[(i, j)] - want[(i, j)]) / s).abs();
            assert!(
                g.is_finite(),
                "({i}, {j}): {} vs {}",
                got[(i, j)],
                want[(i, j)]
            );
            worst = worst.max(g);
        }
    }
    eprintln!("moved levels: worst correlation-scale covariance gap {worst:.3e}");
    assert!(worst < MOVED_LEVELS_BOUND, "{worst}");
}

const MOVED_LEVELS_BOUND: f64 = 1e-6;

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
    let path = unique_temp("ferx_profiled_levels_a1");
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
/// 1.7e-4 (free) / 3.3e-4 (dependent) at the profiled default step (2.2e-3 / 2.8e-3 at the
/// global `1e-2`), against 3.4e-3 / 4.2e-3 for the unprofiled anchor at the same point.
/// The bound is the unprofiled anchor's own.
const A1_OFV: f64 = 1e-5;
const A1_SE: f64 = 1e-2;

// ── EBE convergence: counted and guarded, as unprofiled (#2130 review #2) ──────────

/// A starved inner solve (`inner_maxiter = 0` on F1: no step from the start, 6 subjects
/// unconverged, measured) leaves EBEs unconverged. Since #2093's step-scaled first iterate a
/// single inner iteration already converges every F1 subject at this point (measured over
/// budgets 1..=12), so only the zero budget still starves it. The profiled fit
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
            inner_maxiter: 0,
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

// ── P2: which contrast `auto` takes on S1's shape, by the count (#2326) ─────────

const S1: &str = "tests/data/profile_mbma/s1_profile.ferx";
/// The first 10 studies of S1's `s100t8.csv` (same simulator, same seed).
const S1_DATA_10: &str = "tests/data/profile_mbma/s10t8.csv";

/// S1's `[STUDY, TIME]` block over two arms at each visit, with and without the study
/// baseline `ETA_E0`. Both arms read `ETA_E0`, so the block absorbs it and `auto` takes
/// `sum_to_zero_within`: 80 levels less one per study, **70**. `ETA_EMAX` is read by the
/// drug arm only, so the block cannot reproduce it (#2131): without `ETA_E0` nothing is
/// absorbed and `auto` takes global `sum_to_zero`, **79**. The other contrast would count
/// 85 with `ETA_E0` and 75 without. Mutations (measured): calling co-timed arms with a
/// different `TRT` replicates again (the pre-#2305 rule) refuses the first model ("absorb
/// `ETA_E0` and `ETA_EMAX` together"); resolving the absorbed arm of `auto` to
/// `sum_to_zero` refuses it as not identified. Disabling that arm alone does not reach
/// this test: `admissible` then sends `auto` to `sum_to_zero_within` anyway. S1 itself
/// (`profiled_levels_fits.rs`) leans on the first count: under global `sum_to_zero` its
/// 800 levels are one coupled group and the fit no longer fits the slow tier (#2326).
#[test]
fn s1_auto_contrast_splits_per_study_only_when_a_both_arm_eta_is_absorbed() {
    let count = |with_e0: bool| {
        let text = std::fs::read_to_string(root().join(S1)).unwrap();
        let text = if with_e0 {
            text
        } else {
            let edited = text
                .replace("  omega ETA_E0 ~ 0.1\n", "")
                .replace("E0 = TVE0 + ETA_E0", "E0 = TVE0");
            assert_ne!(edited, text, "fixture moved");
            edited
        };
        let path = unique_temp("ferx_profiled_levels_s1");
        std::fs::write(&path, text).unwrap();
        let p = ferx_core::prepare_run(
            path.to_str().unwrap(),
            Some(root().join(S1_DATA_10).to_str().unwrap()),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p.init_params.omega.eta_names.len(), 1 + with_e0 as usize);
        let opts = FitOptions {
            verbose: false,
            outer_maxiter: 1,
            run_covariance_step: false,
            ..p.parsed.fit_options.clone()
        };
        let f = ferx_core::fit(&p.parsed.model, &p.population, &p.init_params, &opts)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(f.profiled_blocks, ["PLACEBO"]);
        f.n_parameters
    };
    assert_eq!(
        count(true),
        76,
        "3 θ, 70 levels (sum_to_zero_within), 2 ω, σ"
    );
    assert_eq!(count(false), 84, "3 θ, 79 levels (sum_to_zero), ω, σ");
}

// ── #2110: a residual `correlation` line with the profiled block ───────────────────

/// F1 with `theta RHO(0.6, 0.0, 0.95)` and `correlation car1(RHO) by OCC` (ferx-r's MBMA
/// smoke step 13 shape, profiled or not), as text.
fn f1_car1_text(profiled: bool) -> String {
    let text = std::fs::read_to_string(root().join(F1)).unwrap();
    assert!(
        text.contains(PROFILED) && text.contains("weight = SE\n"),
        "fixture moved"
    );
    let text = text
        .replace(
            "  omega ETA_E0",
            "  theta RHO(0.6, 0.0, 0.95)\n  omega ETA_E0",
        )
        .replace(
            "weight = SE\n",
            "weight = SE\n  correlation car1(RHO) by OCC\n",
        );
    if profiled {
        text
    } else {
        text.replace(PROFILED, UNPROFILED)
    }
}

fn f1_car1_evaluated(
    profiled: bool,
    theta: Option<&[f64]>,
    step: Step,
) -> (ferx_core::PreparedRun, FitResult) {
    let path = unique_temp("ferx_profiled_levels_car1");
    std::fs::write(&path, f1_car1_text(profiled)).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(F1_DATA).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut params = p.init_params.clone();
    match theta {
        Some(t) => params.theta = t.to_vec(),
        None => {
            let rho = params.theta.len() - 1;
            params.theta[..4].copy_from_slice(&CAR1_OPTIMUM[..4]);
            params.theta[rho] = CAR1_OPTIMUM[4];
        }
    }
    params.omega = OmegaMatrix::from_diagonal(&[CAR1_OPTIMUM[5]], vec!["ETA_E0".into()]);
    params.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[CAR1_OPTIMUM[6]],
        vec!["KAPPA_ARM".into()],
    ));
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    };
    let opts = step(opts);
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// The profiled car1 fit's optimum, as `ferx` printed it (6 significant digits, macOS
/// arm64, OFV 89.451279; ferx-r's smoke step 13 reaches 89.45128 unprofiled): TVE0, EMAX,
/// ED50, ET50, RHO, then the variances of ETA_E0 and KAPPA_ARM. A point near the
/// minimum, so the profile Hessian is positive definite there — at F1's inits it is not
/// (three negative eigenvalues), and the step reports no SEs.
const CAR1_OPTIMUM: [f64; 7] = [
    48.553123, 14.149350, 35.822851, 4.508109, 0.759796, 4.826057, 141.219036,
];

fn profiled_car1() -> &'static (ferx_core::PreparedRun, FitResult) {
    static F: std::sync::OnceLock<(ferx_core::PreparedRun, FitResult)> = std::sync::OnceLock::new();
    F.get_or_init(|| f1_car1_evaluated(true, None, r))
}

/// C1 with the line: at one joint point ([`CAR1_OPTIMUM`], the levels profiled there)
/// the profiled covariance step and the unprofiled one give the same standard errors for
/// the structural θ, ρ, every level, ω and κ. The profiled side re-profiles the levels at
/// every stencil point through the solve, so this sees the solve's `C̃` inside `R`.
#[test]
fn the_correlated_profiled_covariance_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = profiled_car1();
    let (_, plain) = f1_car1_evaluated(false, Some(&prof.theta), r);
    assert!(
        (prof.ofv - plain.ofv).abs() < 1e-6,
        "the same point: {} vs {}",
        prof.ofv,
        plain.ofv
    );
    let (a, b) = (
        prof.se_theta
            .as_ref()
            .unwrap_or_else(|| panic!("profiled SE: {:?}", prof.warnings)),
        plain
            .se_theta
            .as_ref()
            .unwrap_or_else(|| panic!("unprofiled SE: {:?}", plain.warnings)),
    );
    let rho = a.len() - 1;
    let worst = |idx: &mut dyn Iterator<Item = usize>| {
        idx.map(|k| (a[k] - b[k]).abs() / b[k])
            .fold(0.0_f64, |acc, v| {
                assert!(v.is_finite(), "{a:?} vs {b:?}");
                acc.max(v)
            })
    };
    let worst_struct = worst(&mut (0..4).chain([rho]));
    let worst_level = worst(&mut (4..rho));
    let om = |f: &FitResult| f.se_omega.as_ref().unwrap()[0];
    let ka = |f: &FitResult| f.se_kappa.as_ref().unwrap()[0];
    let worst_var = ((om(prof) - om(&plain)).abs() / om(&plain))
        .max((ka(prof) - ka(&plain)).abs() / ka(&plain));
    assert!(worst_var.is_finite());
    eprintln!(
        "C1-corr: structural + ρ SE {worst_struct:.3e}, level SE {worst_level:.3e}, \
         variance SE {worst_var:.3e}"
    );
    assert!(worst_struct < C1_CORR_SE, "{worst_struct}");
    assert!(worst_level < C1_CORR_SE, "{worst_level}");
    assert!(worst_var < C1_CORR_SE, "{worst_var}");
}

/// The unprofiled side is the analytic R-matrix here too. Worst realised 4.0e-4
/// (structural + ρ), 1.6e-4 (levels), 5.9e-5 (ω, κ), macOS arm64, at the profiled default
/// step. C1's own bound, 7.5× headroom. At `1e-2` the structural gap is 4.5e-3.
const C1_CORR_SE: f64 = C1_BOUND;

/// `run_covariance` on a profiled fit with a `correlation` line repeats the step the fit
/// ran, bit for bit, in memory and after a `.fitrx` round trip (#2173's guarantee on the
/// new combination).
#[test]
fn run_covariance_replays_the_profiled_step_with_a_correlation_line() {
    let (p, f) = profiled_car1();
    assert_replays_bit_for_bit(p, f, &f1_car1_text(true), r);
}

// ── #2108: `covariance_method = s` and `rsr` on a profiled block ──────────────────

/// Worst relative gap of `a` against `b` over `idx`, each term asserted finite first.
fn worst_rel(a: &[f64], b: &[f64], idx: impl Iterator<Item = usize>) -> f64 {
    idx.map(|k| {
        let v = (a[k] - b[k]).abs() / b[k].abs();
        assert!(v.is_finite(), "{k}: {} vs {}", a[k], b[k]);
        v
    })
    .fold(0.0, f64::max)
}

/// Worst correlation-scale gap `|C_p − C_u| / sqrt(C_u,ii · C_u,jj)` over every entry of two
/// packed covariance matrices. A `FIX` coordinate's row and column are zero, and must be zero
/// on both sides.
fn worst_cov_gap(cp: &DMatrix<f64>, cu: &DMatrix<f64>) -> f64 {
    assert_eq!(cp.shape(), cu.shape());
    let mut worst = 0.0_f64;
    for i in 0..cu.nrows() {
        for j in 0..cu.ncols() {
            let s = (cu[(i, i)] * cu[(j, j)]).sqrt();
            if s == 0.0 {
                assert_eq!(cp[(i, j)], 0.0, "({i}, {j})");
                continue;
            }
            let g = ((cp[(i, j)] - cu[(i, j)]) / s).abs();
            assert!(
                g.is_finite(),
                "({i}, {j}): {} vs {}",
                cp[(i, j)],
                cu[(i, j)]
            );
            worst = worst.max(g);
        }
    }
    worst
}

/// The worst gaps of a profiled covariance step against the unprofiled one at the same
/// point, both run as `step`: the structural θ (`structural`), the levels (every other θ),
/// the Ω and κ variances, and the whole packed matrix on the correlation scale.
fn parity(prof: &FitResult, plain: &FitResult, structural: &[usize], step: Step) -> [f64; 4] {
    let method = step(FitOptions::default()).covariance_method;
    for (side, f) in [("profiled", prof), ("unprofiled", plain)] {
        assert_eq!(
            f.covariance_method,
            Some(method),
            "{side}: {:?}",
            f.warnings
        );
    }
    assert!(
        (prof.ofv - plain.ofv).abs() < 1e-6,
        "the same point: {} vs {}",
        prof.ofv,
        plain.ofv
    );
    let se = |f: &FitResult| f.se_theta.clone().unwrap();
    let (a, b) = (se(prof), se(plain));
    let levels = (0..a.len()).filter(|k| !structural.contains(k));
    let var = |f: &FitResult| {
        let mut v = f.se_omega.clone().unwrap();
        v.extend(f.se_kappa.clone().unwrap_or_default());
        v
    };
    let (va, vb) = (var(prof), var(plain));
    [
        worst_rel(&a, &b, structural.iter().copied()),
        worst_rel(&a, &b, levels),
        worst_rel(&va, &vb, 0..va.len()),
        worst_cov_gap(
            prof.covariance_matrix.as_ref().unwrap(),
            plain.covariance_matrix.as_ref().unwrap(),
        ),
    ]
}

/// The profiled F2 evaluation of [`profiled_at_nonmem_optimum`] under `rsr`, shared by C4
/// and its replay.
fn profiled_rsr_at_nonmem_optimum() -> &'static (ferx_core::PreparedRun, FitResult) {
    static F: std::sync::OnceLock<(ferx_core::PreparedRun, FitResult)> = std::sync::OnceLock::new();
    F.get_or_init(|| at_nonmem_optimum(true, None, rsr))
}

/// **C4.** At one joint point — NONMEM's structural optimum, the levels profiled there — the
/// profiled `rsr` step gives the unprofiled `R⁻¹SR⁻¹`: the same SEs for the structural θ,
/// every level, ω and κ, and the same matrix. The profiled structural block is
/// `A_p⁻¹(Σ sᵢᵉᶠᶠsᵢᵉᶠᶠᵀ)A_p⁻¹` over the efficient scores `sᵢθ − BD⁻¹sᵢβ`; with `sᵢθ` in their
/// place (the profiled `R⁻¹`'s θ-level block dropped before `S` is applied) the structural
/// SEs move by 6.8e-3 and the level SEs by 0.97 (macOS arm64), past the bounds. F2 is the
/// fixture for the structural half: on F1 and car1, linear in η with one subject per study,
/// each study's level scores are zero at `β̂`, so `sᵢᵉᶠᶠ = sᵢθ` there and only their level
/// blocks see the mutation. The straddle: the sandwich is not `R⁻¹`, so a step that ignored
/// `S` would fail too.
#[test]
fn c4_the_profiled_sandwich_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = profiled_rsr_at_nonmem_optimum();
    let (_, plain) = at_nonmem_optimum(false, Some(&prof.theta), rsr);
    let [structural, level, variance, matrix] = parity(prof, &plain, &[0, 1, 2], rsr);
    let (_, r_fit) = profiled_at_nonmem_optimum();
    let off_r = worst_rel(
        prof.se_theta.as_ref().unwrap(),
        r_fit.se_theta.as_ref().unwrap(),
        0..3,
    );
    eprintln!(
        "C4: structural SE {structural:.3e}, level SE {level:.3e}, variance SE {variance:.3e}, \
         matrix {matrix:.3e}; rsr against r {off_r:.3e}"
    );
    assert!(structural < C4_SE, "{structural}");
    assert!(level < C4_SE, "{level}");
    assert!(variance < C4_SE, "{variance}");
    assert!(matrix < C4_MATRIX, "{matrix}");
    assert!(off_r > C4_STRADDLE, "the sandwich is R⁻¹ here: {off_r}");
}

/// C4 with a residual `correlation car1(RHO) by OCC` line ([`CAR1_OPTIMUM`]), so `S` is
/// scored and `R` re-profiled through the line's dense `C̃`: the structural θ and ρ, every
/// level, ω and κ against the unprofiled `rsr`.
#[test]
fn c4_the_correlated_profiled_sandwich_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = f1_car1_evaluated(true, None, rsr);
    let (_, plain) = f1_car1_evaluated(false, Some(&prof.theta), rsr);
    let rho = prof.theta.len() - 1;
    let [structural, level, variance, matrix] = parity(&prof, &plain, &[0, 1, 2, 3, rho], rsr);
    eprintln!(
        "C4-corr: structural + ρ SE {structural:.3e}, level SE {level:.3e}, variance SE \
         {variance:.3e}, matrix {matrix:.3e}"
    );
    assert!(structural < C4_SE, "{structural}");
    assert!(level < C4_SE, "{level}");
    assert!(variance < C4_SE, "{variance}");
    assert!(matrix < C4_MATRIX, "{matrix}");
}

/// The profiled F1 fit's optimum (`ferx` CLI, release, macOS arm64, OFV 97.070036,
/// converged): TVE0, EMAX, ED50, ET50, then the variances of ETA_E0 and KAPPA_ARM.
const F1_OPTIMUM: [f64; 6] = [
    48.576414, 13.603909, 33.243705, 4.293363, 4.813371, 156.038443,
];

/// F1 (`mbma_placebo`: η and the arm κ on the block's rows) at [`F1_OPTIMUM`], its levels
/// profiled there, or at `theta` (levels included), with the covariance step `step`.
fn f1_evaluated(
    profiled: bool,
    theta: Option<&[f64]>,
    step: Step,
) -> (ferx_core::PreparedRun, FitResult) {
    let p = prep(F1, F1_DATA, profiled);
    let mut params = p.init_params.clone();
    match theta {
        Some(t) => params.theta = t.to_vec(),
        None => params.theta[..4].copy_from_slice(&F1_OPTIMUM[..4]),
    }
    params.omega = OmegaMatrix::from_diagonal(&[F1_OPTIMUM[4]], vec!["ETA_E0".into()]);
    params.omega_iov = Some(OmegaMatrix::from_diagonal(
        &[F1_OPTIMUM[5]],
        vec!["KAPPA_ARM".into()],
    ));
    let opts = step(FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    });
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// C4 on F1, where the between-arm κ shares the block's rows: the structural θ, every
/// level, ω and κ against the unprofiled `rsr`.
#[test]
fn c4_the_profiled_sandwich_with_an_arm_kappa_is_the_unprofiled_one() {
    let (_, prof) = f1_evaluated(true, None, rsr);
    let (_, plain) = f1_evaluated(false, Some(&prof.theta), rsr);
    let [structural, level, variance, matrix] = parity(&prof, &plain, &[0, 1, 2, 3], rsr);
    eprintln!(
        "C4-kappa: structural SE {structural:.3e}, level SE {level:.3e}, variance SE \
         {variance:.3e}, matrix {matrix:.3e}"
    );
    assert!(structural < C4_SE, "{structural}");
    assert!(level < C4_SE, "{level}");
    assert!(variance < C4_SE, "{variance}");
    assert!(matrix < C4_MATRIX, "{matrix}");
}

/// `nonmem_anchor/nested_levels_3` (sponsor > study > arm, the study and arm kappas inside
/// an `exp`) with a profiled per-study baseline `PLACEBO[STUDYID]` and `correlation ar1(RHO)
/// by ARM` (`profiled_levels_correlation_fits.rs`'s nested fixture), its initial estimates
/// set to the profiled fit's optimum (`ferx` CLI, release, macOS arm64, OFV 80.446292,
/// converged), as text.
fn nested_at_optimum_text(profiled: bool) -> String {
    let text = std::fs::read_to_string(root().join("nonmem_anchor/nested_levels_3.ferx")).unwrap();
    let mut text = text
        .replace(
            "  omega ETA_SP",
            "  theta PLACEBO[STUDYID](0.0) profile\n  theta RHO(0.019865, -0.95, 0.95)\n  omega ETA_SP",
        )
        .replace("  y = TVE0 + ", "  y = TVE0 + PLACEBO + ")
        .replace(
            "DV ~ additive(ADD)",
            "DV ~ additive(ADD)\n  correlation ar1(RHO) by ARM",
        );
    for (from, to) in [
        ("TVE0(10.1818,", "TVE0(10.280824,"),
        ("TVEMAX(22.0140,", "TVEMAX(22.268822,"),
        ("ED50(33.4048,", "ED50(34.640693,"),
        ("ET50(3.20150,", "ET50(3.307384,"),
        ("ETA_SP ~ 0.0699388", "ETA_SP ~ 0.092257"),
        ("K_ST ~ 0.0904362", "K_ST ~ 0.065103"),
        ("K_ARM ~ 3.75478", "K_ARM ~ 2.812739"),
        ("ADD ~ 0.255164", "ADD ~ 0.236655"),
    ] {
        assert!(text.contains(from), "fixture moved: {from}");
        text = text.replace(from, to);
    }
    if profiled {
        text
    } else {
        text.replace(PROFILED, UNPROFILED)
    }
}

/// [`nested_at_optimum_text`] evaluated at its initial estimates, or at `theta` (levels
/// included), with the covariance step `step`.
fn nested_evaluated(
    profiled: bool,
    theta: Option<&[f64]>,
    step: Step,
) -> (ferx_core::PreparedRun, FitResult) {
    let path = unique_temp("ferx_profiled_levels_nested");
    std::fs::write(&path, nested_at_optimum_text(profiled)).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(
            root()
                .join("nonmem_anchor/nested_levels_3.csv")
                .to_str()
                .unwrap(),
        ),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut params = p.init_params.clone();
    if let Some(t) = theta {
        params.theta = t.to_vec();
    }
    let opts = step(FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    });
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

/// C4 on nested levels with a residual correlation line: the structural θ and ρ, every
/// level, ω and both nested kappas against the unprofiled `rsr`.
///
/// Gated: 232 s in a local debug run (macOS arm64), the unprofiled stencil over 30 packed
/// coordinates on a 12-sponsor nested design. The per-PR C4s above die on both of the
/// sandwich's mutations (`sᵢθ` for `sᵢᵉᶠᶠ` on F2; `S` ignored on all three). Its own bounds:
/// this oracle is noise-limited, not truncation-limited — the structural gap is 3.2e-3 at
/// `RSR_FD_STEP`, 8.9e-4 at 3e-3 and 8.6e-2 at 3e-4 (macOS arm64), the re-solved
/// objectives' noise over `h²` taking over below 1e-3.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn c4_the_nested_profiled_sandwich_is_the_unprofiled_one() {
    let (_, prof) = nested_evaluated(true, None, rsr);
    let (_, plain) = nested_evaluated(false, Some(&prof.theta), rsr);
    let rho = prof.theta.len() - 1;
    assert_eq!(prof.theta.len(), plain.theta.len());
    let [structural, level, variance, matrix] = parity(&prof, &plain, &[0, 1, 2, 3, rho], rsr);
    eprintln!(
        "C4-nested: structural + ρ SE {structural:.3e}, level SE {level:.3e}, variance SE \
         {variance:.3e}, matrix {matrix:.3e}"
    );
    assert!(structural < C4_NESTED_SE, "{structural}");
    assert!(level < C4_NESTED_SE, "{level}");
    assert!(variance < C4_NESTED_SE, "{variance}");
    assert!(matrix < C4_NESTED_MATRIX, "{matrix}");
}

/// Worst realised 6.7e-3 (structural + ρ) and 1.3e-2 (matrix) on Linux aarch64 under the
/// nightly of 2026-10-24, 2.1e-3 and 4.2e-3 under that of 2026-10-25, and 3.2e-3 and 6.4e-3
/// on macOS arm64. It is a noise-limited oracle, so the numbers move 3× with the platform
/// and toolchain. 3× headroom on the worst.
const C4_NESTED_SE: f64 = 2e-2;
const C4_NESTED_MATRIX: f64 = 4e-2;

/// `run_covariance` on the profiled `rsr` evaluation repeats it bit for bit, in memory and
/// after a `.fitrx` round trip (#2173's guarantee, for the sandwich).
#[test]
fn run_covariance_replays_the_profiled_sandwich() {
    let (p, f) = profiled_rsr_at_nonmem_optimum();
    let model_text = std::fs::read_to_string(root().join(F2)).unwrap();
    assert_replays_bit_for_bit(p, f, &model_text, rsr);
}

/// Worst realised SE gap 6.3e-4 (F2's levels), Linux aarch64 and macOS arm64 alike to every
/// printed digit; 8.0e-5 (F1) and 1.1e-4 (car1) on the others. 3.2× headroom on F2. The
/// `sᵢθ` mutant lands at 6.8e-3, 3.4× above it.
const C4_SE: f64 = 2e-3;
/// Worst realised correlation-scale matrix gap 1.8e-3 (F2), both platforms; 2.8× headroom.
const C4_MATRIX: f64 = 5e-3;
/// The sandwich against the profiled `R⁻¹` on F2: realised 0.186 (both platforms); with `S`
/// ignored it is 7.4e-3. The floor sits between them.
const C4_STRADDLE: f64 = 5e-2;

/// S1's placebo model (`s1_profile.ferx`) with a visit block `PLACEBO[TIME]` in place of its
/// `[STUDY, TIME]` one, profiled or not, as text: 8 levels over 100 subjects, where a level
/// per study leaves `S` rank-deficient on every fixture above (one subject per study).
fn s1_visit_text(profiled: bool) -> String {
    let text = std::fs::read_to_string(root().join(S1)).unwrap();
    let block = "theta PLACEBO[STUDY, TIME](0.0) profile";
    assert!(text.contains(block), "fixture moved");
    let text = text.replace(block, "theta PLACEBO[TIME](0.0) profile");
    if profiled {
        text
    } else {
        text.replace(PROFILED, UNPROFILED)
    }
}

const S1_DATA_100: &str = "tests/data/profile_mbma/s100t8.csv";

/// [`s1_visit_text`] evaluated at its initial estimates, or at `theta` (levels included),
/// with the covariance step `step`.
fn s1_visit_evaluated(
    profiled: bool,
    theta: Option<&[f64]>,
    step: Step,
) -> (ferx_core::PreparedRun, FitResult) {
    let path = unique_temp("ferx_profiled_levels_s1_visit");
    std::fs::write(&path, s1_visit_text(profiled)).unwrap();
    let p = ferx_core::prepare_run(
        path.to_str().unwrap(),
        Some(root().join(S1_DATA_100).to_str().unwrap()),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut params = p.init_params.clone();
    if let Some(t) = theta {
        assert_eq!(t.len(), params.theta.len(), "one contrast on both sides");
        params.theta = t.to_vec();
    }
    let opts = FitOptions {
        verbose: false,
        outer_maxiter: 0,
        run_covariance_step: true,
        ..p.parsed.fit_options.clone()
    };
    let opts = step(opts);
    let f = ferx_core::fit(&p.parsed.model, &p.population, &params, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (p, f)
}

fn profiled_s1_visit_s() -> &'static (ferx_core::PreparedRun, FitResult) {
    static F: std::sync::OnceLock<(ferx_core::PreparedRun, FitResult)> = std::sync::OnceLock::new();
    F.get_or_init(|| s1_visit_evaluated(true, None, s))
}

/// `covariance_method = s` on a profiled block is the unprofiled `S⁻¹` at the same point:
/// the same matrix over the structural θ, every level, Ω and σ. `S⁻¹` reads no `R`, so this
/// needs no optimum and runs at the inits. Dies when the levels are scored as held (the
/// profile's own template), which leaves their rows of `S` out.
#[test]
fn the_profiled_cross_product_is_the_unprofiled_one_at_the_same_point() {
    let (_, prof) = profiled_s1_visit_s();
    let (_, plain) = s1_visit_evaluated(false, Some(&prof.theta), s);
    let [structural, level, variance, matrix] = parity(prof, &plain, &[0, 1, 2], s);
    eprintln!(
        "s: structural SE {structural:.3e}, level SE {level:.3e}, variance SE {variance:.3e}, \
         matrix {matrix:.3e}"
    );
    assert!(structural < S_SE, "{structural}");
    assert!(level < S_SE, "{level}");
    assert!(variance < S_SE, "{variance}");
    assert!(matrix < S_SE, "{matrix}");
}

/// Worst realised 2.3e-10 (Linux aarch64) and 3.0e-10 (macOS arm64): one estimator over the
/// same scores, the two sides' modes agreeing to `EVAL_INNER_TOL`. With the levels scored
/// as held the gap is 1.0. 3.3× headroom.
const S_SE: f64 = 1e-9;

/// `run_covariance` on the profiled `s` evaluation repeats it bit for bit.
#[test]
fn run_covariance_replays_the_profiled_cross_product() {
    let (p, f) = profiled_s1_visit_s();
    assert_replays_bit_for_bit(p, f, &s1_visit_text(true), s);
}

/// A level per study leaves `S` rank-deficient — F2 has 12 subjects and 41 free packed
/// coordinates — so `s` is refused with the unprofiled step's own reason on both sides,
/// never floored into SEs.
#[test]
fn a_rank_deficient_profiled_cross_product_is_refused_as_unprofiled() {
    let (_, prof) = at_nonmem_optimum(true, None, s);
    let (_, plain) = at_nonmem_optimum(false, Some(&prof.theta), s);
    let reason = "the score cross-product matrix S is singular or rank-deficient";
    for (side, f) in [("profiled", &prof), ("unprofiled", &plain)] {
        assert!(f.covariance_matrix.is_none(), "{side}");
        assert!(
            f.warnings.iter().any(|w| w.contains(reason)),
            "{side}: {:?}",
            f.warnings
        );
    }
}
