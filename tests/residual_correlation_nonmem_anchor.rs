//! NONMEM anchor for a within-group residual correlation, `correlation cs|ar1(RHO)
//! by ARM` (#1081).
//!
//! ## The object
//!
//! The naproxen MBMA (Boucher & Bennetts 2018, `examples/mbma_naproxen.ferx`) on the
//! natural DV scale, each arm's reported SE as a residual `weight =`, and the rows of
//! one arm correlated: `R_arm = σ² · W C(ρ) W`, `C` compound-symmetric or AR(1) over
//! the arm's visits. 18 studies (`ID`), 36 arms, 2–5 visits per arm at irregular
//! times, so every block size from 2 to 5 is exercised and the AR(1) lags go to 4.
//!
//! ## The NONMEM spelling (`nonmem_anchor/residual_correlation/*.ctl`)
//!
//! NONMEM has no residual-correlation record, so the same `R` is built by hand:
//! `L2 = ARM` makes `EPS(1..5)` one shared vector per arm (`$SIGMA 1 FIX` ×5), and
//! the row at visit `j` carries `RES = Σₖ Lⱼₖ(ρ)·EPS(k)`, `L` the Cholesky factor of
//! the arm's correlation matrix in closed form. `ρ = THETA(8)`. The model is linear
//! in η with an additive error, so FOCE is the exact marginal, and a pure-Python
//! closed form (`oracle.py`, in the anchor directory) agrees with NONMEM to
//! ≤ 2e-13 on every evaluation probe below.
//!
//! ## What is pinned
//!
//! * `residual_correlation_matches_nonmem_at_fixed_parameters` — one objective
//!   evaluation per structure and sign of ρ, at NONMEM's fixed parameters. The
//!   negative ones are what an implementation that squared or clamped ρ would get
//!   wrong; `cs_m02` sits near the compound-symmetry floor `−1/(n_max − 1) = −0.25`
//!   of a 5-visit arm. `cs_r0` is the ρ = 0 control (a free ρ at 0, so the line is
//!   kept and the dense path scores it).
//! * `cs_equals_an_arm_kappa_intercept` — a second engine inside ferx: CS with ρ ≥ 0
//!   is the covariance of an arm-level κ intercept loaded by `W`, scored by the IOV
//!   inner objective, which never builds a dense `R`.
//! * `residual_correlation_fit_reaches_nonmem_estimates` (Tier 3) — full fits.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, FitOptions, FitResult};

const DATA: &str = "nonmem_anchor/residual_correlation/napx1081.csv";
const MODEL: &str = include_str!("../nonmem_anchor/residual_correlation/napx_cs.ferx");

/// `(probe, kind, ρ, NONMEM OBJ)` — the `-1000000000` row of
/// `nonmem_anchor/residual_correlation/results/<probe>.ext` (NONMEM 7.6.0, `nm3`
/// anchor build, `MAXEVAL=0`).
const PROBES: &[(&str, &str, f64, f64)] = &[
    ("cs_r0", "cs", 0.0, -173.507_683_912_060_32),
    ("cs_p05", "cs", 0.5, -198.095_096_125_093_65),
    ("cs_m02", "cs", -0.2, -109.372_389_054_464_69),
    ("ar1_p05", "ar1", 0.5, -202.052_628_693_591_57),
    ("ar1_m04", "ar1", -0.4, -121.386_775_569_796_82),
];

/// Worst realised `|ferx − NONMEM|` over [`PROBES`]: 2.806e-9 (`ar1_m04`; macOS
/// arm64 and Linux agree, this is an exact likelihood). The gap is the FOCE inner
/// tolerance on η̂, which enters the objective quadratically. 10× headroom.
const PROBE_TOL: f64 = 3e-8;

/// The twin with `kind` and `ρ` substituted. A ρ of 0 is written free, not `FIX`:
/// a `FIX 0` line is dropped by the parser (pinned in the parser tests), and the
/// point of `cs_r0` is the dense path at ρ = 0.
fn twin(kind: &str, rho: f64) -> String {
    let rho_line = if rho == 0.0 {
        "theta RHO(0.0, -0.24, 0.99)".to_string()
    } else {
        format!("theta RHO({rho}, -0.24, 0.99) FIX")
    };
    MODEL
        .replace("theta RHO(0.5, -0.24, 0.99) FIX", &rho_line)
        .replace("correlation cs(RHO)", &format!("correlation {kind}(RHO)"))
}

fn run(src: &str, maxiter: Option<usize>, covariance: bool) -> FitResult {
    let parsed = parse_full_model(src).expect("twin parses");
    let model = parsed.model;
    let (pop, _) = read_population_for(
        &model,
        &parsed.covariate_decls,
        DATA,
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &[],
    )
    .expect("anchor data reads");
    let mut opts = FitOptions {
        run_covariance_step: covariance,
        ..parsed.fit_options
    };
    if let Some(m) = maxiter {
        opts.outer_maxiter = m;
    }
    fit(&model, &pop, &model.default_params, &opts).expect("fit runs")
}

#[test]
fn residual_correlation_matches_nonmem_at_fixed_parameters() {
    let mut worst = 0.0_f64;
    for &(probe, kind, rho, nm) in PROBES {
        let ofv = run(&twin(kind, rho), Some(0), false).ofv;
        // Finite first: `f64::max` would discard a NaN and keep the rows that worked.
        assert!(ofv.is_finite(), "{probe}: OFV {ofv}");
        let err = (ofv - nm).abs();
        eprintln!("{probe}: ferx {ofv:.12} NONMEM {nm:.12} |Δ| {err:.3e}");
        assert!(
            err < PROBE_TOL,
            "{probe}: ferx {ofv:.10} vs NONMEM {nm:.10} (|Δ| {err:.3e})"
        );
        worst = worst.max(err);
    }
    eprintln!("worst |Δ| = {worst:.3e}");
}

/// CS with ρ ≥ 0 on a weighted additive endpoint, `σ² W (ρJ + (1−ρ)I) W`, is the
/// covariance of an arm intercept `κ ~ N(0, ρσ²)` loaded by `W` plus an
/// independent residual of variance `(1−ρ)σ²` — scored here by the IOV inner
/// objective (`iov_column = ARM`), which never builds a dense `R`. At ρ = 0.5,
/// σ² = 1 both halves are 0.5. The κ model is linear in κ too, so it is exact.
///
/// Realised `|Δ|` 2.3e-10 (macOS arm64): the two engines' inner tolerances. ~9× headroom.
#[test]
fn cs_equals_an_arm_kappa_intercept() {
    let cs = run(&twin("cs", 0.5), Some(0), false).ofv;
    let kappa_src = MODEL
        .replace("theta RHO(0.5, -0.24, 0.99) FIX\n", "")
        .replace("  correlation cs(RHO) by ARM\n", "")
        .replace(
            "sigma ADD_ERR ~ 1.0 (variance) FIX",
            "sigma ADD_ERR ~ 0.5 (variance) FIX\n  kappa K_ARM ~ 0.5 FIX",
        )
        .replace(
            "E0   = TVE0   + B_E0_FLARE * FLARE + ETA_E0",
            "E0   = TVE0   + B_E0_FLARE * FLARE + ETA_E0 + WPSE * K_ARM",
        )
        .replace("method = focei", "method = focei\n  iov_column = ARM");
    assert!(
        kappa_src.contains("WPSE * K_ARM")
            && kappa_src.contains("iov_column = ARM")
            && !kappa_src.contains("correlation cs("),
        "the κ twin must be the κ model and not the correlation one"
    );
    let kappa = run(&kappa_src, Some(0), false).ofv;
    assert!(cs.is_finite() && kappa.is_finite(), "cs {cs} κ {kappa}");
    let err = (cs - kappa).abs();
    eprintln!("cs {cs:.12} κ {kappa:.12} |Δ| {err:.3e}");
    assert!(err < 2e-9, "dense-R CS {cs:.10} vs κ intercept {kappa:.10}");
    // And both are the NONMEM number, so neither engine agrees with the other by
    // sharing a wrong convention.
    assert!((cs - PROBES[1].3).abs() < PROBE_TOL);
}

/// NONMEM's estimates (`cs_est` / `ar1_est`, `METHOD=1`, every θ and ω free, ρ
/// started at 0.3): `(kind, OBJ, ρ̂, SE(ρ̂) from $COV MATRIX=R)`. Both minimisations
/// report `MINIMIZATION SUCCESSFUL`.
const ESTIMATES: &[(&str, f64, f64, f64)] = &[
    ("cs", -199.734_575_047_940_30, 0.509_450, 0.077_591_7),
    ("ar1", -205.932_727_647_201_43, 0.597_282, 0.063_339_4),
];

/// The twin with every θ and ω free and ρ started where NONMEM started it, and
/// tight outer tolerances. At the default tolerances BOBYQA stops ~0.03 OFV short
/// (macOS: −199.7001 for `cs`) on the flat TVEMAX / B_EMAX_FLARE ridge this dataset
/// has — the same objective, measured: ferx scores NONMEM's optimum at −199.734575.
fn free_twin(kind: &str) -> String {
    MODEL
        .lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("theta RHO(") {
                "  theta RHO(0.3, -0.24, 0.99)".to_string()
            } else if t.starts_with("theta ") || t.starts_with("omega ") {
                l.trim_end().trim_end_matches("FIX").trim_end().to_string()
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        .replace("correlation cs(RHO)", &format!("correlation {kind}(RHO)"))
        .replace(
            "method = focei",
            "method = focei\n  optimizer = bobyqa\n  outer_ftol = 1e-10\n  outer_xtol = 1e-8\n  \
             stagnation_guard = false",
        )
}

/// Bands from the realised error. Linux aarch64 (the reference platform):
/// `cs` |ΔOFV| 1.2e-5, |Δρ̂| 6.1e-5, SE 0.05%; `ar1` |ΔOFV| 1e-6, |Δρ̂| 6e-6,
/// SE 0.1%. macOS arm64: `cs` 1e-6 / 4.2e-5 / 0.05%, `ar1` 6.3e-5 / 2.3e-5 / 0.1%.
/// Bounds: OFV 1e-3, ρ̂ 5e-4, SE 1% — ≥ 8× the worst of either platform.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: full fit to convergence (#1081); opt in with --features slow-tests"
)]
fn residual_correlation_fit_reaches_nonmem_estimates() {
    for &(kind, nm_ofv, nm_rho, nm_se) in ESTIMATES {
        let r = run(&free_twin(kind), None, true);
        let rho = r.theta[7];
        let se = r.se_theta.as_ref().map(|s| s[7]).unwrap_or(f64::NAN);
        eprintln!(
            "{kind}: OFV {:.6} (NONMEM {nm_ofv:.6}) ρ̂ {rho:.6} (NONMEM {nm_rho:.6}) \
             SE {se:.6} (NONMEM {nm_se:.6})",
            r.ofv
        );
        assert!(r.ofv.is_finite() && rho.is_finite() && se.is_finite());
        assert!((r.ofv - nm_ofv).abs() < 1e-3, "{kind}: OFV {}", r.ofv);
        assert!((rho - nm_rho).abs() < 5e-4, "{kind}: ρ̂ {rho}");
        assert!((se - nm_se).abs() / nm_se < 0.01, "{kind}: SE(ρ̂) {se}");
    }
}

/// NONMEM's `cs_est` optimum (`.ext` row `-1000000000`): θ₁…θ₇, ρ, ω²(E0), ω²(EMAX).
const CS_EST: [f64; 10] = [
    5.18709, 1.14782, 0.634199, 0.975034, 0.832231, 0.791306, -0.966685, 0.509450, 0.390267,
    0.557777,
];

/// NONMEM's `ar1_est` optimum, same layout as [`CS_EST`].
const AR1_EST: [f64; 10] = [
    5.18727, 1.13276, 0.599296, 0.974667, 0.833842, 0.819166, -0.839378, 0.597282, 0.389372,
    0.557854,
];

/// The free twin of `kind` started at a NONMEM optimum `est`.
fn at_optimum(kind: &str, est: &[f64; 10]) -> String {
    let names = [
        "TVE0",
        "TVEMAX",
        "TVET50",
        "B_E0_FLARE",
        "B_EMAX_FLARE",
        "B_EMAX_TRT",
        "B_LET50_TRT",
    ];
    let mut src = free_twin(kind).replace("theta RHO(0.3,", &format!("theta RHO({},", est[7]));
    for (i, name) in names.iter().enumerate() {
        let decl = MODEL
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("theta {name}(")))
            .expect("declared")
            .trim()
            .trim_end_matches("FIX")
            .trim_end()
            .to_string();
        let init = decl[decl.find('(').unwrap() + 1..decl.find(',').unwrap()].to_string();
        src = src.replace(
            &decl,
            &decl.replacen(&format!("({init},"), &format!("({},", est[i]), 1),
        );
    }
    src = src
        .replace(
            "ETA_E0   ~ 0.3862622500000001",
            &format!("ETA_E0   ~ {}", est[8]),
        )
        .replace("ETA_EMAX ~ 0.54567769", &format!("ETA_EMAX ~ {}", est[9]));
    src
}

/// T13: the covariance step at NONMEM's own optimum (one evaluation, every θ and ω
/// free, `outer_maxiter = 0`). Covariance under a `correlation` line takes the FD
/// Hessian route (`CovScopeDecline::ResidualCorrelations`), and SE(ρ̂) must be
/// NONMEM's `$COV MATRIX=R` value. Realised (macOS arm64): SE 0.077541 against
/// 0.077592, 0.07% (bound 0.5%). The objective is exact there too, so this is Tier 2.
#[test]
fn covariance_at_the_nonmem_optimum_gives_its_se_of_rho() {
    let r = run(&at_optimum("cs", &CS_EST), Some(0), true);
    assert!(
        (r.ofv - ESTIMATES[0].1).abs() < 1e-4,
        "at NONMEM's optimum: {}",
        r.ofv
    );
    let se = r.se_theta.as_ref().map(|s| s[7]).unwrap_or(f64::NAN);
    eprintln!("SE(ρ̂) ferx {se:.6} NONMEM {:.6}", ESTIMATES[0].3);
    assert!(se.is_finite(), "SE(ρ̂) {se}");
    assert!(
        (se - ESTIMATES[0].3).abs() / ESTIMATES[0].3 < 5e-3,
        "SE(ρ̂) ferx {se} vs NONMEM {}",
        ESTIMATES[0].3
    );
}

/// #1081 follow-up: under a `correlation` line the Durbin–Watson statistic is taken
/// on the residuals decorrelated by the fitted `R`, and on this dataset that is a
/// live diagnostic, not a constant. At NONMEM's optima (macOS arm64 and Linux
/// alike):
///
/// | | raw IWRES DW | decorrelated DW |
/// |---|---|---|
/// | `cs` | 0.987 | 1.427 |
/// | `ar1` | 0.938 | 1.691 |
///
/// Raw IWRES would warn on both — they are correlated by construction. Decorrelated,
/// `cs` still warns (a serial decay compound symmetry cannot carry), and with the
/// `ar1` advice; `ar1` does not. The `ar1` fit is the 6-OFV-better one.
#[test]
fn durbin_watson_is_taken_on_decorrelated_residuals() {
    let mut seen = Vec::new();
    for (kind, est, want_dw) in [("cs", &CS_EST, 1.4267), ("ar1", &AR1_EST, 1.6909)] {
        let r = run(&at_optimum(kind, est), Some(0), false);
        let raw = ferx_core::stats::residual_error::iwres_autocorrelation(&r.subjects);
        assert!(r.dw_statistic.is_finite() && raw.1.is_finite());
        eprintln!(
            "{kind}: decorrelated DW {:.4}, raw IWRES DW {:.4}",
            r.dw_statistic, raw.1
        );
        assert!(
            (r.dw_statistic - want_dw).abs() < 1e-3,
            "{kind}: DW {}",
            r.dw_statistic
        );
        assert!(raw.1 < 1.5, "{kind}: raw IWRES would warn (the straddle)");
        let dw: Vec<&String> = r.warnings.iter().filter(|w| w.contains("Durbin")).collect();
        seen.push((kind, dw.len()));
        if kind == "cs" {
            assert_eq!(dw.len(), 1, "{dw:?}");
            assert!(dw[0].starts_with(
                "Positive autocorrelation remains in the residuals decorrelated by \
                 `correlation cs(RHO) by ARM` (Durbin-Watson = 1.43)"
            ));
            assert!(dw[0].contains("`ar1` models that"), "{}", dw[0]);
        } else {
            assert!(dw.is_empty(), "{dw:?}");
        }
    }
    assert_eq!(seen, vec![("cs", 1), ("ar1", 0)]);
}
