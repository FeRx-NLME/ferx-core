//! NONMEM anchor for a within-group residual correlation, `correlation
//! cs|ar1|car1(RHO) by ARM` (#1081).
//!
//! ## The object
//!
//! The naproxen MBMA (Boucher & Bennetts 2018, `examples/mbma_naproxen.ferx`) on the
//! natural DV scale, each arm's reported SE as a residual `weight =`, and the rows of
//! one arm correlated: `R_arm = σ² · W C(ρ) W`, `C` compound-symmetric, AR(1) over
//! the arm's visits, or continuous-time AR(1) over its visit *times*
//! (`C_jk = ρ^|t_j − t_k|`). 18 studies (`ID`), 36 arms, 2–5 visits per arm at
//! irregular times, so every block size from 2 to 5 is exercised, the AR(1) lags go
//! to 4, and the `car1` gaps are not the visit lags (week 0, 2, 12 is lag 1 and 2
//! but Δt 2 and 12).
//!
//! ## The NONMEM spelling (`nonmem_anchor/residual_correlation/*.ctl`)
//!
//! NONMEM has no residual-correlation record, so the same `R` is built by hand:
//! `L2 = ARM` makes `EPS(1..5)` one shared vector per arm (`$SIGMA 1 FIX` ×5), and
//! the row at visit `j` carries `RES = Σₖ Lⱼₖ(ρ)·EPS(k)`, `L` the Cholesky factor of
//! the arm's correlation matrix in closed form. `ρ = THETA(8)`; for `car1` the
//! factor reads the arm's visit times `T1..T5` from the data. The model is linear
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
    ("car_p08", "car1", 0.8, -181.691_383_296_860_49),
    ("car_p03", "car1", 0.3, -176.591_803_240_544_78),
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

/// NONMEM's estimates (`cs_est` / `ar1_est` / `car_est`, `METHOD=1`, every θ and ω
/// free, ρ started at 0.3 — `car1` at 0.5): `(kind, OBJ, ρ̂, SE(ρ̂) from $COV
/// MATRIX=R)`. All three minimisations report `MINIMIZATION SUCCESSFUL`. `car1`'s ρ̂
/// is per week (the data `TIME` unit).
const ESTIMATES: &[(&str, f64, f64, f64)] = &[
    ("cs", -199.734_575_047_940_30, 0.509_450, 0.077_591_7),
    ("ar1", -205.932_727_647_201_43, 0.597_282, 0.063_339_4),
    ("car1", -195.027_166_348_762_39, 0.813_375, 0.037_248_6),
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
                // `car1` needs ρ ≥ 0, and starts where NONMEM's `car_est` did.
                if kind == "car1" {
                    "  theta RHO(0.5, 0.0, 0.99)".to_string()
                } else {
                    "  theta RHO(0.3, -0.24, 0.99)".to_string()
                }
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
/// SE 0.1%; `car1` |ΔOFV| 1e-6, |Δρ̂| 3e-6, SE 0.30%. macOS arm64: `cs` 1e-6 /
/// 4.2e-5 / 0.05%, `ar1` 6.3e-5 / 2.3e-5 / 0.1%.
/// Bounds: OFV 1e-3, ρ̂ 5e-4, SE 1% (`car1` 2.5%) — ≥ 8× the worst realised error.
/// `car1`'s SE sits 0.30% from NONMEM's, and a shared 1% bound would leave it 3.3×
/// headroom. The gap is ferx's covariance step, not the likelihood and not ρ's log
/// packing (#1974 review r1 #2, macOS arm64): the same −0.30% at NONMEM's own optimum
/// (`covariance_at_the_nonmem_optimum_gives_its_se_of_rho`); log-packing `cs`/`ar1`'s ρ
/// moves their SE by < 0.04%; and a central-FD Hessian of ferx's own OFV over all 10
/// (θ, ω²) at NONMEM's optimum gives NONMEM's SE to ~1e-4 for all three kinds. The
/// covariance step runs low on all three (`cs` −0.06%, `ar1` −0.11%, `car1` −0.30%);
/// that is #1975.
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
        let se_tol = if kind == "car1" { 0.025 } else { 0.01 };
        assert!((se - nm_se).abs() / nm_se < se_tol, "{kind}: SE(ρ̂) {se}");
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

/// NONMEM's `car_est` optimum, same layout as [`CS_EST`].
const CAR_EST: [f64; 10] = [
    5.18040, 1.05510, 0.459088, 0.973999, 0.834837, 0.908636, -0.519755, 0.813375, 0.388628,
    0.546406,
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
    // `free_twin` starts ρ at 0.3, or at 0.5 under `car1`.
    let mut src = free_twin(kind)
        .replace("theta RHO(0.3,", &format!("theta RHO({},", est[7]))
        .replace("theta RHO(0.5,", &format!("theta RHO({},", est[7]));
    assert!(
        src.contains(&format!("theta RHO({},", est[7])),
        "{kind}: ρ moved to NONMEM's optimum"
    );
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
/// NONMEM's `$COV MATRIX=R` value. The objective is exact there too, so this is
/// Tier 2. Realised (macOS arm64): `cs` SE 0.077541 against 0.077592, −0.07% (bound
/// 0.5%); `car1` 0.0371379 against 0.0372486, −0.30% (bound 2.5%, the full-fit band) —
/// the same gap the full fit shows, so the optimum is not where it comes from (#1974
/// review r1 #2; the covariance step's bias is #1975).
#[test]
fn covariance_at_the_nonmem_optimum_gives_its_se_of_rho() {
    for (kind, est, nm, tol) in [
        ("cs", &CS_EST, ESTIMATES[0], 5e-3),
        ("car1", &CAR_EST, ESTIMATES[2], 0.025),
    ] {
        assert_eq!(nm.0, kind);
        let r = run(&at_optimum(kind, est), Some(0), true);
        assert!(
            (r.ofv - nm.1).abs() < 1e-4,
            "{kind}: at NONMEM's optimum: {}",
            r.ofv
        );
        let se = r.se_theta.as_ref().map(|s| s[7]).unwrap_or(f64::NAN);
        eprintln!(
            "{kind}: OFV {:.9} SE(ρ̂) ferx {se:.7} NONMEM {:.7} ({:+.3}%)",
            r.ofv,
            nm.3,
            100.0 * (se / nm.3 - 1.0)
        );
        assert!(se.is_finite(), "{kind}: SE(ρ̂) {se}");
        assert!(
            (se - nm.3).abs() / nm.3 < tol,
            "{kind}: SE(ρ̂) ferx {se} vs NONMEM {}",
            nm.3
        );
    }
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

/// Plan T13, SIR leg (#1081 review r1 #8): a short SIR at NONMEM's `cs` optimum
/// keeps every resampled ρ inside its θ box `[−0.24, 0.99]` — ρ packs on the linear
/// scale (its lower bound is negative), so this is the box doing the work, not a
/// log transform — and the draws spread (a degenerate SIR would pass the box
/// check with one repeated value). At ρ̂ = 0.51 ± 0.08 the box is ≥ 6 SE away, so
/// the check that carries weight here is the last one: the draws are centred on ρ̂
/// with the spread of SE(ρ̂), i.e. SIR scores the correlated objective.
#[test]
fn sir_keeps_every_rho_draw_inside_its_theta_box() {
    let parsed = parse_full_model(&at_optimum("cs", &CS_EST)).expect("parses");
    let model = parsed.model;
    let (pop, _) =
        read_population_for(&model, &parsed.covariate_decls, DATA, None, None, None, &[])
            .expect("anchor data reads");
    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: true,
        sir: true,
        sir_samples: 300,
        sir_resamples: 150,
        sir_seed: Some(1081),
        sir_keep_samples: true,
        ..parsed.fit_options
    };
    let r = fit(&model, &pop, &model.default_params, &opts).expect("fit runs");
    let draws = r.sir_resamples_packed.as_ref().expect("resamples kept");
    let rho: Vec<f64> = draws.iter().map(|d| d[7]).collect();
    assert!(rho.len() >= 100, "{} draws", rho.len());
    assert!(rho.iter().all(|v| v.is_finite()), "non-finite ρ draw");
    let (lo, hi) = rho
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let n = rho.len() as f64;
    let mean = rho.iter().sum::<f64>() / n;
    let sd = (rho.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    eprintln!(
        "SIR ρ draws: {} in [{lo:.4}, {hi:.4}], mean {mean:.4} sd {sd:.4}, ESS {:?}",
        rho.len(),
        r.sir_ess
    );
    assert!(
        lo >= -0.24 && hi <= 0.99,
        "ρ draw outside its box: [{lo}, {hi}]"
    );
    assert!(hi - lo > 0.05, "draws do not spread: [{lo}, {hi}]");
    // And they sample the right objective: centred on ρ̂ and spread like SE(ρ̂).
    // Realised (seed 1081): mean 0.4867 against ρ̂ 0.5095 (0.29 SE), sd 0.0818
    // against NONMEM's SE 0.0776 (+5%). Bounds: half an SE, and ±25%.
    let (rho_hat, se) = (CS_EST[7], ESTIMATES[0].3);
    assert!(
        (mean - rho_hat).abs() < 0.5 * se,
        "SIR mean {mean} vs ρ̂ {rho_hat}"
    );
    assert!((sd / se - 1.0).abs() < 0.25, "SIR sd {sd} vs SE {se}");
}

/// Combined-error probes (#1081 review r1 #9): `(probe, kind, ρ, NONMEM OBJ)`, FOCEI
/// (`METHOD=1 INTER MAXEVAL=0`), `nonmem_anchor/residual_correlation/results/comb_*.ext`.
/// NONMEM: `Y = F + F·θ₉·RESP + WPSE·RESA`, `θ₉ = 0.03`, both parts mixed through the
/// same Cholesky factor of `C(ρ)` from independent `EPS(1..5)` / `EPS(6..10)`, so
/// `R_jk = C_jk (f_j f_k θ₉² + w_j w_k)`. Here `R` depends on `f(η)`: the serial
/// `∂R/∂f` enters the FOCEI inner search and its interaction term, which the
/// additive-only probes above cannot see. The ρ = 0 control isolates the
/// combined-error convention from the correlation.
const COMB_PROBES: &[(&str, &str, f64, f64)] = &[
    ("comb_cs_r0", "cs", 0.0, -161.469_171_370_695_25),
    ("comb_cs_p05", "cs", 0.5, -189.777_337_517_105_99),
    ("comb_ar1_m04", "ar1", -0.4, -130.742_944_153_769_29),
    ("comb_ar1_p05", "ar1", 0.5, -189.944_415_907_175_34),
    ("comb_car_p03", "car1", 0.3, -164.480_733_697_776_77),
    ("comb_car_p08", "car1", 0.8, -179.360_463_309_165_65),
];

/// Worst realised `|ferx − NONMEM|` over [`COMB_PROBES`]: 6.2e-8 (`comb_cs_p05`,
/// macOS arm64) — an order above the additive probes because the FOCEI inner search
/// now has an `f`-dependent `R` to converge on. ~10× headroom.
const COMB_TOL: f64 = 6e-7;

/// The twin with a combined error: proportional SD 0.03, the weighted additive part
/// unchanged.
fn comb_twin(kind: &str, rho: f64) -> String {
    twin(kind, rho)
        .replace(
            "  sigma ADD_ERR ~ 1.0 (variance) FIX",
            "  sigma PROP_ERR ~ 0.03 (sd) FIX\n  sigma ADD_ERR ~ 1.0 (variance) FIX",
        )
        .replace(
            "DV ~ additive(ADD_ERR) weight = WPSE",
            "DV ~ combined(PROP_ERR, ADD_ERR) weight = WPSE",
        )
}

#[test]
fn combined_error_correlation_matches_nonmem_focei() {
    let mut worst = 0.0_f64;
    for &(probe, kind, rho, nm) in COMB_PROBES {
        let src = comb_twin(kind, rho);
        assert!(src.contains("DV ~ combined(PROP_ERR, ADD_ERR) weight = WPSE"));
        let ofv = run(&src, Some(0), false).ofv;
        assert!(ofv.is_finite(), "{probe}: OFV {ofv}");
        let err = (ofv - nm).abs();
        eprintln!("{probe}: ferx {ofv:.12} NONMEM {nm:.12} |Δ| {err:.3e}");
        assert!(
            err < COMB_TOL,
            "{probe}: ferx {ofv:.10} vs NONMEM {nm:.10} (|Δ| {err:.3e})"
        );
        worst = worst.max(err);
    }
    eprintln!("combined worst |Δ| = {worst:.3e}");
}
