//! NONMEM anchor for `block_sigma` together with IOV (#979): a correlated total/unbound
//! residual pair on a model with an occasion κ on CL, fitted by FOCEI.
//!
//! ## The kit (`nonmem_anchor/block_sigma_iov.*`)
//!
//! - `block_sigma_iov.csv` — `simulate_block_sigma_iov_data.py` (stdlib, seed 979): 40
//!   subjects, one-compartment IV bolus, three occasions each opened by a dose of 100 (t =
//!   0, 24, 48), so occasions 2 and 3 start on residual drug and κ is live on both sides
//!   of every occasion boundary. A total (`FREE = 0`) / unbound (`FREE = 1`) pair at 1,
//!   4, 12 and 20 h after each dose shares an `L2` id; the two proportional sigmas
//!   (0.10, 0.20) are correlated ρ = 0.6 across the pair.
//! - `block_sigma_iov.ctl` (outputs in `results/`) — `ADVAN1`, κ as occasion `ETA`s with
//!   `$OMEGA BLOCK(1) SAME`, `$SIGMA BLOCK(2)`, `METHOD=1 INTER`. NONMEM 7.5.1 stops with
//!   `ROUNDING ERRORS (ERROR=134)` at OBJ 250.40112720622059 — the estimates are those
//!   of a converged fit (ρ̂ = 0.617 against the simulated 0.6), only the final-digit
//!   test failed.
//! - `block_sigma_iov_eval.ctl` — the same model at that run's estimates as `.ext` prints
//!   them, `MAXEVAL=0 POSTHOC`: OBJ 250.40112720969563 and every η̂, κ̂ in
//!   `results/block_sigma_iov_eval.tab` at full precision.
//! - `block_sigma_iov.ferx` — the ferx twin, with the control stream's initial estimates.
//!
//! ## Measured (macOS arm64)
//!
//! | quantity | worst \|ferx − NONMEM\| | bound |
//! |---|---|---|
//! | OFV at NONMEM's estimates | 5.327e-8 | `5e-7` (9.4× above) |
//! | η̂ and κ̂ (40 × 4 values, `FORMAT=s1PE23.16`) | 1.490e-7 | `1e-6` (6.7× above) |
//! | OFV of the full FOCEI fit vs NONMEM's (slow-gated) | 2.242e-7 (ferx lower) | `1e-3` |
//! | ρ̂ of the full fit vs NONMEM's 0.617090 | 0.61709 | `1e-3` |
//!
//! Both loops run analytic: the EBE search on the adjoint kernel's exact gradient (a
//! `block_sigma` IOV subject always takes it, `iov_inner_residual_correlation`), the
//! outer gradient on the adjoint's — asserted below, with no FD-fallback warning.
//!
//! The twin *declares* a different ρ (0.315) from the one it is scored at (NONMEM's
//! 0.617, passed to `fit()` as a parameter, as the outer loop passes it), so every site
//! on the objective path — the IOV marginal, the EBE search, the reported OFV — must read
//! the live ρ. The controls: the same point at the declared ρ, and with the pair
//! uncorrelated, each move the OFV by whole units.

use ferx_core::api::read_population_for;
use ferx_core::build_info::GradientMethodKind;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, EstimationMethod, FitOptions, FitResult};

/// NONMEM 7.5.1 `OBJ` at the evaluated point (`results/block_sigma_iov_eval.ext`).
const NM_EVAL_OFV: f64 = 250.40112720969563;
/// NONMEM 7.5.1 `OBJ` of the FOCEI fit (`results/block_sigma_iov.ext`).
const NM_FIT_OFV: f64 = 250.40112720622059;
/// Worst realised |ΔOFV| 5.327e-8 (macOS arm64); `5e-7` is 9.4× above it. The
/// uncorrelated control lands 199 units away.
const OFV_TOL: f64 = 5e-7;
/// Worst realised |Δ| over every η̂ and κ̂ 1.490e-7 (macOS arm64); `1e-6` is 6.7× above.
const EBE_TOL: f64 = 1e-6;

const MODEL: &str = include_str!("../nonmem_anchor/block_sigma_iov.ferx");

/// NONMEM's ρ̂, from `block_sigma_iov.ext`'s `SIGMA(2,1)` and the two variances.
fn nm_rho() -> f64 {
    0.0142640 / (0.0119830f64 * 0.0445881).sqrt()
}

/// The twin at `block_sigma_iov_eval.ctl`'s point (`block_sigma_iov.ext`, as printed),
/// except the `block_sigma` off-diagonal, which keeps the file's covariance (0.006,
/// declared ρ = 0.315 at these variances): the anchor hands ρ to `fit()` as a
/// parameter, the way the outer loop does, so a site scoring the *declared* ρ instead
/// of the live one misses NONMEM by whole OFV units.
fn at_nonmem_estimates() -> String {
    let swap = |s: String, from: &str, to: &str| {
        assert!(s.contains(from), "{from}");
        s.replace(from, to)
    };
    let s = swap(
        MODEL.to_string(),
        "theta TVCL(1.2, 0.1, 10.0)",
        "theta TVCL(0.977516, 0.1, 10.0)",
    );
    let s = swap(
        s,
        "theta TVV(12.0, 1.0, 100.0)",
        "theta TVV(9.96803, 1.0, 100.0)",
    );
    let s = swap(s, "omega ETA_CL ~ 0.1", "omega ETA_CL ~ 0.110390");
    let s = swap(s, "kappa KAPPA_CL ~ 0.03", "kappa KAPPA_CL ~ 0.0350388");
    swap(
        s,
        "[0.0144, 0.006, 0.0324]",
        "[0.0119830, 0.006, 0.0445881]",
    )
}

/// `fit()` from the twin's declared parameters, with the `block_sigma` ρ replaced by
/// `rho` when given.
fn run(model_src: &str, outer_maxiter: usize, rho: Option<f64>) -> FitResult {
    let parsed = parse_full_model(model_src).expect("anchor model must parse");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "nonmem_anchor/block_sigma_iov.csv",
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data must load");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        outer_maxiter,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let mut params = parsed.model.default_params.clone();
    if let Some(rho) = rho {
        assert!(
            (params.residual_correlations[0].rho - rho).abs() > 0.1,
            "the live ρ must differ from the declared one"
        );
        params.residual_correlations[0].rho = rho;
    }
    fit(&parsed.model, &pop, &params, &opts).expect("anchor fit")
}

/// `ID → [ETA1..ETA4]` from the `FIRSTONLY` table, in file order.
fn nonmem_ebes() -> Vec<(String, [f64; 4])> {
    let text = std::fs::read_to_string("nonmem_anchor/results/block_sigma_iov_eval.tab").unwrap();
    text.lines()
        .skip(2)
        .map(|l| {
            let f: Vec<f64> = l.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (format!("{}", f[0] as u32), [f[3], f[4], f[5], f[6]])
        })
        .collect()
}

/// ferx's OFV and every η̂ and κ̂ at NONMEM's FOCEI estimates.
#[test]
fn block_sigma_with_iov_matches_nonmem_at_its_estimates() {
    let r = run(&at_nonmem_estimates(), 0, Some(nm_rho()));
    let d_ofv = (r.ofv - NM_EVAL_OFV).abs();
    eprintln!(
        "ferx OFV {:.14}  NONMEM {NM_EVAL_OFV:.14}  |Δ| {d_ofv:.3e}",
        r.ofv
    );
    assert!(
        r.ofv.is_finite() && d_ofv < OFV_TOL,
        "OFV {} vs NONMEM {NM_EVAL_OFV}",
        r.ofv
    );

    let nm = nonmem_ebes();
    assert_eq!(nm.len(), 40);
    assert_eq!(r.subjects.len(), 40);
    let mut worst = 0.0f64;
    for (i, (id, eta)) in nm.iter().enumerate() {
        assert_eq!(&r.subjects[i].id, id, "subject order");
        let kappas = &r.ebe_kappas[i];
        assert_eq!(kappas.len(), 3, "subject {id}: one κ̂ per occasion");
        let got = [
            r.subjects[i].eta[0],
            kappas[0][0],
            kappas[1][0],
            kappas[2][0],
        ];
        for (k, (&g, &w)) in got.iter().zip(eta).enumerate() {
            assert!(g.is_finite(), "subject {id} ETA{}: {g}", k + 1);
            let d = (g - w).abs();
            assert!(
                d < EBE_TOL,
                "subject {id} ETA{}: ferx {g} vs NONMEM {w}",
                k + 1
            );
            worst = worst.max(d);
        }
    }
    eprintln!("worst |Δ EBE| {worst:.3e}");

    // The per-subject contributions the sdtab and the fit YAML report are scored at the
    // same live ρ as the objective (`compute_subject_results`), so they sum to it.
    let summed: f64 = r.subjects.iter().map(|s| s.ofv_contribution).sum();
    eprintln!(
        "Σ subject contributions {summed:.10}, |Δ| {:.3e}",
        (summed - r.ofv).abs()
    );
    assert!(
        (summed - r.ofv).abs() < 1e-6,
        "Σ contributions {summed} vs OFV {}",
        r.ofv
    );

    eprintln!(
        "routes: inner {}, outer {}",
        r.gradient_method_inner, r.gradient_method_outer
    );
    assert_eq!(
        r.gradient_method_inner,
        GradientMethodKind::Analytic.as_str()
    );
    assert_eq!(
        r.gradient_method_outer,
        GradientMethodKind::Analytic.as_str()
    );
    assert!(
        !r.warnings.iter().any(|w| w.contains("finite-difference")),
        "{:?}",
        r.warnings
    );
}

/// The controls: NONMEM's point scored at the *declared* ρ (0.315), and with the pair
/// uncorrelated, are different models whole OFV units away — the anchor above can tell
/// which ρ reached the IOV marginal and its EBE search.
#[test]
fn block_sigma_with_iov_declared_and_uncorrelated_controls_diverge() {
    for (label, rho) in [("declared ρ", None), ("ρ = 0", Some(0.0))] {
        let r = run(&at_nonmem_estimates(), 0, rho);
        let d_ofv = (r.ofv - NM_EVAL_OFV).abs();
        eprintln!("{label} control: OFV {:.6}, |Δ| {d_ofv:.3}", r.ofv);
        assert!(r.ofv.is_finite() && d_ofv > 1.0, "{label}: OFV {}", r.ofv);
    }
}

/// The full FOCEI fit from the control stream's initial estimates (ρ starts at 0.278,
/// NONMEM ends at 0.617) reaches NONMEM's optimum: the fit moves ρ through the IOV
/// marginal and its EBE search.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn block_sigma_with_iov_fit_reaches_the_nonmem_optimum() {
    let r = run(MODEL, 500, None);
    let d_ofv = r.ofv - NM_FIT_OFV;
    let rho = r.residual_correlations[0].rho;
    eprintln!(
        "ferx fit OFV {:.8} (NONMEM {NM_FIT_OFV:.8}, Δ {d_ofv:.3e}), θ {:?}, ρ {rho:.5}",
        r.ofv, r.theta
    );
    assert!(
        r.ofv.is_finite() && d_ofv.abs() < FIT_OFV_TOL,
        "OFV {}",
        r.ofv
    );
    assert!(
        (rho - nm_rho()).abs() < FIT_RHO_TOL,
        "ρ {rho} vs NONMEM {}",
        nm_rho()
    );
}

/// A fit to convergence on another platform's libm stops at a different last digit; the
/// realised gap is 2.2e-7 (macOS arm64), so `1e-3` is a bound on reaching the optimum,
/// not on the stopping rule.
const FIT_OFV_TOL: f64 = 1e-3;
const FIT_RHO_TOL: f64 = 1e-3;
