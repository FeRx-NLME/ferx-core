//! NONMEM 7.5.1 FOCEI cross-check for **mixture models** (`[mixture]`, #977).
//!
//! Two latent subpopulations differing only in clearance (`TVCL1` vs `TVCL2`),
//! with a constant (covariate-free) mixing fraction. 1-cpt IV, proportional
//! error, IIV on CL. Ω and Σ are FIXed at the data-generating values in both
//! engines so the objective is smooth and well-identified (the remaining four
//! estimated parameters are the two class clearances, V, and the mixing logit —
//! exactly the mixture-specific quantities under test). Data:
//! `tests/nonmem/mixture_iv.csv` (30 subjects, seed-977 simulation).
//!
//! ## NONMEM reference (`tests/nonmem/mixture_iv.ctl`)
//!
//! `$SUBROUTINES ADVAN1 TRANS2`; `$MIX NSPOP=2`, `P(1)=THETA(4)`;
//! `IF (MIXNUM.EQ.1) TVCL=THETA(1)` / `IF (MIXNUM.EQ.2) TVCL=THETA(2)`;
//! `$OMEGA 0.09 FIX`, `$SIGMA 0.04 FIX`; `METHOD=1 INTER`.
//! **MINIMIZATION SUCCESSFUL** (3.4 sig. digits), OFV-without-constant
//! `301.582`. Final estimates below are the `.ext` `-1000000000` row.
//! Per-subject `MIXEST` (most-probable class) is in `tests/nonmem/mixture_iv.sdtab`
//! and transcribed into `NM_MIXEST` below.
//!
//! The mixing fraction is parameterised differently in the two engines
//! (NONMEM: `P(1)=THETA(4)`; ferx: `logit(1)=MIXL`, `p(1)=σ(MIXL)`) but denotes
//! the same quantity — compared as the resolved `p(1)` fraction.
//!
//! ## Standard errors (#983 Phase 6)
//!
//! The test also cross-checks the mixture covariance step against
//! `tests/nonmem/mixture_iv_cov.ctl` (`$COVARIANCE MATRIX=R` — the pure
//! Hessian-inverse `R^-1`, the same estimator ferx computes). All four SEs agree
//! to < 1 %. The mixing-fraction SE is compared after delta-method mapping ferx's
//! logit-scale `SE(MIXL)` to the probability scale via `SE(p) = p(1−p)·SE(MIXL)`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 FOCEI MLE (mixture_iv.ext final iteration; OFV without constant).
const NM_TVCL1: f64 = 0.980833;
const NM_TVCL2: f64 = 2.84198;
const NM_TVV: f64 = 9.97859;
const NM_P1: f64 = 0.471970; // P(1), mixing fraction of class 1
const NM_OFV_NO_CONST: f64 = 301.5820;

// NONMEM $COVARIANCE MATRIX=R SEs (mixture_iv_cov.ext, -1000000001 row; natural
// theta scale). MATRIX=R is the pure R^-1 Hessian-inverse — the same estimator
// ferx computes — so these are an apples-to-apples cross-check (#983 Phase 6).
const NM_SE_TVCL1: f64 = 0.105642;
const NM_SE_TVCL2: f64 = 0.254347;
const NM_SE_TVV: f64 = 0.186107;
const NM_SE_P1: f64 = 0.102569; // SE of P(1) on the probability scale

// Per-subject MIXEST (most-probable class, 1-based) from mixture_iv.sdtab,
// indexed by subject order (ID 1..=30).
const NM_MIXEST: [usize; 30] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // IDs 1..=10  (true class 1)
    1, 2, 1, 1, 1, // IDs 11..=15 (true class 1; ID12's draw favours class 2)
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // IDs 16..=25 (true class 2)
    2, 2, 2, 2, 2, // IDs 26..=30 (true class 2)
];

const MODEL: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (NONMEM mixture cross-check)"
)]
fn mixture_fit_matches_nonmem() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();

    let mut opts = FitOptions::default();
    opts.interaction = true; // FOCEI, matching NONMEM METHOD=1 INTER

    let res = fit(&model, &pop, &model.default_params, &opts).expect("mixture fit Ok");

    // ── Population OFV ──────────────────────────────────────────────────────
    // ferx reports the FOCEI objective without the Nobs·ln(2π) constant, the
    // same convention as NONMEM's "OBJECTIVE FUNCTION VALUE WITHOUT CONSTANT".
    // Observed: ferx 301.600 vs NONMEM 301.582 — a ~0.02-unit match.
    assert!(
        (res.ofv - NM_OFV_NO_CONST).abs() < 0.2,
        "OFV {} vs NONMEM {}",
        res.ofv,
        NM_OFV_NO_CONST
    );

    // ── Estimated typical values ────────────────────────────────────────────
    let th = &res.theta;
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(
        rel(th[0], NM_TVCL1) < 0.05,
        "TVCL1 {} vs {}",
        th[0],
        NM_TVCL1
    );
    assert!(
        rel(th[1], NM_TVCL2) < 0.05,
        "TVCL2 {} vs {}",
        th[1],
        NM_TVCL2
    );
    assert!(rel(th[2], NM_TVV) < 0.05, "TVV {} vs {}", th[2], NM_TVV);

    // Mixing fraction: ferx logit → p(1) = σ(MIXL), compared to NONMEM P(1).
    let p1 = 1.0 / (1.0 + (-th[3]).exp());
    assert!((p1 - NM_P1).abs() < 0.05, "p(1) {} vs NONMEM {}", p1, NM_P1);

    // ── Standard errors vs NONMEM $COVARIANCE (#983 Phase 6) ────────────────────
    // The mixture covariance step now runs (it was skipped before): its FD Hessian
    // is built on the K-fold mixture OFV, and SE = sqrt(diag(R^-1)). Compared to
    // `tests/nonmem/mixture_iv_cov.ctl`, which uses `$COVARIANCE MATRIX=R` — the
    // pure Hessian-inverse estimator ferx also computes (NONMEM's default sandwich
    // R^-1 S R^-1 would differ by the usual 10-25% between the two estimators).
    // Observed agreement is <1% on all four (`.ext` `-1000000001` row).
    let se = res
        .se_theta
        .as_ref()
        .expect("mixture fit now reports theta SEs (#983)");
    assert!(
        res.covariance_matrix.is_some(),
        "mixture covariance matrix present"
    );
    let rel_se = |a: f64, b: f64| (a - b).abs() / b.abs();
    assert!(
        rel_se(se[0], NM_SE_TVCL1) < 0.03,
        "SE TVCL1 {} vs {}",
        se[0],
        NM_SE_TVCL1
    );
    assert!(
        rel_se(se[1], NM_SE_TVCL2) < 0.03,
        "SE TVCL2 {} vs {}",
        se[1],
        NM_SE_TVCL2
    );
    assert!(
        rel_se(se[2], NM_SE_TVV) < 0.03,
        "SE TVV {} vs {}",
        se[2],
        NM_SE_TVV
    );
    // Mixing fraction SE: ferx MIXL is on the logit scale; delta-method to p(1),
    // p = σ(MIXL) ⇒ dp/dMIXL = p(1−p), so SE(p) = p(1−p)·SE(MIXL). NONMEM
    // parameterizes P(1)=THETA(4) directly, so its SE is already on the p-scale.
    let se_p1 = p1 * (1.0 - p1) * se[3];
    assert!(
        rel_se(se_p1, NM_SE_P1) < 0.03,
        "SE p(1) {} vs {}",
        se_p1,
        NM_SE_P1
    );

    // ── Per-subject MIXEST classification agreement ─────────────────────────
    assert_eq!(res.subjects.len(), NM_MIXEST.len());
    let mut agree = 0;
    for (i, sr) in res.subjects.iter().enumerate() {
        let m = sr.mixest.expect("MIXEST populated");
        if m == NM_MIXEST[i] {
            agree += 1;
        }
    }
    // Classification must agree on every subject (the two engines compute the
    // same posterior at the same optimum).
    assert_eq!(
        agree,
        NM_MIXEST.len(),
        "MIXEST agreement {}/{}",
        agree,
        NM_MIXEST.len()
    );
}
