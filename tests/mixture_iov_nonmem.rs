//! NONMEM 7.5.1 FOCEI cross-check for **inter-occasion variability under a
//! mixture** (`[mixture]` + `kappa`, #985).
//!
//! Two latent subpopulations differing in clearance (`TVCL1` vs `TVCL2`) with a
//! constant mixing fraction `P(1)`, **plus** a per-occasion κ on CL: every
//! subject has two dosing occasions (`OCC` 1/2), and CL carries an IOV term drawn
//! from a shared IOV Ω (NONMEM `$OMEGA BLOCK(1) … SAME`; ferx `kappa KAPPA_CL`).
//! 1-cpt IV, proportional error. IIV Ω, IOV Ω, and Σ are FIXed at their
//! data-generating values in both engines (mirroring the `mixture_iv_cov` anchor)
//! so the four estimated quantities are exactly the mixture-specific typical
//! values (`TVCL1`, `TVCL2`, `V`, `P(1)`) while the IOV κ still enters the FOCE-I
//! marginal. Data: `tests/nonmem/mixture_iv_iov.csv` (30 subjects × 2 occasions,
//! deterministic #985 simulation).
//!
//! ## NONMEM reference (`tests/nonmem/mixture_iv_iov.ctl`)
//!
//! `$SUBROUTINES ADVAN1 TRANS2`; `$MIX NSPOP=2`, `P(1)=THETA(4)`;
//! `IF (MIXNUM.EQ.1) TVCL=THETA(1)` / `.EQ.2 → THETA(2)`; per-occasion
//! `KACL = OCC1·ETA(2) + OCC2·ETA(3)`, `CL = TVCL·EXP(ETA(1)+KACL)`;
//! `$OMEGA 0.05 FIX`, `$OMEGA BLOCK(1) 0.02 FIX`, `$OMEGA BLOCK(1) SAME`,
//! `$SIGMA 0.04 FIX`; `METHOD=1 INTER`. Final estimates are the `.ext`
//! `-1000000000` row; OFV-without-constant `470.814`.
//!
//! **Standard errors are not cross-checked here.** NONMEM's covariance step for
//! this IOV mixture terminates with rounding errors and substitutes the R matrix
//! (`R MATRIX SUBSTITUTED: YES`, no printable SE block / `.cov`) — a known NONMEM
//! fragility for mixture + IOV covariance, not a ferx issue. ferx's mixture SE
//! machinery is anchored to NONMEM `$COVARIANCE MATRIX=R` separately in
//! `mixture_nonmem.rs` (non-IOV mixture), and the IOV covariance packing/step is
//! exercised by the `iov_mixture_covariance_step_runs` unit test. This anchor
//! therefore compares the **objective, point estimates, and per-subject MIXEST**
//! — the quantities that establish IOV + mixture coexist correctly in the
//! objective (the novel behaviour of #985).

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

// NONMEM 7.5.1 FOCEI MLE (mixture_iv_iov.ext `-1000000000` row; OFV w/o constant).
const NM_TVCL1: f64 = 1.05700;
const NM_TVCL2: f64 = 3.32950;
const NM_TVV: f64 = 10.2591;
const NM_P1: f64 = 0.525988; // P(1), mixing fraction of class 1
const NM_OFV_NO_CONST: f64 = 470.8143;

// Per-subject MIXEST (most-probable class, 1-based) from mixture_iv_iov.sdtab,
// first row per ID (IDs 1..=30). IDs 1..=15 are true class 1, 16..=30 class 2;
// subject 18's posterior favours class 1 (one border case), the rest agree.
const NM_MIXEST: [usize; 30] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // IDs 1..=15 (true class 1)
    2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, // IDs 16..=30 (true class 2; ID18 → class 1)
];

// ferx equivalent: `p(1) = P1` (direct-probability mixing, matching NONMEM
// P(1)=THETA(4)); MIXNUM-branched CL with a per-occasion κ (`+ KAPPA_CL`); IIV,
// IOV, and Σ FIXed at the data-generating values.
const MODEL: &str = r"
[parameters]
  theta TVCL1(1.0, 0.01, 100.0)
  theta TVCL2(3.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta P1(0.5, 0.001, 0.999)
  omega ETA_CL ~ 0.05 FIX
  kappa KAPPA_CL ~ 0.02 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL + KAPPA_CL) else TVCL2 * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (NONMEM IOV+mixture cross-check)"
)]
fn iov_mixture_fit_matches_nonmem() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv_iov.csv"),
        Some(&["WT"]),
        Some("OCC"),
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();
    assert_eq!(model.n_kappa, 1, "one IOV kappa on CL");

    let mut opts = FitOptions::default();
    opts.interaction = true; // FOCEI, matching NONMEM METHOD=1 INTER

    let res = fit(&model, &pop, &model.default_params, &opts).expect("IOV mixture fit Ok");

    // ── Population OFV (without the Nobs·ln(2π) constant, NONMEM convention) ──
    assert!(
        (res.ofv - NM_OFV_NO_CONST).abs() < 0.3,
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
    // ferx `p(1) = P1` is the probability directly, same as NONMEM P(1)=THETA(4).
    assert!(
        (th[3] - NM_P1).abs() < 0.05,
        "p(1) {} vs NONMEM {}",
        th[3],
        NM_P1
    );

    // ── Per-subject MIXEST classification agreement ─────────────────────────
    assert_eq!(res.subjects.len(), NM_MIXEST.len());
    let agree = res
        .subjects
        .iter()
        .enumerate()
        .filter(|(i, sr)| sr.mixest.expect("MIXEST populated") == NM_MIXEST[*i])
        .count();
    // The two engines compute the same posterior at the same optimum, so every
    // subject's most-probable class must match.
    assert_eq!(
        agree,
        NM_MIXEST.len(),
        "MIXEST agreement {}/{}",
        agree,
        NM_MIXEST.len()
    );
}
