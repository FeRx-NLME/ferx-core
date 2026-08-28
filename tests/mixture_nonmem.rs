//! NONMEM 7.5.1 cross-checks for **mixture models** (`[mixture]`): FOCEI (#977)
//! and SAEM (#985).
//!
//! Two latent subpopulations differing only in clearance (`TVCL1` vs `TVCL2`),
//! with a constant (covariate-free) mixing fraction. 1-cpt IV, proportional
//! error, IIV on CL. Ω and Σ are FIXed at the data-generating values in both
//! engines so the objective is smooth and well-identified (the remaining four
//! estimated parameters are the two class clearances, V, and the mixing logit —
//! exactly the mixture-specific quantities under test). Data:
//! `tests/nonmem/mixture_iv.csv` (30 subjects, seed-977 simulation).
//!
//! `impmap_estimating_mixture_recovers_optimum` additionally cross-checks the
//! class-aware mu-referencing θ shift (#996) against
//! `tests/nonmem/mixture_iv_impmap_mu.ctl` — the same model with the
//! class-switched typical value written as `MU_1` *inside* the `MIXNUM` branch
//! and run with `$EST METHOD=IMPMAP`, which is the like-for-like NONMEM anchor
//! for that shift.
//!
//! `mixture_fit_matches_nonmem` is the FOCEI anchor (`mixture_iv.ctl`);
//! `saem_mixture_fit_matches_nonmem_saem` is the SAEM anchor
//! (`mixture_iv_saem.ctl`, `$EST METHOD=SAEM`); and
//! `saem_covariate_mixing_separates_and_recovers_beta` exercises covariate-
//! dependent logit mixing under SAEM end-to-end (#985).
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
use std::io::Write;
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

// NONMEM IMPMAP with the class-switched typical value written as a MU_ reference
// *inside* the MIXNUM branch (`tests/nonmem/mixture_iv_impmap_mu.ctl`, NM 7.6.0,
// `$EST METHOD=IMPMAP NITER=50 ISAMPLE=1500 SEED=20250818`, Ω/Σ FIX). This is the
// anchor for ferx's class-aware mu-ref θ shift under estimating IMP/IMPMAP
// (#996) — the pre-existing anchors use a plain `IF (MIXNUM.EQ.1) TVCL=THETA(1)`
// with no MU_, so they pin the *objective* but not the mu-referenced θ move.
// Values are the `mixture_iv_impmap_mu.ext` `-1000000000` row.
const NM_IMPMAP_MU_TVCL1: f64 = 0.949116;
const NM_IMPMAP_MU_TVCL2: f64 = 2.81929;
const NM_IMPMAP_MU_TVV: f64 = 10.0095;
const NM_IMPMAP_MU_P1: f64 = 0.473425;

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

// ── NONMEM 7.5.1 METHOD=SAEM reference (mixture_iv_saem.ctl / .ext) ──────────
// Same model / data as the FOCEI anchor above, estimated with `$EST
// METHOD=SAEM INTERACTION NBURN=1000 NITER=1000 ISAMPLE=2` followed by a
// `METHOD=IMP EONLY=1` objective-evaluation pass. This is the NONMEM SAEM
// cross-check for the ferx SAEM mixture path (#985). Final SAEM estimates are
// the `mixture_iv_saem.ext` TABLE NO. 1 `-1000000000` row; the OFV is the IMP
// pass's (TABLE NO. 2). SAEM samples the latent class each E-step (exactly the
// ferx scheme), so the two engines' estimates and per-subject MIXEST agree.
const NM_SAEM_TVCL1: f64 = 1.00205;
const NM_SAEM_TVCL2: f64 = 2.73543;
const NM_SAEM_TVV: f64 = 9.99346;
const NM_SAEM_P1: f64 = 0.471245;
const NM_SAEM_OFV_IMP: f64 = 300.8707;

// Per-subject MIXEST from mixture_iv_saem.sdtab (NONMEM SAEM). Differs from the
// FOCEI classification only at ID 5 (a borderline subject SAEM assigns to
// class 2), and at ID 12 both engines pick class 2.
const NM_SAEM_MIXEST: [usize; 30] = [
    1, 1, 1, 1, 2, 1, 1, 1, 1, 1, // IDs 1..=10 (ID5 → class 2 under SAEM)
    1, 2, 1, 1, 1, // IDs 11..=15
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // IDs 16..=25
    2, 2, 2, 2, 2, // IDs 26..=30
];

/// Seeds swept by [`saem_mixture_fit_matches_nonmem_saem`] (#1053 / #1113). The
/// first is the seed the single-realization version of this test shipped with;
/// the rest are the arbitrary seeds of the sweep recorded in #1053, so the
/// distribution measured there describes exactly this set.
const SAEM_ANCHOR_SEEDS: [u64; 10] = [
    20250818, 1, 7, 42, 1234, 99991, 20240101, 555555, 8675309, 20260823,
];

/// SAEM under a mixture (#985): NONMEM `METHOD=SAEM` cross-check. Same
/// two-clearance-class model and data as `mixture_fit_matches_nonmem`, estimated
/// with SAEM instead of FOCEI. Ω/Σ are FIXed, so the estimated quantities are
/// the two class clearances, V, and the mixing logit. ferx SAEM samples the
/// latent class each E-step (drawn from the current posterior `PMIX_i`) and runs
/// η-MCMC within the drawn class; the M-step updates the class-switched thetas
/// (each from its own class members) and the mixing coefficient (from the
/// SA-averaged class frequencies) — the same scheme NONMEM SAEM uses.
///
/// ## Why this anchors a seed-**mean**, not one realization (#1053 / #1113)
///
/// SAEM is a stochastic-approximation estimator: one chain's endpoint is a draw,
/// not a number. Anchoring one ferx realization against one NONMEM realization is
/// therefore a coin flip, and it landed red — the version of this test that
/// shipped with #987 asserted `TVCL2` within 3 % of NONMEM's single SAEM run
/// (2.73543) and was red on every nightly from the day it merged.
///
/// The seed sweep in #1053 settled that this is Monte-Carlo spread, not bias:
/// over ten seeds at this chain length `TVCL2` has sd ≈ 0.13 (CV ≈ 4.7 %), so the
/// old 3 % band was **0.6 sd** wide — narrower than one standard deviation — and
/// `TVCL1` and `p(1)` were just as fragile (3/10 seeds each; **0/10** seeds
/// passed all five assertions together). Lengthening the chain behaves the way a
/// consistent estimator should: sd falls ~28 % from the 600/800 chain and the
/// mean moves *toward* the optimum.
///
/// So the anchor is the **mean over `SAEM_ANCHOR_SEEDS`**, compared against the
/// NONMEM **FOCEI MLE** — the deterministic optimum both engines' SAEM chains are
/// sampling around, and itself a NONMEM number (`mixture_iv.ext`). NONMEM's own
/// single SAEM realization sits 3.75 % *below* that MLE, comfortably inside the
/// observed per-seed range, so it is quoted below as a reference point rather
/// than used as the target. Bands are set from the sweep's sd: each is at least
/// the observed mean offset plus ~3 standard errors of the mean, so the test
/// fails on a shifted estimator rather than on a low draw.
///
/// The chain is NONMEM's (`NBURN=1000 NITER=1000`) rather than the shorter
/// 600/800 the single-seed version used — same reason: it is the lower-variance
/// configuration, and it matches the control stream being anchored against.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (NONMEM SAEM mixture cross-check)"
)]
fn saem_mixture_fit_matches_nonmem_saem() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();

    let k = SAEM_ANCHOR_SEEDS.len() as f64;
    let (mut sum_cl1, mut sum_cl2, mut sum_v, mut sum_p1, mut sum_ofv) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut worst_agree = usize::MAX;
    // Per-subject class-1 vote count across the seeds — the discrete analogue of
    // the seed-mean used for the thetas.
    let mut class1_votes = [0usize; NM_SAEM_MIXEST.len()];

    for seed in SAEM_ANCHOR_SEEDS {
        let mut opts = FitOptions::default();
        opts.method = ferx_core::EstimationMethod::Saem;
        opts.interaction = true;
        opts.saem_n_exploration = 1000;
        opts.saem_n_convergence = 1000;
        opts.saem_seed = Some(seed);

        let res = fit(&model, &pop, &model.default_params, &opts).expect("SAEM mixture fit Ok");
        let th = &res.theta;
        let p1 = 1.0 / (1.0 + (-th[3]).exp());

        // Per-subject MIXEST (recomputed at the SAEM optimum via the mixture
        // marginal). Tallied into `class1_votes` for the modal classification
        // asserted below, and tracked per seed for the loose per-realization floor.
        assert_eq!(res.subjects.len(), NM_SAEM_MIXEST.len());
        let mut agree = 0;
        for (i, sr) in res.subjects.iter().enumerate() {
            let cls = sr.mixest.expect("MIXEST populated");
            if cls == 1 {
                class1_votes[i] += 1;
            }
            if cls == NM_SAEM_MIXEST[i] {
                agree += 1;
            }
        }
        worst_agree = worst_agree.min(agree);

        eprintln!(
            "ferx SAEM mixture seed {seed}: TVCL1={:.4} TVCL2={:.4} TVV={:.4} p1={:.4} \
             OFV={:.4} MIXEST {agree}/{}",
            th[0],
            th[1],
            th[2],
            p1,
            res.ofv,
            NM_SAEM_MIXEST.len()
        );

        sum_cl1 += th[0];
        sum_cl2 += th[1];
        sum_v += th[2];
        sum_p1 += p1;
        sum_ofv += res.ofv;
    }

    let (m_cl1, m_cl2, m_v, m_p1, m_ofv) =
        (sum_cl1 / k, sum_cl2 / k, sum_v / k, sum_p1 / k, sum_ofv / k);
    eprintln!(
        "ferx SAEM mixture seed-mean (K={k}): TVCL1={m_cl1:.4} TVCL2={m_cl2:.4} TVV={m_v:.4} \
         p1={m_p1:.4} OFV={m_ofv:.4} (worst MIXEST {worst_agree}/{})",
        NM_SAEM_MIXEST.len()
    );

    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();

    // ── Seed-mean typical values vs the NONMEM FOCEI MLE ──
    // Bands: observed mean offset + ~3 sem over these seeds (sd/√K), rounded up.
    // NONMEM's single SAEM realization (TVCL1 1.00205, TVCL2 2.73543, TVV 9.99346)
    // is a draw from the same distribution and sits inside every band below.
    assert!(
        rel(m_cl1, NM_TVCL1) < 0.12,
        "SAEM seed-mean TVCL1 {m_cl1} vs NONMEM MLE {NM_TVCL1}"
    );
    assert!(
        rel(m_cl2, NM_TVCL2) < 0.10,
        "SAEM seed-mean TVCL2 {m_cl2} vs NONMEM MLE {NM_TVCL2}"
    );
    assert!(
        rel(m_v, NM_TVV) < 0.03,
        "SAEM seed-mean TVV {m_v} vs NONMEM MLE {NM_TVV}"
    );
    assert!(
        (m_p1 - NM_P1).abs() < 0.08,
        "SAEM seed-mean p(1) {m_p1} vs NONMEM MLE {NM_P1}"
    );

    // Final OFV (K-fold mixture marginal) comparable to NONMEM's IMP objective —
    // the objective is far more stable across seeds than the thetas (sd ≈ 0.8),
    // so this keeps a tight absolute band on the seed mean.
    assert!(
        (m_ofv - NM_SAEM_OFV_IMP).abs() < 4.0,
        "SAEM seed-mean OFV {m_ofv} vs NONMEM IMP {NM_SAEM_OFV_IMP}"
    );

    // Classification (#1053 triage step 4: this check was never reached while the
    // test failed on `TVCL2`). Anchored the same way as the thetas — on the
    // *modal* class over the seeds, the discrete analogue of the seed-mean —
    // because a single chain's per-subject draw is as stochastic as its endpoint:
    // the observed per-seed agreement ranged 27/30 to 30/30 while the modal
    // classification is stable.
    let modal_agree = (0..NM_SAEM_MIXEST.len())
        .filter(|&i| {
            let modal = if 2 * class1_votes[i] >= SAEM_ANCHOR_SEEDS.len() {
                1
            } else {
                2
            };
            modal == NM_SAEM_MIXEST[i]
        })
        .count();
    eprintln!(
        "ferx SAEM mixture modal MIXEST agreement {modal_agree}/{} (worst seed {worst_agree})",
        NM_SAEM_MIXEST.len()
    );
    assert!(
        modal_agree >= NM_SAEM_MIXEST.len() - 1,
        "modal SAEM MIXEST agreement {modal_agree}/{}",
        NM_SAEM_MIXEST.len()
    );
    // Per-realization floor: no single chain may disagree with NONMEM SAEM on more
    // than a tenth of the subjects. Loose on purpose — it guards a collapse of the
    // classification, not the borderline subjects the modal check covers.
    assert!(
        worst_agree >= NM_SAEM_MIXEST.len() - 3,
        "worst-seed SAEM MIXEST agreement {worst_agree}/{}",
        NM_SAEM_MIXEST.len()
    );
}

/// Covariate-dependent mixing under SAEM (#985, full scope). The class logit
/// carries a weight effect `logit(1) = MIXL + BWT·(WT − 75)`, so the mixing
/// M-step is a weighted logistic fit over the sampled classes rather than a bare
/// frequency. Data are generated so the low-clearance class is the low-weight
/// subjects; a correct fit must (a) separate the two clearances and (b) recover
/// a negative `BWT` (heavier ⇒ less likely to be the low-CL class 1). No NONMEM
/// anchor — the constant-mixing NONMEM SAEM cross-check above pins the estimator;
/// this exercises the covariate mixing M-step end-to-end in the full loop.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (covariate-mixing SAEM)"
)]
fn saem_covariate_mixing_separates_and_recovers_beta() {
    // 1-cpt IV, class 1 (low CL) at WT 60, class 2 (high CL) at WT 90.
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT,WT\n");
    let mut sid = 0;
    for &(cl, wt) in [(1.0_f64, 60.0_f64), (3.0, 90.0)].iter() {
        for _ in 0..15 {
            sid += 1;
            csv.push_str(&format!("{sid},0,0,100,1,1,{wt}\n"));
            for (ti, t) in [0.5_f64, 1.0, 2.0, 4.0, 8.0].iter().enumerate() {
                let c = (100.0 / 10.0) * (-(cl / 10.0) * t).exp();
                let dv = c * (1.0 + 0.03 * (((sid + ti) as f64) * 1.3).sin());
                csv.push_str(&format!("{sid},{t},{dv:.5},0,0,1,{wt}\n"));
            }
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    let pop = read_nonmem_csv(f.path(), Some(&["WT"]), None).unwrap();

    const COV_MODEL: &str = r"
[parameters]
  theta TVCL1(1.2, 0.01, 100.0)
  theta TVCL2(2.5, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  theta MIXL(0.0, -10.0, 10.0)
  theta BWT(0.0, -5.0, 5.0)
  omega ETA_CL ~ 0.04
  sigma EPS ~ 0.01

[mixture]
  nsub = 2
  logit(1) = MIXL + BWT*(WT - 75)

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";
    let model = parse_model_string(COV_MODEL).unwrap();
    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Saem;
    opts.interaction = true;
    opts.saem_n_exploration = 500;
    opts.saem_n_convergence = 500;
    opts.saem_seed = Some(20250818);

    let res = fit(&model, &pop, &model.default_params, &opts).expect("covariate SAEM fit Ok");
    let th = &res.theta;
    eprintln!(
        "cov SAEM: TVCL1={:.3} TVCL2={:.3} BWT={:.3}",
        th[0], th[1], th[4]
    );

    // Classes separated (low vs high clearance).
    assert!(th[0] < 1.6, "TVCL1 (low class) {} should be near 1", th[0]);
    assert!(th[1] > 2.4, "TVCL2 (high class) {} should be near 3", th[1]);
    // Heavier subjects are less likely to be class 1 ⇒ BWT < 0.
    assert!(th[4] < 0.0, "BWT {} should be negative", th[4]);

    // Weight-driven classification: low-WT subjects → class 1, high-WT → class 2.
    for (i, sr) in res.subjects.iter().enumerate() {
        let expected = if i < 15 { 1 } else { 2 };
        assert_eq!(sr.mixest.expect("MIXEST"), expected, "subject {i} MIXEST");
    }
}

/// Bayesian (full-MCMC) estimation under a mixture (#985). Same two-clearance
/// model / data as the FOCEI and SAEM anchors. ferx Rao-Blackwellises the latent
/// class (marginalises it out of every Gibbs block), so the sampler targets the
/// K-class marginal and the posterior mean is a point estimate of the mixture
/// MLE.
///
/// **Anchor:** the shared maximum-likelihood optimum (the FOCEI `mixture_iv.ctl`
/// values `NM_*`). The population priors are weakly-informative N(u0, 10²) on the
/// unconstrained scale, centred on this model's initial estimates (the declared
/// `[parameters]` bounds are *not* enforced as uniform priors inside the (θ, σ)
/// block), and Ω/Σ are FIXed — diffuse enough that the posterior mean concentrates
/// at the likelihood optimum, so recovering it *is* the cross-engine check. A direct NONMEM `METHOD=BAYES` run
/// is **not** used: NONMEM's BAYES sampler aborts in burn-in on this model
/// (`$MIX` + FIXed `$OMEGA`/`$SIGMA` — it insists on Gibbs-sampling Ω, which is
/// FIXed), a known NONMEM mixture/BAYES fragility (cf. the covariance-step
/// failure on IOV mixtures). The marginal OFV is additionally checked against the
/// FOCEI optimum, and the same estimator is anchored directly to NONMEM SAEM
/// above, so the mixture-marginal machinery this shares is cross-validated.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (Bayes mixture)"
)]
fn bayes_mixture_recovers_optimum() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();
    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Bayes;
    opts.bayes_warmup = 500;
    opts.bayes_iters = 500;
    opts.bayes_chains = 2;
    opts.bayes_seed = Some(20250818);
    let res = fit(&model, &pop, &model.default_params, &opts).expect("Bayes mixture fit Ok");
    let th = &res.theta;
    let p1 = 1.0 / (1.0 + (-th[3]).exp());
    eprintln!(
        "Bayes mixture: TVCL1={:.4} TVCL2={:.4} TVV={:.4} p1={:.4} OFV={:.4}",
        th[0], th[1], th[2], p1, res.ofv
    );
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();

    // Posterior mean recovers the mixture MLE (the two class clearances are the
    // discriminating quantities — the sampler must separate them).
    assert!(
        rel(th[0], NM_TVCL1) < 0.10,
        "Bayes TVCL1 {} vs {}",
        th[0],
        NM_TVCL1
    );
    assert!(
        rel(th[1], NM_TVCL2) < 0.10,
        "Bayes TVCL2 {} vs {}",
        th[1],
        NM_TVCL2
    );
    assert!(
        rel(th[2], NM_TVV) < 0.05,
        "Bayes TVV {} vs {}",
        th[2],
        NM_TVV
    );
    assert!((p1 - NM_P1).abs() < 0.08, "Bayes p(1) {} vs {}", p1, NM_P1);

    // Marginal OFV (K-fold log-sum-exp) comparable to the FOCEI optimum.
    assert!(
        (res.ofv - NM_OFV_NO_CONST).abs() < 3.0,
        "Bayes OFV {} vs {}",
        res.ofv,
        NM_OFV_NO_CONST
    );

    // Per-subject MIXEST populated (recomputed at the posterior mean).
    assert!(
        res.subjects.iter().all(|s| s.mixest.is_some()),
        "MIXEST populated"
    );
}

// NONMEM IMP objective (mixture_iv_saem.ext TABLE NO. 2, the `METHOD=IMP
// EONLY=1` pass that follows SAEM): the class-marginal −2 log L evaluated by
// importance sampling at the SAEM optimum.
const NM_IMP_MARGINAL: f64 = 300.8707;

/// IMP objective evaluation under a mixture (#985): `method = [saem, imp]` with
/// `imp_eval_only`. The SAEM stage estimates the parameters; the IMP stage
/// evaluates the class-marginal likelihood `−2 Σ log Σ_k p_ik L_ik` by importance
/// sampling per class (per-class MAP + IS, combined via log-sum-exp). Anchored to
/// the NONMEM `METHOD=IMP EONLY=1` objective from `mixture_iv_saem.ctl` (300.87).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (NONMEM IMP mixture marginal)"
)]
fn imp_mixture_marginal_matches_nonmem() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();
    let mut opts = FitOptions::default();
    opts.methods = vec![
        ferx_core::EstimationMethod::Saem,
        ferx_core::EstimationMethod::Imp,
    ];
    opts.interaction = true;
    opts.saem_n_exploration = 600;
    opts.saem_n_convergence = 800;
    opts.saem_seed = Some(20250818);
    opts.imp_eval_only = true;
    opts.imp_samples = 3000;
    opts.imp_seed = Some(20250818);

    let res = fit(&model, &pop, &model.default_params, &opts).expect("saem+imp mixture fit Ok");
    let is = res
        .importance_sampling
        .as_ref()
        .expect("IMP objective evaluation populated on the chain's last stage");
    eprintln!(
        "IMP mixture marginal: -2logL = {:.4} ± {:.4} vs NONMEM {:.4}",
        is.minus2_log_likelihood, is.mc_standard_error, NM_IMP_MARGINAL
    );
    // The IS marginal is a Monte-Carlo estimate at the SAEM optimum; a ~1-unit
    // band covers the MC error plus the small SAEM-vs-NONMEM optimum offset.
    assert!(
        (is.minus2_log_likelihood - NM_IMP_MARGINAL).abs() < 2.0,
        "IMP marginal {} vs NONMEM {}",
        is.minus2_log_likelihood,
        NM_IMP_MARGINAL
    );
}

/// Estimating IMPMAP under a mixture (#985): class-partitioned MCEM recovers the MLE.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests (IMPMAP mixture estimation)"
)]
fn impmap_estimating_mixture_recovers_optimum() {
    // Estimating IMPMAP (class-partitioned MCEM): the per-class IS E-step +
    // responsibility-weighted M-steps must recover the mixture MLE. Ω/Σ FIXed, so
    // the estimated quantities are the two class clearances, V, and the mixing
    // logit. Anchored to the shared optimum (NONMEM FOCEI `mixture_iv.ctl`).
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(MODEL).unwrap();
    let mut opts = FitOptions::default();
    opts.method = ferx_core::EstimationMethod::Impmap;
    opts.interaction = true;
    opts.impmap_iterations = 50;
    opts.impmap_samples = 1500;
    opts.impmap_seed = Some(20250818);
    opts.run_covariance_step = false;

    let res = fit(&model, &pop, &model.default_params, &opts).expect("IMPMAP mixture fit Ok");
    let th = &res.theta;
    let p1 = 1.0 / (1.0 + (-th[3]).exp());
    eprintln!(
        "IMPMAP mixture: TVCL1={:.4} TVCL2={:.4} TVV={:.4} p1={:.4} OFV={:.4}",
        th[0], th[1], th[2], p1, res.ofv
    );
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    // ── vs NONMEM IMPMAP with MU_ inside the MIXNUM branch (#996) ──
    // The like-for-like anchor: both engines now apply a mu-referenced θ move
    // per class, so the class clearances agree to well under 1 %.
    assert!(
        rel(th[0], NM_IMPMAP_MU_TVCL1) < 0.02,
        "IMPMAP TVCL1 {} vs NONMEM MU-IMPMAP {}",
        th[0],
        NM_IMPMAP_MU_TVCL1
    );
    assert!(
        rel(th[1], NM_IMPMAP_MU_TVCL2) < 0.02,
        "IMPMAP TVCL2 {} vs NONMEM MU-IMPMAP {}",
        th[1],
        NM_IMPMAP_MU_TVCL2
    );
    assert!(
        rel(th[2], NM_IMPMAP_MU_TVV) < 0.02,
        "IMPMAP TVV {} vs NONMEM MU-IMPMAP {}",
        th[2],
        NM_IMPMAP_MU_TVV
    );
    assert!(
        (p1 - NM_IMPMAP_MU_P1).abs() < 0.02,
        "IMPMAP p(1) {} vs NONMEM MU-IMPMAP {}",
        p1,
        NM_IMPMAP_MU_P1
    );
    // ── vs the shared FOCEI optimum ──
    // The EM estimator settles slightly below FOCEI's TVCL1/TVCL2 (NONMEM's
    // mu-referenced IMPMAP does the same), so this is the looser of the two
    // checks; before #996 the weighted M-step alone left TVCL2 at ≈ 2.60.
    assert!(
        rel(th[0], NM_TVCL1) < 0.05,
        "IMPMAP TVCL1 {} vs FOCEI {}",
        th[0],
        NM_TVCL1
    );
    assert!(
        rel(th[1], NM_TVCL2) < 0.03,
        "IMPMAP TVCL2 {} vs FOCEI {}",
        th[1],
        NM_TVCL2
    );
    assert!(
        rel(th[2], NM_TVV) < 0.05,
        "IMPMAP TVV {} vs {}",
        th[2],
        NM_TVV
    );
    assert!((p1 - NM_P1).abs() < 0.05, "IMPMAP p(1) {} vs {}", p1, NM_P1);
    assert!(
        res.subjects.iter().all(|s| s.mixest.is_some()),
        "MIXEST populated"
    );

    // The class-aware shift is what buys this: with `mu_referencing = false`
    // every θ falls back to the importance-weighted M-step alone (the pre-#996
    // path), which leaves TVCL2 materially short of the optimum.
    let mut opts_off = opts.clone();
    opts_off.mu_referencing = false;
    let res_off =
        fit(&model, &pop, &model.default_params, &opts_off).expect("IMPMAP mixture fit Ok");
    let th_off = &res_off.theta;
    eprintln!(
        "IMPMAP mixture (mu_referencing off): TVCL1={:.4} TVCL2={:.4}",
        th_off[0], th_off[1]
    );
    assert!(
        rel(th[1], NM_IMPMAP_MU_TVCL2) < rel(th_off[1], NM_IMPMAP_MU_TVCL2),
        "class-aware TVCL2 {} must beat the numerical M-step's {} (target {})",
        th[1],
        th_off[1],
        NM_IMPMAP_MU_TVCL2
    );
}
