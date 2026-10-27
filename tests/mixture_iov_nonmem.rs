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

// ── #1932: evaluation-only (`maxiter = 0`) postfit at NONMEM's own finals ─────
//
// `tests/nonmem/mixture_iv_iov_eval.ctl` is the control stream above with `$THETA`
// set to its `.ext` final row as printed (6 significant figures), `MAXEVAL=0 POSTHOC`,
// and IPRED / IWRES / CWRES / ETA1-3 tabled. NONMEM's table pass runs at
// MIXNUM = MIXEST, so its ETAs are the most-probable class's, which is what ferx
// reports (`mixest_etas` / `mixest_kappas`). The `.phi` is *not* used: for a mixture
// its ETAs differ from the table's in the fourth significant figure (subject 1:
// −1.86198e-2 vs −1.83680e-2), apparently a mix over the two classes.
// NONMEM 7.5.1, ADVAN1 TRANS2, `mixture_iv_iov_eval.ext` −1000000000 row.
const NM_EVAL_OFV: f64 = 470.814_275_316_537_13;

/// One `$TABLE` row of `mixture_iv_iov_eval.tab`.
struct NmRow {
    id: usize,
    evid: u32,
    mixe: usize,
    ipred: f64,
    iwres: f64,
    cwres: f64,
    eta: [f64; 3],
}

fn read_nm_eval_tab() -> Vec<NmRow> {
    let text = std::fs::read_to_string("tests/nonmem/mixture_iv_iov_eval.tab").unwrap();
    let mut lines = text.lines().skip(1); // "TABLE NO.  1"
    let header: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let (id, evid, mixe) = (col("ID"), col("EVID"), col("MIXE"));
    let (ipred, iwres, cwres) = (col("IPRED"), col("IWRES"), col("CWRES"));
    let etas = [col("ETA1"), col("ETA2"), col("ETA3")];
    lines
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            NmRow {
                id: v[id] as usize,
                evid: v[evid] as u32,
                mixe: v[mixe] as usize,
                ipred: v[ipred],
                iwres: v[iwres],
                cwres: v[cwres],
                eta: etas.map(|c| v[c]),
            }
        })
        .collect()
}

/// #1932: an IOV mixture evaluated with `maxiter = 0` reports the MIXEST class's κ̂ and
/// computes IPRED / IWRES with it, as NONMEM's `MAXEVAL=0 POSTHOC` table does. Before the
/// fix ferx left κ̂ empty on this path, so every per-row diagnostic was the κ = 0
/// prediction while the OFV (which integrates κ itself) still agreed.
///
/// The structural checks (observation count, MIXEST, finiteness) run inside the loop. The
/// gap asserts run only after the folded gaps are printed, so a run without the fix
/// reports how far it lands rather than stopping at the κ̂ count. A subject with no κ̂ is
/// compared at κ = 0, the value its postfit then used.
///
/// CWRES linearizes over every ETA, the occasion κs included, as NONMEM's does (#1948): the
/// MIXEST class's `[η̂, κ̂₁, κ̂₂]`, its stacked Jacobian and `blkdiag(Ω, Ω_IOV, Ω_IOV)`.
/// Before #1948 ferx used η alone and this gap was 4.135e-1 with #1932 fixed. Recomputed
/// outside both engines from NONMEM's own tabled ETAs over all 30 subjects, the
/// [η, κ₁, κ₂] recipe reproduces NONMEM's column to 2.1e-10, so what is left is the two
/// engines' η̂ and κ̂ carried through `f0`.
#[test]
fn iov_mixture_eval_only_postfit_matches_nonmem_posthoc() {
    let pop: ferx_core::Population = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv_iov.csv"),
        Some(&["WT"]),
        Some("OCC"),
    )
    .unwrap();
    let src = MODEL
        .replace("TVCL1(1.0,", "TVCL1(1.05700,")
        .replace("TVCL2(3.0,", "TVCL2(3.32950,")
        .replace("TVV(10.0,", "TVV(10.2591,")
        .replace("P1(0.5,", "P1(0.525988,");
    for start in ["(1.05700,", "(3.32950,", "(10.2591,", "(0.525988,"] {
        assert_eq!(src.matches(start).count(), 1, "start {start} substituted");
    }
    let model = parse_model_string(&src).unwrap();
    let mut opts = FitOptions::default();
    opts.interaction = true; // METHOD=1 INTER
    opts.outer_maxiter = 0; // MAXEVAL=0
    let res = fit(&model, &pop, &model.default_params, &opts).expect("eval-only fit Ok");

    let nm = read_nm_eval_tab();
    assert_eq!(res.subjects.len(), 30);
    let ofv_gap = (res.ofv - NM_EVAL_OFV).abs();
    let (mut ipred_gap, mut iwres_gap, mut cwres_gap, mut eta_gap) = (0f64, 0f64, 0f64, 0f64);
    let mut no_kappa = 0usize;
    for (i, sr) in res.subjects.iter().enumerate() {
        // Keyed on the subject's own ID, not its position: every subject has 8
        // observations, so the count check below cannot see a reorder.
        let id: usize = sr.id.parse().expect("numeric subject ID");
        let rows: Vec<&NmRow> = nm.iter().filter(|r| r.id == id).collect();
        assert!(!rows.is_empty(), "subject {id}: not in the NONMEM table");
        let obs: Vec<&&NmRow> = rows.iter().filter(|r| r.evid == 0).collect();
        assert_eq!(sr.ipred.len(), obs.len(), "subject {id}: observation count");
        assert_eq!(sr.mixest, Some(rows[0].mixe), "subject {id}: MIXEST");
        let k = res.ebe_kappas.get(i).map_or(&[][..], |k| k.as_slice());
        if k.len() != 2 {
            no_kappa += 1;
        }
        let kappa = |occ: usize| k.get(occ).map_or(0.0, |v| v[0]);
        let ferx_eta = [sr.eta[0], kappa(0), kappa(1)];
        for (j, (&f, &n)) in ferx_eta.iter().zip(&rows[0].eta).enumerate() {
            assert!(f.is_finite(), "subject {id}: ETA{} = {f}", j + 1);
            eta_gap = eta_gap.max((f - n).abs());
        }
        for (o, r) in obs.iter().enumerate() {
            let (fi, fw, fc) = (sr.ipred[o], sr.iwres[o], sr.cwres[o]);
            assert!(
                fi.is_finite() && fw.is_finite() && fc.is_finite(),
                "subject {id} obs {o}: IPRED {fi} IWRES {fw} CWRES {fc}"
            );
            ipred_gap = ipred_gap.max((fi - r.ipred).abs());
            iwres_gap = iwres_gap.max((fw - r.iwres).abs());
            cwres_gap = cwres_gap.max((fc - r.cwres).abs());
        }
    }
    eprintln!(
        "#1932 eval-only vs NONMEM: subjects without κ̂ {no_kappa}/30  |dOFV| {ofv_gap:.3e}  \
         |dIPRED| {ipred_gap:.3e}  |dIWRES| {iwres_gap:.3e}  |dETA,κ| {eta_gap:.3e}  \
         |dCWRES| {cwres_gap:.3e}"
    );
    assert_eq!(no_kappa, 0, "every subject carries its two occasion κ̂s");
    assert!(
        ofv_gap < EVAL_OFV_TOL,
        "OFV {} vs NONMEM {NM_EVAL_OFV}",
        res.ofv
    );
    assert!(ipred_gap < EVAL_IPRED_TOL, "IPRED gap {ipred_gap:.3e}");
    assert!(iwres_gap < EVAL_IWRES_TOL, "IWRES gap {iwres_gap:.3e}");
    assert!(eta_gap < EVAL_ETA_TOL, "ETA / κ̂ gap {eta_gap:.3e}");
    assert!(
        cwres_gap < EVAL_CWRES_TOL,
        "CWRES gap {cwres_gap:.3e}: not linearized over [eta, kappa_1, kappa_2] (#1948)"
    );
}

// Measured on Linux x86_64 (glibc 2.28) at 45fd69b5 + this change; macOS arm64 gives the same
// digits except dOFV 2.133e-7. Realised worst gaps with the fix / without it (κ̂ empty on the
// eval-only path):
//   OFV    2.132e-7 / 2.132e-7  (the population OFV was right before #1932; this pins the point)
//   IPRED  6.771e-7 / 5.864e-1
//   IWRES  5.349e-7 / 1.294
//   ETA,κ  1.772e-7 / 1.688e-1
// Each bound sits 4.7–9.3× above its realised gap and at least 1e5× below the unfixed one.
// CWRES is measured on macOS arm64 at 48100d27 + #1948, with #1948's fix / without it, #1932's
// fix in both:
//   CWRES  1.140e-7 / 4.135e-1
// Its bound sits 8.8× above the realised gap and 4e5× below the η-only one.
const EVAL_OFV_TOL: f64 = 1e-6;
const EVAL_IPRED_TOL: f64 = 5e-6;
const EVAL_IWRES_TOL: f64 = 5e-6;
const EVAL_ETA_TOL: f64 = 1e-6;
const EVAL_CWRES_TOL: f64 = 1e-6;
