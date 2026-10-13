//! NONMEM anchor: sdtab CWRES under inter-occasion variability (#1948).
//!
//! `tests/nonmem/iov_cwres_{focei,foce}.ctl` are the #985 `mixture_iv_iov` data with the
//! `$MIX` removed: 1-cpt IV, IIV on CL plus a per-occasion κ on CL (`$OMEGA BLOCK(1) …
//! SAME`), proportional error, every variance FIXed, `MAXEVAL=0 POSTHOC`, run in NONMEM
//! 7.5.1 with and without `INTER`. The committed `.ext/.lst/.phi/.tab` are those runs'
//! outputs (OBJ 554.92498260642810 FOCEI, 585.60024484944893 FOCE). ferx evaluates the same
//! fixed point with `outer_maxiter = 0`, so the comparison is not an optimizer race.
//!
//! The object under test is which random effects CWRES linearizes over. NONMEM treats an
//! occasion κ as one more `ETA` (`ETA2`/`ETA3` here), so its CWRES is Hooker's over
//! `b = [η, κ₁, κ₂]` with `W = blkdiag(Ω, Ω_IOV, Ω_IOV)`. ferx used η alone, which leaves
//! `H_κ·κ̂` in the residual and drops `H_κ·Ω_IOV·H_κᵀ` from `R̃`, while its objective, IPRED
//! and EBEs already matched NONMEM to ~1e-7. Measured on macOS arm64 before the fix, worst
//! |ferx − NONMEM| CWRES: FOCEI 0.4259, FOCE 0.2524.
//!
//! Both arms are kept because they put `R` at different points — IPRED under FOCEI,
//! `f(η = 0, κ = 0)` under FOCE — and the FOCE arm is the one that sees whether the post-fit
//! pass takes the κ = 0 population prediction the marginal uses.
//!
//! Engine: the closed-form 1-cpt IV through `pk::predict_iov`; the κ columns of `H` are the
//! marginal's own central differences (`iov_linearization`), shared by the residual pass.
//!
//! Each half of the fix is visible here on its own (macOS arm64, worst |ΔCWRES| FOCEI /
//! FOCE): dropping `H_κ·κ̂` from `f0` lands at 0.838 / 0.470, dropping `H_κ·Ω_IOV·H_κᵀ` from
//! `R̃` at 0.792 / 0.289. What it cannot say is *which* half broke; the Tier-1 tests in
//! `api/tests/iov_cwres_tests.rs` do that, one mutation each.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult};
use std::path::Path;

const MODEL: &str = r"
[parameters]
  theta TVCL(1.876, 0.01, 100.0)
  theta TVV(10.2591, 0.1, 1000.0)
  omega ETA_CL ~ 0.05 FIX
  kappa KAPPA_CL ~ 0.02 FIX
  sigma EPS ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
";

/// NONMEM 7.5.1 `OBJ` at the fixed point (`.ext` `-1000000000` row).
const NM_OFV_FOCEI: f64 = 554.92498260642810;
const NM_OFV_FOCE: f64 = 585.60024484944893;

/// Bounds, from the realised errors (macOS arm64, FOCEI / FOCE; the PR's Linux CI runs this
/// test at these bounds):
/// - CWRES: worst |Δ| 9.613e-8 / 4.196e-8. Before #1948, 0.4259 / 0.2524. `1e-6` is
///   10× / 24× above the first and 4.3e5× / 2.5e5× below the second. The floor is the two
///   engines' EBEs (next lines), carried through `f0 = f(b̂) − H·b̂`; the table's 11
///   significant figures sit well below it.
/// - IPRED: worst relative |Δ| 9.615e-8 / 1.421e-7; `1e-6` is 10× / 7× above.
/// - η̂ and κ̂ against `ETA1..3`: worst |Δ| 1.115e-7 / 2.158e-7; `2e-6` is 18× / 9× above.
/// - OFV: |Δ| 4.371e-7 / 7.855e-8; `5e-6` is 11× / 64× above.
/// - PRED: θ-only, worst relative |Δ| 8.404e-12 (the table's rounding); `1e-9` is 119×
///   above. A miss here means the parameter decoding or the row order is wrong, not the
///   residual pass.
///
/// Every bound but CWRES is the same before and after #1948 — the OFV to the bit
/// (`0x408157665d828b94` / `0x40824ccd4d2164d9`) — so those pin the point the residual is
/// computed at, not the fix.
const CWRES_TOL: f64 = 1e-6;
const IPRED_TOL: f64 = 1e-6;
const EBE_TOL: f64 = 2e-6;
const OFV_TOL: f64 = 5e-6;
const PRED_TOL: f64 = 1e-9;

/// One `EVID = 0` row of a NONMEM table.
struct NmRow {
    id: i64,
    occ: usize,
    pred: f64,
    ipred: f64,
    cwres: f64,
    eta: [f64; 3],
}

fn nonmem_obs(path: &str) -> Vec<NmRow> {
    let text = std::fs::read_to_string(path).expect("NONMEM table");
    text.lines()
        .skip(2)
        .map(|l| {
            let c: Vec<f64> = l
                .split_whitespace()
                .map(|x| x.parse().expect("number"))
                .collect();
            assert_eq!(
                c.len(),
                12,
                "ID TIME OCC EVID DV PRED IPRED IWRES CWRES ETA1 ETA2 ETA3"
            );
            c
        })
        .filter(|c| c[3] == 0.0)
        .map(|c| NmRow {
            id: c[0] as i64,
            occ: c[2] as usize,
            pred: c[5],
            ipred: c[6],
            cwres: c[8],
            eta: [c[9], c[10], c[11]],
        })
        .collect()
}

fn run(method: EstimationMethod, interaction: bool) -> FitResult {
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv_iov.csv"),
        Some(&["WT"]),
        Some("OCC"),
    )
    .expect("data");
    let model = parse_model_string(MODEL).expect("model");
    assert_eq!(model.n_kappa, 1, "one IOV kappa on CL");
    let opts = FitOptions {
        method,
        interaction,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..FitOptions::default()
    };
    fit(&model, &pop, &model.default_params, &opts).expect("fit must run")
}

fn assert_matches_nonmem(label: &str, r: &FitResult, tab: &str, nm_ofv: f64) {
    assert_eq!(r.ebe_kappas.len(), r.subjects.len(), "{label}: kappa rows");
    let want = nonmem_obs(tab);
    let n_got: usize = r.subjects.iter().map(|s| s.cwres.len()).sum();
    assert_eq!(n_got, want.len(), "{label}: observation count");

    let mut rows = want.iter();
    let (mut d_cwres, mut d_ipred, mut d_pred, mut d_ebe) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (s, kappas) in r.subjects.iter().zip(&r.ebe_kappas) {
        assert_eq!(kappas.len(), 2, "{label}: subject {} occasions", s.id);
        for j in 0..s.cwres.len() {
            let nm = rows.next().expect("row");
            assert_eq!(s.id, nm.id.to_string(), "{label}: row order");
            let (c, ip, p) = (s.cwres[j], s.ipred[j], s.pred[j]);
            // f64::max discards NaN, so finiteness is asserted before every fold.
            assert!(
                c.is_finite() && ip.is_finite() && p.is_finite(),
                "{label}: subject {} row {j}: CWRES {c} / IPRED {ip} / PRED {p} not finite",
                s.id
            );
            d_cwres = d_cwres.max((c - nm.cwres).abs());
            d_ipred = d_ipred.max((ip - nm.ipred).abs() / nm.ipred.abs().max(1e-12));
            d_pred = d_pred.max((p - nm.pred).abs() / nm.pred.abs().max(1e-12));
            // ETA1 is η, ETA2/ETA3 the occasion-1/-2 κ (identical on every row of a subject).
            let got = [s.eta[0], kappas[0][0], kappas[1][0]];
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{label}: subject {} EBE {got:?} not finite",
                s.id
            );
            assert!((1..=2).contains(&nm.occ), "{label}: OCC {}", nm.occ);
            for k in 0..3 {
                d_ebe = d_ebe.max((got[k] - nm.eta[k]).abs());
            }
        }
    }
    assert!(rows.next().is_none(), "{label}: unconsumed NONMEM rows");
    let d_ofv = (r.ofv - nm_ofv).abs();
    // The post-fit per-subject OFV, summed in subject order: the IOV marginal evaluated on
    // the linearization the residual pass now shares with it.
    let ofv_i: f64 = r.subjects.iter().map(|s| s.ofv_contribution).sum();
    eprintln!(
        "MEASURE {label}: worst |dCWRES| {d_cwres:.3e}, rel dIPRED {d_ipred:.3e}, \
         rel dPRED {d_pred:.3e}, |dEBE| {d_ebe:.3e}, |dOFV| {d_ofv:.3e} (ferx {:.12}, bits {:#x}; \
         sum OFV_i bits {:#x})",
        r.ofv,
        r.ofv.to_bits(),
        ofv_i.to_bits()
    );
    assert!(
        d_pred < PRED_TOL,
        "{label}: PRED off NONMEM by {d_pred:e} — the point or row order is wrong"
    );
    assert!(
        r.ofv.is_finite() && d_ofv < OFV_TOL,
        "{label}: OFV {} vs NONMEM {nm_ofv}",
        r.ofv
    );
    assert!(d_ebe < EBE_TOL, "{label}: EBEs off NONMEM by {d_ebe:e}");
    assert!(
        d_ipred < IPRED_TOL,
        "{label}: IPRED off NONMEM by {d_ipred:e}"
    );
    assert!(
        d_cwres < CWRES_TOL,
        "{label}: CWRES off NONMEM by {d_cwres:e} (> {CWRES_TOL}): the residual pass is not \
         linearizing over [eta, kappa_1, kappa_2] (#1948)"
    );
}

#[test]
fn iov_cwres_matches_nonmem_focei() {
    let r = run(EstimationMethod::FoceI, true);
    assert!(r.interaction, "FOCEI fixture premise");
    assert_matches_nonmem(
        "focei",
        &r,
        "tests/nonmem/iov_cwres_focei.tab",
        NM_OFV_FOCEI,
    );
}

#[test]
fn iov_cwres_matches_nonmem_foce() {
    let r = run(EstimationMethod::Foce, false);
    assert!(!r.interaction, "FOCE fixture premise");
    assert_matches_nonmem("foce", &r, "tests/nonmem/iov_cwres_foce.tab", NM_OFV_FOCE);
}
