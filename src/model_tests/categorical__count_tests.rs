//! Tier-1 tests for the count endpoint (#2877).
// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::parser::model_parser::with_model_time;
use ferx_model::sens::num::BranchCmp;
use ferx_model::sens::num::PkNum;
use ferx_model::stats::likelihood::SUBJECT_SENTINEL_NLL;
use ferx_model::stats::special::ln_gamma;
use ferx_model::stats::special::negbin_log_pmf_g;
use ferx_model::stats::special::NEGBIN_SUM_MAX_K;
use ferx_model::types::CompiledModel;
use ferx_model::types::CountFamily;
use ferx_model::types::CountMeanScale;
use ferx_model::types::CountParamFn;
use ferx_model::types::DiscreteObsDiagnostic;
use ferx_model::types::EndpointLikelihood;
use ferx_model::types::EndpointPredictionResult;
use ferx_model::types::ObsRecord;
use ferx_model::types::Prediction;
use ferx_model::types::Subject;
use std::collections::HashMap;

use crate::model_tests::categorical::exact_eta_hessian_tests::load;
use crate::sens::dual2::Dual2;
use crate::stats::special::count_log_pmf;
use crate::types::SimOutcome;
use ferx_model::categorical::count::*;

const P: CountFamily = CountFamily::Poisson;
const NB: CountFamily = CountFamily::NegBin;
const LIN: CountMeanScale = CountMeanScale::Linear;
const LOG: CountMeanScale = CountMeanScale::Log;

/// The record kernel against values computed by hand, outside both `count_log_pmf` and the
/// kernel: Poisson `k = 3, λ = 2` is `−log(2³e⁻²/3!) = 2 − 3·ln 2 + ln 6`; NB `k = 2, λ = 3,
/// r = 1.5` is `−log[Γ(3.5)/(Γ(1.5)·2!) · (1.5/4.5)^1.5 · (3/4.5)²]` with
/// `Γ(3.5)/Γ(1.5) = 2.5·1.5 = 3.75`. Both scales give the same value, and the `ln Γ(k + 1)`
/// constant is in (dropping it moves the Poisson value by `ln 6`).
#[test]
fn record_kernel_matches_hand_computed_pmf() {
    let pois = 2.0 - 3.0 * 2.0f64.ln() + 6.0f64.ln();
    let nb = -((3.75f64 / 2.0).ln() + 1.5 * (1.5f64 / 4.5).ln() + 2.0 * (3.0f64 / 4.5).ln());
    for (fam, size, k, lam, want) in [(P, 0.0, 3, 2.0, pois), (NB, 1.5, 2, 3.0, nb)] {
        let lin = count_record_nll_g(fam, LIN, lam, size, k).unwrap();
        let log = count_record_nll_g(fam, LOG, lam.ln(), size, k).unwrap();
        assert!((lin - want).abs() < 1e-12, "{fam:?} linear {lin} vs {want}");
        assert!((log - want).abs() < 1e-12, "{fam:?} log {log} vs {want}");
        let sz = (fam == NB).then_some(size);
        let pmf = -count_log_pmf(lam, sz, f64::from(k));
        assert!((pmf - want).abs() < 1e-12, "{fam:?} count_log_pmf {pmf} vs {want}");
    }
    // `k = 0` reduces to `λ` (Poisson) and `r·log(1 + λ/r)` (NB).
    let p0 = count_record_nll_g(P, LIN, 2.5, 0.0, 0).unwrap();
    assert!((p0 - 2.5).abs() < 1e-14, "{p0}");
    let n0 = count_record_nll_g(NB, LIN, 2.5, 4.0, 0).unwrap();
    assert!((n0 - 4.0 * (1.0f64 + 2.5 / 4.0).ln()).abs() < 1e-13, "{n0}");
}

/// The negative binomial reaches the Poisson as `r → ∞`, and differs from it at a small `r`
/// (so a NB arm that scored Poisson could not pass the hand value above *and* this pair).
#[test]
fn negbin_tends_to_poisson_and_differs_at_small_size() {
    for k in [0u32, 1, 4, 11] {
        let p = count_record_nll_g(P, LIN, 3.7, 0.0, k).unwrap();
        let big = count_record_nll_g(NB, LIN, 3.7, 1e9, k).unwrap();
        let small = count_record_nll_g(NB, LIN, 3.7, 0.8, k).unwrap();
        assert!((p - big).abs() < 1e-6, "k={k}: Poisson {p} vs NB(r=1e9) {big}");
        assert!((p - small).abs() > 0.05, "k={k}: NB(r=0.8) {small} too close to {p}");
    }
}

/// A size large enough that `λ/r` falls below machine epsilon is still a valid law (the
/// Poisson limit), and the kernel must keep `r·log1p(λ/r)` there: `r·ln(1 + λ/r)` rounds
/// `1 + λ/r` to 1 and scored `λ = 1, r = 1e17, k = 0` as ~0 instead of ~1. Values are
/// checked against `count_log_pmf` (which uses `f64::ln_1p`) and against the Poisson; the
/// dual's `∂/∂λ` and `∂²/∂λ²` against the closed forms `(k+r)/(r+λ) − k/λ` and
/// `k/λ² − (k+r)/(r+λ)²`. Both sides of the series switch (`λ/r = 1e-3`) are visited, and
/// the two branches meet without a jump. Measured worst relative value gap 6.8e-16 (bound 1e-13).
#[test]
fn negbin_keeps_its_digits_at_huge_size() {
    let k0 = count_record_nll_g(NB, LIN, 1.0, 1e17, 0).unwrap();
    assert!((k0 - 1.0).abs() < 1e-14, "λ=1, r=1e17, k=0: {k0} (want 1)");
    let mut worst = 0.0f64;
    for (lam, r) in [
        (1.0, 1e17),
        (3.7, 1e300),
        (2.0, 1e12),
        (2.0, 1999.0), // λ/r just above the switch
        (2.0, 2001.0), // just below it
        (0.5, 3.0),
    ] {
        for k in [0u32, 1, 3, 7] {
            let got = count_record_nll_g(NB, LIN, lam, r, k).unwrap();
            let pmf = -count_log_pmf(lam, Some(r), f64::from(k));
            assert!(got.is_finite() && pmf.is_finite());
            let gap = (got - pmf).abs() / pmf.abs().max(1.0);
            assert!(gap.is_finite());
            worst = worst.max(gap);
            assert!(gap < 1e-13, "λ={lam} r={r} k={k}: {got} vs count_log_pmf {pmf}");
            if r >= 1e12 {
                let pois = count_record_nll_g(P, LIN, lam, 0.0, k).unwrap();
                assert!((got - pois).abs() < 1e-10, "λ={lam} r={r} k={k}: {got} vs Poisson {pois}");
            }
            let d = count_record_nll_g(NB, LIN, Dual2::<1>::var(lam, 0), Dual2::constant(r), k)
                .unwrap();
            let kf = f64::from(k);
            let g = (kf + r) / (r + lam) - kf / lam;
            let h = kf / (lam * lam) - (kf + r) / ((r + lam) * (r + lam));
            assert!(d.grad[0].is_finite() && d.hess[0][0].is_finite());
            assert!((d.grad[0] - g).abs() < 1e-12, "λ={lam} r={r} k={k}: ∂λ {} vs {g}", d.grad[0]);
            assert!(
                (d.hess[0][0] - h).abs() < 1e-12,
                "λ={lam} r={r} k={k}: ∂²λ {} vs {h}",
                d.hess[0][0]
            );
        }
    }
    // The two branches agree where they meet: one ulp either side of `λ/r = 1e-3`.
    let r = 1000.0;
    let below = count_record_nll_g(NB, LIN, 1.0 - 1e-12, r, 2).unwrap();
    let above = count_record_nll_g(NB, LIN, 1.0 + 1e-12, r, 2).unwrap();
    assert!((below - above).abs() < 1e-11, "switch jump {below} vs {above}");
    eprintln!("NB huge-size worst relative value gap {worst:.2e}");
}

/// Counts past `NEGBIN_SUM_MAX_K` at sizes where the `ln Γ(r + k) − ln Γ(r)` difference
/// cancels catastrophically. The kernel and `count_log_pmf` are checked against two outside
/// references: R's `-dnbinom(k, size = r, mu = λ, log = TRUE)` (the `pmx` container; its
/// saddle-point form shares no code with ferx) and the definition summed directly in the
/// test with `f64::ln_1p`. The rows visit the Stirling branch at a huge and a moderate size
/// (`r = 1e17, 1e9, 25`) and at its edge (`r = 20`), and the `ln Γ` branch below it; the
/// first is the review's case, which the `ln Γ` difference scored 335.69 for 3.69. The `∂/∂r`, `∂²/∂r²` the dual
/// carries on the Stirling branch match central differences at sizes FD can resolve (at
/// `r = 1e9` the derivative is ~`1/r²`, below FD noise). Measured worst relative gaps:
/// vs R 5.9e-12 (R's own: at `r = 1e9, k = 300` it is 4e-11 off a direct `fsum`, where the
/// kernel is within 3e-13), vs the direct sum 1.5e-13 (the truncated Stirling series at
/// `r = 20`), `∂r` 2.3e-9, `∂²r` 1.1e-8.
#[test]
fn negbin_high_counts_keep_their_digits_at_any_size() {
    let rows: [(f64, f64, u32, f64); 7] = [
        (257.0, 1e17, 257, 3.6938008297041853),
        (257.0, 1e9, 300, 7.1830237020439816),
        (5.0, 1e17, 600, 2281.6125879189144),
        (400.0, 25.0, 300, 5.9378328801361855),
        (300.0, 20.0, 300, 5.1613078124376628),
        (400.0, 25.0, 1000, 20.155978554065801),
        (40.0, 3.3, 700, 49.900385769055134),
    ];
    // The definition summed directly, `Σ_{j<k} ln_1p(j/r)` with `f64::ln_1p` — no branch
    // of the kernel's, and exact to ~`k·eps` plus `ln Γ(k + 1)`'s own rounding.
    let direct = |lam: f64, r: f64, k: u32| {
        let s: f64 = (0..k).map(|j| (f64::from(j) / r).ln_1p()).sum();
        let kf = f64::from(k);
        -(s - ln_gamma(kf + 1.0) + kf * lam.ln() - r * (lam / r).ln_1p() * (1.0 + kf / r))
    };
    let (mut worst_r, mut worst_d) = (0.0f64, 0.0f64);
    for (lam, r, k, want) in rows {
        assert!(k > NEGBIN_SUM_MAX_K);
        let got = count_record_nll_g(NB, LIN, lam, r, k).unwrap();
        let log = count_record_nll_g(NB, LOG, lam.ln(), r, k).unwrap();
        let pmf = -count_log_pmf(lam, Some(r), f64::from(k));
        let dir = direct(lam, r, k);
        assert!(got.is_finite() && log.is_finite() && pmf.is_finite() && dir.is_finite());
        let gap_r = (got - want).abs() / want.abs().max(1.0);
        let gap_d = (got - dir).abs() / dir.abs().max(1.0);
        assert!(gap_r.is_finite() && gap_d.is_finite());
        worst_r = worst_r.max(gap_r);
        worst_d = worst_d.max(gap_d);
        assert!(gap_r < 1e-10, "λ={lam} r={r} k={k}: kernel {got} vs R {want}");
        assert!(gap_d < 1e-12, "λ={lam} r={r} k={k}: kernel {got} vs direct sum {dir}");
        assert!((log - got).abs() / got < 1e-13, "λ={lam} r={r} k={k}: log scale {log}");
        assert_eq!(got, pmf, "λ={lam} r={r} k={k}: kernel and count_log_pmf differ");
    }
    eprintln!("NB high-count worst relative gap: vs R {worst_r:.2e}, vs direct sum {worst_d:.2e}");
    let mut worst_g = 0.0f64;
    let mut worst_h = 0.0f64;
    for (lam, r, k) in [(400.0, 25.0, 1000u32), (30.0, 60.0, 400), (300.0, 500.0, 700)] {
        let d =
            count_record_nll_g(NB, LIN, Dual2::<1>::constant(lam), Dual2::var(r, 0), k).unwrap();
        let f = |rr: f64| count_record_nll_g(NB, LIN, lam, rr, k).unwrap();
        let g = |rr: f64| {
            count_record_nll_g(NB, LIN, Dual2::<1>::constant(lam), Dual2::var(rr, 0), k)
                .unwrap()
                .grad[0]
        };
        let h1 = 1e-5 * r;
        let fd = (f(r + h1) - f(r - h1)) / (2.0 * h1);
        // The Hessian against central differences of the dual gradient (itself pinned to
        // the value above): a second difference of the value is noise-bound here.
        let fd2 = (g(r + h1) - g(r - h1)) / (2.0 * h1);
        assert!(d.grad[0].is_finite() && d.hess[0][0].is_finite());
        assert!(fd.is_finite() && fd2.is_finite());
        let eg = (d.grad[0] - fd).abs() / fd.abs().max(1e-12);
        let eh = (d.hess[0][0] - fd2).abs() / fd2.abs().max(1e-12);
        assert!(eg.is_finite() && eh.is_finite());
        worst_g = worst_g.max(eg);
        worst_h = worst_h.max(eh);
    }
    eprintln!("NB high-count ∂r {worst_g:.2e}, ∂²r {worst_h:.2e}");
    assert!(worst_g < 1e-7, "∂r gap {worst_g:.2e}");
    assert!(worst_h < 1e-6, "∂²r gap {worst_h:.2e}");
}

/// Both sides of `NEGBIN_SUM_MAX_K`: the term-by-term sum and the `ln Γ` difference agree
/// with each other and with `count_log_pmf` at a count either side of the switch (an
/// off-by-one in either branch, or a sum over the wrong range, moves the value by `ln(r+k)`).
#[test]
fn negbin_lgamma_branches_agree_across_the_switch() {
    let r = 3.3;
    let lam = 250.0;
    for k in [NEGBIN_SUM_MAX_K - 1, NEGBIN_SUM_MAX_K, NEGBIN_SUM_MAX_K + 1] {
        let got = count_record_nll_g(NB, LIN, lam, r, k).unwrap();
        let kf = f64::from(k);
        let direct = -(ln_gamma(kf + r) - ln_gamma(r) - ln_gamma(kf + 1.0)
            + r * (r / (r + lam)).ln()
            + kf * (lam / (r + lam)).ln());
        // `ln Γ(k + r)` is ~1200 here: the direct form carries ~1e-10 of rounding.
        assert!((got - direct).abs() < 1e-9, "k={k}: {got} vs {direct}");
        let pmf = -count_log_pmf(lam, Some(r), kf);
        assert!((got - pmf).abs() < 1e-9, "k={k}: {got} vs count_log_pmf {pmf}");
    }
}

/// Every point where the law is undefined is declined, never scored: a non-positive or
/// non-finite `lambda`, a non-finite `log λ` or one whose exponential overflows, and a
/// non-positive or non-finite size on a negative binomial. A Poisson ignores the size slot.
#[test]
fn undefined_laws_are_declined() {
    for lam in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(count_record_nll_g(P, LIN, lam, 0.0, 2).is_none(), "λ = {lam}");
    }
    for ll in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 800.0] {
        assert!(count_record_nll_g(P, LOG, ll, 0.0, 2).is_none(), "log λ = {ll}");
    }
    for r in [0.0, -2.0, f64::NAN, f64::INFINITY] {
        assert!(count_record_nll_g(NB, LIN, 2.0, r, 2).is_none(), "r = {r}");
    }
    assert!(count_record_nll_g(P, LIN, 2.0, f64::NAN, 2).is_some());
    // A very negative `log λ` underflows `λ` to 0 and is still a law (k = 0 certain).
    let v = count_record_nll_g(P, LOG, -800.0, 0.0, 0).unwrap();
    assert_eq!(v, 0.0);
}

/// Dual-vs-FD parity of the record kernel in `(mean, size)` for both families and scales:
/// the gradient and the Hessian `Dual2` carries must match central differences of the
/// `f64` kernel. Measured worst relative gap 2.7e-9 (gradient, h = 1e-6) and 1.8e-7
/// (Hessian, h = 1e-4) — the FD error, not the dual's; the bounds keep ≥ 37× headroom.
/// A wrong digamma / `log1p` derivative or a dropped `k·log λ` term moves them by O(1).
#[test]
fn record_kernel_dual_matches_fd() {
    let f = |fam, sc, m: f64, r: f64, k| count_record_nll_g(fam, sc, m, r, k).unwrap();
    let mut worst_g = 0.0f64;
    let mut worst_h = 0.0f64;
    for (fam, sc, m, r) in
        [(P, LIN, 2.3, 0.0), (P, LOG, 0.4, 0.0), (NB, LIN, 2.3, 1.7), (NB, LOG, 1.1, 6.0)]
    {
        for k in [0u32, 1, 3, 9] {
            let d =
                count_record_nll_g(fam, sc, Dual2::<2>::var(m, 0), Dual2::var(r, 1), k).unwrap();
            assert!((d.value - f(fam, sc, m, r, k)).abs() < 1e-12);
            let n = if fam == NB { 2 } else { 1 };
            let at = |dm: f64, dr: f64| f(fam, sc, m + dm, r + dr, k);
            let h1 = 1e-6;
            let h2 = 1e-4;
            for i in 0..n {
                let e = |s: f64| if i == 0 { (s, 0.0) } else { (0.0, s) };
                let (a, b) = e(h1);
                let fd = (at(a, b) - at(-a, -b)) / (2.0 * h1);
                let g = d.grad[i];
                assert!(g.is_finite() && fd.is_finite());
                worst_g = worst_g.max((g - fd).abs() / fd.abs().max(1.0));
                for j in 0..n {
                    let (ai, bi) = e(h2);
                    let (aj, bj) = if j == 0 { (h2, 0.0) } else { (0.0, h2) };
                    let fd2 = (at(ai + aj, bi + bj) - at(ai - aj, bi - bj) - at(aj - ai, bj - bi)
                        + at(-ai - aj, -bi - bj))
                        / (4.0 * h2 * h2);
                    let hh = d.hess[i][j];
                    assert!(hh.is_finite() && fd2.is_finite());
                    worst_h = worst_h.max((hh - fd2).abs() / fd2.abs().max(1.0));
                }
            }
            if fam == P {
                assert_eq!(d.grad[1], 0.0, "a Poisson record must not read the size slot");
            }
        }
    }
    eprintln!("count kernel dual vs FD: grad {worst_g:.2e}, hess {worst_h:.2e}");
    assert!(worst_g < 1e-7, "gradient gap {worst_g:.2e}");
    assert!(worst_h < 1e-4, "Hessian gap {worst_h:.2e}");
}

const NB_MODEL: &str = r"
[parameters]
  theta BASE(4.3, 0.01, 100.0)
  theta BETA(0.3, -10.0, 10.0)
  theta GAM(-0.6, -10.0, 10.0)
  theta R(2.5, 0.01, 1000.0)
  omega ETA_C ~ 0.2
  omega ETA_G ~ 0.1
[covariates]
  WT continuous
  TRT continuous
[count_model]
  cmt    = 1
  type   = negbin
  lambda = BASE * exp(BETA * (WT / 70 - 1) + GAM * TRT * exp(ETA_G) + ETA_C)
  r      = R
";

/// The data term reached through the production loader is the per-record kernel summed
/// over the subject's records (TRT changes within a subject, so each record reads its own
/// covariate snapshot), and `predict_count` / `count_diagnostics` report the same `λ` at
/// the same η.
#[test]
fn loaded_data_term_sums_the_records_at_their_own_covariates() {
    let prep = load(NB_MODEL, "nonmem_anchor/count_nb.csv");
    let model = &prep.parsed.model;
    assert!(model.has_count() && model.has_discrete());
    let theta = &model.default_params.theta;
    let eta = [0.13, -0.2];
    let s = &prep.population.subjects[0];
    let n_count = s.obs_records.iter().filter(|r| matches!(r, ObsRecord::Count { .. })).count();
    assert_eq!(n_count, 6, "every row is a count record");
    let wt = s.covariates["WT"];
    let trt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut want = 0.0;
    let mut lams = Vec::new();
    for (r, t) in s.obs_records.iter().zip(trt) {
        let ObsRecord::Count { count, .. } = r else { unreachable!() };
        let lam =
            theta[0] * (theta[1] * (wt / 70.0 - 1.0) + theta[2] * t * eta[1].exp() + eta[0]).exp();
        want -= count_log_pmf(lam, Some(theta[3]), f64::from(*count));
        lams.push(lam);
    }
    let got = ferx_model::categorical::discrete_subject_nll(model, s, theta, &eta);
    assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    // The record pair straddling TRT must differ, or the snapshot test above is vacuous.
    assert!((lams[2] - lams[3]).abs() > 0.1, "{lams:?}");

    let mut preds = Vec::new();
    predict_count(model, s, theta, &eta, &mut preds);
    let diags = count_diagnostics(model, s, theta, &eta);
    assert_eq!(preds.len(), 6);
    assert_eq!(diags.len(), 6);
    for ((p, d), lam) in preds.iter().zip(&diags).zip(&lams) {
        let Prediction::Rate { lambda, size } = p.prediction else { panic!("{:?}", p.prediction) };
        assert!((lambda - lam).abs() < 1e-12 && (d.ipred - lam).abs() < 1e-12);
        assert_eq!(size, Some(theta[3]));
        assert_eq!(p.observed, Some(d.dv));
        let var = lam + lam * lam / theta[3];
        assert!((d.iwres - (d.dv - lam) / var.sqrt()).abs() < 1e-12);
        // `prob(k)` is the NB PMF at the record's mean.
        let k = d.dv as usize;
        let pk = p.prediction.prob(k).unwrap();
        assert!((pk - count_log_pmf(*lam, Some(theta[3]), d.dv).exp()).abs() < 1e-14);
    }
    // PRED is the mean at η = 0.
    let pred0 = theta[0] * (theta[1] * (wt / 70.0 - 1.0)).exp();
    assert!((diags[0].pred - pred0).abs() < 1e-12, "{}", diags[0].pred);
}

/// The exact η-gradient and η-Hessian of the count data term come from the subject's
/// compiled plan (the routing predicate says so), and match central differences of the
/// `f64` value path — for the negative binomial, with a second random effect entering
/// non-linearly, so the Hessian has an off-diagonal block. Measured: gradient 9.8e-11,
/// Hessian 1.7e-7 (relative, h = 1e-5 / 1e-4); the bounds keep ≥ 59× headroom.
#[test]
fn planned_eta_gradient_and_hessian_match_fd() {
    let prep = load(NB_MODEL, "nonmem_anchor/count_nb.csv");
    let model = &prep.parsed.model;
    assert!(crate::categorical::all_discrete_endpoints_analytic(model));
    let theta = &model.default_params.theta;
    let eta = [0.13, -0.2];
    let nll =
        |s: &Subject, e: &[f64]| ferx_model::categorical::discrete_subject_nll(model, s, theta, e);
    let (mut wg, mut wh) = (0.0f64, 0.0f64);
    for s in prep.population.subjects.iter().take(5) {
        let (v, g) = ferx_model::categorical::discrete_subject_eta_grad(model, s, theta, &eta)
            .expect("the count term records");
        assert!((v - nll(s, &eta)).abs() < 1e-9);
        let (_, h) = ferx_model::categorical::discrete_subject_eta_hessian(model, s, theta, &eta)
            .expect("the count term records");
        for i in 0..2 {
            let mut p = eta;
            let mut m = eta;
            p[i] += 1e-5;
            m[i] -= 1e-5;
            let fd = (nll(s, &p) - nll(s, &m)) / 2e-5;
            assert!(g[i].is_finite() && fd.is_finite());
            wg = wg.max((g[i] - fd).abs() / fd.abs().max(1.0));
            for j in 0..2 {
                let at = |di: f64, dj: f64| {
                    let mut e = eta;
                    e[i] += di;
                    e[j] += dj;
                    nll(s, &e)
                };
                let hh = 1e-4;
                let fd2 = (at(hh, hh) - at(hh, -hh) - at(-hh, hh) + at(-hh, -hh)) / (4.0 * hh * hh);
                assert!(h[(i, j)].is_finite() && fd2.is_finite());
                wh = wh.max((h[(i, j)] - fd2).abs() / fd2.abs().max(1.0));
            }
        }
        assert!(h[(0, 1)].abs() > 1e-3, "the off-diagonal block must be live");
    }
    eprintln!("count planned η-derivatives vs FD: grad {wg:.2e}, hess {wh:.2e}");
    assert!(wg < 1e-7, "gradient {wg:.2e}");
    assert!(wh < 1e-5, "Hessian {wh:.2e}");
}

/// A subject whose records sit on a count endpoint with a program is served by the exact
/// inner gradient; the same model with a size that has no program would route to FD. Here
/// the routing predicate is pinned on both sides of `param_program`.
#[test]
fn count_records_route_by_program_presence() {
    let prep = load(NB_MODEL, "nonmem_anchor/count_nb.csv");
    let mut model = prep.parsed.model;
    assert!(ferx_model::categorical::count::all_count_endpoints_analytic(&model));
    if let Some(EndpointLikelihood::Count { param_program, .. }) = model.endpoints.get_mut(&1) {
        *param_program = None;
    }
    assert!(!ferx_model::categorical::count::all_count_endpoints_analytic(&model));
    assert!(!crate::categorical::all_discrete_endpoints_analytic(&model));
    let s = &prep.population.subjects[0];
    let theta = model.default_params.theta.clone();
    assert!(
        ferx_model::categorical::discrete_subject_eta_grad(&model, s, &theta, &[0.0, 0.0])
            .is_none(),
        "no program, no planned gradient"
    );
}

/// An undefined record law reaches the objective as the `1e20` sentinel (finite, so the
/// optimizer is repelled rather than handed a NaN), and the plan declines the point.
#[test]
fn an_undefined_record_is_the_sentinel_and_declined_by_the_plan() {
    let prep = load(
        &NB_MODEL.replace("lambda = BASE * exp(", "lambda = (BASE - 5) * exp("),
        "nonmem_anchor/count_nb.csv",
    );
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta; // BASE = 4.3 ⇒ λ < 0
    let s = &prep.population.subjects[0];
    let v = ferx_model::categorical::discrete_subject_nll(model, s, theta, &[0.0, 0.0]);
    assert!(v >= SUBJECT_SENTINEL_NLL && v.is_finite(), "{v}");
    assert!(
        ferx_model::categorical::discrete_subject_eta_grad(model, s, theta, &[0.0, 0.0]).is_none()
    );
}

/// The simulated counts follow their law: Poisson variance ≈ mean, NB variance ≈
/// `μ + μ²/r`. 40 000 draws at μ = 4 (r = 2 for NB, variance 12): the sample variance's
/// standard error is about 0.04 (Poisson) and 0.3 (NB), so the 4% and 6% bands are more
/// than 4 standard errors wide. The Poisson and NB targets differ by 8, so neither family's
/// draw passes the other's check.
#[test]
fn simulated_counts_have_the_law_s_mean_and_variance() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(2877);
    for (size, var_want, band) in [(None, 4.0, 0.04), (Some(2.0), 12.0, 0.06)] {
        let n = 40_000;
        let xs: Vec<f64> =
            (0..n).map(|_| f64::from(draw_count(4.0, size, &mut rng).unwrap())).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        eprintln!("size {size:?}: mean {mean:.4}, var {var:.4} (want {var_want})");
        assert!((mean - 4.0).abs() < 0.05, "mean {mean}");
        assert!((var / var_want - 1.0).abs() < band, "var {var} vs {var_want}");
    }
    // A zero mean draws zero.
    assert_eq!(draw_count(0.0, None, &mut rng), Some(0));
}

/// `simulate_count` emits one `SimOutcome::Count` row per count record with `ipred = λ`, on
/// the user clock, and skips (with one warning naming the subject and the CMT) a record
/// whose law is undefined.
#[test]
fn simulate_count_emits_one_row_per_record_and_warns_on_undefined_laws() {
    use rand::SeedableRng;
    let prep = load(NB_MODEL, "nonmem_anchor/count_nb.csv");
    let model = &prep.parsed.model;
    let theta = &model.default_params.theta;
    let s = &prep.population.subjects[0];
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let (mut out, mut warn) = (Vec::new(), Vec::new());
    simulate_count(model, s, theta, &[0.0, 0.0], 0, 0, &mut rng, &mut out, &mut warn);
    assert_eq!(out.len(), 6);
    assert!(warn.is_empty(), "{warn:?}");
    let mut preds = Vec::new();
    predict_count(model, s, theta, &[0.0, 0.0], &mut preds);
    for (row, p) in out.iter().zip(&preds) {
        assert!(matches!(row.outcome, SimOutcome::Count { .. }));
        assert_eq!(row.time, p.time);
        let Prediction::Rate { lambda, .. } = p.prediction else { unreachable!() };
        assert_eq!(row.ipred, lambda);
    }
    let mut bad = theta.clone();
    bad[3] = -1.0; // size r < 0: no law
    let (mut out, mut warn) = (Vec::new(), Vec::new());
    simulate_count(model, s, &bad, &[0.0, 0.0], 0, 0, &mut rng, &mut out, &mut warn);
    assert!(out.is_empty());
    assert_eq!(warn.len(), 1, "{warn:?}");
    let w = &warn[0];
    assert!(w.contains("[count_model] cmt = 1") && w.contains(&format!("subject {}", s.id)));
    assert!(w.contains("6 record(s)"), "{w}");
    assert!(w.contains("size `r` finite and positive"), "{w}");
}

/// The Pearson residual divides by the family's own variance, and is NaN (an empty sdtab
/// cell) where that variance is not positive.
#[test]
fn pearson_residual_uses_the_family_variance() {
    assert!((count_pearson_residual(6.0, 4.0, None) - 1.0).abs() < 1e-15);
    assert!((count_pearson_residual(6.0, 4.0, Some(2.0)) - 2.0 / 12.0f64.sqrt()).abs() < 1e-15);
    assert!(count_pearson_residual(1.0, 0.0, None).is_nan());
}

fn parse_err(block: &str) -> String {
    let src = format!(
        "[parameters]\n  theta B(2.0, 0.01, 100.0)\n  theta R(2.0, 0.01, 100.0)\n  omega E ~ 0.1\n\
         [count_model]\n{block}"
    );
    match crate::parser::model_parser::parse_model_string(&src) {
        Ok(_) => panic!("parsed:\n{block}"),
        Err(e) => e.to_string(),
    }
}

/// Every refusal of the block names what is wrong and what to write instead; each sentence
/// is asserted, together with what tells it from its neighbours.
#[test]
fn count_model_block_refusals_name_the_fix() {
    let e = parse_err("  cmt = 1\n  type = negbin\n  lambda = B * exp(E)\n");
    assert!(e.contains("`type = negbin` needs the size `r = …`"), "{e}");
    assert!(e.contains("λ + λ²/r"), "{e}");
    let e = parse_err("  cmt = 1\n  lambda = B * exp(E)\n  r = R\n");
    assert!(e.contains("the block is a Poisson endpoint"), "{e}");
    assert!(e.contains("add `type = negbin`, or drop `r`"), "{e}");
    let e = parse_err("  cmt = 1\n  lambda = B\n  log_lambda = log(B) + E\n");
    assert!(e.contains("the mean is given twice (`lambda` and `log_lambda`)"), "{e}");
    let e = parse_err("  cmt = 1\n  lambda = B\n  lambda = B + E\n");
    assert!(e.contains("`lambda` is given twice"), "{e}");
    assert!(!e.contains("log_lambda` and"), "{e}");
    let e = parse_err("  cmt = 1\n  type = zip\n  lambda = B * exp(E)\n");
    assert!(e.contains("type `zip` is not supported"), "{e}");
    assert!(e.contains("`poisson` (the default) and `negbin`"), "{e}");
    let e = parse_err("  cmt = 1\n  mu = B * exp(E)\n");
    assert!(e.contains("unknown key `mu`"), "{e}");
    assert!(e.contains("valid keys: cmt, type, lambda, log_lambda, r"), "{e}");
    let e = parse_err("  cmt = 1\n  type = poisson\n");
    assert!(e.contains("missing the mean"), "{e}");
    assert!(e.contains("`log_lambda = …` (its log)"), "{e}");
    let e = parse_err("  lambda = B * exp(E)\n");
    assert!(e.contains("missing required key `cmt`"), "{e}");
    let e = parse_err("  cmt = 0\n  lambda = B * exp(E)\n");
    assert!(e.contains("cmt = 0 is invalid"), "{e}");
    let e = parse_err("  cmt = 1\n  r = R\n  r = R\n  type = negbin\n  lambda = B\n");
    assert!(e.contains("`r` is given twice"), "{e}");
}

/// A Poisson block defaults its type, takes either mean spelling, and the two spellings of
/// the same mean score the same objective.
#[test]
fn lambda_and_log_lambda_spell_the_same_mean() {
    let pois = |mean: &str| {
        format!(
            "[parameters]\n  theta BASE(4.0, 0.01, 100.0)\n  theta BETA(0.5, -10.0, 10.0)\n  \
             omega ETA_C ~ 0.2\n[covariates]\n  WT continuous\n[count_model]\n  cmt = 1\n  {mean}\n"
        )
    };
    let a = load(
        &pois("lambda = BASE * exp(BETA * (WT / 70 - 1) + ETA_C)"),
        "nonmem_anchor/count_pois.csv",
    );
    let b = load(
        &pois("log_lambda = log(BASE) + BETA * (WT / 70 - 1) + ETA_C"),
        "nonmem_anchor/count_pois.csv",
    );
    assert!(matches!(
        a.parsed.model.endpoints.get(&1),
        Some(EndpointLikelihood::Count {
            family: CountFamily::Poisson,
            scale: CountMeanScale::Linear,
            ..
        })
    ));
    let th = &a.parsed.model.default_params.theta;
    for (sa, sb) in a.population.subjects.iter().zip(&b.population.subjects).take(4) {
        let va = ferx_model::categorical::discrete_subject_nll(&a.parsed.model, sa, th, &[0.3]);
        let vb = ferx_model::categorical::discrete_subject_nll(&b.parsed.model, sb, th, &[0.3]);
        assert!((va - vb).abs() < 1e-10, "{va} vs {vb}");
    }
}

/// `--simulate` (`run_model_simulate`) on a count model: the design gets one count template
/// per (subject × time), each is stamped with a draw (not left at its `0` placeholder), and
/// the refit sees them. A count model whose `[simulation]` has no `times` is refused with a
/// message naming the count endpoint.
#[test]
fn simulate_estimate_stamps_count_draws_and_needs_times() {
    let model = "[parameters]\n  theta BASE(6.0, 0.01, 100.0)\n  theta R(3.0, 0.01, 100.0)\n  \
                 omega ETA_C ~ 0.2\n[count_model]\n  cmt = 2\n  type = negbin\n  \
                 log_lambda = log(BASE) + ETA_C\n  r = R\n[fit_options]\n  method = laplace\n  \
                 maxiter = 3\n";
    let run = |sim: &str| {
        let f = tempfile::Builder::new().suffix(".ferx").tempfile().unwrap();
        std::fs::write(f.path(), format!("{model}[simulation]\n{sim}")).unwrap();
        crate::api::run_model_simulate(f.path().to_str().unwrap())
    };
    let (fit, pop) = run("  n_subjects = 8\n  times = [0, 7, 14, 21]\n  seed = 5\n").unwrap();
    assert!(fit.ofv.is_finite(), "{}", fit.ofv);
    let counts: Vec<u32> = pop
        .subjects
        .iter()
        .flat_map(|s| &s.obs_records)
        .map(|r| match r {
            ObsRecord::Count { count, cmt, .. } => {
                assert_eq!(*cmt, 2);
                *count
            }
            other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(counts.len(), 32, "one record per subject × time");
    // At mean 6 a placeholder-only population (all 0) has probability ~1e-30.
    assert!(counts.iter().filter(|&&c| c > 0).count() > 20, "{counts:?}");
    let err = run("  n_subjects = 2\n  horizon = 10\n  seed = 5\n").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.starts_with("[simulation] has no `times`, but the model has a [count_model] endpoint."),
        "{msg}"
    );
    assert!(msg.contains("needs `times = [...]`"), "{msg}");
}

/// A gather index read by a count mean is validated on the count records' own snapshots:
/// an out-of-range value on a later count record is the `E_THETA_GATHER_INDEX_RANGE` data
/// error naming its TIME, a missing one `E_THETA_GATHER_INDEX_MISSING`, and an in-range
/// one passes (review r2 #2).
#[test]
fn gather_index_is_validated_on_count_records() {
    use crate::types::Population;
    let model = crate::parser::model_parser::parse_full_model(
        r"
[parameters]
  theta RATE[2](2.0, 0.01, 100.0)
[count_model]
  cmt    = 1
  lambda = RATE[IDX]
",
    )
    .unwrap()
    .model;
    let population = |second: Option<f64>| {
        let idx =
            |v: Option<f64>| v.map(|v| HashMap::from([("IDX".to_string(), v)])).unwrap_or_default();
        let mut s = Subject {
            id: "1".into(),
            covariates: idx(Some(1.0)),
            endpoint_covariates: vec![idx(Some(1.0)), idx(second)],
            ..Default::default()
        };
        s.obs_records = [0.0, 4.0]
            .map(|t| ObsRecord::Count { time: t, raw_time: t, count: 2, cmt: 1 })
            .to_vec();
        Population {
            subjects: vec![s],
            covariate_names: vec!["IDX".into()],
            dv_column: "DV".into(),
            input_columns: Vec::new(),
            exclusions: None,
            warnings: Vec::new(),
        }
    };
    let codes = |second: Option<f64>| -> Vec<(String, String)> {
        crate::api::check_model_data(&model, &population(second))
            .into_iter()
            .filter(|d| d.code.starts_with("E_THETA_GATHER_INDEX"))
            .map(|d| (d.code.to_string(), d.message))
            .collect()
    };
    let refused = codes(Some(3.0));
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0].0, "E_THETA_GATHER_INDEX_RANGE");
    assert!(refused[0].1.contains("is 3 on subject `1` at TIME 4"), "{}", refused[0].1);
    let missing = codes(None);
    assert_eq!(missing.len(), 1, "{missing:?}");
    assert_eq!(missing[0].0, "E_THETA_GATHER_INDEX_MISSING");
    assert!(missing[0].1.contains("at TIME 4"), "{}", missing[0].1);
    assert!(codes(Some(2.0)).is_empty(), "in range");
}

/// The zero-mean limit (a `log_lambda` that underflows `λ` to 0) is the law with all mass
/// at 0 everywhere: the likelihood, the PMF helper behind `Prediction::prob`, and the
/// sampler for both families — the negative binomial included, whose gamma mixture cannot
/// be built at a zero scale (review r2 #3).
#[test]
fn zero_mean_limit_is_consistent_across_likelihood_pmf_and_sampler() {
    use rand::SeedableRng;
    for size in [None, Some(2.0)] {
        assert_eq!(count_log_pmf(0.0, size, 0.0), 0.0, "{size:?}");
        assert_eq!(count_log_pmf(0.0, size, 3.0), f64::NEG_INFINITY, "{size:?}");
        let p = Prediction::Rate { lambda: 0.0, size };
        assert_eq!(p.prob(0), Some(1.0), "{size:?}");
        assert_eq!(p.prob(1), Some(0.0), "{size:?}");
    }
    assert!(count_log_pmf(0.0, Some(-1.0), 0.0).is_nan(), "a bad size stays undefined");
    assert!(count_log_pmf(-1.0, None, 0.0).is_nan());
    assert_eq!(count_record_nll_g(NB, LOG, -800.0, 2.0, 0), Some(0.0));
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    assert_eq!(draw_count(0.0, Some(2.0), &mut rng), Some(0));
    // Through `simulate_count`: every record draws 0 and nothing is skipped or warned.
    let prep = load(
        &NB_MODEL.replace(
            "lambda = BASE * exp(BETA * (WT / 70 - 1) + GAM * TRT * exp(ETA_G) + ETA_C)",
            "log_lambda = -800 + ETA_C",
        ),
        "nonmem_anchor/count_nb.csv",
    );
    let model = &prep.parsed.model;
    let s = &prep.population.subjects[0];
    let (mut out, mut warn) = (Vec::new(), Vec::new());
    let th = &model.default_params.theta;
    simulate_count(model, s, th, &[0.0, 0.0], 0, 0, &mut rng, &mut out, &mut warn);
    assert!(warn.is_empty(), "{warn:?}");
    assert_eq!(out.len(), 6);
    assert!(out.iter().all(|r| matches!(r.outcome, SimOutcome::Count { count: 0 })));
}

/// A data-driven θ level block read bare by a count mean, on count-only data (review r3):
/// the count records define the levels (TRT is 0 then 1 within every subject, so both
/// levels come from the records alone, no Gaussian row exists), each record carries its own
/// index, and the data term scores each record at its own level. Before the count records
/// were level rows this model was refused as having no levels.
#[test]
fn count_records_define_and_carry_theta_levels() {
    let prep = load(
        "[parameters]\n  theta BASE(4.0, 0.01, 100.0)\n  theta EFF[TRT](0.0, -5.0, 5.0)\n\
         [covariates]\n  TRT categorical\n[count_model]\n  cmt = 1\n  \
         log_lambda = log(BASE) + EFF\n",
        "nonmem_anchor/count_pois.csv",
    );
    let model = &prep.parsed.model;
    // Two levels under the default sum-to-zero contrast: one free θ, the TRT = 1 level its
    // negative. Without the count records as level rows only TRT = 0 (the baseline) is a
    // level, and every record is scored at it.
    let blocks = &model.indiv_param_partials.theta_blocks.level_blocks;
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].record_cmts, vec![1], "the count endpoint reads the block");
    assert_eq!(model.theta_names, vec!["BASE", "EFF[TRT=0]"]);
    let theta = [4.0, 0.35];
    let s = &prep.population.subjects[0];
    let mut want = 0.0;
    for (k, r) in s.obs_records.iter().enumerate() {
        let ObsRecord::Count { count, .. } = r else { unreachable!() };
        // Records 0..3 are TRT = 0, records 3..6 TRT = 1.
        let eff = if k < 3 { theta[1] } else { -theta[1] };
        want -= count_log_pmf(4.0 * eff.exp(), None, f64::from(*count));
    }
    let got = ferx_model::categorical::discrete_subject_nll(model, s, &theta, &[]);
    assert!((got - want).abs() < 1e-10, "{got} vs {want}");
}

/// A count past `u32::MAX` — the largest a record carries — is outside the PMF's domain:
/// `count_log_pmf` returns `NaN` for both families rather than narrowing it to `u32::MAX`
/// (the negative binomial's cast saturated, so `2³²` was scored as `2³² − 1`), and
/// `Prediction::prob` returns `None`. `u32::MAX` itself is still a count, finite on both.
///
/// Mutations: drop the domain bound and the NB value at `2³²` equals the one at `2³² − 1`;
/// narrow in `prob` with `as` and it returns `Some`.
#[test]
fn counts_past_u32_max_are_out_of_domain_not_narrowed() {
    let max = f64::from(u32::MAX);
    for size in [None, Some(3.5)] {
        let at_max = count_log_pmf(4e9, size, max);
        assert!(at_max.is_finite(), "size {size:?}: P(Y = u32::MAX) = {at_max}");
        let past = count_log_pmf(4e9, size, max + 1.0);
        assert!(past.is_nan(), "size {size:?}: P(Y = 2³²) = {past}, not NaN");
    }
    let rate = Prediction::Rate { lambda: 4e9, size: Some(3.5) };
    assert!(rate.prob(u32::MAX as usize).is_some_and(|p| p.is_finite()));
    if let Ok(big) = usize::try_from(1u64 << 32) {
        assert_eq!(rate.prob(big), None, "a count past u32::MAX is out of range");
    }
}
