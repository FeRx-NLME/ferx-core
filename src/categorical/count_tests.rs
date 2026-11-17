//! Tier-1 tests for the count endpoint (#2877).

use super::*;
use crate::categorical::exact_eta_hessian_tests::load;
use crate::sens::dual2::Dual2;
use crate::stats::special::count_log_pmf;
use crate::types::SimOutcome;

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
        assert!(
            (pmf - want).abs() < 1e-12,
            "{fam:?} count_log_pmf {pmf} vs {want}"
        );
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
        assert!(
            (p - big).abs() < 1e-6,
            "k={k}: Poisson {p} vs NB(r=1e9) {big}"
        );
        assert!(
            (p - small).abs() > 0.05,
            "k={k}: NB(r=0.8) {small} too close to {p}"
        );
    }
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
        assert!(
            (got - pmf).abs() < 1e-9,
            "k={k}: {got} vs count_log_pmf {pmf}"
        );
    }
}

/// Every point where the law is undefined is declined, never scored: a non-positive or
/// non-finite `lambda`, a non-finite `log λ` or one whose exponential overflows, and a
/// non-positive or non-finite size on a negative binomial. A Poisson ignores the size slot.
#[test]
fn undefined_laws_are_declined() {
    for lam in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            count_record_nll_g(P, LIN, lam, 0.0, 2).is_none(),
            "λ = {lam}"
        );
    }
    for ll in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 800.0] {
        assert!(
            count_record_nll_g(P, LOG, ll, 0.0, 2).is_none(),
            "log λ = {ll}"
        );
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
    for (fam, sc, m, r) in [
        (P, LIN, 2.3, 0.0),
        (P, LOG, 0.4, 0.0),
        (NB, LIN, 2.3, 1.7),
        (NB, LOG, 1.1, 6.0),
    ] {
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
                assert_eq!(
                    d.grad[1], 0.0,
                    "a Poisson record must not read the size slot"
                );
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
    let n_count = s
        .obs_records
        .iter()
        .filter(|r| matches!(r, ObsRecord::Count { .. }))
        .count();
    assert_eq!(n_count, 6, "every row is a count record");
    let wt = s.covariates["WT"];
    let trt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut want = 0.0;
    let mut lams = Vec::new();
    for (r, t) in s.obs_records.iter().zip(trt) {
        let ObsRecord::Count { count, .. } = r else {
            unreachable!()
        };
        let lam =
            theta[0] * (theta[1] * (wt / 70.0 - 1.0) + theta[2] * t * eta[1].exp() + eta[0]).exp();
        want -= count_log_pmf(lam, Some(theta[3]), f64::from(*count));
        lams.push(lam);
    }
    let got = super::super::discrete_subject_nll(model, s, theta, &eta);
    assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    // The record pair straddling TRT must differ, or the snapshot test above is vacuous.
    assert!((lams[2] - lams[3]).abs() > 0.1, "{lams:?}");

    let mut preds = Vec::new();
    predict_count(model, s, theta, &eta, &mut preds);
    let diags = count_diagnostics(model, s, theta, &eta);
    assert_eq!(preds.len(), 6);
    assert_eq!(diags.len(), 6);
    for ((p, d), lam) in preds.iter().zip(&diags).zip(&lams) {
        let Prediction::Rate { lambda, size } = p.prediction else {
            panic!("{:?}", p.prediction)
        };
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
    let nll = |s: &Subject, e: &[f64]| super::super::discrete_subject_nll(model, s, theta, e);
    let (mut wg, mut wh) = (0.0f64, 0.0f64);
    for s in prep.population.subjects.iter().take(5) {
        let (v, g) = super::super::discrete_subject_eta_grad(model, s, theta, &eta)
            .expect("the count term records");
        assert!((v - nll(s, &eta)).abs() < 1e-9);
        let (_, h) = super::super::discrete_subject_eta_hessian(model, s, theta, &eta)
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
        assert!(
            h[(0, 1)].abs() > 1e-3,
            "the off-diagonal block must be live"
        );
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
    assert!(super::all_count_endpoints_analytic(&model));
    if let Some(EndpointLikelihood::Count { param_program, .. }) = model.endpoints.get_mut(&1) {
        *param_program = None;
    }
    assert!(!super::all_count_endpoints_analytic(&model));
    assert!(!crate::categorical::all_discrete_endpoints_analytic(&model));
    let s = &prep.population.subjects[0];
    let theta = model.default_params.theta.clone();
    assert!(
        super::super::discrete_subject_eta_grad(&model, s, &theta, &[0.0, 0.0]).is_none(),
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
    let v = super::super::discrete_subject_nll(model, s, theta, &[0.0, 0.0]);
    assert!(v >= SUBJECT_SENTINEL_NLL && v.is_finite(), "{v}");
    assert!(super::super::discrete_subject_eta_grad(model, s, theta, &[0.0, 0.0]).is_none());
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
        let xs: Vec<f64> = (0..n)
            .map(|_| f64::from(draw_count(4.0, size, &mut rng).unwrap()))
            .collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        eprintln!("size {size:?}: mean {mean:.4}, var {var:.4} (want {var_want})");
        assert!((mean - 4.0).abs() < 0.05, "mean {mean}");
        assert!(
            (var / var_want - 1.0).abs() < band,
            "var {var} vs {var_want}"
        );
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
    simulate_count(
        model,
        s,
        theta,
        &[0.0, 0.0],
        0,
        0,
        &mut rng,
        &mut out,
        &mut warn,
    );
    assert_eq!(out.len(), 6);
    assert!(warn.is_empty(), "{warn:?}");
    let mut preds = Vec::new();
    predict_count(model, s, theta, &[0.0, 0.0], &mut preds);
    for (row, p) in out.iter().zip(&preds) {
        assert!(matches!(row.outcome, SimOutcome::Count { .. }));
        assert_eq!(row.time, p.time);
        let Prediction::Rate { lambda, .. } = p.prediction else {
            unreachable!()
        };
        assert_eq!(row.ipred, lambda);
    }
    let mut bad = theta.clone();
    bad[3] = -1.0; // size r < 0: no law
    let (mut out, mut warn) = (Vec::new(), Vec::new());
    simulate_count(
        model,
        s,
        &bad,
        &[0.0, 0.0],
        0,
        0,
        &mut rng,
        &mut out,
        &mut warn,
    );
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
    assert!(
        e.contains("the mean is given twice (`lambda` and `log_lambda`)"),
        "{e}"
    );
    let e = parse_err("  cmt = 1\n  lambda = B\n  lambda = B + E\n");
    assert!(e.contains("`lambda` is given twice"), "{e}");
    assert!(!e.contains("log_lambda` and"), "{e}");
    let e = parse_err("  cmt = 1\n  type = zip\n  lambda = B * exp(E)\n");
    assert!(e.contains("type `zip` is not supported"), "{e}");
    assert!(e.contains("`poisson` (the default) and `negbin`"), "{e}");
    let e = parse_err("  cmt = 1\n  mu = B * exp(E)\n");
    assert!(e.contains("unknown key `mu`"), "{e}");
    assert!(
        e.contains("valid keys: cmt, type, lambda, log_lambda, r"),
        "{e}"
    );
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
    for (sa, sb) in a
        .population
        .subjects
        .iter()
        .zip(&b.population.subjects)
        .take(4)
    {
        let va = super::super::discrete_subject_nll(&a.parsed.model, sa, th, &[0.3]);
        let vb = super::super::discrete_subject_nll(&b.parsed.model, sb, th, &[0.3]);
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
        let idx = |v: Option<f64>| {
            v.map(|v| HashMap::from([("IDX".to_string(), v)]))
                .unwrap_or_default()
        };
        let mut s = Subject {
            id: "1".into(),
            covariates: idx(Some(1.0)),
            endpoint_covariates: vec![idx(Some(1.0)), idx(second)],
            ..Default::default()
        };
        s.obs_records = [0.0, 4.0]
            .map(|t| ObsRecord::Count {
                time: t,
                raw_time: t,
                count: 2,
                cmt: 1,
            })
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
    assert!(
        refused[0].1.contains("is 3 on subject `1` at TIME 4"),
        "{}",
        refused[0].1
    );
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
    assert!(
        count_log_pmf(0.0, Some(-1.0), 0.0).is_nan(),
        "a bad size stays undefined"
    );
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
    simulate_count(
        model,
        s,
        th,
        &[0.0, 0.0],
        0,
        0,
        &mut rng,
        &mut out,
        &mut warn,
    );
    assert!(warn.is_empty(), "{warn:?}");
    assert_eq!(out.len(), 6);
    assert!(out
        .iter()
        .all(|r| matches!(r.outcome, SimOutcome::Count { count: 0 })));
}
