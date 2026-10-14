//! NONMEM `$COVARIANCE MATRIX=S` / default (`R⁻¹SR⁻¹`) anchor for a **mixture**
//! covariance step (#1956).
//!
//! Before #1956 the score cross-product `S` scored a mixture subject through the
//! single-population objective, under the default `MIXNUM = 1` and the base Ω/Σ: the
//! class-2-only θ and the mixing θ had identically-zero score columns, so
//! `covariance_method = s` was refused as singular and `rsr` silently lost both
//! coordinates. On `mixture_iv` at the NONMEM finals the old S was
//! `diag ≈ [2454, 0, 5930, 0]` with eigenvalues `[7975, 409, ~0, 0]` (macOS arm64).
//!
//! The model and data are the existing FOCEI mixture anchor's
//! (`tests/nonmem/mixture_iv_cov.ctl`, `mixture_iv.csv`): two classes differing in CL,
//! constant mixing, Ω and Σ FIXed, so the four free coordinates are TVCL1, TVCL2, V
//! and the mixing fraction. ferx spells the mixing fraction as NONMEM does —
//! `p(1) = P1`, a θ on the probability scale — so the SEs compare without a delta map.
//!
//! Both engines evaluate one fixed point, the `mixture_iv_cov.ext` finals at their
//! six printed significant figures: NONMEM with `MAXEVAL=0`, ferx with
//! `outer_maxiter = 0`. Three NONMEM 7.5.1 runs share that point (`tests/nonmem/`),
//! all at OBJ 301.58201838:
//!
//! | stem                 | `$COVARIANCE`       | ferx `covariance_method` |
//! |----------------------|---------------------|--------------------------|
//! | `mixture_iv_cov_s`   | `MATRIX=S`          | `s`                      |
//! | `mixture_iv_cov_rsr` | default (`R⁻¹SR⁻¹`) | `rsr`                    |
//! | `mixture_iv_cov_r0`  | `MATRIX=R`          | `r` (control)            |
//!
//! ## ferx's `S` is not NONMEM's `S` — on any model, not only a mixture
//!
//! The two engines differentiate different per-subject objectives. NONMEM's score is
//! the **total** derivative of `−2LLᵢ(x)`, the EBE η̂ᵢ(x) re-estimated and the FOCEI
//! `log|H̃|` recomputed at every perturbed `x`. ferx's (`subject_nll_pop_grad`, shared
//! by the single-population path this fix reuses per class) holds the mode and its
//! prediction Jacobian at `x̂`: the EBE-response term is the documented zero of
//! `subject_eta_response_correction` (#338 Part A, the ω/σ blocks deferred to #1817).
//! At the MLE both conventions are unbiased, so the gap is a finite-sample one — but
//! on 30 subjects it is 14 % on `SE(TVCL2)`.
//!
//! `tests/nonmem/mixture_iv_cov_score_oracle.py` computes the per-subject mixture
//! `−2LLᵢ` in closed form, outside both engines, and forms `S = Σ gᵢgᵢᵀ` from central
//! differences under each convention. It reproduces NONMEM's OBJ to 4e-8, and its two
//! conventions land one on each engine (macOS arm64):
//!
//! | `S` SE    | NONMEM `MATRIX=S` | oracle, η̂ re-estimated | oracle, mode held | ferx `s`  |
//! |-----------|-------------------|-------------------------|-------------------|-----------|
//! | TVCL1     | 0.132375          | 0.132374                | 0.134032          | 0.134032  |
//! | TVCL2     | 0.293660          | 0.293660                | 0.252108          | 0.252108  |
//! | V         | 0.213639          | 0.213639                | 0.211925          | 0.211925  |
//! | P1        | 0.108812          | 0.108812                | 0.122599          | 0.122599  |
//!
//! The same oracle's `--single` run shows the gap on the one-population twin of this
//! model (`CL = TVCL·exp(η)`, same data): η̂ re-estimated `SE(TVCL) = 0.053787`, mode
//! held `0.062615`, ferx `0.062615` — a pre-existing single-population convention, not
//! a mixture defect. So `s`/`rsr` are anchored here on the oracle's mode-held column,
//! and the oracle's re-estimated column is checked against NONMEM, which pins *which*
//! convention each engine implements. When ferx's score gains the EBE response, the
//! straddle assertion below reddens and the anchor moves to NONMEM's `.ext` directly.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, CovarianceMethod, FitOptions, FitResult};
use std::path::Path;

/// `mixture_iv_cov.ext` `-1000000000` row (NONMEM 7.5.1, FOCEI), as printed.
const NM_TVCL1: f64 = 0.980833;
const NM_TVCL2: f64 = 2.84198;
const NM_TVV: f64 = 9.97859;
const NM_P1: f64 = 0.471970;

/// `mixture_iv_cov_score_oracle.py` at the point above (TVCL1, TVCL2, V, P1), numpy
/// on macOS arm64. `R` is the central-difference Hessian of the total objective;
/// `TOTAL` re-estimates η̂ at every stencil point (NONMEM's convention), `HELD` holds
/// the mode and its prediction Jacobian at `x̂` (ferx's).
const ORACLE_R: [f64; 4] = [0.10580139, 0.25430878, 0.18610565, 0.10257858];
const ORACLE_S_TOTAL: [f64; 4] = [0.13237448, 0.29366024, 0.2136393, 0.1088116];
const ORACLE_RSR_TOTAL: [f64; 4] = [0.08546294, 0.22427684, 0.16875011, 0.10125359];
const ORACLE_S_HELD: [f64; 4] = [0.13403189, 0.25210754, 0.21192505, 0.12259852];
const ORACLE_RSR_HELD: [f64; 4] = [0.08741343, 0.30434308, 0.17043115, 0.11422785];

const NAMES: [&str; 4] = ["TVCL1", "TVCL2", "V", "P1"];

fn model_src() -> String {
    format!(
        r"
[parameters]
  theta TVCL1({NM_TVCL1}, 0.01, 100.0)
  theta TVCL2({NM_TVCL2}, 0.01, 100.0)
  theta TVV({NM_TVV}, 0.1, 1000.0)
  theta P1({NM_P1}, 0.001, 0.999)
  omega ETA_CL ~ 0.09 FIX
  sigma EPS ~ 0.04 FIX

[mixture]
  nsub = 2
  p(1) = P1

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)
"
    )
}

fn eval_only_fit(method: CovarianceMethod) -> FitResult {
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/mixture_iv.csv"),
        Some(&["WT"]),
        None,
    )
    .unwrap();
    let model = parse_model_string(&model_src()).unwrap();
    let opts = FitOptions {
        interaction: true, // NONMEM METHOD=1 INTER
        outer_maxiter: 0,  // NONMEM MAXEVAL=0
        covariance_method: method,
        covariance_method_set: true,
        ..Default::default()
    };
    fit(&model, &pop, &model.default_params, &opts).expect("eval-only mixture fit Ok")
}

fn theta_ses(res: &FitResult, label: &str) -> Vec<f64> {
    let se = res
        .se_theta
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: no θ SEs; warnings: {:?}", res.warnings))
        .clone();
    assert_eq!(se.len(), 4, "{label}: four θ");
    for (j, s) in se.iter().enumerate() {
        assert!(
            s.is_finite() && *s > 0.0,
            "{label}: SE θ{} = {s}; warnings: {:?}",
            j + 1,
            res.warnings
        );
    }
    se
}

/// The `-1000000001` (standard error) row of a NONMEM `.ext` file: θ1..θ4.
fn nonmem_theta_ses(stem: &str) -> Vec<f64> {
    let path = format!("tests/nonmem/{stem}.ext");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let row = text
        .lines()
        .find(|l| l.split_whitespace().next() == Some("-1000000001"))
        .unwrap_or_else(|| panic!("{path}: no -1000000001 (SE) row"));
    let se: Vec<f64> = row
        .split_whitespace()
        .skip(1)
        .take(4)
        .map(|v| v.parse::<f64>().unwrap())
        .collect();
    assert!(
        se.len() == 4 && se.iter().all(|s| s.is_finite() && *s > 0.0),
        "{path}: SE row {se:?}"
    );
    se
}

/// Worst relative gap `|got − want| / want`, every entry asserted finite first so a
/// `NaN` cannot hide behind `f64::max`.
fn worst_rel(label: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f64;
    for ((name, g), w) in NAMES.iter().zip(got).zip(want) {
        assert!(g.is_finite() && w.is_finite(), "{label} {name}: {g} vs {w}");
        let rel = (g - w).abs() / w;
        eprintln!("{label} {name}: {g:.8e} vs {w:.8e} rel {rel:.3e}");
        worst = worst.max(rel);
    }
    eprintln!("{label}: worst rel {worst:.3e}");
    worst
}

/// Every free θ carries information under `s` and `rsr` — the property #1956 broke.
/// Runs on every PR: one objective evaluation plus the covariance step on 30 subjects.
#[test]
fn mixture_s_and_rsr_report_every_theta_se() {
    for (method, label) in [
        (CovarianceMethod::CrossProduct, "s"),
        (CovarianceMethod::Sandwich, "rsr"),
    ] {
        let res = eval_only_fit(method);
        let se = theta_ses(&res, label);
        assert!(
            res.warnings.iter().all(|w| !w.contains("singular")),
            "{label}: {:?}",
            res.warnings
        );
        eprintln!("ferx {label} SEs (TVCL1, TVCL2, V, P1): {se:?}");
    }
}

/// The `MATRIX=R` control: the same point, the same objective, ferx's `r` against
/// NONMEM's. It separates an `S` disagreement from an `R` one, since `rsr` inherits
/// both.
///
/// Measured worst 4.49e-3 on TVCL1 (the other three ≤ 4.9e-4), macOS arm64. The
/// 0.45 % sits on NONMEM's side: the oracle's `R` agrees with ferx's to 2.4e-5 on
/// TVCL1 (0.10580139 vs 0.10580397), and NONMEM's own `MATRIX=R` SE(TVCL1) moves from
/// 0.105642 at its full-precision MLE (`mixture_iv_cov.ext`) to 0.105331 at the
/// six-figure point here — a 0.3 % swing from rounding the point, i.e. its FD `R`
/// noise floor. Bound 1 %: 2.2× the measured worst.
#[test]
fn mixture_r_se_matches_nonmem_r() {
    let se = theta_ses(&eval_only_fit(CovarianceMethod::Hessian), "r");
    let worst = worst_rel(
        "ferx r vs NONMEM MATRIX=R",
        &se,
        &nonmem_theta_ses("mixture_iv_cov_r0"),
    );
    assert!(
        worst < 1e-2,
        "ferx r vs NONMEM MATRIX=R: worst rel {worst:.3e}"
    );
}

/// ferx's `s` / `rsr` against the third-way oracle's mode-held convention, and the
/// oracle's re-estimated convention against NONMEM's `MATRIX=S` / `R⁻¹SR⁻¹` — see the
/// module doc for why the anchor is split this way.
///
/// Measured worst relative gaps, macOS arm64, and the bound each gets:
///
/// | comparison                        | worst            | bound | why that bound                       |
/// |-----------------------------------|------------------|-------|--------------------------------------|
/// | ferx `s` vs oracle held           | 2.7e-7 (TVCL1)   | 1e-5  | 37×; no `R` in `S⁻¹`                 |
/// | ferx `r` vs oracle                | 2.1e-4 (V)       | 1e-3  | 4.7×; the two FD `R` stencils        |
/// | ferx `rsr` vs oracle held         | 4.0e-4 (V)       | 1e-3  | 2.5×; inherits the `R` row above     |
/// | NONMEM `S` vs oracle total        | 3.9e-6 (TVCL1)   | 2e-5  | 5×; NONMEM prints 6 figures (≤ 4e-6) |
/// | NONMEM `R⁻¹SR⁻¹` vs oracle total  | 8.9e-3 (TVCL1)   | 2e-2  | 2.2×; NONMEM's FD `R` (see `r` test) |
/// | oracle held vs total, `S`         | 1.4e-1 (TVCL2)   | > 0.1 | the straddle, ≥ 5000× the `s` bound  |
/// | oracle held vs total, `R⁻¹SR⁻¹`   | 3.6e-1 (TVCL2)   | > 0.1 | the straddle, ≥ 5× the NONMEM bound  |
#[test]
fn mixture_s_rsr_ses_match_score_oracle_and_nonmem() {
    // ferx implements the mode-held convention, to the oracle's FD accuracy.
    let s = theta_ses(&eval_only_fit(CovarianceMethod::CrossProduct), "s");
    let rsr = theta_ses(&eval_only_fit(CovarianceMethod::Sandwich), "rsr");
    let r = theta_ses(&eval_only_fit(CovarianceMethod::Hessian), "r");
    let w_s = worst_rel("ferx s vs oracle held", &s, &ORACLE_S_HELD);
    let w_r = worst_rel("ferx r vs oracle", &r, &ORACLE_R);
    let w_rsr = worst_rel("ferx rsr vs oracle held", &rsr, &ORACLE_RSR_HELD);
    assert!(w_s < 1e-5, "ferx s vs oracle (mode held): {w_s:.3e}");
    assert!(w_r < 1e-3, "ferx r vs oracle: {w_r:.3e}");
    assert!(w_rsr < 1e-3, "ferx rsr vs oracle (mode held): {w_rsr:.3e}");

    // NONMEM implements the re-estimated (total-derivative) convention.
    let w_nm_s = worst_rel(
        "NONMEM S vs oracle total",
        &nonmem_theta_ses("mixture_iv_cov_s"),
        &ORACLE_S_TOTAL,
    );
    let w_nm_rsr = worst_rel(
        "NONMEM RSR vs oracle total",
        &nonmem_theta_ses("mixture_iv_cov_rsr"),
        &ORACLE_RSR_TOTAL,
    );
    assert!(
        w_nm_s < 2e-5,
        "NONMEM MATRIX=S vs oracle (η̂ re-estimated): {w_nm_s:.3e}"
    );
    assert!(
        w_nm_rsr < 2e-2,
        "NONMEM R⁻¹SR⁻¹ vs oracle (η̂ re-estimated): {w_nm_rsr:.3e}"
    );

    // The straddle: the two conventions differ by far more than any agreement above,
    // so passing on one column says nothing about the other. If ferx's score gains
    // the EBE response, `w_s` fails first — move the s/rsr anchor to NONMEM's `.ext`.
    let w_conv_s = worst_rel("oracle S held vs total", &ORACLE_S_HELD, &ORACLE_S_TOTAL);
    let w_conv_rsr = worst_rel(
        "oracle RSR held vs total",
        &ORACLE_RSR_HELD,
        &ORACLE_RSR_TOTAL,
    );
    assert!(
        w_conv_s > 0.1 && w_conv_rsr > 0.1,
        "the conventions must stay distinguishable: S {w_conv_s:.3e}, RSR {w_conv_rsr:.3e}"
    );
}
