//! NONMEM anchor for the random-effect shape transforms (#1716).
//!
//! `boxcox(η, λ)` and `tdist(η, ν)` are Petersson et al. 2009's transforms as
//! they are hand-coded in `$PK` (and as Pharmpy generates them) — `ETATR = (EXP(ETA(1))**λ − 1)/λ`, the
//! third-order t series — so this is an ordinary anchored comparison against
//! the same arithmetic in NONMEM. Four control streams, all
//! `$EST METHOD=1 INTERACTION MAXEVAL=0 POSTHOC` on one multi-dose dataset
//! (`nonmem_anchor/eta_shape.csv`: 30 subjects, 100 at 0/12/24 h, 9
//! observations each, the later doses landing on residual drug):
//!
//! | stream | CL |
//! |---|---|
//! | `eta_shape_null` | `TVCL·exp(η)` — the null twin |
//! | `eta_shape_bc_pos` | `TVCL·exp(boxcox(η, +0.5))` |
//! | `eta_shape_bc_neg` | `TVCL·exp(boxcox(η, −0.5))` |
//! | `eta_shape_td` | `TVCL·exp(tdist(η, 5))` |
//!
//! Each shaped arm differs from the null by its transform alone. Three objects
//! are compared per arm: the FOCEI objective, both absolute and as the
//! **difference** the transform makes against the null (the convention of
//! `additive_covariate_nonmem_anchor`, whose dataset carries an absolute
//! offset between the engines; this one measured none), the posthoc η against
//! `.phi`, and `IPRED` per observation against the `$TABLE`. The η and `IPRED` comparisons are where a
//! wrong transform shows first: the EBE is the argmin of an objective the
//! transform enters at every record.
//!
//! The transform is live in every arm: the shaped objectives move by −1.39,
//! +13.27 and −0.72 against the null, and `the_shape_is_what_is_being_anchored`
//! states that as a test. John-Draper has no NONMEM anchor: spelled in
//! NM-TRAN as `((ABS(η)+1)**λ − 1)·(ABS(η)/η)/λ` its sign factor is `0/0` at η = 0, where NONMEM's inner
//! search starts. It is checked against external closed-form values and by
//! `Dual2`-vs-FD parity in `src/parser/eta_shape_tests.rs`.
//!
//! Run in `nonmem:7.6.0-anchor` (the outputs in `nonmem_anchor/results/`);
//! `nonmem:7.6.0-arm64` reproduces each objective to 3e-8. Fast (an
//! evaluation, no fit), so it runs on every PR.

use std::collections::HashMap;
use std::path::Path;

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::types::{FitOptions, FitResult, Population};
use ferx_core::{fit, read_nonmem_csv};

const DATA: &str = "nonmem_anchor/eta_shape.csv";

/// `OBJECTIVE FUNCTION VALUE` of each stream (`nonmem:7.6.0-anchor`).
const NM_OFV_NULL: f64 = -202.55964036923595;
const NM_OFV_BC_POS: f64 = -203.94545795400700;
const NM_OFV_BC_NEG: f64 = -189.28933134482733;
const NM_OFV_TD: f64 = -203.28275851948356;

/// The ferx model at the control streams' parameter vector, `CL` as given and
/// `extra` appended (an `[eta_shape]` block, say).
fn model(cl: &str, extra_theta: &str, extra: &str) -> String {
    format!(
        "
[parameters]
  theta TVCL(2.0, 0.01, 100.0)
  theta TVV(20.0, 0.1, 1000.0)
{extra_theta}
  omega ETA_CL ~ 0.2
  omega ETA_V ~ 0.1
  sigma PROP_ERR ~ 0.1 (sd)

[individual_parameters]
  CL = {cl}
  V = TVV * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
{extra}"
    )
}

fn null_model() -> String {
    model("TVCL * exp(ETA_CL)", "", "")
}
fn bc_model(lambda: f64) -> String {
    model(
        "TVCL * exp(boxcox(ETA_CL, LAMBDA))",
        &format!("  theta LAMBDA({lambda}, -3.0, 3.0, FIX)"),
        "",
    )
}
fn td_model() -> String {
    model(
        "TVCL * exp(tdist(ETA_CL, NU))",
        "  theta NU(5.0, 3.0, 100.0, FIX)",
        "",
    )
}

/// The ferx FOCEI evaluation at the declared parameters (`outer_maxiter = 0`
/// is `MAXEVAL=0`), with the population it ran on.
fn evaluate(src: &str) -> (FitResult, Population) {
    let parsed = parse_full_model(src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let result = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("the evaluation must run");
    assert!(result.ofv.is_finite(), "ferx OFV is not finite");
    (result, pop)
}

/// Rows of a committed NONMEM output, as `header → column` maps.
fn nonmem_rows(file: &str, skip: usize) -> Vec<HashMap<String, f64>> {
    let text = std::fs::read_to_string(Path::new("nonmem_anchor/results").join(file))
        .unwrap_or_else(|e| panic!("{file} must be readable: {e}"));
    let mut lines = text.lines().skip(skip);
    let header: Vec<String> = lines
        .next()
        .expect("a header line")
        .split_whitespace()
        .map(str::to_string)
        .collect();
    lines
        .map(|l| {
            header
                .iter()
                .cloned()
                .zip(l.split_whitespace().map(|v| v.parse().expect("a number")))
                .collect()
        })
        .collect()
}

/// Worst `|ferx − NONMEM|` over every subject's posthoc η (against `.phi`) and
/// every observation's `IPRED` (relative, against the `$TABLE`). Every value is
/// asserted finite before it is folded: `f64::max` drops a `NaN`.
fn worst_errors(result: &FitResult, pop: &Population, stem: &str) -> (f64, f64) {
    let phi = nonmem_rows(&format!("{stem}.phi"), 1);
    let tab = nonmem_rows(&format!("{stem}.tab"), 1);
    let (mut worst_eta, mut worst_ipred) = (0.0f64, 0.0f64);
    let mut n_obs = 0;
    for (s, subj) in result.subjects.iter().zip(&pop.subjects) {
        let id: f64 = s.id.parse().expect("numeric IDs");
        let row = phi.iter().find(|r| r["ID"] == id).expect("a .phi row");
        for k in 0..2 {
            let (got, want) = (s.eta[k], row[&format!("ETA({})", k + 1)]);
            assert!(got.is_finite(), "subject {id} η{k} = {got}");
            worst_eta = worst_eta.max((got - want).abs());
        }
        for (j, &t) in subj.obs_times.iter().enumerate() {
            let want = tab
                .iter()
                .find(|r| r["ID"] == id && (r["TIME"] - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("no $TABLE row for {id} at {t}"))["IPRED"];
            let got = s.ipred[j];
            assert!(got.is_finite(), "subject {id} IPRED at {t} = {got}");
            worst_ipred = worst_ipred.max((got - want).abs() / want.abs());
            n_obs += 1;
        }
    }
    assert_eq!(n_obs, 270, "every observation is compared");
    (worst_eta, worst_ipred)
}

/// One shaped arm against NONMEM: ΔOFV vs the null, posthoc η, `IPRED`. Also
/// asserts the evaluation ran on the analytic gradient, not an FD fallback —
/// the dual path is what a shaped ETA must not lose.
///
/// Unlike `additive_covariate_nonmem_anchor`'s dataset, this one shows no
/// absolute-objective offset between the engines, so the absolute OFV is
/// checked too, not only the difference.
///
/// Measured worst over the three arms (Linux aarch64 via `tools/linux-test.sh`
/// and macOS arm64 print the same digits): ΔOFV
/// 1.9e-7, absolute OFV 1.3e-7, posthoc η 2.7e-8, `IPRED` 4.3e-8 relative — the
/// null twin alone reads η 1.3e-8 / `IPRED` 2.5e-8, so the transform adds
/// nothing measurable. Bounds 2e-6 (OFV) / 3e-7 (η) / 5e-7 (`IPRED`): ~10×
/// headroom, and six orders below the smallest shape effect anchored (ΔOFV
/// 0.72).
fn check_arm(src: &str, stem: &str, nm_ofv: f64) {
    let (null, _) = evaluate(&null_model());
    let (arm, pop) = evaluate(src);
    for w in &arm.warnings {
        assert!(!w.contains("finite-difference"), "{stem}: FD fallback: {w}");
    }
    let d_ferx = arm.ofv - null.ofv;
    let d_nm = nm_ofv - NM_OFV_NULL;
    let (eta, ipred) = worst_errors(&arm, &pop, stem);
    eprintln!(
        "{stem}: ΔOFV ferx {d_ferx:.9} NONMEM {d_nm:.9} (|Δ| {:.3e}); abs OFV ferx {:.9} \
         NONMEM {nm_ofv:.9}; worst η {eta:.3e}, IPRED {ipred:.3e}",
        (d_ferx - d_nm).abs(),
        arm.ofv
    );
    assert!(
        (d_ferx - d_nm).abs() < 2e-6,
        "{stem}: ΔOFV {d_ferx} vs {d_nm}"
    );
    assert!(
        (arm.ofv - nm_ofv).abs() < 2e-6,
        "{stem}: OFV {} vs {nm_ofv}",
        arm.ofv
    );
    assert!(eta < 3e-7, "{stem}: worst η error {eta:e}");
    assert!(ipred < 5e-7, "{stem}: worst IPRED error {ipred:e}");
}

#[test]
fn box_cox_positive_lambda_matches_nonmem() {
    check_arm(&bc_model(0.5), "eta_shape_bc_pos", NM_OFV_BC_POS);
}

#[test]
fn box_cox_negative_lambda_matches_nonmem() {
    check_arm(&bc_model(-0.5), "eta_shape_bc_neg", NM_OFV_BC_NEG);
}

#[test]
fn t_distribution_matches_nonmem() {
    check_arm(&td_model(), "eta_shape_td", NM_OFV_TD);
}

/// The `[eta_shape]` spelling evaluates the same objective and EBEs as the
/// inline one, against the same NONMEM stream.
#[test]
fn the_eta_shape_block_matches_nonmem() {
    let src = model(
        "TVCL * exp(ETA_CL)",
        "  theta LAMBDA(0.5, -3.0, 3.0, FIX)",
        "\n[eta_shape]\n  ETA_CL ~ boxcox(LAMBDA)\n",
    );
    check_arm(&src, "eta_shape_bc_pos", NM_OFV_BC_POS);
}

/// The null twin itself: η and `IPRED` against NONMEM, so the arms' agreement
/// is not an artefact of a shared offset in the base model.
#[test]
fn the_null_twin_matches_nonmem() {
    let (r, pop) = evaluate(&null_model());
    let (eta, ipred) = worst_errors(&r, &pop, "eta_shape_null");
    eprintln!(
        "null: abs OFV ferx {:.9} NONMEM {NM_OFV_NULL:.9}; η {eta:.3e}, IPRED {ipred:.3e}",
        r.ofv
    );
    assert!(
        eta < 3e-7 && ipred < 5e-7 && (r.ofv - NM_OFV_NULL).abs() < 2e-6,
        "null: η {eta:e}, IPRED {ipred:e}"
    );
}

/// Non-degeneracy: a ferx model that ignored the transform (the null model) is
/// compared against each shaped NONMEM stream and must **fail** the bound the
/// arms pass — otherwise the anchors would be satisfied by an implementation
/// that never applies the shape.
#[test]
fn the_shape_is_what_is_being_anchored() {
    let (null, pop) = evaluate(&null_model());
    for (stem, nm) in [
        ("eta_shape_bc_pos", NM_OFV_BC_POS),
        ("eta_shape_bc_neg", NM_OFV_BC_NEG),
        ("eta_shape_td", NM_OFV_TD),
    ] {
        let d = nm - NM_OFV_NULL;
        assert!(
            d.abs() > 0.5,
            "{stem}: the shape moves NONMEM's objective by {d}"
        );
        let (eta, _) = worst_errors(&null, &pop, stem);
        assert!(
            eta > 1e-2,
            "{stem}: an unshaped model is {eta:e} from the shaped EBEs"
        );
    }
}

/// SAEM runs on a shaped model: the shaped ETA has no mu-reference, so its
/// typical value and the free shape θ go to the numerical M-step. A short run
/// (not to convergence) must finish with a finite objective and move λ off its
/// start, which shows the shape θ is estimated rather than held.
#[test]
fn saem_estimates_a_free_shape_theta() {
    use ferx_core::types::EstimationMethod;
    let src = model(
        "TVCL * exp(boxcox(ETA_CL, LAMBDA))",
        "  theta LAMBDA(0.01, -3.0, 3.0)",
        "",
    );
    let parsed = parse_full_model(&src).expect("the ferx model must parse");
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("the dataset must load");
    let options = FitOptions {
        method: EstimationMethod::Saem,
        saem_n_exploration: 30,
        saem_n_convergence: 20,
        run_covariance_step: false,
        saem_seed: Some(1716),
        ..parsed.fit_options.clone()
    };
    let r = fit(&parsed.model, &pop, &parsed.model.default_params, &options)
        .expect("SAEM must run on a shaped model");
    assert!(r.ofv.is_finite(), "SAEM OFV {}", r.ofv);
    let i = r.theta_names.iter().position(|n| n == "LAMBDA").unwrap();
    eprintln!("SAEM λ after 50 iterations: {}", r.theta[i]);
    assert!(
        r.theta[i].is_finite() && (r.theta[i] - 0.01).abs() > 1e-3,
        "λ = {}",
        r.theta[i]
    );
}
