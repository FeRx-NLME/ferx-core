//! NONMEM anchor for fixed-effects-only (naive-pooled) fits — #989.
//!
//! # The reference run
//!
//! `nonmem_anchor/results/zero_omega_pooled.{ctl,lst,ext}`, NONMEM 7.6.0,
//! `examples/one_cpt_iv_pooled.ferx` transcribed to ADVAN1 TRANS2 against the
//! repo's own `data/one_cpt_iv.csv`:
//!
//! ```text
//! $PK    CL = THETA(1)*EXP(ETA(1))   V = THETA(2)   S1 = V
//! $ERROR Y  = F*(1 + EPS(1))
//! $OMEGA 0 FIX
//! $EST   METHOD=0 MAXEVAL=9999 SIGDIGITS=4
//! ```
//!
//! **`$OMEGA 0 FIX`, not an omitted `$OMEGA`.** Leaving `$OMEGA` out entirely is
//! the obvious transcription and it is wrong: NM-TRAN then reports
//! `(WARNING 1) NM-TRAN INFERS THAT THE DATA ARE SINGLE-SUBJECT`, drops the ID
//! grouping, and — because each subject's TIME restarts at 0 — aborts the run
//! with `TIME DATA ITEM IS LESS THAN PREVIOUS TIME DATA ITEM`. A degenerate
//! `ETA(1)` with variance fixed at 0 keeps NONMEM in population mode with
//! per-subject dose bookkeeping, which is what ferx does at `n_eta = 0`.
//!
//! # Reference values (`.ext` final row)
//!
//! ```text
//! THETA1 4.84070E+00   THETA2 5.28324E+01   SIGMA(1,1) 1.66323E-02
//! OBJ (without constant) -269.63700440359776
//! ```
//!
//! Standard errors are checked too, but only under `covariance_method = rsr`:
//! NONMEM's `$COVARIANCE` default is the `RSR` sandwich while ferx's default is
//! `r`, so the two disagree **by design**. On a naive-pooled model that is not a
//! cosmetic difference — the model ignores within-subject correlation on purpose,
//! so the sandwich runs about twice the naive inverse-Hessian (SE TVCL 0.166 vs
//! 0.078). Comparing the two defaults would look like a factor-of-two bug.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{fit, read_nonmem_csv, FitOptions};
use std::path::Path;

const MODEL: &str = "examples/one_cpt_iv_pooled.ferx";
const DATA: &str = "data/one_cpt_iv.csv";

/// NONMEM `OBJECTIVE FUNCTION VALUE WITHOUT CONSTANT` — the convention ferx's
/// `ofv` uses (`Σ IWRES² + Σ log Var`, no `n·log 2π`).
const NM_OBJV: f64 = -269.63700440359776;
const NM_TVCL: f64 = 4.84070;
const NM_TVV: f64 = 52.8324;
/// `sqrt(1.66323E-02)` — ferx stores the proportional sigma on the SD scale.
const NM_SIGMA_SD: f64 = 0.12896627466124622;

/// Standard errors from the `.ext` `-1000000001` row, which is NONMEM's own
/// `$COVARIANCE` default — the `RSR` sandwich. `.cov` agrees: `sqrt(2.76552E-02)`
/// and `sqrt(3.09835E+00)`.
const NM_SE_TVCL: f64 = 0.166298;
const NM_SE_TVV: f64 = 1.76021;
/// `-1000000005` row: the SD-scale sigma SE, matching how ferx reports it.
const NM_SE_SIGMA_SD: f64 = 0.0182395;

fn anchor_opts(outer_maxiter: usize) -> FitOptions {
    FitOptions {
        outer_maxiter,
        run_covariance_step: false,
        ..Default::default()
    }
}

/// Evaluate ferx's objective **at NONMEM's optimum** and require it to reproduce
/// NONMEM's OBJ.
///
/// This is the load-bearing half of the anchor: it compares the two objective
/// *functions* at one shared point, so it cannot be satisfied by an optimizer
/// that happens to land somewhere similar. `outer_maxiter = 0` is evaluation-only
/// (NONMEM `MAXEVAL=0`), so it returns immediately and needs no `slow-tests` gate.
///
/// With `n_eta = 0` there is no inner EBE solve and no `log|Ω|` term, so the two
/// objectives are the same closed form and the agreement is exact to print
/// precision — hence a far tighter tolerance than the usual 0.5 anchor band.
#[test]
fn ofv_at_nonmem_optimum_matches_nonmem_objv() {
    let parsed = parse_full_model_file(Path::new(MODEL)).expect("pooled example must parse");
    let model = parsed.model;
    assert_eq!(
        model.n_eta, 0,
        "the anchor model must carry no random effects"
    );

    let mut params = model.default_params.clone();
    params.theta[0] = NM_TVCL;
    params.theta[1] = NM_TVV;
    params.sigma.values[0] = NM_SIGMA_SD;

    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("anchor data must read");
    let res =
        fit(&model, &pop, &params, &anchor_opts(0)).expect("evaluation-only fit must return Ok");

    // Observed delta at the time of writing: 3.7e-8 (ferx -269.63700436625965).
    // That residual is NONMEM's own `SIGDIGITS=4` print precision on the θ/σ we
    // feed back in, not a difference between the two objectives. The band is set
    // at 1e-3 — 500x tighter than the usual 0.5 anchor tolerance, with ample
    // headroom for cross-platform libm variation in the closed-form solution,
    // since this compares against a hard-coded constant rather than another ferx
    // run. If this ever drifts past 1e-3, the objective changed; do not widen it.
    //
    // The tight band is not decoration: perturbing `NM_TVCL` by 1% moves this OFV
    // by 0.458, which the conventional 0.5 anchor tolerance would have accepted.
    // On a fixed-effects-only model the objective is flat enough near the optimum
    // that a 0.5 band cannot distinguish a converged fit from a 1%-wrong one.
    assert!(
        (res.ofv - NM_OBJV).abs() < 1e-3,
        "ferx OFV {} vs NONMEM OBJ {} (delta {:.3e})",
        res.ofv,
        NM_OBJV,
        (res.ofv - NM_OBJV).abs()
    );
}

/// Fit to convergence from the model file's own starting values and require the
/// parameters to land on NONMEM's.
///
/// Slow-gated: this is a real convergence loop. The evaluation-only test above is
/// what runs on every PR.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn converged_estimates_match_nonmem_naive_pooled_run() {
    let parsed = parse_full_model_file(Path::new(MODEL)).expect("pooled example must parse");
    let model = parsed.model;
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("anchor data must read");
    let res = fit(&model, &pop, &model.default_params, &anchor_opts(300))
        .expect("naive-pooled fit must return Ok");

    assert!(res.converged, "fit must converge");
    assert!(
        (res.ofv - NM_OBJV).abs() < 1e-3,
        "OFV {} vs NONMEM {}",
        res.ofv,
        NM_OBJV
    );
    // NONMEM ran SIGDIGITS=4, so it reports five figures; compare on relative
    // error at that precision rather than an absolute band that would mean
    // something different for TVCL (~5) than for TVV (~53).
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    assert!(
        rel(res.theta[0], NM_TVCL) < 1e-3,
        "TVCL {} vs NONMEM {}",
        res.theta[0],
        NM_TVCL
    );
    assert!(
        rel(res.theta[1], NM_TVV) < 1e-3,
        "TVV {} vs NONMEM {}",
        res.theta[1],
        NM_TVV
    );
    assert!(
        rel(res.sigma[0], NM_SIGMA_SD) < 1e-3,
        "PROP_ERR (sd) {} vs NONMEM {}",
        res.sigma[0],
        NM_SIGMA_SD
    );
}

/// Run the **covariance step** at `n_eta = 0` and require its standard errors to
/// reproduce NONMEM's.
///
/// The two tests above both disable the covariance step, so without this one the
/// whole SE path is unexercised at `n_eta = 0` — and the SE table in
/// `docs/faq.qmd` would be a documented numeric claim with nothing behind it.
/// That path is not trivially safe here: the packed optimizer vector carries **no**
/// Omega coordinates at `n_eta = 0`, so the FD-of-OFV Hessian, the score
/// cross-product and the eigen-floor inverse all run on a θ/σ-only parameter
/// space that no Gaussian model could reach before #989.
///
/// `covariance_method = rsr` because that — not ferx's default `r` — is what
/// NONMEM's `$COVARIANCE` computes. On a naive-pooled model the distinction is
/// not cosmetic: the model deliberately ignores within-subject correlation, so
/// the sandwich is roughly twice the naive inverse-Hessian (SE TVCL 0.166 vs
/// 0.078). Comparing ferx's default against NONMEM's default would look like a
/// factor-of-two bug.
///
/// Evaluation-only (`outer_maxiter = 0`, NONMEM `MAXEVAL=0`) at NONMEM's own
/// optimum, so it returns immediately and needs no `slow-tests` gate — and so the
/// Hessian is formed at the same point NONMEM formed its own.
#[test]
fn covariance_step_at_nonmem_optimum_matches_nonmem_rsr_standard_errors() {
    use ferx_core::types::{CovarianceMethod, CovarianceStatus};

    let parsed = parse_full_model_file(Path::new(MODEL)).expect("pooled example must parse");
    let model = parsed.model;
    assert_eq!(
        model.n_eta, 0,
        "the anchor model must carry no random effects"
    );

    let mut params = model.default_params.clone();
    params.theta[0] = NM_TVCL;
    params.theta[1] = NM_TVV;
    params.sigma.values[0] = NM_SIGMA_SD;

    let opts = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: true,
        covariance_method: CovarianceMethod::Sandwich,
        ..Default::default()
    };
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("anchor data must read");
    let res = fit(&model, &pop, &params, &opts).expect("the covariance step must run at n_eta = 0");

    assert_ne!(
        res.covariance_status,
        CovarianceStatus::Failed,
        "the covariance step must not fail at n_eta = 0"
    );
    // A zero-Omega fit must report no Omega SEs at all — not a phantom entry, and
    // not a `None` standing in for a failed step (the status assert above rules
    // that reading out).
    assert!(
        res.se_omega.as_ref().is_none_or(|s| s.is_empty()),
        "an Omega-less fit must carry no Omega SEs, got {:?}",
        res.se_omega
    );

    let se_theta = res
        .se_theta
        .as_ref()
        .expect("theta SEs must be produced at n_eta = 0");
    let se_sigma = res
        .se_sigma
        .as_ref()
        .expect("sigma SEs must be produced at n_eta = 0");

    // Observed at the time of writing: 2.0e-6 (TVCL), 7.6e-6 (TVV), 2.7e-6 (sigma)
    // — an order of magnitude closer than the *converged*-fit comparison in
    // `docs/faq.qmd` (8e-5 / 1.6e-4 / 2e-4), because this forms the Hessian at
    // exactly the point NONMEM formed its own rather than at ferx's own optimum.
    //
    // The band is 1e-4, and it cannot usefully go tighter: NONMEM prints these to
    // six figures, so `1.76021` alone carries ±3e-6 of relative rounding. That
    // still leaves it a real assertion — ferx's *default* `r` covariance gives SE
    // TVCL 0.0776, a relative error of 0.53, five thousand times outside the band.
    // Naive-pooled ignores within-subject correlation by construction, so the
    // sandwich and the naive inverse-Hessian differ by about a factor of two here;
    // this test is what keeps the two from being silently interchanged.
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    assert!(
        rel(se_theta[0], NM_SE_TVCL) < 1e-4,
        "SE TVCL {} vs NONMEM {} (rel {:.2e})",
        se_theta[0],
        NM_SE_TVCL,
        rel(se_theta[0], NM_SE_TVCL)
    );
    assert!(
        rel(se_theta[1], NM_SE_TVV) < 1e-4,
        "SE TVV {} vs NONMEM {} (rel {:.2e})",
        se_theta[1],
        NM_SE_TVV,
        rel(se_theta[1], NM_SE_TVV)
    );
    assert!(
        rel(se_sigma[0], NM_SE_SIGMA_SD) < 1e-4,
        "SE PROP_ERR (sd) {} vs NONMEM {} (rel {:.2e})",
        se_sigma[0],
        NM_SE_SIGMA_SD,
        rel(se_sigma[0], NM_SE_SIGMA_SD)
    );
}
