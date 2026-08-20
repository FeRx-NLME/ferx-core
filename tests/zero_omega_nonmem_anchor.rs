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
//! Standard errors are NOT checked here. NONMEM's `$COVARIANCE` default is the
//! `RSR` sandwich while ferx's `covariance_method` default is `r`, so the two
//! disagree by design; under `covariance_method = rsr` they agree to four
//! significant figures (SE TVCL 0.166312 vs 1.66298E-01, SE TVV 1.760499 vs
//! 1.76021E+00). That comparison lives in the docs, not in an assertion, because
//! pinning it would be pinning NONMEM's print precision.

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
