//! NONMEM 7.6.0 anchor for combined proportional + additive error under
//! log-transform-both-sides (#2729):
//!
//! ```text
//! IPR = F
//! IF (IPR.LT.1E-12) IPR = 1E-12
//! Y = LOG(IPR) + SQRT(PROP**2 + ADD**2/IPR**2)*EPS(1)
//! ```
//!
//! against ferx's `log(DV) ~ combined(PROP, ADD)` and `DV ~ log_combined(PROP, ADD)`.
//!
//! Data, control streams and outputs are committed under `tests/nonmem/`
//! (`make_ltbs_combined_data.py`, seeded): 16 subjects, one-compartment oral, sampled
//! from ~9 down to ~0.06 mg/L, so `ADD/F` runs from below `PROP` to several times it and
//! both terms are live. Every subject also has an observation at `t = 0`, where `F = 0`:
//! that row sits on the LTBS floor, where `W = ADD/1e-12`, so the floor's spelling
//! (floor `F`, then both log it and divide `ADD` by it) is compared too.
//!
//! * `ltbs_combined_eval.ctl` — FOCEI (`METHOD=COND INTERACTION`), `MAXEVAL=0`, at
//!   the initial estimates.
//! * `ltbs_combined_eval_foce.ctl` — the same under FOCE (`METHOD=COND`).
//! * `ltbs_combined.ctl` — the FOCEI fit (terminated `ROUNDING ERRORS (ERROR=134)`
//!   at OBJ 417.2305); `ltbs_combined_final_eval.ctl` re-evaluates `MAXEVAL=0` at its
//!   printed final estimates, a second point with `ADD` less than half its start.
//!
//! | quantity | NONMEM 7.6.0 | ferx (Linux x86_64) |
//! |---|---:|---:|
//! | FOCEI OBJ, initial estimates | 482.53324149657630 | 482.533241490887 (−5.7e-9) |
//! | FOCE OBJ, initial estimates | 1464.3513956553350 | 1464.351395649576 (−5.8e-9) |
//! | FOCEI OBJ, NONMEM's final estimates | 417.23048569895218 | 417.230485698363 (−5.9e-10) |
//!
//! Both spellings give these numbers to the printed digit: the natural-scale CSV holds
//! the log file's values exponentiated at full precision.
//!
//! The FOCE value is ~1000 above FOCEI's: under FOCE the variance is read at the
//! population prediction, where `ADD/F` is far from its conditional value. Both engines
//! agree on that convention, which is what the FOCE row pins.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// `ltbs_combined_eval.ext`, the `-1000000000` row.
const NM_FOCEI_OBJ: f64 = 482.53324149657630;
/// `ltbs_combined_eval_foce.ext`.
const NM_FOCE_OBJ: f64 = 1464.3513956553350;
/// `ltbs_combined_final_eval.ext`.
const NM_FINAL_OBJ: f64 = 417.23048569895218;
/// The measured gaps are ≤ 5.8e-9 (summation order between the engines); the bound sits
/// ~9× above them and far below what a wrong variance moves the objective by: dropping
/// `ADD` moves it by ~1100, and each floor row alone contributes ~50.
const OFV_TOL: f64 = 5e-8;

const INITS: Params = Params {
    theta: [2.2, 9.0, 1.3],
    prop: 0.12,
    add: 0.07,
    omega: [0.09, 0.04],
};
/// `ltbs_combined.ext`'s printed final estimates (6 significant figures), which is
/// exactly the point `ltbs_combined_final_eval.ctl` evaluates.
const FINAL: Params = Params {
    theta: [2.00036, 9.34966, 1.53330],
    prop: 0.104379,
    add: 0.0321163,
    omega: [0.0648868, 0.0401850],
};

struct Params {
    theta: [f64; 3],
    prop: f64,
    add: f64,
    omega: [f64; 2],
}

/// `statement` is the `[error_model]` line; NONMEM's `THETA(4)`/`THETA(5)` are ferx's
/// σ on the SD scale (NONMEM's `$SIGMA 1 FIX` makes `W` the SD).
fn model(p: &Params, statement: &str) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL({}, 0.01, 50.0)
  theta TVV({}, 0.1, 500.0)
  theta TVKA({}, 0.01, 50.0)
  omega ETA_CL ~ {}
  omega ETA_V  ~ {}
  sigma PROP_ERR ~ {} (sd)
  sigma ADD_ERR ~ {} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  {statement}
",
        p.theta[0], p.theta[1], p.theta[2], p.omega[0], p.omega[1], p.prop, p.add
    );
    let m = parse_model_string(&src).expect("model must parse");
    assert_eq!(m.error_model, ferx_core::types::ErrorModel::LogCombined);
    m
}

/// The objective at the model's own parameters, with no outer step.
fn evaluate(m: &ferx_core::types::CompiledModel, data: &str, method: EstimationMethod) -> f64 {
    let pop = read_nonmem_csv(Path::new(data), None, None).expect("data");
    let opts = FitOptions {
        method,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(2),
        ..FitOptions::default()
    };
    fit(m, &pop, &m.default_params, &opts)
        .expect("evaluation")
        .ofv
}

/// Both spellings: `DV ~ log_combined` on the log-scale data NONMEM reads, and
/// `log(DV) ~ combined` on the same data exponentiated.
fn check(p: &Params, method: EstimationMethod, want: f64, label: &str) {
    for (statement, data) in [
        (
            "DV ~ log_combined(PROP_ERR, ADD_ERR)",
            "tests/nonmem/ltbs_combined.csv",
        ),
        (
            "log(DV) ~ combined(PROP_ERR, ADD_ERR)",
            "tests/nonmem/ltbs_combined_natural.csv",
        ),
    ] {
        let ofv = evaluate(&model(p, statement), data, method);
        eprintln!(
            "{label} [{statement}]: ferx {ofv:.12} vs NONMEM {want:.12}, diff {:e}",
            ofv - want
        );
        assert!(ofv.is_finite(), "{label} [{statement}]: ferx OFV {ofv}");
        assert!(
            (ofv - want).abs() < OFV_TOL,
            "{label} [{statement}]: ferx {ofv} vs NONMEM {want}"
        );
    }
}

#[test]
fn ltbs_combined_focei_objective_matches_nonmem_at_the_initial_estimates() {
    check(
        &INITS,
        EstimationMethod::FoceI,
        NM_FOCEI_OBJ,
        "FOCEI, inits",
    );
}

#[test]
fn ltbs_combined_foce_objective_matches_nonmem_at_the_initial_estimates() {
    check(&INITS, EstimationMethod::Foce, NM_FOCE_OBJ, "FOCE, inits");
}

#[test]
fn ltbs_combined_focei_objective_matches_nonmem_at_its_final_estimates() {
    check(
        &FINAL,
        EstimationMethod::FoceI,
        NM_FINAL_OBJ,
        "FOCEI, NONMEM final",
    );
}

/// Non-degeneracy: with `ADD → 0` (the reduced model the issue's port was forced into)
/// the objective is far from NONMEM's combined one, so the anchor above cannot be met by
/// an implementation that drops the additive term.
#[test]
fn dropping_add_moves_the_objective_far_beyond_the_tolerance() {
    let reduced = Params { add: 1e-9, ..INITS };
    let ofv = evaluate(
        &model(&reduced, "DV ~ log_combined(PROP_ERR, ADD_ERR)"),
        "tests/nonmem/ltbs_combined.csv",
        EstimationMethod::FoceI,
    );
    eprintln!("ADD → 0: ferx {ofv:.6} vs combined NONMEM {NM_FOCEI_OBJ:.6}");
    assert!(ofv.is_finite());
    assert!(
        (ofv - NM_FOCEI_OBJ).abs() > 100.0,
        "dropping ADD left the objective at {ofv}"
    );
}

/// The FOCEI fit from the initial estimates lands where NONMEM's did: OBJ 417.2305
/// (`ltbs_combined.ext`) with `ADD` well away from zero, so the additive term is
/// identified rather than collapsed. Measured (Linux x86_64, release): ferx OFV
/// 417.230410, 7.6e-5 *below* NONMEM's (which stopped on `ROUNDING ERRORS`), and
/// `ADD` 0.0321140 against NONMEM's 0.0321163 (7e-5 relative).
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn ltbs_combined_focei_fit_matches_nonmem() {
    let m = model(&INITS, "log(DV) ~ combined(PROP_ERR, ADD_ERR)");
    let pop = read_nonmem_csv(
        Path::new("tests/nonmem/ltbs_combined_natural.csv"),
        None,
        None,
    )
    .expect("data");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        run_covariance_step: false,
        verbose: false,
        checkpoint: false,
        threads: Some(2),
        ..FitOptions::default()
    };
    let r = fit(&m, &pop, &m.default_params, &opts).expect("fit");
    eprintln!(
        "LTBS combined fit: ferx OFV {:.6} (NONMEM {NM_FINAL_OBJ:.6}), sigma {:?}, theta {:?}",
        r.ofv, r.sigma, r.theta
    );
    assert!(r.ofv.is_finite());
    assert!(
        r.ofv < NM_FINAL_OBJ + 1e-3,
        "ferx {} is worse than NONMEM's {NM_FINAL_OBJ}",
        r.ofv
    );
    assert!(
        (r.sigma[1] - FINAL.add).abs() < 0.01 * FINAL.add,
        "ADD {} vs NONMEM {}",
        r.sigma[1],
        FINAL.add
    );
}
