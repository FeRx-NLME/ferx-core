//! NONMEM 7.6.0 anchor for per-endpoint LTBS combined error (#2729): the issue's
//! motivating shape, plasma concentration (`CMT=2`) and cumulative urine amount
//! (`CMT=3`), each with its own `PROP` and `ADD`:
//!
//! ```text
//! CMT=2: log(DV) ~ combined(PROP_P, ADD_P)
//! CMT=3: log(DV) ~ combined(PROP_U, ADD_U)
//! ```
//!
//! against `$ERROR` switching `THETA` pairs on `CMT` and scoring
//! `Y = LOG(IPR) + SQRT(PROP**2 + ADD**2/IPR**2)*EPS(1)` with `IPR` floored at 1e-12.
//!
//! Data (`tests/nonmem/make_ltbs_combined_cmt_data.py`, seeded): 16 subjects, both
//! endpoints sampled from `t = 0` — where both predictions are 0 and sit on the floor —
//! to 24 h. NONMEM reads `LNDV`, the log of the natural `DV` ferx reads.
//!
//! * `ltbs_combined_cmt_eval.ctl` / `_eval_foce.ctl` — FOCEI / FOCE, `MAXEVAL=0`, at
//!   the initial estimates (`ADVAN13 TOL=12`).
//! * `ltbs_combined_cmt.ctl` — the FOCEI fit; it terminated with `ROUNDING ERRORS` and
//!   `ADD_U` collapsed to 0.004. `ltbs_combined_cmt_final_eval.ctl` re-evaluates
//!   `MAXEVAL=0` at its printed estimates: a second point where the urine endpoint is
//!   nearly pure proportional and the plasma one still combined.
//!
//! | quantity | NONMEM 7.6.0 | ferx (Linux x86_64) |
//! |---|---:|---:|
//! | FOCEI OBJ, initial estimates | 1236.2501551968762 | 1236.250155138609 |
//! | FOCE OBJ, initial estimates | 1279.7110938059068 | 1279.711093903299 |
//! | FOCEI OBJ, NONMEM's final estimates | 1020.2975589006809 | 1020.297558907486 |
//! | FOCEI OBJ, inits + early-time plasma `PROP` factor | 1265.1650674155283 | 1265.165067356047 |
//!
//! The last row is the motivating model's whole residual shape:
//! `ltbs_combined_cmt_mag_eval.ctl` multiplies the plasma `PROP` by `FAC = 1.8` while
//! `TIME < 3` (one dose at 0, so that is the time after dose), which ferx writes on the
//! per-CMT argument itself:
//! `CMT=2: log(DV) ~ combined(PROP_P * (if (TAD < 3) FAC else 1), ADD_P)`.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const DATA: &str = "tests/nonmem/ltbs_combined_cmt.csv";
const NM_FOCEI_OBJ: f64 = 1236.2501551968762;
const NM_FOCE_OBJ: f64 = 1279.7110938059068;
const NM_FINAL_OBJ: f64 = 1020.2975589006809;
/// `ltbs_combined_cmt_mag_eval.ext`.
const NM_MAG_OBJ: f64 = 1265.1650674155283;
/// Realised gaps (Linux x86_64): −5.8e-8 (FOCEI, inits), 9.7e-8 (FOCE, inits), 6.8e-9
/// (FOCEI, final) — the ODE solvers' tolerance, not the residual model, which the
/// single-endpoint closed-form anchor pins to 6e-9. ~5× headroom; swapping the two
/// endpoints' σ pairs moves the objective by 263.
const OFV_TOL: f64 = 5e-7;

struct Params {
    theta: [f64; 4],
    /// `[PROP_P, ADD_P, PROP_U, ADD_U]`, SDs.
    sigma: [f64; 4],
    omega: [f64; 2],
}

const INITS: Params = Params {
    theta: [2.2, 9.0, 1.3, 0.35],
    sigma: [0.12, 0.07, 0.25, 0.4],
    omega: [0.09, 0.04],
};
const FINAL: Params = Params {
    theta: [2.04299, 9.97957, 1.43556, 0.389745],
    sigma: [0.0916699, 0.0282388, 0.227097, 0.004],
    omega: [0.0639221, 0.0314136],
};

fn model(p: &Params) -> ferx_core::types::CompiledModel {
    model_with(p, "PROP_P", "")
}

/// `prop_p` is the plasma endpoint's first argument; `extra` adds `[parameters]` lines.
fn model_with(p: &Params, prop_p: &str, extra: &str) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL({}, 0.01, 50.0)
  theta TVV({}, 0.1, 500.0)
  theta TVKA({}, 0.01, 50.0)
  theta TVFE({}, 0.01, 0.99)
  omega ETA_CL ~ {}
  omega ETA_V  ~ {}
  sigma PROP_P ~ {} (sd)
  sigma ADD_P ~ {} (sd)
  sigma PROP_U ~ {} (sd)
  sigma ADD_U ~ {} (sd)
{extra}
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA
  FE = TVFE

[structural_model]
  ode(states=[depot, central, urine])

[odes]
  d/dt(depot) = -KA * depot
  d/dt(central) = KA * depot - (CL/V) * central
  d/dt(urine) = FE * (CL/V) * central

[scaling]
  y[CMT=2] = central / V
  y[CMT=3] = urine

[error_model]
  CMT=2: log(DV) ~ combined({prop_p}, ADD_P)
  CMT=3: log(DV) ~ combined(PROP_U, ADD_U)

[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
",
        p.theta[0],
        p.theta[1],
        p.theta[2],
        p.theta[3],
        p.omega[0],
        p.omega[1],
        p.sigma[0],
        p.sigma[1],
        p.sigma[2],
        p.sigma[3]
    );
    let m = parse_model_string(&src).expect("model must parse");
    assert!(m.log_transform && !m.dv_pre_logged);
    m
}

fn evaluate(m: &ferx_core::types::CompiledModel, method: EstimationMethod) -> f64 {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
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

fn check(p: &Params, method: EstimationMethod, want: f64, label: &str) {
    check_model(&model(p), method, want, label);
}

fn check_model(
    m: &ferx_core::types::CompiledModel,
    method: EstimationMethod,
    want: f64,
    label: &str,
) {
    let ofv = evaluate(m, method);
    eprintln!(
        "{label}: ferx {ofv:.12} vs NONMEM {want:.12}, diff {:e}",
        ofv - want
    );
    assert!(ofv.is_finite(), "{label}: ferx OFV {ofv}");
    assert!(
        (ofv - want).abs() < OFV_TOL,
        "{label}: ferx {ofv} vs NONMEM {want}"
    );
}

#[test]
fn ltbs_combined_per_cmt_focei_objective_matches_nonmem_at_the_initial_estimates() {
    check(
        &INITS,
        EstimationMethod::FoceI,
        NM_FOCEI_OBJ,
        "FOCEI, inits",
    );
}

#[test]
fn ltbs_combined_per_cmt_foce_objective_matches_nonmem_at_the_initial_estimates() {
    check(&INITS, EstimationMethod::Foce, NM_FOCE_OBJ, "FOCE, inits");
}

#[test]
fn ltbs_combined_per_cmt_focei_objective_matches_nonmem_at_its_final_estimates() {
    check(
        &FINAL,
        EstimationMethod::FoceI,
        NM_FINAL_OBJ,
        "FOCEI, NONMEM final",
    );
}

/// Non-degeneracy: the endpoints really are scored with their own σ pairs. Swapping the
/// plasma and urine pairs moves the objective far beyond the tolerance, so an
/// implementation that read one endpoint's σ for both could not pass the anchor.
#[test]
fn swapping_the_endpoint_sigma_pairs_moves_the_objective() {
    let swapped = Params {
        sigma: [
            INITS.sigma[2],
            INITS.sigma[3],
            INITS.sigma[0],
            INITS.sigma[1],
        ],
        ..INITS
    };
    let ofv = evaluate(&model(&swapped), EstimationMethod::FoceI);
    eprintln!("swapped σ pairs: ferx {ofv:.6} vs NONMEM {NM_FOCEI_OBJ:.6}");
    assert!(ofv.is_finite());
    assert!(
        (ofv - NM_FOCEI_OBJ).abs() > 10.0,
        "swapping the pairs left the objective at {ofv}"
    );
}

/// The motivating model's whole residual shape: per-endpoint `PROP`/`ADD` and, on the
/// plasma `PROP` only, a factor while `TAD < 3`. Its NONMEM objective is 29 above the
/// factor-free one, so the factor is live and a model that ignored it could not pass.
#[test]
fn ltbs_combined_per_cmt_with_an_early_time_prop_factor_matches_nonmem() {
    let m = model_with(
        &INITS,
        "PROP_P * (if (TAD < 3) FAC else 1)",
        "  theta FAC(1.8, 0.1, 10.0)\n",
    );
    assert!(m.has_custom_ruv_magnitude());
    assert!((NM_MAG_OBJ - NM_FOCEI_OBJ).abs() > 10.0);
    check_model(
        &m,
        EstimationMethod::FoceI,
        NM_MAG_OBJ,
        "FOCEI, inits + PROP factor",
    );
}
