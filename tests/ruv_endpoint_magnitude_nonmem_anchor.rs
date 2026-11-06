//! NONMEM 7.5.1 anchor for magnitude expressions on `CMT=N:` lines and in `if/else`
//! branches (#2731), the joint-model shape: a residual per endpoint with a factor on one.
//!
//! Every test evaluates the FOCEI objective once at fixed parameters, against `MAXEVAL=0`
//! runs of `nonmem_anchor/ruv_endpoint_magnitude_*.ctl` (`ADVAN6 TOL=9`; outputs in
//! `nonmem_anchor/results/`). The data (`ruv_endpoint_magnitude.csv`, simulated by
//! `ruv_endpoint_magnitude_sim.ctl`) has two endpoints, two doses and a flag `EARLY` that
//! changes within a subject on both endpoints, so the factor is live on some rows and off
//! on others of the same endpoint.
//!
//! | arm | what it pins | NONMEM | realised gap |
//! |---|---|---:|---:|
//! | `cmt` | factor on `CMT=1`, own sigma on `CMT=2` | −354.349027673 | 6.1e-8 |
//! | `cmt_b` | same model, second point (`TPF` below 1, other σ) | −294.153411988 | 1.4e-8 |
//! | `workaround` | the single statement the issue replaces | −354.349027666 | 6.8e-8 |
//! | `mixed` | `TAD` factor on the proportional half of a `combined` | −185.934433389 | 4.1e-9 |
//! | `shared` | one sigma on both endpoints, factor on one | −282.091324165 | 5.9e-8 |
//! | `ifelse` | one sigma in both branches, factor on one | −200.188323133 | 3.8e-8 |
//!
//! Gaps measured on Linux x86_64 (`tools/linux-test.sh`); macOS arm64 prints the same digits.
//!
//! `shared` and `ifelse` are the arms a per-sigma multiplier cannot pass: the factor is on
//! the same sigma slot the other endpoint reads bare, so it has to follow the row's
//! endpoint. `ifelse` and `mixed` also reach a `combined` endpoint, whose sigmas are not
//! the leading slots of the flat vector.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

const DATA: &str = "nonmem_anchor/ruv_endpoint_magnitude.csv";

/// Largest realised gap to NONMEM is 6.8e-8 (see the table above); about 15× headroom.
const OFV_TOL: f64 = 1e-6;

const NM_CMT: f64 = -354.34902767324832;
const NM_CMT_B: f64 = -294.15341198817254;
const NM_WORKAROUND: f64 = -354.34902766613953;
const NM_MIXED: f64 = -185.93443338905377;
const NM_SHARED: f64 = -282.09132416510522;
const NM_IFELSE: f64 = -200.18832313308110;

/// The structural model every control stream shares. `params` is the arm's extra
/// `theta` and `sigma` declarations, `error` its `[error_model]` body.
fn model(params: &str, error: &str) -> ferx_core::types::CompiledModel {
    let src = format!(
        "[parameters]
  theta TVCL(2.2, FIX)
  theta TVV(18.0, FIX)
  theta TVKE0(0.6, FIX)
{params}
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KE0 = TVKE0

[structural_model]
  ode(states=[central, effect])

[odes]
  d/dt(central) = -CL/V * central
  d/dt(effect) = KE0 * (central/V - effect)

[scaling]
  y[CMT=1] = central / V
  y[CMT=2] = effect

[covariates]
  EARLY continuous
  ISUR continuous

[error_model]
{error}

[fit_options]
  method = focei
"
    );
    parse_model_string(&src).expect("model must parse")
}

fn evaluate(m: &ferx_core::types::CompiledModel) -> f64 {
    let pop = read_nonmem_csv(Path::new(DATA), None, None).expect("data");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
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

fn assert_anchor(label: &str, ofv: f64, nonmem: f64) {
    eprintln!(
        "{label}: ferx {ofv:.12} vs NONMEM {nonmem:.12}, diff {:e}",
        ofv - nonmem
    );
    assert!(
        (ofv - nonmem).abs() < OFV_TOL,
        "{label}: ferx {ofv} vs NONMEM {nonmem}"
    );
}

const CMT_ERROR: &str = "  CMT=1: DV ~ additive(S * (1.0 + (TPF - 1.0) * EARLY))
  CMT=2: DV ~ additive(U)";

fn cmt_model(tpf: f64, s: f64, u: f64) -> ferx_core::types::CompiledModel {
    model(
        &format!(
            "  theta TPF({tpf}, FIX)
  sigma S ~ {s} (sd) FIX
  sigma U ~ {u} (sd) FIX"
        ),
        CMT_ERROR,
    )
}

fn workaround_model() -> ferx_core::types::CompiledModel {
    model(
        "  theta TPF(1.6, FIX)
  theta TRU(1.6, FIX)
  sigma S ~ 0.25 (sd) FIX",
        "  DV ~ additive(S * ((1.0 + (TPF - 1.0) * EARLY) * (1.0 - ISUR) + TRU * ISUR))",
    )
}

#[test]
fn factor_on_one_cmt_line_matches_nonmem() {
    assert_anchor("cmt", evaluate(&cmt_model(1.6, 0.25, 0.4)), NM_CMT);
}

#[test]
fn factor_on_one_cmt_line_matches_nonmem_at_a_second_point() {
    assert_anchor("cmt_b", evaluate(&cmt_model(0.7, 0.3, 0.35)), NM_CMT_B);
}

#[test]
fn single_statement_workaround_matches_nonmem() {
    assert_anchor("workaround", evaluate(&workaround_model()), NM_WORKAROUND);
}

/// The issue's validation: the per-endpoint spelling and the single-statement workaround
/// are one model, so the objective is the same on the same data. NONMEM's own two runs
/// differ by 7.1e-9, from `SD = 0.4` written as `0.25 * 1.6`. ferx's gap measures 0.
#[test]
fn cmt_lines_and_the_single_statement_workaround_give_the_same_objective() {
    let per_endpoint = evaluate(&cmt_model(1.6, 0.25, 0.4));
    let workaround = evaluate(&workaround_model());
    eprintln!(
        "per-endpoint {per_endpoint:.12} vs workaround {workaround:.12}, diff {:e}",
        per_endpoint - workaround
    );
    assert!(
        (per_endpoint - workaround).abs() < 1e-9,
        "per-endpoint {per_endpoint} vs workaround {workaround}"
    );
}

#[test]
fn tad_factor_on_a_combined_endpoint_matches_nonmem() {
    let m = model(
        "  theta TPF(1.6, FIX)
  sigma A ~ 0.25 (sd) FIX
  sigma P ~ 0.2 (sd) FIX
  sigma B ~ 0.3 (sd) FIX",
        "  CMT=1: DV ~ additive(A)
  CMT=2: DV ~ combined(P * (if (TAD < 3.0) TPF else 1.0), B)",
    );
    assert_anchor("mixed", evaluate(&m), NM_MIXED);
}

#[test]
fn one_sigma_on_both_endpoints_with_a_factor_on_one_matches_nonmem() {
    let m = model(
        "  theta TPF(1.6, FIX)
  sigma S ~ 0.25 (sd) FIX",
        "  CMT=1: DV ~ additive(S * TPF)
  CMT=2: DV ~ additive(S)",
    );
    assert_anchor("shared", evaluate(&m), NM_SHARED);
}

#[test]
fn if_else_branches_sharing_a_sigma_match_nonmem() {
    let m = model(
        "  theta TPF(1.6, FIX)
  sigma S ~ 0.25 (sd) FIX
  sigma P ~ 0.2 (sd) FIX",
        "  if (EARLY > 0.5) {
    DV ~ additive(S * TPF)
  } else {
    DV ~ combined(P, S)
  }",
    );
    assert_anchor("ifelse", evaluate(&m), NM_IFELSE);
}
