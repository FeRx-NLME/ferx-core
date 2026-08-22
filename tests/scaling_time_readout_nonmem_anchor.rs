//! NONMEM anchor for the `TIME` built-in in a `[scaling]` Form C readout (#1028).
//!
//! Everything else behind #1028 is ferx measuring ferx: the sensitivity parity
//! tests assert `∂y/∂BETA == t`, and the production test differences the same model
//! at two `BETA` values to recover `BETA·t`. Both are exact oracles, but both run on
//! the same engine — they pin that ferx is *self-consistent* about which time the
//! readout sees, not that it is the time a reference implementation would use.
//!
//! ferx's `[scaling] y = <expr>` and NONMEM's `$ERROR IPRED = ...` are the same
//! object: the structural prediction handed to the residual error model, evaluated
//! once per observation record. NONMEM has always had `TIME` in scope there, so a
//! response-versus-time readout translates line for line — which makes it the right
//! reference for the behaviour this issue restored.
//!
//! `nonmem_anchor/scaling_time_readout.ctl` (NONMEM 7.6.0, `ADVAN13 TOL=9`,
//! `MAXEVAL=0`, every `$THETA` `FIX`, `$OMEGA 0 FIX`):
//!
//! ```text
//! $ERROR
//!   IPRED = A(1)/V + BETA*TIME/(TIME + T50)
//! ```
//!
//! against the ferx twin below, whose readout is the same expression:
//!
//! ```text
//! [scaling]
//!   y = central / V + BETA * TIME / (TIME + T50)
//! ```
//!
//! Measured (`nonmem_anchor/results/scaling_time_readout.tab`), 100 mg IV bolus,
//! `CL = 5`, `V = 50`, `BETA = 3`, `T50 = 4`:
//!
//! ```text
//!    t   NONMEM IPRED   A(1)/V (decay)   BETA·t/(t+T50)
//!  0.0   2.0000         2.0000           0.0000
//!  0.5   2.2358         1.9025           0.3333
//!  1.0   2.4097         1.8097           0.6000
//!  2.0   2.6375         1.6375           1.0000
//!  4.0   2.8406         1.3406           1.5000
//!  8.0   2.8987         0.8987           2.0000
//! 12.0   2.8524         0.6024           2.2500
//! 24.0   2.7529         0.1814           2.5714
//! ```
//!
//! The time term overtakes the disposition by 8 h, so a readout reading a stale
//! `TIME = 0` — the pre-#1028 behaviour — would land on the bare decay column and
//! miss by 1.18x at the first sample, rising to 15x by 24 h. This test would have
//! failed loudly on it; `the_nonmem_anchor_discriminates_a_stale_time_of_zero`
//! pins that separation so the anchor cannot quietly stop discriminating.
//!
//! Tier 2: `predict` at fixed parameters, no convergence loop, so no `slow-tests`
//! gate.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/scaling_time_readout.csv`. Single 100 mg bolus into the central
/// compartment, sampled 0.5–24 h. The `t = 0` record is the dose (`MDV=1`), which
/// NONMEM still tables but ferx does not return a prediction for, so the comparison
/// below runs over the seven observation rows.
const CSV: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,0,1,1,100,1\n\
1,0.5,0,0,0,0,1\n\
1,1,0,0,0,0,1\n\
1,2,0,0,0,0,1\n\
1,4,0,0,0,0,1\n\
1,8,0,0,0,0,1\n\
1,12,0,0,0,0,1\n\
1,24,0,0,0,0,1\n";

/// ferx twin of the control stream: the same 1-cpt IV disposition written as an
/// ODE, with the `$ERROR` expression as a Form C readout. `BETA` and `T50` are
/// non-structural individual parameters, so the readout is the only place they act.
const FERX: &str = r#"
[parameters]
  theta TVCL(5.0, 0.0, 1e15)
  theta TVV(50.0, 0.0, 1e15)
  theta TVBETA(3.0, 0.0, 1e15)
  theta TVT50(4.0, 0.0, 1e15)
  omega ETA_CL ~ 0.0
  sigma PROP ~ 0.01 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BETA = TVBETA
  T50  = TVT50

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V + BETA * TIME / (TIME + T50)

[error_model]
  DV ~ proportional(PROP)
"#;

/// NONMEM `IPRED` at the seven observation times, t = 0.5, 1, 2, 4, 8, 12, 24 h.
/// (The tabled `t = 0` dose row, `IPRED = 2.0000`, is the pure baseline `A(1)/V`
/// with the time term identically zero — it carries no information about which time
/// the readout saw, and ferx returns no prediction for an `MDV=1` record.)
const NONMEM_IPRED: [f64; 7] = [
    2.2358E+00, 2.4097E+00, 2.6375E+00, 2.8406E+00, 2.8987E+00, 2.8524E+00, 2.7529E+00,
];

/// The observation times `NONMEM_IPRED` is indexed by.
const TIMES: [f64; 7] = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

/// The readout a stale `TIME = 0` would have produced — the bare `A(1)/V` decay,
/// `2·exp(-0.1·t)`. Not a NONMEM run: it is the closed form of the same
/// disposition, used only to show the anchor above *discriminates*. An anchor both
/// engines pass while reading a stale zero would be worthless.
fn decay_only(t: f64) -> f64 {
    2.0 * (-0.1 * t).exp()
}

#[test]
fn ferx_form_c_time_readout_matches_nonmem_error_block() {
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
    let model = parse_full_model(FERX)
        .expect("TIME in a Form C readout parses")
        .model;

    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect();
    assert_eq!(
        preds.len(),
        NONMEM_IPRED.len(),
        "one prediction per observation"
    );

    // 1e-4 relative: NONMEM's table carries 5 significant figures, so this sits just
    // above the reference's own precision — tight enough that a wrong time anchor
    // (off by an observation, or stuck at 0) cannot slip through.
    for (i, (&got, &want)) in preds.iter().zip(NONMEM_IPRED.iter()).enumerate() {
        let rel = (got - want).abs() / want;
        assert!(
            rel < 1e-4,
            "obs {i}: ferx {got:.6} vs NONMEM {want:.6} (rel {rel:.2e})"
        );
    }
}

/// Discrimination check on the anchor itself: the reference curve must be far from
/// the one a stale `TIME = 0` produces, or matching it would prove nothing. The gap
/// is ≥14% at the earliest sample and grows to 3.2x by 24 h, so the `1e-4` tolerance
/// above has three orders of magnitude of margin over the effect it is detecting.
#[test]
fn the_nonmem_anchor_discriminates_a_stale_time_of_zero() {
    for (i, (&t, &want)) in TIMES.iter().zip(NONMEM_IPRED.iter()).enumerate() {
        let rel = (want - decay_only(t)).abs() / want;
        assert!(
            rel > 0.14,
            "obs {i} (t={t}): NONMEM {want:.4} vs stale-TIME decay {:.4} differ by \
             only {rel:.2e} — the anchor would not catch the bug",
            decay_only(t)
        );
    }
}
