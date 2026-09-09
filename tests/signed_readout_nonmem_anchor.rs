//! NONMEM anchor for #1020 — a Form C `[scaling]` readout that goes negative.
//!
//! The bug was ferx-internal (the ODE overshoot guard, a statement about a
//! compartment amount, was applied to the *readout*), so the regression tests in
//! `src/` are ferx measuring ferx. This file pins the convention against the
//! reference implementation: NONMEM applies no non-negativity guard to `$ERROR`
//! output, so a signed readout comes back signed.
//!
//! `nonmem_anchor/signed_readout.ctl` — one-compartment IV bolus written as
//! `$DES` (`ADVAN13`), read out as a change from a baseline:
//!
//! ```text
//! $DES    DADT(1) = -K*A(1)
//! $ERROR  IPRED = A(1)/V - BASE
//! ```
//!
//! with `CL = 4`, `V = 12`, `BASE = 2`, all `FIX`, `$OMEGA 0 FIX`, `MAXEVAL=0` —
//! a pure prediction anchor. Measured (`nonmem_anchor/results/signed_readout.tab`,
//! NONMEM 7.5.1):
//!
//! ```text
//!     t   NONMEM IPRED
//!  0.25     5.6670
//!  1.00     3.9711
//!  4.00     0.19664
//! 12.00    -1.8474
//! 24.00    -1.9972
//! ```
//!
//! The last two are negative and NONMEM reports them as such. Before #1020 ferx
//! returned exactly `0.0` for both.
//!
//! Tier 2: `predict` at fixed parameters, no convergence loop, so no `slow-tests`
//! gate is needed.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{predict, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/signed_readout.csv`, keyed for ferx.
const CSV: &str = "ID,TIME,DV,MDV,EVID,AMT,CMT\n\
1,0,.,1,1,100,1\n\
1,0.25,1.0,0,0,0,1\n\
1,1,1.0,0,0,0,1\n\
1,4,1.0,0,0,0,1\n\
1,12,1.0,0,0,0,1\n\
1,24,1.0,0,0,0,1\n";

/// ferx equivalent of the control stream: the same `$DES`, the same fixed
/// parameters, and the `$ERROR` output expression written as a Form C readout.
const FERX: &str = r#"
[parameters]
  theta TVCL(4.0, 0.0, 1e15)
  theta TVV(12.0, 0.0, 1e15)
  theta TVBASE(2.0, 0.0, 1e15)
  omega ETA_CL ~ 0.0
  sigma ADD ~ 0.1 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVBASE

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[scaling]
  y = central / V - BASE

[error_model]
  DV ~ additive(ADD)

[fit_options]
  ode_reltol = 1e-10
  ode_abstol = 1e-12
"#;

/// NONMEM `IPRED` at t = 0.25, 1, 4, 12, 24 h (5 significant figures).
const NONMEM: [f64; 5] = [5.6670E+00, 3.9711E+00, 1.9664E-01, -1.8474E+00, -1.9972E+00];

#[test]
fn ferx_form_c_readout_reproduces_nonmem_including_its_negative_predictions() {
    let mut f = NamedTempFile::new().expect("temp csv");
    write!(f, "{CSV}").expect("write csv");
    f.flush().expect("flush csv");
    let pop = read_nonmem_csv(f.path(), None, None).expect("dataset loads");
    let model = parse_full_model(FERX).expect("model parses").model;

    let preds: Vec<f64> = predict(&model, &pop, &model.default_params)
        .into_iter()
        .map(|p| p.pred)
        .collect();
    assert_eq!(preds.len(), NONMEM.len(), "one prediction per observation");

    // The two late points are the whole point of the anchor: assert they are
    // genuinely negative before checking the values, so a re-introduced clamp
    // fails here with the diagnosis rather than as a bare tolerance miss.
    assert!(
        preds[3] < 0.0 && preds[4] < 0.0,
        "the sub-baseline predictions must stay negative, got {preds:?}"
    );

    // 1e-4 relative, matching the other NONMEM prediction anchors: the reference
    // table carries 5 significant figures. The model pins `ode_reltol`/`ode_abstol`
    // tight because this readout is a *difference* of two numbers of size ~2 — at
    // t=4 it is ~0.2, so any LSODA-vs-RK45 disagreement in the concentration is
    // amplified by the cancellation. At those tolerances every point agrees to
    // within the reference table's own rounding (worst case 1.4e-5 relative).
    for (i, (&got, &want)) in preds.iter().zip(NONMEM.iter()).enumerate() {
        let rel = (got - want).abs() / want.abs();
        assert!(
            rel < 1e-4,
            "obs {i}: ferx {got:.6} vs NONMEM {want:.6} (rel {rel:.2e})"
        );
    }
}
