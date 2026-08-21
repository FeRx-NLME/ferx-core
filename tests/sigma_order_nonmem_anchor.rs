//! NONMEM anchor for #1001 — single-endpoint `[error_model]` sigma binding.
//!
//! #1001 is a *binding* bug: a single-endpoint `[error_model]` validated the sigma
//! name it was given and then discarded it, consuming the flat sigma vector
//! positionally instead. `DV ~ proportional(S_SMALL)` with `S_BIG` declared first
//! therefore fitted against `S_BIG`, silently.
//!
//! Every measurement behind the issue is ferx measuring ferx — the same file with
//! the two `sigma` lines transposed. That shows the two spellings *disagree*, but
//! it cannot say which one is right, because both sides run on the same engine.
//! This file closes that gap against NONMEM 7.6.0 and, in doing so, pins the
//! invariant the new parse check enforces: **in-order positional binding is what
//! the file reads as describing.**
//!
//! Two control streams differing in exactly one line
//! (`nonmem_anchor/sigma_order_{small,big}.ctl`), `ADVAN1 TRANS1`, `$THETA` all
//! `FIX`, `$OMEGA 0.09 FIX`, `MAXEVAL=0 METHOD=1 INTER`:
//!
//! * **small** — `$SIGMA 0.0025 FIX` (0.05² — `S_SMALL`), the fit the model text
//!   reads as describing.
//! * **big** — `$SIGMA 4.0 FIX` (2.0² — `S_BIG`), the fit a mis-ordered model
//!   silently got instead.
//!
//! Measured (`nonmem_anchor/results/sigma_order_{small,big}.lst`), against ferx at
//! `inner_tol = 1e-10`, `ode_reltol = 1e-10`, `ode_abstol = 1e-12` — the default
//! `inner_tol = 1e-5` leaves ~8e-4 of residual EBE noise in the objective, which is
//! tolerance, not disagreement (both sides move together when it is tightened):
//!
//! ```text
//!   sigma (SD)   NONMEM 7.6.0     ferx          |diff|
//!   0.05         -6.83893110      -6.838931     < 1e-6
//!   2.00        103.82896422     103.828964     < 1e-6
//!   ---------------------------------------------------
//!   delta        110.66789532     110.667895
//! ```
//!
//! So the silent misbind was worth **110.67 OFV units** on 24 observations, and
//! both endpoints of that interval are independently confirmed. NONMEM cannot
//! express the bug at all — `$ERROR` indexes `EPS(1)`/`EPS(2)` positionally with no
//! name to disagree with — which is why the fix is a rejection rather than a
//! re-binding: there is no reference behaviour to match, only a spelling that
//! promises something the engine does not do.
//!
//! Tier 2: `fit` with `maxiter = 0` (objective at fixed parameters) plus one parse,
//! no convergence loop, so it needs no `slow-tests` gate.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv};
use std::io::Write;
use tempfile::NamedTempFile;

/// `nonmem_anchor/sigma_order.csv`, verbatim — 6 subjects, one 100 mg bolus into
/// the single state, four observations each.
const CSV: &str = "ID,TIME,AMT,EVID,MDV,CMT,DV\n\
1,0,100,1,1,1,0\n\
1,0.5,0,0,0,1,9.5\n\
1,2,0,0,0,1,8.2\n\
1,8,0,0,0,1,4.5\n\
1,24,0,0,0,1,0.9\n\
2,0,100,1,1,1,0\n\
2,0.5,0,0,0,1,10.4\n\
2,2,0,0,0,1,8.9\n\
2,8,0,0,0,1,5.1\n\
2,24,0,0,0,1,1.2\n\
3,0,100,1,1,1,0\n\
3,0.5,0,0,0,1,8.7\n\
3,2,0,0,0,1,7.4\n\
3,8,0,0,0,1,3.9\n\
3,24,0,0,0,1,0.7\n\
4,0,100,1,1,1,0\n\
4,0.5,0,0,0,1,11.2\n\
4,2,0,0,0,1,9.6\n\
4,8,0,0,0,1,5.6\n\
4,24,0,0,0,1,1.4\n\
5,0,100,1,1,1,0\n\
5,0.5,0,0,0,1,9.1\n\
5,2,0,0,0,1,7.9\n\
5,8,0,0,0,1,4.2\n\
5,24,0,0,0,1,0.8\n\
6,0,100,1,1,1,0\n\
6,0.5,0,0,0,1,10.0\n\
6,2,0,0,0,1,8.5\n\
6,8,0,0,0,1,4.8\n\
6,24,0,0,0,1,1.0\n";

/// NONMEM `sigma_order_small.ctl`: `$SIGMA 0.0025 FIX`.
const NONMEM_OFV_SMALL: f64 = -6.8389311000825916;
/// NONMEM `sigma_order_big.ctl`: `$SIGMA 4.0 FIX`.
const NONMEM_OFV_BIG: f64 = 103.82896421994307;

/// The `[fit_options]` both ferx models share. `maxiter = 0` holds the thetas at
/// their initials so the objective is a deterministic function of the model text;
/// the tolerances are tightened to NONMEM's inner precision so the comparison is
/// not dominated by EBE noise (see the module docs).
const FIT_OPTIONS: &str = "\
[fit_options]
  method = focei
  maxiter = 0
  covariance = false
  inner_tol = 1e-10
  ode_reltol = 1e-10
  ode_abstol = 1e-12
";

/// The ferx model, parameterised by its `sigma` declarations and error statement.
fn model(sigma_decls: &str, error_stmt: &str) -> String {
    format!(
        "\
[parameters]
  theta TVKE(0.1, 0.001, 1.0)
  theta TVV(10.0, 1.0, 100.0)

  omega ETA1 ~ 0.09

{sigma_decls}

[individual_parameters]
  KE = TVKE * exp(ETA1)
  V = TVV

[structural_model]
  ode(obs_cmt=CENT, states=[CENT])

[odes]
  d/dt(CENT) = -KE * CENT

[error_model]
  {error_stmt}

[scaling]
  obs_scale = V

{FIT_OPTIONS}"
    )
}

/// Fit `model_str` against the anchor dataset and return the objective.
fn ofv(model_str: &str) -> f64 {
    let parsed = match parse_full_model(model_str) {
        Ok(p) => p,
        Err(e) => panic!("model must parse: {e}"),
    };
    let mut csv = NamedTempFile::new().expect("temp csv");
    csv.write_all(CSV.as_bytes()).expect("write csv");
    csv.flush().expect("flush csv");
    let pop = read_nonmem_csv(csv.path(), None, None).expect("anchor data must read");
    let params = parsed.model.default_params.clone();
    let result = fit(&parsed.model, &pop, &params, &parsed.fit_options).expect("fit returns");
    result.ofv
}

#[test]
fn single_sigma_matches_nonmem() {
    // The baseline: one declared sigma, no order to get wrong. Establishes that
    // ferx's proportional-error objective agrees with NONMEM's `Y = F*(1+EPS(1))`
    // on this dataset before any ordering question is asked.
    let got = ofv(&model(
        "  sigma S_SMALL ~ 0.05 (sd)",
        "DV ~ proportional(S_SMALL)",
    ));
    assert!(
        (got - NONMEM_OFV_SMALL).abs() < 1e-5,
        "ferx {got} vs NONMEM {NONMEM_OFV_SMALL}"
    );
}

#[test]
fn in_order_trailing_sigma_matches_nonmem_and_does_not_change_the_fit() {
    // Declaring a second, unreferenced sigma *after* the one the error model names
    // must not perturb the fit — it occupies slot 1, which a `proportional` model
    // never loads. This is the invariant that makes the in-order spelling safe, and
    // it is pinned against NONMEM rather than only against ferx's own single-sigma
    // run.
    let got = ofv(&model(
        "  sigma S_SMALL ~ 0.05 (sd)\n  sigma S_BIG ~ 2.0 (sd)",
        "DV ~ proportional(S_SMALL)",
    ));
    assert!(
        (got - NONMEM_OFV_SMALL).abs() < 1e-5,
        "ferx {got} vs NONMEM {NONMEM_OFV_SMALL}"
    );
}

#[test]
fn leading_sigma_is_the_one_that_binds_and_matches_nonmem() {
    // The other endpoint of the interval: with `S_BIG` declared first and *named*,
    // ferx reproduces the big-sigma NONMEM run. Before #1001 a model naming
    // `S_SMALL` in this same file produced this same number — that was the bug.
    // Now that spelling is rejected, so this is the only way to reach it, and it
    // agrees with the reference implementation.
    let got = ofv(&model(
        "  sigma S_BIG ~ 2.0 (sd)\n  sigma S_SMALL ~ 0.05 (sd)",
        "DV ~ proportional(S_BIG)",
    ));
    assert!(
        (got - NONMEM_OFV_BIG).abs() < 1e-5,
        "ferx {got} vs NONMEM {NONMEM_OFV_BIG}"
    );
    // The span the silent misbind used to cover, confirmed on both engines.
    let nonmem_delta = NONMEM_OFV_BIG - NONMEM_OFV_SMALL;
    assert!(
        (nonmem_delta - 110.667_895_32).abs() < 1e-6,
        "NONMEM delta {nonmem_delta}"
    );
}

#[test]
fn mis_ordered_model_is_rejected_rather_than_silently_bound() {
    // The whole of #1001: the spelling that used to reach `NONMEM_OFV_BIG` while
    // naming `S_SMALL` no longer parses at all. Paired with the test above — which
    // shows the number itself is a legitimate NONMEM-matching fit when the model
    // asks for it honestly — this pins that the fix removed the *silence*, not the
    // behaviour.
    let err = match parse_full_model(&model(
        "  sigma S_BIG ~ 2.0 (sd)\n  sigma S_SMALL ~ 0.05 (sd)",
        "DV ~ proportional(S_SMALL)",
    )) {
        Ok(_) => panic!("a mis-ordered single-endpoint error model must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("consumed positionally") && err.contains("S_SMALL") && err.contains("S_BIG"),
        "got: {err}"
    );
}
