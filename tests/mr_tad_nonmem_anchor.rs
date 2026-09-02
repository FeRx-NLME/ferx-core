//! NONMEM cross-check for an `[odes]` RHS that reads the `TAD` built-in on a model
//! whose absorption is **closed-form eligible** — the cell #1124 is about.
//!
//! ## What is being anchored
//!
//! A static model with a built-in input-rate forcing (`first_order` here) and a
//! linear one-compartment disposition is served by the closed-form
//! modified-release fast path, which evaluates a superposition of single-route
//! kernels and **never evaluates the `[odes]` right-hand side at all**. A `TAD`
//! term on the elimination was therefore not approximated but *absent*: output was
//! bit-identical to the same model with the term deleted.
//!
//! Nothing internal to that path could see it. `identify_disposition` establishes
//! time-invariance behaviourally, by probing the RHS at two times — but it pinned
//! `tad = 0.0`, so a `TAD`-reading RHS evaluated identically at both probes and was
//! admitted as time-invariant. And the routing site tries the closed form *before*
//! its ODE twin, so comparing production against the closed form agreed by
//! construction (the difference was exactly `0.0`). This anchor is the external
//! reference that does see it.
//!
//! Measured at the shared parameter point:
//!
//! | route | OFV | Δ vs NONMEM |
//! |---|---|---|
//! | NONMEM | −234.478426 | — |
//! | ferx, closed form (pre-fix) | −12.4832 | **222.0** |
//! | ferx, integrated (post-fix) | −234.478426 | **2.5e-7** |
//!
//! ## Why two doses
//!
//! Under a single dose `TAD == TAFD`, so a one-dose anchor cannot distinguish a
//! `TAD` defect from a `TAFD` one and would agree with an engine that was wrong
//! about which anchor it used. The second dose at `t = 12` re-anchors `TAD`
//! strictly inside the observation range, with records on both sides (11.5, 12.5),
//! and lands on residual drug — the central amount just before it is 22.90 against
//! an eventual peak of 91.21 — so the incoming side of the dose event is live
//! rather than cancelling against `g(x⁻) = 0`.
//!
//! ## Why there is no lagtime and no `WT` column
//!
//! #1124 is independent of #1070: the closed form dropped the term with no lag
//! anywhere. Leaving the lag out keeps this anchor clear of the `MTIME`/`MTDIFF`
//! machinery `tad_lag_B` needs (`TDOS` switches exactly *at* a dose record, which
//! NONMEM already breaks the integration on) and clear of #1070's separate
//! gradient question. `tad_lag_{A,B}` also carry a FIXED-zero-exponent `WT` purely
//! to force ferx onto its event-driven predictor, because the dense one returned
//! `NaN` for a `TAD`-reading RHS under a lagtime (#1110); #1124 routes every
//! model-time-reading RHS there directly, so that workaround is obsolete here.
//!
//! ## Why the control stream is trusted
//!
//! NONMEM has no `TAD` built-in in `$DES`, so it is authored — and authoring it is
//! exactly where the `tad_lag` anchors went wrong twice. The committed stream is
//! checked, not assumed: `nonmem_anchor/mr_tad_det.ctl` runs it with `$OMEGA 0 FIX`
//! so `IPRED` is a pure function of `THETA`, and
//! `nonmem_anchor/mr_tad_deterministic_truth.py` re-derives the same trajectory
//! with a plain stdlib RK4 (no NONMEM, no ferx). They agree to **1.7e-5**, which is
//! the `$TABLE` print precision rather than either integrator's error.
//!
//! ## The kit (`nonmem_anchor/`)
//!
//! - `simulate_mr_tad_data.py` — deterministic generator (pure stdlib, seed
//!   20260829), 12 subjects, doses at 0 and 12 h.
//! - `mr_tad.ctl` — the control stream; outputs in `results/mr_tad.{ext,lst,tab}`.
//! - `mr_tad_det.ctl` + `mr_tad_deterministic_truth.py` — the construction check
//!   above; output in `results/mr_tad_det.tab`.
//! - `mr_tad_fit.ferx` — the matching ferx model.
//! - `data/mr_tad.csv` — the same CSV both engines read (dose `CMT=1`, obs
//!   `CMT=1`, so no re-keying is needed).

use ferx_core::ode::OdeMethod;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions};
use std::path::Path;

/// NONMEM `#OBJV` from `nonmem_anchor/results/mr_tad.lst`.
const NONMEM_OFV: f64 = -234.478_425_749_153_05;

/// `{ELIM}` is the elimination term: the anchor uses the `TAD` factor, and the
/// non-degeneracy check below re-fits with it deleted. Everything else is shared,
/// so the two differ by exactly one edit.
const MODEL: &str = r"
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(20.0, 0.001, 500.0)
  theta TVKA(0.8, 0.001, 24.0)
  theta THETA_TAD(0.3, 0.0, 10.0)

  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.1 (sd)

[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  KA   = TVKA
  KTAD = THETA_TAD

[structural_model]
  ode(states=[central])

[odes]
  d/dt(central) = first_order(ka=KA) - (CL/V) * central{ELIM}

[scaling]
  y = central / V

[error_model]
  DV ~ proportional(PROP)

[fit_options]
  ode_reltol = 1e-9
  ode_abstol = 1e-11
";

fn ofv_for(elim: &str) -> f64 {
    let model = parse_full_model(&MODEL.replace("{ELIM}", elim))
        .expect("TAD model must parse")
        .model;
    let pop =
        read_nonmem_csv(Path::new("data/mr_tad.csv"), None, None).expect("anchor data must load");
    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        // Evaluate at the shared parameter point: the NONMEM run is
        // `MAXEVAL=0 POSTHOC`, so its "optimum" IS the initial vector above.
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        // NONMEM-equivalent ODE accuracy, and the stepper actually pinned:
        // `FitOptions::default()` is `OdeMethod::Auto`, so without this line a
        // change to the stiffness probe could move the anchor, and the committed
        // companion (`nonmem_anchor/*_fit.ferx`, which sets `ode_method = rk45`)
        // and this test would disagree at birth with nothing to catch it.
        ode_method: OdeMethod::Rk45,
        ode_reltol: 1e-9,
        ode_abstol: 1e-11,
        inner_tol: 1e-6,
        ..Default::default()
    };
    fit(&model, &pop, &model.default_params, &opts)
        .expect("fit must evaluate")
        .ofv
}

#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow + NONMEM-anchored TAD-in-[odes] (#1124) acceptance: opt in with --features slow-tests"
)]
fn mr_shaped_tad_rhs_matches_nonmem() {
    const OFV_TOLERANCE: f64 = 0.5;

    let ofv = ofv_for(" * (1.0 + KTAD * TAD)");
    assert!(
        ofv.is_finite(),
        "OFV must be finite (a NaN here means #1110 has regressed into this cell)"
    );
    let delta = (ofv - NONMEM_OFV).abs();
    assert!(
        delta < OFV_TOLERANCE,
        "ferx OFV {ofv} vs NONMEM {NONMEM_OFV} — delta {delta} exceeds {OFV_TOLERANCE}"
    );

    // Non-degeneracy: the `TAD` term must actually be *doing* something here.
    //
    // This is the assertion the anchor turns on. The closed form's failure mode
    // was not "slightly off" — it returned precisely the numbers of the model with
    // this term deleted, which is what the pre-fix −12.4832 is. If deleting the
    // term moved the objective only a little, agreement above would be weak
    // evidence; it moves it by ~222 units, so it is not.
    let ofv_without = ofv_for("");
    let gap = (ofv_without - ofv).abs();
    assert!(
        gap > 100.0,
        "deleting the `TAD` factor moved the objective by only {gap} — this anchor \
         cannot distinguish an engine that integrates the term from one that drops \
         it, so its agreement above proves nothing"
    );
}
