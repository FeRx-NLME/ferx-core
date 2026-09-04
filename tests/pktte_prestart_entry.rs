//! End-to-end: a joint PK-TTE subject whose `TENTRY` falls **before its first record**
//! must score, and must score the same on both hazard engines — issue #1223.
//!
//! The one-solve share (#570) NaN-filled a `chz_times` entry below the integration start;
//! `tte_ode_nll_from_shared` skips a non-finite state and `tte_nll_from_curves` maps the
//! resulting NaN `H` to its `1e20` sentinel. The dedicated two-solve engine
//! (`ode_dense_solve_states`) has filled the same node with the seeded state since the CTMM
//! scorer needed it. So the objective a subject got depended on which engine
//! `try_joint_pktte_shared_solve` admitted it to — a question about resets and covariates,
//! not about where its `TENTRY` falls.
//!
//! Two arms, and they are **not** symmetric:
//!
//!   * **A1** — a dose, a PK observation and the event. Qualifies for the share, so this is
//!     the arm that was red: `TENTRY = 5` returned the sentinel while `TENTRY = 0` returned
//!     a finite objective.
//!   * **A4** — A1 plus an `EVID=3` reset, which `try_joint_pktte_shared_solve` declines
//!     (`subject.has_resets()`), routing it to the dedicated engine. Green before the fix.
//!     It is the **control**: it pins that the fix did not move the arm that was already
//!     right, and it is what makes "both engines agree" a statement about two engines.
//!
//!   * **A5** — A1 plus an `MDV=1` row at `t = 5`. The reader drops such a row outright
//!     (`datareader.rs`, the observation arm is `else if evid == 0 && mdv == 0`, and the
//!     only other non-dose arm needs `EVID=2` *and* time-varying covariates), so it never
//!     reaches `obs_times`, `pk_only_times` or `obs_records` and cannot move the
//!     integration start. Asserted **bit-identically** against A1, because a dropped row
//!     leaves a bit-identical `Population`. This is a *reader* claim — it is the one the
//!     docs sentence in `docs/estimation/tte.qmd` makes — not a test of the fix.
//!
//! The population is loaded through the **routed** `read_population_for`, not
//! `read_nonmem_csv`: since #1199 the model-blind reader builds no event records at all, so
//! a `read_nonmem_csv` load would produce a subject with no TTE record and quietly test
//! nothing.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{fit, EstimationMethod, FitOptions};
use std::io::Write;

/// FIXed thetas, `[event_model] cmt = 3` on a `[depot, central]` PK block, so CMT 1 doses,
/// CMT 2 is the PK observation compartment and CMT 3 the ODE-accumulated hazard.
const MODEL: &str = "nonmem_anchor/ss_chz_drug_fit.ferx";

const HEADER: &str = "ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,SS,II,TENTRY\n";

/// A1: dose at 10, PK observation at 12, exact event at 20 with `TENTRY = entry`.
/// The first record is the dose, so any `0 < entry < 10` is pre-start.
fn arm_a1(entry: f64) -> String {
    format!(
        "{HEADER}\
         1,10,.,1,100,1,0,1,0,0,0\n\
         1,12,5,0,.,2,0,0,0,0,0\n\
         1,20,1,0,0,3,0,0,0,0,{entry}\n"
    )
}

/// A4: A1 plus an `EVID=3` reset at 15 and a re-dose at 16 — the reset is what makes
/// `try_joint_pktte_shared_solve` decline, so this arm runs the dedicated engine.
fn arm_a4(entry: f64) -> String {
    format!(
        "{HEADER}\
         1,10,.,1,100,1,0,1,0,0,0\n\
         1,12,5,0,.,2,0,0,0,0,0\n\
         1,15,.,3,0,1,0,1,0,0,0\n\
         1,16,.,1,100,1,0,1,0,0,0\n\
         1,20,1,0,0,3,0,0,0,0,{entry}\n"
    )
}

/// A5: A1 plus an `MDV=1` row at `t = 5`, which the reader drops entirely.
fn arm_a5(entry: f64) -> String {
    format!(
        "{HEADER}\
         1,5,.,0,.,2,0,1,0,0,0\n\
         1,10,.,1,100,1,0,1,0,0,0\n\
         1,12,5,0,.,2,0,0,0,0,0\n\
         1,20,1,0,0,3,0,0,0,0,{entry}\n"
    )
}

/// Objective at the model file's FIXed initial values: no outer iterations, no covariance
/// step. The ODE tolerances are set to the model file's own `1e-9` / `1e-11` because `fit`
/// carries *its* `FitOptions` the last hop to the integrator (#1212) and would otherwise
/// override what `parse_model_string` baked onto the spec.
fn ofv(csv: &str) -> f64 {
    let mut f = tempfile::NamedTempFile::new().expect("create temp csv");
    f.write_all(csv.as_bytes()).expect("write temp csv");

    let src = std::fs::read_to_string(MODEL).unwrap_or_else(|e| panic!("{MODEL}: {e}"));
    let model = parse_model_string(&src).expect("anchor model must parse");
    let path = f.path().to_str().expect("temp path is utf-8");
    let (pop, _) = read_population_for(&model, &None, path, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    assert_eq!(pop.subjects.len(), 1, "fixture is a single subject");

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        ode_reltol: 1e-9,
        ode_abstol: 1e-11,
        ..FitOptions::default()
    };
    let res = fit(&model, &pop, &model.default_params, &opts).expect("maxiter-0 fit must run");
    res.ofv
}

/// The share-admitted arm: a pre-start `TENTRY` must give the same objective as no
/// truncation at all, because `H(entry) = 0` there. Red before the fix — `TENTRY = 5`
/// returned the `1e20` sentinel.
#[test]
fn prestart_entry_matches_no_entry_on_the_shared_engine() {
    let (pre, none) = (ofv(&arm_a1(5.0)), ofv(&arm_a1(0.0)));
    assert!(
        pre.is_finite() && none.is_finite(),
        "A1 objectives must be finite: TENTRY=5 {pre}, TENTRY=0 {none}"
    );
    assert!(
        (pre - none).abs() <= 1e-9 * none.abs().max(1.0),
        "A1: a pre-start TENTRY must contribute nothing — TENTRY=5 {pre} vs TENTRY=0 {none}"
    );
    assert!(
        (none - A1_OFV).abs() <= 1e-6 * A1_OFV.abs(),
        "A1 objective moved: {none} vs the pinned {A1_OFV}"
    );
}

/// The control: the same subject with an `EVID=3` reset declines the share and runs the
/// dedicated engine, which already filled the pre-start node. Green before the fix; here to
/// pin that the fix did not move it, and that the two engines land on the same convention.
#[test]
fn prestart_entry_matches_no_entry_on_the_dedicated_engine() {
    let (pre, none) = (ofv(&arm_a4(5.0)), ofv(&arm_a4(0.0)));
    assert!(
        pre.is_finite() && none.is_finite(),
        "A4 objectives must be finite: TENTRY=5 {pre}, TENTRY=0 {none}"
    );
    assert!(
        (pre - none).abs() <= 1e-9 * none.abs().max(1.0),
        "A4: a pre-start TENTRY must contribute nothing — TENTRY=5 {pre} vs TENTRY=0 {none}"
    );
    assert!(
        (none - A4_OFV).abs() <= 1e-6 * A4_OFV.abs(),
        "A4 objective moved: {none} vs the pinned {A4_OFV}"
    );
    // The reset arm must not accidentally coincide with A1 — if it did, the "two engines"
    // claim would rest on one engine having been reached twice.
    assert!(
        (A4_OFV - A1_OFV).abs() > 1e-3,
        "A4 must be a materially different subject from A1 ({A4_OFV} vs {A1_OFV})"
    );
}

/// An `MDV=1` row before the first dose does not start the hazard clock — the reader drops
/// it, so the objective is *bit-identical* to the same dataset without it. Pins the reader
/// claim the `docs/estimation/tte.qmd` sentence makes; it does not exercise the fix.
#[test]
fn mdv_one_row_before_the_first_dose_changes_nothing() {
    for entry in [0.0_f64, 5.0] {
        let (with_row, without) = (ofv(&arm_a5(entry)), ofv(&arm_a1(entry)));
        assert!(
            with_row.is_finite() && without.is_finite(),
            "A5/A1 objectives must be finite at TENTRY={entry}: {with_row}, {without}"
        );
        assert_eq!(
            with_row.to_bits(),
            without.to_bits(),
            "TENTRY={entry}: a dropped MDV=1 row must leave the objective bit-identical \
             ({with_row} vs {without})"
        );
    }
}

/// Measured on this branch with the fix in place (`24.417862686939927`). Pinned to `1e-6`
/// relative: the equality legs above are what test the fix, and this is the guard against
/// both of them drifting together to some other objective.
const A1_OFV: f64 = 24.417862686939927;
/// Measured the same way (`16.97903614243416`). Unchanged by the fix — this arm declines
/// the share and runs the dedicated engine, which already filled the pre-start node.
const A4_OFV: f64 = 16.97903614243416;
