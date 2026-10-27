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
//! Four arms, and they are **not** symmetric:
//!
//!   * **A1** — a dose, a PK observation and the event. Qualifies for the share, so this is
//!     the arm that was red: `TENTRY = 5` returned the sentinel while `TENTRY = 0` returned
//!     a finite objective.
//!   * **A4** — A1 plus an `EVID=3` reset, which `try_joint_pktte_shared_solve` declines
//!     (`subject.has_resets()`), routing it to the dedicated engine. Green before the fix.
//!     It is the **control**: it pins that the fix did not move the arm that was already
//!     right, and it is what makes "both engines agree" a statement about two engines.
//!
//!   * **A5** — A1 plus an `MDV=1` row at `t = 5`. Since #1809 that row is a record: the
//!     reader keeps its time in `pk_only_times`, so it starts the integration, and with it
//!     the hazard clock, as NONMEM's first record of any `EVID` does. Before the dose at 10
//!     there is no drug, so `h = H0` on `[5, 10]`, and the objective rises over A1 by
//!     exactly `2 · H0 · 5`, a closed form. A5 with an `EVID=2` row in place of the
//!     `MDV=1` row is the same record and must score bit-identically. Before #1809 the
//!     reader dropped the row, and A5 was bit-identical to A1.
//!   * **A6** — A1 plus an `EVID=3` reset at 25, after the event, with and without A5's
//!     record. The reset sends it to the dedicated engine, and the same closed form must
//!     hold there. A4 cannot carry this check, because its reset at 15 zeroes the
//!     accumulated hazard before the event.
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

/// Measured on this branch with the fix in place. Pinned to `1e-6` relative: the bit-equality
/// legs below are what test the fix, and this is the guard against both sides of such a pair
/// drifting together to some other objective.
const A1_OFV: f64 = 24.417862686939927;
/// Measured the same way. Unchanged by the fix — A4 declines the share and runs the dedicated
/// engine, which already filled the pre-start node.
const A4_OFV: f64 = 16.97903614243416;

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

/// A5: A1 plus an `EVID=0, MDV=1` row at `t = 5`.
fn arm_a5(entry: f64) -> String {
    arm_a5_evid(entry, 0)
}

/// A6: A1 plus an `EVID=3` reset at 25, *after* the event, and optionally the `EVID=0,
/// MDV=1` record at `t = 5`. The reset makes `try_joint_pktte_shared_solve` decline, so
/// this arm runs the dedicated engine. Unlike A4's reset at 15, it cannot zero the hazard
/// accumulated before the event, so the record's `[5, 10]` window reaches `H(20)`.
fn arm_a6(record: bool) -> String {
    let record_row = if record {
        "1,5,.,0,.,2,0,1,0,0,0\n"
    } else {
        ""
    };
    format!(
        "{HEADER}{record_row}\
         1,10,.,1,100,1,0,1,0,0,0\n\
         1,12,5,0,.,2,0,0,0,0,0\n\
         1,20,1,0,0,3,0,0,0,0,0\n\
         1,25,.,3,0,1,0,1,0,0,0\n"
    )
}

/// A5 with the record at `t = 5` written with `EVID = evid` (`0` with `MDV=1`, or `2`).
fn arm_a5_evid(entry: f64, evid: u32) -> String {
    format!(
        "{HEADER}\
         1,5,.,{evid},.,2,0,1,0,0,0\n\
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
    // Bit-exact, not merely close: the extra CHZ node the `TENTRY` adds is filled before the
    // break walk and lies in no segment, and it does not move the horizon, so the integration
    // is identical. Measured — see the sibling assertion in `prestart_entry_time_contributes_
    // nothing_post_start_entry_does`, which reports a difference of exactly `0.0`.
    assert_eq!(
        pre.to_bits(),
        none.to_bits(),
        "A1: a pre-start TENTRY must contribute nothing — TENTRY=5 {pre} vs TENTRY=0 {none}"
    );
    assert!(
        (none - A1_OFV).abs() <= 1e-6 * A1_OFV.abs(),
        "A1 objective moved: {none} vs the pinned {A1_OFV}"
    );

    // The other side of the gate, or the test above is satisfied by an implementation that
    // ignores `TENTRY` entirely — never pushing it into the shared solve's CHZ times, never
    // subtracting `H(entry)`. That implementation also passes A4 and A5, so nothing else in this
    // file would notice. The first record is the dose at 10, so 12 is post-start and must move
    // the objective by `H(12) > 0`.
    let post = ofv(&arm_a1(12.0));
    assert!(
        post.is_finite(),
        "A1 with a post-start TENTRY must be finite: {post}"
    );
    assert!(
        (post - none).abs() > 1e-6,
        "A1: a post-start TENTRY must change the objective — TENTRY=12 {post} vs TENTRY=0 {none}"
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
    assert_eq!(
        pre.to_bits(),
        none.to_bits(),
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

/// An `MDV=1` row before the first dose is a record, so it starts the hazard clock (#1809).
/// NONMEM integrates from the subject's first record of any `EVID`. Before the fix the reader
/// dropped the row and the objective was bit-identical to A1, the claim
/// `docs/estimation/tte.qmd` used to make.
///
/// Closed form, computed outside both engines: no drug is present before the dose at 10, so
/// `h = H0 = 0.02` on `[5, 10]`, and `H(20)` gains `0.02 · 5 = 0.1`. The event is exact, so
/// the objective (`-2 log L`) gains `2 · 0.1 = 0.2`. The pre-#1809 behaviour, `ΔOFV = 0`,
/// is on the other side of the bound by the whole 0.2.
#[test]
fn mdv_one_row_before_the_first_dose_starts_the_clock() {
    for entry in [0.0_f64, 5.0] {
        let (with_row, without) = (ofv(&arm_a5(entry)), ofv(&arm_a1(entry)));
        assert!(
            with_row.is_finite() && without.is_finite(),
            "A5/A1 objectives must be finite at TENTRY={entry}: {with_row}, {without}"
        );
        // Not the `1e20` sentinel either: both sides are near the A1 objective.
        assert!(
            (without - A1_OFV).abs() < 1.0 && (with_row - A1_OFV).abs() < 1.0,
            "TENTRY={entry}: an objective was repelled ({with_row}, {without}; A1 {A1_OFV})"
        );
        // Measured on Linux x86_64 at the model's `ode_reltol = 1e-9`: ΔOFV − 0.2 = −4.6e-10
        // at both TENTRY values. The bound is ~200x that.
        let delta = with_row - without;
        assert!(
            (delta - 0.2).abs() < 1e-7,
            "TENTRY={entry}: the MDV=1 record at 5 must start the clock: want ΔOFV = 2·H0·5 \
             = 0.2 over A1, got {delta} ({with_row} vs {without})"
        );
        // An EVID=2 row in its place is the same record: the reader builds the same subject.
        let evid2 = ofv(&arm_a5_evid(entry, 2));
        assert_eq!(
            evid2.to_bits(),
            with_row.to_bits(),
            "TENTRY={entry}: an EVID=2 and an EVID=0/MDV=1 row at 5 must score identically \
             ({evid2} vs {with_row})"
        );
    }
}

/// The same clock on the dedicated two-solve engine (A6). Its reset at 25 makes
/// `try_joint_pktte_shared_solve` decline, so the record at 5 reaches the
/// `ode_dense_solve_states` start instead of the shared solve's. The share arm passing alone
/// would leave this engine's start unchecked. The closed form is unchanged: `h = H0` on
/// `[5, 10]`, ΔOFV = 0.2.
///
/// A4 cannot carry this check, and a first version of this test tried: its reset at 15
/// zeroes every state, the `__chz` accumulator included, so the `[5, 10]` window is wiped
/// before the event at 20. ΔOFV there is 0 by construction (measured −4.0e-10), which is
/// the old behaviour's answer too.
#[test]
fn mdv_one_row_before_the_first_dose_starts_the_dedicated_engine_clock() {
    let (with_row, without) = (ofv(&arm_a6(true)), ofv(&arm_a6(false)));
    assert!(
        with_row.is_finite() && without.is_finite() && (with_row - without).abs() < 1.0,
        "A6 objectives must be finite and not repelled: {with_row}, {without}"
    );
    // Measured on Linux x86_64: ΔOFV − 0.2 = −4.6e-10, as on the shared engine.
    let delta = with_row - without;
    assert!(
        (delta - 0.2).abs() < 1e-7,
        "dedicated engine: the MDV=1 record at 5 must start the clock: want ΔOFV = 0.2, got \
         {delta} ({with_row} vs {without})"
    );
}
