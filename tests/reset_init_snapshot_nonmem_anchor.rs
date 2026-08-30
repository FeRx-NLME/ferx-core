//! NONMEM anchor for #1133 — which record's `$PK` snapshot re-seeds `init(...)` at an
//! EVID=3/4 reset.
//!
//! #1073 established the rule ferx now follows on every other boundary: **NONMEM runs
//! `$PK` at every data record and ADVANs *to* it, so a segment is governed by the record
//! that terminates it.** An EVID=3/4 row is a data record. ferx did not treat it as one:
//! `ode_predictions_event_driven`'s `Kind::Reset` arm re-seeded the state with `last_pk`,
//! the *previous* record's snapshot, so an `[odes] init(state) = <covariate-driven expr>`
//! restarted the episode on a stale covariate.
//!
//! # Why this needed NONMEM rather than an in-repo oracle
//!
//! Both of the usual twins are blind here, for reasons that are written into the source:
//!
//! * **`Dual2`-vs-FD parity cannot see it.** `sens/ode_provider.rs` mirrors the production
//!   arm deliberately, so the twin differentiates the same convention the value path uses
//!   and both sides move together. That is the documented failure mode in CLAUDE.md and
//!   how #1079 survived.
//! * **An analytic-vs-ODE twin cannot see it either.** The closed-form walk
//!   (`pk/event_driven.rs`) zeroes every compartment unconditionally and has no `init(...)`
//!   concept, so there is no second implementation to disagree.
//!
//! # The control streams
//!
//! `ADVAN13 TOL=9`, one compartment, every `$THETA` `FIX`, `$OMEGA 0 FIX`, `MAXEVAL=0`.
//! `CL = 5` and `V = 50` are plain thetas — **`WT` reaches the prediction only through**
//! `A_0(1) = THETA(3)*WT`, so a divergence here is the reset re-seed's snapshot and
//! nothing else. A 100 mg bolus at `t = 4` puts residual drug in the compartment when the
//! reset arrives at `t = 8`, so the reset's *incoming* side is live (CLAUDE.md's
//! non-degeneracy rule) — NONMEM's arm B shows `9.3208 → 14.0` across it, i.e. the
//! residue is genuinely discarded and replaced by the seed, not added to.
//!
//! | arm | dataset | `WT` at the reset row | `WT` on the next record | `#OBJV` |
//! |---|---|---|---|---|
//! | A | `reset_init_snapshot.csv`       | 140 | 140 | 115.73262077584336 |
//! | B | `reset_init_snapshot_flat.csv`  |  70 |  70 |  56.99027970108172 |
//! | C | `reset_init_snapshot_split.csv` | 140 | **200** | 115.73262077584336 |
//! | D | `reset_init_snapshot_evid4.csv` | 140 (EVID=4, +100 mg) | 200 | 128.80319627590555 |
//! | E | `reset_init_snapshot_multi.csv` | **two** resets: 140 at t=8, 200 at t=12 | — | 234.84197355103731 |
//! | F | `reset_init_snapshot.csv` + `$OMEGA 0.09` on an η inside `A_0` | 140 | 140 | 71.223755303866326 |
//!
//! # Measured (`nonmem_anchor/results/reset_init_snapshot_*.tab`)
//!
//! ```text
//!    t      A            B            C            D
//!  0.0     14.000000    14.000000    14.000000    14.000000
//!  6.0      9.320824     9.320824     9.320824     9.320824   <- residue before the reset
//!  8.0     28.000000    14.000000    28.000000    30.000000   <- reset row (MDV=1)
//!  9.0     25.335448    12.667724    25.335448    27.145123
//! ```
//!
//! Three facts, each readable straight off that table:
//!
//! 1. **NONMEM re-applies `A_0` at a reset.** Arm B seeds `700/50 = 14.0` at `t = 8`
//!    where the decayed residue was `9.32` — so a reset is not "zero and continue".
//! 2. **It uses the reset row's own `$PK`.** Arm A reads `28.0 = (10·140)/50` at `t = 8`.
//!    The previous record carried `WT = 70`, which would have given `14.0` — the number
//!    ferx produced.
//! 3. **Not the next record ahead, either.** Arm C is **bit-identical to A** at every
//!    timepoint (10 decimals) and to the last digit of `#OBJV`, despite carrying
//!    `WT = 200` on every post-reset record. Three candidate conventions, three distinct
//!    predictions at `t = 8` (14.0 / 28.0 / 40.0), and NONMEM picks the middle one.
//!
//! Arm D adds the EVID=4 case, where the reset and its dose share one row: `30.0 =
//! (10·140 + 100)/50`, i.e. the row's own `WT` seeds *and* the co-timed dose then lands on
//! the re-seeded state.
//!
//! A and C are what make this an anchor rather than a fixture. B alone would be satisfied
//! by any convention (the covariate is flat), and A alone is satisfied by both "the reset
//! row's own snapshot" and "the next record ahead" — the natural LOCF dataset makes those
//! two agree. C is the one dataset that separates them, and it is why the fix resolves the
//! reset row's *own* covariates rather than reusing `governing_record`, which would have
//! passed A and B and failed only here.
//!
//! # Arms E and F: what A–D still could not see
//!
//! **E — two resets.** Every one of A–D has exactly one reset, so `pk_at_reset[idx]` is
//! always `pk_at_reset[0]` and the per-reset index is unpinned: replacing `[idx]` with
//! `[0]` in either engine passes all of them. E carries a second reset at `t = 12` with
//! `WT = 200`, so the two seeds are 1400 and 2000 and the engines must index them apart —
//! `IPRED(13) = 36.193497` against the first reset's snapshot's `25.335448`.
//!
//! **F — the objective, not just the prediction.** A–E are `$OMEGA 0 FIX` with an η-free
//! `A_0`, so they pin the *value* path only; a defect confined to the `Dual2` reset seed
//! (`init_taylor_seed_at` at the `K_RESET` branch) would be caught by nothing outside the
//! repo. F puts `ETA(1)` inside `A_0` under `$OMEGA 0.09` and reads `#OBJV` with `POSTHOC`.
//! The FOCEI `h` matrix is the analytic Jacobian, so a wrong `∂init/∂η` at the reset moves
//! the EBE and therefore the objective — which makes the sensitivity path externally
//! anchorable rather than only twin-checked. That the EBE genuinely moves is visible in
//! NONMEM's own table: `IPRED(0) = 10.328` against `PRED(0) = 14.0`.
//!
//! Raw NONMEM output for all six arms is committed under
//! `nonmem_anchor/results/reset_init_snapshot_{A,B,C,D,E,F}.tab`, so every constant below
//! is auditable against the file it came from.
//!
//! Tier 2: `predict` at fixed parameters, no convergence loop, so no `slow-tests` gate.
//! This is deliberate — #1132 records that the slow tier never runs on a PR.

use std::path::{Path, PathBuf};

fn anchor(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// Population `PRED` (η = 0) at every observation row of `data`, in file order.
/// `$OMEGA 0 FIX` on the NONMEM side and `omega ETA_CL ~ 0.0` here make `PRED`
/// and `IPRED` the same number, so this reads against either column.
fn ferx_preds(data: &str) -> Vec<(f64, f64)> {
    let parsed = ferx_core::parser::model_parser::parse_full_model_file(Path::new(&anchor(
        "reset_init_snapshot_fit.ferx",
    )))
    .expect("the anchor model parses");
    let pop = ferx_core::read_nonmem_csv(Path::new(&anchor(data)), None, None)
        .expect("the anchor dataset loads");
    ferx_core::predict(&parsed.model, &pop, &parsed.model.default_params)
        .iter()
        .map(|p| (p.time, p.pred))
        .collect()
}

/// NONMEM `IPRED` at the **observation** rows (`MDV = 0`) of each dataset. The `t = 4`
/// dose row and the `t = 8` reset row are `MDV = 1`, so ferx emits no prediction there
/// and they are documented in the module header instead.
const NM_A: &[(f64, f64)] = &[
    (0.0, 1.400_000_000_0e1),
    (1.0, 1.266_772_384_0e1),
    (2.0, 1.146_223_052_1e1),
    (6.0, 9.320_824_388_2e0),
    (9.0, 2.533_544_769_1e1),
    (10.0, 2.292_446_106_2e1),
    (12.0, 1.876_896_126_7e1),
    (16.0, 1.258_121_097_3e1),
];

const NM_B: &[(f64, f64)] = &[
    (0.0, 1.400_000_000_0e1),
    (1.0, 1.266_772_384_0e1),
    (2.0, 1.146_223_052_1e1),
    (6.0, 9.320_824_388_2e0),
    (9.0, 1.266_772_384_6e1),
    (10.0, 1.146_223_053_1e1),
    (12.0, 9.384_480_633_7e0),
    (16.0, 6.290_605_486_7e0),
];

const NM_D: &[(f64, f64)] = &[
    (0.0, 1.400_000_000_0e1),
    (1.0, 1.266_772_384_0e1),
    (2.0, 1.146_223_052_1e1),
    (6.0, 9.320_824_388_2e0),
    (9.0, 2.714_512_252_6e1),
    (10.0, 2.456_192_256_7e1),
    (12.0, 2.010_960_135_8e1),
    (16.0, 1.347_986_890_0e1),
];

/// NONMEM `IPRED` for arm E, the two-reset dataset. The second reset at `t = 12` seeds
/// `2000/50 = 40.0`; had the engine reused the FIRST reset's snapshot it would seed 28.0
/// and read `25.335448` at `t = 13` instead of `36.193497`.
const NM_E: &[(f64, f64)] = &[
    (0.0, 1.400_000_000_0e1),
    (1.0, 1.266_772_384_0e1),
    (2.0, 1.146_223_052_1e1),
    (6.0, 9.320_824_388_2e0),
    (9.0, 2.533_544_769_1e1),
    (10.0, 2.292_446_106_2e1),
    (13.0, 3.619_349_670_8e1),
    (14.0, 3.274_923_010_0e1),
    (16.0, 2.681_280_182_3e1),
];

/// Relative tolerance against the NONMEM table. The two solvers are independent
/// implementations at `TOL=9` / `reltol 1e-10`, so this is a numerical-agreement band,
/// not a convention band: the defect under test is a factor of **two** at every
/// post-reset timepoint, five orders of magnitude clear of it.
const REL_TOL: f64 = 1e-7;

fn assert_matches_nonmem(data: &str, expected: &[(f64, f64)]) {
    let got = ferx_preds(data);
    assert_eq!(
        got.len(),
        expected.len(),
        "{data}: ferx returned {} predictions, NONMEM table has {} observation rows \
         (times {:?})",
        got.len(),
        expected.len(),
        got.iter().map(|(t, _)| *t).collect::<Vec<_>>()
    );
    for ((t_got, p), (t_exp, nm)) in got.iter().zip(expected) {
        assert!(
            (t_got - t_exp).abs() < 1e-9,
            "{data}: prediction {t_got} does not line up with NONMEM row {t_exp}"
        );
        let rel = (p - nm).abs() / nm.abs();
        assert!(
            rel < REL_TOL,
            "{data} @ t={t_exp}: ferx {p:.10} vs NONMEM {nm:.10} (rel {rel:.2e}). \
             Before #1133's fix the post-reset points were a factor of two out, because \
             the reset re-seeded `init(...)` with the PREVIOUS record's covariate snapshot."
        );
    }
}

#[test]
fn reset_reseeds_init_from_the_reset_rows_own_covariates() {
    // A: `WT` steps 70 -> 140 *at* the reset row and stays there — the natural LOCF
    // shape. Post-reset predictions double relative to the stale-snapshot answer.
    assert_matches_nonmem("reset_init_snapshot.csv", NM_A);
}

#[test]
fn a_flat_covariate_across_the_reset_is_unchanged() {
    // B: the control. Every candidate convention agrees when `WT` does not move, so
    // this pins that the fix is reading the reset row and not perturbing the walk.
    assert_matches_nonmem("reset_init_snapshot_flat.csv", NM_B);
}

#[test]
fn the_reset_seed_ignores_the_record_after_it() {
    // C, the discriminator. Identical to A except every post-reset record carries
    // `WT = 200` instead of 140. NONMEM's answer does not move, so the seed is the
    // reset row's *own* snapshot — not the next record ahead, which is what #1073's
    // `governing_record` resolution would have handed it.
    assert_matches_nonmem("reset_init_snapshot_split.csv", NM_A);
}

#[test]
fn split_and_natural_datasets_agree_prediction_for_prediction() {
    // The invariance form of the test above, asserted directly between the two ferx runs.
    // Given the two `assert_matches_nonmem` neighbours this is implied rather than
    // independent — it earns its place on tolerance, not logic: `1e-9` absolute is ~2500x
    // tighter than `REL_TOL` at these magnitudes, so a sub-threshold leak of the following
    // record into the seed shows up here first.
    let a = ferx_preds("reset_init_snapshot.csv");
    let c = ferx_preds("reset_init_snapshot_split.csv");
    assert_eq!(
        a.len(),
        c.len(),
        "the two datasets have the same observation rows"
    );
    for ((t, pa), (_, pc)) in a.iter().zip(&c) {
        assert!(
            (pa - pc).abs() < 1e-9,
            "t={t}: changing `WT` on the records AFTER the reset moved the prediction \
             ({pa:.10} vs {pc:.10}); NONMEM gives the same value for both."
        );
    }
}

/// ferx FOCEI OFV at the same fixed thetas NONMEM evaluated (`maxiter = 0`, covariance
/// off — both set in `reset_init_snapshot_ofv.ferx`), on one subject with eight
/// observations. Cheap enough to stay Tier 2.
fn ferx_ofv(data: &str) -> f64 {
    let (result, _pop) = ferx_core::run_model_with_data(
        &anchor("reset_init_snapshot_ofv.ferx"),
        Some(&anchor(data)),
    )
    .expect("the OFV anchor model and dataset load and evaluate");
    result.ofv
}

/// `nonmem_anchor/results/reset_init_snapshot_F.tab` / its `.ext`.
const NM_OBJV_F: f64 = 71.223_755_303_866_326;

#[test]
fn the_objective_matches_nonmem_with_an_eta_inside_the_reset_seed() {
    // F. Every other arm is `$OMEGA 0 FIX`, so they pin the value path and leave the
    // `Dual2` reset seed anchored only by an in-repo twin. Here `ETA_BASE` sits inside
    // `init(central)`, so the EBE search reads `∂init/∂η` evaluated at the reset row's
    // snapshot and a wrong one moves the objective.
    //
    // The repo's standard anchor tolerance, half an OFV unit. Seeding from the previous
    // record instead (`WT = 70` rather than 140) halves the post-reset baseline and moves
    // this by far more than that.
    let ofv = ferx_ofv("reset_init_snapshot.csv");
    let delta = (ofv - NM_OBJV_F).abs();
    assert!(
        delta < 0.5,
        "OFV anchor: ferx {ofv:.6} vs NONMEM {NM_OBJV_F:.6} (Δ {delta:.3e})"
    );
}

#[test]
fn each_reset_seeds_from_its_own_row() {
    // E: two resets, `WT` 140 then 200. This is the arm that pins the per-reset *index* —
    // with a single reset everywhere else, `pk_at_reset[idx]` and `pk_at_reset[0]` are the
    // same expression and the plumbing is unverified.
    assert_matches_nonmem("reset_init_snapshot_multi.csv", NM_E);
}

#[test]
fn evid4_seeds_from_its_own_row_then_lands_its_dose() {
    // D: reset and dose share one row. `30.0 = (10*140 + 100)/50` at `t = 8` — the
    // row's own `WT` seeds, then the co-timed dose lands on the re-seeded state.
    assert_matches_nonmem("reset_init_snapshot_evid4.csv", NM_D);
}
