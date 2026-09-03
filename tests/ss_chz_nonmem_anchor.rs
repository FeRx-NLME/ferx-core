//! NONMEM anchor for an `SS=1` dose on a joint PK-TTE model — issue #1210.
//!
//! The steady-state equilibration used to cycle the injected `d/dt(__chz_<cmt>)` accumulator
//! along with the PK compartments. That row is a pure integrator with no elimination term, so
//! it has no steady state; it just counted up, and its run-in total landed in `H(0)`.
//!
//! **NONMEM has no native *steady-state* comparator here, and that is a measured result.** Handing NONMEM's
//! own SS routine the augmented system (`nonmem_anchor/ss_chz_r1_{const,drug}.ctl`) returns
//! `A(3) = -2.39e7` with `#OBJV = +INF` for the constant-hazard arm and
//! `NUMERICAL DIFFICULTIES WITH STEADY STATE SOLUTION` / `PROGRAM TERMINATED BY OBJ` for the
//! drug-driven one. `H(0) = 0` is therefore a **ferx definition** — the cumulative hazard is
//! measured from the start of the subject's record — and the anchor is the construction that
//! expresses it in NONMEM: `ss_chz_r2_*.ctl`, an explicit 51-dose pulse train at
//! 0, 12, …, 600 with `$DES` gating the hazard on `T >= 600`. At `T = 600` the PK is at its
//! periodic steady state and `A(3)` restarts, so NONMEM `T = 600 + t` is ferx `t`.
//!
//! Three arms, in the order that makes them trustworthy:
//!
//!   * **const** (`BETA = 0`) — the hazard collapses to `H0`, so `A(3)` must equal the closed
//!     form `H0 * t`. Asserted here against the *reference table*, not just against ferx: it
//!     is what certifies the pulse train, the gate and the `T - 600` alignment without an
//!     integrator on either side.
//!   * **drug** (`BETA = 0.5`) — the hazard rides `A(2)/V`. No closed form; this is the oracle
//!     for what the const arm cannot see. Non-degenerate on the incoming side by construction:
//!     the dose is `SS=1`, so drug from the preceding interval is present at every record
//!     (trough 4.78896241), never the `g(x-) = 0` of a first dose.
//!   * **tdep** — drug times a Gompertz baseline `exp(GAM * (T - 600))`, anchored on the
//!     record start. It pins the model clock, which is the quantity an SS run-in displaces:
//!     before the fix ferx evaluated that baseline 600 hours into the run-in and read
//!     `H(1) = 4.5e10` against NONMEM's 1.667.
//!
//! A fourth construction, `ss_chz_r3_{const,drug}.ctl`, anchors the rule the other three
//! cannot reach: that a **second** `SS=1` dose keeps the hazard accrued so far. It needs no SS
//! record on the NONMEM side at all. A second SS dose on the regimen a subject is already at
//! steady state under asserts two things — the compartments re-equilibrate to that same trough,
//! and the accumulated hazard survives — and an uninterrupted 56-dose train through `T = 660`
//! satisfies both by construction. Zeroing the accumulator there (the issue's own suggested
//! fix) disagrees with that train by 99.8% relative on `H`; verified by mutation.
//!
//! `IPRED` is compared too, and it is the control rather than the finding: the PK side was
//! never wrong — ferx matched it to 8 digits *before* the fix — so its agreement localises
//! #1210 to the accumulator row alone. A regression that broke the SS equilibration outright
//! would show up here and nowhere else in this file.
//!
//! Bounds are measured, then set an order of magnitude above the observed residual; both
//! numbers are recorded at each assertion. The model files pin reltol 1e-9 / abstol 1e-11 —
//! at ferx's defaults this comparison is good only to ~1e-3.

#![cfg(feature = "survival")]

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::{predict, predict_survival, CompiledModel, Population};

const DATA: &str = "nonmem_anchor/ss_chz_ss.csv";

/// `H0` in every arm's `$THETA` and `[parameters]` block.
const H0: f64 = 0.02;

/// ferx times of the four compared records — NONMEM `T` minus the pulse train's 600 h.
const GRID: [f64; 4] = [1.0, 4.0, 8.0, 12.0];

fn load(arm: &str) -> (CompiledModel, Population) {
    load_with(arm, DATA)
}

/// The same model files against a different dataset. `r3` reuses the `const` and `drug`
/// models unchanged and swaps only the records, which is what makes it a test of the *dose
/// semantics* rather than of a second model.
fn load_with(arm: &str, data: &str) -> (CompiledModel, Population) {
    let path = format!("nonmem_anchor/ss_chz_{arm}_fit.ferx");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let m = parse_model_string(&src).expect("anchor model must parse");
    let (pop, _) = read_population_for(&m, &None, data, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    (m, pop)
}

/// `nonmem_anchor/ss_chz_ss2.csv`: an `SS=1` dose at `t = 0` **and** a second at `t = 48`.
const DATA_MID_RECORD: &str = "nonmem_anchor/ss_chz_ss2.csv";

/// ferx times of the r3 records — NONMEM `T` minus 600. `47.9`/`48.1` straddle the second
/// dose; `60` is a full interval past it, where a discarded `H(48)` would still be missing.
const GRID_R3: [f64; 4] = [12.0, 47.9, 48.1, 60.0];

/// `(ferx time, IPRED, CHZ, HAZ)` for the four post-gate records of an r2 table.
///
/// Columns are `ID TIME CMT DV EVID IPRED CHZ HAZ MDV`. Only records at `T > 600` are
/// returned — the 51 dose rows and the pre-gate window carry `A(3) = 0` by construction and
/// comparing them would assert the gate against itself.
fn nonmem_rows(arm: &str) -> Vec<(f64, f64, f64, f64)> {
    let path = format!("nonmem_anchor/results/ss_chz_r2_{arm}.tab");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows: Vec<(f64, f64, f64, f64)> = raw
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            (f.len() >= 9 && f[8] == 0.0 && f[1] > 600.0).then(|| (f[1] - 600.0, f[5], f[6], f[7]))
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite times"));
    assert_eq!(
        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        GRID.to_vec(),
        "{arm}: the reference table's post-gate records are not the four expected ones"
    );
    rows
}

/// `(ferx time, IPRED, CHZ, HAZ)` for the four post-gate **observation** records of an r3
/// table. Same column layout and same `T - 600` alignment as [`nonmem_rows`]; the r3 train
/// runs on to `T = 660` and its records straddle the second dose.
fn nonmem_rows_r3(arm: &str) -> Vec<(f64, f64, f64, f64)> {
    let path = format!("nonmem_anchor/results/ss_chz_r3_{arm}.tab");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows: Vec<(f64, f64, f64, f64)> = raw
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            // `f[2]` is CMT: keep the PK observations only, so the TTE row at T = 660 does not
            // collide with the PK row at the same time.
            (f.len() >= 9 && f[8] == 0.0 && f[2] == 2.0 && f[1] > 600.0)
                .then(|| (f[1] - 600.0, f[5], f[6], f[7]))
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite times"));
    // `647.9 - 600.0` is `47.89999999999998`, so the grid is matched to tolerance rather than
    // bit-exactly. Still a strict count-and-position check: a missing, extra or misaligned
    // record fails here before anything is compared.
    let times: Vec<f64> = rows.iter().map(|r| r.0).collect();
    assert_eq!(
        times.len(),
        GRID_R3.len(),
        "{arm}: the r3 table has {} post-gate observation records, expected {}: {times:?}",
        times.len(),
        GRID_R3.len()
    );
    for (got, want) in times.iter().zip(&GRID_R3) {
        assert!(
            (got - want).abs() < 1e-9,
            "{arm}: r3 record at ferx t={got}, expected {want}"
        );
    }
    rows
}

/// **A second `SS=1` dose keeps the hazard accrued so far, and NONMEM says so.**
///
/// This is the rule that separates #1210's fix from the issue's own suggestion ("zero the
/// accumulator"), and it is the one I first judged un-anchorable, because NONMEM's steady-state
/// routine cannot be handed this system at all (`ss_chz_r1_*`). That was too pessimistic: the
/// r2 construction generalises, and the generalisation needs **no SS record on the NONMEM side
/// whatsoever**.
///
/// A second `SS=1` dose at `t = 48`, on the same regimen the subject is already at steady state
/// under, asserts exactly two things: the PK compartments re-equilibrate to that same periodic
/// trough (so nothing changes), and the accumulated hazard is a fact about the record that
/// survives the re-equilibration. A subject who has simply been dosed every 12 h without
/// interruption satisfies both by construction. So `ss_chz_r3_*.ctl` is one uninterrupted
/// 56-dose train through `T = 660` with the hazard gated on at `T = 600` — and ferx's
/// `SS@0 + SS@48` must reproduce it record for record.
///
/// The discrimination is wide, not marginal. Reference `A(3)` on the const arm reads
/// `0.958 -> 0.960 -> 0.962` across the dose and `1.200` a full interval later. Zeroing at the
/// second dose would give `0.002` and `0.240`: every record after `t = 48` short by `H(48)`.
/// The straddle is asserted below so the arm cannot quietly stop discriminating.
///
/// Non-degenerate on the incoming side, which is what the dose-event rule in CLAUDE.md
/// requires: the reference's own `IPRED` reads `4.83708578` at `t = 47.9` against `4.78896241`
/// at the dose times, so drug from the preceding interval is genuinely present when the second
/// dose lands — never the `g(x-) = 0` of a first dose.
#[test]
fn a_second_ss_dose_keeps_the_hazard_and_matches_nonmem() {
    for arm in ["const", "drug"] {
        let (m, pop) = load_with(arm, DATA_MID_RECORD);
        let sv = predict_survival(&m, &pop, &m.default_params, &GRID_R3);
        let pred = predict(&m, &pop, &m.default_params);
        let reference = nonmem_rows_r3(arm);

        // The straddle: the reference must carry a materially non-zero hazard into the second
        // dose, or "preserve" and "zero" agree there and this arm proves nothing.
        let h_before = reference[1].2;
        assert!(
            h_before > 0.9,
            "{arm}: the reference's pre-dose hazard is {h_before}; with nothing accrued this              arm cannot tell preserve from zero"
        );

        let mut worst_h = 0.0f64;
        let mut worst_pred = 0.0f64;
        let mut compared_pred = 0usize;
        for (t, ipred, chz, haz) in reference {
            let r = sv
                .iter()
                .find(|r| (r.time - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("{arm}: no ferx survival row at t={t}"));
            assert!(
                r.cum_hazard.is_finite() && r.hazard.is_finite(),
                "{arm}: ferx returned a non-finite H/h at t={t}: {}, {}",
                r.cum_hazard,
                r.hazard
            );
            worst_h = worst_h.max(((r.cum_hazard - chz) / chz).abs());
            worst_h = worst_h.max(((r.hazard - haz) / haz).abs());

            if let Some(pr) = pred.iter().find(|p| (p.time - t).abs() < 1e-9) {
                assert!(pr.pred.is_finite(), "{arm}: non-finite PRED at t={t}");
                worst_pred = worst_pred.max(((pr.pred - ipred) / ipred).abs());
                compared_pred += 1;
            }
        }
        assert_eq!(
            compared_pred, 4,
            "{arm}: expected all four PK records to be compared, not {compared_pred}"
        );
        // Measured worst relative disagreement: const 4.6e-16 on H/h and 1.0e-9 on PRED,
        // drug 7.8e-9 and 1.0e-9. 2e-8 is ~2.6x the largest, which is the drug arm's H —
        // the reference's nine printed digits on `A(3) = 1.3e2` are themselves worth ~1e-9.
        assert!(
            worst_h < 2e-8,
            "{arm}: worst relative H/h disagreement vs the r3 train = {worst_h}"
        );
        assert!(
            worst_pred < 2e-8,
            "{arm}: worst relative PRED disagreement vs the r3 train = {worst_pred}"
        );
    }
}

/// The reference certifies itself before anything is compared against it: on the constant
/// arm the pulse train's `A(3)` must equal `H0 * t` to the table's printed precision.
///
/// This is the only check in the file whose truth is independent of both engines. If the gate
/// slipped, the train were short, or the `T - 600` alignment were off by an interval, this is
/// what says so — and it says so before the drug and tdep arms, which have no closed form, are
/// allowed to stand on the same construction.
#[test]
fn the_constant_arm_of_the_reference_reproduces_the_closed_form() {
    let mut worst = 0.0f64;
    for (t, _, chz, haz) in nonmem_rows("const") {
        assert!(
            chz.is_finite() && haz.is_finite(),
            "NONMEM table carries a non-finite value at t={t}: CHZ={chz}, HAZ={haz}"
        );
        assert!(
            (haz - H0).abs() < 1e-12,
            "the constant arm's hazard is not constant at t={t}: {haz}"
        );
        worst = worst.max((chz - H0 * t).abs());
    }
    // Measured 0.0 — the table's nine printed digits are exact against `0.02 * t` at every
    // record. The bound is the table's own print resolution (1e-8 relative on `A(3) = 0.24`).
    assert!(
        worst < 1e-8,
        "the reference pulse train does not reproduce H(t) = 0.02*t; worst |A(3) - H0*t| = \
         {worst}"
    );
}

/// ferx's `H(t)` after an `SS=1` dose equals NONMEM's gated pulse-train `A(3)`, on all three
/// arms.
///
/// Before the fix ferx's `H` was displaced by exactly its own `H(0)` at every record —
/// `+12.000000` (const), `+1268.971796` (drug), `+4.536e10` (tdep). The shape was right and
/// only the origin was wrong, so an anchor on the *increments* would have passed throughout;
/// the absolute comparison is what catches it.
#[test]
fn cumulative_hazard_after_an_ss_dose_matches_nonmem() {
    for arm in ["const", "drug", "tdep"] {
        let (m, pop) = load(arm);
        let sv = predict_survival(&m, &pop, &m.default_params, &GRID);
        let mut worst = 0.0f64;
        for (t, _, chz, _) in nonmem_rows(arm) {
            let r = sv
                .iter()
                .find(|r| (r.time - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("{arm}: no ferx survival row at t={t}"));
            // `f64::max` returns the OTHER operand when one side is NaN, so a NaN folded
            // through `.max()` would leave `worst` at whatever the finite records produced
            // and this bound would pass on the strength of the rows that worked. A solver
            // returning NaN is the likeliest way to break the thing being anchored, so the
            // value is checked before it reaches the fold.
            assert!(
                r.cum_hazard.is_finite(),
                "{arm}: ferx returned a non-finite H at t={t}: {}",
                r.cum_hazard
            );
            worst = worst.max(((r.cum_hazard - chz) / chz).abs());
        }
        // Measured worst relative disagreement: const 0, drug 1.5e-9, tdep 1.7e-9 (see the
        // bound rationale in the module docs). 2e-8 is ~12x the largest of those.
        assert!(
            worst < 2e-8,
            "{arm}: worst relative H disagreement vs NONMEM = {worst}"
        );
    }
}

/// The instantaneous hazard at the same records. `H` is an integral, so an error in `h` that
/// changes sign across the interval can cancel out of it; `h` is read pointwise and cannot.
#[test]
fn instantaneous_hazard_after_an_ss_dose_matches_nonmem() {
    for arm in ["const", "drug", "tdep"] {
        let (m, pop) = load(arm);
        let sv = predict_survival(&m, &pop, &m.default_params, &GRID);
        let mut worst = 0.0f64;
        for (t, _, _, haz) in nonmem_rows(arm) {
            let r = sv
                .iter()
                .find(|r| (r.time - t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("{arm}: no ferx survival row at t={t}"));
            assert!(
                r.hazard.is_finite(),
                "{arm}: ferx returned a non-finite h at t={t}: {}",
                r.hazard
            );
            worst = worst.max(((r.hazard - haz) / haz).abs());
        }
        assert!(
            worst < 2e-8,
            "{arm}: worst relative h disagreement vs NONMEM = {worst}"
        );
    }
}

/// The PK control. `IPRED` was already exact before #1210's fix, on all three arms, which is
/// what localises the defect to the accumulator row; keeping it asserted means a change that
/// breaks the *PK* half of the SS equilibration cannot hide behind a green hazard anchor.
#[test]
fn steady_state_predictions_match_nonmem() {
    for arm in ["const", "drug", "tdep"] {
        let (m, pop) = load(arm);
        let pred = predict(&m, &pop, &m.default_params);
        let mut worst = 0.0f64;
        let mut compared = 0usize;
        for (t, ipred, _, _) in nonmem_rows(arm) {
            let Some(r) = pred.iter().find(|r| (r.time - t).abs() < 1e-9) else {
                // The TTE record at t = 12 carries no Gaussian prediction on the ferx side.
                continue;
            };
            assert!(r.pred.is_finite(), "{arm}: non-finite PRED at t={t}");
            worst = worst.max(((r.pred - ipred) / ipred).abs());
            compared += 1;
        }
        assert_eq!(
            compared, 3,
            "{arm}: expected the three PK records to be compared, not {compared}"
        );
        assert!(
            worst < 2e-8,
            "{arm}: worst relative PRED disagreement vs NONMEM = {worst}"
        );
    }
}

/// The run-in does not reach the objective: `H(0) = 0` on the fit path, and `-2 log L` no
/// longer carries `2 * H_runin`.
///
/// The two assertions are one claim measured twice. `H(0) = 0` is the property #1210 is about,
/// read off this model and this dataset rather than a unit fixture. The objective is the same
/// statement where a user would actually notice it, and it is pinned because the contamination
/// entered it *additively* and in closed form: the fixture is one subject with one `SS=1` dose
/// at `II = 12`, so the 50-cycle run-in banked `H_runin = H0 * 50 * II = 12.0` into `H(0)` and
/// the objective carried `2 * H_runin = 24.0` of it.
///
/// Measured on this fixture by running the pre-#1210 engine against it (`M7`'s edits plus
/// `M8`'s): `94.462287664653` before, `70.462287654391` after — a gap of `24.0000000103`
/// against a predicted `24.0`. That closed form is what keeps the pinned number from being a
/// bare change-detector; if it moves, the doc says what the number means.
///
/// The number is ferx-vs-ferx, not a NONMEM value — NONMEM cannot fit this model at all
/// (`ss_chz_r1_const.ctl` returns `#OBJV = +INF`). The NONMEM oracles are the sibling tests on
/// `H`, `h` and `PRED`.
///
/// **The SS-warning check below is a guard, not a discriminator, and the difference is
/// measured.** #1210 did make the exact fixed point decline for every joint model — the
/// accumulator's one-cycle map is the identity, so `I - M` had a zero row — and the capped
/// 50-cycle train then ran with `SsStopTracker` watching a row that never settles. But on
/// *this* fixture the pre-fix engine emits **zero** `Steady-state (SS=1) equilibration`
/// warnings (its five warnings are the FOCEI default, FD inner gradients, the thread count,
/// EPS shrinkage and IWRES autocorrelation). So the assertion holds before the fix as well as
/// after, and it is kept only to catch a future regression that starts warning here. The arm
/// that actually pins the cycle behaviour is the Tier-1
/// `a_nonlinear_joint_model_judges_convergence_on_the_pk_rows`, which asserts the cycle count
/// directly and dies when the equilibration is unmasked.
#[test]
fn a_joint_steady_state_fit_does_not_bank_the_run_in_into_the_objective() {
    use ferx_core::types::{EstimationMethod, FitOptions};

    let (m, pop) = load("const");

    // The property, on the fit path's own model and data. Measured exactly 0.0.
    //
    // Asked for on its own. Until #1218 a grid whose maximum is 0 returned `NaN` and this had
    // to pad the grid with `GRID` to get a finite row; the single-instant path is pinned by
    // `predict_survival_on_a_single_instant_grid_matches_the_multi_point_grid` below.
    let sv = predict_survival(&m, &pop, &m.default_params, &[0.0]);
    let at_zero: Vec<_> = sv.iter().filter(|r| r.time == 0.0).collect();
    assert!(!at_zero.is_empty(), "no survival row at t = 0");
    for r in at_zero {
        assert!(
            r.cum_hazard.is_finite(),
            "non-finite H at t = 0: {}",
            r.cum_hazard
        );
        assert!(
            r.cum_hazard.abs() < 1e-12,
            "the SS run-in is still banked into H(0): {} (pre-#1210 this read 12.0 = H0*50*II)",
            r.cum_hazard
        );
    }

    let opts = FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 0,
        run_covariance_step: false,
        verbose: false,
        ..Default::default()
    };
    let r = ferx_core::fit(&m, &pop, &m.default_params, &opts).expect("joint SS fit runs");

    let ss_warnings: Vec<&String> = r
        .warnings
        .iter()
        .filter(|w| w.contains("Steady-state (SS=1) equilibration"))
        .collect();
    assert!(
        ss_warnings.is_empty(),
        "a joint SS model reported a non-converged equilibration: {ss_warnings:?}"
    );

    assert!(r.ofv.is_finite(), "objective is not finite: {}", r.ofv);
    assert!(
        (r.ofv - 70.462287654391).abs() < 1e-6,
        "objective moved: {} (measured 70.462287654391 after the fix, 94.462287664653 before, \
         a gap of 2 * H_runin = 24.0)",
        r.ofv
    );
}

/// A simulated `SS=1` subject draws a distribution of event times, not a column of zeros.
///
/// This was #1210's loudest symptom and the one no fixture covered. With the run-in banked,
/// the drug-driven arm reached `H(0) = 1268.97`; the inversion `H(t) = -log U` therefore
/// crossed at the first step for every draw, and 20 of 20 simulated subjects had their event
/// at `t = 0.000`.
///
/// The assertion is on the *spread*, not on any particular time: the sampler's own stream is
/// pinned elsewhere, and a bound on individual draws would be a change-detector. `t = 0` is
/// asserted separately because it is not a plausible draw at all — it is the signature of an
/// `H` that starts above every `-log U`.
#[test]
fn a_simulated_steady_state_subject_draws_a_distribution_of_event_times() {
    use ferx_core::api::{simulate_with_options, SimulateOptions};

    let (m, pop) = load("drug");
    let opts = SimulateOptions {
        horizon: Some(24.0),
        seed: Some(7),
        ..Default::default()
    };
    let sim = simulate_with_options(&m, &pop, &m.default_params, 20, &opts)
        .expect("joint SS simulation runs");

    let times: Vec<f64> = sim.iter().filter(|r| r.cmt == 3).map(|r| r.time).collect();
    assert_eq!(times.len(), 20, "one event row per draw");
    for t in &times {
        assert!(t.is_finite(), "a simulated event time is not finite: {t}");
        assert!(
            *t > 0.0,
            "a simulated event landed at t = {t}; an event at the record's own start is the \
             signature of a cumulative hazard that begins above -log U, not a draw"
        );
    }
    let lo = times.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi - lo > 0.5,
        "simulated event times are degenerate: min {lo}, max {hi}"
    );
}

/// `nonmem_anchor/ss_chz_reset.csv`: an `SS=1` dose at `t = 0`, then `EVID=4` (reset + dose)
/// at `t = 6`.
const DATA_RESET: &str = "nonmem_anchor/ss_chz_reset.csv";

/// ferx times of the r4 records — NONMEM `T` minus 600. `4`/`5.9` sit before the reset with
/// the hazard live; `8`/`12` after it, where a preserved `H` would still be too high.
const GRID_R4: [f64; 4] = [4.0, 5.9, 8.0, 12.0];

/// `(ferx time, IPRED, CHZ, HAZ)` for the four post-gate observation records of the r4 table.
fn nonmem_rows_r4() -> Vec<(f64, f64, f64, f64)> {
    let path = "nonmem_anchor/results/ss_chz_r4_const.tab";
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows: Vec<(f64, f64, f64, f64)> = raw
        .lines()
        .skip(2)
        .filter_map(|l| {
            let f: Vec<f64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            (f.len() >= 9 && f[8] == 0.0 && f[2] == 2.0 && f[1] > 600.0)
                .then(|| (f[1] - 600.0, f[5], f[6], f[7]))
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite times"));
    let times: Vec<f64> = rows.iter().map(|r| r.0).collect();
    assert_eq!(
        times.len(),
        GRID_R4.len(),
        "the r4 table has {} post-gate observation records, expected {}: {times:?}",
        times.len(),
        GRID_R4.len()
    );
    for (got, want) in times.iter().zip(&GRID_R4) {
        assert!(
            (got - want).abs() < 1e-9,
            "r4 record at ferx t={got}, expected {want}"
        );
    }
    rows
}

/// **A reset zeroes the accumulated hazard, and NONMEM agrees.**
///
/// Zeroing on reset is a *convention choice*, not a derivation: carrying `H` across a reset is
/// just as implementable, and #1210's fix deliberately keeps a later `SS=1` dose from zeroing
/// while leaving `EVID=3`/`4` as the things that do. Until this arm, that choice was checked
/// only against ferx's own closed form — self-referential in exactly the way the
/// preserve-not-zero rule was before r3, which is the comparison that proved the issue's
/// suggested alternative wrong.
///
/// The r2 harness needs no change: the `$DES` gate is on absolute `T`, so after a reset the
/// hazard keeps accruing while `A(3)` restarts from zero. NONMEM's `EVID=4` resets every
/// compartment, accumulator included, and the reference reads `0.080 / 0.118` before the
/// reset and `0.040 / 0.120` after — i.e. the clock restarts at `t = 6`. Preserving instead
/// would give `0.158` and `0.238`.
///
/// **`EVID=4`, not `EVID=3`, and that is measured rather than stylistic.** An `EVID=3` reset
/// zeroes the PK compartments too, so with no further dose `IPRED = 0` at the post-reset
/// records and the proportional error model has zero variance there — NONMEM returns
/// `PROGRAM TERMINATED BY OBJ` / `ERROR IN CELS`. `EVID=4` keeps drug present afterwards,
/// which the anchor needs anyway to be non-degenerate on the *outgoing* side: the reference's
/// `IPRED` runs `10.45 / 8.78` before and `7.59 / 6.07` after, so the PK is live throughout.
///
/// Const arm only, deliberately: after a reset a drug-driven hazard rides a PK profile that
/// restarted from one dose, so it says nothing the const arm does not, while giving up the
/// exact closed form that makes a convention test readable.
#[test]
fn a_reset_zeroes_the_accumulated_hazard_and_matches_nonmem() {
    let (m, pop) = load_with("const", DATA_RESET);
    let sv = predict_survival(&m, &pop, &m.default_params, &GRID_R4);
    let pred = predict(&m, &pop, &m.default_params);
    let reference = nonmem_rows_r4();

    // The straddle: the reference must carry a materially non-zero hazard into the reset, or
    // "zero" and "preserve" agree there and this arm proves nothing.
    let h_before = reference[1].2;
    assert!(
        h_before > 0.1,
        "the reference's pre-reset hazard is {h_before}; with nothing accrued this arm cannot \
         tell zeroing from preserving"
    );
    // And it must drop across the reset, or the table is not showing a reset at all.
    assert!(
        reference[2].2 < h_before,
        "the reference's hazard did not drop across the reset ({} -> {}); the fixture is not \
         exercising a reset",
        h_before,
        reference[2].2
    );

    let mut worst_h = 0.0f64;
    let mut worst_pred = 0.0f64;
    let mut compared = 0usize;
    for (t, ipred, chz, haz) in reference {
        let r = sv
            .iter()
            .find(|r| (r.time - t).abs() < 1e-9)
            .unwrap_or_else(|| panic!("no ferx survival row at t={t}"));
        assert!(
            r.cum_hazard.is_finite() && r.hazard.is_finite(),
            "ferx returned a non-finite H/h at t={t}: {}, {}",
            r.cum_hazard,
            r.hazard
        );
        worst_h = worst_h.max(((r.cum_hazard - chz) / chz).abs());
        worst_h = worst_h.max(((r.hazard - haz) / haz).abs());

        let pr = pred
            .iter()
            .find(|p| (p.time - t).abs() < 1e-9)
            .unwrap_or_else(|| panic!("no ferx PK row at t={t}"));
        assert!(pr.pred.is_finite(), "non-finite PRED at t={t}");
        worst_pred = worst_pred.max(((pr.pred - ipred) / ipred).abs());
        compared += 1;
    }
    assert_eq!(
        compared, 4,
        "expected all four records to be compared, not {compared}"
    );
    // Measured worst relative disagreement: 3.5e-16 on H/h — the const arm's hazard is exact
    // against the closed form on both sides of the reset — and 2.4e-9 on PRED, which is the
    // reference table's own nine-digit print resolution. 2e-8 is ~8x the larger.
    assert!(
        worst_h < 2e-8,
        "worst relative H/h disagreement vs the r4 train = {worst_h}"
    );
    assert!(
        worst_pred < 2e-8,
        "worst relative PRED disagreement vs the r4 train = {worst_pred}"
    );
}

/// #1218: a grid with no point past the first event returned `NaN, NaN` — `[0.0]`, `[0.0, 0.0]`
/// and the `0` row of `[-1.0, 0.0]` — while the same `t = 0` asked for alongside a later point
/// was finite. Measured before the fix on both arms.
///
/// The single-instant row must be the multi-point row bit for bit. The drug arm is the one that
/// can tell the post-dose state from the seeded one — `h(0) = 0.2193` at the SS trough against
/// the drug-free `H0 = 0.02` — so its `h(0)` is asserted away from `H0` explicitly: the guard
/// against fixing this by widening the pre-first-event prefill (which would return a finite,
/// wrong `h(0)` and pass every other assertion here). The `-1.0` row *is* that prefill's
/// territory and is asserted unchanged: seeded state, drug-free hazard.
#[test]
fn predict_survival_on_a_single_instant_grid_matches_the_multi_point_grid() {
    fn at(rows: &[ferx_core::SurvivalPredictionResult], t: f64) -> Vec<(f64, f64)> {
        rows.iter()
            .filter(|r| r.time == t)
            .map(|r| (r.cum_hazard, r.hazard))
            .collect()
    }
    for arm in ["const", "drug"] {
        let (m, pop) = load(arm);
        let want = at(
            &predict_survival(&m, &pop, &m.default_params, &[0.0, 1.0]),
            0.0,
        );
        assert_eq!(want.len(), 1, "{arm}: one subject, one row at t = 0");
        let (want_h, want_haz) = want[0];
        assert!(
            want_h.is_finite() && want_haz.is_finite(),
            "{arm}: the multi-point reference row is not finite: {want_h}, {want_haz}"
        );
        assert_eq!(want_h, 0.0, "{arm}: H(0) is not exactly zero");

        for grid in [&[0.0][..], &[0.0, 0.0], &[-1.0, 0.0]] {
            let got = at(&predict_survival(&m, &pop, &m.default_params, grid), 0.0);
            let n_zero = grid.iter().filter(|&&t| t == 0.0).count();
            assert_eq!(
                got.len(),
                n_zero,
                "{arm} {grid:?}: one row per requested t = 0"
            );
            for (h, haz) in got {
                assert!(
                    h.is_finite() && haz.is_finite(),
                    "{arm} {grid:?}: non-finite (H, h) at t = 0: {h}, {haz}"
                );
                assert_eq!(
                    h.to_bits(),
                    want_h.to_bits(),
                    "{arm} {grid:?}: H(0) {h} != {want_h}"
                );
                assert_eq!(
                    haz.to_bits(),
                    want_haz.to_bits(),
                    "{arm} {grid:?}: h(0) {haz} != {want_haz}"
                );
            }
        }

        // Before the first event: the seeded state, so no drug and the bare `H0`.
        let pre = at(
            &predict_survival(&m, &pop, &m.default_params, &[-1.0, 0.0]),
            -1.0,
        );
        assert_eq!(pre.len(), 1);
        assert_eq!(pre[0].0, 0.0, "{arm}: H(-1) is not zero");
        assert!(
            (pre[0].1 - H0).abs() < 1e-15,
            "{arm}: h(-1) = {} is not the drug-free hazard {H0}",
            pre[0].1
        );

        if arm == "drug" {
            assert!(
                want_haz > 5.0 * H0,
                "drug arm: h(0) = {want_haz} is not distinguishable from the drug-free {H0}; \
                 this test could not tell a post-dose row from a seeded one"
            );
        }
    }
}
