//! NONMEM anchor for an `SS=1` dose on a joint PK-TTE model — issue #1210.
//!
//! The steady-state equilibration used to cycle the injected `d/dt(__chz_<cmt>)` accumulator
//! along with the PK compartments. That row is a pure integrator with no elimination term, so
//! it has no steady state; it just counted up, and its run-in total landed in `H(0)`.
//!
//! **NONMEM has no native comparator here, and that is a measured result.** Handing NONMEM's
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
    let path = format!("nonmem_anchor/ss_chz_{arm}_fit.ferx");
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let m = parse_model_string(&src).expect("anchor model must parse");
    let (pop, _) = read_population_for(&m, &None, DATA, None, None, None, &[])
        .expect("endpoint-routed load must succeed");
    (m, pop)
}

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

/// `fit()` on a joint `SS=1` model emits no steady-state non-convergence warning.
///
/// It used to emit two. The accumulator's one-cycle map is the identity, so `I - M` was
/// singular, the exact fixed point declined for every joint model, and the capped 50-cycle
/// train ran — with `SsStopTracker` watching a row that grows by `H0 * II` every cycle and so
/// never converging. The warnings were real reports of a real failure to converge; they are
/// gone because the failure is.
///
/// Paired with the objective, because "no warning" alone is satisfiable by deleting the
/// warning. The constant-hazard arm's contribution is `-log S - log h` with `H(12) = 0.24` and
/// `h = 0.02` — a small, finite number. Before the fix the same fit reported `83.592027`
/// against `59.592027` now, the difference being exactly `2 x H_runin = 24.0`.
#[test]
fn a_joint_steady_state_fit_no_longer_reports_a_non_converged_equilibration() {
    use ferx_core::types::{EstimationMethod, FitOptions};

    let (m, pop) = load("const");
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
        (r.ofv - 59.592027).abs() < 1e-4,
        "objective moved: {} (pre-#1210 this fixture read 83.592027, i.e. 2 x H_runin higher)",
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
