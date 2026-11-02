//! mrgsolve cross-check for **schedule adaptation** (#945, epic #391): a fixed review grid,
//! a due mask, and a dosing interval the controller moves on a ladder.
//!
//! NONMEM has no native feedback dosing, so the external anchor for this feature family is
//! **mrgsolve**. The kit lives in `tests/reference/vanco_interval_mrgsolve/`:
//!
//!   - `vanco_interval_mrgsolve.R` runs a 1-cpt IV vancomycin model (fixed 1000 mg, 1-h
//!     infusion, `CL = 4·RF`) over a q12 review grid in an R loop that mirrors the driver's
//!     due mask exactly: a grid point is due once the current interval has passed since the
//!     last consulted decision; at a due point the first matching rule picks hold / dose and
//!     lengthen / shorten on the ladder 12/24/36/48 h; a consulted decision starts the next
//!     interval; a not-due point only reads the trough.
//!   - `expected.md` is the frozen per-grid-point log: trough, outcome, interval after,
//!     and each consulted trough's distance to its nearest threshold.
//!
//! The ferx side is `vanco_interval.ferx` driven by a programmatic controller below, with
//! `start_interval = 12` and `DoseAction::SetInterval` for every rung change. Each subject
//! runs at a **supplied η = 0** (`SuppliedEtas`), the typical-value patient mrgsolve
//! simulates: the model's `omega ~ 1e-10` still draws an η of order 1e-5 on CL, and that
//! drift — not either solver — was the whole of a 3.8e-4 mg/L one-sided gap at t = 288
//! when the anchor drew it.
//!
//! Non-degeneracy: two subjects, multi-dose with drug on board at every later dose.
//! Subject 1 (RF 0.15) climbs to the top rung, stays pinned there and holds at due points
//! (once on the top rung). Subject 2 (RF 0.5) lengthens twice, shortens, and lengthens
//! again. Decision 0 of both shortens at the bottom rung. Every consulted trough is at
//! least 0.13 mg/L from its nearest threshold, so a cross-solver difference cannot flip a
//! decision: the outcome and interval sequences must match exactly.

use ferx_core::parser::model_parser::parse_full_model_file;
use ferx_core::{
    read_nonmem_csv, simulate_adaptive_with_inputs, AdaptiveInputs, AdaptiveSimulateOptions,
    ControllerCtx, DecisionOutcome, DoseAction, MonitorSpec, ObserveMode, SuppliedEtas,
};
use std::path::Path;

const KIT: &str = "tests/reference/vanco_interval_mrgsolve";
const LADDER: [f64; 4] = [12.0, 24.0, 36.0, 48.0];
const DOSE: f64 = 1000.0;
const TH_HOLD: f64 = 25.0;
const TH_LEN: f64 = 14.0;
const TH_SHORT: f64 = 9.0;

/// One row of `expected.md`: `(time, trough, outcome, interval after)`.
type RefRow = (f64, f64, String, f64);

/// The frozen per-subject logs, read from the kit's `expected.md`.
fn reference() -> Vec<Vec<RefRow>> {
    let md = std::fs::read_to_string(format!("{KIT}/expected.md")).expect("expected.md");
    let mut subjects: Vec<Vec<RefRow>> = Vec::new();
    for line in md.lines() {
        if line.starts_with("## Subject") {
            subjects.push(Vec::new());
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // `| decision | time | trough | outcome | interval | margin |`
        if cells.len() >= 7 && cells[1].parse::<usize>().is_ok() {
            let row = (
                cells[2].parse().expect("time"),
                cells[3].parse().expect("trough"),
                cells[4].to_string(),
                cells[5].parse().expect("interval"),
            );
            subjects
                .last_mut()
                .expect("a subject header first")
                .push(row);
        }
    }
    subjects
}

/// The controller the R loop mirrors: the rung comes from `ctx.interval`, the first matching
/// rule picks the dose effect and the interval effect, and a rung change is a `SetInterval`.
fn controller() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |ctx: &ControllerCtx| {
        let trough = ctx.signal("C").expect("monitor C declared");
        let interval = ctx.interval.expect("start_interval is set");
        let rung = LADDER
            .iter()
            .position(|&l| l == interval)
            .expect("interval on the ladder");
        let (dose, new_rung) = if trough > TH_HOLD {
            (false, (rung + 1).min(LADDER.len() - 1))
        } else if trough > TH_LEN {
            (true, (rung + 1).min(LADDER.len() - 1))
        } else if trough < TH_SHORT {
            (true, rung.saturating_sub(1))
        } else {
            (true, rung)
        };
        let mut actions = vec![if dose {
            DoseAction::Infuse {
                amt: DOSE,
                cmt: 1,
                rate: DOSE,
            }
        } else {
            DoseAction::Hold
        }];
        if new_rung != rung {
            actions.push(DoseAction::SetInterval {
                interval: LADDER[new_rung],
            });
        }
        actions
    }
}

/// Cross-solver trough band (mg/L), ferx (RK45 at the model's default tolerances) against
/// mrgsolve (LSODA at 1e-10) at η = 0. The worst gap over all 50 grid points, printed by
/// the test, is 2.109e-8 mg/L on Linux aarch64 (`tools/linux-test.sh`) and identical on
/// macOS arm64, so 1e-7 is 4.7× headroom; `expected.md`'s 9 decimals floor the comparison
/// at 5e-10. A dosing defect worth naming moves a trough by orders of magnitude more (one
/// misplaced 1000 mg dose is ~12 mg/L), and the smallest decision margin is 0.13 mg/L.
const TROUGH_TOL: f64 = 1e-7;

#[test]
fn vanco_interval_extension_matches_mrgsolve() {
    let parsed = parse_full_model_file(Path::new(&format!("{KIT}/vanco_interval.ferx")))
        .expect("model parses");
    let pop = read_nonmem_csv(
        Path::new(&format!("{KIT}/vanco_interval_subjects.csv")),
        None,
        None,
    )
    .expect("subjects load");
    let grid: Vec<f64> = (0..=24).map(|k| 12.0 * k as f64).collect();
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(1);
    opts.decision_times = grid.clone();
    opts.monitors = vec![MonitorSpec::new("C", 1, ObserveMode::Ipred)];
    opts.start_interval = Some(12.0);
    let model = &parsed.model;
    let etas = SuppliedEtas::from_table(
        vec!["1".into(), "2".into()],
        model.eta_names.clone(),
        vec![vec![0.0; model.n_eta]; 2],
    )
    .expect("η = 0 table");
    let inputs = AdaptiveInputs::new(&model.default_params).with_etas(&etas);
    let res = simulate_adaptive_with_inputs(model, &pop, &inputs, 1, controller, &opts)
        .expect("runs and passes the default verifier");

    let reference = reference();
    assert_eq!(reference.len(), 2, "two subjects in the kit");
    let mut worst: f64 = 0.0;
    for (s, (id, want)) in ["1", "2"].iter().zip(&reference).enumerate() {
        assert_eq!(
            want.len(),
            grid.len(),
            "subject {id}: one reference row per grid point"
        );
        let got: Vec<_> = res.decisions.iter().filter(|d| d.subject == *id).collect();
        assert_eq!(
            got.len(),
            grid.len(),
            "subject {id}: one decision row per grid point"
        );
        for (d, (t, trough, outcome, interval)) in got.iter().zip(want) {
            assert_eq!(d.time, *t, "subject {id}");
            let label = match d.outcome {
                DecisionOutcome::Dosed { .. } => "dosed",
                DecisionOutcome::Hold => "hold",
                DecisionOutcome::NotDue => "not_due",
                _ => "other",
            };
            assert_eq!(label, outcome, "subject {id} t={t}: outcome");
            assert_eq!(
                d.interval,
                Some(*interval),
                "subject {id} t={t}: interval after"
            );
            let sig = d.observed_signals[0].value;
            assert!(sig.is_finite(), "subject {id} t={t}: trough {sig}");
            let gap = (sig - trough).abs();
            worst = worst.max(gap);
            assert!(
                gap < TROUGH_TOL,
                "subject {id} t={t}: ferx trough {sig:.6} vs mrgsolve {trough:.6}"
            );
        }
        // The ledger holds exactly the dosed grid points, each 1000 mg.
        let dosed: Vec<f64> = want
            .iter()
            .filter(|r| r.2 == "dosed")
            .map(|r| r.0)
            .collect();
        let ledger: Vec<f64> = res
            .ledger
            .iter()
            .filter(|e| e.subject == *id)
            .map(|e| e.time)
            .collect();
        assert_eq!(ledger, dosed, "subject {id}: realized dose times");
        assert!(res
            .ledger
            .iter()
            .filter(|e| e.subject == *id)
            .all(|e| e.amt == DOSE));
        let m = &res.metrics[s];
        assert_eq!(m.subject, *id);
        let count = |o: &str| want.iter().filter(|r| r.2 == o).count();
        assert_eq!(m.n_not_due, count("not_due"), "subject {id}");
        assert_eq!(m.n_holds, count("hold"), "subject {id}");
    }
    eprintln!("worst ferx-vs-mrgsolve trough gap: {worst:.3e} mg/L");
}

/// The scenario the anchor relies on, read off the frozen reference so a regenerated kit
/// that lost a case fails here by name: subject 1 holds on the top rung, subject 2 shortens
/// after lengthening, and both saturate at the bottom at decision 0.
#[test]
fn vanco_interval_kit_is_non_degenerate() {
    let r = reference();
    let (s1, s2) = (&r[0], &r[1]);
    assert!(s1
        .iter()
        .any(|row| row.2 == "hold" && row.3 == 48.0 && row.0 > 0.0));
    let first_len = s2.iter().position(|row| row.3 > 12.0).expect("lengthens");
    let shortened = s2
        .windows(2)
        .skip(first_len)
        .any(|w| w[1].2 == "dosed" && w[1].3 < w[0].3);
    assert!(shortened, "subject 2 shortens after lengthening");
    for s in [s1, s2] {
        assert_eq!((s[0].1, s[0].3), (0.0, 12.0), "bottom-rung shorten at t=0");
        assert!(s.iter().any(|row| row.2 == "not_due"));
    }
}
