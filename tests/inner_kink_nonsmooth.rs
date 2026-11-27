//! #3159 end to end: on an `[odes]` model whose observation is clipped at a threshold, with
//! `inner_kick = true`, the returned EBE of every subject is the minimum of its individual
//! objective.
//!
//! The fixture is a one-state decay restarted at each occasion, observed through
//! `max(0, eff − TH)` with IIV and IOV on `TH` (`tests/fixtures/clipped_decay`). Six of its
//! subjects: 14 and 17 on whom the objective-stall stop certified a point on a kink's valley
//! floor at ‖g‖ ≈ 7 and 10, 0.175 and 0.055 above the minimum; 20 and 27 whose descent ended in
//! a second local mode 0.011 away, 0.165 and 0.032 above it; and 1 and 2 as controls. The stall
//! escape is unconditional (free on a smooth objective); the kicks are opt-in (#3159, cost).
//!
//! The reference is an independent Nelder–Mead minimum of the same objective written out in
//! Python (`sum (DV − max(0, BASE·e^{−K·tad} − TH))²/σ² + ηᵀΩ⁻¹η + Σ κ²/ω_κ`, μ-shifted), from
//! three simplex sizes and two starts each. Worst of the six, measured: 1.8e-4 under the dense
//! driver (subject 27); 3.2e-3 under L-BFGS (subject 17), whose limited-memory metric resolves
//! the shallow, prior-constrained direction along `ETA_TH`/`KAPPA_1`/`KAPPA_2` less exactly — ten
//! times the escapes or iterations reaches the same point, so it is not short of either. The
//! bound is 5e-3, ≥ 1.6× the worse of the two.
//!
//! Mutations (run): the stall escape off in the dense driver → subject 17 fails (14 is then
//! reached by the kicks), and in the L-BFGS driver → the same under `lbfgs`; the kick restarts off
//! (the default) → subject 20 fails (27 too). On `main` all four fail.

use std::path::Path;

/// `[ETA_BASE, ETA_TH, KAPPA_1, KAPPA_2]` of each subject's minimum.
const ORACLE: [(&str, [f64; 4]); 6] = [
    ("1", [-0.081773, 0.092269, 0.480459, -0.43945]),
    ("2", [-0.217644, -0.201366, 0.072804, -0.162301]),
    ("14", [0.302834, 0.29221, -0.37159, 0.501461]),
    ("17", [-0.237811, 0.535079, 0.118906, 0.118906]),
    ("20", [-0.283402, 0.141441, 0.435102, -0.372239]),
    ("27", [-0.096806, -0.356722, -0.051522, -0.107021]),
];

/// The fixture's fit under `driver` (`bfgs` forces the dense driver, `lbfgs` the limited-memory
/// one: the fixture has 4 coordinates, below the default switch at 32), plus `extra` options.
fn fit_with(driver: &str, extra: &str) -> ferx_core::FitResult {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clipped_decay");
    let model = std::fs::read_to_string(dir.join("clipped.ferx")).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("clipped.ferx");
    // One thread: the EBE solves are independent of each other, but a parallel run's rayon
    // worker count is free to change, and the oracle comparison wants one fixed path.
    std::fs::write(&path, format!("{model}  inner_optimizer = {driver}\n  threads = 1\n{extra}"))
        .unwrap();
    ferx_core::run_model_with_data(
        path.to_str().unwrap(),
        Some(dir.join("clipped.csv").to_str().unwrap()),
    )
    .expect("fit")
    .0
}

/// How far each subject's EBE is from its [`ORACLE`] minimum, by id.
fn distances(fit: &ferx_core::FitResult) -> Vec<(&'static str, f64)> {
    ORACLE
        .iter()
        .map(|&(id, want)| {
            let i = fit.subjects.iter().position(|s| s.id == id).unwrap_or_else(|| panic!("{id}"));
            let (eta, kappas) = (&fit.subjects[i].eta, &fit.ebe_kappas[i]);
            let got = [eta[0], eta[1], kappas[0][0], kappas[1][0]];
            let worst = got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max); // nan-ok: asserted below
            assert!(worst.is_finite(), "subject {id}: EBE {got:?}");
            (id, worst)
        })
        .collect()
}

fn check(driver: &str) {
    for (id, worst) in distances(&fit_with(driver, "  inner_kick = true\n")) {
        assert!(worst < 5e-3, "{driver}, subject {id}: EBE is {worst:.1e} from its minimum");
    }
}

#[test]
fn a_clipped_models_ebes_are_the_minima_of_their_individual_objectives() {
    check("bfgs");
}

/// The same fixture on the L-BFGS driver, which has its own copy of the stall exit.
#[test]
fn a_clipped_models_ebes_are_the_minima_under_the_limited_memory_driver() {
    check("lbfgs");
}

/// `inner_kick` defaults to `false`: the second mode of subject 20, 0.011 away and 0.165 lower,
/// is not reached without it (a differential pair with `inner_kick = true`, which reaches it).
#[test]
fn inner_kick_is_off_by_default_and_leaves_a_second_mode_unreached() {
    let by_id = |fit: &ferx_core::FitResult| {
        distances(fit).into_iter().find(|&(id, _)| id == "20").map(|(_, d)| d).unwrap()
    };
    assert!(by_id(&fit_with("bfgs", "")) > 5e-3, "premise: the default leaves it unreached");
    assert!(by_id(&fit_with("bfgs", "  inner_kick = true\n")) < 5e-3);
}
