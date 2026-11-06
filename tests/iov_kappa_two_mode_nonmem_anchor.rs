//! NONMEM anchor for the guarded multi-start on an IOV subject (#2730): `inner_restarts`
//! must reach a mode that differs in one occasion's κ, which a descent from κ = 0 misses.
//!
//! ## The kit (`nonmem_anchor/iov_kappa_two_mode*`)
//!
//! - `simulate_iov_kappa_two_mode_data.py` (stdlib, seed 2730) writes both datasets: six
//!   subjects, one-compartment oral, two occasions, samples 8–24 h after each dose. Those
//!   are past `1/k`, where `C(t; ka)` rises and falls in `ka`, so one concentration has a
//!   solution either side of the peak. `iov_kappa_two_mode.csv` opens occasion 2 with
//!   `EVID=4` (every coordinate is scanned); `_noreset.csv` with `EVID=1` (the flatness and
//!   barrier probes decide). Occasion 2 lands on residual drug in the second one.
//! - `iov_kappa_two_mode.ferx`, the twin. All parameters are fixed on the NONMEM side.
//! - Three NONMEM 7.5.1 runs of the reset data, all `METHOD=1 INTER MAXEVAL=0`, in
//!   `results/`: `.ctl` alone (the descent from η = 0), `_mceta.ctl` (`MCETA=200`, random
//!   EBE starts, which finds subject 1's deep mode by itself), and `_start.ctl` (`$ETAS` +
//!   `MCETA=1`, the EBE search started at ferx's EBEs). Two for the no-reset data.
//!
//! ## Why the reset data needs the `$ETAS` run
//!
//! Subject 5 has a mode at κ = (−1.21, −0.82) that the multi-start finds and `MCETA=200`
//! and `MCETA=3000` do not. It is genuinely lower in the inner objective: 4.920 against
//! 5.178, computed a third way in R, outside both engines. NONMEM started there stays
//! there and scores it to 7e-8 of ferx. So this subject's *OFV contribution* goes up
//! (−1.807 to +0.118), because the FOCEI objective is not monotone in the posterior. The
//! inner loop selects on the inner objective, as `MCETA` does: on the no-reset data
//! subject 3 moves up the same way, in NONMEM unaided and in ferx.
//!
//! ## Measured (macOS arm64 and Linux x86_64, equal in every printed digit)
//!
//! | quantity | worst \|ferx − NONMEM\| | bound |
//! |---|---|---|
//! | total OFV, all four arms | 4.923e-7 | `5e-6` (10× above) |
//! | per-subject OFV, all four arms | 4.292e-7 | `5e-6` (11× above) |
//!
//! The arms straddle the fix: restarts off must equal NONMEM's descent from η = 0 and sit
//! whole OFV units above the arm with them on.

use ferx_core::api::read_population_for;
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, FitOptions, FitResult};

const MODEL: &str = include_str!("../nonmem_anchor/iov_kappa_two_mode.ferx");

/// Worst realised |ΔOFV| 4.923e-7. The other mode is 8.9 units away.
const OFV_TOL: f64 = 5e-6;
/// Worst realised per-subject |Δ| 4.292e-7.
const SUBJECT_TOL: f64 = 5e-6;

/// `OBJ` of `results/<name>.ext`, and each subject's from `results/<name>.phi`.
fn nonmem(name: &str) -> (f64, Vec<f64>) {
    let read = |ext: &str| {
        std::fs::read_to_string(format!("nonmem_anchor/results/{name}.{ext}")).expect("results")
    };
    let ext = read("ext");
    let total = ext
        .lines()
        .find(|l| l.trim_start().starts_with("-1000000000 "))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .expect("final OBJ row");
    let phi = read("phi");
    let subjects = phi
        .lines()
        .skip(2)
        .map(|l| l.split_whitespace().last().unwrap().parse().unwrap())
        .collect();
    (total, subjects)
}

fn evaluate(csv: &str, inner_restarts: usize) -> FitResult {
    let parsed = parse_full_model(MODEL).expect("anchor model must parse");
    let (pop, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        &format!("nonmem_anchor/{csv}"),
        None,
        parsed.fit_options.iov_column.as_deref(),
        None,
        &parsed.column_map,
    )
    .expect("anchor data must load");
    let opts = FitOptions {
        inner_restarts,
        outer_maxiter: 0,
        run_covariance_step: false,
        ..parsed.fit_options.clone()
    };
    let params = parsed.model.default_params.clone();
    fit(&parsed.model, &pop, &params, &opts).expect("anchor evaluation")
}

fn assert_matches(label: &str, r: &FitResult, reference: &str) {
    let (total, subjects) = nonmem(reference);
    assert!(r.ofv.is_finite(), "{label}: OFV {}", r.ofv);
    let d_total = (r.ofv - total).abs();
    assert_eq!(r.subjects.len(), subjects.len(), "{label}");
    let mut worst = 0.0_f64;
    for (s, want) in r.subjects.iter().zip(&subjects) {
        let gap = (s.ofv_contribution - want).abs();
        assert!(gap.is_finite(), "{label}: subject {}", s.id);
        worst = worst.max(gap);
    }
    eprintln!("MEASURE {label}: |ΔOFV| {d_total:.3e}, worst subject {worst:.3e}");
    assert!(d_total < OFV_TOL, "{label}: {} vs {total}", r.ofv);
    assert!(worst < SUBJECT_TOL, "{label}: worst subject {worst:e}");
}

/// A reset subject: every coordinate scanned, so the default reaches subject 1's deep mode.
#[test]
fn reset_data_reaches_nonmems_deep_kappa_mode() {
    let off = evaluate("iov_kappa_two_mode.csv", 0);
    assert_matches("reset, restarts off", &off, "iov_kappa_two_mode");
    let on = evaluate("iov_kappa_two_mode.csv", 1);
    assert_matches("reset, default", &on, "iov_kappa_two_mode_start");
    // Subject 1 against the run that found the mode with no help from ferx.
    let (_, unaided) = nonmem("iov_kappa_two_mode_mceta");
    let d = (on.subjects[0].ofv_contribution - unaided[0]).abs();
    eprintln!("MEASURE reset, subject 1 vs unaided MCETA: {d:.3e}");
    assert!(d < SUBJECT_TOL, "subject 1: {d:e}");
    assert!(
        off.subjects[0].ofv_contribution > on.subjects[0].ofv_contribution + 10.0,
        "subject 1 must straddle the fix"
    );
}

/// No resets: the flatness probe passes subject 1's sharp κ₂ over, the barrier probe does
/// not, and every subject then matches a NONMEM run that had no help from ferx.
#[test]
fn no_reset_data_reaches_nonmems_deep_kappa_mode() {
    let off = evaluate("iov_kappa_two_mode_noreset.csv", 0);
    assert_matches("no reset, restarts off", &off, "iov_kappa_two_mode_noreset");
    let on = evaluate("iov_kappa_two_mode_noreset.csv", 1);
    assert_matches("no reset, default", &on, "iov_kappa_two_mode_noreset_mceta");
    assert!(off.ofv > on.ofv + 5.0, "{} vs {}", off.ofv, on.ofv);
}
