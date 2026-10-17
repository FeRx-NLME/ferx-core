//! #1605: a declared `omega ~ 0.0` / `kappa ~ 0.0` (FIX or not) draws exactly zero in
//! every simulation consumer, and every other η draws from its own declared variance.
//!
//! Before the fix the constructor's `+ 1e-8·I` regularisation reached the draw: the zero
//! η came out with sd `1e-4` and the non-zero ones from `ω + 1e-8`. Measured on Linux at
//! `3b88da7f` on this file's ODE model: `simulate_adaptive` IPRED 1.4e-4 relative off the
//! closed form (seed 1), `simulate_with_seed` 2.2e-4, while `predict` at η = 0 was 4e-16.
//!
//! One test per draw site, each failure naming its consumer: `simulate` (η, the
//! propensity-matched pool's η, κ), `simulate_adaptive` (η, κ, and the verifier's κ
//! re-draw) and the NPDE reference (η, κ). Each was killed by pointing that one site
//! back at `.chol`; the PR body lists which test died for which site.

use super::*;
use crate::parser::model_parser::parse_model_string;
use crate::sim::adaptive::{ControllerCtx, DoseAction};
use crate::types::OmegaMatrix;
use nalgebra::DMatrix;
use rand::{RngExt, SeedableRng};
use rand_distr::Normal;
use std::collections::HashMap;
use std::io::Write;

/// One-compartment oral ODE. `{omegas}` declares the random effects, `{cl}` / `{v}` are
/// the CL and V expressions, so the same structure serves the zero model and its twins.
fn model_src(omegas: &str, cl: &str, v: &str) -> String {
    format!(
        r#"
[parameters]
  theta TVCL(2.0, 0.01, 50.0)
  theta TVV(20.0, 0.5, 200.0)
  theta TVKA(0.15, 0.005, 20.0)
{omegas}
  sigma PROP ~ 1e-10 (sd)
[individual_parameters]
  CL = {cl}
  V  = {v}
  KA = TVKA
[structural_model]
  ode(obs_cmt=central, states=[depot, central])
[odes]
  d/dt(depot)   = -KA*depot
  d/dt(central) = KA*depot - CL/V*central
[scaling]
  y = central / V
[error_model]
  DV ~ proportional(PROP)
[fit_options]
  ode_reltol = 1e-11
  ode_abstol = 1e-11
"#
    )
}

fn model(omegas: &str, cl: &str, v: &str) -> CompiledModel {
    parse_model_string(&model_src(omegas, cl, v)).expect("#1605 fixture parses")
}

/// `ETA_CL ~ 0.0 FIX` on CL — the model a user writes to switch a random effect off.
fn zero_fix() -> CompiledModel {
    model("  omega ETA_CL ~ 0.0 FIX", "TVCL * exp(ETA_CL)", "TVV")
}

/// `ETA_CL ~ 0.0 FIX` and `KAPPA_CL ~ 0.0 FIX`, both on CL.
fn zero_fix_kappa() -> CompiledModel {
    model(
        "  omega ETA_CL ~ 0.0 FIX\n  kappa KAPPA_CL ~ 0.0 FIX",
        "TVCL * exp(ETA_CL + KAPPA_CL)",
        "TVV",
    )
}

const OBS: [f64; 6] = [2.0, 6.0, 26.0, 30.0, 50.0, 54.0];
const DOSES: [f64; 3] = [0.0, 24.0, 48.0];

/// `nid` subjects, 100 mg into the depot at each of `DOSES` (when `with_doses`), the obs
/// of `OBS` on occasion 1 before t = 24 and occasion 2 after. `dv(t)` is the DV column.
fn population(name: &str, nid: usize, with_doses: bool, dv: impl Fn(f64) -> f64) -> Population {
    let mut s = String::from("ID,TIME,DV,AMT,EVID,CMT,MDV,OCC\n");
    for id in 1..=nid {
        let mut rows: Vec<(f64, u8, String)> = Vec::new();
        if with_doses {
            for t in DOSES {
                let occ = if t < 24.0 { 1 } else { 2 };
                rows.push((t, 0, format!("{id},{t},0,100,1,1,1,{occ}\n")));
            }
        }
        for t in OBS {
            let occ = if t < 24.0 { 1 } else { 2 };
            rows.push((t, 1, format!("{id},{t},{},0,0,2,0,{occ}\n", dv(t))));
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        rows.into_iter().for_each(|r| s.push_str(&r.2));
    }
    let path = std::env::temp_dir().join(format!("ferx_1605_{name}_{}.csv", std::process::id()));
    std::fs::File::create(&path)
        .and_then(|mut f| f.write_all(s.as_bytes()))
        .expect("write #1605 fixture csv");
    let pop = crate::read_nonmem_csv(&path, None, Some("OCC")).expect("read #1605 fixture csv");
    let _ = std::fs::remove_file(&path);
    pop
}

/// `predict` at η = 0, keyed by (id, time bits).
fn predict_at_zero(m: &CompiledModel, pop: &Population) -> HashMap<(String, u64), f64> {
    predict(m, pop, &m.default_params)
        .expect("predict runs")
        .into_iter()
        .map(|p| ((p.id, p.time.to_bits()), p.pred))
        .collect()
}

/// Every simulated IPRED equals `reference` to the bit, and is finite. `consumer` names
/// the draw site in the failure.
fn assert_rows_bit_equal(
    consumer: &str,
    rows: &[SimulationResult],
    reference: &HashMap<(String, u64), f64>,
) {
    assert!(!rows.is_empty(), "{consumer}: no rows to compare");
    for r in rows {
        let want = reference[&(r.id.clone(), r.time.to_bits())];
        assert!(
            r.ipred.is_finite(),
            "{consumer}: IPRED {} at t={}",
            r.ipred,
            r.time
        );
        assert_eq!(
            r.ipred.to_bits(),
            want.to_bits(),
            "{consumer}: IPRED {:e} != predict at η = 0 {:e} (id {}, t={}, rel {:.3e}) — a \
             zero ω/κ was drawn as non-zero (#1605)",
            r.ipred,
            want,
            r.id,
            r.time,
            ((r.ipred - want) / want).abs()
        );
    }
}

// ── OmegaMatrix: the factor itself ───────────────────────────────────────────────────

#[test]
fn positive_definite_omega_draws_from_chol_unchanged() {
    let om = OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["A".into(), "B".into()]);
    assert!(om.sim_chol.is_none());
    let f = om.sim_draw_factor(Some(&om), &[true, true]);
    assert!(
        std::ptr::eq(&*f, &om.chol),
        "a PD Ω must borrow `chol`, not rebuild it"
    );
}

#[test]
fn structural_zero_leaves_the_fit_matrices_regularised_and_the_draw_exact() {
    let om = OmegaMatrix::from_diagonal(&[0.0, 0.09], vec!["ZERO".into(), "V".into()]);
    // The fit path is untouched: `matrix`, `chol`, `inv`, `log_det` still describe the
    // regularised Ω, exactly as at 3b88da7f.
    assert_eq!(om.matrix[(0, 0)], 1e-8);
    assert_eq!(om.matrix[(1, 1)], 0.09 + 1e-8);
    assert_eq!(om.chol[(0, 0)], 1e-8f64.sqrt());
    // The draw is not: the zero row is exactly 0, the other η exactly its declared sd.
    let f = om.sim_draw_factor(None, &[]);
    assert_eq!(f.as_slice(), &[0.0, 0.0, 0.0, 0.09f64.sqrt()]);
    assert_eq!(om.structural_zeros(), vec![true, false]);
}

#[test]
fn structural_zero_beside_a_block_factors_the_block_on_its_own() {
    let mut m = DMatrix::zeros(3, 3);
    m[(0, 0)] = 0.09;
    m[(2, 2)] = 0.04;
    m[(0, 2)] = 0.01;
    m[(2, 0)] = 0.01;
    let om = OmegaMatrix::from_matrix(m, vec!["A".into(), "Z".into(), "B".into()], false);
    let f = om.sim_draw_factor(None, &[]);
    let sub = DMatrix::from_row_slice(2, 2, &[0.09, 0.01, 0.01, 0.04])
        .cholesky()
        .unwrap()
        .l();
    assert_eq!(f[(0, 0)], sub[(0, 0)]);
    assert_eq!(f[(2, 0)], sub[(1, 0)]);
    assert_eq!(f[(2, 2)], sub[(1, 1)]);
    assert!(f.row(1).iter().all(|&x| x == 0.0) && f.column(1).iter().all(|&x| x == 0.0));
}

#[test]
fn indefinite_omega_without_a_zero_row_keeps_drawing_from_chol() {
    // A zero diagonal whose row is not zero is not a structural zero: no split exists,
    // so the draw falls back to the regularised `chol` as before.
    let m = DMatrix::from_row_slice(2, 2, &[0.0, 0.01, 0.01, 0.09]);
    let om = OmegaMatrix::from_matrix(m, vec!["A".into(), "B".into()], false);
    assert!(om.sim_chol.is_none());
    assert!(std::ptr::eq(&*om.sim_draw_factor(None, &[]), &om.chol));
}

#[test]
fn all_zero_omega_draws_zero() {
    let om = OmegaMatrix::from_diagonal(&[0.0, 0.0], vec!["A".into(), "B".into()]);
    assert!(om.sim_draw_factor(None, &[]).iter().all(|&x| x == 0.0));
}

/// A fit's Ω carries a `FIX`-ed zero as the regularised `1e-8`, through the packed
/// `ln L_ii` (the parameterisation round trip `fit` runs) or rebuilt from a bare matrix
/// (ferx-r, `.fitrx`). Both recover the zero from the declaration — and only when the
/// declaration fixes it at 0 and the carried value is still the floor.
#[test]
fn a_fit_carried_fix_zero_is_recovered_from_the_declaration() {
    use crate::estimation::parameterization::{pack_params, unpack_params};
    let m = model(
        "  omega ETA_CL ~ 0.0 FIX\n  omega ETA_V ~ 0.09",
        "TVCL * exp(ETA_CL)",
        "TVV * exp(ETA_V)",
    );
    let declared = &m.default_params;
    let want = [0.0, 0.0, 0.0, 0.09f64.sqrt()];

    // (a) through the fit's own pack/unpack: `from_chol_factor`, no `sim_chol`.
    let fitted = unpack_params(&pack_params(declared), declared);
    assert!(fitted.omega.sim_chol.is_none());
    assert!(
        fitted.omega.matrix[(0, 0)] > 0.0,
        "the fit carries the floor, not 0"
    );
    let f = fitted
        .omega
        .sim_draw_factor(Some(&declared.omega), &declared.omega_fixed);
    assert_eq!(f[(0, 0)], 0.0, "unpacked FIX-0 η not recovered as zero");
    assert_eq!(f[(1, 0)], 0.0);
    assert_eq!(f[(1, 1)], fitted.omega.matrix[(1, 1)].sqrt());

    // (b) a bare matrix as `FitResult.omega` stores it, rebuilt by `from_matrix`.
    let fit_matrix = DMatrix::from_row_slice(2, 2, &[1e-8, 0.0, 0.0, 0.09]);
    let rebuilt = OmegaMatrix::from_matrix(fit_matrix, declared.omega.eta_names.clone(), true);
    let f = rebuilt.sim_draw_factor(Some(&declared.omega), &declared.omega_fixed);
    assert_eq!(f.as_slice(), &want);

    // Not FIX in the declaration → the carried 1e-8 is the caller's value, drawn as is.
    let f = rebuilt.sim_draw_factor(Some(&declared.omega), &[false, false]);
    assert!(std::ptr::eq(&*f, &rebuilt.chol));
    // A real variance supplied for the FIX-0 η is honoured, not overwritten by the
    // declaration.
    let supplied = OmegaMatrix::from_diagonal(&[0.04, 0.09], declared.omega.eta_names.clone());
    let f = supplied.sim_draw_factor(Some(&declared.omega), &declared.omega_fixed);
    assert!(std::ptr::eq(&*f, &supplied.chol));
}

// ── simulate ─────────────────────────────────────────────────────────────────────────

#[test]
fn simulate_eta_zero_fix_is_bit_equal_to_predict_at_eta_zero() {
    let m = zero_fix();
    let pop = population("sim_eta", 4, true, |_| 0.0);
    let rows = simulate_with_seed(&m, &pop, &m.default_params, 3, 7).expect("simulate runs");
    assert_rows_bit_equal("simulate η", &rows, &predict_at_zero(&m, &pop));
}

#[test]
fn simulate_free_zero_omega_is_a_structural_zero_in_a_simulation_only_run() {
    // No FIX: a simulation-only run reads a bare `~ 0.0` as η ≡ 0 too (#1605 decision;
    // NONMEM refuses a bare `$OMEGA 0` outright, and ferx's fits keep refusing it).
    let m = model("  omega ETA_CL ~ 0.0", "TVCL * exp(ETA_CL)", "TVV");
    let pop = population("sim_free", 3, true, |_| 0.0);
    let rows = simulate_with_seed(&m, &pop, &m.default_params, 2, 5).expect("simulate runs");
    assert_rows_bit_equal("simulate η (free zero)", &rows, &predict_at_zero(&m, &pop));
}

#[test]
fn simulate_propensity_pool_eta_zero_fix_is_bit_equal_to_predict_at_eta_zero() {
    let m = zero_fix();
    let pop = population("sim_pool", 4, true, |t| (t / 10.0).sin().abs() + 0.5);
    let mut opts = SimulateOptions::default();
    opts.seed = Some(9);
    opts.match_method = Some(crate::MatchMethod::Optimal);
    let rows = simulate_with_options(&m, &pop, &m.default_params, 2, &opts).expect("simulate runs");
    assert_rows_bit_equal(
        "simulate η (propensity-matched pool)",
        &rows,
        &predict_at_zero(&m, &pop),
    );
}

#[test]
fn simulate_kappa_zero_fix_is_bit_equal_to_predict_iov_at_zero() {
    let m = zero_fix_kappa();
    let pop = population("sim_kappa", 3, true, |_| 0.0);
    let rows = simulate_with_seed(&m, &pop, &m.default_params, 2, 13).expect("simulate runs");
    let mut reference = HashMap::new();
    for s in &pop.subjects {
        let n_groups = crate::stats::stacked_layout::group_count(s);
        assert_eq!(
            n_groups, 2,
            "the fixture must carry two occasions, or κ is never drawn"
        );
        let kappas = vec![vec![0.0; m.n_kappa]; n_groups];
        let p = crate::pk::predict_iov(&m, s, &m.default_params.theta, &[0.0], &kappas);
        for (t, v) in s.obs_times.iter().zip(p) {
            reference.insert((s.id.clone(), t.to_bits()), v);
        }
    }
    assert_rows_bit_equal("simulate κ", &rows, &reference);
}

/// The non-degenerate pair: `ETA_CL ~ 0.0 FIX` beside `ETA_V ~ 0.09`. The zero η is
/// exactly 0, the other draws from exactly its declared sd (not `√(0.09 + 1e-8)`, which
/// the regularisation leaked into it), and the `z` stream is the one drawn before the
/// fix: `ETA_V` reads the **second** normal, so the full vector is still drawn.
#[test]
fn simulate_mixed_omega_zero_eta_is_exact_and_the_other_keeps_its_variance() {
    let m = model(
        "  omega ETA_CL ~ 0.0 FIX\n  omega ETA_V ~ 0.09",
        "TVCL * exp(ETA_CL)",
        "TVV * exp(ETA_V)",
    );
    let pop = population("sim_mixed", 1, true, |_| 0.0);
    let seed = 21;
    let rows = simulate_with_seed(&m, &pop, &m.default_params, 1, seed).expect("simulate runs");

    // The first subject's η is the first `n_eta` normals of the seeded stream.
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z: Vec<f64> = (0..2).map(|_| rng.sample(normal)).collect();
    let eta_v = 0.09f64.sqrt() * z[1];
    let s = &pop.subjects[0];
    let want =
        crate::pk::compute_predictions_with_tv(&m, s, &m.default_params.theta, &[0.0, eta_v]);
    // The regularised draw this replaces, so the pair straddles the fix: it must differ.
    let old = crate::pk::compute_predictions_with_tv(
        &m,
        s,
        &m.default_params.theta,
        &[1e-4 * z[0], (0.09f64 + 1e-8).sqrt() * z[1]],
    );
    assert!(
        want.iter().zip(&old).any(|(a, b)| a != b),
        "the twin does not straddle"
    );
    assert_eq!(rows.len(), want.len());
    for (r, w) in rows.iter().zip(&want) {
        assert!(r.ipred.is_finite());
        assert_eq!(
            r.ipred.to_bits(),
            w.to_bits(),
            "simulate η (mixed Ω): IPRED {:e} != η = (0, √0.09·z₂) prediction {:e} at t={}",
            r.ipred,
            w,
            r.time
        );
    }
}

// ── simulate_adaptive ────────────────────────────────────────────────────────────────

fn adaptive_opts(verify: bool) -> AdaptiveSimulateOptions {
    let mut o = AdaptiveSimulateOptions::default();
    o.seed = Some(1);
    o.decision_times = DOSES.to_vec();
    o.verify = verify;
    o
}

fn bolus() -> impl FnMut(&ControllerCtx) -> Vec<DoseAction> {
    |_ctx: &ControllerCtx| vec![DoseAction::Bolus { amt: 100.0, cmt: 1 }]
}

/// The zero model against its twin: the same structure with the random effect declared
/// non-zero but read by nothing, so its trajectory is the η = 0 one by construction.
/// A static `predict` is not bit-comparable here (the adaptive driver integrates in
/// decision segments), so the twin is the η = 0 reference on the same engine.
fn assert_adaptive_matches_twin(
    consumer: &str,
    zero: &CompiledModel,
    twin: &CompiledModel,
    verify: bool,
) {
    let pop = population(&format!("adaptive_{}", zero.n_kappa), 2, false, |_| 0.0);
    let run = |m: &CompiledModel| {
        simulate_adaptive(m, &pop, &m.default_params, 2, bolus, &adaptive_opts(verify))
            .unwrap_or_else(|e| panic!("{consumer}: simulate_adaptive refused: {e}"))
    };
    let (a, b) = (run(zero), run(twin));
    assert_eq!(a.trajectories.len(), b.trajectories.len());
    assert!(!a.trajectories.is_empty());
    for (x, y) in a.trajectories.iter().zip(&b.trajectories) {
        assert!(x.ipred.is_finite());
        assert_eq!(
            x.ipred.to_bits(),
            y.ipred.to_bits(),
            "{consumer}: IPRED {:e} != the η = 0 twin's {:e} at t={} (rel {:.3e}) — a zero \
             ω/κ was drawn as non-zero (#1605)",
            x.ipred,
            y.ipred,
            x.time,
            ((x.ipred - y.ipred) / y.ipred).abs()
        );
    }
}

#[test]
fn adaptive_eta_zero_fix_is_bit_equal_to_the_eta_zero_twin() {
    let twin = model("  omega ETA_CL ~ 0.09", "TVCL", "TVV");
    assert_adaptive_matches_twin("simulate_adaptive η", &zero_fix(), &twin, false);
}

#[test]
fn adaptive_kappa_zero_fix_is_bit_equal_to_the_kappa_zero_twin() {
    let twin = model(
        "  omega ETA_CL ~ 0.0 FIX\n  kappa KAPPA_CL ~ 0.09",
        "TVCL * exp(ETA_CL)",
        "TVV",
    );
    assert_adaptive_matches_twin("simulate_adaptive κ", &zero_fix_kappa(), &twin, false);
}

#[test]
fn adaptive_verifier_redraws_a_zero_kappa_from_the_build_loops_factor() {
    // Verifier on: its independent κ re-draw must resolve the same factor as the build
    // loop, or the bit-equality it asserts fails and the run is refused.
    let twin = model(
        "  omega ETA_CL ~ 0.0 FIX\n  kappa KAPPA_CL ~ 0.09",
        "TVCL * exp(ETA_CL)",
        "TVV",
    );
    assert_adaptive_matches_twin(
        "simulate_adaptive verifier κ re-draw",
        &zero_fix_kappa(),
        &twin,
        true,
    );
}

/// The adaptive IPRED against the closed form, the measurement #1605 was filed on: 1.4e-4
/// relative at 3b88da7f (seed 1), now at the ODE engines' own tolerance.
#[test]
fn adaptive_eta_zero_fix_matches_the_closed_form() {
    let m = zero_fix();
    let pop = population("adaptive_closed", 1, false, |_| 0.0);
    let res = simulate_adaptive(&m, &pop, &m.default_params, 1, bolus, &adaptive_opts(true))
        .expect("simulate_adaptive runs");
    let (cl, v, ka) = (2.0f64, 20.0f64, 0.15f64);
    let k = cl / v;
    let closed = |t: f64| -> f64 {
        DOSES
            .iter()
            .filter(|&&td| t >= td)
            .map(|&td| {
                let s = t - td;
                100.0 * ka / (v * (ka - k)) * ((-k * s).exp() - (-ka * s).exp())
            })
            .sum()
    };
    let mut worst = 0.0f64;
    for r in &res.trajectories {
        let rel = ((r.ipred - closed(r.time)) / closed(r.time)).abs();
        assert!(rel.is_finite(), "non-finite IPRED at t={}", r.time);
        worst = worst.max(rel);
    }
    // Bound: the ODE engines' 1e-10 (as #1611's anchors). Realised: see the PR body.
    assert!(
        worst < 1e-10,
        "simulate_adaptive η: worst rel vs closed form {worst:.3e}"
    );
}

// ── NPDE ─────────────────────────────────────────────────────────────────────────────

/// DV sits `2e-5` relative above the η = 0 prediction. The residual is at the simulator's
/// variance floor (`1e-12`, sd `1e-6` absolute, since the declared σ is below it), so with
/// η and κ exactly 0 the smallest DV offset (`0.87 · 2e-5`) is 17 sd clear and every
/// replicate falls below the DV: every npd sits at the top clamp, `Φ⁻¹(1 − 1/800)` =
/// 3.02 at K = 400. A draw of sd `1e-4` (the regularised zero) moves the late IPREDs by
/// up to ~2e-4 relative (the gap measured on `simulate`), scattering their replicates
/// across the DV and their npd back towards 0.
fn assert_npde_at_top_clamp(consumer: &str, m: &CompiledModel) {
    let reference = population("npde_ref", 2, true, |_| 0.0);
    let pred = predict_at_zero(m, &reference);
    let pop = population(&format!("npde_{}", m.n_kappa), 2, true, |t| {
        pred[&("1".to_string(), t.to_bits())] * (1.0 + 2e-5)
    });
    let scores =
        crate::stats::npde::compute_npde_npd(m, &pop, &m.default_params, 400, Some(3), None)
            .expect("compute_npde_npd runs");
    for (i, s) in scores.iter().enumerate() {
        for &npd in &s.npd {
            assert!(npd.is_finite(), "{consumer}: subject {i} npd {npd}");
            assert!(
                npd > 2.5,
                "{consumer}: subject {i} npd {npd:.3} — replicates scatter around a DV 2e-5 \
                 above the η = 0 prediction, so a zero ω/κ was drawn as non-zero (#1605)"
            );
        }
    }
}

#[test]
fn npde_reference_draws_a_zero_fix_eta_as_zero() {
    assert_npde_at_top_clamp("NPDE η", &zero_fix());
}

#[test]
fn npde_reference_draws_a_zero_fix_kappa_as_zero() {
    assert_npde_at_top_clamp("NPDE κ", &zero_fix_kappa());
}
