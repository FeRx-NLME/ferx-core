//! Tests for the VI driver.

use super::*;
use crate::types::{
    BloqMethod, DoseEvent, EstimationMethod, GradientMethod, Subject, ViEtaGrad, ViFamily,
    ViFinalOfv, ViKl, ViOmegaUpdate,
};
use std::collections::HashMap;

/// Closed-form 1-cpt IV, two random effects, three subjects. Data simulated from
/// the model's own initial estimates so the fit has a real optimum to move toward
/// rather than an arbitrary one.
fn fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("fixture parses");
    let params = model.default_params.clone();

    let make = |id: &str, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: vec![1.0, 4.0, 8.0],
        obs_raw_times: Vec::new(),
        observations: vec![9.0 * scale, 6.7 * scale, 4.5 * scale],
        obs_cmts: vec![1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; 3],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let population = Population {
        subjects: vec![make("1", 1.0), make("2", 1.15), make("3", 0.87)],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    (model, population, params)
}

/// As [`fixture`], but with a **mixed** `block_omega` + standalone-eta `Ω` — the
/// shape `examples/warfarin_block_omega.ferx` uses. The `ETA_KA` row is outside the
/// `(ETA_CL, ETA_V)` block, so `Ω(2,0)` and `Ω(2,1)` are structural zeros.
fn mixed_omega_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  theta TVKA(1.5, 0.01, 50.0)
  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("mixed omega fixture parses");
    let params = model.default_params.clone();
    let (_, population, _) = fixture();
    (model, population, params)
}

fn opts(iters: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Vi,
        vi_iters: iters,
        vi_mc_samples: 2,
        vi_seed: Some(99),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Convergence bookkeeping
// ---------------------------------------------------------------------------

/// The moving-average criterion must ignore per-iteration noise (which is what a
/// naive `|f[n] − f[n−1]| < tol` test would fire on) and respond to genuine drift.
#[test]
fn trace_settles_on_a_flat_noisy_tail_but_not_on_a_drifting_one() {
    // Flat + alternating noise: a single-step test would see a change of 2.0
    // every iteration and never converge; the moving average sees ~0.
    let flat: Vec<f64> = (0..100)
        .map(|i| 500.0 + if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    assert!(trace_has_settled(&flat, 20, 1e-4));

    // Steady descent of 1 per iteration: must NOT be reported as settled.
    let drifting: Vec<f64> = (0..100).map(|i| 500.0 - i as f64).collect();
    assert!(!trace_has_settled(&drifting, 20, 1e-4));

    // Too short to judge.
    assert!(!trace_has_settled(&[1.0, 2.0, 3.0], 20, 1e-4));
    assert!(!trace_has_settled(&flat, 0, 1e-4));

    // A non-finite tail is never "settled".
    let mut poisoned = flat.clone();
    poisoned[99] = f64::NAN;
    assert!(!trace_has_settled(&poisoned, 20, 1e-4));
}

// ---------------------------------------------------------------------------
// End-to-end behaviour
// ---------------------------------------------------------------------------

/// A short run must complete, report a full trace, and drive the objective down.
#[test]
fn run_vi_decreases_the_objective_and_reports_a_trace() {
    let (model, population, params) = fixture();
    let out = run_vi(&model, &population, &params, &opts(60)).expect("VI runs");
    let vi = out.vi.as_ref().expect("VI result present");

    assert_eq!(vi.elbo_trace.len(), 60);
    assert_eq!(vi.n_iterations, 60);
    assert!(vi.neg_two_elbo.is_finite());
    assert_eq!(vi.family, "full_rank");
    assert_eq!(vi.eta_means.len(), 3);
    assert_eq!(vi.eta_covs.len(), 3);
    assert_eq!(vi.eta_covs[0].len(), 2);

    // The objective must be lower at the end than at the start. Compare block
    // means rather than endpoints: the trace is a Monte-Carlo estimate.
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let first = mean(&vi.elbo_trace[..10]);
    let last = mean(&vi.elbo_trace[50..]);
    assert!(
        last < first,
        "objective should decrease: first block {first:.4}, last block {last:.4}"
    );
}

/// `OuterResult` must carry conditional **modes** and the `n_obs × n_eta`
/// sensitivity matrix, not the variational moments.
///
/// `h_matrices` is NONMEM's "H" — `∂f/∂η`, consumed by `compute_cwres` as
/// `ipred − H·η̂` — and every downstream diagnostic (CWRES, IWRES, shrinkage,
/// sdtab) is defined against the mode and that matrix. Reporting `μ` as `eta_hat`
/// while sourcing `H` from elsewhere would make them mutually inconsistent, so VI
/// runs one inner-loop pass at its converged estimate. The variational moments
/// are still reported, on `ViResult`.
#[test]
fn outer_result_carries_conditional_modes_and_sensitivities() {
    let (model, population, params) = fixture();
    let out = run_vi(&model, &population, &params, &opts(30)).expect("VI runs");
    let vi = out.vi.as_ref().unwrap();

    assert_eq!(out.eta_hats.len(), 3);
    assert_eq!(out.h_matrices.len(), 3);
    for (i, eta) in out.eta_hats.iter().enumerate() {
        assert_eq!(eta.len(), 2, "eta_hat is per-random-effect");
        // H is n_obs × n_eta: three observations, two random effects.
        assert_eq!(
            out.h_matrices[i].nrows(),
            3,
            "H has one row per observation"
        );
        assert_eq!(out.h_matrices[i].ncols(), 2, "H has one column per eta");
        assert!(out.h_matrices[i].iter().all(|v| v.is_finite()));

        // The mode should land near the variational mean — warm-started from it —
        // without being required to equal it.
        for k in 0..2 {
            assert!(
                (eta[k] - vi.eta_means[i][k]).abs() < 0.5,
                "subject {i} eta[{k}]: mode {} vs variational mean {}",
                eta[k],
                vi.eta_means[i][k]
            );
        }
    }

    // And the variational covariances are still reported, positive-definite.
    for cov in &vi.eta_covs {
        assert!(cov[0][0] > 0.0 && cov[1][1] > 0.0);
        assert!(
            (cov[0][1] - cov[1][0]).abs() < 1e-12,
            "covariance must be symmetric"
        );
    }
}

/// Same seed ⇒ same fit. Different seed ⇒ a different (but still valid) fit.
#[test]
fn runs_are_reproducible_from_the_seed() {
    let (model, population, params) = fixture();
    let a = run_vi(&model, &population, &params, &opts(25)).unwrap();
    let b = run_vi(&model, &population, &params, &opts(25)).unwrap();
    for (x, y) in a.params.theta.iter().zip(b.params.theta.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "same seed must give the same fit");
    }

    let mut other = opts(25);
    other.vi_seed = Some(1234);
    let c = run_vi(&model, &population, &params, &other).unwrap();
    assert!(
        a.params
            .theta
            .iter()
            .zip(c.params.theta.iter())
            .any(|(x, y)| x != y),
        "a different seed should draw different common random numbers"
    );
}

// ---------------------------------------------------------------------------
// The ELBO-is-not-an-OFV contract
// ---------------------------------------------------------------------------

/// By default `ofv` is `NaN` with a warning saying why. This is the load-bearing
/// contract of the whole method: the ELBO is a lower bound, and silently writing
/// it where a `−2 log L` is expected would invite invalid model comparisons.
#[test]
fn default_leaves_ofv_nan_and_explains_why() {
    let (model, population, params) = fixture();
    let out = run_vi(&model, &population, &params, &opts(20)).unwrap();

    assert!(
        out.ofv.is_nan(),
        "default vi_final_ofv must not fabricate an OFV"
    );
    assert!(
        out.warnings.iter().any(|w| w.contains("lower bound")),
        "a NaN OFV must be explained; warnings were {:?}",
        out.warnings
    );
    // And the bound itself is still reported, just not as the OFV.
    assert!(out.vi.as_ref().unwrap().neg_two_elbo.is_finite());
}

/// `vi_final_ofv = laplace` reconverges EBEs at the VI estimate and reports a
/// genuine FOCE-comparable objective — a *different* number from the ELBO.
#[test]
fn laplace_final_ofv_reports_a_real_objective() {
    let (model, population, params) = fixture();
    let mut o = opts(40);
    o.vi_final_ofv = ViFinalOfv::Laplace;
    let out = run_vi(&model, &population, &params, &o).unwrap();

    assert!(
        out.ofv.is_finite(),
        "laplace mode must produce a finite OFV"
    );
    let elbo = out.vi.as_ref().unwrap().neg_two_elbo;
    assert!(
        (out.ofv - elbo).abs() > 1e-9,
        "the Laplace objective and the ELBO bound should not coincide \
         (ofv {}, -2*ELBO {elbo})",
        out.ofv
    );
    assert!(
        !out.warnings.iter().any(|w| w.contains("lower bound")),
        "no NaN-OFV warning should fire when an OFV was requested"
    );
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// The mean-field family runs and reports itself, with the smaller `φ` it implies.
#[test]
fn mean_field_family_is_selectable() {
    let (model, population, params) = fixture();
    let mut o = opts(20);
    o.vi_family = ViFamily::MeanField;
    let out = run_vi(&model, &population, &params, &o).unwrap();
    let vi = out.vi.as_ref().unwrap();
    assert_eq!(vi.family, "mean_field");
    // Mean-field cannot represent posterior correlation, so the off-diagonal of
    // every reported covariance must be exactly zero.
    for cov in &vi.eta_covs {
        assert_eq!(cov[0][1], 0.0);
        assert_eq!(cov[1][0], 0.0);
    }
}

/// Both `Ω` update routes run and land in the same neighbourhood. They are not
/// expected to agree exactly — one is an exact maximizer, the other a gradient
/// step — but a gross disagreement would mean the Cholesky chaining in one of
/// them is wrong.
#[test]
fn both_omega_update_routes_agree_approximately() {
    let (model, population, params) = fixture();

    let mut closed = opts(150);
    closed.vi_omega_update = ViOmegaUpdate::ClosedForm;
    let a = run_vi(&model, &population, &params, &closed).unwrap();

    let mut adam = opts(150);
    adam.vi_omega_update = ViOmegaUpdate::Adam;
    let b = run_vi(&model, &population, &params, &adam).unwrap();

    for k in 0..2 {
        let (x, y) = (a.params.omega.matrix[(k, k)], b.params.omega.matrix[(k, k)]);
        assert!(x.is_finite() && y.is_finite());
        assert!(
            (x - y).abs() / x.abs().max(y.abs()).max(1e-6) < 1.0,
            "omega[{k},{k}]: closed-form {x:.6}, adam {y:.6} — routes disagree grossly"
        );
    }
}

/// A FIXed `θ` must not move. Adam's step for a zeroed gradient is exactly zero,
/// so this is a bit-equality check rather than a tolerance.
#[test]
fn fixed_parameters_do_not_move() {
    let (model, population, mut params) = fixture();
    params.theta_fixed = vec![false, true];
    params.sigma_fixed = vec![true];
    let want_theta = params.theta[1];
    let want_sigma = params.sigma.values[0];

    let out = run_vi(&model, &population, &params, &opts(40)).unwrap();
    assert_eq!(
        out.params.theta[1].to_bits(),
        want_theta.to_bits(),
        "a FIXed theta moved"
    );
    assert_eq!(
        out.params.sigma.values[0].to_bits(),
        want_sigma.to_bits(),
        "a FIXed sigma moved"
    );
    assert!(
        out.params.theta[0] != params.theta[0],
        "the free theta should still have been estimated"
    );
}

/// A declared `θ` box must hold.
///
/// Adam is unconstrained, so the box is enforced only by the projection in
/// `run_vi`; without it a bound is *silently* ignored, which is worse than
/// refusing it. The box below is positioned to exclude wherever the free fit
/// actually goes, so the constraint is genuinely active rather than incidentally
/// satisfied — and the free estimate is asserted to lie outside it, which is what
/// makes this a regression test rather than a tautology.
#[test]
fn declared_theta_bounds_are_enforced() {
    let (model, population, params) = fixture();

    let free = run_vi(&model, &population, &params, &opts(60))
        .unwrap()
        .params
        .theta[0];
    assert!(
        (free - params.theta[0]).abs() > 1e-6,
        "the free fit must move TVCL for this test to constrain anything (got {free})"
    );

    // A box containing the initial estimate but not `free`, on whichever side of
    // the init the free fit did not travel toward.
    let midpoint = 0.5 * (free + params.theta[0]);
    let (lo, hi) = if free < params.theta[0] {
        (midpoint, 10.0)
    } else {
        (0.1, midpoint)
    };

    let mut bounded = params.clone();
    bounded.theta_lower[0] = lo;
    bounded.theta_upper[0] = hi;
    let got = run_vi(&model, &population, &bounded, &opts(60))
        .unwrap()
        .params
        .theta[0];

    // ULP-scale slack: the box is enforced in *packed* space, exactly as it is for
    // FOCE/FOCEI, so `x` is clamped to `ln(lo)` and `exp(ln(lo))` need not be
    // bit-identical to `lo`. Snapping the natural-scale value instead would make VI
    // behave differently from every other estimator. This is still ~4 orders tighter
    // than the pre-fix escape (a few percent of the bound).
    let slack = |b: f64| 1e-12 * (1.0 + b.abs());
    assert!(
        got >= lo - slack(lo) && got <= hi + slack(hi),
        "TVCL = {got} escaped its declared bounds [{lo}, {hi}]"
    );
    assert!(
        free < lo || free > hi,
        "test is vacuous: the unbounded estimate {free} is already inside [{lo}, {hi}]"
    );
}

/// The covariance step runs at the VI estimate, so a VI fit reports standard
/// errors like every other method instead of a `FAILED` covariance status.
///
/// It is the ordinary FD-of-OFV Hessian — a *Laplace* covariance at the VI point —
/// not anything derived from `vi.eta_covs`, which are per-subject posterior
/// variances that variational families are known to understate.
#[test]
fn covariance_step_runs_at_the_vi_estimate() {
    let (model, population, params) = fixture();
    let mut o = opts(40);
    o.run_covariance_step = true;
    o.vi_final_ofv = ViFinalOfv::Laplace;

    let out = run_vi(&model, &population, &params, &o).unwrap();
    let cov = out
        .covariance_matrix
        .as_ref()
        .expect("VI must run the covariance step");

    let n = crate::estimation::parameterization::pack_params(&out.params).len();
    assert_eq!(cov.nrows(), n, "covariance is packed-parameter sized");
    assert_eq!(cov.ncols(), n);
    for i in 0..n {
        assert!(cov[(i, i)] >= 0.0, "negative variance at [{i},{i}]");
        for j in 0..n {
            assert!(
                (cov[(i, j)] - cov[(j, i)]).abs() <= 1e-10 * (1.0 + cov[(i, j)].abs()),
                "covariance must be symmetric at [{i},{j}]"
            );
        }
    }
    assert!(out.covariance_wall_time_secs > 0.0, "cov time not recorded");

    // The exact packed vector the step ran at, so a standalone `run_covariance`
    // reproduces it rather than re-deriving `chol(Ω)` from `L·Lᵀ`.
    assert_eq!(
        out.packed_estimate.as_deref(),
        Some(crate::estimation::parameterization::pack_params(&out.params).as_slice()),
        "packed_estimate must be the vector the covariance step used"
    );
}

/// And the step honours its gate: `fit_inner` clears `run_covariance_step` on every
/// non-terminal stage of a chain, so a `methods = vi, focei` run must not pay for a
/// covariance matrix twice.
#[test]
fn covariance_step_is_skipped_when_not_requested() {
    let (model, population, params) = fixture();
    let mut o = opts(20);
    o.run_covariance_step = false;

    let out = run_vi(&model, &population, &params, &o).unwrap();
    assert!(out.covariance_matrix.is_none());
    assert_eq!(out.covariance_wall_time_secs, 0.0);
}

/// `vi_kl = mc` runs end to end, reports which route it took, and lands in the same
/// neighbourhood as the analytic KL.
///
/// The two are not expected to agree closely — the whole point of the default is that
/// sampling the KL is noisier — but a gross disagreement would mean the Monte-Carlo
/// kernel has the wrong sign or scale somewhere.
#[test]
fn mc_kl_route_runs_and_reports_itself() {
    let (model, population, params) = fixture();

    let analytic = run_vi(&model, &population, &params, &opts(80)).unwrap();
    assert_eq!(analytic.vi.as_ref().unwrap().kl, "analytic");

    let mut o = opts(80);
    o.vi_kl = ViKl::Mc;
    // The KL is sampled here, so give the estimator more draws than the default.
    o.vi_mc_samples = 8;
    let mc = run_vi(&model, &population, &params, &o).unwrap();
    let vi = mc.vi.as_ref().unwrap();

    assert_eq!(vi.kl, "mc", "the route actually taken must be reported");
    assert_eq!(
        vi.n_kl_fallback_subjects, 0,
        "asking for mc outright is not a fallback"
    );
    assert!(vi.neg_two_elbo.is_finite());
    assert!(
        !mc.warnings.iter().any(|w| w.contains("no closed-form KL")),
        "no fallback warning should fire when mc was requested: {:?}",
        mc.warnings
    );

    // The objective still descends.
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    assert!(
        mean(&vi.elbo_trace[70..]) < mean(&vi.elbo_trace[..10]),
        "the mc-KL objective should still decrease"
    );

    for k in 0..2 {
        let (a, m) = (analytic.params.theta[k], mc.params.theta[k]);
        assert!(
            (a - m).abs() / a.abs().max(1e-6) < 0.5,
            "theta[{k}]: analytic {a:.6}, mc {m:.6} — routes disagree grossly"
        );
    }
    for k in 0..2 {
        let (a, m) = (
            analytic.params.omega.matrix[(k, k)],
            mc.params.omega.matrix[(k, k)],
        );
        assert!(m > 0.0, "omega[{k},{k}] must stay positive under mc-KL");
        assert!(
            (a - m).abs() / a.abs().max(1e-6) < 1.0,
            "omega[{k},{k}]: analytic {a:.6}, mc {m:.6}"
        );
    }
}

/// The fallback warning fires only when a family actually had no closed form, and
/// names the option that silences it.
#[test]
fn kl_fallback_warning_reports_only_a_real_fallback() {
    assert!(kl_fallback_warning(0, 10, "full_rank").is_none());

    let w = kl_fallback_warning(3, 10, "some_flow").expect("a real fallback must warn");
    assert!(w.contains("some_flow"), "the family should be named: {w}");
    assert!(w.contains('3') && w.contains("10"), "counts missing: {w}");
    assert!(
        w.contains("vi_kl = mc"),
        "the warning should name the option that silences it: {w}"
    );
}

/// A mixed `block_omega` + standalone-eta `Ω` keeps its declared structure through a
/// whole fit, under **both** `Ω` update routes.
///
/// The two routes reach it differently and both need checking. `closed_form` masks
/// `Ω` in `Ω` space every iteration. `adam` relies on the gradient skip in
/// `chain_omega_grad` plus a fact about the parameterization: the Cholesky factor of
/// a matrix that is block-diagonal under permutation is itself zero wherever the
/// matrix is, so a packed slot that starts at `0.0` and is never stepped
/// reconstructs `Ω[i,j] = Σₖ L[i,k]·L[j,k]` as *exactly* `0.0`. Hence the
/// bit-equality assertion rather than a tolerance — a tolerance would hide a slot
/// that had drifted a little.
#[test]
fn mixed_omega_keeps_its_structural_zeros_through_a_fit() {
    for route in [ViOmegaUpdate::ClosedForm, ViOmegaUpdate::Adam] {
        let (model, population, params) = mixed_omega_fixture();
        let mut o = opts(60);
        o.vi_omega_update = route;

        let out = run_vi(&model, &population, &params, &o).unwrap();
        let om = &out.params.omega.matrix;

        for (i, j) in [(2usize, 0usize), (2, 1), (0, 2), (1, 2)] {
            assert_eq!(
                om[(i, j)],
                0.0,
                "{route:?}: Ω({i},{j}) is a structural zero but came back {}",
                om[(i, j)]
            );
        }
        // The within-block covariance is a real parameter and must still be estimated.
        assert!(
            om[(1, 0)] != 0.0,
            "{route:?}: the (ETA_CL, ETA_V) block covariance was not estimated"
        );
        assert!(
            (om[(1, 0)] - om[(0, 1)]).abs() < 1e-15,
            "{route:?}: Ω must stay symmetric"
        );
        for k in 0..3 {
            assert!(om[(k, k)] > 0.0, "{route:?}: Ω({k},{k}) must stay positive");
        }
    }
}

/// A FIXed eta inside a block keeps its whole declared row and column through a fit
/// on the default (closed-form) route, where the restoration is exact.
#[test]
fn fixed_eta_in_a_block_keeps_its_declared_row() {
    let (model, population, mut params) = mixed_omega_fixture();
    params.omega_fixed = vec![false, true, false];
    let declared = params.omega.matrix.clone();

    let out = run_vi(&model, &population, &params, &opts(50)).unwrap();
    let om = &out.params.omega.matrix;

    for k in 0..3 {
        assert_eq!(
            om[(1, k)].to_bits(),
            declared[(1, k)].to_bits(),
            "Ω(1,{k}) touches FIXed ETA_V and must keep its declared value"
        );
        assert_eq!(om[(k, 1)].to_bits(), declared[(k, 1)].to_bits());
    }
    assert!(
        (om[(0, 0)] - declared[(0, 0)]).abs() > 1e-9,
        "the free ETA_CL variance should still have been estimated"
    );
}

/// `vi_eta_grad = fd` runs, reports every subject as finite-differenced, and lands
/// close to the analytic route — the switch changes speed, not the fit.
#[test]
fn fd_eta_grad_mode_agrees_with_the_analytic_route() {
    let (model, population, params) = fixture();
    let a = run_vi(&model, &population, &params, &opts(60)).unwrap();

    let mut o = opts(60);
    o.vi_eta_grad = ViEtaGrad::Fd;
    let b = run_vi(&model, &population, &params, &o).unwrap();

    assert_eq!(a.vi.as_ref().unwrap().n_fd_subjects, 0);
    assert_eq!(b.vi.as_ref().unwrap().n_fd_subjects, 3);
    for k in 0..2 {
        let (x, y) = (a.params.theta[k], b.params.theta[k]);
        assert!(
            (x - y).abs() / x.abs().max(1e-6) < 1e-3,
            "theta[{k}]: analytic {x:.8}, fd {y:.8}"
        );
    }
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// IOV is rejected with a message naming the alternative, rather than fitting a
/// model whose `κ` the variational family cannot represent.
#[test]
fn iov_models_are_refused() {
    let (mut model, population, params) = fixture();
    model.n_kappa = 1;
    // `OuterResult` is not `Debug`, so match rather than `expect_err`.
    let err = match run_vi(&model, &population, &params, &opts(5)) {
        Err(e) => e,
        Ok(_) => panic!("IOV must be refused"),
    };
    assert!(err.contains("IOV"), "unhelpful error: {err}");
    assert!(err.contains("saem") || err.contains("focei"));
}

/// An empty population is refused rather than producing a vacuous fit.
#[test]
fn empty_population_is_refused() {
    let (model, _, params) = fixture();
    let empty = Population {
        subjects: vec![],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    assert!(run_vi(&model, &empty, &params, &opts(5)).is_err());
}

/// M3 censoring routes through the full-FD branch of the data term rather than
/// being silently mis-scored, so a BLOQ model must still fit.
#[test]
fn m3_censored_models_run() {
    let (mut model, mut population, params) = fixture();
    model.bloq_method = BloqMethod::M3;
    population.subjects[0].cens = vec![0, 0, 1];
    let out = run_vi(&model, &population, &params, &opts(20)).expect("M3 model fits");
    assert!(out.vi.as_ref().unwrap().neg_two_elbo.is_finite());
}

/// The `analytical_model` shell has no differentiable closed form, so every
/// subject falls back to finite differences — and the run must say so in a
/// warning rather than quietly taking 40× longer.
#[test]
fn fd_fallback_is_warned_about() {
    let (_, population, _) = fixture();
    let shell = crate::types::test_helpers::analytical_model(GradientMethod::Auto);
    let shell_params = shell.default_params.clone();
    let out = run_vi(&shell, &population, &shell_params, &opts(10)).unwrap();
    assert_eq!(out.vi.as_ref().unwrap().n_fd_subjects, 3);
    assert!(
        out.warnings.iter().any(|w| w.contains("finite-difference")),
        "an all-FD run must warn; warnings were {:?}",
        out.warnings
    );
}
