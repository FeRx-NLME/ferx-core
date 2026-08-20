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

/// An IOV model now fits end to end, with a variational posterior over the stacked
/// `[η, κ₁ … κ_K]` and per-occasion `κ` means reported alongside the `η` moments.
///
/// The two subjects carry **different** occasion counts, so this exercises the
/// per-subject family dimension rather than a population that happens to be uniform.
#[test]
fn iov_models_fit_and_report_per_occasion_kappa_means() {
    let (model, population, params) = iov_fixture();
    assert!(model.n_kappa > 0);

    let out = run_vi(&model, &population, &params, &opts(40)).expect("IOV must fit");
    let vi = out.vi.as_ref().expect("VI result present");

    // eta moments stay `n_eta`-shaped: they are the BSV head of the stacked posterior.
    assert_eq!(vi.eta_means.len(), population.subjects.len());
    for m in &vi.eta_means {
        assert_eq!(m.len(), model.n_eta);
    }
    // kappa means are per subject, per occasion.
    assert_eq!(vi.kappa_means.len(), population.subjects.len());
    assert_eq!(vi.kappa_means[0].len(), 2, "subject 1 has two occasions");
    assert_eq!(vi.kappa_means[1].len(), 3, "subject 2 has three occasions");
    for subj in &vi.kappa_means {
        for occ in subj {
            assert_eq!(occ.len(), model.n_kappa);
            assert!(occ.iter().all(|v| v.is_finite()));
        }
    }
    assert!(vi.neg_two_elbo.is_finite());
    // The closed form must have produced an Omega_iov rather than leaving it untouched.
    let iov = out
        .params
        .omega_iov
        .as_ref()
        .expect("an IOV fit must report Omega_iov");
    assert!(iov.matrix[(0, 0)] > 0.0 && iov.matrix[(0, 0)].is_finite());
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

// ---------------------------------------------------------------------------
// Early stopping
// ---------------------------------------------------------------------------

/// `vi_iters` is a ceiling, so a fit that settles must stop well short of it, report the
/// iterations it *actually* ran, and still come back converged without a warning.
///
/// The regression this guards is the pairing, not either half: raising the `vi_iters`
/// default without early stopping would make every easy fit pay the ceiling, and early
/// stopping that forgot to re-window `trace_has_settled` onto the realised run length
/// would report `converged: false` for every early stop.
#[test]
fn settled_runs_stop_well_short_of_the_iteration_ceiling() {
    let (model, population, params) = fixture();
    let out = run_vi(&model, &population, &params, &opts(25_000)).expect("VI runs");
    let vi = out.vi.as_ref().expect("VI result present");

    assert!(
        vi.n_iterations < 25_000,
        "a settled fit must stop before the ceiling, but ran all {} iterations",
        vi.n_iterations
    );
    assert_eq!(
        vi.n_iterations,
        vi.elbo_trace.len(),
        "reported n_iterations must equal the number of iterations actually evaluated"
    );
    assert!(
        vi.converged,
        "a run stops early precisely *because* it settled, so it must report converged"
    );
    assert!(
        !out.warnings.iter().any(|w| w.contains("still moving")),
        "an early-stopped run must not warn that the objective was still moving: {:?}",
        out.warnings
    );
}

/// A run given no room to settle still terminates at the ceiling and reports it
/// honestly — the path that has to keep working now that stopping is conditional.
#[test]
fn unsettled_runs_still_stop_at_the_ceiling() {
    let (model, population, params) = fixture();
    let out = run_vi(&model, &population, &params, &opts(40)).expect("VI runs");
    let vi = out.vi.as_ref().expect("VI result present");
    assert_eq!(vi.n_iterations, 40);
    assert_eq!(vi.elbo_trace.len(), 40);
}

/// The noise-aware half of [`trace_has_settled`], on traces shaped like a real fit
/// rather than the synthetic ones above.
///
/// This is the regression for why early stopping never fired: a plateaued warfarin run
/// sits around `-286` with window-to-window scatter of order 0.1, while the purely
/// relative threshold is `1e-4 * (1 + 286) ~ 0.029`. The old criterion could therefore
/// never be satisfied by a converged fit, and the run always burned its whole budget.
#[test]
fn settling_is_judged_against_noise_not_objective_magnitude() {
    // Deterministic pseudo-noise, so this test cannot flake.
    let noise = |i: usize| ((i * 37 + 11) as f64).sin();

    // Flat at a realistic OFV magnitude, with noise far larger than the relative
    // threshold. This is a converged fit and must be recognised as one.
    let flat: Vec<f64> = (0..200).map(|i| -286.0 + noise(i)).collect();
    assert!(
        trace_has_settled(&flat, 50, CONVERGENCE_REL_TOL),
        "a flat but noisy trace at OFV scale must count as settled; this is exactly the \
         case the purely relative criterion could never satisfy"
    );

    // The same trace judged with the noise term removed (rel_tol only, as the old
    // criterion effectively was) is NOT settled — pinning the behaviour that changed.
    let n = flat.len();
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let d = (mean(&flat[n - 100..n - 50]) - mean(&flat[n - 50..])).abs();
    assert!(
        d > CONVERGENCE_REL_TOL * (1.0 + 286.0),
        "fixture must actually exceed the old relative threshold, else it proves nothing \
         (delta {d:.6})"
    );

    // A drift that is small in relative terms but large against the noise must still be
    // rejected: the criterion has to catch slow descent, not just obvious descent.
    let drifting: Vec<f64> = (0..200)
        .map(|i| -286.0 + noise(i) - 0.05 * i as f64)
        .collect();
    assert!(
        !trace_has_settled(&drifting, 50, CONVERGENCE_REL_TOL),
        "a trace still descending well above its own noise level must not count as settled"
    );

    // A noiseless trace still settles on the relative floor alone, so the new criterion
    // is never stricter than the one it replaces.
    let constant = vec![-286.0; 200];
    assert!(trace_has_settled(&constant, 50, CONVERGENCE_REL_TOL));
}

/// A two-subject IOV population on a closed-form 1-cpt IV model, with a differing
/// occasion count per subject.
///
/// The differing `K` is the point: it is what makes the stacked dimension a per-subject
/// quantity, so a fixture where every subject agreed would not exercise
/// [`Families::PerSubject`] or catch an offset computed from the wrong subject's `K`.
fn iov_fixture() -> (CompiledModel, Population, ModelParameters) {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  kappa KAPPA_CL ~ 0.02
  sigma PROP_ERR ~ 0.04

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  iov_column = OCC
",
    )
    .expect("IOV fixture parses");

    let subject = |id: &str, occ: Vec<u32>, scale: f64| Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times: (0..occ.len()).map(|k| 1.0 + 2.0 * k as f64).collect(),
        obs_raw_times: Vec::new(),
        observations: (0..occ.len())
            .map(|k| scale * (8.0 - 0.9 * k as f64))
            .collect(),
        obs_cmts: vec![1; occ.len()],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        cens: vec![0; occ.len()],
        occasions: occ,
        obs_l2: Vec::new(),
        dose_occasions: vec![1],
        fremtype: Vec::new(),
        obs_records: vec![],
    };

    let population = Population {
        // Two occasions for the first subject, three for the second.
        subjects: vec![
            subject("1", vec![1, 1, 2, 2], 1.0),
            subject("2", vec![1, 2, 2, 3, 3], 1.15),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    };
    let params = model.default_params.clone();
    (model, population, params)
}

/// `max_relative_change` is scale-aware and does not blow up near zero.
#[test]
fn parameter_change_is_relative_with_an_absolute_floor() {
    // A 1% move on an O(1) coordinate.
    assert!((max_relative_change(&[1.01], &[1.0]) - 0.0099).abs() < 1e-3);
    // The same *absolute* move on a coordinate 100x larger is 100x smaller relatively.
    assert!(max_relative_change(&[100.01], &[100.0]) < 1e-3);
    // Two coordinates: the worst one wins, so a single unsettled parameter blocks.
    assert!(max_relative_change(&[1.0, 2.0], &[1.0, 1.0]) > 0.4);
    // Near zero the floor keeps it finite rather than dividing by ~0.
    let near_zero = max_relative_change(&[1e-9], &[0.0]);
    assert!(near_zero.is_finite() && near_zero < 1.0, "got {near_zero}");
    // Identical vectors have not moved.
    assert_eq!(
        max_relative_change(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0]),
        0.0
    );
}

/// A run whose **parameters** have settled must be reported as converged even when the
/// objective is still creeping.
///
/// This is the case that motivated the criterion. A neural-network weight vector has
/// exact permutation and layer-scale symmetries, so an unregularised fit drifts along
/// those flat directions forever: the objective test says "still moving" indefinitely
/// while the estimates, and everything a user reads off them, have stopped. Measured on a
/// 146-weight DCM, the objective test never fired in 25 000 iterations even though the
/// fit was sound (weights bounded, sigma sensible, OFV better than SAEM's).
///
/// Exercised here through the ordinary fixture rather than a DCM, which needs
/// `--features nn`: the assertion is that the two criteria are independent and either
/// suffices, not that this particular model has flat directions.
#[test]
fn parameter_stability_alone_can_certify_convergence() {
    let (model, population, params) = fixture();
    // Long enough that the parameters settle well before the ceiling.
    let out = run_vi(&model, &population, &params, &opts(25_000)).expect("VI runs");
    let vi = out.vi.as_ref().expect("VI result present");
    assert!(vi.converged, "a settled fit must report converged");
    assert!(
        vi.n_iterations < 25_000,
        "must stop before the ceiling, ran {}",
        vi.n_iterations
    );
    assert!(
        !out.warnings.iter().any(|w| w.contains("had settled")),
        "a converged run must not warn about settling: {:?}",
        out.warnings
    );
}

// ---------------------------------------------------------------------------
// Closed-form σ route
// ---------------------------------------------------------------------------

/// The M-step must actually be wired into the loop, not merely available.
///
/// Asserted by the two routes disagreeing on `σ` from the same fixture, seed and
/// draw stream: `ViSigmaUpdate::ClosedForm` replaces `σ` with the exact maximizer
/// every iteration where `Adam` steps it, so an identical `σ` would mean the option
/// is being ignored. (The formula's *correctness* is pinned separately, against the
/// vanishing σ-gradient, in `closed_form_sigma_zeroes_the_sigma_gradient`.)
#[test]
fn sigma_closed_form_route_moves_sigma_off_the_adam_path() {
    let (model, population, params) = fixture();

    let mut o_cf = opts(60);
    o_cf.vi_sigma_update = crate::types::ViSigmaUpdate::ClosedForm;
    let cf = run_vi(&model, &population, &params, &o_cf).unwrap();

    let mut o_adam = opts(60);
    o_adam.vi_sigma_update = crate::types::ViSigmaUpdate::Adam;
    let adam = run_vi(&model, &population, &params, &o_adam).unwrap();

    assert_ne!(
        cf.params.sigma.values[0], adam.params.sigma.values[0],
        "both σ routes returned {}, so vi_sigma_update is not reaching the loop",
        cf.params.sigma.values[0]
    );
    // Neither route may emit the fallback warning on this model: the fixture is a
    // single proportional σ, which is exactly the supported case.
    assert!(
        !cf.warnings.iter().any(|w| w.contains("vi_sigma_update")),
        "unexpected σ fallback on a supported model: {:?}",
        cf.warnings
    );
}

/// A model outside the scalar derivation's scope must fall back to Adam **and say
/// so**. A silent fallback would apply a formula that does not describe the model,
/// which is the failure mode the support predicate exists to prevent.
#[test]
fn sigma_closed_form_falls_back_loudly_on_combined_error() {
    let model = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP_ERR ~ 0.04
  sigma ADD_ERR ~ 0.1

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
",
    )
    .expect("combined-error fixture parses");
    let (_m, population, _p) = fixture();
    let params = model.default_params.clone();

    let mut o = opts(40);
    o.vi_sigma_update = crate::types::ViSigmaUpdate::ClosedForm;
    let out = run_vi(&model, &population, &params, &o).unwrap();

    let w = out
        .warnings
        .iter()
        .find(|w| w.contains("vi_sigma_update"))
        .unwrap_or_else(|| panic!("no σ fallback warning; warnings were {:?}", out.warnings));
    assert!(
        w.contains("combined") && w.contains("Adam"),
        "the warning must name the reason and the route taken: {w}"
    );
    // The combined model is refused for the *right* reason. It necessarily declares two
    // σ, so an ordering that checked the count first would report that instead — true,
    // but not the thing the user needs to know.
    assert!(
        !w.contains("σ parameters"),
        "refused on the σ count rather than on the error structure: {w}"
    );
}

// ---------------------------------------------------------------------------
// Vacuous parameter-stability criterion
// ---------------------------------------------------------------------------

/// A model whose population vector cannot move must not be judged by whether it moved.
///
/// The regression: with `θ`, `Ω` and `σ` all FIXed, `max_relative_change` compares a constant
/// against itself, reports "settled" at its first opportunity, and stops the fit — while `φ`, the
/// only thing being optimized in that configuration and the one thing the criterion cannot see,
/// is nowhere near converged. Measured on warfarin before the guard: 500 iterations,
/// `converged: true`, `elbo_tightness_ratio: 78` (implausible above 25) and `−2·ELBO = +2026` on
/// a model that reaches `−283` once `φ` is allowed to finish.
#[test]
fn param_criterion_is_vacuous_when_every_population_parameter_is_fixed() {
    let (model, _population, params) = fixture();
    assert!(
        super::param_criterion_applies(&params),
        "a model with free θ/Ω/σ must keep the parameter-stability criterion"
    );

    let all_fixed = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0) FIX
  theta TVV(10.0, 1.0, 100.0) FIX
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V  ~ 0.04 FIX
  sigma PROP_ERR ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("all-FIXed fixture parses");
    assert!(
        !super::param_criterion_applies(&all_fixed.default_params),
        "with every population coordinate FIXed the criterion is vacuous and must be disabled"
    );

    // A single free coordinate is enough to make it meaningful again — the guard must key on
    // "nothing can move", not on "something is fixed", which is the common and harmless case.
    let one_free = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0)
  theta TVV(10.0, 1.0, 100.0) FIX
  omega ETA_CL ~ 0.09 FIX
  sigma PROP_ERR ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("one-free fixture parses");
    assert!(
        super::param_criterion_applies(&one_free.default_params),
        "one free θ is enough for the criterion to mean something"
    );
}

/// And the user is told, because it changes what `converged` is based on.
#[test]
fn all_fixed_population_says_convergence_rests_on_the_objective_alone() {
    let all_fixed = crate::parser::model_parser::parse_model_string(
        r"
[parameters]
  theta TVCL(1.0, 0.1, 10.0) FIX
  theta TVV(10.0, 1.0, 100.0) FIX
  omega ETA_CL ~ 0.09 FIX
  omega ETA_V  ~ 0.04 FIX
  sigma PROP_ERR ~ 0.04 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
",
    )
    .expect("all-FIXed fixture parses");
    let (_m, population, _p) = fixture();
    let out = run_vi(
        &all_fixed,
        &population,
        &all_fixed.default_params,
        &opts(40),
    )
    .unwrap();
    assert!(
        out.warnings
            .iter()
            .any(|w| w.contains("every population parameter is FIXed")),
        "the all-FIXed case must say what convergence now rests on; warnings were {:?}",
        out.warnings
    );

    // The ordinary case must stay quiet about it.
    let (model, population, params) = fixture();
    let free = run_vi(&model, &population, &params, &opts(40)).unwrap();
    assert!(
        !free
            .warnings
            .iter()
            .any(|w| w.contains("every population parameter is FIXed")),
        "a model with free parameters must not get the all-FIXed warning: {:?}",
        free.warnings
    );
}
