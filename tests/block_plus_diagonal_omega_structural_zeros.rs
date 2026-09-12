//! A `block_omega` next to a separate diagonal `omega` declares the cross-block
//! covariances as structural zeros (docs/model-file/parameters.qmd, "Mixing
//! diagonal and block"); so does a `block_kappa` next to a separate `kappa`.
//! Before #1018 the outer optimizer searched the cross-block Cholesky entries
//! anyway: `pack_with_bounds` pinned only FIX coordinates and the analytic
//! gradient zeroed only FIX coordinates, so a `block_omega (ETA_CL, ETA_V)` +
//! `omega ETA_KA` fit estimated `Cov(KA, CL)` and `Cov(KA, V)` and landed on the
//! full 3×3 block optimum — same OFV, same Ω — while `n_parameters` and the
//! covariance step (#243, #1177) both treated those entries as absent.
//!
//! Tier 2: every fit here is capped at a handful of outer iterations.

use ferx_core::parser::model_parser::parse_model_string;
use ferx_core::types::omega_se_at;
use ferx_core::{fit, read_nonmem_csv, EstimationMethod, FitOptions, FitResult, Optimizer};
use std::path::Path;

/// `(data file, occasion column)`.
type DataSet = (&'static str, Option<&'static str>);
const WARFARIN: DataSet = ("data/warfarin_block_omega.csv", None);
const WARFARIN_IOV: DataSet = ("data/warfarin_iov.csv", Some("OCC"));

const PARTIAL_BLOCK_SRC: &str = "
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.07, 0.02, 0.02]
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

const PARTIAL_OMEGA_DECL: &str =
    "block_omega (ETA_CL, ETA_V) = [0.07, 0.02, 0.02]\n  omega ETA_KA ~ 0.40";

/// The same starting Ω written as one full 3×3 block: the KA covariances start
/// at 0 but are free. This is the control that makes the zero assertions
/// meaningful — it proves the capped fit *does* move a cross covariance off 0
/// when the model allows it, so a partial-block fit that keeps them at 0 is the
/// mask at work, not an optimizer that never took a step.
fn full_block_src() -> String {
    PARTIAL_BLOCK_SRC.replace(
        PARTIAL_OMEGA_DECL,
        "block_omega (ETA_CL, ETA_V, ETA_KA) = [0.07, 0.02, 0.02, 0.0, 0.0, 0.40]",
    )
}

/// The partial block with the diagonal η declared *first*, so the structural
/// zeros sit at `(1,0)` and `(2,0)` instead of `(2,0)` and `(2,1)`.
fn diagonal_first_src() -> String {
    PARTIAL_BLOCK_SRC.replace(
        PARTIAL_OMEGA_DECL,
        "omega ETA_KA ~ 0.40\n  block_omega (ETA_CL, ETA_V) = [0.07, 0.02, 0.02]",
    )
}

const PARTIAL_BLOCK_KAPPA_SRC: &str = "
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.01]
  kappa KAPPA_KA ~ 0.01
  sigma PROP_ERR ~ 0.2 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V + KAPPA_V)
  KA = TVKA * exp(ETA_KA + KAPPA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)
";

/// The kappa counterpart of [`full_block_src`].
fn full_block_kappa_src() -> String {
    PARTIAL_BLOCK_KAPPA_SRC.replace(
        "block_kappa (KAPPA_CL, KAPPA_V) = [0.01, 0.002, 0.01]\n  kappa KAPPA_KA ~ 0.01",
        "block_kappa (KAPPA_CL, KAPPA_V, KAPPA_KA) = [0.01, 0.002, 0.01, 0.0, 0.0, 0.01]",
    )
}

/// Fit capped at 4 outer iterations, or an evaluation-only run
/// (`outer_maxiter = 0`) that reports the start as the fit sees it.
#[allow(clippy::too_many_arguments)]
fn fit_with(
    src: &str,
    data: DataSet,
    method: EstimationMethod,
    interaction: bool,
    optimizer: Optimizer,
    eval_only: bool,
    covariance: bool,
) -> FitResult {
    let model = parse_model_string(src).expect("model parses");
    let pop = read_nonmem_csv(Path::new(data.0), None, data.1).expect("data loads");
    let mut opts = FitOptions::default();
    opts.method = method;
    opts.interaction = interaction;
    opts.optimizer = optimizer;
    opts.outer_maxiter = if eval_only { 0 } else { 4 };
    opts.run_covariance_step = covariance;
    opts.verbose = false;
    fit(&model, &pop, &model.default_params, &opts).expect("capped fit runs")
}

/// The FOCE (no interaction) arm most of this file uses.
fn capped_fit(
    src: &str,
    data: DataSet,
    optimizer: Optimizer,
    eval_only: bool,
    covariance: bool,
) -> FitResult {
    fit_with(
        src,
        data,
        EstimationMethod::Foce,
        false,
        optimizer,
        eval_only,
        covariance,
    )
}

/// The four cross-block entries of a mixed block + diagonal matrix are all
/// exactly 0. `at` reads the matrix under test (Ω or Ω_IOV), `names` are its
/// row labels.
fn assert_cross_zeros(
    label: &str,
    names: &[String],
    at: impl Fn(usize, usize) -> f64,
    block: (&str, &str),
    diag: &str,
) {
    let (ia, ib, id) = (idx(names, block.0), idx(names, block.1), idx(names, diag));
    for (i, j) in [(id, ia), (ia, id), (id, ib), (ib, id)] {
        assert!(
            at(i, j) == 0.0,
            "{label}: [{},{}] = {:e} must be a structural zero",
            names[i],
            names[j],
            at(i, j)
        );
    }
}

fn idx(names: &[String], name: &str) -> usize {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("{name} not in {names:?}"))
}

/// A free covariance moved off its start. The reference is an evaluation-only
/// fit of the same model, not the declared number: Ω is rebuilt from the packed
/// `exp(ln L₀₀)`·`L₁₀`, which is not bit-identical to the declared value, so a
/// comparison against the literal would pass on a fit that never stepped.
fn assert_moved(what: &str, got: f64, start: f64) {
    assert!(
        got.is_finite() && (got - start).abs() > 1e-6 * start.abs(),
        "{what}: {got:e} did not move from its start {start:e}"
    );
}

/// Every optimizer family that walks the packed vector through
/// `pack_with_bounds` + the held mask: NLopt derivative-free (BOBYQA), NLopt
/// gradient (L-BFGS, SLSQP), the built-in BFGS, and the argmin trust region.
const OPTIMIZERS: [Optimizer; 5] = [
    Optimizer::Bobyqa,
    Optimizer::Lbfgs,
    Optimizer::Slsqp,
    Optimizer::Bfgs,
    Optimizer::TrustRegion,
];

#[test]
fn partial_block_holds_structural_zeros_and_the_full_block_does_not() {
    let orders = [
        ("block then diagonal", PARTIAL_BLOCK_SRC.to_string()),
        ("diagonal then block", diagonal_first_src()),
    ];
    // No optimizer runs in an evaluation-only fit, so one per declaration order
    // serves every optimizer below.
    let starts: Vec<FitResult> = orders
        .iter()
        .map(|(_, src)| capped_fit(src, WARFARIN, Optimizer::Lbfgs, true, false))
        .collect();

    for optimizer in OPTIMIZERS {
        let mut partial_ofv = f64::NAN;
        for ((label, src), start) in orders.iter().zip(&starts) {
            let r = capped_fit(src, WARFARIN, optimizer, false, false);
            let (cl, v, ka) = (
                idx(&r.eta_names, "ETA_CL"),
                idx(&r.eta_names, "ETA_V"),
                idx(&r.eta_names, "ETA_KA"),
            );
            for (i, j) in [(ka, cl), (cl, ka), (ka, v), (v, ka)] {
                // Exact: a structural zero is a declared model property, not an
                // estimate that happens to be small. `== 0.0` also rejects NaN.
                assert!(
                    r.omega[(i, j)] == 0.0,
                    "{optimizer:?} / {label}: Ω[{},{}] = {:e} must be a structural zero; Ω = {}",
                    r.eta_names[i],
                    r.eta_names[j],
                    r.omega[(i, j)],
                    r.omega
                );
            }
            // The in-block covariance is still estimated — it must not be frozen
            // together with the cross-block entries.
            assert_moved(
                &format!("{optimizer:?} / {label}: in-block Cov(CL, V)"),
                r.omega[(v, cl)],
                start.omega[(v, cl)],
            );
            // 3 θ + 3 Ω variances + 1 in-block covariance + 1 σ.
            assert_eq!(r.n_parameters, 8, "{optimizer:?} / {label}");
            if partial_ofv.is_nan() {
                partial_ofv = r.ofv;
            }
        }

        // The straddle: the identical start and data, with the KA covariances
        // free, leaves them off 0 within the same iteration cap.
        let full = capped_fit(&full_block_src(), WARFARIN, optimizer, false, false);
        let (cl, v, ka) = (0, 1, 2);
        assert_eq!(full.eta_names, ["ETA_CL", "ETA_V", "ETA_KA"]);
        assert!(
            full.omega[(ka, cl)] != 0.0 || full.omega[(ka, v)] != 0.0,
            "{optimizer:?}: full block left both KA covariances at 0 — the capped fit \
             cannot observe the structural-zero hold; Ω = {}",
            full.omega
        );
        assert_eq!(full.n_parameters, 10, "{optimizer:?}");
        assert!(
            full.ofv.is_finite() && partial_ofv.is_finite() && full.ofv != partial_ofv,
            "{optimizer:?}: partial-block OFV {partial_ofv} must differ from full-block OFV {}",
            full.ofv
        );
    }
}

/// The same hold on Ω_IOV: a `block_kappa` beside a separate `kappa`. L-BFGS
/// only — the trust-region outer loop does not take IOV models.
#[test]
fn partial_block_kappa_holds_structural_zeros_and_the_full_block_does_not() {
    let iov = |f: &FitResult| f.omega_iov.clone().expect("fit carries omega_iov");

    let start = capped_fit(
        PARTIAL_BLOCK_KAPPA_SRC,
        WARFARIN_IOV,
        Optimizer::Lbfgs,
        true,
        false,
    );
    let r = capped_fit(
        PARTIAL_BLOCK_KAPPA_SRC,
        WARFARIN_IOV,
        Optimizer::Lbfgs,
        false,
        false,
    );
    let (cl, v, ka) = (
        idx(&r.kappa_names, "KAPPA_CL"),
        idx(&r.kappa_names, "KAPPA_V"),
        idx(&r.kappa_names, "KAPPA_KA"),
    );
    let om = iov(&r);
    for (i, j) in [(ka, cl), (cl, ka), (ka, v), (v, ka)] {
        assert!(
            om[(i, j)] == 0.0,
            "Ω_IOV[{},{}] = {:e} must be a structural zero; Ω_IOV = {}",
            r.kappa_names[i],
            r.kappa_names[j],
            om[(i, j)],
            om
        );
    }
    assert_moved(
        "in-block Cov(KAPPA_CL, KAPPA_V)",
        om[(v, cl)],
        iov(&start)[(v, cl)],
    );
    // 3 θ + 3 Ω + 3 κ variances + 1 in-block κ covariance + 1 σ.
    assert_eq!(r.n_parameters, 11);

    let full = capped_fit(
        &full_block_kappa_src(),
        WARFARIN_IOV,
        Optimizer::Lbfgs,
        false,
        false,
    );
    let fom = iov(&full);
    let (fcl, fv, fka) = (
        idx(&full.kappa_names, "KAPPA_CL"),
        idx(&full.kappa_names, "KAPPA_V"),
        idx(&full.kappa_names, "KAPPA_KA"),
    );
    assert!(
        fom[(fka, fcl)] != 0.0 || fom[(fka, fv)] != 0.0,
        "full block_kappa left both KAPPA_KA covariances at 0 — the capped fit cannot \
         observe the hold; Ω_IOV = {fom}"
    );
    assert_eq!(full.n_parameters, 13);
}

/// The two gradient builders the FOCE arms above never reach, each of which lost
/// a `free_mask` gate in #1018 and is now held by `packed_fixed_mask` alone:
///
/// - the FOCEI Laplace-cached builder (`interaction = true`), and
/// - the finite-difference fallback, which every IOV model takes because the
///   analytic dispatcher requires `kappas.is_empty()` — so `method = gn` on a
///   `block_kappa` model is the arm that exercises it.
#[test]
fn focei_and_gauss_newton_builders_hold_structural_zeros() {
    let cases: [(&str, EstimationMethod, bool, Optimizer, DataSet, &str); 4] = [
        (
            "focei / lbfgs",
            EstimationMethod::FoceI,
            true,
            Optimizer::Lbfgs,
            WARFARIN,
            PARTIAL_BLOCK_SRC,
        ),
        (
            "focei / trust region",
            EstimationMethod::FoceI,
            true,
            Optimizer::TrustRegion,
            WARFARIN,
            PARTIAL_BLOCK_SRC,
        ),
        (
            "gn",
            EstimationMethod::FoceGn,
            false,
            Optimizer::Lbfgs,
            WARFARIN,
            PARTIAL_BLOCK_SRC,
        ),
        (
            "gn / IOV (finite-difference fallback)",
            EstimationMethod::FoceGn,
            false,
            Optimizer::Lbfgs,
            WARFARIN_IOV,
            PARTIAL_BLOCK_KAPPA_SRC,
        ),
    ];

    for (label, method, interaction, optimizer, data, src) in cases {
        let r = fit_with(src, data, method, interaction, optimizer, false, false);
        match r.omega_iov.as_ref() {
            Some(iov) => {
                assert_cross_zeros(
                    label,
                    &r.kappa_names,
                    |i, j| iov[(i, j)],
                    ("KAPPA_CL", "KAPPA_V"),
                    "KAPPA_KA",
                );
                assert_eq!(r.n_parameters, 11, "{label}");
            }
            None => {
                assert_cross_zeros(
                    label,
                    &r.eta_names,
                    |i, j| r.omega[(i, j)],
                    ("ETA_CL", "ETA_V"),
                    "ETA_KA",
                );
                assert_eq!(r.n_parameters, 8, "{label}");
            }
        }
        assert!(r.ofv.is_finite(), "{label}: OFV must be finite");
    }
}

/// After the fix `n_parameters` (8) and the covariance step agree on the free
/// set. `se_omega` for a non-diagonal Ω is the full column-major lower triangle
/// (#226) — 6 entries for 3 η — and the two structural-zero entries carry an SE
/// of exactly 0, like a FIX parameter.
#[test]
fn partial_block_covariance_step_reports_zero_se_on_structural_zeros() {
    let r = capped_fit(PARTIAL_BLOCK_SRC, WARFARIN, Optimizer::Lbfgs, false, true);
    assert_eq!(r.n_parameters, 8);
    let se = r
        .se_omega
        .as_ref()
        .expect("covariance step produced omega SEs");
    assert_eq!(se.len(), 6, "full lower triangle of a 3×3 Ω");
    for (i, j) in [(2, 0), (2, 1)] {
        let s = omega_se_at(&r.se_omega, 3, i, j).expect("lower-triangle SE present");
        assert!(
            s == 0.0,
            "SE(Ω[{i},{j}]) = {s:e} must be 0 for a structural zero"
        );
    }
    for (i, j) in [(0, 0), (1, 0), (1, 1), (2, 2)] {
        let s = omega_se_at(&r.se_omega, 3, i, j).expect("lower-triangle SE present");
        assert!(
            s.is_finite() && s > 0.0,
            "SE(Ω[{i},{j}]) = {s:e} must be a positive estimate"
        );
    }
    let cov = r.covariance_matrix.as_ref().expect("covariance matrix");
    // Free packed coordinates carry a non-zero diagonal; the two structural
    // zeros (and nothing else — no parameter here is FIX) carry exactly 0.
    let zero_diag = (0..cov.nrows()).filter(|&k| cov[(k, k)] == 0.0).count();
    assert_eq!(zero_diag, 2, "covariance diagonal = {:?}", cov.diagonal());
    assert_eq!(cov.nrows() - zero_diag, r.n_parameters);
}
