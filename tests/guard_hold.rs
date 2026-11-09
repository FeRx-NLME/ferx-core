//! #2797: a coordinate collapsed on ferx's internal lower guard is reported in
//! `FitResult::held_at_bound`, and the covariance step holds it — the other SEs conditional
//! on it at its estimate — only after a clean verdict on the unconditional `R`.
//!
//! Tier 2: every fit is an evaluation (`maxiter = 0`) at a fixed point, which runs the
//! covariance step once, so nothing here converges. A `maxiter = 0` fit reports
//! `converged = false`, so no test asserts it.
//!
//! The fixture (`tests/data/guard_hold/`, with its generator) is a one-compartment oral
//! population simulated with **no** KA variability, fitted with an ETA on KA: the ETA
//! collapses onto the rail ω = e⁻¹² = 6.1442123533282098e-6. The two points straddle the
//! verdict:
//! - **stationary** — the converged estimate (OFV 353.754540470156, macOS arm64): a clean
//!   `R`, so ETA_KA is held;
//! - **poor σ** — the same with σ = 0.02 (OFV 7887.7): the analytic `R` is indefinite
//!   along ETA_KA itself (min eig −21.96), floored and reported "severe". Holding there
//!   first would hide that floor and move σ's SE by −98.6% (the audit's probe G), so the
//!   step must not hold.

use std::path::{Path, PathBuf};

use ferx_core::io::output::parameter_table;
use ferx_core::parser::model_parser::parse_model_file;
use ferx_core::types::{FitOptions, FitResult, HeldReason, ParameterKind};
use ferx_core::{run_covariance, run_model_with_data};

const DATA: &str = "tests/data/guard_hold/sim_oral_no_ka_iiv.csv";
/// The rail: ω = exp(−12), the log-Cholesky diagonal's lower guard −6, written with 17
/// significant digits so it packs back onto −6 exactly.
const RAIL: &str = "6.1442123533282098e-6";
/// The converged σ of the fixture (SD scale).
const SIGMA_HAT: &str = "0.10069194732313741";
/// ETA_KA's packed row: `[TVCL, TVV, TVKA, ETA_CL, ETA_V, ETA_KA, PROP_ERR, …]`.
const ETA_KA_ROW: usize = 5;

fn model_source(sigma: &str, kappa: bool, analytic_cov: bool) -> String {
    let (kappa_decl, ka, iov) = if kappa {
        (
            format!("  kappa K_KA ~ {RAIL}\n"),
            "KA = TVKA * exp(ETA_KA + K_KA)",
            "  iov_column = OCC\n",
        )
    } else {
        (String::new(), "KA = TVKA * exp(ETA_KA)", "")
    };
    format!(
        "[parameters]
  theta TVCL(0.12250099819205257, 0.001, 10.0)
  theta TVV(7.753670975867449, 0.1, 500.0)
  theta TVKA(0.9982755334099482, 0.01, 50.0)
  omega ETA_CL ~ 0.0792688562315328
  omega ETA_V  ~ 0.05944155243978235
  omega ETA_KA ~ {RAIL}
{kappa_decl}  sigma PROP_ERR ~ {sigma} (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  {ka}

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = true
  analytic_cov_hessian = {analytic_cov}
{iov}"
    )
}

/// Fit `src` (an evaluation) and return the fit, the model path and the population, for
/// the FIX-clone `run_covariance` comparisons.
fn evaluate(
    name: &str,
    src: &str,
) -> (
    FitResult,
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join(format!("{name}.ferx"));
    std::fs::write(&path, src).unwrap();
    let (fit, pop) = run_model_with_data(path.to_str().unwrap(), Some(DATA)).expect("evaluate");
    let model = parse_model_file(Path::new(&path)).expect("parse");
    (fit, model, pop, dir)
}

fn all_ses(r: &FitResult) -> Vec<f64> {
    let mut v = r.se_theta.clone().unwrap_or_default();
    v.extend(r.se_omega.clone().unwrap_or_default());
    v.extend(r.se_sigma.clone().unwrap_or_default());
    v.extend(r.se_kappa.clone().unwrap_or_default());
    v
}

/// The same fit with every collapse FIX, through `run_covariance`: the matrix a NONMEM
/// `$OMEGA … FIX` run computes, from the same engine. It pins *conditional versus
/// marginal*, not `R` itself — the analytic route builds both from the same `R` entries;
/// the NONMEM anchor (`nonmem_anchor/guard_hold_*`) is the external reference.
fn fix_clone(
    fit: &FitResult,
    model: &ferx_core::types::CompiledModel,
    pop: &ferx_core::types::Population,
    analytic: bool,
) -> FitResult {
    let mut held = fit.clone();
    for h in &fit.held_at_bound {
        match h.kind {
            ParameterKind::Omega => held.omega_fixed[h.index] = true,
            ParameterKind::Kappa => held.kappa_fixed[h.index] = true,
            ParameterKind::Sigma => held.sigma_fixed[h.index] = true,
            ParameterKind::Theta => held.theta_fixed[h.index] = true,
        }
    }
    let options = FitOptions {
        analytic_cov_hessian: analytic,
        ..FitOptions::default()
    };
    run_covariance(&held, Some(model), Some(pop), &options).expect("FIX clone")
}

fn regularized(r: &FitResult) -> bool {
    r.warnings
        .iter()
        .any(|w| w.starts_with("Covariance step regularized"))
}

fn guard_warning(r: &FitResult) -> &str {
    r.warnings
        .iter()
        .find(|w| w.starts_with("Internal optimizer parameter guard reached"))
        .expect("the guard warning")
}

/// Tests 5 and T-6b in one, so the pair cannot silently land on one side: at the
/// stationary point ETA_KA is held, at the poor-σ point it is listed but not held, and the
/// straddle itself (no floor / a floor) is asserted.
///
/// Mutations, each named by the assertion that dies: never hold → `cov_fixed` on the
/// stationary point; hold before the verdict → the poor-σ floor and σ-SE ratio; drop the
/// list fill → `held_at_bound`.
#[test]
fn a_collapse_is_held_at_a_stationary_point_and_only_listed_at_a_poor_one() {
    // ── Stationary: held ──────────────────────────────────────────────────────────────
    let (f, model, pop, _d) = evaluate("stationary", &model_source(SIGMA_HAT, false, true));
    assert!(f.covariance_matrix.is_some(), "{:?}", f.warnings);
    assert!(
        !regularized(&f),
        "the stationary point has a clean R: {:?}",
        f.warnings
    );
    let held: Vec<(&str, ParameterKind, usize, usize, HeldReason)> = f
        .held_at_bound
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index, h.reason))
        .collect();
    assert_eq!(
        held,
        [(
            "ETA_KA",
            ParameterKind::Omega,
            2,
            ETA_KA_ROW,
            HeldReason::CollapsedAtGuard
        )]
    );
    assert!(
        f.cov_fixed[ETA_KA_ROW],
        "the step held ETA_KA: {:?}",
        f.cov_fixed
    );
    assert_eq!(f.cov_fixed.iter().filter(|&&b| b).count(), 1);
    let cov = f.covariance_matrix.as_ref().unwrap();
    for k in 0..cov.nrows() {
        assert_eq!(cov[(ETA_KA_ROW, k)], 0.0);
    }
    assert_eq!(f.se_omega.as_ref().unwrap()[2], 0.0);
    for (i, se) in all_ses(&f).iter().enumerate() {
        assert!(se.is_finite(), "SE {i} = {se}");
        if i != ETA_KA_ROW {
            assert!(*se > 0.0, "SE {i} = {se}");
        }
    }
    assert!(
        guard_warning(&f).ends_with(
            "Their standard errors are not reported; the other standard errors are computed \
             with them held at the floor."
        ),
        "{}",
        guard_warning(&f)
    );
    let row = parameter_table(&f)
        .lines()
        .find(|l| l.starts_with("ETA_KA "))
        .unwrap()
        .to_string();
    assert_eq!(
        row.split_whitespace().collect::<Vec<_>>()[2..4],
        ["N/A", "N/A"],
        "{row}"
    );

    // The held SEs are exactly the FIX clone's on the analytic R: a block of the analytic
    // R along the free axes is the full R's sub-block to the bit (`analytic_cov_axes`), and
    // both invert it the same way. Measured: 0 ULP on every SE (macOS arm64).
    let clone = fix_clone(&f, &model, &pop, true);
    assert_eq!(
        all_ses(&clone),
        all_ses(&f),
        "analytic: held SEs = FIX clone SEs"
    );
    assert_eq!(clone.covariance_matrix, f.covariance_matrix);

    // ── Poor σ: listed, not held ──────────────────────────────────────────────────────
    let (g, model, pop, _d) = evaluate("poor_sigma", &model_source("0.02", false, true));
    assert!(g.covariance_matrix.is_some(), "{:?}", g.warnings);
    assert!(
        regularized(&g),
        "the straddle: poor σ floors R: {:?}",
        g.warnings
    );
    let floor = g
        .warnings
        .iter()
        .find(|w| w.starts_with("Covariance step regularized"))
        .unwrap();
    assert!(floor.contains("severity: severe"), "{floor}");
    assert_eq!(g.held_at_bound.len(), 1);
    assert_eq!(g.held_at_bound[0].name, "ETA_KA");
    assert!(
        !g.cov_fixed[ETA_KA_ROW],
        "a floored R is not held: {:?}",
        g.cov_fixed
    );
    // The unconditional SE is kept in the field; only the display says N/A.
    assert!(g.se_omega.as_ref().unwrap()[2] > 0.0);
    let row = parameter_table(&g)
        .lines()
        .find(|l| l.starts_with("ETA_KA "))
        .unwrap()
        .to_string();
    assert_eq!(
        row.split_whitespace().collect::<Vec<_>>()[2..4],
        ["N/A", "N/A"],
        "{row}"
    );
    assert!(
        guard_warning(&g).ends_with(" Their standard errors are not reported."),
        "{}",
        guard_warning(&g)
    );
    // Holding before the verdict is what this pins against: the FIX clone at the poor
    // point has no floor and a σ SE about 70× smaller (0.00015 against 0.0105, measured);
    // the reported SE must stay the unconditional one.
    let clone = fix_clone(&g, &model, &pop, true);
    assert!(!regularized(&clone), "{:?}", clone.warnings);
    let (sg, sc) = (
        g.se_sigma.as_ref().unwrap()[0],
        clone.se_sigma.as_ref().unwrap()[0],
    );
    assert!(
        sg.is_finite() && sc.is_finite() && sg > 10.0 * sc,
        "σ SE {sg} vs held {sc}"
    );
}

/// Test 6, FD side: on the FD stencil the step holds too, with the other SEs equal to the
/// FD FIX clone's. Measured: 0 ULP on every SE (macOS arm64) — each stencil entry is the
/// same objective evaluations whichever free set the step runs over — so the bound is
/// equality. The route itself is asserted, so a fixture that silently ran the same engine
/// twice fails by name: the stencil's SEs differ from the analytic R's (by up to 1.8e-4
/// relative, measured).
#[test]
fn a_collapse_is_held_on_the_fd_stencil_route_too() {
    let (f, model, pop, _d) = evaluate("stationary_fd", &model_source(SIGMA_HAT, false, false));
    assert!(!regularized(&f), "{:?}", f.warnings);
    assert!(f.cov_fixed[ETA_KA_ROW], "{:?}", f.cov_fixed);
    assert_eq!(f.se_omega.as_ref().unwrap()[2], 0.0);
    let clone = fix_clone(&f, &model, &pop, false);
    assert_eq!(all_ses(&clone), all_ses(&f), "FD: held SEs = FIX clone SEs");
    let (analytic, _, _, _d) = evaluate("stationary_an", &model_source(SIGMA_HAT, false, true));
    assert_ne!(
        all_ses(&analytic),
        all_ses(&f),
        "the two fits must run different routes"
    );
}

/// Test 7: an IOV κ collapsed with the ETA — both held, two entries in packed order.
#[test]
fn a_collapsed_kappa_is_held_with_the_eta() {
    let (f, model, pop, _d) = evaluate("kappa", &model_source(SIGMA_HAT, true, true));
    assert!(!regularized(&f), "{:?}", f.warnings);
    let held: Vec<(&str, ParameterKind, usize, usize)> = f
        .held_at_bound
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index))
        .collect();
    // Packed: [θ×3, Ω×3, σ, κ].
    assert_eq!(
        held,
        [
            ("ETA_KA", ParameterKind::Omega, 2, ETA_KA_ROW),
            ("K_KA", ParameterKind::Kappa, 0, 7),
        ]
    );
    assert!(
        f.cov_fixed[ETA_KA_ROW] && f.cov_fixed[7],
        "{:?}",
        f.cov_fixed
    );
    assert_eq!(f.se_kappa.as_ref().unwrap()[0], 0.0);
    let clone = fix_clone(&f, &model, &pop, true);
    assert_eq!(all_ses(&clone), all_ses(&f));
}

/// Test 10 and D7: with nothing on the rail the list is empty and nothing is marked; and a
/// `run_covariance` re-run on the held fit rebuilds the list, the mask and the warning from
/// its own matrix — here the same ones, bit for bit.
#[test]
fn no_collapse_lists_nothing_and_a_re_run_rebuilds_the_report() {
    let src = model_source(SIGMA_HAT, false, true)
        .replace(&format!("omega ETA_KA ~ {RAIL}"), "omega ETA_KA ~ 0.1");
    let (f, _, _, _d) = evaluate("interior", &src);
    assert!(f.held_at_bound.is_empty());
    assert!(!f.cov_fixed.iter().any(|&b| b), "{:?}", f.cov_fixed);
    assert!(!f.warnings.iter().any(|w| w.contains("parameter guard")));

    // The re-run rebuilds the report from its own step, not from the incoming fit — which
    // `run_covariance` clones, so an incoming fit that already carried the right report
    // could not tell (#2812 review r1 #2). The incoming fit here is stale on every count:
    // no list, the no-matrix wording of the guard warning, no mask, and ETA_KA's
    // unconditional SE. Mutation: drop `refresh_held_at_bound_report` → every assertion
    // below fails.
    let (f, model, pop, _d) = evaluate("rerun", &model_source(SIGMA_HAT, false, true));
    let stale_text = guard_warning(&f)
        .split(" Their standard errors")
        .next()
        .unwrap()
        .to_string();
    let mut stale = f.clone();
    stale.held_at_bound.clear();
    stale.cov_fixed.clear();
    stale.se_omega.as_mut().unwrap()[2] = 1.2e-4;
    for w in stale.warnings.iter_mut() {
        if w.starts_with("Internal optimizer parameter guard") {
            *w = stale_text.clone();
        }
    }
    let options = FitOptions::default();
    let again = run_covariance(&stale, Some(&model), Some(&pop), &options).expect("re-run");
    assert_eq!(again.held_at_bound, f.held_at_bound);
    assert_eq!(again.cov_fixed, f.cov_fixed);
    assert!(again.cov_fixed[ETA_KA_ROW]);
    assert_eq!(all_ses(&again), all_ses(&f));
    assert_eq!(again.se_omega.as_ref().unwrap()[2], 0.0);
    assert_eq!(guard_warning(&again), guard_warning(&f));
    assert_ne!(guard_warning(&again), stale_text);
    assert_eq!(
        again
            .warnings
            .iter()
            .filter(|w| w.starts_with("Internal optimizer parameter guard"))
            .count(),
        1,
        "the stale copy is dropped"
    );
}

/// #2812 review r1 #3: under `covariance_method = s` the step holds the collapse on `S⁻¹`
/// (once `S` itself inverts cleanly), equal to the same model with ETA_KA FIX. Measured:
/// bit-equal on every SE (macOS arm64). Mutation: drop the `condition_on_held` call on
/// the `s` arm → the zero row and the equality fail.
#[test]
fn a_collapse_is_held_under_covariance_method_s() {
    let held_src = model_source(SIGMA_HAT, false, true).replace(
        "covariance = true",
        "covariance = true\n  covariance_method = s",
    );
    let (h, _, _, _d) = evaluate("held_s", &held_src);
    let fix_src = held_src.replace(
        &format!("omega ETA_KA ~ {RAIL}\n"),
        &format!("omega ETA_KA ~ {RAIL} FIX\n"),
    );
    assert_ne!(fix_src, held_src);
    let (f, _, _, _d) = evaluate("fix_s", &fix_src);
    assert_eq!(h.covariance_method, f.covariance_method);
    let cov = h.covariance_matrix.as_ref().expect("an S⁻¹");
    for k in 0..cov.nrows() {
        assert_eq!(cov[(ETA_KA_ROW, k)], 0.0, "row {k}");
    }
    assert!(h.cov_fixed[ETA_KA_ROW], "{:?}", h.cov_fixed);
    assert_eq!(h.held_at_bound.len(), 1);
    assert!(f.held_at_bound.is_empty(), "a FIX parameter is not listed");
    assert_eq!(all_ses(&h), all_ses(&f));
    assert_eq!(h.covariance_matrix, f.covariance_matrix);
}

/// #2812 review r1 #1: on a block Ω, `se_omega` is the whole lower triangle, so the held
/// variance's SE is zeroed at `chol_lt_idx(i, i, n)`, not at the ETA index. ETA_KA is the
/// **third** ETA of a 3×3 block, so the two layouts disagree: index 2 is
/// `ETA_KA × ETA_CL`'s slot, and Ω_KA,KA's is 5. The reviewer's fixture: Ω_KA,KA set so
/// that `L_22` packs onto the rail. Before the fix the table printed
/// `ETA_KA × ETA_CL … SE 0.000000` and Ω_KA,KA kept 3.16e-4.
#[test]
fn a_held_variance_in_a_block_omega_zeroes_its_own_se_slot() {
    let src = model_source(SIGMA_HAT, false, true)
        .replace("  omega ETA_CL ~ 0.0792688562315328\n", "")
        .replace("  omega ETA_V  ~ 0.05944155243978235\n", "")
        .replace(
            &format!("  omega ETA_KA ~ {RAIL}\n"),
            "  block_omega (ETA_CL, ETA_V, ETA_KA) = [0.0792688562315328, 0.01, \
             0.05944155243978235, 0.002, 0.001, 6.621429974700735e-05]\n",
        );
    let (f, _, _, _d) = evaluate("block3", &src);
    // Packed: [θ×3, L00, L10, L20, L11, L21, L22, σ] — L22 is row 8.
    let held: Vec<(&str, ParameterKind, usize, usize)> = f
        .held_at_bound
        .iter()
        .map(|h| (h.name.as_str(), h.kind, h.index, h.packed_index))
        .collect();
    assert_eq!(held, [("ETA_KA", ParameterKind::Omega, 2, 8)]);
    assert!(f.cov_fixed[8], "the step held L22: {:?}", f.warnings);
    let se = f.se_omega.as_ref().unwrap();
    assert_eq!(se.len(), 6, "a block Ω's se_omega is its lower triangle");
    assert_eq!(se[5], 0.0, "Ω_KA,KA's own slot is zeroed: {se:?}");
    for (k, v) in se.iter().enumerate().filter(|&(k, _)| k != 5) {
        assert!(v.is_finite() && *v > 0.0, "slot {k} keeps its SE: {se:?}");
    }
    let table = parameter_table(&f);
    let cells = |name: &str| {
        let l = table
            .lines()
            .find(|l| l.starts_with(name))
            .unwrap_or_else(|| panic!("{name}: {table}"))
            .to_string();
        let w: Vec<String> = l.split_whitespace().map(str::to_string).collect();
        w[w.len() - 4..w.len() - 2].join(" ")
    };
    assert_eq!(cells("ETA_KA "), "N/A N/A", "{table}");
    assert_ne!(
        cells("ETA_KA × ETA_CL").split(' ').next().unwrap(),
        "0.000000",
        "{table}"
    );
}

// ── #2809: a collapsed `[mixture]` per-class override ───────────────────────────────────
//
// `tests/data/guard_hold/mix_override_collapse.{py,csv}`: a two-class 1-cpt IV population
// whose class 2 has **no** CL variability, fitted with `omega(2) ETA_CL` — the override
// collapses. The point is the full fit's estimate with the override on its rail (OFV
// −37.391597, macOS arm64; the plan's M2). Packed: `[TVCL1, TVCL2, TVV, MIXL, ETA_CL, EPS,
// ETA_CL_MIX2]`.

const MIX_DATA: &str = "tests/data/guard_hold/mix_override_collapse.csv";
/// ETA_CL_MIX2's packed row.
const OVERRIDE_ROW: usize = 6;

fn mixture_source(override_fixed: bool, analytic_cov: bool) -> String {
    let fix = if override_fixed { " FIX" } else { "" };
    format!(
        "[parameters]
  theta TVCL1(1.2921355862556707, 0.01, 100.0)
  theta TVCL2(2.526598380988927, 0.01, 100.0)
  theta TVV(10.040303488207734, 0.1, 1000.0)
  theta MIXL(-0.11953563542469542, -10.0, 10.0)
  omega ETA_CL ~ 0.06227533113210565
  sigma EPS ~ 0.10298957033853946 (sd)

[mixture]
  nsub = 2
  logit(1) = MIXL
  omega(2) ETA_CL ~ {RAIL}{fix}

[individual_parameters]
  CL = if (MIXNUM == 1) TVCL1 * exp(ETA_CL) else TVCL2 * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(EPS)

[fit_options]
  method     = focei
  maxiter    = 0
  covariance = true
  analytic_cov_hessian = {analytic_cov}
"
    )
}

fn evaluate_mixture(name: &str, src: &str) -> (FitResult, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join(format!("{name}.ferx"));
    std::fs::write(&path, src).unwrap();
    let (fit, _) = run_model_with_data(path.to_str().unwrap(), Some(MIX_DATA)).expect("evaluate");
    (fit, dir)
}

/// The ETA_CL row of `parameter_table`: its estimate, SE and RSE cells.
fn base_eta_cl_cells(r: &FitResult) -> Vec<String> {
    let table = parameter_table(r);
    let row = table
        .lines()
        .find(|l| l.starts_with("ETA_CL "))
        .unwrap_or_else(|| panic!("{table}"));
    row.split_whitespace()
        .skip(1)
        .take(3)
        .map(str::to_string)
        .collect()
}

/// T7/T8's shared assertions for one route: the override is held and listed with its
/// class, the step marks it, every SE is bit-equal to the FIX twin fitted from the FIX
/// model file (`FitResult` has no override-FIX field, so `fix_clone` cannot build it), and
/// the base ETA_CL keeps a positive SE that the table prints as a number.
///
/// Mutations, each named by the assertion that dies: restore the D3 filter in
/// `held_guard_coordinates` → the list and `cov_fixed[6]`, and the SEs (the step no longer
/// conditions); drop the override mapping in `packed_diagonal_parameter` → the list; drop
/// the `class` skip in `apply_held_at_bound` → base `se_omega[0]` is 0.0, not the twin's;
/// drop the `class` filter in `se_held` → the table's ETA_CL row prints `N/A`.
fn assert_override_held_like_its_fix_twin(analytic: bool) -> FitResult {
    let tag = if analytic { "analytic" } else { "fd" };
    let (f, _d) = evaluate_mixture(&format!("held_{tag}"), &mixture_source(false, analytic));
    assert!(f.covariance_matrix.is_some(), "{tag}: {:?}", f.warnings);
    assert!(!regularized(&f), "{tag}: a clean R: {:?}", f.warnings);
    let held: Vec<_> = f
        .held_at_bound
        .iter()
        .map(|h| {
            (
                h.name.as_str(),
                h.kind,
                h.index,
                h.packed_index,
                h.reason,
                h.class,
            )
        })
        .collect();
    assert_eq!(
        held,
        [(
            "ETA_CL_MIX2",
            ParameterKind::Omega,
            0,
            OVERRIDE_ROW,
            HeldReason::CollapsedAtGuard,
            Some(2)
        )],
        "{tag}"
    );
    assert!(f.cov_fixed[OVERRIDE_ROW], "{tag}: {:?}", f.cov_fixed);
    assert_eq!(f.cov_fixed.iter().filter(|&&b| b).count(), 1, "{tag}");
    let w = guard_warning(&f);
    assert!(w.contains("ETA_CL_MIX2"), "{w}");
    assert!(
        w.ends_with(
            "Their standard errors are not reported; the other standard errors are computed \
             with them held at the floor."
        ),
        "{w}"
    );

    let (twin, _d) = evaluate_mixture(&format!("fix_{tag}"), &mixture_source(true, analytic));
    assert!(twin.held_at_bound.is_empty(), "{:?}", twin.held_at_bound);
    assert!(twin.cov_fixed[OVERRIDE_ROW], "{:?}", twin.cov_fixed);
    assert!(
        !twin.warnings.iter().any(|w| w.contains("parameter guard")),
        "a FIX override raises no guard warning: {:?}",
        twin.warnings
    );
    assert_eq!(
        all_ses(&f),
        all_ses(&twin),
        "{tag}: held SEs = FIX twin SEs"
    );
    assert_eq!(f.covariance_matrix, twin.covariance_matrix, "{tag}");

    let se = f.se_omega.as_ref().unwrap()[0];
    assert!(se.is_finite() && se > 0.0, "{tag}: base ETA_CL SE {se}");
    let cells = base_eta_cl_cells(&f);
    assert!(!cells.contains(&"N/A".to_string()), "{tag}: {cells:?}");
    assert_eq!(cells, base_eta_cl_cells(&twin), "{tag}");
    f
}

/// T7 (#2809), analytic R. Base `se_omega[0]` = 0.019272550413150978 on macOS arm64 and
/// 1.927255e-2 on Linux aarch64 (`tools/linux-test.sh`, every printed digit equal); every SE
/// bit-equal to the FIX twin, 0 ULP, on both platforms (the plan's M3/M4).
#[test]
fn a_collapsed_mixture_override_is_held_like_its_fix_twin_on_the_analytic_r() {
    assert_override_held_like_its_fix_twin(true);
}

/// T8 (#2809), FD stencil. The route is asserted, so a fixture that silently ran the same
/// engine twice fails by name: the stencil's base ETA_CL SE is 0.01926268 against the
/// analytic R's 0.01927255 (macOS arm64).
#[test]
fn a_collapsed_mixture_override_is_held_like_its_fix_twin_on_the_fd_stencil() {
    let fd = assert_override_held_like_its_fix_twin(false);
    let (an, _d) = evaluate_mixture("route_analytic", &mixture_source(false, true));
    assert_ne!(
        all_ses(&an),
        all_ses(&fd),
        "the two fits must run different routes"
    );
}

/// T9 (#2809): a `run_covariance` re-run on a mixture fit rebuilds the override's entry
/// and its `cov_fixed` mark from its own step. The incoming fit is stale (no list, no
/// mask), so an incoming copy cannot satisfy it. Mutation: `refresh_held_at_bound_report`
/// skips `apply_held_at_bound` → the list is empty.
#[test]
fn a_re_run_rebuilds_the_held_override_entry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rerun.ferx");
    std::fs::write(&path, mixture_source(false, true)).unwrap();
    let (f, pop) = run_model_with_data(path.to_str().unwrap(), Some(MIX_DATA)).expect("fit");
    let model = parse_model_file(&path).expect("parse");
    assert_eq!(f.held_at_bound.len(), 1);
    let mut stale = f.clone();
    stale.held_at_bound.clear();
    stale.cov_fixed.clear();
    let again =
        run_covariance(&stale, Some(&model), Some(&pop), &FitOptions::default()).expect("re-run");
    assert_eq!(again.held_at_bound, f.held_at_bound);
    assert_eq!(again.held_at_bound[0].class, Some(2));
    assert_eq!(again.cov_fixed, f.cov_fixed);
    assert!(again.cov_fixed[OVERRIDE_ROW]);
    assert_eq!(all_ses(&again), all_ses(&f));
}
