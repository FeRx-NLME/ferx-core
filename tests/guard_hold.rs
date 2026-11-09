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

// ── #2816: SIR holds a step-held collapse as FIX ─────────────────────────────────────

/// The fit's `SIR:` warnings, in order.
fn sir_warnings(r: &FitResult) -> Vec<&str> {
    r.warnings
        .iter()
        .filter(|w| w.starts_with("SIR"))
        .map(String::as_str)
        .collect()
}

/// `sir_ess` and `sir_ci_{theta,omega,sigma}` of `got` bit-for-bit those of `want`.
fn assert_same_sir(got: &FitResult, want: &FitResult, how: &str) {
    let ess = |r: &FitResult| r.sir_ess.map(f64::to_bits);
    assert!(want.sir_ess.is_some(), "{how}: no SIR on the reference");
    assert_eq!(
        ess(got),
        ess(want),
        "{how}: ESS {:?} vs {:?}",
        got.sir_ess,
        want.sir_ess
    );
    assert_eq!(got.sir_ci_theta, want.sir_ci_theta, "{how}: θ CIs");
    assert_eq!(got.sir_ci_omega, want.sir_ci_omega, "{how}: Ω CIs");
    assert_eq!(got.sir_ci_sigma, want.sir_ci_sigma, "{how}: σ CIs");
}

/// SIR on the held fit at the stationary point against the same fit with ETA_KA `FIX`,
/// under `method` (`""` for `r`, or a `covariance_method` line). Returns the held fit,
/// its model, population, source and temp dir for the `run_sir` replays.
fn held_sir_is_fix_sir(
    method: &str,
    how: &str,
) -> (
    FitResult,
    ferx_core::types::CompiledModel,
    ferx_core::types::Population,
    String,
    tempfile::TempDir,
) {
    let sir = format!("{method}  sir = true\n  sir_seed = 7\n");
    let held_src = format!("{}{sir}", model_source(SIGMA_HAT, false, true));
    let fix_src = held_src.replace(
        &format!("omega ETA_KA ~ {RAIL}\n"),
        &format!("omega ETA_KA ~ {RAIL} FIX\n"),
    );
    assert_ne!(fix_src, held_src);
    let (h, model, pop, dir) = evaluate("held_sir", &held_src);
    let (f, _, _, _d) = evaluate("fix_sir", &fix_src);
    // Precondition: the step held ETA_KA, so the two proposals are one matrix.
    assert!(h.cov_fixed[ETA_KA_ROW], "{how}: {:?}", h.warnings);
    assert_eq!(h.held_at_bound.len(), 1, "{how}");
    assert_eq!(h.covariance_method, f.covariance_method, "{how}");
    assert_eq!(
        h.covariance_matrix, f.covariance_matrix,
        "{how}: proposals differ"
    );
    assert_same_sir(&h, &f, how);
    assert_eq!(sir_warnings(&h), sir_warnings(&f), "{how}");
    assert!(
        !sir_warnings(&h).iter().any(|w| w.contains("ETA_KA")),
        "{how}: a SIR sentence names the held coordinate: {:?}",
        sir_warnings(&h)
    );
    (h, model, pop, held_src, dir)
}

/// #2816 T1: at the stationary point the covariance step holds ETA_KA (#2797), so SIR on
/// the analytic R-matrix proposal is SIR on the same fit with ETA_KA `FIX` — the same ESS
/// bits, the same CIs and the same `SIR:` sentences. `run_sir` on the held fit, in memory
/// and after a `.fitrx` reload, reproduces the in-fit SIR: the gate reads the reloaded
/// matrix and the guard walk the reloaded estimates, so a reload that rounded ETA_KA off
/// the rail would redden it.
///
/// Mutation: delete the SIR hold in `run_sir_in_box` → ESS 129.71 against 747.85, and a
/// rank-deficiency sentence naming ETA_KA appears (macOS arm64).
#[test]
fn sir_holds_a_step_held_collapse_as_fix_under_r() {
    let (h, model, pop, held_src, dir) = held_sir_is_fix_sir("", "r");
    let opts = FitOptions {
        sir: true,
        sir_seed: Some(7),
        ..FitOptions::default()
    };
    let mem = ferx_core::run_sir(&h, Some(&model), Some(&pop), &opts).expect("run_sir");
    assert_same_sir(&mem, &h, "run_sir in memory");
    let path = dir.path().join("held_sir.fitrx");
    ferx_core::io::fitrx::save_fit(&h, &pop, &held_src, &path, Default::default()).expect("saves");
    let loaded = ferx_core::io::fitrx::load_fit(&path).expect("loads");
    assert_eq!(
        loaded.fit.held_at_bound.len(),
        1,
        "the hold survives the reload"
    );
    let re = ferx_core::run_sir(&loaded.fit, Some(&model), Some(&pop), &opts)
        .expect("run_sir after reload");
    assert_same_sir(&re, &h, "run_sir after .fitrx reload");
}

/// #2816 T2: T1 on the unprofiled `S` proposal (`covariance_method = s`). Its low ESS on
/// the unheld proposal also emits the "heaviest draw … moves ETA_KA" sentence, which the
/// equality with the FIX twin's sentences rules out.
///
/// Mutation: delete the SIR hold → ESS 43.15 against 432.47 (macOS arm64).
#[test]
fn sir_holds_a_step_held_collapse_as_fix_under_s() {
    held_sir_is_fix_sir("  covariance_method = s\n", "s");
}

/// #2816 T3, the straddle: at the poor-σ point the analytic `R` is floored along ETA_KA, so
/// the step does not hold it (`cov_fixed` false) and SIR must keep sampling it. There SIR
/// shrinks the proposal along ETA_KA and says so, naming it — a property of the proposal,
/// not of the draws, so the ESS of ≈1 at this point cannot flake it.
///
/// Mutation: drop the `step_held` gate (hold every guard coordinate) → ETA_KA leaves the
/// free block and the shrink note disappears (measured, macOS arm64).
#[test]
fn sir_keeps_sampling_a_collapse_the_step_did_not_hold() {
    let src = format!(
        "{}  sir = true\n  sir_seed = 7\n",
        model_source("0.02", false, true)
    );
    let (f, _, _, _d) = evaluate("poor_sir", &src);
    assert!(
        !f.cov_fixed[ETA_KA_ROW],
        "the step held it: {:?}",
        f.warnings
    );
    assert_eq!(f.held_at_bound.len(), 1, "listed, not held");
    assert!(
        sir_warnings(&f)
            .iter()
            .any(|w| w.starts_with("SIR: proposal was shrunk") && w.contains("[ETA_KA")),
        "{:?}",
        sir_warnings(&f)
    );
}

/// #2830 review r1 #1: a proposal whose shape does not match the packed vector is refused
/// with `E_FIT_INPUT_MODEL`, also when the parameters carry a guard collapse. The hold reads
/// the proposal's diagonal at the collapse's packed index, so it must run after the shape
/// check: a 7×5 proposal (ETA_KA is packed row 5, columns 0–4 only) otherwise indexes
/// `(5, 5)` and panics instead of returning the refusal.
///
/// Mutation: move the hold loop back above the dimension check → panic (index out of
/// bounds), not `Err`.
#[test]
fn run_sir_core_refuses_a_misshapen_proposal_with_a_collapse() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shape.ferx");
    std::fs::write(&path, model_source(SIGMA_HAT, false, true)).unwrap();
    let p = ferx_core::prepare_run(path.to_str().unwrap(), Some(DATA)).expect("prepare");
    let params = &p.init_params;
    // Precondition: the inits carry the collapse, so the hold loop has a coordinate to read.
    assert_eq!(
        params.omega.matrix[(2, 2)].to_bits(),
        RAIL.parse::<f64>().unwrap().to_bits()
    );
    let etas = vec![nalgebra::DVector::zeros(3); p.population.subjects.len()];
    let opts = FitOptions {
        sir: true,
        sir_seed: Some(7),
        ..FitOptions::default()
    };
    let proposal = nalgebra::DMatrix::<f64>::zeros(7, ETA_KA_ROW);
    let err = ferx_core::estimation::sir::run_sir_core(
        &p.parsed.model,
        &p.population,
        params,
        &etas,
        &proposal,
        0.0,
        &opts,
    )
    .err()
    .expect("a 7×5 proposal is refused");
    assert!(
        err.contains("Covariance matrix dimensions (7,5) don't match packed parameters (7)"),
        "{err}"
    );
}
