//! Tier-1 tests for #1613: `fit()` honours `FitOptions::gradient_method` per call.
//!
//! **The gap.** Every gradient decision read `model.gradient_method`, which only the file
//! entry points stamped from the options. `fit` takes `&CompiledModel` (not `Clone`), so it
//! could not stamp, and a direct `fit()` with `FitOptions { gradient_method: Fd, .. }` on a
//! parsed model ran on the analytic `Dual2` gradient — bit-identical to the run without the
//! setting, with no warning. Now `fit` arms the call's flag (`api::pool::FitScope`) and every
//! reader asks `GradientMethod::forced_fd`, a union with the model's own flag.
//!
//! Every case reads **which engine ran** — the `gradient_method_inner` / `_outer` labels,
//! which `fit_inner` computes from the same predicates the loops read — and not only the
//! objective. The R-matrix scope (G4) and the IOV FD reason (G5) are pinned at the reader in
//! `estimation/cov_diagnostics_tests.rs` and `estimation/inner_optimizer_iov_tests.rs`.

use super::*;
use crate::types::{GradientMethod, Optimizer};

const FD: &str = "finite differences";
const ANALYTIC: &str = "analytic (Dual2)";

/// `examples/warfarin.ferx` on `data/warfarin.csv`: closed-form, in analytic scope on both
/// loops, so `Auto` and `Fd` take different engines.
fn warfarin(stamp: GradientMethod) -> (CompiledModel, Population) {
    let mut parsed =
        crate::parser::model_parser::parse_full_model_file(Path::new("examples/warfarin.ferx"))
            .expect("parse warfarin");
    let (population, _) = read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        "data/warfarin.csv",
        None,
        None,
        None,
        &parsed.column_map,
    )
    .expect("read warfarin");
    parsed.model.gradient_method = stamp;
    (parsed.model, population)
}

/// A short FOCE run with the optimizer pinned: under `optimizer = auto` an FD outer gradient
/// resolves to BOBYQA, whose label is `N/A`, and that would hide the outer half of the fix.
fn opts(gradient: GradientMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::Foce,
        interaction: false,
        optimizer: Optimizer::NloptLbfgs,
        gradient_method: gradient,
        outer_maxiter: 2,
        run_covariance_step: false,
        verbose: false,
        threads: Some(1),
        ..Default::default()
    }
}

fn run(stamp: GradientMethod, gradient: GradientMethod) -> FitResult {
    let (model, population) = warfarin(stamp);
    fit(&model, &population, &model.default_params, &opts(gradient)).expect("fit")
}

/// **G1 + G2 — the direct `fit()` with `options = Fd` on an unstamped model runs FD on both
/// loops and lands on the stamped path's objective bit for bit; the `Auto` control in the same
/// test runs analytic and lands elsewhere.**
///
/// At `d43afca9` the G1 row was bit-identical to the G2 row (inner and outer analytic), which
/// is the reported defect. The straddle is asserted, so the bit-equality with the stamped run
/// cannot pass because `Fd` stopped mattering.
///
/// Mutations, each of which names its side: drop the `fd` member from `FitScope::of` → both
/// labels; revert `analytic_inner_common_bail` to
/// `model.gradient_method` → inner only; revert `analytic_outer_gradient_available` → outer
/// only. A predicate that always says FD fails the G2 half.
#[test]
fn a_direct_fit_with_gradient_fd_in_the_options_runs_fd_on_both_loops() {
    let direct = run(GradientMethod::Auto, GradientMethod::Fd);
    let stamped = run(GradientMethod::Fd, GradientMethod::Fd);
    let control = run(GradientMethod::Auto, GradientMethod::Auto);

    assert_eq!(
        direct.gradient_method_inner, FD,
        "inner: options.gradient_method = Fd must reach the EBE solve"
    );
    assert_eq!(
        direct.gradient_method_outer, FD,
        "outer: options.gradient_method = Fd must reach the outer gradient"
    );
    assert!(direct.ofv.is_finite() && stamped.ofv.is_finite() && control.ofv.is_finite());
    assert_eq!(
        direct.ofv.to_bits(),
        stamped.ofv.to_bits(),
        "direct fit() OFV {:.17e} vs stamped-model OFV {:.17e}",
        direct.ofv,
        stamped.ofv
    );

    // G2, the straddle.
    assert_eq!(control.gradient_method_inner, ANALYTIC, "control inner");
    assert_eq!(control.gradient_method_outer, ANALYTIC, "control outer");
    assert_ne!(
        control.ofv.to_bits(),
        direct.ofv.to_bits(),
        "Fd and Auto must reach different objectives on this fixture, or the equality above \
         holds whether the options are read or not"
    );
}

/// **G3 — the model's own `gradient = fd` still wins over `options = Auto`** (the union;
/// today's behaviour, and the one a caller who stamped by hand relies on).
///
/// Mutation: rewrite `forced_fd` as "the options win" (`scope.unwrap_or(model flag)`).
#[test]
fn a_stamped_model_runs_fd_under_auto_options() {
    let r = run(GradientMethod::Fd, GradientMethod::Auto);
    assert_eq!(r.gradient_method_inner, FD, "inner");
    assert_eq!(r.gradient_method_outer, FD, "outer");
}

/// **G6 — a pool's workers carry the `gradient = fd` of the call they serve, and pools are
/// not reused across a different flag.**
///
/// The labels cannot see this: `fit_inner` computes them on the fit's own thread, which
/// `FitScope::armed` arms whatever pool it runs on. What the pool decides is what the
/// *workers* read, i.e. the per-subject EBE solves the `par_iter` fans out — so this compares
/// objectives. A pinned `threads` leases from one cache keyed by the scope, and an `Auto` fit
/// leases the default-scope pool there; alternating `Auto` → `Fd` → `Auto` at one width leaves
/// each fit an idle pool of the *other* flag to (wrongly) reuse. The `Fd` objective is
/// compared with a stamped model's, whose flag rides on the model and so reaches the workers
/// regardless of the pool.
///
/// Mutation: drop `fd` from `FitScope::same_pool_key` → the `Fd` fit reuses the `Auto` pool
/// (workers analytic) or the second `Auto` fit reuses the `Fd` pool; drop it from
/// `install_on_worker` → the `Fd` fit's workers never see it.
#[test]
fn pool_workers_carry_the_calls_gradient_fd() {
    // An odd width other tests do not pin, so the idle pool this test leaves is the one its
    // next fit finds.
    const WIDTH: usize = 5;
    let pinned = |g| FitOptions {
        threads: Some(WIDTH),
        ..opts(g)
    };
    let fit_with = |stamp, g| {
        let (model, population) = warfarin(stamp);
        fit(&model, &population, &model.default_params, &pinned(g)).expect("fit")
    };
    let auto_before = fit_with(GradientMethod::Auto, GradientMethod::Auto);
    let direct_fd = fit_with(GradientMethod::Auto, GradientMethod::Fd);
    let auto_after = fit_with(GradientMethod::Auto, GradientMethod::Auto);
    let stamped_fd = fit_with(GradientMethod::Fd, GradientMethod::Fd);

    assert_eq!(direct_fd.gradient_method_inner, FD);
    assert_eq!(auto_before.gradient_method_inner, ANALYTIC);
    assert_ne!(
        auto_before.ofv.to_bits(),
        stamped_fd.ofv.to_bits(),
        "premise: Fd and Auto reach different objectives at this width"
    );
    assert_eq!(
        direct_fd.ofv.to_bits(),
        stamped_fd.ofv.to_bits(),
        "the Fd fit's workers: {:.17e} vs stamped {:.17e}",
        direct_fd.ofv,
        stamped_fd.ofv
    );
    assert_eq!(
        auto_after.ofv.to_bits(),
        auto_before.ofv.to_bits(),
        "an Auto fit after an Fd one: {:.17e} vs {:.17e}",
        auto_after.ofv,
        auto_before.ofv
    );
}

/// The post-hoc fixtures: a FOCEI fit (non-interaction FOCE with a prediction-dependent
/// residual declines the analytic R-matrix for its own reason, which would hide the
/// gradient clause) with a covariance matrix, so `run_sir` has a proposal.
fn posthoc_base() -> (CompiledModel, Population, FitResult) {
    let (model, population) = warfarin(GradientMethod::Auto);
    let o = FitOptions {
        run_covariance_step: true,
        ..posthoc_opts(GradientMethod::Auto)
    };
    let base = fit(&model, &population, &model.default_params, &o).expect("base fit");
    assert!(
        base.covariance_matrix.is_some(),
        "premise: the base fit has a covariance"
    );
    (model, population, base)
}

fn posthoc_opts(gradient: GradientMethod) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        interaction: true,
        ..opts(gradient)
    }
}

fn cov_bits(r: &FitResult) -> Vec<u64> {
    let m = r.covariance_matrix.as_ref().expect("covariance");
    assert!(
        m.iter().all(|v| v.is_finite()),
        "non-finite covariance entry"
    );
    m.iter().map(|v| v.to_bits()).collect()
}

/// **G4b — a post-hoc `run_covariance` honours the `gradient_method` in its own options**
/// (#1829 review r1 row 2). `Fd` options on an unstamped model must take the FD stencil — bit
/// for bit the matrix a stamped model gives under `Auto` options, whose flag rides on the
/// model — while the `Auto` control takes the analytic R-matrix and lands elsewhere.
///
/// Mutation: hand `with_fit_scope` a gradient-`Auto` copy of the options
/// (`run_covariance.rs`) → the `Fd` call equals the control, not the stamped run.
#[test]
fn a_posthoc_run_covariance_honours_the_options_gradient_fd() {
    let (model, population, base) = posthoc_base();
    let (stamped, _) = warfarin(GradientMethod::Fd);
    let cov = |m: &CompiledModel, g| {
        crate::run_covariance(&base, Some(m), Some(&population), &posthoc_opts(g))
            .expect("run_covariance")
    };
    let fd = cov_bits(&cov(&model, GradientMethod::Fd));
    let reference = cov_bits(&cov(&stamped, GradientMethod::Auto));
    let control = cov_bits(&cov(&model, GradientMethod::Auto));
    assert_ne!(
        control, reference,
        "premise: the analytic R-matrix and the FD stencil differ on this fit"
    );
    assert_eq!(
        fd, reference,
        "run_covariance with options Fd must take the FD stencil, as a stamped model does"
    );
}

/// **G4c — the same for `run_sir`**, whose importance weights re-solve every subject's EBE:
/// the inner η-gradient route is what the options' `Fd` changes there.
///
/// Mutation: hand `run_sir`'s scope a gradient-`Auto` copy of the options (`run_sir.rs`) →
/// the `Fd` call equals the control.
#[test]
fn a_posthoc_run_sir_honours_the_options_gradient_fd() {
    let (model, population, base) = posthoc_base();
    let (stamped, _) = warfarin(GradientMethod::Fd);
    let sir = |m: &CompiledModel, g| {
        let o = FitOptions {
            sir_samples: 40,
            sir_resamples: 20,
            sir_seed: Some(7),
            ..posthoc_opts(g)
        };
        let r = crate::run_sir(&base, Some(m), Some(&population), &o).expect("run_sir");
        let ess = r.sir_ess.expect("ess");
        assert!(ess.is_finite(), "ess {ess}");
        r.sir_ci_theta
            .expect("SIR CIs")
            .into_iter()
            .flat_map(|(lo, hi)| [lo.to_bits(), hi.to_bits()])
            .chain(std::iter::once(ess.to_bits()))
            .collect::<Vec<u64>>()
    };
    let fd = sir(&model, GradientMethod::Fd);
    let reference = sir(&stamped, GradientMethod::Auto);
    let control = sir(&model, GradientMethod::Auto);
    assert_ne!(
        control, reference,
        "premise: the analytic and FD inner routes weight the draws differently"
    );
    assert_eq!(
        fd, reference,
        "run_sir with options Fd must re-solve on FD, as a stamped model does"
    );
}

// ── #1835: the fit's `gradient = fd` is in its stage record ─────────────────────────────

/// `fit` with its stage record's `gradient_method` set to `g`: the straddle's other side,
/// the same estimates with only the recorded route changed.
fn with_recorded_gradient(fit: &FitResult, g: GradientMethod) -> FitResult {
    let mut f = fit.clone();
    f.scoring_settings
        .as_mut()
        .expect("premise: a fit carries its stage record")
        .gradient_method = g;
    f
}

/// A direct `fit()` with `options = Fd` on a parsed (unstamped) model, carrying a covariance
/// so `run_sir` has a proposal.
fn direct_fd_fit() -> (CompiledModel, Population, FitResult) {
    let (model, population) = warfarin(GradientMethod::Auto);
    let o = FitOptions {
        run_covariance_step: true,
        ..posthoc_opts(GradientMethod::Fd)
    };
    let f = fit(&model, &population, &model.default_params, &o).expect("fd fit");
    assert_eq!(f.gradient_method_inner, FD, "premise: the fit ran FD");
    assert!(
        f.covariance_matrix.is_some(),
        "premise: the fd fit has a covariance"
    );
    (model, population, f)
}

/// **G4e (#1835) — the `run_sir` twin of G4d.** The fit has no SIR record, so `run_sir`
/// resolves through the stage record; its draws re-solve every EBE on the recorded FD route.
///
/// Mutation: drop `recorded!(gradient_method)` → the default run re-solves analytically and
/// equals the `Auto`-record run, not the caller-`Fd` one.
#[test]
fn a_direct_fd_fits_posthoc_run_sir_with_default_options_stays_fd() {
    let (model, population, f) = direct_fd_fit();
    assert!(f.sir_settings.is_none(), "premise: no SIR record");
    let sir = |fit: &FitResult, g: GradientMethod| {
        let o = FitOptions {
            sir_samples: 40,
            sir_resamples: 20,
            sir_seed: Some(7),
            gradient_method: g,
            ..FitOptions::default()
        };
        let r = crate::run_sir(fit, Some(&model), Some(&population), &o).expect("run_sir");
        let ess = r.sir_ess.expect("ess");
        assert!(ess.is_finite(), "ess {ess}");
        r.sir_ci_theta
            .expect("SIR CIs")
            .into_iter()
            .flat_map(|(lo, hi)| [lo.to_bits(), hi.to_bits()])
            .chain(std::iter::once(ess.to_bits()))
            .collect::<Vec<u64>>()
    };
    let default = sir(&f, GradientMethod::Auto);
    let reference = sir(&f, GradientMethod::Fd);
    let analytic = sir(
        &with_recorded_gradient(&f, GradientMethod::Auto),
        GradientMethod::Auto,
    );
    assert_ne!(
        analytic, reference,
        "premise: the analytic and FD inner routes weight the draws differently"
    );
    assert_eq!(
        default, reference,
        "run_sir with default options must re-solve on the fit's FD route"
    );
}

/// **#1860 review r1 #1 — a post-hoc `run_sir` with the retired `gradient_method = Ad` records
/// `auto`, and its bundle loads.** Only `fit()` refuses `Ad` (`E_AD_RETIRED`); `run_sir` runs it
/// as the analytic route (`inner_optimizer.rs`, "AD (retired → analytic)"). Recording it
/// verbatim made `save_fit` succeed and `load_fit` fail with `sir.settings.gradient_method must
/// be auto or fd`, on a bundle that loaded before #1835.
///
/// Mutation: drop the `Ad → Auto` arm in `ScoringSettings::of_run` → the record says `Ad` and
/// `load_fit` is `Corrupt`.
#[test]
fn a_posthoc_run_sir_with_retired_ad_records_auto_and_its_bundle_loads() {
    let (model, population, base) = posthoc_base();
    let o = FitOptions {
        sir_samples: 20,
        sir_resamples: 10,
        sir_seed: Some(7),
        gradient_method: GradientMethod::Ad,
        ..FitOptions::default()
    };
    let r = crate::run_sir(&base, Some(&model), Some(&population), &o).expect("run_sir(Ad)");
    assert_eq!(
        r.sir_settings.as_ref().map(|s| s.scoring.gradient_method),
        Some(GradientMethod::Auto),
        "the run went analytic, and the record says so"
    );
    let dir = tempfile::tempdir().unwrap();
    let bundle = dir.path().join("sir_ad.fitrx");
    crate::io::fitrx::save_fit(&r, &population, "src\n", &bundle, Default::default())
        .expect("save_fit");
    let loaded = crate::io::fitrx::load_fit(&bundle)
        .unwrap_or_else(|e| panic!("a bundle from run_sir(Ad) must load: {e}"))
        .fit;
    assert_eq!(
        loaded.sir_settings.map(|s| s.scoring.gradient_method),
        Some(GradientMethod::Auto)
    );
}

// ── #1862: a pre-#1835 bundle of an FD fit warns that the post-hoc step goes analytic ───

/// `fit` (carrying both records) through a `.fitrx` bundle whose `fit.json` has
/// `gradient_method` removed from `scoring_settings` and `sir.settings` — the shape of a
/// bundle saved before #1835 — reloaded.
fn through_pre_1835_bundle(fit: &FitResult, population: &Population) -> FitResult {
    use std::io::{Read as _, Write as _};
    let dir = tempfile::tempdir().unwrap();
    let saved = dir.path().join("new.fitrx");
    crate::io::fitrx::save_fit(fit, population, "src\n", &saved, Default::default())
        .expect("save_fit");
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&saved).unwrap()).unwrap();
    let old = dir.path().join("pre_1835.fitrx");
    let mut zw = zip::ZipWriter::new(std::fs::File::create(&old).unwrap());
    for i in 0..archive.len() {
        let mut f = archive.by_index(i).unwrap();
        let name = f.name().unwrap().to_string();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        if name == "fit.json" {
            let mut v: serde_json::Value = serde_json::from_slice(&buf).unwrap();
            let removed = [
                v["scoring_settings"]
                    .as_object_mut()
                    .and_then(|o| o.remove("gradient_method")),
                v["sir"]["settings"]
                    .as_object_mut()
                    .and_then(|o| o.remove("gradient_method")),
            ];
            assert!(
                removed.iter().all(Option::is_some),
                "premise: the key is written in both blocks"
            );
            buf = serde_json::to_vec(&v).unwrap();
        }
        zw.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zw.write_all(&buf).unwrap();
    }
    zw.finish().unwrap();
    let loaded = crate::io::fitrx::load_fit(&old).expect("load_fit").fit;
    assert_eq!(
        loaded.scoring_settings.as_ref().map(|s| s.gradient_method),
        Some(GradientMethod::Auto),
        "premise: the stripped stage record loads auto"
    );
    assert_eq!(
        loaded
            .sir_settings
            .as_ref()
            .map(|s| s.scoring.gradient_method),
        Some(GradientMethod::Auto),
        "premise: the stripped SIR record loads auto"
    );
    loaded
}

/// `fit` with a SIR record (a short `run_sir` under the fit's own route), so the bundle
/// carries both blocks the field is removed from.
fn with_sir_record(fit: &FitResult, model: &CompiledModel, population: &Population) -> FitResult {
    let o = FitOptions {
        sir_samples: 20,
        sir_resamples: 10,
        sir_seed: Some(7),
        ..FitOptions::default()
    };
    let r = crate::run_sir(fit, Some(model), Some(population), &o).expect("run_sir");
    assert!(r.sir_settings.is_some(), "premise: the SIR record is set");
    r
}

fn warns(r: &FitResult) -> bool {
    let n = r
        .warnings
        .iter()
        .filter(|w| *w == crate::estimation::fit_inputs::UNRECORDED_FD_ROUTE_WARNING)
        .count();
    assert!(n <= 1, "the note is pushed once: {:?}", r.warnings);
    n == 1
}

/// **#1862 — an FD fit's pre-#1835 bundle: the post-hoc steps under default options warn
/// that they go analytic, and a caller's `gradient_method = fd` silences it.** Both sides of
/// the gate in one test, on the same loaded fit: the default call warns, the `Fd` call does
/// not. The `run_covariance` output is fed back into `run_covariance` with `Fd` too, so a
/// note carried over from the earlier step is taken off rather than kept.
///
/// Mutations: drop either `note_unrecorded_fd_route` call → that step's `warns` dies; drop
/// the `inner_reports_analytic_model` clause → the `Fd` calls warn; key on the outer label
/// alone (`gradient_method_inner` ignored) → the analytic control below warns.
#[test]
fn a_pre_1835_bundle_of_an_fd_fit_warns_that_posthoc_goes_analytic() {
    let (model, population, f) = direct_fd_fit();
    assert_ne!(
        f.gradient_method_outer, ANALYTIC,
        "premise: the outer ran FD"
    );
    let fd_record = with_sir_record(&f, &model, &population);
    assert!(
        !warns(&fd_record),
        "a fit that records gradient_method = fd is silent"
    );
    let loaded = through_pre_1835_bundle(&fd_record, &population);
    assert_eq!(
        loaded.gradient_method_inner, FD,
        "premise: the label survives"
    );

    let cov = |fit: &FitResult, g| {
        let o = FitOptions {
            gradient_method: g,
            ..FitOptions::default()
        };
        crate::run_covariance(fit, Some(&model), Some(&population), &o).expect("run_covariance")
    };
    let sir = |fit: &FitResult, g| {
        let o = FitOptions {
            sir_samples: 20,
            sir_resamples: 10,
            sir_seed: Some(7),
            gradient_method: g,
            ..FitOptions::default()
        };
        crate::run_sir(fit, Some(&model), Some(&population), &o).expect("run_sir")
    };
    let cov_default = cov(&loaded, GradientMethod::Auto);
    assert!(
        warns(&cov_default),
        "run_covariance: {:?}",
        cov_default.warnings
    );
    assert!(
        !warns(&cov(&loaded, GradientMethod::Fd)),
        "run_covariance with gradient_method = fd runs as the fit did"
    );
    assert!(
        !warns(&cov(&cov_default, GradientMethod::Fd)),
        "a carried-over note is taken off when it no longer applies"
    );
    assert!(
        warns(&cov(&cov_default, GradientMethod::Auto)),
        "and kept once"
    );
    assert!(warns(&sir(&loaded, GradientMethod::Auto)), "run_sir");
    assert!(
        !warns(&sir(&loaded, GradientMethod::Fd)),
        "run_sir with gradient_method = fd"
    );
}

/// **#1862 control — an analytic fit's pre-#1835 bundle stays analytic and is silent.** The
/// same stripping as above; the labels say analytic, so the `auto` the bundle loads is what
/// the fit ran.
#[test]
fn a_pre_1835_bundle_of_an_analytic_fit_is_silent() {
    let (model, population, base) = posthoc_base();
    assert_eq!(base.gradient_method_inner, ANALYTIC, "premise");
    let loaded = through_pre_1835_bundle(&with_sir_record(&base, &model, &population), &population);
    let r = crate::run_covariance(
        &loaded,
        Some(&model),
        Some(&population),
        &FitOptions::default(),
    )
    .expect("run_covariance");
    assert!(!warns(&r), "run_covariance: {:?}", r.warnings);
    let r = crate::run_sir(
        &loaded,
        Some(&model),
        Some(&population),
        &FitOptions {
            sir_samples: 20,
            sir_resamples: 10,
            sir_seed: Some(7),
            ..FitOptions::default()
        },
    )
    .expect("run_sir");
    assert!(!warns(&r), "run_sir: {:?}", r.warnings);
}
