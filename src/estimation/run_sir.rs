//! Standalone SIR — run Sampling Importance Resampling against an existing
//! `FitResult` without re-fitting.
//!
//! Mirrors the SIR step that [`fit()`](crate::api::fit) runs inline when
//! `options.sir = true`, but lets callers drive SIR after a fit has completed
//! (potentially from a different session, loaded via `.fitrx`).
//!
//! Input rules (shared with `run_covariance` through
//! `estimation::fit_inputs::resolve_fit_inputs`, #1622):
//! - `None` re-reads from `fit.model_path` / `fit.data_path`, reads the data the
//!   way the fit did (`[data_selection]` included) and binds the model from
//!   `fit.data_bindings`. If a stored hash exists, a mismatch is a **hard error**.
//! - A supplied `model` / `population` is not hash-checked (the in-memory values
//!   don't carry their source bytes), but must be the fitted one: the fit's
//!   bindings and θ count, the fit's subjects in the fit's order.
//! - A fit that records its reader settings (#1685) re-reads `fit.data_path` with
//!   them, so `Some(model)` with `population = None` does not read the model file.
//!   An older fit takes the model file's settings, which with `Some(model)` means
//!   reading the hash-verified model file; pass both to avoid it.
//! - A fit that carries a population fingerprint (every fit since #1685) refuses
//!   any population, supplied or re-read, that is not the one it was given.

use crate::estimation::uncertainty_samples::fitted_params_from_result;
use crate::types::*;
use nalgebra::DVector;

/// Put this SIR run's notes on a fit's warnings **in place of** any earlier
/// run's.
///
/// `run_sir` clones its input fit, and that fit may already carry `SIR:` lines
/// — it was produced with `sir = true` (the inline path in `fit_inner` pushes
/// the same prefix), or its own `run_sir` output is being piped back in. Those
/// lines describe the SIR result this call replaces: appending would print the
/// same line once per pass (#1037), and a re-run under different settings
/// would keep the old run's diagnosis beside the new numbers — the low-ESS
/// warning's own advice, re-running with `sir_scale = natural`, would leave
/// "ESS 3.5" next to an ESS of 140 (#1723). The in-fit SIR's `SIR failed: `
/// line goes too — this run succeeded in its place. `SIR requested` and every
/// `SIR fallback…` line are a different step's story and stay.
fn replace_sir_warnings(warnings: &mut Vec<String>, sir_warnings: &[String]) {
    warnings.retain(|w| !w.starts_with("SIR: ") && !w.starts_with("SIR failed: "));
    warnings.extend(sir_warnings.iter().map(|w| format!("SIR: {w}")));
}

/// The **data** −2 log L of a completed fit — what `sir::run_sir_core` takes as
/// its `ofv_hat` (#254).
///
/// `FitResult::ofv` is the *penalized* total once a `prior(...)` is declared, so
/// handing it to `run_sir_core` — which adds the penalty itself — counts the
/// penalty twice. Today that cancels, because `dofv` only ever enters normalized
/// log-weights, but the contract is what the next reader will rely on.
///
/// `ofv - ofv_prior` rather than the `ofv_data` field, because both are
/// `#[serde(default)]`: a `FitResult` deserialized from a pre-#254 YAML carries
/// `ofv_data = 0.0`, whereas `ofv_prior = 0.0` is the truth there. One spelling
/// is therefore correct for an old fit and a new one alike.
fn data_ofv(fit: &FitResult) -> f64 {
    fit.ofv - fit.ofv_prior
}

/// Run SIR against an existing fit. Returns a new `FitResult` that is a clone
/// of `fit` with the `sir_*` fields populated. This run's diagnostics — the
/// proposal conditioning (#1021) and the low-ESS warning (#1723) — **replace**
/// the input fit's `SIR: …` and `SIR failed: …` lines rather than being
/// appended to them, and `warnings_structured` is rebuilt; `sir_ess` remains
/// the quantitative signal for a poorly-matched proposal.
///
/// # Notes on integrity
///
/// Hash verification (when stored on the fit) hits the filesystem once for
/// the hash and again during parse / CSV read. On a fast local filesystem
/// the window between those two reads is too small to be a practical TOCTOU
/// concern; on a shared filesystem or network mount, a file modified
/// in that window would pass the check and then be parsed in its modified
/// form. The intended threat model is accidental edits, not adversarial
/// substitution.
///
/// Paths recorded on the fit are stored verbatim (no canonicalisation), so
/// relative paths resolve against whatever the cwd is at `run_sir` time,
/// not at fit time. Pass absolute paths to `fit_from_files` if your
/// downstream code may run from a different working directory — or
/// canonicalise the path on the fit yourself before save / re-use, e.g.
/// `fit.model_path = Some(std::fs::canonicalize(&path)?.to_string_lossy().into_owned())`.
///
/// # IOV models (n_kappa > 0)
///
/// For models with inter-occasion variability, re-reading the dataset
/// requires the `iov_column` the fit read with — that name doesn't survive on
/// a `CompiledModel`. A fit that records its reader settings (#1685) carries
/// it, so every cell runs. On an older fit, `None` for both `model` and
/// `population` parses the full model file (including `[fit_options]`) and
/// threads its `iov_column` into the reader, while `Some(model)` for an IOV
/// model with `population = None` is an error rather than read occasions with
/// an `iov_column` the supplied model may not share: the model carries none,
/// and the model file's need not be the one it was built with. Workaround:
/// pass both `Some(model)` and `Some(population)`.
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to SIR-refine. Must carry a
///   `covariance_matrix` (i.e. the original fit ran with `covariance = true`).
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path` (with
///   the `iov_column` constraint above for IOV models), routed by the model so
///   a joint model's event rows come back as event records (#1199). A supplied
///   population read without that routing is rejected (`E_ENDPOINT_UNROUTED`).
/// - `options`: the settings recorded in [`SirSettings`](crate::estimation::sir::SirSettings)
///   (`sir_samples`, `sir_resamples`, `sir_seed`, `sir_df`, `sir_scale`,
///   `sir_keep_samples`, `inner_maxiter`, `inner_tol`, `mu_referencing`, `n_agq`,
///   `inner_optimizer`, `ebe_warm_start` and the six `ode_*` overrides), plus
///   `verbose` and `cancel`. **Each recorded setting the caller leaves at its
///   [`FitOptions::default`] value is taken from `fit.sir_settings`** (#1758), so
///   `run_sir` with default options repeats the fit's own SIR — same scale, degrees of
///   freedom, draws and seed, bit for bit. The test is by value: an explicit default
///   (`sir_scale = SirScale::Packed` on a fit recorded `natural`) cannot be told from
///   an unset one and yields to the record. To override with a default value, set it
///   on `fit.sir_settings` (or clear that field) before calling. A fit without a
///   record (`sir_settings = None`: no SIR ran, or a `.fitrx` written before #1758)
///   uses `options` as given, except that an unset `sir_seed` falls back to
///   `fit.sir_seed` (the seed such a fit was given). `inner_optimizer` and
///   `ebe_warm_start` hold for the draws only: the process's previous values are
///   restored when SIR returns.
///   `method` and `interaction` are **not** read from `options`: they come from the
///   fit (`fit.method`, then `fit.interaction` for a method that does not fix it), so
///   the draws are weighted with the objective the estimates minimise (#1710, #1755)
///   and the result equals the in-fit SIR at the same settings. Other fields are
///   ignored.
///
/// # `[mixture]` models
///
/// Each draw is scored with the K-class mixture marginal, at per-class Ω/Σ rebuilt
/// from the fit (#1704). A class override's fitted value is carried only by the
/// in-memory result of a packed-space `fit()`, so for a mixture model with at least
/// one override this returns an error on a fit read from `.fitrx`, built in R, or
/// estimated by SAEM / IMP / Bayes (see
/// [`fitted_params_from_result`]).
pub fn run_sir(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, crate::diagnostics::EngineError> {
    // #1758: the fit's recorded SIR settings, wherever the caller left the default. Resolved
    // first, so every reader below sees them: this ODE scope and the one `run_sir_core` opens
    // itself (#1212), which is the one the draws' solves run under. The resolved
    // `inner_optimizer` / `ebe_warm_start` reach the draws through `run_sir_core`, which sets
    // those process globals for its run and restores them (#1767).
    let options = &resolve_sir_options(fit, options);
    // #1710: score the fit's own marginal, whatever `interaction` the caller carries.
    let options = &crate::estimation::fit_inputs::fitted_marginal_options(fit, options);
    // #1212: carry this call's ODE solver settings to the integrator, as `fit()` does. Every
    // SIR sample re-solves the inner loop, so without this a caller-supplied `ode_reltol` /
    // `ode_method` would be ignored and the sampled OFVs would come from a different
    // integration accuracy than the fit being refined. The scope also puts the sample
    // fan-out on a pool whose workers carry the same settings.
    crate::api::with_fit_ode_scope(options, || run_sir_scoped(fit, model, population, options))?
}

/// `options` with every SIR setting the caller left at its default taken from
/// `fit.sir_settings` (#1758), so `run_sir(fit, …, &FitOptions::default())`
/// repeats the SIR the fit reports. Value-based, as `ode_solver_override` is: a
/// field equal to its default reads as "no opinion". A fit with no record (SIR
/// never ran, or written before #1758) leaves `options` as given, but for an unset
/// `sir_seed`, which takes `fit.sir_seed`: before #1758 that field echoed the seed
/// the fit was given, so it is the seed a pre-#1758 SIR drew with (#1767).
fn resolve_sir_options(fit: &FitResult, options: &FitOptions) -> FitOptions {
    let mut o = options.clone();
    let Some(rec) = fit.sir_settings.as_ref() else {
        if o.sir_seed.is_none() {
            o.sir_seed = fit.sir_seed;
        }
        return o;
    };
    let d = FitOptions::default();
    macro_rules! recorded {
        ($field:ident = $value:expr) => {
            if o.$field == d.$field {
                o.$field = $value;
            }
        };
    }
    recorded!(sir_samples = rec.samples);
    recorded!(sir_resamples = rec.resamples);
    recorded!(sir_seed = Some(rec.seed));
    recorded!(sir_df = rec.df);
    recorded!(sir_scale = rec.scale);
    recorded!(sir_keep_samples = rec.keep_samples);
    recorded!(inner_maxiter = rec.inner_maxiter);
    recorded!(inner_tol = rec.inner_tol);
    recorded!(mu_referencing = rec.mu_referencing);
    recorded!(n_agq = rec.n_agq);
    recorded!(inner_optimizer = rec.inner_optimizer);
    recorded!(ebe_warm_start = rec.ebe_warm_start);
    recorded!(ode_reltol = rec.ode_reltol);
    recorded!(ode_abstol = rec.ode_abstol);
    recorded!(ode_max_steps = rec.ode_max_steps);
    recorded!(ode_method = rec.ode_method);
    recorded!(ode_stiff_abort_after = rec.ode_stiff_abort_after);
    recorded!(ode_auto_switch = rec.ode_auto_switch);
    o
}

fn run_sir_scoped(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, crate::diagnostics::EngineError> {
    // Hash verification runs before the covariance check so a stale-input
    // error wins over a missing-cov error. A user pointing at the wrong
    // model or dataset should hear about that first; the cov-missing case
    // is downstream and only matters once the inputs are confirmed.

    // --- Resolve model and population (#1622) ------------------------------
    //
    // Shared with `run_covariance`: re-parsed and re-read the way the fit read
    // them, `[data_selection]` included, and bound from `fit.data_bindings`.
    let inputs =
        crate::estimation::fit_inputs::resolve_fit_inputs(fit, model, population, "run_sir")?;
    let model_ref = inputs.model();
    let pop_ref = inputs.population();

    // Re-runs the inner loop (EBEs → the prediction walk), so it needs the same
    // dose-compartment precondition `fit()` enforces (#375) — a `Result`-returning
    // API must not abort the process from inside the walk. Mirrors `run_covariance`.
    crate::diagnostics::first_error(&crate::api::check_dose_compartments(model_ref, pop_ref))?;
    // …and the endpoint-routing precondition (#1199), as `fit()` enforces it: SIR on
    // a population read model-blind would resample the Gaussian half of a joint
    // likelihood. The re-read above is routed; this covers a supplied population.
    crate::diagnostics::first_error(&crate::api::check_endpoint_routing(
        model_ref, pop_ref, true,
    ))?;

    // --- Sanity-check dimensions ------------------------------------------
    if model_ref.n_eta != fit.omega.nrows() {
        return Err(crate::diagnostics::EngineError::from(format!(
            "supplied model has n_eta = {} but fit.omega is {}×{}. \
             Verify you supplied the same model used for the fit.",
            model_ref.n_eta,
            fit.omega.nrows(),
            fit.omega.ncols()
        ))
        .in_context("run_sir"));
    }
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(crate::diagnostics::EngineError::from(format!(
            "fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ))
        .in_context("run_sir"));
    }

    // --- Reconstruct ModelParameters and eta_hats -------------------------
    let params = fitted_params_from_result(fit, model_ref).map_err(|e| format!("run_sir: {e}"))?;
    let eta_hats: Vec<DVector<f64>> = fit.subjects.iter().map(|s| s.eta.clone()).collect();

    // --- Now require a covariance matrix to seed the proposal -------------
    let cov = fit.covariance_matrix.as_ref().ok_or_else(|| {
        "run_sir requires fit.covariance_matrix; re-run the original fit \
         with the covariance step enabled (FitOptions.run_covariance_step = \
         true, or `covariance = true` in the model file's [fit_options])."
            .to_string()
    })?;

    // --- Run SIR (identical to the inline path in fit()) ------------------
    let sir = crate::estimation::sir::run_sir_core(
        model_ref,
        pop_ref,
        &params,
        &eta_hats,
        cov,
        data_ofv(fit),
        options,
    )?;

    // --- Build the augmented FitResult ------------------------------------
    let mut out = fit.clone();
    replace_sir_warnings(&mut out.warnings, &sir.warnings);
    inputs.note_warnings(&mut out.warnings);
    crate::api::rebuild_warnings_structured(&mut out);
    crate::api::apply_sir_result(&mut out, Some(&sir), None);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::fit_from_files;
    use crate::io::hash::sha256_file;

    /// #1037: a fit that already carries the same `SIR:` line — because it was
    /// fitted with `sir = true`, or because its `run_sir` output is being piped
    /// back in — must not collect a second copy.
    #[test]
    fn replace_sir_warnings_does_not_duplicate() {
        let mut warnings = vec![
            "Covariance step: matrix was not positive definite".to_string(),
            "SIR: proposal covariance is rank-deficient [CL +0.71, V -0.70]".to_string(),
        ];
        let sir = vec![
            "proposal covariance is rank-deficient [CL +0.71, V -0.70]".to_string(),
            "proposal was shrunk in 1 direction(s) [KA +1.00]".to_string(),
        ];
        replace_sir_warnings(&mut warnings, &sir);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(
            warnings[2].starts_with("SIR: proposal was shrunk"),
            "{warnings:?}"
        );

        // Idempotent: a second pass over the same kernel output adds nothing.
        replace_sir_warnings(&mut warnings, &sir);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
    }

    /// #1723: a re-run's notes replace the earlier run's — a stale low-ESS line
    /// is dropped when the new run is healthy — while every non-kernel line,
    /// including the other `SIR …` phrasings, stays where it was.
    #[test]
    fn replace_sir_warnings_drops_the_replaced_runs_notes_only() {
        let mut warnings = vec![
            "Minimization terminated".to_string(),
            "SIR: effective sample size is 3.5 of 1000 draws".to_string(),
            "SIR fallback: proposal was shrunk in 1 direction(s).".to_string(),
            "SIR failed: covariance not positive definite".to_string(),
            "SIR fallback failed: proposal could not be made PD".to_string(),
            "SIR requested but no covariance".to_string(),
        ];
        replace_sir_warnings(&mut warnings, &[]);
        assert_eq!(
            warnings,
            [
                "Minimization terminated",
                "SIR fallback: proposal was shrunk in 1 direction(s).",
                "SIR fallback failed: proposal could not be made PD",
                "SIR requested but no covariance",
            ]
        );
    }

    // Use the in-tree warfarin example + data. They live at repo paths
    // `examples/warfarin.ferx` and `data/warfarin.csv` (see AGENTS.md);
    // tests run from the crate root, so relative paths work directly.
    const MODEL_PATH: &str = "examples/warfarin.ferx";
    const DATA_PATH: &str = "data/warfarin.csv";

    fn copy_example_to_tempdir(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        // Hash-mismatch tests need to mutate the source files. Copy them to a
        // tempdir so we don't touch the checked-in examples.
        let model = dir.join("model.ferx");
        let data = dir.join("data.csv");
        std::fs::copy(MODEL_PATH, &model).unwrap();
        std::fs::copy(DATA_PATH, &data).unwrap();
        (model, data)
    }

    fn quick_opts() -> FitOptions {
        // Small SIR settings so the test stays under a few seconds.
        FitOptions {
            verbose: false,
            run_covariance_step: true,
            // Pin the derivative-free outer optimizer: this test exercises the
            // SIR plumbing, not the optimizer-default choice, so keep it on the
            // path it was validated against rather than the `auto` default (#490).
            optimizer: crate::types::Optimizer::Bobyqa,
            sir_samples: 8,
            sir_resamples: 4,
            sir_seed: Some(1),
            ..FitOptions::default()
        }
    }

    #[test]
    fn paths_and_hashes_are_populated_by_fit_from_files() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");

        assert_eq!(
            fit.model_path.as_deref(),
            Some(model_path.to_str().unwrap())
        );
        assert_eq!(fit.data_path.as_deref(), Some(data_path.to_str().unwrap()));
        assert_eq!(fit.model_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(fit.data_hash.as_deref().map(|s| s.len()), Some(64));
        assert_eq!(
            fit.model_hash.as_deref(),
            Some(sha256_file(&model_path).unwrap().as_str())
        );
        assert_eq!(
            fit.data_hash.as_deref(),
            Some(sha256_file(&data_path).unwrap().as_str())
        );
    }

    #[test]
    fn run_sir_rejects_when_no_covariance() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = FitOptions {
            verbose: false,
            run_covariance_step: false,
            ..FitOptions::default()
        };
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        assert!(fit.covariance_matrix.is_none());
        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.to_string().contains("covariance_matrix"),
            "expected cov-missing message, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_detects_modified_model_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(opts.clone()),
        )
        .expect("fit must converge");

        // Tamper with the model file (append whitespace — enough to flip the
        // SHA-256). The next run_sir call must refuse.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap();
        writeln!(f, "  ").unwrap();
        drop(f);

        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.to_string().contains("model hash mismatch"),
            "expected hash-mismatch message, got: {}",
            err
        );
    }

    /// Run a fit and skip the test body with a logged message when the
    /// covariance step doesn't converge. SIR requires a non-None
    /// covariance matrix; the warfarin FD cov step is flaky. This helper
    /// centralises the skip pattern so the SIR happy-path tests don't fail
    /// spuriously.
    fn fit_with_cov_or_skip(
        model_path: &str,
        data_path: &str,
        opts: FitOptions,
    ) -> Option<FitResult> {
        let fit = fit_from_files(model_path, Some(data_path), None, Some(opts))
            .expect("fit must converge");
        if fit.covariance_matrix.is_none() {
            eprintln!(
                "[skip] covariance step did not produce a matrix (likely FD \
                 instability); skipping SIR happy-path assertions"
            );
            return None;
        }
        Some(fit)
    }

    #[test]
    fn run_sir_happy_path_populates_sir_fields() {
        // Integration test: fit_from_files → run_sir(None, None) → verify
        // the returned FitResult carries the four SIR diagnostics.
        // Exercises the "re-read model + data from paths + verify hashes"
        // code path, which is the primary use case for the public API.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        let out = run_sir(&fit, None, None, &opts).expect("run_sir succeeds");

        // Every SIR field populated.
        let ci_theta = out.sir_ci_theta.as_ref().expect("ci_theta populated");
        let ci_omega = out.sir_ci_omega.as_ref().expect("ci_omega populated");
        let ci_sigma = out.sir_ci_sigma.as_ref().expect("ci_sigma populated");
        let ess = out.sir_ess.expect("sir_ess populated");

        assert_eq!(ci_theta.len(), fit.theta.len());
        assert_eq!(ci_omega.len(), fit.omega.nrows());
        assert_eq!(ci_sigma.len(), fit.sigma.len());
        // #1705: warfarin declares no kappa, so there is no κ interval — `None`,
        // not `Some(vec![])`, so the writers and the bundle stay byte-identical.
        assert_eq!(out.sir_ci_kappa, None);
        // ESS is bounded by sir_samples; with sir_samples=8, sir_resamples=4
        // we expect ess > 0 and ess <= sir_samples.
        assert!(ess > 0.0 && ess <= opts.sir_samples as f64);

        // Lower <= upper on every CI band.
        for (lo, hi) in ci_theta {
            assert!(lo <= hi, "theta CI: {} > {}", lo, hi);
        }
        for (lo, hi) in ci_omega {
            assert!(lo <= hi, "omega CI: {} > {}", lo, hi);
        }

        // Non-SIR fields unchanged (we copy fit, then stamp on the SIR
        // fields — the rest must round-trip).
        assert_eq!(out.theta, fit.theta);
        assert_eq!(out.omega, fit.omega);
        assert_eq!(out.sigma, fit.sigma);
        assert_eq!(out.ofv, fit.ofv);
    }

    #[test]
    fn run_sir_errors_when_iov_model_supplied_without_population() {
        // Caller passes Some(model) for an IOV (n_kappa > 0) model but
        // None for population. The wrapper must refuse rather than
        // silently re-read the data without iov_column (which would drop
        // occasion parsing and produce wrong SIR results).
        //
        // The fit object here is shape-wise nonsense (warfarin non-IOV
        // fit + IOV model) — but the IOV check fires before any
        // dimension check, so this triggers the intended branch first.
        //
        // #1685 T8: on a **legacy** fit — no recorded reader settings or
        // fingerprint, as an older `.fitrx` — the refusal stands; a fit that
        // records its `iov_column` runs instead (T9).
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(mut fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };
        fit.reader_settings = None;
        fit.population_fingerprint = None;

        let iov_model = crate::parser::model_parser::parse_full_model_file(std::path::Path::new(
            "examples/warfarin_iov.ferx",
        ))
        .expect("parse warfarin_iov.ferx")
        .model;
        assert!(
            iov_model.n_kappa > 0,
            "warfarin_iov.ferx must declare kappa"
        );

        let err = run_sir(&fit, Some(&iov_model), None, &opts).unwrap_err();
        assert!(
            err.to_string().contains("IOV") && err.to_string().contains("population"),
            "expected IOV-needs-population error, got: {}",
            err
        );
    }

    /// The bits of a SIR CI vector, so `assert_eq!` compares exactly and a `NaN`
    /// endpoint cannot compare equal to anything but the same `NaN`.
    fn ci_bits(ci: &Option<Vec<(f64, f64)>>) -> Option<Vec<(u64, u64)>> {
        ci.as_ref()
            .map(|v| v.iter().map(|(a, b)| (a.to_bits(), b.to_bits())).collect())
    }

    /// #1705: `sir_ci_kappa` is filled on **both** SIR paths and they agree to the
    /// bit — `fit(sir = true)` and `run_sir` on the same fit, model, population
    /// and options. `warfarin_iov` (one kappa on CL, analytic one-compartment
    /// oral, FOCE as the file declares it; covariance forced on here).
    ///
    /// The standalone input has every SIR field cleared, so a `run_sir` that
    /// stopped filling `sir_ci_kappa` cannot pass on the value it inherited from
    /// the clone. Mutations, one per side: drop the fill in `run_sir` → the
    /// "standalone" `expect` dies; drop it in `fit()` → the "in-fit" `expect`
    /// dies. θ/Ω/σ/ESS ride along as bits, so a κ column that perturbed the
    /// resampling on one path only would show here too.
    ///
    /// Not a bracketing test: this fixture's ESS is low (4.9 of 1000 measured on
    /// macOS), so its CI is a handful of distinct draws. Bracketing and the
    /// agreement with the Wald interval are pinned on `mbma_placebo`
    /// (`tests/sir_ci_kappa.rs`, Tier 3), where ESS is ~143.
    #[test]
    fn in_fit_and_standalone_sir_report_the_same_kappa_ci() {
        let prep =
            crate::api::prepare_run("examples/warfarin_iov.ferx", Some("data/warfarin_iov.csv"))
                .expect("prepare warfarin_iov");
        let opts = FitOptions {
            verbose: false,
            run_covariance_step: true,
            sir: true,
            sir_samples: 300,
            sir_resamples: 100,
            sir_seed: Some(1705),
            ..prep.parsed.fit_options.clone()
        };
        let model = &prep.parsed.model;
        let pop = &prep.population;
        let fit = crate::api::fit(model, pop, &prep.init_params, &opts).expect("fit warfarin_iov");
        assert!(
            fit.covariance_matrix.is_some(),
            "the covariance step must succeed, or neither SIR path runs"
        );

        let mut bare = fit.clone();
        bare.sir_ci_theta = None;
        bare.sir_ci_omega = None;
        bare.sir_ci_sigma = None;
        bare.sir_ci_kappa = None;
        bare.sir_ess = None;
        let standalone = run_sir(&bare, Some(model), Some(pop), &opts).expect("run_sir");

        let k_fit = fit
            .sir_ci_kappa
            .as_ref()
            .expect("in-fit SIR filled no sir_ci_kappa");
        let k_sa = standalone
            .sir_ci_kappa
            .as_ref()
            .expect("standalone run_sir filled no sir_ci_kappa");
        assert_eq!(k_fit.len(), fit.kappa_names.len(), "one CI per kappa");
        for &(lo, hi) in k_fit {
            assert!(lo.is_finite() && hi.is_finite(), "κ CI [{lo}, {hi}]");
            assert!(
                0.0 < lo && lo <= hi,
                "κ CI [{lo}, {hi}] is not a variance interval"
            );
        }
        assert_eq!(
            ci_bits(&standalone.sir_ci_kappa),
            ci_bits(&fit.sir_ci_kappa),
            "κ: {k_sa:?} vs {k_fit:?}"
        );
        assert_eq!(
            ci_bits(&standalone.sir_ci_theta),
            ci_bits(&fit.sir_ci_theta),
            "θ"
        );
        assert_eq!(
            ci_bits(&standalone.sir_ci_omega),
            ci_bits(&fit.sir_ci_omega),
            "Ω"
        );
        assert_eq!(
            ci_bits(&standalone.sir_ci_sigma),
            ci_bits(&fit.sir_ci_sigma),
            "σ"
        );
        assert_eq!(
            standalone.sir_ess.map(f64::to_bits),
            fit.sir_ess.map(f64::to_bits),
            "ESS"
        );
    }

    #[test]
    fn run_sir_detects_modified_data_file() {
        // Symmetric to `run_sir_detects_modified_model_file` — verify the
        // data-hash branch fires on tamper. Without this test the data
        // side of the integrity check has no coverage.
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Append a comment line so the CSV still parses but the hash flips.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&data_path)
            .unwrap();
        writeln!(f, "# tampered").unwrap();
        drop(f);

        let err = run_sir(&fit, None, None, &opts).unwrap_err();
        assert!(
            err.to_string().contains("data hash mismatch"),
            "expected data hash-mismatch message, got: {}",
            err
        );
    }

    #[test]
    fn run_sir_with_caller_supplied_model_and_pop_skips_hash_check() {
        // When the caller passes Some(model) AND Some(population), the
        // wrapper uses them as-is — no hash verification. Tampering with
        // the on-disk files (so the recorded hashes no longer match)
        // must NOT trigger a hash mismatch error in this branch.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Tamper only the model file — changing its hash is sufficient to
        // verify the hash-bypass behaviour. Appending to the CSV would add
        // a fake subject (the CSV reader doesn't skip comment lines), which
        // would make pop.subjects.len() != fit.eta_hats.len() and panic in
        // run_inner_loop_warm; the data file hash is checked separately in
        // run_sir_detects_modified_data_file.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&model_path)
                .unwrap();
            writeln!(f, "# tampered").unwrap();
        }

        // Build model + population in memory (from the tampered/original files —
        // the caller-supplied branch doesn't verify hashes).
        let parsed = crate::parser::model_parser::parse_full_model_file(&model_path)
            .expect("parse tampered model");
        let pop =
            crate::io::datareader::read_nonmem_csv(&data_path, None, None).expect("read data");

        // Should succeed despite the on-disk tampering, because the
        // caller-supplied branch bypasses the hash check entirely.
        let out = run_sir(&fit, Some(&parsed.model), Some(&pop), &opts)
            .expect("caller-supplied model+pop must skip the hash check");
        assert!(out.sir_ess.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn run_sir_succeeds_with_fixed_parameters() {
        // Regression: before run_sir_core was restricted to the free
        // subspace, every SIR sample for a model with at least one FIX-ed
        // parameter failed the bounds check. `compute_covariance` zeroes the
        // rows/cols of FIX-ed indices, the proposal-cov regularisation
        // (eigenvalue floor of 1e-8) then perturbs them by ~1e-4, and
        // `compute_bounds` pins them with `lower == upper == x_hat[i]` — so
        // the strict bounds check rejected every sample and SIR returned
        // "All SIR samples had invalid weights".
        //
        // `examples/warfarin_fix.ferx` exercises FIX on a theta, an omega,
        // and a sigma in one go.
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model_fix.ferx");
        let data_path = dir.path().join("data.csv");
        std::fs::copy("examples/warfarin_fix.ferx", &model_path).unwrap();
        std::fs::copy(DATA_PATH, &data_path).unwrap();

        let opts = quick_opts();
        let Some(fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Sanity-check the fit actually carries FIX-ed parameters — without
        // this the test would silently degenerate into a non-FIX path.
        assert!(
            fit.theta_fixed.iter().any(|&b| b)
                || fit.omega_fixed.iter().any(|&b| b)
                || fit.sigma_fixed.iter().any(|&b| b),
            "expected at least one FIX-ed parameter in warfarin_fix.ferx"
        );

        let out = run_sir(&fit, None, None, &opts)
            .expect("SIR must succeed on a model with FIX-ed parameters");
        let ess = out.sir_ess.expect("sir_ess populated");
        assert!(
            ess > 0.0 && ess <= opts.sir_samples as f64,
            "ess = {} out of (0, {}]",
            ess,
            opts.sir_samples
        );
    }

    /// #1021: the covariance step floors non-identified eigenvalues of the FD
    /// Hessian before inverting it, so such a direction comes back in
    /// `covariance_matrix` with a variance of ~1/floor. Sampling that direction
    /// unshrunk put every SIR draw outside the packed bounds, and SIR failed
    /// with the uninformative "All SIR samples had invalid weights". The
    /// proposal is now capped at the bounds: SIR runs, and says so.
    #[test]
    fn run_sir_survives_an_explosive_proposal_direction() {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let opts = quick_opts();
        let Some(mut fit) = fit_with_cov_or_skip(
            model_path.to_str().unwrap(),
            data_path.to_str().unwrap(),
            opts.clone(),
        ) else {
            return;
        };

        // Inflate one free direction to the magnitude an eigenvalue-floored
        // Hessian produces. Adding to a diagonal keeps the matrix PSD, so this
        // is a covariance a real fit could hand us — not a malformed input.
        let mut cov = fit.covariance_matrix.clone().expect("covariance present");
        cov[(0, 0)] += 1e8;
        fit.covariance_matrix = Some(cov);

        let out = run_sir(&fit, None, None, &opts)
            .expect("SIR must survive an eigenvalue-floored proposal direction");
        let ess = out.sir_ess.expect("sir_ess populated");
        assert!(ess > 0.0, "ess = {ess}");
        assert!(
            out.warnings.iter().any(|w| w.contains("shrunk")),
            "the shrinkage must be reported to the user: {:?}",
            out.warnings
        );
    }

    #[test]
    fn run_sir_errors_when_no_model_path_recorded_and_no_caller_model() {
        // Cover the "no model path recorded and caller didn't supply one"
        // branch — the in-memory `fit()` path leaves `model_path = None`,
        // and a downstream caller that also passes `None` should get a
        // clear error rather than a panic or generic NPE-style failure.
        //
        // Cheapest way to get a valid FitResult with empty paths is to run
        // `fit_from_files` and then null out the path/hash fields.
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = copy_example_to_tempdir(dir.path());

        let mut fit = fit_from_files(
            model_path.to_str().unwrap(),
            Some(data_path.to_str().unwrap()),
            None,
            Some(quick_opts()),
        )
        .expect("fit must converge");
        fit.model_path = None;
        fit.data_path = None;
        fit.model_hash = None;
        fit.data_hash = None;

        let err = run_sir(&fit, None, None, &quick_opts()).unwrap_err();
        assert!(
            err.to_string().contains("no model supplied"),
            "expected 'no model supplied' error, got: {}",
            err
        );
    }

    /// `run_sir_core` adds the prior penalty to the `ofv_hat` it is handed
    /// (#254), so what `run_sir` passes has to be the **data** OFV.
    ///
    /// Both spellings that look right are wrong on one of these two inputs, and
    /// each is a distinct mutation: returning `fit.ofv` (the bug this replaced)
    /// reddens the priored case only, and returning `fit.ofv_data` reddens the
    /// legacy case only — `ofv_data` is `#[serde(default)]`, so a `FitResult`
    /// deserialized from a pre-#254 YAML carries `0.0` there.
    #[test]
    fn data_ofv_strips_the_prior_penalty_and_survives_a_legacy_fit() {
        // A priored fit: ofv is the penalized total, ofv_data the data half.
        let mut fit = FitResult {
            ofv: 110.0,
            ofv_prior: 10.0,
            ofv_data: 100.0,
            ..crate::types::test_helpers::empty_fit_result()
        };
        assert_eq!(data_ofv(&fit), 100.0);
        assert_ne!(
            data_ofv(&fit),
            fit.ofv,
            "passing the penalized total would double-count the penalty"
        );

        // A pre-#254 fit read back from YAML: neither new field was serialized,
        // so both default to 0.0 and the whole objective is the data half.
        fit.ofv = 100.0;
        fit.ofv_prior = 0.0;
        fit.ofv_data = 0.0;
        assert_eq!(data_ofv(&fit), 100.0);
    }

    // ── #1758: run_sir repeats the fit's own SIR ───────────────────────────

    /// A record with every field off its default, so each `recorded!` line has a
    /// value to take that the caller's default options do not already hold.
    fn off_default_settings() -> crate::estimation::sir::SirSettings {
        crate::estimation::sir::SirSettings {
            samples: 321,
            resamples: 123,
            seed: 4242,
            df: 3.0,
            scale: SirScale::Natural,
            keep_samples: true,
            inner_maxiter: 17,
            inner_tol: 3e-4,
            mu_referencing: false,
            n_agq: 5,
            inner_optimizer: InnerOptimizer::Lbfgs,
            ebe_warm_start: true,
            ode_reltol: 1e-7,
            ode_abstol: 1e-9,
            ode_max_steps: 777,
            ode_method: crate::ode::OdeMethod::Rodas5P,
            ode_stiff_abort_after: Some(9),
            ode_auto_switch: false,
        }
    }

    /// The resolution rule, field by field. Default caller options take every
    /// recorded value; a caller's non-default value wins; no record leaves the
    /// caller's options alone. Mutations: delete any one `recorded!` line (the
    /// first equality dies, the diff naming the field); invert the test to
    /// `!=` (the second dies); return the record when there is none (third).
    /// `SirSettings::from_options` is the reader on both sides, so a field it
    /// dropped would also fail the first equality.
    #[test]
    fn resolve_sir_options_takes_each_recorded_setting_the_caller_left_default() {
        let rec = off_default_settings();
        let mut fit = crate::types::test_helpers::minimal_fit_result();
        fit.sir_settings = Some(rec.clone());
        let d = FitOptions::default();
        assert_ne!(
            crate::estimation::sir::SirSettings::from_options(&d),
            rec,
            "premise: the record differs from the defaults"
        );

        let resolved = resolve_sir_options(&fit, &d);
        assert_eq!(
            crate::estimation::sir::SirSettings::from_options(&resolved),
            rec
        );

        // The caller's explicit, non-default value wins over the record.
        let caller = FitOptions {
            sir_df: 9.0,
            sir_scale: SirScale::Packed, // a default value: yields to the record
            sir_seed: Some(1),
            ode_reltol: 1e-5,
            ..FitOptions::default()
        };
        let resolved = resolve_sir_options(&fit, &caller);
        assert_eq!(resolved.sir_df, 9.0);
        assert_eq!(resolved.sir_seed, Some(1));
        assert_eq!(resolved.ode_reltol, 1e-5);
        assert_eq!(
            resolved.sir_scale,
            SirScale::Natural,
            "value-based: an explicit default cannot override the record"
        );
        assert_eq!(resolved.sir_samples, rec.samples);

        // No record: the caller's options as given.
        fit.sir_settings = None;
        let resolved = resolve_sir_options(&fit, &caller);
        assert_eq!(
            crate::estimation::sir::SirSettings::from_options(&resolved),
            crate::estimation::sir::SirSettings::from_options(&caller)
        );
    }

    /// #1767 finding 3: a fit with no record (a pre-#1758 `.fitrx`) still carries the seed
    /// it was given in `sir_seed`, and `run_sir` draws with it when the caller sets none —
    /// not with the built-in default. Both sides of the gate in one test: an unset caller
    /// seed takes the fit's, an explicit one wins. Mutations: drop the fallback (the first
    /// assertion gets `None`); apply it unconditionally (the second gets 7, not 1).
    #[test]
    fn resolve_sir_options_without_a_record_takes_the_fits_seed() {
        let mut fit = crate::types::test_helpers::minimal_fit_result();
        fit.sir_settings = None;
        fit.sir_seed = Some(7);
        let unset = FitOptions::default();
        assert_eq!(unset.sir_seed, None, "premise: the caller sets no seed");
        assert_eq!(resolve_sir_options(&fit, &unset).sir_seed, Some(7));
        let explicit = FitOptions {
            sir_seed: Some(1),
            ..FitOptions::default()
        };
        assert_eq!(resolve_sir_options(&fit, &explicit).sir_seed, Some(1));
    }

    /// `fit`'s SIR outputs cleared, its record kept, so a `run_sir` that failed
    /// to fill a field cannot pass on the value it inherited from the clone.
    fn sir_outputs_cleared(fit: &FitResult) -> FitResult {
        let mut bare = fit.clone();
        bare.sir_ci_theta = None;
        bare.sir_ci_omega = None;
        bare.sir_ci_sigma = None;
        bare.sir_ci_kappa = None;
        bare.sir_ess = None;
        bare.sir_resamples_packed = None;
        bare.sir_seed = None;
        bare
    }

    /// Fit with `sir = true`, then `run_sir` the result twice: once without its
    /// record and the explicit draw options (the pre-#1758 call, the premise),
    /// once with the record and **default** options. The second must repeat the
    /// in-fit SIR to the bit; the first must not, or the row tests nothing.
    fn assert_run_sir_repeats_in_fit_sir(
        prep: &crate::api::PreparedRun,
        opts: &FitOptions,
        row: &str,
    ) -> FitResult {
        let model = &prep.parsed.model;
        let pop = &prep.population;
        let fit = crate::api::fit(model, pop, &prep.init_params, opts).expect("fit");
        assert!(fit.covariance_matrix.is_some(), "{row}: no covariance");
        let ess_fit = fit.sir_ess.expect("in-fit SIR ran");
        assert!(ess_fit.is_finite(), "{row}: in-fit ESS {ess_fit}");
        assert_eq!(
            fit.sir_settings,
            Some(crate::estimation::sir::SirSettings::from_options(opts)),
            "{row}: the fit records what it scored under"
        );

        // Premise: the pre-#1758 call (no record, the draw options only) differs.
        let mut no_record = sir_outputs_cleared(&fit);
        no_record.sir_settings = None;
        let draws_only = FitOptions {
            verbose: false,
            sir_samples: opts.sir_samples,
            sir_resamples: opts.sir_resamples,
            sir_seed: opts.sir_seed,
            ..FitOptions::default()
        };
        let before = run_sir(&no_record, Some(model), Some(pop), &draws_only).expect("run_sir");
        assert_ne!(
            before.sir_ess.map(f64::to_bits),
            Some(ess_fit.to_bits()),
            "{row}: premise — default options must score differently, or this row tests nothing"
        );

        let quiet = FitOptions {
            verbose: false,
            ..FitOptions::default()
        };
        let out =
            run_sir(&sir_outputs_cleared(&fit), Some(model), Some(pop), &quiet).expect("run_sir");
        assert_eq!(
            out.sir_ess.map(f64::to_bits),
            Some(ess_fit.to_bits()),
            "{row}: ESS {:?} vs in-fit {ess_fit}",
            out.sir_ess
        );
        assert_eq!(
            ci_bits(&out.sir_ci_theta),
            ci_bits(&fit.sir_ci_theta),
            "{row}: θ"
        );
        assert_eq!(
            ci_bits(&out.sir_ci_omega),
            ci_bits(&fit.sir_ci_omega),
            "{row}: Ω"
        );
        assert_eq!(
            ci_bits(&out.sir_ci_sigma),
            ci_bits(&fit.sir_ci_sigma),
            "{row}: σ"
        );
        assert_eq!(out.sir_settings, fit.sir_settings, "{row}: settings");
        assert_eq!(out.sir_seed, fit.sir_seed, "{row}: seed");
        fit
    }

    /// The #1758 oracle fixture (ferx-r#472, measured in core on the plan):
    /// warfarin, FOCEI, covariance step, 200 / 100 draws, seed 7.
    fn warfarin_sir_opts(prep: &crate::api::PreparedRun) -> FitOptions {
        FitOptions {
            verbose: false,
            method: crate::types::EstimationMethod::FoceI,
            run_covariance_step: true,
            sir: true,
            sir_samples: 200,
            sir_resamples: 100,
            sir_seed: Some(7),
            ..prep.parsed.fit_options.clone()
        }
    }

    /// T5 (#1758): `run_sir(fit, default options)` is the in-fit SIR, to the
    /// bit, for each setting ferx-r#472 found ignored. Warfarin FOCEI, FD
    /// inner gradients. Mutations, one per row: delete `recorded!(sir_scale …)`,
    /// `recorded!(sir_df …)`, `recorded!(inner_maxiter …)` — that row's ESS
    /// assertion dies. (The process-global `inner_optimizer` row lives in its
    /// own binary, `tests/run_sir_inner_optimizer_global.rs`, so no concurrent
    /// test's `fit()` can move the global under it.)
    #[test]
    fn run_sir_with_default_options_repeats_the_in_fit_sir() {
        let prep = crate::api::prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
            .expect("prepare warfarin");
        let base = warfarin_sir_opts(&prep);
        let natural = FitOptions {
            sir_scale: SirScale::Natural,
            ..base.clone()
        };
        assert_run_sir_repeats_in_fit_sir(&prep, &natural, "sir_scale = natural");
        let df3 = FitOptions {
            sir_df: 3.0,
            ..base.clone()
        };
        assert_run_sir_repeats_in_fit_sir(&prep, &df3, "sir_df = 3");
        let maxiter5 = FitOptions {
            inner_maxiter: 5,
            ..base
        };
        assert_run_sir_repeats_in_fit_sir(&prep, &maxiter5, "inner_maxiter = 5");
    }

    /// T6 (#1758): the ODE twin. `mm_iv` (Michaelis–Menten, the smallest ODE
    /// example), FOCEI, a non-default `ode_reltol` / `ode_abstol` and a pinned
    /// `rk45`, FD inner gradients. The recorded solver settings must reach the
    /// integrator. Mutations: delete `recorded!(ode_reltol …)`, `(ode_abstol …)`
    /// or `(ode_method …)` — ESS differs. Opening `run_sir`'s own ODE scope with
    /// the caller's options instead is an *equivalent* mutation: `run_sir_core`
    /// opens its own scope from the resolved options (#1212), and that is the
    /// one the draws run under.
    #[test]
    fn run_sir_repeats_an_ode_fit_sir_under_its_recorded_tolerances() {
        let prep = crate::api::prepare_run("examples/mm_iv.ferx", Some("data/mm_iv.csv"))
            .expect("prepare mm_iv");
        let opts = FitOptions {
            verbose: false,
            run_covariance_step: true,
            sir: true,
            sir_samples: 100,
            sir_resamples: 50,
            sir_seed: Some(7),
            ode_reltol: 1e-3,
            ode_abstol: 1e-5,
            ode_method: crate::ode::OdeMethod::Rk45,
            ..prep.parsed.fit_options.clone()
        };
        assert_run_sir_repeats_in_fit_sir(&prep, &opts, "ode_reltol = 1e-3");
    }

    /// T7 (#1758): a caller's non-default setting overrides the record, and the
    /// output records what this run used — not the input fit's seed or scale.
    /// The input keeps its SIR outputs (seed 7, `packed`), so a `run_sir` that
    /// left them in place reports the old ones. Mutation: drop the
    /// `apply_sir_result` call in `run_sir` (seed stays 7, scale `packed`).
    #[test]
    fn run_sir_records_the_settings_it_ran_not_the_inputs() {
        let prep = crate::api::prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
            .expect("prepare warfarin");
        let opts = FitOptions {
            sir_df: 3.0,
            ..warfarin_sir_opts(&prep)
        };
        let model = &prep.parsed.model;
        let pop = &prep.population;
        let fit = crate::api::fit(model, pop, &prep.init_params, &opts).expect("fit");
        assert_eq!(fit.sir_seed, Some(7));
        let caller = FitOptions {
            verbose: false,
            sir_scale: SirScale::Natural,
            sir_seed: Some(99),
            ..FitOptions::default()
        };
        let out = run_sir(&fit, Some(model), Some(pop), &caller).expect("run_sir");
        let st = out.sir_settings.as_ref().expect("settings recorded");
        assert_eq!(st.scale, SirScale::Natural, "the caller's scale");
        assert_eq!(st.seed, 99, "the caller's seed");
        assert_eq!(out.sir_seed, Some(99));
        assert_eq!(st.df, 3.0, "the record's df, which the caller left default");
        assert_eq!(st.samples, 200, "the record's draw count");
    }

    /// T10 (#1758): the record survives a `.fitrx` round trip, and `run_sir` on
    /// the reloaded fit repeats the in-fit SIR — to the bit up to the one input
    /// `.fitrx` stores lossily. `ebes.csv` writes each EBE at 6 dp (#1631), and the EBEs
    /// warm-start every draw's inner solve, so the reloaded ESS moved by 7.1e-9
    /// (8.5e-11 relative, measured on macOS arm64). The test pins that this
    /// rounding is the *whole* gap: the in-memory fit with its EBEs rounded the
    /// same way gives the reloaded result bit for bit. Without the record the
    /// draws are scored at df 5, an ESS gap of ~4 (T5's premise), so the 1e-8
    /// relative bound has ~100× headroom over the measured gap and ~10⁶× margin
    /// to the defect. Mutation: drop the `settings` wire field (the record
    /// assertion dies, and `run_sir` scores at df 5).
    #[test]
    fn run_sir_after_a_fitrx_round_trip_repeats_the_in_fit_sir() {
        let prep = crate::api::prepare_run("examples/warfarin.ferx", Some("data/warfarin.csv"))
            .expect("prepare warfarin");
        let opts = FitOptions {
            sir_df: 3.0,
            ..warfarin_sir_opts(&prep)
        };
        let model = &prep.parsed.model;
        let pop = &prep.population;
        let fit = crate::api::fit(model, pop, &prep.init_params, &opts).expect("fit");
        let ess_fit = fit.sir_ess.expect("in-fit SIR ran");
        assert!(ess_fit.is_finite());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("df3.fitrx");
        crate::io::fitrx::save_fit(
            &fit,
            pop,
            "src\n",
            &path,
            crate::io::fitrx::SaveFitOptions::default(),
        )
        .expect("save");
        let loaded = crate::io::fitrx::load_fit(&path).expect("load").fit;
        assert_eq!(loaded.sir_settings, fit.sir_settings, "record round-trips");

        let quiet = FitOptions {
            verbose: false,
            ..FitOptions::default()
        };
        let out = run_sir(
            &sir_outputs_cleared(&loaded),
            Some(model),
            Some(pop),
            &quiet,
        )
        .expect("run_sir");

        // The in-memory fit with its EBEs rounded as `ebes.csv` writes them.
        let mut rounded = sir_outputs_cleared(&fit);
        for s in &mut rounded.subjects {
            for e in s.eta.iter_mut() {
                *e = crate::io::output::fmt_num(*e).parse().unwrap();
            }
        }
        let control = run_sir(&rounded, Some(model), Some(pop), &quiet).expect("run_sir");
        assert_eq!(
            out.sir_ess.map(f64::to_bits),
            control.sir_ess.map(f64::to_bits),
            "the EBE rounding must be the whole round-trip gap: {:?} vs {:?}",
            out.sir_ess,
            control.sir_ess
        );
        assert_eq!(
            ci_bits(&out.sir_ci_theta),
            ci_bits(&control.sir_ci_theta),
            "θ"
        );

        let ess = out.sir_ess.expect("standalone SIR ran");
        assert!(ess.is_finite(), "ESS {ess}");
        let rel = (ess - ess_fit).abs() / ess_fit;
        assert!(rel <= 1e-8, "ESS {ess} vs in-fit {ess_fit} (rel {rel:e})");
    }
}

/// #1622 T7: `run_sir(None, None)` runs on the model **as fitted**, so it equals the
/// call with `prepare_run`'s bound model and population to the bit. Same fixture and
/// engine as `run_covariance::from_fit_bindings` (analytic two-compartment oral,
/// FOCEI, analytic Dual2 inner gradient).
#[cfg(test)]
mod from_fit_bindings {
    use super::*;
    use crate::estimation::fit_inputs::test_fixtures::{design, sir_case, Kind};

    /// Every kind, `sir_ess` and `sir_ci_theta` `to_bits`. The ESS is asserted above
    /// 10 on every arm first: on a degenerate proposal (ESS ≈ 1, measured with the
    /// level block on `Q`) a bit match compares a single draw and would pass on a
    /// wrong model too.
    ///
    /// Tier 3: five fits with their covariance step and ten SIR runs. The resolver it
    /// exercises is the one `run_covariance` shares, whose per-PR test
    /// (`run_covariance::from_fit_bindings::the_re_read_binds_and_filters_as_the_fit_did`)
    /// dies on the same two mutations.
    ///
    /// Mutations — skip the bind in `resolve_fit_inputs`: `Median`'s ESS moves (29.87
    /// vs 126.31 measured before the fix) and `Level` is refused on `n_theta`; drop the
    /// selection from the re-read: `Select` is refused on the subject count (a panic in
    /// the inner loop before #1622).
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn the_re_read_equals_the_model_bound_on_the_fit_data() {
        for kind in [
            Kind::Plain,
            Kind::Median,
            Kind::Level,
            Kind::LevelMedian,
            Kind::Select,
        ] {
            let c = sir_case(kind);
            let want = run_sir(
                &c.fit,
                Some(&c.prep.parsed.model),
                Some(&c.prep.population),
                &c.opts,
            )
            .unwrap_or_else(|e| panic!("{kind:?} oracle: {e}"));
            let got =
                run_sir(&c.fit, None, None, &c.opts).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            let ess = want.sir_ess.expect("the oracle reports an ESS");
            assert!(ess > 10.0, "{kind:?}: degenerate SIR oracle, ESS {ess}");
            assert_eq!(
                got.sir_ess.map(f64::to_bits),
                want.sir_ess.map(f64::to_bits),
                "{kind:?}: ESS {:?} vs {:?}",
                got.sir_ess,
                want.sir_ess
            );
            let ci = |f: &FitResult| {
                f.sir_ci_theta.as_ref().map(|v| {
                    v.iter()
                        .map(|(a, b)| (a.to_bits(), b.to_bits()))
                        .collect::<Vec<_>>()
                })
            };
            assert!(ci(&want).is_some(), "{kind:?}");
            assert_eq!(ci(&got), ci(&want), "{kind:?}: sir_ci_theta bits");
        }
    }

    /// #1729 T5: SIR shares the resolver's empty-bindings check. A fit that records
    /// no bindings, lent the `Median` model bound on a design (WT × 1.3), is refused
    /// under SIR's own prefix rather than weighting draws centred on the design.
    ///
    /// Mutation — bypass the resolver's check: the call returns `Ok`.
    #[test]
    fn an_empty_bindings_fit_refuses_a_design_bound_model() {
        let c = sir_case(Kind::Median);
        let mut fit = c.fit.clone();
        fit.data_bindings = Default::default();
        let design = design(&c);
        let err = run_sir(
            &fit,
            Some(&design.parsed.model),
            Some(&c.prep.population),
            &c.opts,
        )
        .map(|_| ())
        .expect_err("refused");
        assert!(
            err.to_string().starts_with(
                "run_sir: this fit records no data-derived bindings (an older `.fitrx`), so \
                 the supplied model's covariate statistics were checked against the supplied \
                 population, and they differ: the median of `WT` is "
            ),
            "{err}"
        );
    }
}
