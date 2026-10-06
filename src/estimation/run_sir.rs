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
//! - `Some(model)` with `population = None` still reads the hash-verified model
//!   file, for the reader settings the re-read needs; pass both to avoid it.

use crate::estimation::uncertainty_samples::fitted_params_from_result;
use crate::types::*;
use nalgebra::DVector;

/// Append the SIR kernel's proposal-conditioning notes to a fit's warnings,
/// skipping any that are already there.
///
/// `run_sir` clones its input fit, and that fit may already carry identical
/// `SIR:` lines — it was produced with `sir = true` (the inline path in
/// `fit_inner` pushes the same text), or its own `run_sir` output is being piped
/// back in. Without the dedupe the same line accumulates and is printed once per
/// pass (#1037).
fn push_sir_warnings(warnings: &mut Vec<String>, sir_warnings: &[String]) {
    for w in sir_warnings {
        let line = format!("SIR: {}", w);
        if !warnings.contains(&line) {
            warnings.push(line);
        }
    }
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
/// of `fit` with the `sir_*` fields populated. Proposal-conditioning
/// diagnostics (rank deficiency / bound-driven shrinkage, #1021) are appended
/// to the returned fit's `warnings` as `SIR: …` lines, deduplicated against
/// what the input fit already carried; `sir_ess` remains the quantitative
/// signal for a poorly-matched proposal.
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
/// requires the `iov_column` name from the model file's `[fit_options]`
/// block — that name doesn't survive on a `CompiledModel`. When the
/// caller passes `None` for both `model` and `population`, this function
/// parses the full model file (including `[fit_options]`) and threads
/// `iov_column` into the model-routed reader. When the caller supplies
/// `Some(model)` for an IOV model but leaves `population = None`, `run_sir`
/// returns an error rather than read occasions with an `iov_column` the
/// supplied model may not share: the model carries none, and the model file's
/// need not be the one it was built with. Workaround: pass both `Some(model)`
/// and `Some(population)` for IOV cases.
///
/// # Arguments
/// - `fit`: the maximum-likelihood fit to SIR-refine. Must carry a
///   `covariance_matrix` (i.e. the original fit ran with `covariance = true`).
/// - `model`: pre-compiled model. When `None`, re-parsed from `fit.model_path`.
/// - `population`: dataset. When `None`, re-read from `fit.data_path` (with
///   the `iov_column` constraint above for IOV models), routed by the model so
///   a joint model's event rows come back as event records (#1199). A supplied
///   population read without that routing is rejected (`E_ENDPOINT_UNROUTED`).
/// - `options`: SIR-relevant fields read are `sir_samples`, `sir_resamples`,
///   `sir_seed`, `sir_keep_samples`, plus the inner-loop settings
///   (`inner_maxiter`, `inner_tol`, `mu_referencing`, `verbose`, `cancel`).
///   `interaction` is **not** read from `options`: it comes from the fit
///   (`fit.method`, then `fit.interaction`), so the draws are weighted with the
///   objective the estimates minimise (#1710). Other fields (e.g. `method`) are
///   ignored.
pub fn run_sir(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // #1710: score the fit's own marginal, whatever `interaction` the caller carries.
    let options = &crate::estimation::fit_inputs::fitted_marginal_options(fit, options);
    // #1212: carry this call's ODE solver settings to the integrator, as `fit()` does. Every
    // SIR sample re-solves the inner loop, so without this a caller-supplied `ode_reltol` /
    // `ode_method` would be ignored and the sampled OFVs would come from a different
    // integration accuracy than the fit being refined. The scope also puts the sample
    // fan-out on a pool whose workers carry the same settings.
    crate::api::with_fit_ode_scope(options, || run_sir_scoped(fit, model, population, options))?
}

fn run_sir_scoped(
    fit: &FitResult,
    model: Option<&CompiledModel>,
    population: Option<&Population>,
    options: &FitOptions,
) -> Result<FitResult, String> {
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
        return Err(format!(
            "run_sir: supplied model has n_eta = {} but fit.omega is {}×{}. \
             Verify you supplied the same model used for the fit.",
            model_ref.n_eta,
            fit.omega.nrows(),
            fit.omega.ncols()
        ));
    }
    if !fit.subjects.is_empty() && fit.subjects[0].eta.len() != model_ref.n_eta {
        return Err(format!(
            "run_sir: fit.subjects[0] has eta dim {} but model has n_eta = {}. \
             Subject EBEs are inconsistent with the supplied model.",
            fit.subjects[0].eta.len(),
            model_ref.n_eta
        ));
    }

    // --- Reconstruct ModelParameters and eta_hats -------------------------
    let params = fitted_params_from_result(fit, model_ref);
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
    push_sir_warnings(&mut out.warnings, &sir.warnings);
    out.sir_ci_kappa = sir.kappa_ci();
    out.sir_ci_theta = Some(sir.ci_theta);
    out.sir_ci_omega = Some(sir.ci_omega);
    out.sir_ci_sigma = Some(sir.ci_sigma);
    out.sir_ess = Some(sir.effective_sample_size);
    out.sir_resamples_packed = sir.resamples_packed;
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
    fn push_sir_warnings_does_not_duplicate() {
        let mut warnings = vec![
            "Covariance step: matrix was not positive definite".to_string(),
            "SIR: proposal covariance is rank-deficient [CL +0.71, V -0.70]".to_string(),
        ];
        let sir = vec![
            "proposal covariance is rank-deficient [CL +0.71, V -0.70]".to_string(),
            "proposal was shrunk in 1 direction(s) [KA +1.00]".to_string(),
        ];
        push_sir_warnings(&mut warnings, &sir);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        assert!(
            warnings[2].starts_with("SIR: proposal was shrunk"),
            "{warnings:?}"
        );

        // Idempotent: a second pass over the same kernel output adds nothing.
        push_sir_warnings(&mut warnings, &sir);
        assert_eq!(warnings.len(), 3, "{warnings:?}");
    }

    /// A clean SIR run leaves the fit's warnings untouched.
    #[test]
    fn push_sir_warnings_is_a_no_op_when_the_proposal_was_clean() {
        let mut warnings = vec!["Minimization terminated".to_string()];
        push_sir_warnings(&mut warnings, &[]);
        assert_eq!(warnings, vec!["Minimization terminated".to_string()]);
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
            err.contains("covariance_matrix"),
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
            err.contains("model hash mismatch"),
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
            err.contains("IOV") && err.contains("population"),
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
            err.contains("data hash mismatch"),
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
            err.contains("no model supplied"),
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
            err.starts_with(
                "run_sir: this fit records no data-derived bindings (an older `.fitrx`), so \
                 the supplied model's covariate statistics were checked against the supplied \
                 population, and they differ: the median of `WT` is "
            ),
            "{err}"
        );
    }
}
