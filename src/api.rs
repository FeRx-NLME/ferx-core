use crate::diagnostics::{first_error, CheckReport, Diagnostic};
use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::parameterization::theta_packs_log;
use crate::estimation::saem;
use crate::io::datareader::read_nonmem_csv;
use crate::pk;
use crate::stats::likelihood::{compute_cwres, foce_subject_nll, foce_subject_nll_iov};
use crate::stats::residual_error::{compute_iwres, iwres_autocorrelation};
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Build the `FitResult.neural_networks` summary from the compiled model's
/// `[covariate_nn]` blocks. Empty when no NN blocks are present, so output
/// writers can always iterate `result.neural_networks` without branching.
#[cfg(feature = "nn")]
fn build_neural_network_infos(model: &CompiledModel) -> Vec<NeuralNetworkInfo> {
    use crate::nn::CovariateMapper;
    model
        .covariate_nns
        .iter()
        .map(|nn| NeuralNetworkInfo {
            name: nn.name.clone(),
            shape: nn.mapper.mlp().layer_sizes().to_vec(),
            hidden_activation: nn.mapper.mlp().hidden_activation().as_str().to_string(),
            output_activation: nn.mapper.mlp().output_activation().as_str().to_string(),
            n_weights: nn.mapper.n_weights(),
            weights_offset: nn.weights_offset,
            input_names: nn.mapper.input_names().to_vec(),
            output_names: nn.mapper.output_names().to_vec(),
        })
        .collect()
}
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

/// Route predictions through analytical PK or ODE solver, then apply
/// `model.scaling` so simulate / predict / post-fit IPRED see the same
/// scaled output as the estimation dispatcher in `pk::compute_predictions_with_tv_into_with_schedule`.
///
/// `theta` and `eta` are required so that `ScalingSpec::ExpressionScale`
/// can evaluate its `scale_fn(theta, eta, covariates)`. Callers that don't
/// have a separate eta vector (population predictions) pass an all-zero eta.
pub(crate) fn model_preds(
    model: &CompiledModel,
    subject: &Subject,
    pk_params: &PkParams,
    theta: &[f64],
    eta: &[f64],
) -> Vec<f64> {
    let mut preds = if let Some(ref ode_spec) = model.ode_spec {
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values, theta, eta)
    } else {
        pk::compute_predictions(model.pk_model, subject, pk_params)
    };
    pk::apply_scaling(model, subject, theta, eta, &mut preds);
    pk::apply_log_transform(model, &mut preds);
    preds
}

/// Log-transform every observation (including M3 LLOQ values carried on CENS
/// rows — they live in the same `observations` vector) in place, for LTBS case 2
/// (`log(DV) ~ additive`, natural-scale data). Returns the count of non-positive
/// DV values, which are floored to [`crate::pk::LTBS_FLOOR`] before the log so the
/// result stays finite. Case 1 (`DV ~ log_additive`, `dv_pre_logged`) must NOT
/// call this — the DV is already on the log scale.
fn log_transform_observations(pop: &mut Population) -> usize {
    let mut n_nonpos = 0usize;
    for subject in &mut pop.subjects {
        for v in &mut subject.observations {
            if *v <= 0.0 {
                n_nonpos += 1;
            }
            *v = v.max(crate::pk::LTBS_FLOOR).ln();
        }
    }
    n_nonpos
}

/// Run a model file with a NONMEM-format CSV dataset.
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_with_data(
    model_path: &str,
    data_path: &str,
) -> Result<(FitResult, Population), String> {
    run_model_with_data_inits(model_path, data_path, None)
}

/// Like [`run_model_with_data`], but lets the caller (e.g. the CLI's
/// `--inits-from-nca` flag) override the model file's `inits_from_nca` fit
/// option. When `inits_override` is `None` the model-file value is used as-is;
/// when `Some(method)` it forces that NCA strategy regardless of the file.
pub fn run_model_with_data_inits(
    model_path: &str,
    data_path: &str,
    inits_override: Option<crate::suggest_start::NcaInit>,
) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    set_model_name(&mut parsed.model, model_path);
    if let Some(method) = inits_override {
        parsed.fit_options.inits_from_nca = Some(method);
    }

    eprintln!("Model: {}", parsed.model.name);

    let iov_col = parsed.fit_options.iov_column.as_deref();
    let population = read_nonmem_csv(Path::new(data_path), None, iov_col)?;
    eprintln!(
        "Data:  {} subjects, {} observations from {}",
        population.subjects.len(),
        population.n_obs(),
        data_path
    );

    let init_params = build_init_params(&parsed);
    let mut result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    // Hash both inputs *after* the fit so we don't double up disk reads
    // (the model and CSV are already in the page cache from parse + read
    // upstream). Errors here are non-fatal: the fit already succeeded, and
    // a missing hash just disables the integrity check in run_sir.
    result.model_path = Some(model_path.to_string());
    result.data_path = Some(data_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    result.data_hash = crate::io::hash::sha256_file(Path::new(data_path)).ok();
    Ok((result, population))
}

/// Run a model file with simulated data (from [simulation] block).
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_simulate(model_path: &str) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;
    use std::collections::HashMap;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    let sim_spec = parsed
        .simulation
        .clone()
        .ok_or("Model file has no [simulation] block — use --data instead")?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    // Build template population
    let subjects: Vec<Subject> = (1..=sim_spec.n_subjects)
        .map(|i| Subject {
            id: format!("{}", i),
            doses: vec![DoseEvent::new(
                0.0,
                sim_spec.dose_amt,
                sim_spec.dose_cmt,
                0.0,
                false,
                0.0,
            )],
            obs_times: sim_spec.obs_times.clone(),
            observations: vec![0.0; sim_spec.obs_times.len()],
            obs_cmts: vec![1; sim_spec.obs_times.len()],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0; sim_spec.obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
    };

    // Simulate
    eprintln!(
        "Simulating {} subjects (seed={})...",
        sim_spec.n_subjects, sim_spec.seed
    );
    let sim_results = simulate_with_seed(
        &parsed.model,
        &template,
        &parsed.model.default_params,
        1,
        sim_spec.seed,
    );

    let mut population = template;
    for subject in &mut population.subjects {
        let sims: Vec<_> = sim_results.iter().filter(|s| s.id == subject.id).collect();
        for (j, s) in sims.iter().enumerate() {
            if j < subject.observations.len() {
                // Under LTBS the simulated DV is on the log scale and may be
                // negative, so the positivity floor only applies to natural-scale
                // simulation.
                subject.observations[j] = if parsed.model.log_transform {
                    s.dv_sim
                } else {
                    s.dv_sim.max(0.001)
                };
            }
        }
    }

    eprintln!(
        "Loaded {} subjects, {} observations",
        population.subjects.len(),
        population.n_obs()
    );

    let init_params = build_init_params(&parsed);
    let mut result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    // No data file to hash — data is simulated in-process. Hash the model
    // post-fit (same pattern as `run_model_with_data`); failures are
    // non-fatal and just disable the integrity check in `run_sir`.
    result.model_path = Some(model_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    Ok((result, population))
}

/// Legacy alias
pub fn run_from_file(path: &str) -> Result<FitResult, String> {
    run_model_simulate(path).map(|(r, _)| r)
}

fn set_model_name(model: &mut CompiledModel, path: &str) {
    if model.name == "Unnamed" {
        if let Some(stem) = Path::new(path).file_stem().and_then(|s| s.to_str()) {
            model.name = stem.to_string();
        }
    }
}

fn build_init_params(parsed: &ParsedModel) -> ModelParameters {
    parsed.model.default_params.clone()
}

/// Fail early if the model references covariates that the data doesn't carry.
/// Case-sensitive: `CRCL` and `crcl` are distinct names. Historically a missing
/// covariate silently evaluated to zero, which left fits stuck at the initial
/// estimates with no visible diagnostic (see commit introducing this check).
///
/// Returns a diagnostic per problem (here, at most one). The message text is
/// kept byte-for-byte identical to the historical `Err(String)` so `fit()`'s
/// error — produced via [`first_error`] — is unchanged.
fn check_covariates(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();

    if missing.is_empty() {
        return Vec::new();
    }

    let available = if population.covariate_names.is_empty() {
        "(none)".to_string()
    } else {
        population.covariate_names.join(", ")
    };
    vec![Diagnostic::error(
        "E_MISSING_COVARIATE",
        format!(
            "Model references covariate(s) not found in data (case-sensitive): {}. \
             Available covariate columns: {}.",
            missing.join(", "),
            available
        ),
    )
    .with_suggestion(format!("available covariate columns: {}", available))]
}

/// Per-CMT scaling needs every observed CMT to have an entry in the
/// `ScalingSpec::PerCmt` / `OdeReadout::PerCmt` map. Wraps the existing
/// `pk::validate_per_cmt_scaling` (which the parser can't run — it doesn't see
/// the data), preserving its message verbatim.
fn check_per_cmt_scaling(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    match pk::validate_per_cmt_scaling(model, &population.subjects) {
        Ok(()) => Vec::new(),
        Err(msg) => vec![Diagnostic::error("E_PER_CMT_SCALING", msg).with_block("scaling")],
    }
}

/// Per-CMT (multi-endpoint) error models: every observed CMT must have a
/// matching `CMT=N:` entry in `[error_model]`.
fn check_per_cmt_error_model(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let crate::types::ErrorSpec::PerCmt(map) = &model.error_spec else {
        return Vec::new();
    };
    use std::collections::BTreeSet;
    let mut missing = BTreeSet::new();
    for subj in &population.subjects {
        for &cmt in &subj.obs_cmts {
            if !map.contains_key(&cmt) {
                missing.insert(cmt);
            }
        }
    }
    if missing.is_empty() {
        return Vec::new();
    }
    let list = missing
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    vec![Diagnostic::error(
        "E_PER_CMT_ERROR_MODEL",
        format!(
            "[error_model] has no entry for observed compartment(s) {}; \
             add a `CMT=N: DV ~ ...` line for each observed CMT.",
            list
        ),
    )
    .with_block("error_model")]
}

/// All data-dependent *fatal* compatibility checks between a compiled model and
/// a dataset, collected into one diagnostic list. Shared by `fit()` (which
/// stops at the first error via [`first_error`]) and `ferx check` (which
/// reports every finding). Check order matches the historical inline order in
/// `fit()` so the first error is unchanged: covariates, scaling, error model.
pub fn check_model_data(model: &CompiledModel, population: &Population) -> Vec<Diagnostic> {
    let mut diags = check_covariates(model, population);
    diags.extend(check_per_cmt_scaling(model, population));
    diags.extend(check_per_cmt_error_model(model, population));
    diags
}

/// Model + estimation-option *compatibility* checks that don't depend on data:
/// estimation method vs an SDE (`[diffusion]`) model, IMP chain placement, and
/// optimizer vs IOV. These mirror the guards at the top of `fit_inner`, so a
/// clean `ferx check` and a `fit()` agree on which method/model combinations are
/// rejected (rather than reporting `valid: true` and then failing at fit time).
/// `fit_inner` consumes these via [`first_error`]; message text is identical to
/// the historical inline guards. Check order matches `fit_inner` so the first
/// error is unchanged.
pub fn check_model_options(model: &CompiledModel, options: &FitOptions) -> Vec<Diagnostic> {
    let chain = options.method_chain();
    let mut diags = Vec::new();

    // SDE ([diffusion]) is incompatible with SAEM, with the Gauss-Newton
    // methods, and with the autodiff gradient path (EKF estimation requires
    // FD-FOCE/FOCEI).
    if model.is_sde() {
        if chain.iter().any(|&m| m == EstimationMethod::Saem) {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "method = saem is not compatible with a [diffusion] block. \
                     SDE / EKF estimation requires FOCE or FOCEI. Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if chain
            .iter()
            .any(|&m| matches!(m, EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid))
        {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "SDE ([diffusion]) is not supported with method = gn or gn_hybrid. \
                     Use method = foce or method = focei.",
                )
                .with_block("fit_options"),
            );
        }
        if options.gradient_method == crate::types::GradientMethod::Ad {
            diags.push(
                Diagnostic::error(
                    "E_SDE_INCOMPATIBLE",
                    "gradient_method = ad is not compatible with a [diffusion] block. \
                     Set gradient_method = fd (or leave it unset — fd is selected automatically).",
                )
                .with_block("fit_options"),
            );
        }
    }

    // Explicit `gradient_method = ad` on a build compiled WITHOUT the `autodiff`
    // feature: AD is unavailable, so the inner loop would silently fall back to
    // FD and run a different method than the user asked for. Reject it instead.
    // `auto` (defined to fall back) and `fd` are unaffected.
    #[cfg(not(feature = "autodiff"))]
    if options.gradient_method == crate::types::GradientMethod::Ad {
        diags.push(
            Diagnostic::error(
                "E_AD_UNAVAILABLE",
                "gradient_method = ad was requested, but this build was compiled without the \
                 `autodiff` feature, so automatic differentiation is unavailable — the fit would \
                 silently use finite differences. Rebuild with the Enzyme toolchain \
                 (`--features autodiff`), or set gradient_method = auto (falls back to FD \
                 automatically) or fd.",
            )
            .with_block("fit_options"),
        );
    }

    // IMP is a likelihood evaluation, not an estimator: it must follow a
    // parameter-estimating stage, appear at most once, and be the terminal stage.
    if chain.iter().any(|&m| m == EstimationMethod::Imp) {
        if chain.first().copied() == Some(EstimationMethod::Imp) {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` cannot be the first stage in a chain — it consumes \
                     EBEs and Hessians from a preceding estimator. Try `methods = [focei, imp]` \
                     or `methods = [saem, imp]`.",
                )
                .with_block("fit_options"),
            );
        }
        let n_imp = chain
            .iter()
            .filter(|&&m| m == EstimationMethod::Imp)
            .count();
        if n_imp > 1 {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` may appear at most once in a chain.",
                )
                .with_block("fit_options"),
            );
        }
        if chain.last().copied() != Some(EstimationMethod::Imp) {
            diags.push(
                Diagnostic::error(
                    "E_IMP_CHAIN",
                    "method `imp` must be the final stage of the chain — placing it mid-chain \
                     would leave `FitResult.importance_sampling` populated with a log-likelihood \
                     computed at parameters that the following stage then overwrites. Move `imp` \
                     to the end.",
                )
                .with_block("fit_options"),
            );
        }
    }

    // The trust-region outer optimizer does not thread kappas through its OFV.
    if model.n_kappa > 0 && options.optimizer == Optimizer::TrustRegion {
        diags.push(
            Diagnostic::error(
                "E_OPTIMIZER_IOV",
                "optimizer = trust_region does not support IOV (n_kappa > 0). \
                 Use optimizer = bobyqa, slsqp, lbfgs, nlopt_lbfgs, mma, or bfgs \
                 for models with kappa declarations.",
            )
            .with_block("fit_options"),
        );
    }

    diags
}

/// Data-dependent *warning*-level checks: malformed steady-state rows, EVID=3/4
/// resets under an SDE model, and a negative typical-value lag time. These are
/// non-fatal — `fit()` pushes their messages into `FitResult.warnings` and
/// proceeds; `ferx check` reports them as `Warning` diagnostics. Message text
/// is identical to the historical inline strings.
pub fn check_model_data_warnings(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // SS=1 with II ≤ 0 — the SS branch is gated on `dose.ii > 0`, so the dose
    // is silently treated as a single (non-SS) dose.
    let n_ss_bad_ii = population
        .subjects
        .iter()
        .filter(|s| s.doses.iter().any(|d| d.ss && d.ii <= 0.0))
        .count();
    if n_ss_bad_ii > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_II",
            format!(
                "{} subject(s) have SS=1 doses with missing or non-positive II. \
                 SS predictions require II > 0 — these doses are treated as \
                 non-SS (no steady-state pre-equilibration). Set II in the \
                 dataset or remove the SS flag.",
                n_ss_bad_ii
            ),
        ));
    }

    // SS=1 infusion with T_inf > II — overlapping pulses have no closed form;
    // the SS pre-equilibration is skipped.
    let n_ss_overlapping_inf = population
        .subjects
        .iter()
        .filter(|s| {
            s.doses
                .iter()
                .any(|d| d.ss && d.ii > 0.0 && d.rate > 0.0 && d.duration > d.ii)
        })
        .count();
    if n_ss_overlapping_inf > 0 {
        diags.push(Diagnostic::warning(
            "W_STEADY_STATE_INFUSION",
            format!(
                "{} subject(s) have SS=1 infusions with T_inf > II (overlapping \
                 pulses). No closed form or pulse-expansion scheme covers this \
                 case — the SS pre-equilibration is skipped and the dose is \
                 applied as a single (non-SS) infusion, so the system is not at \
                 steady state at the dose time. Use a shorter infusion (T_inf \
                 ≤ II) or remove the SS flag.",
                n_ss_overlapping_inf
            ),
        ));
    }

    // EVID=3/4 resets are not honoured on the EKF/SDE path.
    if model.is_sde() {
        let n_reset_sde = population
            .subjects
            .iter()
            .filter(|s| s.has_resets())
            .count();
        if n_reset_sde > 0 {
            diags.push(Diagnostic::warning(
                "W_SDE_RESET",
                format!(
                    "{} subject(s) have EVID=3/4 reset rows with a [diffusion] (SDE) \
                     model. System resets are not yet honoured on the EKF/SDE path — \
                     the resets are ignored and compartment amounts carry through. \
                     Use an ODE or analytical model if resets are required.",
                    n_reset_sde
                ),
            ));
        }
    }

    // Negative typical-value lag time at the initial point (eta = 0).
    if model.has_lagtime() {
        if let Some(first_subj) = population.subjects.first() {
            let zero_eta = vec![0.0_f64; model.n_eta];
            let pk = (model.pk_param_fn)(&init_params.theta, &zero_eta, &first_subj.covariates);
            if pk.lagtime() < 0.0 {
                diags.push(Diagnostic::warning(
                    "W_NEGATIVE_LAGTIME",
                    format!(
                        "Lagtime evaluates to {:.4} (< 0) at the initial typical-value \
                         point (eta = 0). Negative lagtimes are physically nonsensical \
                         and are not clamped — consider an exp() or other positive-link \
                         parameterisation.",
                        pk.lagtime()
                    ),
                ));
            }
        }
    }

    diags
}

/// Map a free-text parser error string to a single structured [`Diagnostic`].
/// Recognises the `"Missing [X] block"` shape (→ `E_MISSING_BLOCK`, with the
/// block name attached) and the `--features nn` gate (→ `E_NN_FEATURE_DISABLED`);
/// everything else is a generic `E_PARSE`.
fn parse_error_to_diagnostic(err: &str) -> Diagnostic {
    if let Some(rest) = err.strip_prefix("Missing [") {
        if let Some(end) = rest.find(']') {
            let block = &rest[..end];
            return Diagnostic::error("E_MISSING_BLOCK", err.to_string()).with_block(block);
        }
    }
    if err.contains("[covariate_nn]") && err.contains("--features nn") {
        return Diagnostic::error("E_NN_FEATURE_DISABLED", err.to_string())
            .with_block("covariate_nn");
    }
    Diagnostic::error("E_PARSE", err.to_string())
}

/// Validate a model file (and optionally a dataset) **without fitting**.
///
/// Runs the parser plus every data-independent and data-dependent check,
/// collecting *all* findings into a [`CheckReport`] (rather than stopping at the
/// first, as `fit()` does). This is the engine behind the `ferx check` CLI
/// command and is the fast `author → diagnose → fix` loop for tools and agents.
///
/// When `data_path` is `None`, only parse/structural and model/option
/// compatibility validation runs (no data is read). When it is `Some`, the CSV
/// is read and the covariate / per-CMT / steady-state / lag-time checks run as
/// well.
pub fn validate_model_file(model_path: &str, data_path: Option<&str>) -> CheckReport {
    use crate::parser::model_parser::parse_full_model_file;

    let model_name = Path::new(model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let data = data_path.map(|s| s.to_string());

    // 1. Parse. A parse failure is terminal — without an AST there is nothing
    //    further to validate, so return a report carrying just that diagnostic.
    let parsed = match parse_full_model_file(Path::new(model_path)) {
        Ok(p) => p,
        Err(e) => {
            return CheckReport::new(model_name, data, vec![parse_error_to_diagnostic(&e)]);
        }
    };

    let mut diags: Vec<Diagnostic> = Vec::new();

    // 2. Model / estimation-option compatibility (data-independent): catches
    //    method/model combinations that `fit()` rejects before fitting, so a
    //    clean check and a fit agree. Uses the parsed `[fit_options]`, mirroring
    //    what the CLI fit path (`run_model_with_data`) passes to `fit()`.
    diags.extend(check_model_options(&parsed.model, &parsed.fit_options));

    // 3. Data-dependent checks (only when a dataset is supplied).
    if let Some(path) = data_path {
        let iov_col = parsed.fit_options.iov_column.as_deref();
        match read_nonmem_csv(Path::new(path), None, iov_col) {
            Ok(population) => {
                diags.extend(check_model_data(&parsed.model, &population));
                let init_params = parsed.model.default_params.clone();
                diags.extend(check_model_data_warnings(
                    &parsed.model,
                    &population,
                    &init_params,
                ));
            }
            Err(e) => {
                diags.push(Diagnostic::error(
                    "E_DATA",
                    format!("Failed to read data file '{}': {}", path, e),
                ));
            }
        }
    }

    // 4. Attach block-level line numbers to any diagnostic that named a block.
    for d in &mut diags {
        if d.line.is_none() {
            if let Some(block) = &d.block {
                if let Some(&ln) = parsed.block_lines.get(block) {
                    d.line = Some(ln);
                }
            }
        }
    }

    CheckReport::new(model_name, data, diags)
}

/// High-level fit: model file path + data file path → FitResult
pub fn fit_from_files(
    model_path: &str,
    data_path: &str,
    covariate_columns: Option<&[&str]>,
    options: Option<FitOptions>,
) -> Result<FitResult, String> {
    let mut model = crate::parser::model_parser::parse_model_file(Path::new(model_path))?;
    let population = read_nonmem_csv(Path::new(data_path), covariate_columns, None)?;
    let opts = options.unwrap_or_default();
    model.bloq_method = opts.bloq_method;
    // SDE models cannot use autodiff — force FD.
    model.gradient_method =
        if model.is_sde() && opts.gradient_method != crate::types::GradientMethod::Fd {
            crate::types::GradientMethod::Fd
        } else {
            opts.gradient_method
        };
    let mut result = fit(&model, &population, &model.default_params, &opts)?;
    // Hash inputs post-fit (same pattern as `run_model_with_data`). The
    // model and CSV were already read by `parse_model_file` and
    // `read_nonmem_csv` upstream, so the OS page cache typically serves
    // these reads; failures are non-fatal and just disable the integrity
    // check in `run_sir`.
    result.model_path = Some(model_path.to_string());
    result.data_path = Some(data_path.to_string());
    result.model_hash = crate::io::hash::sha256_file(Path::new(model_path)).ok();
    result.data_hash = crate::io::hash::sha256_file(Path::new(data_path)).ok();
    Ok(result)
}

/// Perturb initial parameters for multi-start optimisation.
///
/// Start 0 always returns the unmodified params. Starts 1..n multiply each
/// log-packed theta by `exp(N(0, sigma))` and shift identity-packed thetas
/// (negative lower bound) by `sigma * N(0,1)`. Omega and sigma are left
/// unchanged — their starting values are typically less important than theta.
fn perturb_init(
    params: &ModelParameters,
    start_idx: usize,
    sigma: f64,
    base_seed: u64,
) -> ModelParameters {
    if start_idx == 0 {
        return params.clone();
    }
    let mut rng = rand::rngs::SmallRng::seed_from_u64(base_seed.wrapping_add(start_idx as u64));
    let normal = Normal::new(0.0_f64, 1.0_f64).expect("normal dist");
    let mut p = params.clone();
    for (i, t) in p.theta.iter_mut().enumerate() {
        let lower = p.theta_lower.get(i).copied().unwrap_or(0.0);
        if theta_packs_log(lower) {
            *t *= (sigma * normal.sample(&mut rng)).exp();
        } else {
            *t += sigma * normal.sample(&mut rng);
        }
        // Clamp to bounds to avoid starting outside the feasible region
        let lo = p.theta_lower.get(i).copied().unwrap_or(f64::NEG_INFINITY);
        let hi = p.theta_upper.get(i).copied().unwrap_or(f64::INFINITY);
        *t = t.clamp(lo, hi);
    }
    p
}

/// Main fit entry point: CompiledModel + Population → FitResult.
///
/// When `options.threads` is `Some(n)`, the fit runs inside a scoped rayon
/// pool of `n` workers, so this setting is per-call (different fits in the
/// same process can use different thread counts). When `None`, rayon's
/// global pool is used (one worker per logical CPU).
pub fn fit(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    // LTBS sanity checks for hand-built `CompiledModel`s. The parser already
    // enforces these for `.ferx` models, but a Rust caller could otherwise set
    // `log_transform = true` together with a proportional/combined error or a
    // per-CMT spec, which would make the likelihood inconsistent (predictions
    // log-wrapped while variance still expects natural-scale `f`). Fail fast.
    if model.log_transform {
        if !matches!(model.error_model, ErrorModel::Additive) {
            return Err(
                "LTBS (`log_transform = true`) requires `error_model = Additive`; \
                 proportional/combined error on the log scale is not supported"
                    .to_string(),
            );
        }
        if !matches!(model.error_spec, ErrorSpec::Single(_)) {
            return Err(
                "LTBS (`log_transform = true`) is not supported with per-CMT \
                 (`ErrorSpec::PerCmt`) error models"
                    .to_string(),
            );
        }
        if model.diffusion_theta_start.is_some() {
            return Err(
                "LTBS (`log_transform = true`) is not supported with an SDE \
                 model (`diffusion_theta_start = Some(_)`)"
                    .to_string(),
            );
        }
    }
    // Data-dependent fatal checks (covariates present, per-CMT scaling and
    // per-CMT error-model coverage). These can't run in the parser — it doesn't
    // see the data. `ferx check` runs the same `check_model_data` to report
    // every finding; here we stop at the first error to preserve fit()'s
    // historical fail-fast behavior and exact error strings.
    first_error(&check_model_data(model, population))?;
    // If any subject has per-event covariate snapshots that don't carry
    // a variation in covariates the model actually references (e.g.
    // DAY / STIME columns in NONMEM-format datasets), clear those
    // snapshots so the downstream prediction path routes through the
    // cheap analytical/no-TV fast path instead of the event-driven
    // path. Bigger wins on SAD-style datasets where every subject has
    // a varying DAY column but no model expression touches DAY.
    // Log-transform-both-sides (LTBS) case 2 (`log(DV) ~ additive`): the data's
    // DV is on the natural scale, so log-transform every observation once here,
    // before any prediction is scored against it. Case 1 (`DV ~ log_additive`,
    // `dv_pre_logged`) leaves the already-log DV untouched. Logging into the
    // owned clone leaves the caller's `Population` (and any `simulate` reuse of
    // it) unmodified, and avoids double-logging on repeated `fit()` calls.
    let needs_dv_log = model.log_transform && !model.dv_pre_logged;
    let mut ltbs_warnings: Vec<String> = Vec::new();
    let pop_pruned: std::borrow::Cow<Population> = {
        let needs_prune = population.subjects.iter().any(|s| {
            !s.dose_covariates.is_empty()
                || !s.obs_covariates.is_empty()
                || !s.pk_only_covariates.is_empty()
        });
        if needs_prune || needs_dv_log {
            let mut p = population.clone();
            if needs_prune {
                p.prune_irrelevant_tv_covariates(&model.referenced_covariates);
            }
            if needs_dv_log {
                let n_nonpos = log_transform_observations(&mut p);
                if n_nonpos > 0 {
                    ltbs_warnings.push(format!(
                        "LTBS (log(DV) ~ ...): {n_nonpos} observation(s) had DV ≤ 0, which \
                         cannot be log-transformed; they were floored to log({LTBS_FLOOR:e}). \
                         Check the data scale, or use `DV ~ log_additive(...)` if DV is \
                         already log-transformed.",
                        LTBS_FLOOR = crate::pk::LTBS_FLOOR,
                    ));
                }
            }
            std::borrow::Cow::Owned(p)
        } else {
            std::borrow::Cow::Borrowed(population)
        }
    };
    let pop_ref: &Population = &*pop_pruned;

    // Single-start fast path (default)
    if options.n_starts <= 1 {
        let res = match options.threads {
            Some(n) if n > 0 => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| format!("failed to build rayon pool with {} threads: {}", n, e))?;
                pool.install(|| fit_inner(model, pop_ref, init_params, options))
            }
            _ => fit_inner(model, pop_ref, init_params, options),
        };
        return res.map(|mut result| {
            result.warnings.splice(0..0, ltbs_warnings);
            rebuild_warnings_structured(&mut result);
            result
        });
    }

    // Multi-start: run n_starts fits in parallel, return the lowest-OFV converged result.
    // `threads` controls per-subject parallelism inside each start; in multi-start mode
    // we let the global rayon pool handle both levels (outer start × inner per-subject).
    // Creating a new ThreadPool per start inside an outer into_par_iter() spawns n_starts
    // independent pools that all compete on the same CPUs, causing oversubscription —
    // so we only honour `threads` when the global pool hasn't been entered yet (single-start
    // path above). Here we always use the global pool for the outer par_iter.
    let base_seed: u64 = options.multi_start_seed.unwrap_or(42);
    let base_saem_seed: u64 = options.saem_seed.unwrap_or(12345);
    let n = options.n_starts;
    let sigma = options.start_sigma;

    // Warn once (before the parallel section) that global_search only runs on start 0.
    let mut pre_warnings: Vec<String> = ltbs_warnings;
    if options.global_search && n > 1 {
        pre_warnings.push(format!(
            "global_search = true with n_starts = {n}: CRS2-LM only runs on start 0 \
             (it ignores the starting point and would override the theta perturbation \
             on starts 1..{n})"
        ));
    }

    let results: Vec<(usize, Result<FitResult, String>)> = (0..n)
        .into_par_iter()
        .map(|k| {
            let init_k = perturb_init(init_params, k, sigma, base_seed);
            // Per-start option overrides for k > 0:
            // - saem_seed: derive from base so each start gets a different MH trajectory.
            //   Start 0 keeps the user's seed for reproducibility of the unperturbed run.
            // - global_search: CRS2-LM ignores the starting point and samples freely in
            //   [lower, upper], so running it on starts 1..n overrides the perturbation
            //   and makes multi-start a no-op for those starts. Only run it on start 0.
            let opts_k_storage;
            let opts_ref: &FitOptions = if k == 0 {
                options
            } else {
                opts_k_storage = FitOptions {
                    saem_seed: Some(base_saem_seed.wrapping_add(k as u64)),
                    global_search: false,
                    ..options.clone()
                };
                &opts_k_storage
            };
            (k, fit_inner(model, pop_ref, &init_k, opts_ref))
        })
        .collect();

    // Pick best converged result; fall back to best unconverged if none converged.
    let mut best: Option<(usize, FitResult)> = None;
    let mut failed_starts: Vec<String> = Vec::new();
    for (k, res) in results {
        match res {
            Ok(r) => {
                let better = match &best {
                    None => true,
                    Some((_, b)) => {
                        // Prefer converged over unconverged; then lower OFV
                        (!b.converged && r.converged)
                            || (b.converged == r.converged && r.ofv < b.ofv)
                    }
                };
                if better {
                    best = Some((k, r));
                }
            }
            Err(e) => failed_starts.push(format!("start {k}: {e}")),
        }
    }

    match best {
        None => Err("All multi-start fits failed".to_string()),
        Some((k, mut result)) => {
            result.warnings.splice(0..0, pre_warnings);
            if !failed_starts.is_empty() {
                result.warnings.push(format!(
                    "Multi-start: {} of {n} starts failed: {}",
                    failed_starts.len(),
                    failed_starts.join("; ")
                ));
            }
            if !result.converged {
                result.warnings.push(format!(
                    "No multi-start run converged ({n} starts); returning best OFV from start {k}"
                ));
            } else if k > 0 {
                result.warnings.push(format!(
                    "Multi-start: best result from start {k}/{n} (OFV = {:.4})",
                    result.ofv
                ));
            }
            rebuild_warnings_structured(&mut result);
            Ok(result)
        }
    }
}

/// Rebuild `warnings_structured` from the current `warnings` vec.
///
/// Called after all late-injected warnings (LTBS splice, multi-start metadata)
/// have been appended so the structured field is always in sync with the flat list.
fn rebuild_warnings_structured(result: &mut FitResult) {
    result.warnings_structured = result
        .warnings
        .iter()
        .map(|w| crate::types::classify_warning(w))
        .collect();
}

/// Probe whether NLopt CRS2-LM (used for global_search) is available.
fn probe_nlopt_algorithms() -> Vec<String> {
    fn dummy_obj(_x: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()) -> f64 {
        0.0
    }
    let available = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            1,
            dummy_obj,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if available.is_err() {
        vec![
            "NLopt CRS2-LM not available in this build — global_search = true will fail. \
             Install a full NLopt build: brew install nlopt / apt install libnlopt-dev"
                .to_string(),
        ]
    } else {
        vec![]
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    let fit_start = Instant::now();
    let chain = options.method_chain();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let unsupported_warnings = options.unsupported_keys_warnings();

    // Capture thread count before chain runs (current_num_threads() reports
    // whichever Rayon pool is active — scoped pool when threads=Some, else global).
    let n_threads_used = rayon::current_num_threads();

    // Initialise the per-iteration optimizer trace if requested.
    if options.optimizer_trace {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("/tmp/ferx_trace_{}_{}.csv", pid, ts);
        if let Err(e) = crate::estimation::trace::init(path.clone()) {
            eprintln!("[ferx] warning: could not open trace file {}: {}", path, e);
        } else {
            eprintln!("[ferx] optimizer trace → {}", path);
        }
    }

    // Reset gradient timing counters for this fit so FERX_TIME_GRADIENTS
    // readouts are per-call rather than cumulative across a long R session.
    let time_gradients = std::env::var("FERX_TIME_GRADIENTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if time_gradients {
        crate::estimation::inner_optimizer::GRADIENT_TIMINGS.reset();
    }
    if options.verbose {
        let chain_str: Vec<&str> = chain.iter().map(|m| m.label()).collect();
        // rayon::current_num_threads() reports whichever pool par_iter would use
        // from the current call — the scoped pool when options.threads is Some,
        // otherwise the global pool. So this stays accurate in both paths.
        let n_threads = rayon::current_num_threads();
        let thread_word = if n_threads == 1 { "thread" } else { "threads" };
        if !unsupported_warnings.is_empty() {
            eprintln!("--- Warnings ---");
            for w in &unsupported_warnings {
                eprintln!("  * {}", w);
            }
            eprintln!();
        }
        eprintln!(
            "Starting estimation (chain: {}) on {} {}...",
            chain_str.join(" → "),
            n_threads,
            thread_word
        );
        eprintln!(
            "  {} subjects, {} observations",
            population.subjects.len(),
            population.n_obs()
        );
        eprintln!(
            "  {} thetas, {} etas, {} sigmas",
            model.n_theta, model.n_eta, model.n_epsilon
        );
        // Report the route each method actually uses. Gradient-based estimators
        // (FOCE/FOCEI/GN) are driven by the inner-loop gradient; IMP consumes
        // the EBE Hessian built via that same route. SAEM is sampling-based, so
        // it reports its E-step kernel (MH/HMC) instead of a gradient route.
        let uses_gradient_route = chain.iter().any(|m| {
            matches!(
                m,
                EstimationMethod::Foce
                    | EstimationMethod::FoceI
                    | EstimationMethod::FoceGn
                    | EstimationMethod::FoceGnHybrid
                    | EstimationMethod::Imp
            )
        });
        if uses_gradient_route {
            eprintln!(
                "  gradient: {}",
                crate::estimation::inner_optimizer::gradient_route_summary(
                    model,
                    population,
                    options.gradient_method,
                )
            );
        }
        if chain.iter().any(|m| *m == EstimationMethod::Saem) {
            eprintln!(
                "  sampler:  {}",
                crate::estimation::saem::saem_sampler_summary(model, options)
            );
        }
    }

    // Model / estimation-option compatibility guards: SDE vs SAEM / GN / AD,
    // IMP chain placement, and trust-region vs IOV. Extracted into
    // `check_model_options` so `ferx check` reports the same incompatibilities;
    // here we stop at the first error to preserve fail-fast behavior and exact
    // error strings. (Per-CMT error models cannot reach the EKF path — the
    // parser rejects Form C `y[CMT=N]` readouts on SDE models — so an SDE model
    // is always single-endpoint here, which the EKF residual-variance
    // assumption in stats/likelihood.rs relies on.)
    first_error(&check_model_options(model, options))?;

    // Pre-compute n_params (uses init_params, available before chain runs).
    let fixed_mask = crate::estimation::parameterization::packed_fixed_mask(init_params);
    let n_params_pre = fixed_mask.iter().filter(|&&b| !b).count();

    // Probe NLopt algorithm availability only when global_search will actually
    // run — otherwise the CRS2-LM warning is misleading for users who never
    // requested it.
    let nlopt_missing = if options.global_search {
        probe_nlopt_algorithms()
    } else {
        Vec::new()
    };

    // Covariance step cost warning: fire before chain so user sees it
    // immediately. Use checked_mul so an absurd parameter count cannot wrap
    // and produce a bogus estimate; on overflow we still warn but suppress
    // the numeric estimate.
    let covariance_n_evals_estimated = if options.run_covariance_step && n_params_pre > 30 {
        n_params_pre.checked_mul(n_params_pre)
    } else {
        None
    };

    // Run each stage in sequence, feeding params forward.
    let n_stages = chain.len();
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = model.parse_warnings.clone();
    accumulated_warnings.extend(unsupported_warnings);

    // Emit NLopt / covariance warnings before any work starts.
    accumulated_warnings.extend(nlopt_missing.iter().cloned());

    // Data-dependent warnings: malformed steady-state rows, EVID=3/4 resets
    // under an SDE model, and a negative typical-value lag time. Extracted into
    // `check_model_data_warnings` so `ferx check` reports the same findings;
    // message text is unchanged. Probed against `population` (not the pruned
    // copy) and `init_params`, matching the historical inline checks.
    for d in check_model_data_warnings(model, population, init_params) {
        accumulated_warnings.push(d.message);
    }
    if options.run_covariance_step && n_params_pre > 30 {
        if let Some(n_evals) = covariance_n_evals_estimated {
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → {} OFV evaluations \
                 (finite-difference Hessian). This may take several minutes \
                 on complex models.",
                n_params_pre, n_evals
            ));
        } else {
            // n_params² overflowed usize — warn without the (wrapped) number.
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → n² OFV evaluations \
                 (finite-difference Hessian). Estimate exceeds usize range; \
                 expect this to be very slow.",
                n_params_pre
            ));
        }
    }

    // inits_from_nca: derive NCA-based starting values before the optimizer
    // loop, using the strategy the user selected (nca / nca_sweep / nca_ebe).
    if let Some(method) = options.inits_from_nca {
        let suggested = crate::suggest_start::inits_from_nca(model, population, method);
        stage_params = suggested.params;
        accumulated_warnings.extend(suggested.warnings);
    }

    let mut total_iterations: usize = 0;
    let mut is_result: Option<ImportanceSamplingResult> = None;

    for (stage_idx, &method) in chain.iter().enumerate() {
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }
        let is_last = stage_idx + 1 == n_stages;
        let mut stage_opts = options.clone();
        stage_opts.method = method;
        stage_opts.methods = Vec::new();
        // Per-stage interaction flag: FOCEI=on, FOCE=off, others inherit from user options.
        match method {
            EstimationMethod::FoceI => stage_opts.interaction = true,
            EstimationMethod::Foce => stage_opts.interaction = false,
            _ => {}
        }
        // Only run the covariance step on the final stage to avoid wasted work.
        if !is_last {
            stage_opts.run_covariance_step = false;
            stage_opts.sir = false;
        }

        if options.verbose && n_stages > 1 {
            eprintln!(
                "\n── Stage {}/{}: {} ──",
                stage_idx + 1,
                n_stages,
                method.label()
            );
        }

        // IMP stage: not an estimator. Consumes the previous stage's params /
        // EBEs / Hessians, writes its result to `is_result`, and skips the
        // params/result update at the bottom of the loop so the preceding
        // stage's `OuterResult` continues to be the canonical one.
        if method == EstimationMethod::Imp {
            let prev = result.as_ref().expect(
                "IMP guard above should have rejected an IMP-first chain — \
                 prior stage's OuterResult must exist here",
            );
            match crate::estimation::importance_sampling::run_importance_sampling(
                model,
                population,
                &prev.params,
                &prev.eta_hats,
                &prev.h_matrices,
                &prev.kappas,
                &stage_opts,
            ) {
                Ok(r) => {
                    // Surface a *separate* warning for any subject whose
                    // ESS-fraction collapsed to zero. These are already in
                    // `low_ess_subjects` (assuming threshold > 0), but
                    // complete proposal collapse is qualitatively distinct
                    // from merely-low ESS — each collapsed subject inflates
                    // the reported MC SE by ~1 unit (see Geweke variance
                    // fallback in `importance_sampling.rs`).
                    let collapsed: Vec<&str> = r
                        .low_ess_subjects
                        .iter()
                        .filter(|(_, f)| *f <= 0.0)
                        .map(|(id, _)| id.as_str())
                        .collect();
                    if !collapsed.is_empty() {
                        let preview = if collapsed.len() <= 5 {
                            collapsed.join(", ")
                        } else {
                            let head = collapsed[..5].join(", ");
                            format!("{} (+{} more)", head, collapsed.len() - 5)
                        };
                        let msg = format!(
                            "IMP: {} subject(s) had ESS = 0 (proposal collapse): {}. \
                             The reported MC SE is inflated by ~1 per collapsed subject; \
                             consider raising `is_samples` or `is_proposal_df`, \
                             or check the EBE/Hessian quality of these subjects.",
                            collapsed.len(),
                            preview
                        );
                        accumulated_warnings.push(if n_stages > 1 {
                            format!("[IMP] {}", msg)
                        } else {
                            msg
                        });
                    }
                    is_result = Some(r);
                }
                Err(e) => {
                    accumulated_warnings.push(if n_stages > 1 {
                        format!("[IMP] {}", e)
                    } else {
                        format!("IMP: {}", e)
                    });
                }
            }
            continue;
        }

        let stage_result = match method {
            EstimationMethod::Saem => {
                saem::run_saem(model, population, &stage_params, &stage_opts)?
            }
            EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
                crate::estimation::gauss_newton::run_foce_gn(
                    model,
                    population,
                    &stage_params,
                    &stage_opts,
                )
            }
            EstimationMethod::Imp => unreachable!("handled by the IMP branch above"),
            _ => optimize_population(model, population, &stage_params, &stage_opts),
        };

        stage_params = stage_result.params.clone();
        total_iterations += stage_result.n_iterations;
        for w in &stage_result.warnings {
            accumulated_warnings.push(if n_stages > 1 {
                format!("[{}] {}", method.label(), w)
            } else {
                w.clone()
            });
        }
        result = Some(stage_result);
    }

    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    let mut result = result.expect("method chain must have at least one stage");
    // Overwrite with chain-aware totals
    result.n_iterations = total_iterations;
    result.warnings = accumulated_warnings;

    // Thread efficiency warnings (post-chain, uses n_threads_used captured above).
    let n_subjects = population.subjects.len();
    if n_subjects > 0 && n_threads_used > n_subjects {
        // `threads = 0` is not a valid Rayon pool size, so for n_subjects = 1
        // we still suggest a 1-thread pool.
        let suggested = n_subjects.max(1);
        result.warnings.push(format!(
            "{} threads configured but only {} subject(s) — consider threads = {} to reduce \
             scheduling overhead (no speed benefit beyond n_subjects)",
            n_threads_used, n_subjects, suggested
        ));
    }
    // SAEM-specific: MH scheduling has higher per-subject overhead than FOCE.
    // Skip when n_subjects < 2 (n_subjects/2 = 0 is meaningless and the prior
    // warning already covers the n_threads > n_subjects case).
    if chain.iter().any(|&m| m == EstimationMethod::Saem) && n_subjects >= 2 {
        let suggested = (n_subjects / 2).max(1);
        if n_threads_used > suggested {
            result.warnings.push(format!(
                "SAEM with more threads than subjects/2 may be slower due to MH scheduling \
                 overhead. Consider threads = {} for SAEM.",
                suggested
            ));
        }
    }

    // Compute per-subject diagnostics
    let subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        &result.kappas,
        options.interaction,
    );

    let n_obs = population.n_obs();
    let n_params = n_params_pre;

    let ofv = result.ofv;
    let aic = ofv + 2.0 * n_params as f64;
    let bic = ofv + n_params as f64 * (n_obs as f64).ln();

    // Extract SEs from covariance matrix using converged parameter values
    let (se_theta, se_omega, se_sigma, se_kappa) =
        extract_standard_errors(&result.covariance_matrix, &result.params);

    // Optional SIR step
    let mut warnings = result.warnings;

    // Report detected mu-referencing relationships (only when feature is enabled)
    if options.mu_referencing && !model.mu_refs.is_empty() {
        let mut names: Vec<&String> = model.mu_refs.keys().collect();
        names.sort();
        warnings.push(format!(
            "mu-ref: {}",
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // When M3 BLOQ is combined with non-interaction FOCE, mixing linearized
    // Gaussian residuals with non-linearized log Φ terms gives inconsistent
    // OFVs near the LLOQ boundary. The FOCE dispatcher routes affected
    // subjects through FOCEI internally — surface the promotion to the user.
    if matches!(model.bloq_method, BloqMethod::M3)
        && matches!(
            options.method,
            EstimationMethod::Foce | EstimationMethod::FoceGn
        )
        && !options.interaction
        && population.subjects.iter().any(|s| s.has_bloq())
    {
        warnings.push(
            "M3 BLOQ handling requires FOCEI semantics; subjects with CENS=1 \
             rows were evaluated with η-interaction. Set method=focei explicitly \
             to silence this notice."
                .to_string(),
        );
    }
    let sir_result = if options.sir && !crate::cancel::is_cancelled(&options.cancel) {
        if let Some(ref cov) = result.covariance_matrix {
            if options.verbose {
                eprintln!("\nRunning SIR...");
            }
            match crate::estimation::sir::run_sir_core(
                model,
                population,
                &result.params,
                &result.eta_hats,
                cov,
                result.ofv,
                options,
            ) {
                Ok(sir) => Some(sir),
                Err(e) => {
                    warnings.push(format!("SIR failed: {}", e));
                    None
                }
            }
        } else {
            warnings.push(
                "SIR requested but covariance matrix is not available. \
                 Enable covariance = true in [fit_options]."
                    .to_string(),
            );
            None
        }
    } else {
        None
    };

    // `final_method` reports the last *estimating* stage — IMP is a likelihood
    // evaluation and doesn't produce parameters, so a chain like `[saem, imp]`
    // surfaces as `method = SAEM`. The full chain (including IMP) is preserved
    // in `method_chain`.
    let final_method = chain
        .iter()
        .rev()
        .copied()
        .find(|&m| m != EstimationMethod::Imp)
        .unwrap_or(*chain.last().expect("chain non-empty"));
    let grad_inner =
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, model);
    let grad_outer = crate::build_info::gradient_method_outer(
        &crate::build_info::BUILD_INFO,
        final_method,
        options.optimizer,
    );

    // Flush and close the trace file; capture path for FitResult.
    let trace_path = crate::estimation::trace::finish();

    // Shrinkage
    let shrinkage_eta = compute_eta_shrinkage(&subjects, &result.params.omega.matrix);
    let shrinkage_eps = compute_eps_shrinkage(&subjects);

    if let Some(w) = eps_shrinkage_warning(shrinkage_eps) {
        warnings.push(w);
    }

    let (iwres_lag1_r, dw_statistic) = iwres_autocorrelation(&subjects);

    // Covariance status
    let covariance_status = if !options.run_covariance_step {
        CovarianceStatus::NotRequested
    } else if result.covariance_matrix.is_some() {
        CovarianceStatus::Computed
    } else {
        CovarianceStatus::Failed
    };

    let wall_time_secs = fit_start.elapsed().as_secs_f64();

    let (cov_eigenvalues, cov_condition_number) =
        cov_diagnostics(result.covariance_matrix.as_ref());

    // Derive per-eta lognormal flags from mu_refs, keyed by eta name.
    // Etas absent from mu_refs (conditional / complex / logit) are treated as
    // additive (false) and a warning is added when they participate in a block.
    let eta_log_transformed: Vec<bool> = result
        .params
        .omega
        .eta_names
        .iter()
        .map(|name| {
            model
                .mu_refs
                .get(name)
                .map(|r| r.log_transformed)
                .unwrap_or(false)
        })
        .collect();

    let omega_param_corr = compute_param_corr(
        &result.params.omega.matrix,
        &eta_log_transformed,
        &result.params.omega.eta_names,
        "omega_param_corr",
        &mut warnings,
    );

    let omega_iov_param_corr = result.params.omega_iov.as_ref().and_then(|iov| {
        let kappa_log: Vec<bool> = model
            .kappa_names
            .iter()
            .map(|name| {
                model
                    .kappa_mu_refs
                    .get(name)
                    .map(|r| r.log_transformed)
                    .unwrap_or(false)
            })
            .collect();
        compute_param_corr(
            &iov.matrix,
            &kappa_log,
            &model.kappa_names,
            "omega_iov_param_corr",
            &mut warnings,
        )
    });

    // DW autocorrelation warnings
    if dw_statistic.is_finite() {
        if dw_statistic < 1.5 {
            let mut msg = format!(
                "Positive IWRES autocorrelation detected (Durbin-Watson = {:.2}). \
                Structural model may be missing dynamics. Consider a transit \
                absorption model, additional compartment, or IOV on ka/F.",
                dw_statistic
            );
            if model.ode_spec.is_some() {
                msg.push_str(" For ODE models, SDE process noise may also help.");
            }
            warnings.push(msg);
        } else if dw_statistic > 2.5 {
            warnings.push(format!(
                "Negative IWRES autocorrelation detected (Durbin-Watson = {:.2}). \
                Possible over-parameterization or misspecified error model.",
                dw_statistic
            ));
        }
    }

    let fit_result = FitResult {
        method: final_method,
        method_chain: chain.clone(),
        converged: result.converged,
        ofv,
        aic,
        bic,
        theta: result.params.theta.clone(),
        theta_names: result.params.theta_names.clone(),
        eta_names: result.params.omega.eta_names.clone(),
        omega: result.params.omega.matrix.clone(),
        sigma: result.params.sigma.values.clone(),
        sigma_names: result.params.sigma.names.clone(),
        error_model: model.error_model,
        covariance_matrix: result.covariance_matrix,
        se_theta,
        se_omega,
        se_sigma,
        theta_fixed: result.params.theta_fixed.clone(),
        omega_fixed: result.params.omega_fixed.clone(),
        sigma_fixed: result.params.sigma_fixed.clone(),
        omega_init_as_sd: model.omega_init_as_sd.clone(),
        sigma_init_as_sd: model.sigma_init_as_sd.clone(),
        subjects,
        n_obs,
        n_subjects: population.subjects.len(),
        n_parameters: n_params,
        n_iterations: result.n_iterations,
        interaction: options.interaction,
        warnings,
        warnings_structured: vec![],
        sir_ci_theta: sir_result.as_ref().map(|s| s.ci_theta.clone()),
        sir_ci_omega: sir_result.as_ref().map(|s| s.ci_omega.clone()),
        sir_ci_sigma: sir_result.as_ref().map(|s| s.ci_sigma.clone()),
        sir_ess: sir_result.as_ref().map(|s| s.effective_sample_size),
        sir_resamples_packed: sir_result.as_ref().and_then(|s| s.resamples_packed.clone()),
        importance_sampling: is_result,
        omega_iov: result.params.omega_iov.as_ref().map(|m| m.matrix.clone()),
        kappa_names: model.kappa_names.clone(),
        kappa_fixed: result.params.kappa_fixed.clone(),
        kappa_init_as_sd: model.kappa_init_as_sd.clone(),
        se_kappa,
        shrinkage_kappa: Vec::new(),
        ebe_kappas: result.kappas.clone(),
        saem_mu_ref_m_step_evals_saved: result.saem_mu_ref_m_step_evals_saved,
        saem_n_subjects_hmc: result.saem_n_subjects_hmc,
        gradient_method_inner: grad_inner.as_str().to_string(),
        gradient_method_outer: grad_outer.as_str().to_string(),
        uses_ode_solver: model.is_ode_based(),
        uses_sde: model.is_sde(),
        n_threads_used,
        nlopt_missing_algorithms: nlopt_missing,
        covariance_n_evals_estimated,
        trace_path,
        ebe_convergence_warnings: result.ebe_convergence_warnings,
        max_unconverged_subjects: result.max_unconverged_subjects,
        total_ebe_fallbacks: result.total_ebe_fallbacks,
        covariance_status,
        shrinkage_eta,
        shrinkage_eps,
        iwres_lag1_r,
        dw_statistic,
        wall_time_secs,
        model_name: model.name.clone(),
        ferx_version: env!("CARGO_PKG_VERSION").to_string(),
        eta_param_info: model.eta_param_info.clone(),
        theta_transform: model.theta_transform.clone(),
        sigma_types: model
            .error_spec
            .sigma_types(result.params.sigma.values.len()),
        cov_eigenvalues,
        cov_condition_number,
        eta_log_transformed,
        omega_param_corr,
        omega_iov_param_corr,
        // Path/hash fields stay None at this layer; `fit_from_files` and the
        // CLI populate them after a successful fit. In-memory `fit()` callers
        // don't have meaningful paths.
        model_path: None,
        data_path: None,
        model_hash: None,
        data_hash: None,
        #[cfg(feature = "nn")]
        neural_networks: build_neural_network_infos(model),
    };

    if time_gradients {
        let (ad_c, ad_n, fd_c, fd_n, jac_ad_c, jac_ad_n, jac_fd_c, jac_fd_n) =
            crate::estimation::inner_optimizer::GRADIENT_TIMINGS.snapshot();
        let ms = |n: u64| (n as f64) / 1_000_000.0;
        let avg_us = |n: u64, c: u64| {
            if c == 0 {
                0.0
            } else {
                (n as f64) / (c as f64) / 1_000.0
            }
        };
        eprintln!("--- Gradient timings (FERX_TIME_GRADIENTS=1) ---");
        eprintln!(
            "  BFGS (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            ad_c,
            ms(ad_n),
            avg_us(ad_n, ad_c)
        );
        eprintln!(
            "  BFGS (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            fd_c,
            ms(fd_n),
            avg_us(fd_n, fd_c)
        );
        eprintln!(
            "  Jac  (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_ad_c,
            ms(jac_ad_n),
            avg_us(jac_ad_n, jac_ad_c)
        );
        eprintln!(
            "  Jac  (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_fd_c,
            ms(jac_fd_n),
            avg_us(jac_fd_n, jac_fd_c)
        );
    }

    Ok(fit_result)
}

/// Eigenvalues and condition number of the correlation matrix of free
/// (non-fixed) parameters.  Fixed parameters have zero diagonal in the
/// covariance matrix and are excluded so that the correlation scaling does not
/// divide by zero and the condition number reflects only the identifiable
/// parameter space.
///
/// Returns `(None, None)` when `cov` is `None` or fewer than two free
/// parameters exist (after excluding parameters whose diagonal entry is
/// `<= 0`).  Parameters with non-positive diagonals are treated as fixed and
/// silently excluded; the remaining free subblock is used for the computation.
/// Threshold below which an off-diagonal omega/kappa entry is treated as
/// structurally zero for correlation reporting.  Matches the threshold used
/// in `io/output.rs` when emitting the `correlation:` field.
const OFFDIAG_EPS: f64 = 1e-15;

/// Compute a parameter-level correlation matrix from an omega/kappa matrix.
///
/// For lognormal pairs uses `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`.
/// For additive pairs uses `ω_ij/√(ω_ii·ω_jj)` (eta-level).
/// Mixed pairs fall back to eta-level and append a warning.
/// Returns `None` when the matrix is diagonal (no off-diagonals above
/// `OFFDIAG_EPS`).
fn compute_param_corr(
    omega: &DMatrix<f64>,
    log_transformed: &[bool],
    names: &[String],
    warn_prefix: &str,
    warnings: &mut Vec<String>,
) -> Option<DMatrix<f64>> {
    let n = omega.nrows();
    debug_assert_eq!(
        log_transformed.len(),
        n,
        "log_transformed must be parallel to omega diagonal (got {} for n={})",
        log_transformed.len(),
        n,
    );
    debug_assert_eq!(
        names.len(),
        n,
        "names must be parallel to omega diagonal (got {} for n={})",
        names.len(),
        n,
    );
    let has_offdiag = (0..n).any(|i| (0..i).any(|j| omega[(i, j)].abs() > OFFDIAG_EPS));
    if !has_offdiag {
        return None;
    }
    let mut corr = DMatrix::identity(n, n);
    for i in 0..n {
        for j in 0..i {
            let cov = omega[(i, j)];
            if cov.abs() <= OFFDIAG_EPS {
                continue;
            }
            let w_ii = omega[(i, i)];
            let w_jj = omega[(j, j)];
            let lt_i = *log_transformed.get(i).unwrap_or(&false);
            let lt_j = *log_transformed.get(j).unwrap_or(&false);
            let c = if lt_i && lt_j {
                let num = cov.exp() - 1.0;
                let den = ((w_ii.exp() - 1.0) * (w_jj.exp() - 1.0)).sqrt();
                if den > 0.0 {
                    num / den
                } else {
                    0.0
                }
            } else if !lt_i && !lt_j {
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            } else {
                let name_i = names.get(i).map(|s| s.as_str()).unwrap_or("?");
                let name_j = names.get(j).map(|s| s.as_str()).unwrap_or("?");
                warnings.push(format!(
                    "{}: {} × {} have mixed lognormal/additive parameterizations; \
                     falling back to eta-level correlation",
                    warn_prefix, name_i, name_j
                ));
                let den = (w_ii * w_jj).sqrt();
                if den > 0.0 {
                    cov / den
                } else {
                    0.0
                }
            };
            corr[(i, j)] = c;
            corr[(j, i)] = c;
        }
    }
    Some(corr)
}

fn cov_diagnostics(cov: Option<&DMatrix<f64>>) -> (Option<Vec<f64>>, Option<f64>) {
    let cov = match cov {
        Some(m) => m,
        None => return (None, None),
    };
    let n = cov.nrows();
    let free: Vec<usize> = (0..n).filter(|&i| cov[(i, i)] > 0.0).collect();
    if free.len() < 2 {
        return (None, None);
    }
    let sub = DMatrix::from_fn(free.len(), free.len(), |a, b| cov[(free[a], free[b])]);
    let std_devs: Vec<f64> = (0..free.len()).map(|a| sub[(a, a)].sqrt()).collect();
    let cor = DMatrix::from_fn(free.len(), free.len(), |a, b| {
        sub[(a, b)] / (std_devs[a] * std_devs[b])
    });
    let eig = cor.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().cloned().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let min_ev = eigenvalues.last().copied().unwrap_or(0.0);
    let max_ev = eigenvalues.first().copied().unwrap_or(0.0);
    let condition_number = if min_ev > 1e-10 {
        max_ev / min_ev
    } else {
        f64::INFINITY
    };
    (Some(eigenvalues), Some(condition_number))
}

/// Compute per-subject diagnostics (IPRED, PRED, IWRES, CWRES)
fn compute_subject_results(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];
            let kappas: &[DVector<f64>] = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };

            // Individual predictions: f(eta_hat), with occasion-specific kappas for IOV.
            // Uses the continuous per-occasion-aware prediction (issue #104) for IOV
            // and the TV-aware dispatcher for everyone else — so the sdtab IPRED/IWRES
            // match the IPRED that drove the FOCEI marginal at fit time.
            //
            // Previously this branch called `model_preds` with a single per-subject
            // `pk_params_ind` from `subject.covariates`, which on TV-covariate data
            // silently took the **non-TV** dose-superposition path while the OFV
            // was being computed on the event-driven path that honours per-event
            // covariate snapshots. Result: sdtab IPRED collapsed to ~0 in the
            // terminal phase for subjects with even mild TV covariates, IWRES
            // exploded, and the EPS-shrinkage warning fired even when the actual
            // fit (and the inner-loop EBE) were fine. Caught on the jasmine peds
            // vancomycin testdata — see `[[focei-laplace-not-sheiner-beal]]`.
            let ipred = if !kappas.is_empty() {
                let kappa_slices: Vec<Vec<f64>> =
                    kappas.iter().map(|k| k.as_slice().to_vec()).collect();
                crate::pk::predict_iov(model, subject, &params.theta, eta.as_slice(), &kappa_slices)
            } else {
                crate::pk::compute_predictions_with_tv(
                    model,
                    subject,
                    &params.theta,
                    eta.as_slice(),
                )
            };

            // Population predictions: f(eta = 0, kappa = 0).
            let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
            let pk_params_pop = (model.pk_param_fn)(&params.theta, &zero_eta, &subject.covariates);
            let pred = model_preds(model, subject, &pk_params_pop, &params.theta, &zero_eta);

            // IWRES (NaN on BLOQ rows — see compute_cwres for CWRES handling).
            let mut iwres = compute_iwres(
                &subject.observations,
                &ipred,
                &subject.obs_cmts,
                &model.error_spec,
                &params.sigma.values,
            );
            for (j, c) in subject.cens.iter().enumerate() {
                if *c != 0 {
                    iwres[j] = f64::NAN;
                }
            }

            // CWRES
            let cwres = compute_cwres(
                subject,
                &ipred,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                &model.error_spec,
            );

            // OFV contribution
            let ofv_i = if !kappas.is_empty() {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when kappas non-empty");
                foce_subject_nll_iov(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                    kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                )
            };

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
                n_obs: subject.observations.len(),
            }
        })
        .collect()
}

/// ETA shrinkage: `1 - sqrt(mean(eta_hat_k^2)) / sqrt(omega_kk)` for each random effect k.
///
/// Uses the uncentered second moment with `n` divisor (NONMEM / PsN / Monolix
/// convention), reflecting the population assumption that `E[eta_k] = 0`. This
/// differs from the centered, unbiased sample variance (n-1 divisor) — for small
/// `n` the unbiased form inflates SD by sqrt(n/(n-1)) and routinely produces
/// spurious negative shrinkage even on well-fit models.
pub(crate) fn compute_eta_shrinkage(subjects: &[SubjectResult], omega: &DMatrix<f64>) -> Vec<f64> {
    let n_eta = omega.nrows();
    let n_subj = subjects.len();
    if n_subj < 2 || n_eta == 0 {
        return vec![f64::NAN; n_eta];
    }
    (0..n_eta)
        .map(|k| {
            let omega_var = omega[(k, k)];
            if omega_var <= 0.0 {
                return f64::NAN;
            }
            let omega_sd = omega_var.sqrt();
            let ms = subjects.iter().map(|s| s.eta[k].powi(2)).sum::<f64>() / n_subj as f64;
            1.0 - ms.sqrt() / omega_sd
        })
        .collect()
}

/// EPS shrinkage: `1 - sqrt(mean(IWRES^2))` across all valid (non-NaN) residuals.
///
/// IWRES has model-imposed mean 0 and variance 1, so the uncentered second
/// moment with `n` divisor is the natural estimator (matches NONMEM).
pub(crate) fn compute_eps_shrinkage(subjects: &[SubjectResult]) -> f64 {
    let vals: Vec<f64> = subjects
        .iter()
        .flat_map(|s| s.iwres.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let ms = vals.iter().map(|v| v.powi(2)).sum::<f64>() / n as f64;
    1.0 - ms.sqrt()
}

/// Threshold below which negative `shrinkage_eps` triggers a warning.
///
/// Small negative values are normal sampling noise around 0 on well-fit models
/// (the NONMEM uncentered estimator has a small downward bias when the sample
/// mean of IWRES is non-zero). Past this threshold the residual error model
/// genuinely fails to absorb the residuals at the EBE etas and the user should
/// see it.
const EPS_SHRINKAGE_WARN_THRESHOLD: f64 = -0.05;

/// Build the user-facing warning for notably-negative EPS shrinkage, or
/// `None` if the value is finite and above the threshold (or NaN).
pub(crate) fn eps_shrinkage_warning(shrinkage_eps: f64) -> Option<String> {
    if !shrinkage_eps.is_finite() || shrinkage_eps >= EPS_SHRINKAGE_WARN_THRESHOLD {
        return None;
    }
    Some(format!(
        "EPS shrinkage is notably negative ({:.1}%): mean(IWRES^2) > 1, \
         which means the residual error model does not absorb the residuals \
         at the final EBE etas. Common causes: SAEM converged to a local \
         optimum with under-fit sigma (try `method = [saem, focei]` to polish \
         with FOCEI, or different starts); model misspecification on a subset \
         of subjects; sigma at a bound. Inspect the IWRES distribution in the \
         sdtab.",
        100.0 * shrinkage_eps
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_subject(eta: Vec<f64>, iwres: Vec<f64>) -> SubjectResult {
        let n = iwres.len();
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::from_vec(eta),
            ipred: vec![0.0; n],
            pred: vec![0.0; n],
            iwres,
            cwres: vec![0.0; n],
            ofv_contribution: 0.0,
            cens: vec![0; n],
            n_obs: n,
        }
    }

    #[test]
    fn test_eta_shrinkage_zero_when_eta_rms_matches_omega_sd() {
        // NONMEM convention: shrinkage = 1 - sqrt(mean(eta^2)) / sqrt(omega).
        // omega = 1.0, eta = [+1, -1] => mean(eta^2) = 1 => shrinkage = 0.
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert_eq!(sh.len(), 1);
        assert!(
            (sh[0]).abs() < 1e-10,
            "expected ~0 shrinkage, got {}",
            sh[0]
        );
    }

    #[test]
    fn test_eta_shrinkage_positive_when_etas_shrunk() {
        // Etas close to zero → shrinkage > 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects: Vec<SubjectResult> = (0..10)
            .map(|_| make_subject(vec![0.01], vec![0.0]))
            .collect();
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0] > 0.5, "expected high shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_uses_n_not_n_minus_1_divisor() {
        // Regression: with the old (centered, n-1) estimator, eta = [+a, -a] gave
        // SD = a*sqrt(2), so omega = 2.0 was needed to land at shrinkage 0.
        // With the NONMEM (uncentered, n) estimator, mean(eta^2) = a^2, so the
        // same eta values + omega = 2.0 must now give a clearly positive value:
        // shrinkage = 1 - 1/sqrt(2) ≈ 0.293.
        let omega = DMatrix::from_diagonal_element(1, 1, 2.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        let expected = 1.0 - 1.0 / 2.0_f64.sqrt();
        assert!(
            (sh[0] - expected).abs() < 1e-10,
            "expected {}, got {}",
            expected,
            sh[0]
        );
    }

    #[test]
    fn test_eta_shrinkage_nan_when_omega_zero() {
        let omega = DMatrix::zeros(1, 1);
        let subjects = vec![
            make_subject(vec![0.1], vec![0.0]),
            make_subject(vec![-0.1], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN when omega=0");
    }

    #[test]
    fn test_eta_shrinkage_nan_when_fewer_than_2_subjects() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![make_subject(vec![0.5], vec![0.0])];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN with only 1 subject");
    }

    #[test]
    fn test_eps_shrinkage_near_zero_for_unit_normal_iwres() {
        // NONMEM convention: shrinkage = 1 - sqrt(mean(IWRES^2)).
        // IWRES = [+1, -1] => mean(IWRES^2) = 1 => shrinkage = 0.
        let subjects = vec![
            make_subject(vec![0.0], vec![1.0]),
            make_subject(vec![0.0], vec![-1.0]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "expected ~0 eps shrinkage, got {}", sh);
    }

    #[test]
    fn test_eps_shrinkage_nan_for_fewer_than_2_residuals() {
        let subjects = vec![make_subject(vec![0.0], vec![0.5])];
        assert!(compute_eps_shrinkage(&subjects).is_nan());
    }

    #[test]
    fn test_eps_shrinkage_negative_when_iwres_inflated() {
        // IWRES with mean(IWRES^2) > 1 must produce a negative value, not be
        // clamped. Matches the SAEM repro on `nmdata_20230216_1.csv` where
        // mean(IWRES^2) ~ 2.45 -> shrinkage ~ -0.566.
        let subjects = vec![
            make_subject(vec![0.0], vec![2.0]),
            make_subject(vec![0.0], vec![-2.0]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!(sh < 0.0, "expected negative shrinkage, got {}", sh);
        assert!((sh - (1.0 - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_eps_shrinkage_warning_emits_below_threshold() {
        let w = eps_shrinkage_warning(-0.10).expect("expected warning");
        assert!(w.contains("mean(IWRES^2) > 1"));
        assert!(w.contains("-10.0%"));
    }

    #[test]
    fn test_eps_shrinkage_warning_silent_above_threshold() {
        // Tiny negatives are noise — no warning.
        assert!(eps_shrinkage_warning(-0.01).is_none());
        // Positive shrinkage — no warning.
        assert!(eps_shrinkage_warning(0.20).is_none());
        // Right at the boundary — no warning (uses `<`).
        assert!(eps_shrinkage_warning(-0.05).is_none());
        // NaN — no warning.
        assert!(eps_shrinkage_warning(f64::NAN).is_none());
    }

    #[test]
    fn test_eps_shrinkage_ignores_nan_iwres() {
        // BLOQ rows have NaN IWRES — they must be filtered out.
        // After filtering, two values with mean(IWRES^2)=1 remain => shrinkage = 0.
        let subjects = vec![
            make_subject(vec![0.0], vec![1.0, f64::NAN]),
            make_subject(vec![0.0], vec![-1.0, f64::NAN]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "NaN IWRES not filtered, got {}", sh);
    }
}

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
fn extract_standard_errors(
    cov: &Option<DMatrix<f64>>,
    template: &ModelParameters,
) -> (
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
) {
    let cov = match cov {
        Some(c) => c,
        None => return (None, None, None, None),
    };

    let n = cov.nrows();
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    // SE on packed scale
    let se_packed: Vec<f64> = (0..n)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Theta: SE on original scale via delta method
    // If x = log(theta), then SE(theta) = theta * SE(x)
    let se_theta: Vec<f64> = (0..n_theta)
        .map(|i| template.theta[i] * se_packed[i])
        .collect();

    // Omega: SE for diagonal variances
    // omega_ii = L_ii^2, so SE(omega_ii) ≈ 2*L_ii * SE(L_ii)
    // L_ii = exp(x_i), SE(L_ii) = L_ii * SE(x_i)
    // SE(omega_ii) = 2 * L_ii^2 * SE(x_i) = 2 * omega_ii * SE(x_i)
    let omega_start = n_theta;
    let se_omega: Vec<f64> = (0..n_eta)
        .map(|i| {
            let idx = if template.omega.diagonal {
                omega_start + i
            } else {
                // L[i,i] in column-major lower-triangle packing (see `pack_params`):
                // packed entries before column j are sum_{k<j} (n_eta - k), so the
                // i-th diagonal sits at i*n_eta - i*(i-1)/2.  The previous formula
                // i*(i+1)/2 + i was row-major and gave the wrong index for n_eta ≥ 3
                // (e.g. picked L[2,0] instead of L[1,1] for n_eta=3).
                omega_start + i * n_eta - i * i.saturating_sub(1) / 2
            };
            if idx < n {
                2.0 * template.omega.matrix[(i, i)] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start
        + if template.omega.diagonal {
            n_eta
        } else {
            n_eta * (n_eta + 1) / 2
        };
    let se_sigma: Vec<f64> = (0..n_sigma)
        .map(|i| {
            let idx = sigma_start + i;
            if idx < n {
                template.sigma.values[i] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // IOV (kappa): SE for diagonal variances of omega_iov.
    //
    // The packed Cholesky layout is column-major (see `pack_params`):
    // L[i,i] sits at offset `i*n - i*(i-1)/2` within the IOV block.
    // Same delta-method approximation as `se_omega`: SE(var_i) ≈ 2 * var_i * SE(log L_ii),
    // which is exact for diagonal IOV and a first-order approximation for block_kappa.
    // Off-diagonal covariance SEs are not currently reported (matches BSV omega).
    let kappa_start = sigma_start + n_sigma;
    let se_kappa: Option<Vec<f64>> = template.omega_iov.as_ref().map(|iov| {
        let n_kappa = iov.dim();
        (0..n_kappa)
            .map(|i| {
                let idx = if iov.diagonal {
                    kappa_start + i
                } else {
                    kappa_start + i * n_kappa - i * (i.saturating_sub(1)) / 2
                };
                if idx < n {
                    2.0 * iov.matrix[(i, i)] * se_packed[idx]
                } else {
                    0.0
                }
            })
            .collect()
    });

    (Some(se_theta), Some(se_omega), Some(se_sigma), se_kappa)
}

/// Simulate observations from a model with given parameters (random seed).
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult> {
    use rand::prelude::*;
    simulate_inner(model, population, params, n_sim, &mut thread_rng())
}

/// Simulate with a fixed seed for reproducibility.
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(model, population, params, n_sim, &mut rng)
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    simulate_inner_with_draw(model, population, params, n_sim, 1, rng)
}

fn simulate_inner_with_draw<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    draw: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        for subject in &population.subjects {
            // Sample eta from N(0, Omega); append zero kappas for IOV models.
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let z_vec = DVector::from_column_slice(&z);
            let eta = &params.omega.chol * z_vec;
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            // Compute individual parameters
            let pk_params = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates);

            // Predict concentrations
            let ipreds = model_preds(model, subject, &pk_params, &params.theta, &eta_slice);

            // Add residual error
            for (j, &ipred) in ipreds.iter().enumerate() {
                let var =
                    model.residual_variance_at(subject.obs_cmts[j], ipred, &params.sigma.values);
                let eps: f64 = rng.sample(normal);
                let dv_sim = ipred + var.sqrt() * eps;

                results.push(SimulationResult {
                    draw,
                    sim: sim_idx + 1,
                    id: subject.id.clone(),
                    time: subject.obs_times[j],
                    ipred,
                    dv_sim,
                });
            }
        }
    }

    results
}

/// Options controlling `simulate_with_uncertainty()`.
#[derive(Debug, Clone)]
pub struct SimulateUncertaintyOptions {
    /// Number of parameter sets to draw from the uncertainty distribution.
    pub n_uncertainty_draws: usize,
    /// Number of eta/eps replicates simulated *per* parameter draw.
    pub n_sim_per_draw: usize,
    /// How to draw the parameter sets — asymptotic MVN or SIR resamples.
    pub method: crate::estimation::uncertainty_samples::UncertaintyMethod,
    /// Optional seed for reproducibility. `None` uses `thread_rng`.
    pub seed: Option<u64>,
}

/// Simulate observations while propagating parameter uncertainty.
///
/// For each of `opts.n_uncertainty_draws` parameter sets drawn from the
/// uncertainty distribution (asymptotic MVN around the ML estimate or stored
/// SIR resamples), simulate `opts.n_sim_per_draw` replicates of every subject
/// — sampling etas from the drawn Omega and epsilons from the drawn Sigma.
///
/// Total rows returned: `n_uncertainty_draws * n_sim_per_draw * n_subjects *
/// n_obs`. Each `SimulationResult` carries the originating `draw` and `sim`
/// indices so downstream code can compute per-time uncertainty bands.
pub fn simulate_with_uncertainty(
    model: &CompiledModel,
    population: &Population,
    fit_result: &FitResult,
    opts: &SimulateUncertaintyOptions,
) -> Result<Vec<SimulationResult>, String> {
    use rand::SeedableRng;

    let mut rng: rand::rngs::StdRng = match opts.seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        // Re-seed StdRng from entropy so simulate-without-seed is still
        // independent across calls but uses a uniform RNG type internally.
        None => rand::rngs::StdRng::from_entropy(),
    };

    let template =
        crate::estimation::uncertainty_samples::fitted_params_from_result(fit_result, model);
    let draws = crate::estimation::uncertainty_samples::draw_parameter_samples(
        fit_result,
        &template,
        opts.n_uncertainty_draws,
        opts.method,
        &mut rng,
    )?;

    // Final size is deterministic, so we can size the buffer once and avoid
    // repeated reallocations for large simulations.
    let total_obs: usize = population.subjects.iter().map(|s| s.obs_times.len()).sum();
    let mut results =
        Vec::with_capacity(opts.n_uncertainty_draws * opts.n_sim_per_draw * total_obs);
    for (k, params) in draws.iter().enumerate() {
        let mut rows = simulate_inner_with_draw(
            model,
            population,
            params,
            opts.n_sim_per_draw,
            k + 1,
            &mut rng,
        );
        results.append(&mut rows);
    }
    Ok(results)
}

/// A single simulated observation.
///
/// `draw` is the uncertainty draw index (1-based). For `simulate()` /
/// `simulate_with_seed()`, which use point-estimate parameters, `draw` is
/// always `1`. For `simulate_with_uncertainty()` it spans
/// `1..=n_uncertainty_draws`. `sim` is the replicate index *within* a draw.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub draw: usize,
    pub sim: usize,
    pub id: String,
    pub time: f64,
    pub ipred: f64,
    pub dv_sim: f64,
}

/// Predict concentrations for a population using given parameters (no random effects).
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult> {
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();

    for subject in &population.subjects {
        let pk_params = (model.pk_param_fn)(&params.theta, &zero_eta, &subject.covariates);
        let preds = model_preds(model, subject, &pk_params, &params.theta, &zero_eta);

        for (j, &pred) in preds.iter().enumerate() {
            results.push(PredictionResult {
                id: subject.id.clone(),
                time: subject.obs_times[j],
                pred,
            });
        }
    }

    results
}

/// A single prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci` (no autodiff required).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod iov_integration {
    use super::fit;
    use crate::types::*;

    use std::collections::HashMap;

    // ── Model ────────────────────────────────────────────────────────────────
    fn make_iov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
        };
        CompiledModel {
            name: "iov_test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            // pk_param_fn: eta[0]=BSV for CL, eta[1]=KAPPA_CL (appended by IOV path)
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL = TVCL * exp(ETA_CL + KAPPA_CL)
                p.values[1] = theta[1]; // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: vec![false],
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        }
    }

    // ── Population ───────────────────────────────────────────────────────────
    // 4 subjects, each with 2 occasions.  Times 1–3 = occasion 1, times 4–6 = occ 2.
    // Observations are plausible IV-bolus concentrations (dose=100).
    fn make_iov_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let dose_occ = vec![1u32, 2]; // two dose rows: one per occasion
        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0]),
            ("S2", vec![40.0, 32.0, 24.0, 38.0, 29.0, 22.0]),
            ("S3", vec![33.0, 25.0, 19.0, 31.0, 24.0, 18.0]),
            ("S4", vec![42.0, 33.0, 25.0, 39.0, 30.0, 23.0]),
        ];

        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 6],
                occasions: occasions.clone(),
                dose_occasions: dose_occ.clone(),
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    fn fast_opts(
        method: EstimationMethod,
        optimizer: Optimizer,
        mu_referencing: bool,
    ) -> FitOptions {
        FitOptions {
            method,
            methods: Vec::new(),
            outer_maxiter: 60,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: method == EstimationMethod::FoceI,
            mu_referencing,
            optimizer,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────
    fn assert_iov_fit_ok(result: &FitResult) {
        assert!(result.ofv.is_finite(), "OFV must be finite");
        assert!(result.omega_iov.is_some(), "omega_iov must be populated");
        let iov_diag = result.omega_iov.as_ref().unwrap()[(0, 0)];
        assert!(
            iov_diag > 0.0,
            "omega_iov diagonal must be positive, got {iov_diag}"
        );
        assert_eq!(result.kappa_names, vec!["KAPPA_CL"], "kappa name mismatch");
        assert_eq!(result.ebe_kappas.len(), 4, "expected kappas for 4 subjects");
        for (i, subj_kappas) in result.ebe_kappas.iter().enumerate() {
            assert_eq!(subj_kappas.len(), 2, "subject {i} should have 2 occasions");
        }
    }

    // ── Tests: FOCE + all outer optimizers ───────────────────────────────────

    #[test]
    fn test_iov_foce_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_slsqp() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Slsqp, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Lbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_nlopt_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::NloptLbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_mma() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Mma, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_bfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: FOCEI ─────────────────────────────────────────────────────────

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_focei_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: mu-referencing ─────────────────────────────────────────────────

    #[test]
    fn test_iov_foce_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts)
            .expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_focei_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts)
            .expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: GN and GN_Hybrid ───────────────────────────────────────────────

    #[test]
    fn test_iov_gn() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("GN fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_iov_gn_hybrid() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
        let result =
            fit(&model, &pop, &model.default_params, &opts).expect("GN hybrid fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Test: SAEM + IOV now supported (Step 11) ─────────────────────────────

    #[test]
    fn test_iov_saem_succeeds() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_ok(),
            "SAEM with IOV must succeed, got: {:?}",
            result.err()
        );
        let fr = result.unwrap();
        assert!(
            fr.ofv.is_finite(),
            "SAEM IOV OFV must be finite, got {}",
            fr.ofv
        );
        assert!(
            fr.omega_iov.is_some(),
            "omega_iov must be present in result"
        );
    }

    // ── Test: SAEM in a chained methods sequence + IOV succeeds ──────────────
    #[test]
    fn test_iov_saem_in_methods_chain_succeeds() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        opts.methods = vec![EstimationMethod::Saem, EstimationMethod::Foce];
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_ok(),
            "SAEM in methods chain with IOV must succeed, got: {:?}",
            result.err()
        );
        let fr = result.unwrap();
        assert!(
            fr.ofv.is_finite(),
            "chained SAEM+FOCE IOV OFV must be finite"
        );
        assert!(
            fr.omega_iov.is_some(),
            "omega_iov must survive the FOCE polishing step in a chained run"
        );
    }

    // ── Test: trust-region optimizer + IOV must return Err ────────────────────
    // trust_region.rs currently passes `&[]` for kappas to pop_nll, which would
    // silently route the OFV through the non-IOV path. Guard at api.rs blocks
    // that before any wrong number escapes.
    #[test]
    fn test_iov_trust_region_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(
            result.is_err(),
            "trust_region with IOV must return an error"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("trust_region") && msg.contains("IOV"),
            "error message should mention trust_region and IOV, got: {msg}"
        );
    }

    // `ferx check` must surface the same trust_region+IOV incompatibility that
    // `fit()` rejects — without it, a model could report `valid: true` and then
    // fail at fit time. `check_model_options` is the shared source of truth.
    #[test]
    fn test_check_model_options_flags_trust_region_iov() {
        let model = make_iov_model();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::TrustRegion, false);
        let diags = super::check_model_options(&model, &opts);
        let d = diags
            .iter()
            .find(|d| d.code == "E_OPTIMIZER_IOV")
            .expect("expected E_OPTIMIZER_IOV diagnostic");
        // Same wording fit() produces (regression against the extracted guard).
        assert!(d.message.contains("trust_region") && d.message.contains("IOV"));

        // A compatible optimizer produces no compatibility diagnostics.
        let ok_opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        assert!(super::check_model_options(&model, &ok_opts).is_empty());
    }

    // On a build without the `autodiff` feature, explicitly requesting AD must
    // error rather than silently running FD. `auto`/`fd` must still pass.
    #[cfg(not(feature = "autodiff"))]
    #[test]
    fn ad_requested_without_autodiff_feature_errors() {
        let model = make_iov_model();
        let mut opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);

        opts.gradient_method = crate::types::GradientMethod::Ad;
        let diags = super::check_model_options(&model, &opts);
        assert!(
            diags
                .iter()
                .any(|d| d.code == "E_AD_UNAVAILABLE" && d.is_error()),
            "explicit gradient_method=ad on a non-autodiff build must error, got: {diags:?}"
        );

        for gm in [
            crate::types::GradientMethod::Auto,
            crate::types::GradientMethod::Fd,
        ] {
            opts.gradient_method = gm;
            assert!(
                !super::check_model_options(&model, &opts)
                    .iter()
                    .any(|d| d.code == "E_AD_UNAVAILABLE"),
                "gradient_method={gm:?} must not trigger E_AD_UNAVAILABLE"
            );
        }
    }
}

#[cfg(test)]
mod extract_se_tests {
    use super::extract_standard_errors;
    use crate::types::*;
    use nalgebra::DMatrix;

    /// Helper for tests where the BSV omega is a single diagonal eta and only
    /// the IOV block varies. BSV-omega tests below build their own template
    /// inline because they need a 3-eta block omega.
    fn make_template(omega_iov: Option<OmegaMatrix>, kappa_fixed: Vec<bool>) -> ModelParameters {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov,
            kappa_fixed,
        }
    }

    // ── BSV omega ────────────────────────────────────────────────────────────

    /// Block omega with n_eta = 3.  The packed Cholesky layout is column-major
    /// (see `pack_params`): L[0,0]=0, L[1,0]=1, L[2,0]=2, L[1,1]=3, L[2,1]=4,
    /// L[2,2]=5.  The previous index formula `i*(i+1)/2 + i` was row-major and
    /// returned offsets 0, 2, 5 — picking L[2,0] for the L[1,1] slot.
    /// For n_eta ≤ 2 the row- and column-major formulas coincide, which is why
    /// this regressed silently.
    #[test]
    fn test_se_omega_block_n3_uses_column_major_indexing() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let _chol = mat.clone().cholesky().unwrap().l();
        let omega =
            OmegaMatrix::from_matrix(mat, vec!["E1".into(), "E2".into(), "E3".into()], false);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 3],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_block(6) + sigma(1) = 8.
        // Within the omega block (start = 1): L[0,0] at idx 1, L[1,1] at idx 4,
        // L[2,2] at idx 6.  Use distinct cov diagonals so we can tell which one
        // each SE pulls from.
        let n = 8;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
        }
        let (_, se_omega, _, _) = extract_standard_errors(&Some(cov), &template);
        let se = se_omega.unwrap();
        // L[0,0] at packed idx 1 → se_packed = 2 → se_omega[0] = 2 * 0.04 * 2 = 0.16
        assert!((se[0] - 2.0 * 0.04 * 2.0).abs() < 1e-12, "got {}", se[0]);
        // L[1,1] at packed idx 4 → se_packed = 5 → se_omega[1] = 2 * 0.09 * 5 = 0.90
        // Pre-fix this would have used idx 3 (= L[2,0]) → 2 * 0.09 * 4 = 0.72.
        assert!((se[1] - 2.0 * 0.09 * 5.0).abs() < 1e-12, "got {}", se[1]);
        // L[2,2] at packed idx 6 → se_packed = 7 → se_omega[2] = 2 * 0.16 * 7 = 2.24
        assert!((se[2] - 2.0 * 0.16 * 7.0).abs() < 1e-12, "got {}", se[2]);
    }

    /// Diagonal omega path is unaffected by the fix; this guards the simple case.
    #[test]
    fn test_se_omega_diagonal_unchanged() {
        let omega = OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["E1".into(), "E2".into()]);
        let template = ModelParameters {
            theta: vec![5.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.1],
            theta_upper: vec![50.0],
            theta_fixed: vec![false],
            omega,
            omega_fixed: vec![false; 2],
            sigma: SigmaVector {
                values: vec![0.05],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: vec![],
        };
        // Packed layout: theta(1) + omega_diag(2) + sigma(1) = 4. Identity cov.
        let cov = Some(DMatrix::<f64>::identity(4, 4));
        let (_, se_omega, _, _) = extract_standard_errors(&cov, &template);
        let se = se_omega.unwrap();
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    // ── IOV (kappa) ──────────────────────────────────────────────────────────

    /// Identity covariance matrix gives unit SEs on the packed scale.  After
    /// the delta-method transform `SE(var_i) = 2 * var_i * SE(log L_ii)`, the
    /// returned se_kappa[i] equals 2 * variance_i for diagonal IOV.
    #[test]
    fn test_se_kappa_diagonal_uses_correct_index() {
        let iov =
            OmegaMatrix::from_diagonal(&[0.04, 0.09], vec!["KAPPA_CL".into(), "KAPPA_V".into()]);
        let template = make_template(Some(iov), vec![false, false]);
        // Packed layout: [theta, omega(1), sigma, kappa_1, kappa_2] → 5 entries.
        // Identity cov means se_packed = [1, 1, 1, 1, 1].
        let cov = Some(DMatrix::<f64>::identity(5, 5));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        let se = se_kappa.expect("se_kappa should be Some when omega_iov is set");
        assert_eq!(se.len(), 2);
        assert!((se[0] - 2.0 * 0.04).abs() < 1e-12);
        assert!((se[1] - 2.0 * 0.09).abs() < 1e-12);
    }

    /// Block kappa with n=3: column-major Cholesky packing places L[1,1] at
    /// offset 3 and L[2,2] at offset 5 within the IOV block.  Mirrors the BSV
    /// regression test above.
    #[test]
    fn test_se_kappa_block_n3_column_major_indexing() {
        let mut mat = DMatrix::<f64>::zeros(3, 3);
        mat[(0, 0)] = 0.04;
        mat[(1, 1)] = 0.09;
        mat[(2, 2)] = 0.16;
        let _chol = mat.clone().cholesky().unwrap().l();
        let iov = OmegaMatrix::from_matrix(mat, vec!["K1".into(), "K2".into(), "K3".into()], false);
        let template = make_template(Some(iov), vec![false; 3]);
        // Packed layout: theta(1) + omega_diag(1) + sigma(1) + kappa_block(6) = 9
        // Within the kappa block (start = 3): L11=0, L21=1, L31=2, L22=3, L32=4, L33=5
        // → diagonals at packed indices 3, 6, 8.
        // Build a cov matrix with distinct diagonal entries so we can verify
        // which index each SE pulls from.
        let n = 9;
        let mut cov = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            cov[(i, i)] = ((i + 1) as f64).powi(2); // se_packed[i] = i + 1
        }
        let (_, _, _, se_kappa) = extract_standard_errors(&Some(cov), &template);
        let se = se_kappa.unwrap();
        // L[0,0] at idx 3 → se_packed = 4 → se_kappa[0] = 2 * 0.04 * 4 = 0.32
        assert!((se[0] - 2.0 * 0.04 * 4.0).abs() < 1e-12, "got {}", se[0]);
        // L[1,1] at idx 6 → se_packed = 7 → se_kappa[1] = 2 * 0.09 * 7 = 1.26
        assert!((se[1] - 2.0 * 0.09 * 7.0).abs() < 1e-12, "got {}", se[1]);
        // L[2,2] at idx 8 → se_packed = 9 → se_kappa[2] = 2 * 0.16 * 9 = 2.88
        assert!((se[2] - 2.0 * 0.16 * 9.0).abs() < 1e-12, "got {}", se[2]);
    }

    #[test]
    fn test_se_kappa_none_when_no_iov() {
        let template = make_template(None, vec![]);
        let cov = Some(DMatrix::<f64>::identity(3, 3));
        let (_, _, _, se_kappa) = extract_standard_errors(&cov, &template);
        assert!(se_kappa.is_none());
    }

    #[test]
    fn test_se_kappa_none_when_no_cov() {
        let iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let template = make_template(Some(iov), vec![false]);
        let (_, _, _, se_kappa) = extract_standard_errors(&None, &template);
        assert!(se_kappa.is_none());
    }
}

#[cfg(test)]
mod tests_cov_diagnostics {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_cov_diagnostics_none_input() {
        let (ev, cn) = cov_diagnostics(None);
        assert!(ev.is_none());
        assert!(cn.is_none());
    }

    #[test]
    fn test_cov_diagnostics_fewer_than_two_free_params() {
        // 2×2 matrix where only one param is free (second has zero diagonal)
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 4.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        assert!(ev.is_none());
        assert!(cn.is_none());
    }

    #[test]
    fn test_cov_diagnostics_excludes_fixed_params_zero_diagonal() {
        // 3×3 covariance; middle param is fixed (zero row/col).
        // Free subblock [[4, 0.5], [0.5, 2]] is non-singular, so condition
        // number must be finite and eigenvalues length must be 2.
        let mut m = DMatrix::<f64>::zeros(3, 3);
        m[(0, 0)] = 4.0;
        m[(0, 2)] = 0.5;
        m[(2, 0)] = 0.5;
        m[(2, 2)] = 2.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        let ev = ev.expect("eigenvalues must be Some");
        let cn = cn.expect("condition_number must be Some");
        assert_eq!(ev.len(), 2, "must have 2 eigenvalues (one per free param)");
        assert!(
            cn.is_finite(),
            "condition_number must be finite for non-singular subblock"
        );
        assert!(cn > 0.0);
        // Eigenvalues must be sorted descending
        assert!(ev[0] >= ev[1]);
    }

    #[test]
    fn test_cov_diagnostics_inf_condition_number_for_non_positive_eigenvalue() {
        // Construct a 2×2 covariance matrix whose free-param correlation matrix
        // is [[1, r], [r, 1]] with |r| > 1 — not PSD, so min eigenvalue < 0.
        // (r = 1.5 → eigenvalues 2.5 and -0.5)
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.5; // cor = 1.5/sqrt(1*1) = 1.5 > 1 → non-PSD
        m[(1, 0)] = 1.5;
        m[(1, 1)] = 1.0;
        let (ev, cn) = cov_diagnostics(Some(&m));
        let cn = cn.expect("condition_number must be Some");
        assert!(
            cn.is_infinite(),
            "condition_number must be Inf when min eigenvalue ≤ 0, got {cn}"
        );
        let ev = ev.expect("eigenvalues must be Some");
        assert!(
            ev.last().copied().unwrap_or(1.0) <= 0.0,
            "min eigenvalue must be ≤ 0"
        );
    }

    #[test]
    fn test_cov_diagnostics_inf_condition_number_for_near_zero_eigenvalue() {
        // Simulate a floating-point near-zero negative eigenvalue (e.g. -1e-15)
        // that a well-conditioned matrix could produce due to numerical noise.
        // The tolerance guard (> 1e-10) must treat this as singular → INFINITY.
        let mut m = DMatrix::<f64>::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 1.0 - 1e-15; // cor ≈ 1 → min eigenvalue ≈ 0 (or tiny negative)
        m[(1, 0)] = 1.0 - 1e-15;
        m[(1, 1)] = 1.0;
        let (_, cn) = cov_diagnostics(Some(&m));
        let cn = cn.expect("condition_number must be Some");
        assert!(
            cn.is_infinite(),
            "condition_number must be Inf for near-singular matrix (min_ev ≤ 1e-10), got {cn}"
        );
    }

    #[test]
    fn test_cov_diagnostics_identity_covariance() {
        // Diagonal covariance → correlation matrix is identity → all eigenvalues 1.
        let m = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_vec(vec![4.0, 9.0]));
        let (ev, cn) = cov_diagnostics(Some(&m));
        let ev = ev.expect("eigenvalues must be Some");
        let cn = cn.expect("condition_number must be Some");
        for &e in &ev {
            assert!((e - 1.0).abs() < 1e-12, "eigenvalue must be 1.0, got {e}");
        }
        assert!(
            (cn - 1.0).abs() < 1e-12,
            "condition_number must be 1.0, got {cn}"
        );
    }
}

#[cfg(test)]
mod tests_param_corr {
    use super::compute_param_corr;
    use nalgebra::DMatrix;

    fn names(ns: &[&str]) -> Vec<String> {
        ns.iter().map(|s| s.to_string()).collect()
    }

    /// Lognormal pair: uses the bivariate lognormal formula.
    #[test]
    fn lognormal_pair() {
        // ω = [[0.09, 0.045], [0.045, 0.09]]
        let w11 = 0.09_f64;
        let w12 = 0.045_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[true, true],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some for block omega");

        assert!(warnings.is_empty());
        // diagonal must be 1
        assert!((corr[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-12);
        // lognormal formula: (exp(w12) - 1) / sqrt((exp(w11)-1)*(exp(w11)-1))
        let expected = (w12.exp() - 1.0) / (w11.exp() - 1.0);
        assert!(
            (corr[(0, 1)] - expected).abs() < 1e-10,
            "lognormal corr {:.6} != expected {:.6}",
            corr[(0, 1)],
            expected
        );
    }

    /// Additive pair: falls back to eta-level formula (cov/sqrt(var_i*var_j)).
    #[test]
    fn additive_pair() {
        let w11 = 4.0_f64;
        let w12 = 1.0_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[false, false],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some");

        assert!(warnings.is_empty());
        let expected = w12 / w11;
        assert!((corr[(0, 1)] - expected).abs() < 1e-12);
    }

    /// Mixed pair (one lognormal, one additive) falls back to eta-level and emits a warning.
    #[test]
    fn mixed_pair_warns_and_falls_back() {
        let w11 = 0.09_f64;
        let w12 = 0.03_f64;
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = w11;
        omega[(1, 1)] = w11;
        omega[(0, 1)] = w12;
        omega[(1, 0)] = w12;

        let mut warnings = Vec::new();
        let corr = compute_param_corr(
            &omega,
            &[true, false],
            &names(&["ETA_CL", "ETA_V"]),
            "test",
            &mut warnings,
        )
        .expect("should return Some");

        assert_eq!(warnings.len(), 1, "expected one warning");
        assert!(warnings[0].contains("mixed"));
        // eta-level fallback
        let expected = w12 / w11;
        assert!((corr[(0, 1)] - expected).abs() < 1e-12);
    }

    /// Diagonal omega returns None (no off-diagonals to report).
    #[test]
    fn diagonal_returns_none() {
        let mut omega = DMatrix::zeros(2, 2);
        omega[(0, 0)] = 0.09;
        omega[(1, 1)] = 0.04;
        let mut warnings = Vec::new();
        let result = compute_param_corr(
            &omega,
            &[true, true],
            &names(&["A", "B"]),
            "test",
            &mut warnings,
        );
        assert!(result.is_none());
        assert!(warnings.is_empty());
    }
}

#[cfg(test)]
mod simulate_with_uncertainty_tests {
    //! End-to-end smoke tests for `simulate_with_uncertainty`. The parameter
    //! sampler itself is exercised in `estimation::uncertainty_samples::tests`;
    //! these tests verify the wiring: row count, draw index range, and SIR
    //! pool reuse.

    use super::*;
    use crate::estimation::uncertainty_samples::UncertaintyMethod;
    use nalgebra::DMatrix;
    use std::collections::HashMap;

    fn tiny_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        CompiledModel {
            name: "uncertainty_smoke".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                p.values[0] = theta[0] * eta[0].exp();
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params,
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        }
    }

    fn tiny_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0];
        let subjects: Vec<Subject> = (0..2)
            .map(|i| Subject {
                id: format!("S{}", i + 1),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                observations: vec![30.0, 22.0, 16.0],
                obs_cmts: vec![1, 1, 1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0, 0, 0],
                occasions: vec![1, 1, 1],
                dose_occasions: vec![1],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    /// Build a synthetic `FitResult` carrying the fitted theta/Omega/Sigma
    /// from `template` plus a small identity covariance in packed log-space.
    /// Avoids invoking `fit()` (slow) while still exercising the full
    /// `simulate_with_uncertainty` wiring.
    fn synthetic_fit(template: &ModelParameters) -> FitResult {
        let n_packed = crate::estimation::parameterization::packed_len(template);
        let cov = DMatrix::identity(n_packed, n_packed) * 0.01;
        FitResult {
            method: EstimationMethod::FoceI,
            method_chain: vec![EstimationMethod::FoceI],
            converged: true,
            ofv: 0.0,
            aic: 0.0,
            bic: 0.0,
            theta: template.theta.clone(),
            theta_names: template.theta_names.clone(),
            eta_names: template.omega.eta_names.clone(),
            omega: template.omega.matrix.clone(),
            sigma: template.sigma.values.clone(),
            sigma_names: template.sigma.names.clone(),
            error_model: ErrorModel::Proportional,
            covariance_matrix: Some(cov),
            se_theta: None,
            se_omega: None,
            se_sigma: None,
            theta_fixed: template.theta_fixed.clone(),
            omega_fixed: template.omega_fixed.clone(),
            sigma_fixed: template.sigma_fixed.clone(),
            omega_init_as_sd: vec![false; template.omega.matrix.nrows()],
            sigma_init_as_sd: vec![false; template.sigma.values.len()],
            subjects: vec![],
            n_obs: 6,
            n_subjects: 2,
            n_parameters: n_packed,
            n_iterations: 0,
            interaction: true,
            warnings: vec![],
            warnings_structured: vec![],
            sir_ci_theta: None,
            sir_ci_omega: None,
            sir_ci_sigma: None,
            sir_ess: None,
            sir_resamples_packed: None,
            importance_sampling: None,
            omega_iov: None,
            kappa_names: vec![],
            kappa_fixed: vec![],
            kappa_init_as_sd: vec![],
            se_kappa: None,
            shrinkage_kappa: vec![],
            ebe_kappas: vec![],
            saem_mu_ref_m_step_evals_saved: None,
            saem_n_subjects_hmc: None,
            gradient_method_inner: String::new(),
            gradient_method_outer: String::new(),
            uses_ode_solver: false,
            uses_sde: false,
            n_threads_used: 1,
            nlopt_missing_algorithms: vec![],
            covariance_n_evals_estimated: None,
            trace_path: None,
            ebe_convergence_warnings: 0,
            max_unconverged_subjects: 0,
            total_ebe_fallbacks: 0,
            covariance_status: CovarianceStatus::Computed,
            shrinkage_eta: vec![],
            shrinkage_eps: f64::NAN,
            iwres_lag1_r: f64::NAN,
            dw_statistic: f64::NAN,
            wall_time_secs: 0.0,
            model_name: String::new(),
            ferx_version: String::new(),
            eta_param_info: vec![],
            theta_transform: vec![],
            sigma_types: vec![],
            cov_eigenvalues: None,
            cov_condition_number: None,
            eta_log_transformed: vec![],
            omega_param_corr: None,
            omega_iov_param_corr: None,
            model_path: None,
            data_path: None,
            model_hash: None,
            data_hash: None,
            #[cfg(feature = "nn")]
            neural_networks: Vec::new(),
        }
    }

    #[test]
    fn asymptotic_row_count_and_draw_range() {
        let model = tiny_model();
        let pop = tiny_population();
        let fit = synthetic_fit(&model.default_params);

        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 3,
            n_sim_per_draw: 2,
            method: UncertaintyMethod::Asymptotic,
            seed: Some(7),
        };
        let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
        // 3 draws * 2 sims * 2 subjects * 3 obs = 36 rows
        assert_eq!(rows.len(), 36);

        let mut draws: Vec<usize> = rows.iter().map(|r| r.draw).collect();
        draws.sort();
        draws.dedup();
        assert_eq!(draws, vec![1, 2, 3]);

        let mut sims: Vec<usize> = rows.iter().map(|r| r.sim).collect();
        sims.sort();
        sims.dedup();
        assert_eq!(sims, vec![1, 2]);
    }

    #[test]
    fn legacy_simulate_emits_draw_one() {
        // The original simulate() path should tag every row with draw = 1,
        // preserving a sensible default for callers that don't propagate
        // parameter uncertainty.
        let model = tiny_model();
        let pop = tiny_population();
        let rows = simulate_with_seed(&model, &pop, &model.default_params, 2, 42);
        assert!(rows.iter().all(|r| r.draw == 1));
    }

    #[test]
    fn fit_rejects_ltbs_with_proportional_error() {
        // Defensive guard against hand-built `CompiledModel`s: the parser already
        // rejects this combination, but a Rust caller flipping `log_transform = true`
        // on a model with proportional/combined error would silently mis-fit (the
        // prediction is log-wrapped but the variance still expects natural-scale f).
        let mut model = tiny_model(); // tiny_model uses Proportional error
        model.log_transform = true;
        let pop = tiny_population();
        let opts = FitOptions::default();
        let err = fit(&model, &pop, &model.default_params, &opts).unwrap_err();
        assert!(
            err.contains("LTBS") && err.contains("Additive"),
            "expected LTBS+proportional rejection, got: {err}"
        );
    }

    #[test]
    fn sir_path_reuses_pool() {
        let model = tiny_model();
        let pop = tiny_population();
        let mut fit = synthetic_fit(&model.default_params);
        // Build a 4-element resample pool: small perturbations of x_hat in
        // packed log-space. Tests will sample with replacement from this pool.
        let x_hat = crate::estimation::parameterization::pack_params(&model.default_params);
        let pool: Vec<Vec<f64>> = (0..4)
            .map(|k| {
                let mut xk = x_hat.clone();
                xk[0] += 0.005 * (k as f64);
                xk
            })
            .collect();
        fit.sir_resamples_packed = Some(pool);

        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 5,
            n_sim_per_draw: 1,
            method: UncertaintyMethod::Sir,
            seed: Some(11),
        };
        let rows = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap();
        // 5 draws * 1 sim * 2 subjects * 3 obs = 30 rows
        assert_eq!(rows.len(), 30);
    }

    #[test]
    fn asymptotic_errors_without_covariance_step() {
        let model = tiny_model();
        let pop = tiny_population();
        let mut fit = synthetic_fit(&model.default_params);
        fit.covariance_matrix = None;
        let opts = SimulateUncertaintyOptions {
            n_uncertainty_draws: 2,
            n_sim_per_draw: 1,
            method: UncertaintyMethod::Asymptotic,
            seed: Some(0),
        };
        let err = simulate_with_uncertainty(&model, &pop, &fit, &opts).unwrap_err();
        assert!(err.contains("covariance"));
    }
}

// ── SDE end-to-end integration ───────────────────────────────────────────────

#[cfg(test)]
mod sde_integration {
    use super::fit;
    use crate::parser::model_parser::parse_full_model;
    use crate::types::*;
    use std::collections::HashMap;

    /// 1-cpt IV ODE model with a [diffusion] block on the central compartment.
    /// Sigma (ADD) is fixed so that the diffusion parameter must absorb residual
    /// variance. We verify:
    ///   (a) uses_sde = true
    ///   (b) DIFF_CENTRAL is estimated positive
    ///   (c) OFV is finite and the fit converges
    ///   (d) OFV with diffusion <= OFV without diffusion (diffusion can only help)
    const SDE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[diffusion]
  central ~ 0.5

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    /// Same model without the [diffusion] block (for OFV comparison).
    const BASE_MODEL_SRC: &str = r#"
[parameters]
  theta TVCL(5.0, 0.1, 50.0)
  theta TVV(50.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD ~ 1.0 FIX

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  ode(obs_cmt=central, states=[central])

[odes]
  d/dt(central) = -(CL/V) * central

[error_model]
  DV ~ additive(ADD)

[fit_options]
  method = foce
"#;

    fn make_sde_population() -> Population {
        // 4 subjects, single IV bolus dose=100 at t=0, observations at 3 times.
        // The ODE `d/dt(central) = -(CL/V) * central` describes the amount in
        // the central compartment (mg) — ferx adds `dose.amt` directly to the
        // state, so for an IV bolus the state IS the dose in amount units.
        // Observations must therefore also be in amount (mg), not concentration.
        // True amounts from a 1-cpt model with CL=5, V=50 (k = 0.1/h):
        //   t=1: A(t) = 100·exp(-0.1) = 90.48
        //   t=4: A(t) = 100·exp(-0.4) = 67.03
        //   t=8: A(t) = 100·exp(-0.8) = 44.93
        // Values below are symmetric ±5% perturbations of the true amounts
        // (two subjects below, two above) so the population sample remains
        // centered on the analytical trajectory.
        let obs_times = vec![1.0, 4.0, 8.0];
        let dvs: &[(&str, Vec<f64>)] = &[
            // -5% across all times
            ("S1", vec![85.96, 63.68, 42.68]),
            // +5% across all times
            ("S2", vec![95.00, 70.38, 47.18]),
            // -3% across all times
            ("S3", vec![87.77, 65.02, 43.58]),
            // +3% across all times
            ("S4", vec![93.19, 69.04, 46.28]),
        ];
        let subjects = dvs
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 3],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0; 3],
                occasions: vec![1u32; 3],
                dose_occasions: vec![1u32],
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    fn fast_foce_opts() -> FitOptions {
        FitOptions {
            method: EstimationMethod::Foce,
            methods: Vec::new(),
            outer_maxiter: 80,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: false,
            mu_referencing: false,
            optimizer: Optimizer::Slsqp,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    #[test]
    fn test_sde_fit_smoke() {
        // Combined smoke test: one SDE fit, three assertions. Each EKF FOCE
        // fit takes ~30–50 min on the 2-core CI runner, so the previous
        // 3-tests-1-assertion split tripled CI wall for no extra coverage.
        let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let pop = make_sde_population();
        let opts = fast_foce_opts();
        let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts)
            .expect("SDE fit should succeed");
        assert!(result.uses_sde, "uses_sde must be true");
        assert!(
            result.ofv.is_finite(),
            "OFV must be finite, got {}",
            result.ofv
        );
        let diff_idx = result
            .theta_names
            .iter()
            .position(|n| n == "DIFF_CENTRAL")
            .expect("DIFF_CENTRAL must be in theta_names");
        let diff_val = result.theta[diff_idx];
        assert!(
            diff_val > 0.0,
            "DIFF_CENTRAL must be positive, got {diff_val}"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: opt in with --features slow-tests"
    )]
    fn test_sde_ofv_le_base_ofv() {
        // Reference: the OFV from the identical model fit without the
        // [diffusion] block (BASE_MODEL_SRC). Since [diffusion] adds an extra
        // free parameter (DIFF_CENTRAL ≥ 0) and the EKF observation variance
        // collapses to the residual-only variance when DIFF_CENTRAL → 0, the
        // SDE OFV must be ≤ the base OFV at the optimum.
        // The +1 unit of slack absorbs numerical noise from finite-difference
        // gradients, NLopt's stopping tolerance (`outer_gtol = 1e-3`), and the
        // truncated `outer_maxiter = 80` cap in `fast_foce_opts`; without the
        // slack we'd flake on iterations where the SDE fit stopped a hair
        // short of the base fit's OFV.
        let pop = make_sde_population();
        let opts = fast_foce_opts();

        let parsed_base = parse_full_model(BASE_MODEL_SRC).expect("base model should parse");
        let base_result = fit(
            &parsed_base.model,
            &pop,
            &parsed_base.model.default_params,
            &opts,
        )
        .expect("base fit should succeed");

        let parsed_sde = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let sde_result = fit(
            &parsed_sde.model,
            &pop,
            &parsed_sde.model.default_params,
            &opts,
        )
        .expect("SDE fit should succeed");

        assert!(
            sde_result.ofv <= base_result.ofv + 1.0,
            "SDE OFV ({}) should not be worse than base OFV ({}) by more than 1 unit",
            sde_result.ofv,
            base_result.ofv,
        );
    }

    /// SDE + gn / gn_hybrid must fail with a clear error message.
    #[test]
    fn sde_gn_returns_error() {
        use crate::types::EstimationMethod;

        let parsed = parse_full_model(SDE_MODEL_SRC).expect("SDE model should parse");
        let pop = {
            // Minimal single-subject population (no data needed — error fires before fitting).
            use crate::types::{DoseEvent, Population, Subject};
            let subj = Subject {
                id: "1".into(),
                doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
                obs_times: vec![1.0],
                observations: vec![1.0],
                obs_cmts: vec![1],
                covariates: HashMap::new(),
                dose_covariates: Vec::new(),
                obs_covariates: Vec::new(),
                pk_only_times: Vec::new(),
                pk_only_covariates: Vec::new(),
                reset_times: Vec::new(),
                cens: vec![0],
                occasions: Vec::new(),
                dose_occasions: Vec::new(),
            };
            Population {
                subjects: vec![subj],
                covariate_names: Vec::new(),
                dv_column: "DV".into(),
            }
        };

        for method in [EstimationMethod::FoceGn, EstimationMethod::FoceGnHybrid] {
            let opts = FitOptions {
                method,
                ..FitOptions::default()
            };
            let result = fit(&parsed.model, &pop, &parsed.model.default_params, &opts);
            assert!(result.is_err(), "expected error for {:?} + SDE", method);
            let msg = result.unwrap_err();
            assert!(
                msg.contains("gn") || msg.contains("gn_hybrid"),
                "error message should mention gn: {msg}"
            );
        }
    }
}

#[cfg(test)]
mod multi_start_tests {
    use super::perturb_init;
    use crate::estimation::parameterization::theta_packs_log;
    use crate::types::{FitOptions, ModelParameters, OmegaMatrix, SigmaVector};

    fn make_params(
        theta: Vec<f64>,
        theta_lower: Vec<f64>,
        theta_upper: Vec<f64>,
    ) -> ModelParameters {
        let n = theta.len();
        ModelParameters {
            theta,
            theta_names: (0..n).map(|i| format!("T{i}")).collect(),
            theta_lower,
            theta_upper,
            theta_fixed: vec![false; n],
            omega: OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]),
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        }
    }

    #[test]
    fn test_perturb_start0_is_identity() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        let perturbed = perturb_init(&p, 0, 0.5, 42);
        assert_eq!(perturbed.theta, p.theta);
    }

    #[test]
    fn test_perturb_changes_theta() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        let perturbed = perturb_init(&p, 1, 0.3, 42);
        // With sigma=0.3 and seed=43 (42+1), at least one theta should differ
        let changed = perturbed
            .theta
            .iter()
            .zip(p.theta.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "start 1 should perturb theta");
    }

    #[test]
    fn test_perturb_stays_in_bounds() {
        let p = make_params(vec![5.0, 50.0], vec![0.1, 1.0], vec![100.0, 500.0]);
        for k in 1..=10 {
            let perturbed = perturb_init(&p, k, 2.0, 42); // large sigma to stress-test bounds
            for (i, &t) in perturbed.theta.iter().enumerate() {
                assert!(
                    t >= p.theta_lower[i],
                    "start {k}: theta[{i}]={t} < lower={}",
                    p.theta_lower[i]
                );
                assert!(
                    t <= p.theta_upper[i],
                    "start {k}: theta[{i}]={t} > upper={}",
                    p.theta_upper[i]
                );
            }
        }
    }

    #[test]
    fn test_perturb_identity_packed_theta() {
        // theta_lower < 0 → identity packing → additive perturbation
        let p = make_params(vec![0.5], vec![-5.0], vec![5.0]);
        assert!(!theta_packs_log(p.theta_lower[0]));
        let perturbed = perturb_init(&p, 1, 0.3, 99);
        assert!(perturbed.theta[0] >= -5.0 && perturbed.theta[0] <= 5.0);
    }

    #[test]
    fn test_n_starts_option_parsed() {
        let mut opts = FitOptions::default();
        assert_eq!(opts.n_starts, 1);
        opts.n_starts = 4;
        assert_eq!(opts.n_starts, 4);
    }

    #[test]
    fn test_n_starts_and_seed_via_parser() {
        use crate::parser::model_parser::apply_fit_option;
        let mut opts = FitOptions::default();
        apply_fit_option(&mut opts, "n_starts", "4").expect("n_starts parses");
        assert_eq!(opts.n_starts, 4);
        apply_fit_option(&mut opts, "multi_start_seed", "123").expect("multi_start_seed parses");
        assert_eq!(opts.multi_start_seed, Some(123));
        apply_fit_option(&mut opts, "start_sigma", "0.5").expect("start_sigma parses");
        assert!((opts.start_sigma - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_per_start_saem_seed_derivation() {
        let base: u64 = 12345;
        // Each start k > 0 gets base + k; start 0 keeps the base unchanged.
        assert_eq!(base.wrapping_add(0), 12345);
        assert_eq!(base.wrapping_add(1), 12346);
        assert_eq!(base.wrapping_add(7), 12352);
        // All derived seeds are distinct.
        let seeds: Vec<u64> = (0..8).map(|k| base.wrapping_add(k)).collect();
        let unique: std::collections::HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(unique.len(), 8);
        // wrapping_add is defined at u64::MAX.
        assert_eq!(u64::MAX.wrapping_add(1), 0);
    }
}

#[cfg(test)]
mod tests_sdtab_tv_cov {
    use super::*;
    use crate::types::{
        BloqMethod, CompiledModel, DoseEvent, ErrorModel, ErrorSpec, GradientMethod,
        ModelParameters, OmegaMatrix, PkModel, PkParams, Population, ScalingSpec, SigmaVector,
        Subject,
    };
    use nalgebra::{DMatrix, DVector};
    use std::collections::HashMap;

    /// Regression: on a subject with time-varying covariates the sdtab IPRED
    /// (`SubjectResult.ipred`) must come from the **TV-aware** prediction path
    /// — `compute_predictions_with_tv` — so it agrees with the IPRED used by
    /// the FOCEI marginal during the fit.
    ///
    /// Before this fix, `compute_subject_results` called `model_preds` with a
    /// single per-subject `pk_params` derived from `subject.covariates`. For
    /// TV-covariate subjects that silently used the wrong covariate snapshot
    /// for every observation, producing sdtab IPREDs that drifted to ~0 after
    /// the first dose and inflated IWRES into 30+. The OFV was fine because
    /// the FOCEI marginal already routed through `compute_predictions_with_tv`
    /// — only the post-fit diagnostic path was broken.
    ///
    /// This test constructs a minimal 1-cpt IV bolus model where `pk_param_fn`
    /// reads `WT` to scale CL, and a subject whose `obs_covariates` carry
    /// `WT = [70, 140, 210]` (vs `subject.covariates["WT"] = 70`). The TV
    /// path gives a strictly different concentration profile from the no-TV
    /// path, so the assertion `sdtab IPRED == compute_predictions_with_tv`
    /// fails *loudly* if the dispatch ever regresses to `model_preds`.
    #[test]
    fn test_sdtab_ipred_honours_tv_covariates() {
        // ── Minimal CompiledModel: 1-cpt IV bolus, CL scaled by per-event WT ──
        let omega = OmegaMatrix::from_diagonal(&[0.04], vec!["ETA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0], // TVCL = 5, TVV = 50
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector {
                values: vec![0.1],
                names: vec!["PROP_ERR".into()],
            },
            sigma_fixed: vec![false],
            omega_iov: None,
            kappa_fixed: Vec::new(),
        };
        let model = CompiledModel {
            name: "tv_cov_sdtab_regression".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            error_spec: crate::types::ErrorSpec::Single(ErrorModel::Proportional),
            // CL = TVCL · exp(η_CL) · (WT/70) — reads WT from the covariate map
            // that `compute_predictions_with_tv` substitutes per-event from
            // `obs_covariates` / `dose_covariates`. With WT changing per obs
            // the TV path produces a profile that the (broken) no-TV path
            // can't match.
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], cov: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let wt = cov.get("WT").copied().unwrap_or(70.0);
                p.values[0] = theta[0] * eta[0].exp() * (wt / 70.0);
                p.values[1] = theta[1];
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            kappa_names: Vec::new(),
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            indiv_param_names: vec!["CL".into(), "V".into()],
            indiv_param_partials: crate::types::IndivParamPartials::empty(),
            default_params: default_params.clone(),
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec!["WT".into()],
            gradient_method: GradientMethod::Fd,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
            log_transform: false,
            dv_pre_logged: false,
        };

        // Subject with TV WT: subject.covariates["WT"] = 70 (the no-TV snapshot)
        // but obs_covariates have WT = [70, 140, 210] at the three observation
        // times. dose_covariates set to the same WT=70 the dose-time snapshot
        // would carry.
        let mut subj_cov = HashMap::new();
        subj_cov.insert("WT".to_string(), 70.0);
        let mut obs_covs: Vec<HashMap<String, f64>> = Vec::new();
        for wt in [70.0_f64, 140.0, 210.0] {
            let mut m = HashMap::new();
            m.insert("WT".to_string(), wt);
            obs_covs.push(m);
        }
        let mut dose_covs: Vec<HashMap<String, f64>> = Vec::new();
        let mut m = HashMap::new();
        m.insert("WT".to_string(), 70.0);
        dose_covs.push(m);
        let subject = Subject {
            id: "TVS".to_string(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 2.0, 3.0],
            observations: vec![10.0, 5.0, 2.5], // placeholders; values don't matter for the IPRED check
            obs_cmts: vec![1, 1, 1],
            covariates: subj_cov,
            dose_covariates: dose_covs,
            obs_covariates: obs_covs,
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![1, 1, 1],
            dose_occasions: vec![1],
        };
        // Sanity: this subject must be classified TV — that's the regime the
        // bug lived in.
        assert!(
            subject.has_tv_covariates(),
            "test setup wrong: subject must have TV covariates"
        );

        let population = Population {
            subjects: vec![subject.clone()],
            covariate_names: vec!["WT".into()],
            dv_column: "DV".into(),
        };

        // Fixed EBE at η = 0; H matrix is irrelevant for the IPRED check but
        // must be the right shape for CWRES not to panic.
        let eta_hats = vec![DVector::from_vec(vec![0.0])];
        let h_matrices = vec![DMatrix::from_element(3, 1, 0.5)];
        let kappas: Vec<Vec<DVector<f64>>> = vec![Vec::new()];

        // Reference IPRED: the TV-aware dispatcher the FOCEI marginal uses.
        let ipred_reference = crate::pk::compute_predictions_with_tv(
            &model,
            &subject,
            &default_params.theta,
            eta_hats[0].as_slice(),
        );
        // Sanity: the TV path must NOT collapse to the no-TV path here. If
        // both paths produced the same IPRED, this regression test would
        // trivially pass even if the dispatch in `compute_subject_results`
        // regressed to `model_preds` — so we verify the TV vs no-TV gap is
        // visible before relying on the equality assertion below.
        let pk_no_tv = (model.pk_param_fn)(
            &default_params.theta,
            eta_hats[0].as_slice(),
            &subject.covariates,
        );
        let ipred_no_tv = model_preds(
            &model,
            &subject,
            &pk_no_tv,
            &default_params.theta,
            eta_hats[0].as_slice(),
        );
        let gap: f64 = ipred_reference
            .iter()
            .zip(ipred_no_tv.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            gap > 1e-3,
            "test setup wrong: TV and no-TV IPRED paths must differ noticeably \
             for this regression test to mean anything; got gap = {gap}, \
             ipred_tv = {ipred_reference:?}, ipred_no_tv = {ipred_no_tv:?}"
        );

        // The actual regression assertion: `compute_subject_results` IPRED
        // must equal the TV-aware reference. If the dispatch ever falls back
        // to `model_preds` it will be `ipred_no_tv` instead — failure here.
        let results = compute_subject_results(
            &model,
            &population,
            &default_params,
            &eta_hats,
            &h_matrices,
            &kappas,
            true,
        );
        assert_eq!(results.len(), 1);
        let sdtab_ipred = &results[0].ipred;
        assert_eq!(sdtab_ipred.len(), 3);
        for (j, (&got, &expected)) in sdtab_ipred.iter().zip(ipred_reference.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-12,
                "sdtab IPRED at obs {j} = {got}, expected (TV-aware) {expected} \
                 — `compute_subject_results` must route IPRED through \
                 `compute_predictions_with_tv` for TV-covariate subjects"
            );
        }
    }
}
