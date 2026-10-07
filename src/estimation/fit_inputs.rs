//! The model and population a post-hoc step on a fit runs on (#1622).
//!
//! `run_sir` and `run_covariance` evaluate the objective at a fit's θ, so they
//! must run on the model **as it was fitted** — the fit's θ layout and the fit's
//! covariate centres — and on the population the fit saw. Both used to carry a
//! copy of the same resolve block, and both copies re-read `fit.data_path`
//! without the model's `[data_selection]` filter and without binding the model's
//! data-derived halves. A level-block model then panicked from a `Result` API
//! (the unbound model has fewer θ than the fit), and a `center = median` model
//! returned `Ok` with the covariate effect absent. One resolver, called by both,
//! is the fix: it reads with the fit's own reader and binds with
//! [`bind_from_fit`](crate::bind_from_fit) and the fit's `data_bindings`.

use std::borrow::Cow;
use std::path::Path;

use crate::diagnostics::EngineError;
use crate::io::hash::sha256_file;
use crate::types::{CompiledModel, FitOptions, FitResult, ParsedModel, Population};

/// The model a post-hoc step runs on: the caller's, or one rebuilt from the fit.
/// Not a `Cow`, since `CompiledModel` (closures) is not `Clone`.
enum ModelRef<'a> {
    Lent(&'a CompiledModel),
    Built(Box<CompiledModel>),
}

/// The resolved inputs of a post-hoc step.
pub(crate) struct FitInputs<'a> {
    model: ModelRef<'a>,
    population: Cow<'a, Population>,
}

impl FitInputs<'_> {
    pub(crate) fn model(&self) -> &CompiledModel {
        match &self.model {
            ModelRef::Lent(m) => m,
            ModelRef::Built(m) => m,
        }
    }
    pub(crate) fn population(&self) -> &Population {
        &self.population
    }
}

/// Why the model file is read: it decides the wording of a failure to read it.
#[derive(Clone, Copy)]
enum ModelFileUse {
    /// `model = None`: the model is rebuilt from it.
    Rebuild,
    /// `model = Some`, `population = None`: only its reader settings are needed.
    ReaderSettings,
}

/// Read, hash and parse `fit.model_path` from **one** read of the file, so the
/// hashed bytes, the parsed declarations and the text `bind_from_fit` compiles are
/// the same version of it.
fn read_model_file(
    fit: &FitResult,
    path: &str,
    why: ModelFileUse,
) -> Result<(ParsedModel, String), String> {
    // In the reader-settings cell the caller holds a model already, so a failure
    // here says why the file is still needed, and how to not need it.
    let because = match why {
        ModelFileUse::Rebuild => String::new(),
        ModelFileUse::ReaderSettings => " The population is re-read with this file's \
             `[data]` renames and `[data_selection]`, which decide the rows the fit saw. \
             Pass `population = Some(&pop)` as well, and the model file is not read."
            .to_string(),
    };
    let src = crate::io::model_source::ModelSource::load(
        Path::new(path),
        fit.model_hash.as_deref(),
        "",
        &because,
    )
    .map_err(crate::io::model_source::ModelSourceError::into_message)?;
    Ok((src.parsed, src.text))
}

/// Refuse a population that is not the fit's subjects in the fit's order: the
/// fit's EBEs are matched to subjects by position. One message per cause, the
/// order one naming the first position that differs.
fn check_subjects(fit: &FitResult, population: &Population) -> Result<(), String> {
    if fit.subjects.is_empty() {
        return Ok(());
    }
    const WHY: &str = "The fit's EBEs are matched to subjects by position, so the population \
                       must be the one the fit saw.";
    if population.subjects.len() != fit.subjects.len() {
        return Err(format!(
            "the population has {} subjects but the fit has {}. {WHY}",
            population.subjects.len(),
            fit.subjects.len()
        ));
    }
    let first = population
        .subjects
        .iter()
        .zip(&fit.subjects)
        .position(|(p, f)| p.id != f.id);
    if let Some(i) = first {
        return Err(format!(
            "subject {} of the population is `{}`, but the fit's is `{}`. {WHY}",
            i + 1,
            population.subjects[i].id,
            fit.subjects[i].id
        ));
    }
    Ok(())
}

/// #1729: refuse a lent model whose covariate statistics the population it will be
/// scored on does not reproduce, for a fit that records no bindings to compare. The
/// fitted model passes bit for bit (`check_stats_on` re-summarises with the binder's
/// own function); one bound on a simulation design is refused rather than scoring
/// the fit's θ centred on the design.
fn check_lent_stats(
    m: &CompiledModel,
    population: &Population,
    supplied: bool,
    with_model_file: bool,
) -> Result<(), String> {
    let mismatch = crate::api::check_stats_on(m, population)?;
    let Some(d) = mismatch else {
        return Ok(());
    };
    let (against, there) = if supplied {
        ("the supplied population", "the supplied population")
    } else {
        ("the fit's data, re-read from `fit.data_path`", "the data")
    };
    let routed = if supplied || with_model_file {
        ""
    } else {
        " The data was re-read without the model file (the fit records no \
         `model_path`), so a `[data_selection]` in it was not applied: if the model \
         has one, pass `population = Some(&pop)` with the fit's population."
    };
    // `bind_covariate_stats` binds the statistics only: on a model with a level block
    // it leaves the layout unbound, and the call is then refused on its θ count.
    let fix = if m.theta_blocks().level_blocks().is_empty() {
        "`prepare_run` on the fit's model and data files, or `bind_covariate_stats` on a \
         freshly parsed model with the fit's population."
    } else {
        "`prepare_run` on the fit's model and data files. The model has a theta level \
         block, which `bind_covariate_stats` does not bind."
    };
    Err(format!(
        "this fit records no data-derived bindings (an older `.fitrx`), so the \
         supplied model's covariate statistics were checked against {against}, and they \
         differ: the {field} of `{cov}` is {mv} in the model but {dv} in {there}. The model \
         was bound on other data, and scoring the fit's θ with it would centre the \
         relations on that data, not on the data the θ was estimated from. Re-parse the \
         model and bind it on the fit's data: {fix}{routed}",
        field = d.field,
        cov = d.covariate,
        mv = d.model,
        dv = d.data,
    ))
}

/// Resolve the model and population `entry` (`"run_sir"` / `"run_covariance"`)
/// runs on, refusing every input that would evaluate the fit's θ against a model
/// or population other than the fitted ones.
///
/// - `model = None`: re-parsed from `fit.model_path` (hash-verified), then bound
///   with `bind_from_fit` and `fit.data_bindings`. A model with a level block binds
///   on the re-read population or a copy of the supplied one; any other binds
///   without touching the population. A fit whose bindings are empty (an older
///   `.fitrx`) is refused when the model needs them; a plain model is unaffected.
/// - `model = Some(m)`: used as supplied, but refused when `m` is not bound with
///   the fit's (non-empty) bindings, or still has a relation waiting on statistics.
///   A re-read population gets `m`'s level index columns. When the fit records no
///   bindings, `m`'s covariate statistics must be the ones the population summarises
///   to (#1729): the fitted model passes, one bound on other data is refused.
/// - `population = None`: re-read from `fit.data_path` (hash-verified) the way the
///   fit read it, `[data_selection]` included. With `Some(model)` that needs the
///   (hash-verified) model file for its reader settings; a fit that recorded none
///   falls back to the model-routed reader.
///
/// The population must be the fit's subjects in the fit's order, checked before any
/// binding, and the model must have the fit's θ count. Both were panics (#1622).
/// Last, a categorical covariate value outside the model's levels is refused
/// (`E_COV_LEVEL_UNKNOWN`, #1740): the subject IDs can match while a covariate
/// column was recoded.
///
/// Every refusal is attributed to `entry` as the [`EngineError`]'s context, so it
/// prints as `"{entry}: {message}"`; a refusal `ferx check` codes
/// (`E_COV_LEVEL_UNKNOWN`, `E_COVSTAT_UNRESOLVED`) keeps its diagnostic (#1746).
pub(crate) fn resolve_fit_inputs<'a>(
    fit: &FitResult,
    model: Option<&'a CompiledModel>,
    population: Option<&'a Population>,
    entry: &str,
) -> Result<FitInputs<'a>, EngineError> {
    resolve_fit_inputs_unattributed(fit, model, population).map_err(|e| e.in_context(entry))
}

/// [`resolve_fit_inputs`] before its refusal is attributed to the entry point.
fn resolve_fit_inputs_unattributed<'a>(
    fit: &FitResult,
    model: Option<&'a CompiledModel>,
    population: Option<&'a Population>,
) -> Result<FitInputs<'a>, EngineError> {
    // A supplied model carries no `iov_column`, and the one in the model file need
    // not be the one this model was built with, so refuse rather than parse
    // occasions out of the data with settings the model may not share. First, so
    // nothing is read for a call that cannot run.
    if let (Some(m), None) = (model, population) {
        if m.n_kappa > 0 {
            return Err(EngineError::from(
                "caller-supplied `model` for an IOV (n_kappa > 0) model \
                 requires `population` to also be supplied — the model carries no \
                 `iov_column`, and per-occasion kappas are parsed with it.",
            ));
        }
    }

    // Where the population comes from, for the #1729 refusal's wording.
    let supplied = population.is_some();

    let mut file: Option<(ParsedModel, String)> = None;
    if model.is_none() {
        let path = fit.model_path.as_deref().ok_or(
            "no model supplied and fit.model_path is None. \
             Either pass `model = Some(&model)` or re-fit via fit_from_files \
             so the path is recorded.",
        )?;
        file = Some(read_model_file(fit, path, ModelFileUse::Rebuild)?);
    } else if population.is_none() {
        if let Some(path) = fit.model_path.as_deref() {
            file = Some(read_model_file(fit, path, ModelFileUse::ReaderSettings)?);
        }
    }

    // --- Population --------------------------------------------------------
    let mut population: Cow<'a, Population> = match population {
        Some(p) => Cow::Borrowed(p),
        None => {
            let path = fit.data_path.as_deref().ok_or(
                "no population supplied and fit.data_path is None. \
                 Either pass `population = Some(&pop)` or re-fit via fit_from_files \
                 so the path is recorded.",
            )?;
            if let Some(expected) = &fit.data_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(EngineError::from(format!(
                        "data hash mismatch for {path}. Stored: {expected}, current: \
                         {actual}. The dataset has changed since the fit was produced — \
                         refusing to run against stale data."
                    )));
                }
            }
            let p = match (&file, model) {
                (Some((parsed, _)), _) => crate::api::read_population_as_fitted(parsed, path)?.0,
                (None, Some(m)) => {
                    crate::api::read_population_routed_by(m, Path::new(path), None, &[])?
                }
                (None, None) => unreachable!("model = None always reads the model file"),
            };
            Cow::Owned(p)
        }
    };
    // Before any binding: a population that is not the fit's should hear that, not
    // a level refusal worded for a simulation design.
    check_subjects(fit, &population)?;

    // --- Model -------------------------------------------------------------
    let model: ModelRef<'a> = match model {
        None => {
            let (mut parsed, text) = file.expect("read above when model is None");
            // Only a level block writes to the population, so only then is a
            // supplied one copied.
            let pop = if parsed.model.theta_blocks().level_blocks().is_empty() {
                None
            } else {
                Some(population.to_mut())
            };
            crate::api::bind_from_fit_on(&mut parsed, &text, pop, &fit.data_bindings)?;
            ModelRef::Built(Box::new(parsed.model))
        }
        Some(m) => {
            if !fit.data_bindings.is_empty() && m.data_bindings() != &fit.data_bindings {
                return Err(EngineError::from(
                    "the supplied model is not bound with this fit's bindings: its \
                     data-derived bindings (level layout, covariate statistics) differ from \
                     the fit's `data_bindings`. Pass `model = None` to rebuild it from the \
                     fit, or bind it with `ferx_core::api::bind_from_fit` and the fit's \
                     `data_bindings`.",
                ));
            }
            if let Cow::Owned(p) = &mut population {
                crate::api::write_fitted_level_columns(m, p, &m.data_bindings().levels)?;
            }
            crate::diagnostics::first_error(&crate::api::check_covariate_model_bound(m))?;
            // No recorded bindings to compare (an older `.fitrx`): the population
            // decides. After the level index write, so a population with a level the
            // model never saw keeps that refusal.
            if fit.data_bindings.is_empty() && !m.data_bindings().covariate_stats.is_empty() {
                check_lent_stats(m, &population, supplied, file.is_some())?;
            }
            ModelRef::Lent(m)
        }
    };

    let inputs = FitInputs { model, population };
    if inputs.model().n_theta != fit.theta.len() {
        return Err(EngineError::from(format!(
            "the model has n_theta = {} but the fit has {} θ. Verify you supplied \
             the same model the fit used, bound the way the fit was (`bind_from_fit` with \
             the fit's `data_bindings`).",
            inputs.model().n_theta,
            fit.theta.len()
        )));
    }
    // The same subjects can carry a recoded covariate: a categorical value outside
    // the model's levels would be scored as the reference level (#1740).
    crate::diagnostics::first_error(&crate::api::check_covariate_levels(
        inputs.model(),
        &inputs.population,
    ))?;
    Ok(inputs)
}

/// The caller's options with `interaction` set to the marginal `fit` was estimated
/// under (#1710), for a post-hoc step that re-scores the objective at the fit's
/// estimates.
///
/// Keyed on `fit.method` first, through the same [`interaction_for`] rule the stage
/// loop uses, and on the stored `fit.interaction` only for a method the rule passes
/// through (Gauss-Newton). So a `.fitrx` written before #1710 — whose FOCE fits carry
/// the leaked `interaction: true` — still re-scores under FOCE, and a caller passing
/// `FitOptions::default()` (`interaction = true`) to `run_sir` on a FOCE fit no longer
/// weights its draws with the FOCEI objective: on warfarin_iov that was an SIR ESS of
/// 4.9 / 1000 and a covariance step reporting SE(TVKA) = 7576 against the fit's 0.74.
///
/// [`interaction_for`]: crate::types::interaction_for
pub(crate) fn fitted_marginal_options(fit: &FitResult, options: &FitOptions) -> FitOptions {
    FitOptions {
        interaction: crate::types::interaction_for(fit.method, fit.interaction),
        ..options.clone()
    }
}

/// The #1619 / #1622 fixture, shared by the `run_covariance` and `run_sir` tests:
/// the analytic two-compartment oral model on `data/two_cpt_oral_cov.csv` (30
/// subjects) with a `STUDY = (ID − 1) mod 3 + 1` column, in four model kinds.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use std::path::{Path, PathBuf};

    /// Which data-derived halves the model has.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) enum Kind {
        /// `center = 70`: nothing data-derived — the control.
        Plain,
        /// `center = median`: a symbolic statistic.
        Median,
        /// `theta SHIFT[STUDY]` on `KA`. Ten subjects per study, so the block cannot
        /// absorb `ETA_KA` and `auto` resolves to global `sum_to_zero`. On `KA`, not
        /// `Q` or `V2`: there the fitted SIR proposal degenerates (ESS 1.0–1.02 of 100
        /// resamples, measured), and a `to_bits` match of one draw compares nothing.
        Level,
        /// Both.
        LevelMedian,
        /// `Plain` with `[data_selection] ignore_subjects = [3]`.
        Select,
    }

    pub(crate) fn model_text(kind: Kind, fit_options: &str) -> String {
        let level = matches!(kind, Kind::Level | Kind::LevelMedian);
        let median = matches!(kind, Kind::Median | Kind::LevelMedian);
        format!(
            r#"
[parameters]
  theta TVCL(4.0, 0.1, 100.0)
  theta TVV1(40.0, 1.0, 500.0)
  theta TVQ(8.0, 0.1, 100.0)
  theta TVV2(80.0, 1.0, 500.0)
  theta TVKA(1.0, 0.01, 10.0)
{lvl}
  omega ETA_CL ~ 0.15
  omega ETA_V1 ~ 0.15
  omega ETA_KA ~ 0.20
  sigma PROP_ERR ~ 0.04 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = TVQ
  V2 = TVV2
  KA = TVKA * {ka}exp(ETA_KA)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V1, q=Q, v2=V2, ka=KA)

[covariates]
  WT   continuous
  CRCL continuous
  STUDY categorical

[covariate_model]
  CL ~ WT   power(center = {c}) => THETA_CL_WT(0.6, 0.01, 5.0)

[error_model]
  DV ~ proportional(PROP_ERR)
{sel}
{fit_options}
"#,
            lvl = if level {
                "  theta SHIFT[STUDY](0.0, -2.0, 2.0)"
            } else {
                ""
            },
            ka = if level { "exp(SHIFT) * " } else { "" },
            c = if median { "median" } else { "70" },
            sel = if kind == Kind::Select {
                "[data_selection]\n  ignore_subjects = [3]\n"
            } else {
                ""
            },
        )
    }

    /// The fixture data with its `STUDY` column, as CSV text.
    pub(crate) fn data_text() -> String {
        let src = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/two_cpt_oral_cov.csv"),
        )
        .expect("data/two_cpt_oral_cov.csv");
        let mut out = String::new();
        for (i, line) in src.lines().enumerate() {
            if i == 0 {
                out.push_str(&format!("{line},STUDY\n"));
                continue;
            }
            let id: usize = line.split(',').next().unwrap().parse().unwrap();
            out.push_str(&format!("{line},{}\n", (id - 1) % 3 + 1));
        }
        out
    }

    /// Write the model and data into `dir`; return their paths.
    pub(crate) fn write(dir: &Path, kind: Kind, fit_options: &str) -> (PathBuf, PathBuf) {
        let model = dir.join("model.ferx");
        let data = dir.join("data.csv");
        std::fs::write(&model, model_text(kind, fit_options)).unwrap();
        std::fs::write(&data, data_text()).unwrap();
        (model, data)
    }

    /// The options `run_sir` / `run_covariance` read from the file: FOCEI, and a SIR
    /// configuration. `fit_from_files` ignores a file's `[fit_options]` by design, so
    /// the fit's own options are passed to it explicitly (see [`case`]).
    pub(crate) const FIT_OPTIONS: &str = "[fit_options]\n  method = focei\n  \
         sir_samples = 300\n  sir_resamples = 100\n  sir_seed = 5\n";

    /// Outer iterations of the fixture fit. The oracles here are bit-identities of
    /// two runs at the same fitted point, so they need no convergence: measured, the
    /// `None == Some` covariance match holds at 0, 1, 3 and at full convergence. The
    /// SIR proposal does not: at 1 the ESS is 3.2–4.9 on three kinds; at 3 it is
    /// 84–111 of 100 resamples on every kind.
    const OUTER_MAXITER: usize = 3;

    /// A fitted case: the fit through `fit_from_files`, and the oracle inputs —
    /// `prepare_run` on the same files, i.e. the model bound on the fit's own data.
    pub(crate) struct Case {
        pub(crate) _dir: tempfile::TempDir,
        pub(crate) model_path: PathBuf,
        pub(crate) fit: crate::types::FitResult,
        pub(crate) prep: crate::api::PreparedRun,
        pub(crate) opts: crate::types::FitOptions,
    }

    /// A case whose fit skips the inline covariance step: every test but SIR's,
    /// which needs the fit's covariance matrix as its proposal ([`sir_case`]).
    pub(crate) fn case(kind: Kind) -> Case {
        build(kind, false)
    }

    /// A case whose fit carries its covariance matrix, for `run_sir`.
    pub(crate) fn sir_case(kind: Kind) -> Case {
        build(kind, true)
    }

    fn build(kind: Kind, covariance: bool) -> Case {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = write(dir.path(), kind, FIT_OPTIONS);
        let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
        let prep = crate::api::prepare_run(m, Some(d)).expect("fixture prepares");
        let opts = prep.parsed.fit_options.clone();
        let fit_opts = crate::types::FitOptions {
            outer_maxiter: OUTER_MAXITER,
            run_covariance_step: covariance,
            ..opts.clone()
        };
        let fit = crate::api::fit_from_files(m, Some(d), None, Some(fit_opts)).expect("fits");
        assert_eq!(fit.covariance_matrix.is_some(), covariance, "{kind:?}");
        Case {
            _dir: dir,
            model_path,
            fit,
            prep,
            opts,
        }
    }
    /// The fixture model bound on a simulation design (#1729): the fit's data with
    /// `WT × 1.3` and `STUDY 3 → 4`, through `prepare_run`. Same subjects, so it
    /// passes the subject check; its WT statistics and its level set are not the fit's.
    pub(crate) fn design(case: &Case) -> crate::api::PreparedRun {
        let mut out = String::new();
        for (i, line) in data_text().lines().enumerate() {
            if i == 0 {
                assert!(line.starts_with("ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT,"));
                out.push_str(line);
            } else {
                let mut cols: Vec<String> = line.split(',').map(str::to_string).collect();
                let wt: f64 = cols[8].parse().unwrap();
                cols[8] = format!("{}", wt * 1.3);
                if cols[10] == "3" {
                    cols[10] = "4".to_string();
                }
                out.push_str(&cols.join(","));
            }
            out.push('\n');
        }
        let dir = case.model_path.parent().unwrap();
        let path = dir.join("design.csv");
        std::fs::write(&path, out).unwrap();
        crate::api::prepare_run(case.model_path.to_str().unwrap(), path.to_str())
            .expect("the design binds")
    }

    /// The model parsed from the fixture file and never bound.
    pub(crate) fn unbound(case: &Case) -> crate::types::CompiledModel {
        crate::parser::model_parser::parse_full_model_file(&case.model_path)
            .unwrap()
            .model
    }
}
