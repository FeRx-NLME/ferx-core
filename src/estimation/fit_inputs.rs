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

use std::path::Path;

use crate::io::hash::sha256_file;
use crate::types::{CompiledModel, FitResult, ParsedModel, Population};

/// A value the caller lent, or one this resolver built.
enum Held<'a, T> {
    Lent(&'a T),
    Built(Box<T>),
}

impl<T> std::ops::Deref for Held<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Held::Lent(t) => t,
            Held::Built(t) => t,
        }
    }
}

/// The resolved inputs of a post-hoc step.
pub(crate) struct FitInputs<'a> {
    model: Held<'a, CompiledModel>,
    population: Held<'a, Population>,
}

impl FitInputs<'_> {
    pub(crate) fn model(&self) -> &CompiledModel {
        &self.model
    }
    pub(crate) fn population(&self) -> &Population {
        &self.population
    }
}

/// Resolve the model and population `entry` (`"run_sir"` / `"run_covariance"`)
/// runs on, refusing every input that would evaluate the fit's θ against a model
/// or population other than the fitted ones.
///
/// - `model = None`: re-parsed from `fit.model_path` (hash-verified), then bound
///   with `bind_from_fit` and `fit.data_bindings`, on the re-read population or on a
///   copy of the supplied one. A fit whose bindings are empty (an older `.fitrx`)
///   is refused when the model needs them; a plain model is unaffected.
/// - `model = Some(m)`: used as supplied, but refused when `m` is not bound with
///   the fit's (non-empty) bindings, or still has a relation waiting on statistics.
///   A re-read population gets `m`'s level index columns.
/// - `population = None`: re-read from `fit.data_path` (hash-verified) the way the
///   fit read it, `[data_selection]` included, whenever the fit recorded its model
///   file.
///
/// Then, for every cell: the model must have the fit's θ count, and the population
/// the fit's subjects in the fit's order. Both were panics before (#1622).
pub(crate) fn resolve_fit_inputs<'a>(
    fit: &FitResult,
    model: Option<&'a CompiledModel>,
    population: Option<&'a Population>,
    entry: &str,
) -> Result<FitInputs<'a>, String> {
    let prefix = |e: String| format!("{entry}: {e}");

    // The model file: parsed when the model is to be rebuilt, or when the
    // population is to be re-read (its reader settings live in the file).
    let mut file: Option<(ParsedModel, String)> = None;
    if model.is_none() || (population.is_none() && fit.model_path.is_some()) {
        let path = fit.model_path.as_deref().ok_or_else(|| {
            format!(
                "{entry}: no model supplied and fit.model_path is None. \
                 Either pass `model = Some(&model)` or re-fit via fit_from_files \
                 so the path is recorded."
            )
        })?;
        if let Some(expected) = &fit.model_hash {
            let actual = sha256_file(Path::new(path))?;
            if &actual != expected {
                return Err(format!(
                    "{entry}: model hash mismatch for {path}. Stored: {expected}, current: \
                     {actual}. The .ferx file has changed since the fit was produced — \
                     refusing to run against stale source."
                ));
            }
        }
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("{entry}: failed to read model file {path}: {e}"))?;
        let parsed = crate::parser::model_parser::parse_full_model_file(Path::new(path))?;
        file = Some((parsed, text));
    }

    // --- Population --------------------------------------------------------
    let mut population: Held<'a, Population> = match population {
        Some(p) => Held::Lent(p),
        None => {
            // A supplied model carries no `iov_column`, so refuse rather than parse
            // occasions out of the data with settings the model may not share.
            if let Some(m) = model {
                if m.n_kappa > 0 {
                    return Err(format!(
                        "{entry}: caller-supplied `model` for an IOV (n_kappa > 0) model \
                         requires `population` to also be supplied — `iov_column` from \
                         `[fit_options]` is needed to parse per-occasion kappas correctly."
                    ));
                }
            }
            let path = fit.data_path.as_deref().ok_or_else(|| {
                format!(
                    "{entry}: no population supplied and fit.data_path is None. \
                     Either pass `population = Some(&pop)` or re-fit via fit_from_files \
                     so the path is recorded."
                )
            })?;
            if let Some(expected) = &fit.data_hash {
                let actual = sha256_file(Path::new(path))?;
                if &actual != expected {
                    return Err(format!(
                        "{entry}: data hash mismatch for {path}. Stored: {expected}, current: \
                         {actual}. The dataset has changed since the fit was produced — \
                         refusing to run against stale data."
                    ));
                }
            }
            let p = match &file {
                Some((parsed, _)) => crate::api::read_population_as_fitted(parsed, path)?.0,
                None => crate::api::read_population_routed_by(
                    model.expect("no model file implies a supplied model"),
                    Path::new(path),
                    None,
                    &[],
                )?,
            };
            Held::Built(Box::new(p))
        }
    };

    // --- Model -------------------------------------------------------------
    let model: Held<'a, CompiledModel> = match model {
        None => {
            let (mut parsed, text) = file.expect("parsed above when model is None");
            let needs = !parsed.model.theta_blocks().level_blocks().is_empty()
                || parsed.model.covariate_model.is_some()
                || !fit.data_bindings.is_empty();
            if needs {
                // The binder writes the level index columns, so it binds a copy
                // of a supplied population rather than the caller's.
                if let Held::Lent(p) = &population {
                    let copy = (*p).clone();
                    population = Held::Built(Box::new(copy));
                }
                let Held::Built(pop) = &mut population else {
                    unreachable!("made owned just above")
                };
                crate::api::bind_from_fit(&mut parsed, &text, pop, &fit.data_bindings)
                    .map_err(prefix)?;
            }
            Held::Built(Box::new(parsed.model))
        }
        Some(m) => {
            if !fit.data_bindings.is_empty() && m.data_bindings() != &fit.data_bindings {
                return Err(format!(
                    "{entry}: the supplied model is not bound with this fit's bindings: its \
                     data-derived bindings (level layout, covariate statistics) differ from \
                     the fit's `data_bindings`. Pass `model = None` to rebuild it from the \
                     fit, or bind it with `ferx_core::api::bind_from_fit` and the fit's \
                     `data_bindings`."
                ));
            }
            if let Held::Built(p) = &mut population {
                crate::api::write_fitted_level_columns(m, p.as_mut(), &m.data_bindings().levels)
                    .map_err(prefix)?;
            }
            crate::api::assert_covariate_model_bound(m).map_err(prefix)?;
            Held::Lent(m)
        }
    };

    // --- Shape: the fit's θ, the fit's subjects -----------------------------
    if model.n_theta != fit.theta.len() {
        return Err(format!(
            "{entry}: the model has n_theta = {} but the fit has {} θ. Verify you supplied \
             the same model the fit used, bound the way the fit was (`bind_from_fit` with \
             the fit's `data_bindings`).",
            model.n_theta,
            fit.theta.len()
        ));
    }
    if !fit.subjects.is_empty() {
        let same = population.subjects.len() == fit.subjects.len()
            && population
                .subjects
                .iter()
                .zip(&fit.subjects)
                .all(|(p, f)| p.id == f.id);
        if !same {
            return Err(format!(
                "{entry}: the population has {} subjects but the fit has {}, or not in the \
                 fit's order. The fit's EBEs are matched to subjects by position, so the \
                 population must be the one the fit saw.",
                population.subjects.len(),
                fit.subjects.len()
            ));
        }
    }

    Ok(FitInputs { model, population })
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

    /// FOCEI with the inline covariance step, and a SIR configuration whose
    /// proposal is not degenerate on any kind (ESS 60–126 of 100 resamples on
    /// this fixture, measured).
    pub(crate) const FIT_OPTIONS: &str = "[fit_options]\n  method = focei\n  maxiter = 40\n  \
         covariance = true\n  sir_samples = 300\n  sir_resamples = 100\n  sir_seed = 5\n";

    /// A fitted case: the fit through `fit_from_files`, and the oracle inputs —
    /// `prepare_run` on the same files, i.e. the model bound on the fit's own data.
    pub(crate) struct Case {
        pub(crate) _dir: tempfile::TempDir,
        pub(crate) model_path: PathBuf,
        pub(crate) fit: crate::types::FitResult,
        pub(crate) prep: crate::api::PreparedRun,
        pub(crate) opts: crate::types::FitOptions,
    }

    pub(crate) fn case(kind: Kind) -> Case {
        let dir = tempfile::tempdir().unwrap();
        let (model_path, data_path) = write(dir.path(), kind, FIT_OPTIONS);
        let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
        let fit = crate::api::fit_from_files(m, Some(d), None, None).expect("fixture fits");
        let prep = crate::api::prepare_run(m, Some(d)).expect("fixture prepares");
        let opts = prep.parsed.fit_options.clone();
        Case {
            _dir: dir,
            model_path,
            fit,
            prep,
            opts,
        }
    }

    /// The model parsed from the fixture file and never bound.
    pub(crate) fn unbound(case: &Case) -> crate::types::CompiledModel {
        crate::parser::model_parser::parse_full_model_file(&case.model_path)
            .unwrap()
            .model
    }
}
