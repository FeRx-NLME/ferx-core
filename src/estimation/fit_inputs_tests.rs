//! #1685: a fit owns how it read its data. `run_sir` / `run_covariance` re-read
//! with the fit's recorded reader settings, and refuse any population — supplied
//! or re-read — that is not the one the fit was given.
//!
//! Fixture: the #1622 two-compartment oral model (`test_fixtures`), FOCEI, analytic
//! Dual2 inner gradient, 30 subjects × 10 observations, 3 outer iterations. Every
//! oracle is a bit-identity at the fitted point or a refusal text, so no fit number
//! enters and the platform does not either.

use super::test_fixtures::{case, case_with_call_ignore, unbound, Case, Kind};
use super::*;
use crate::estimation::run_covariance::run_covariance;
use crate::estimation::run_sir::run_sir;
use crate::types::CovarianceStatus;

const SUPPLIED: &str = "Pass the population the fit was given, or `population = None` to \
                        re-read it from `fit.data_path` with the fit's reader settings.";
const RECORDED: &str = "Re-reading `fit.data_path` with the fit's recorded reader settings did \
                        not reproduce the fitted population, so this version of ferx reads \
                        the file differently from the one that made the fit. Pass the fit's \
                        population as `population = Some(&pop)`, or refit.";
const FILE: &str = "The fit records no reader settings (it was given its population in \
                    memory), so `fit.data_path` was re-read with the model file's `[data]` \
                    renames and `[data_selection]`, which did not reproduce it. Pass the \
                    fit's population as `population = Some(&pop)`.";
const ROUTED: &str = "The fit records neither reader settings nor a `model_path`, so \
                      `fit.data_path` was re-read with the model's endpoint routing only: no \
                      `[data]` renames and no `[data_selection]` were applied. Pass the fit's \
                      population as `population = Some(&pop)`.";
const DOSE_CAUSE: &str = " A dose-row filter (`ignore = EVID == 1 && ...`) or an edited dose \
                          record changes the doses without changing any observation.";

fn refusal(entry: &str, what: &str, advice: &str) -> String {
    format!("{entry}: this population is not the one the fit was given: {what} {advice}")
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The covariance and the θ SEs of two runs, to the bit.
fn assert_same_covariance(got: &FitResult, want: &FitResult, what: &str) {
    assert_eq!(want.covariance_status, CovarianceStatus::Computed, "{what}");
    assert_eq!(got.covariance_status, CovarianceStatus::Computed, "{what}");
    let m = |f: &FitResult| bits(f.covariance_matrix.as_ref().unwrap().as_slice());
    assert_eq!(m(got), m(want), "{what}: covariance bits");
    let se = |f: &FitResult| bits(f.se_theta.as_ref().unwrap());
    assert_eq!(se(got), se(want), "{what}: se_theta bits");
}

fn err_of(r: Result<FitResult, EngineError>) -> String {
    r.map(|_| ()).expect_err("refused").to_string()
}

fn data(c: &Case) -> &str {
    c.fit.data_path.as_deref().unwrap()
}

/// The population read with the fit's recorded settings, routed by the fitted model.
fn read_recorded(c: &Case) -> Population {
    crate::api::read_population_with(
        &c.prep.parsed.model,
        c.fit.reader_settings.as_ref().expect("recorded"),
        data(c),
    )
    .unwrap()
    .0
}

/// T1, probe A (`[inherited]` from the plan, re-measured at `045ca1b2`: re-read 300
/// observations against the fit's 240, SE(TVV2) 3.29 inline vs 1.17 standalone).
/// A `fit_from_files` row filter the model file does not state is recorded and
/// replayed: `(None, None)` and `(Some, None)` are bit-identical to `(Some, Some)` on
/// the fitted population. The straddle is asserted: the model file's settings read
/// 300 observations.
///
/// The same fixture carries the `Records` row of the refusal text from all three
/// sources: supplied (the 300-observation read), re-read with the recorded settings
/// after they are tampered, and re-read with the model file's on a fit that records
/// none.
///
/// Mutations — re-read with the model file's settings when the fit records its own:
/// `(None, None)` is refused (the `Records` row) instead of matching; skip the
/// fingerprint check: the three refusals return `Ok`; swap any two advice texts or
/// delete a sentence of one: its equality dies.
#[test]
fn a_caller_row_filter_is_replayed_on_the_re_read() {
    let c = case_with_call_ignore(Kind::Plain, false, &["TIME > 24"]);
    let settings = c
        .fit
        .reader_settings
        .as_ref()
        .expect("fit_from_files records");
    assert_eq!(settings.ignore_exprs, ["TIME > 24"]);
    let fitted = read_recorded(&c);
    let (file_read, _) = crate::api::read_population_as_fitted(&c.prep.parsed, data(&c)).unwrap();
    assert_eq!((c.fit.n_obs, fitted.n_obs()), (240, 240));
    assert_eq!(
        file_read.n_obs(),
        300,
        "the straddle: the model file has no clause"
    );

    let model = &c.prep.parsed.model;
    let want = run_covariance(&c.fit, Some(model), Some(&fitted), &c.opts).expect("oracle");
    for (m, what) in [(None, "None/None"), (Some(model), "Some/None")] {
        let got =
            run_covariance(&c.fit, m, None, &c.opts).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_same_covariance(&got, &want, what);
    }
    // Review r1 #5: a fingerprint of another scheme cannot be compared, but the
    // recorded settings are still replayed: the re-read is the fitted 240
    // observations, not the model file's 300 (here the two differ, so the cell
    // straddles), unverified and with the warning.
    let mut stale = c.fit.clone();
    stale.population_fingerprint = stale
        .population_fingerprint
        .map(|f| f.with_scheme(crate::types::POPULATION_FINGERPRINT_SCHEME + 1));
    let got = run_covariance(&stale, None, None, &c.opts).expect("stale scheme");
    assert_same_covariance(&got, &want, "stale scheme, recorded settings");
    assert!(got.warnings.iter().any(|w| w == STALE_FINGERPRINT_WARNING));

    let (n_file, n_fit) = (
        file_read.subjects[0].observations.len(),
        fitted.subjects[0].observations.len(),
    );
    assert_eq!((n_file, n_fit), (10, 8));
    let records = format!(
        "the records of subject `1` differ: {n_file} observations in this population, \
         {n_fit} in the fit's."
    );
    let err = err_of(run_covariance(
        &c.fit,
        Some(model),
        Some(&file_read),
        &c.opts,
    ));
    assert_eq!(err, refusal("run_covariance", &records, SUPPLIED));

    let mut drifted = c.fit.clone();
    drifted
        .reader_settings
        .as_mut()
        .unwrap()
        .ignore_exprs
        .clear();
    let err = err_of(run_covariance(&drifted, None, None, &c.opts));
    assert_eq!(err, refusal("run_covariance", &records, RECORDED));

    let mut in_memory = c.fit.clone();
    in_memory.reader_settings = None;
    let err = err_of(run_sir(&in_memory, None, None, &c.opts));
    assert_eq!(err, refusal("run_sir", &records, FILE));
}

/// T1, the SIR half: the recorded filter reaches the SIR re-read too (probe A: SIR
/// returned `Ok` on the 300-observation population). `(None, None)` is bit-identical
/// to `(Some, Some)` on the fitted population, CIs and ESS.
///
/// Mutation — re-read with the model file's settings: refused instead of matching.
#[test]
fn sir_replays_a_caller_row_filter() {
    let c = case_with_call_ignore(Kind::Plain, true, &["TIME > 24"]);
    let fitted = read_recorded(&c);
    let want = run_sir(&c.fit, Some(&c.prep.parsed.model), Some(&fitted), &c.opts).expect("oracle");
    let got = run_sir(&c.fit, None, None, &c.opts).expect("re-read with the recorded settings");
    let ci = |f: &FitResult| {
        f.sir_ci_theta
            .as_ref()
            .unwrap()
            .iter()
            .flat_map(|(a, b)| [a.to_bits(), b.to_bits()])
            .collect::<Vec<_>>()
    };
    assert!(want.sir_ess.unwrap().is_finite());
    assert_eq!(ci(&got), ci(&want));
    assert_eq!(
        got.sir_ess.map(f64::to_bits),
        want.sir_ess.map(f64::to_bits)
    );
}

/// T2, probe B (re-measured at `045ca1b2`: covariance `Ok` with every θ "zero
/// diagonal — flat objective", SIR `Ok` with ESS 298 of 300 — the objective is NaN
/// everywhere). A bound level model handed a population never bound for it is now
/// refused with #1647's text: the block and the binder, never the engine-internal
/// column, a re-read or `[data_selection]`. The straddle: a fit without a
/// fingerprint (an older `.fitrx`) keeps today's `Ok`.
///
/// Mutations — skip the fingerprint check: both cells return `Ok`; drop the level
/// classifier in `population_refusal`: the text is the covariate-columns row.
#[test]
fn a_population_never_bound_for_the_level_block_hears_the_binder() {
    let c = case(Kind::Level);
    let bare = crate::parser::model_parser::parse_full_model_file(&c.model_path).unwrap();
    let never_bound = crate::api::read_population_as_fitted(&bare, data(&c))
        .unwrap()
        .0;
    let model = &c.prep.parsed.model;
    for entry in ["run_covariance", "run_sir"] {
        let r = match entry {
            "run_covariance" => run_covariance(&c.fit, Some(model), Some(&never_bound), &c.opts),
            _ => run_sir(&c.fit, Some(model), Some(&never_bound), &c.opts),
        };
        let e = r.map(|_| ()).expect_err("refused");
        // #1746: the #1647 refusal keeps its `ferx check` code through the resolver.
        assert_eq!(e.code(), Some("E_THETA_LEVELS_DATA_UNBOUND"), "{e}");
        assert_eq!(e.context(), Some(entry), "{e}");
        let err = e.to_string();
        assert_eq!(
            err,
            format!(
                "{entry}: `theta SHIFT[...]` is bound, but this population was never bound \
                 for it, so its records carry no index into the block's levels. Bind the \
                 population with `bind_from_fit(&mut parsed, &model_text, &mut population, \
                 &fit.data_bindings)`, passing the bindings the θ you run was laid out on: \
                 the fit's `data_bindings`, or, for the model's own θ, a clone of \
                 `parsed.model.data_bindings()` taken before the call. Then run the model it \
                 re-parses into `parsed`."
            )
        );
        for never in ["__level_SHIFT", "re-read", "[data_selection]"] {
            assert!(!err.contains(never), "{entry}: says {never}: {err}");
        }
    }
    let mut legacy = c.fit.clone();
    legacy.reader_settings = None;
    legacy.population_fingerprint = None;
    let old = run_covariance(&legacy, Some(model), Some(&never_bound), &c.opts)
        .expect("a fit without a fingerprint is not verified");
    assert!(old.se_theta.is_none(), "the old garbage: a flat objective");
}

/// #1792 review r2 (a): the "`k` of `n`" cell of `E_THETA_LEVELS_DATA_UNBOUND` reaches
/// a post-hoc caller too. The fit's own population with the index stripped from one
/// subject is partly bound: `population_refusal` keeps that cell rather than the
/// fingerprint text, since the fix is the binder, not another population.
///
/// Mutation — drop the "`k` of `n`" cell in `population_refusal`'s filter (keep
/// only "never bound"): the fingerprint text comes back and both entries die.
#[test]
fn a_partly_bound_population_hears_k_of_n_on_a_post_hoc_step() {
    let c = case(Kind::Level);
    let model = &c.prep.parsed.model;
    let mut partly = c.prep.population.clone();
    let s = &mut partly.subjects[0];
    for m in std::iter::once(&mut s.covariates)
        .chain(s.obs_covariates.iter_mut())
        .chain(s.dose_covariates.iter_mut())
        .chain(s.pk_only_covariates.iter_mut())
        .chain(s.reset_covariates.iter_mut())
    {
        m.remove("__level_SHIFT");
    }
    let id = partly.subjects[0].id.clone();
    let n = partly.subjects.len();
    for entry in ["run_covariance", "run_sir"] {
        let r = match entry {
            "run_covariance" => run_covariance(&c.fit, Some(model), Some(&partly), &c.opts),
            _ => run_sir(&c.fit, Some(model), Some(&partly), &c.opts),
        };
        let e = r.map(|_| ()).expect_err("refused");
        assert_eq!(e.code(), Some("E_THETA_LEVELS_DATA_UNBOUND"), "{e}");
        assert_eq!(e.context(), Some(entry), "{e}");
        let err = e.to_string();
        assert!(
            err.starts_with(&format!(
                "{entry}: `theta SHIFT[...]` is bound, but 1 of {n} subjects in this population \
                 carry no index into the block's levels, the first being subject {id}: they \
                 were not bound with the rest. Bind the population with `bind_from_fit("
            )),
            "{err}"
        );
        for never in [
            "not the one the fit was given",
            "never bound",
            "__level_SHIFT",
        ] {
            assert!(!err.contains(never), "{entry}: says {never}: {err}");
        }
    }
}

/// #1792 review r1 #1: a population bound **on its own** for other levels (every
/// `STUDY` relabelled `+100`) carries the index, so `check_level_index_columns` finds
/// `E_THETA_LEVELS_DATA_MISMATCH` — whose "a fit of a model bound on this population" is
/// advice for new data. On a post-hoc step it is not the fit's population, and hears
/// the fingerprint refusal: the first covariate difference and "pass the population
/// the fit was given". The straddle, in the same test: the gate does see a mismatch.
///
/// Mutation — let `population_refusal` return every level diagnostic (the pre-fix
/// code): both entries' equality dies on the MISMATCH text.
#[test]
fn a_population_bound_for_other_levels_hears_the_fingerprint_refusal() {
    let c = case(Kind::Level);
    let text = std::fs::read_to_string(&c.model_path).unwrap();
    let mut parsed = crate::parser::model_parser::parse_full_model_file(&c.model_path).unwrap();
    let mut other = crate::api::read_population_as_fitted(&parsed, data(&c))
        .unwrap()
        .0;
    for s in &mut other.subjects {
        for m in std::iter::once(&mut s.covariates).chain(s.obs_covariates.iter_mut()) {
            if let Some(v) = m.get_mut("STUDY") {
                *v += 100.0;
            }
        }
    }
    crate::api::bind_theta_levels(&mut parsed, &text, &mut other).expect("bound on its own");
    let model = &c.prep.parsed.model;
    let codes: Vec<_> =
        crate::api::check_level_index_columns(model, &other, crate::api::LevelDataEntry::Run)
            .into_iter()
            .map(|d| d.code)
            .collect();
    assert_eq!(codes, ["E_THETA_LEVELS_DATA_MISMATCH"], "the straddle");

    for entry in ["run_covariance", "run_sir"] {
        let r = match entry {
            "run_covariance" => run_covariance(&c.fit, Some(model), Some(&other), &c.opts),
            _ => run_sir(&c.fit, Some(model), Some(&other), &c.opts),
        };
        let e = r.map(|_| ()).expect_err("refused");
        assert_eq!((e.code(), e.context()), (None, Some(entry)), "{e}");
        let err = e.to_string();
        assert!(
            err.starts_with(&format!(
                "{entry}: this population is not the one the fit was given: "
            )),
            "{err}"
        );
        assert!(err.ends_with(SUPPLIED), "{err}");
        for never in ["a fit of a model bound on this population", "level(s)"] {
            assert!(!err.contains(never), "{entry}: says {never}: {err}");
        }
    }
}

/// T3, probe C (re-measured at `045ca1b2`: SE(TVCL) 2410 against 0.310 on the fitted
/// population; 60 doses against 30, the same 300 observations). A dose-row filter
/// changes the doses and no observation; a population read without it is refused
/// on the doses of subject `1`. The refusal's dose sentence is the cause only this
/// row has.
///
/// T12 on the same fixture, both sides in one test: a fingerprint of another scheme
/// is treated as absent — the same call runs, with the warning — and the current
/// scheme refuses.
///
/// The covariate rows ride along: an extra covariate column, and a covariate value
/// changed on subject `1`, each refused with its own text (a level-free model, so
/// no level classifier fires).
///
/// Mutations — drop the dose digest from the encoding: the unfiltered population
/// passes; compare without the scheme: the stale cell is refused; drop the warning:
/// its assertion dies; delete the dose sentence: the equality dies.
#[test]
fn a_dose_row_filter_is_seen_and_a_stale_scheme_is_not_compared() {
    let c = case(Kind::DoseFilter);
    let model = &c.prep.parsed.model;
    let s = c.fit.reader_settings.clone().unwrap();
    let mut unfiltered_settings = s.clone();
    unfiltered_settings.ignore_exprs.clear();
    let unfiltered = crate::api::read_population_with(model, &unfiltered_settings, data(&c))
        .unwrap()
        .0;
    let doses = |p: &Population| p.subjects.iter().map(|s| s.doses.len()).sum::<usize>();
    assert_eq!((doses(&c.prep.population), doses(&unfiltered)), (30, 60));
    assert_eq!(
        c.prep.population.n_obs(),
        unfiltered.n_obs(),
        "observations equal"
    );

    let what = format!(
        "the doses of subject `1` differ: 2 in this population, 1 in the fit's, with the \
         same observation records.{DOSE_CAUSE}"
    );
    let e = run_covariance(&c.fit, Some(model), Some(&unfiltered), &c.opts)
        .map(|_| ())
        .expect_err("refused");
    // #1746: no `ferx check` code for a population mismatch; attributed to the entry.
    assert_eq!(
        (e.code(), e.context()),
        (None, Some("run_covariance")),
        "{e}"
    );
    let err = e.to_string();
    assert_eq!(err, refusal("run_covariance", &what, SUPPLIED));

    let mut stale = c.fit.clone();
    stale.population_fingerprint = stale
        .population_fingerprint
        .map(|f| f.with_scheme(crate::types::POPULATION_FINGERPRINT_SCHEME + 1));
    let out = run_covariance(&stale, Some(model), Some(&unfiltered), &c.opts)
        .expect("another scheme is not compared");
    assert!(
        out.warnings.iter().any(|w| w == STALE_FINGERPRINT_WARNING),
        "{:?}",
        out.warnings
    );
    // Once, however often the result is piped back in.
    let again = run_covariance(&out, Some(model), Some(&unfiltered), &c.opts).unwrap();
    let n = again
        .warnings
        .iter()
        .filter(|w| *w == STALE_FINGERPRINT_WARNING)
        .count();
    assert_eq!(n, 1);

    let mut extra = c.prep.population.clone();
    extra.covariate_names.push("EXTRA".to_string());
    let err = err_of(run_covariance(&c.fit, Some(model), Some(&extra), &c.opts));
    assert_eq!(
        err,
        refusal(
            "run_covariance",
            "its covariate columns are not the fit's: missing none, extra `EXTRA`.",
            SUPPLIED
        )
    );

    let mut heavier = c.prep.population.clone();
    heavier.subjects[0]
        .covariates
        .insert("WT".to_string(), 99.0);
    let err = err_of(run_covariance(&c.fit, Some(model), Some(&heavier), &c.opts));
    assert_eq!(
        err,
        refusal(
            "run_covariance",
            "the covariate values of subject `1` differ, with the same records and doses.",
            SUPPLIED
        )
    );

    // Review r1 #4: a record that differs with the observation count unchanged says
    // so, rather than printing "10 in this population, 10 in the fit's".
    let mut nudged = c.prep.population.clone();
    nudged.subjects[0].observations[3] *= 1.5;
    let err = err_of(run_covariance(&c.fit, Some(model), Some(&nudged), &c.opts));
    assert_eq!(
        err,
        refusal(
            "run_covariance",
            "the records of subject `1` differ with the same 10 observations: an observation \
             time, value, compartment, censoring flag or occasion, or an `EVID = 2` or reset \
             row.",
            SUPPLIED
        )
    );

    // Review r1 #3: no recorded settings and no `model_path` — the routed re-read,
    // which opens no model file — says what that read did not apply.
    let mut routed = c.fit.clone();
    routed.model_path = None;
    routed.reader_settings = None;
    let err = err_of(run_covariance(&routed, Some(model), None, &c.opts));
    assert_eq!(err, refusal("run_covariance", &what, ROUTED));
}

/// T9: an IOV model with recorded settings. `(Some(model), None)` was refused (the
/// model carries no `iov_column`); the fit now records the one it read with, so the
/// call re-reads with it and is bit-identical to `(Some, Some)`. `prepare_run` → `fit`
/// with the settings stamped, as `run_model_with_overrides` (the CLI) does it.
///
/// Mutation — drop `iov_column` from the recorded settings: the re-read carries no
/// occasions and is refused on the fingerprint.
#[test]
fn an_iov_model_re_reads_with_its_recorded_occasion_column() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("warfarin_iov.ferx");
    let data_path = dir.path().join("warfarin_iov.csv");
    std::fs::copy("examples/warfarin_iov.ferx", &model_path).unwrap();
    std::fs::copy("data/warfarin_iov.csv", &data_path).unwrap();
    let prep = crate::api::prepare_run(model_path.to_str().unwrap(), data_path.to_str())
        .expect("prepares");
    assert!(prep.parsed.model.n_kappa > 0);
    assert!(
        prep.reader_settings.iov_column.is_some(),
        "the file names its column"
    );
    let opts = FitOptions {
        outer_maxiter: 2,
        run_covariance_step: false,
        ..prep.parsed.fit_options.clone()
    };
    let mut fit = crate::api::fit(
        &prep.parsed.model,
        &prep.population,
        &prep.init_params,
        &opts,
    )
    .unwrap();
    fit.model_path = model_path.to_str().map(String::from);
    fit.data_path = Some(prep.data_path.clone());
    fit.model_hash = prep.model_hash.clone();
    fit.data_hash = prep.data_hash.clone();
    fit.reader_settings = Some(prep.reader_settings.clone());

    let model = &prep.parsed.model;
    let want = run_covariance(&fit, Some(model), Some(&prep.population), &opts).expect("oracle");
    let got = run_covariance(&fit, Some(model), None, &opts).expect("Some/None on an IOV model");
    assert_same_covariance(&got, &want, "warfarin_iov Some/None");
}

/// The resolver alone on every kind, every cell — the no-false-refusal control for
/// the fingerprint (T10's resolver half; `run_covariance`'s `assert_re_read_matches`
/// is its numeric half). Each cell resolves, and resolves to the fitted population.
/// `Level` / `LevelMedian` exercise the level write before the comparison,
/// `DoseFilter` the dose digest, `Select` a subject filter.
#[test]
fn every_legitimate_cell_resolves_to_the_fitted_population() {
    for kind in [
        Kind::Plain,
        Kind::Median,
        Kind::Level,
        Kind::LevelMedian,
        Kind::Select,
        Kind::DoseFilter,
    ] {
        let c = case(kind);
        let fp = c.fit.population_fingerprint.as_ref().expect("fit() stamps");
        let model = &c.prep.parsed.model;
        let pop = &c.prep.population;
        let cells: [(Option<&CompiledModel>, Option<&Population>, &str); 4] = [
            (None, None, "None/None"),
            (Some(model), None, "Some/None"),
            (Some(model), Some(pop), "Some/Some"),
            (None, Some(pop), "None/Some"),
        ];
        for (m, p, what) in cells {
            let inputs = resolve_fit_inputs(&c.fit, m, p, "t10")
                .unwrap_or_else(|e| panic!("{kind:?} {what}: {e}"));
            assert_eq!(
                fp.first_difference(inputs.population()),
                None,
                "{kind:?} {what}"
            );
        }
        // And the unbound parse with the fit's population: refused on its bindings
        // where it has any, never on the fingerprint.
        let bare = unbound(&c);
        if let Err(e) = resolve_fit_inputs(&c.fit, Some(&bare), Some(pop), "t10") {
            assert!(
                !e.to_string().contains("not the one the fit was given"),
                "{kind:?}: {e}"
            );
        }
    }
}

/// Review r1 #1: `run_model_with_data` hands back its population with the occasions
/// a derived `iov_occasion` rule wrote (`derive_output_occasions`, for sdtab's `OCC`),
/// after `fit()` fingerprinted the population it was given. Passing that population
/// back is passing the fit's population, and must not be refused.
///
/// Mutation — hash `occasions` / `dose_occasions` under a derived rule too: the
/// supplied cell is refused ("…the records of subject `1` differ…").
#[test]
fn the_population_a_file_fit_returns_with_derived_occasions_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("warfarin_iov_dose.ferx");
    let data_path = dir.path().join("warfarin_iov.csv");
    let text = std::fs::read_to_string("examples/warfarin_iov.ferx").unwrap();
    assert!(text.contains("iov_column = OCC"));
    let text = text.replace(
        "iov_column = OCC\n  covariance = false",
        "iov_occasion = dose\n  maxiter = 2\n  checkpoint = false\n  covariance = true",
    );
    std::fs::write(&model_path, text).unwrap();
    std::fs::copy("data/warfarin_iov.csv", &data_path).unwrap();
    let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
    let (fit, returned) = crate::api::run_model_with_data(m, Some(d)).expect("fits");
    let prep = crate::api::prepare_run(m, Some(d)).unwrap();
    // Live: the returned population carries derived occasions the reader did not.
    assert!(prep.population.subjects[0].occasions.is_empty());
    assert!(!returned.subjects[0].occasions.is_empty());

    let opts = prep.parsed.fit_options.clone();
    for (p, what) in [
        (Some(&returned), "returned"),
        (Some(&prep.population), "read"),
    ] {
        crate::estimation::fit_inputs::resolve_fit_inputs(&fit, None, p, "probe")
            .map(|_| ())
            .unwrap_or_else(|e| panic!("{what}: {e}"));
    }
    // #1783: and it is the population the fit scored — the step is the inline one.
    // At 2 outer iterations the inline step is `Computed` (measured; asserted).
    assert_eq!(fit.covariance_status, CovarianceStatus::Computed);
    let got =
        run_covariance(&fit, None, Some(&returned), &opts).expect("the returned population runs");
    let kappa = |f: &FitResult| bits(f.se_kappa.as_ref().expect("kappa SEs"));
    assert!(!kappa(&fit).is_empty());
    assert_eq!(kappa(&got), kappa(&fit), "se_kappa");
}

// ---------------------------------------------------------------------------
// #1783: a post-hoc step prepares the population as `fit()` did — occasions
// derived under the fit's recorded `iov_occasion` rule, DV log-transformed for
// `log(DV) ~ …` — after the fingerprint check. The oracle is the fit's own inline
// step: the same objective at the same point, so every cell must match it to the
// bit. There is no NONMEM spelling of "a post-hoc step on a stored fit".
//
// Fixtures: `examples/warfarin_iov.ferx` (doses at t = 0 and 120 h on every
// subject, so `dose` makes two occasions) and `examples/warfarin_ltbs.ferx`, each
// fitted once per test binary and shared.
// ---------------------------------------------------------------------------

mod fitted_population_1783 {
    use super::*;
    use crate::api::PreparedRun;
    use crate::types::IovOccasionRule;
    use std::path::PathBuf;
    use std::sync::OnceLock;

    /// Outer iterations of every #1783 fixture fit: the smallest at which every
    /// inline covariance step is `Computed` and the `dose` fit's SIR ESS exceeds 50
    /// of 100 resamples (both asserted by the tests that lean on them). Measured on
    /// macOS arm64: the ESS is 9.4 at 2, 13.6 at 3, 95.6 at 4 and 109.4 at 5; every
    /// covariance step is `Computed` from 2.
    const MAXITER: usize = 4;

    /// The SIR settings of T2.
    fn sir_opts(opts: &FitOptions) -> FitOptions {
        FitOptions {
            sir_samples: 300,
            sir_resamples: 100,
            sir_seed: Some(1),
            ..opts.clone()
        }
    }

    struct FileFit {
        _dir: tempfile::TempDir,
        model_path: PathBuf,
        data_path: PathBuf,
        fit: FitResult,
        /// The population `run_model_with_data` returned (derived occasions written
        /// for sdtab); `None` for a `fit_from_files` fixture.
        returned: Option<Population>,
        /// `prepare_run` on the same files: the population as read, unlabelled under
        /// a derived rule.
        prep: PreparedRun,
        /// The model file's `[fit_options]`, what a caller passes to the step.
        opts: FitOptions,
    }

    fn write_case(
        model_src: &str,
        data_src: &str,
        edits: &[(&str, &str)],
        data_edit: Option<fn(&str) -> String>,
    ) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.ferx");
        let data_path = dir.path().join("data.csv");
        let mut text = std::fs::read_to_string(model_src).unwrap();
        for (from, to) in edits {
            assert!(text.contains(from), "{model_src}: `{from}`");
            text = text.replacen(from, to, 1);
        }
        std::fs::write(&model_path, text).unwrap();
        let data = std::fs::read_to_string(data_src).unwrap();
        let data = match data_edit {
            Some(f) => f(&data),
            None => data,
        };
        std::fs::write(&data_path, data).unwrap();
        (dir, model_path, data_path)
    }

    /// A fit through `run_model_with_data`, whose rule is the model file's.
    fn run_fit(
        model_src: &str,
        data_src: &str,
        edits: &[(&str, &str)],
        data_edit: Option<fn(&str) -> String>,
    ) -> FileFit {
        let (dir, model_path, data_path) = write_case(model_src, data_src, edits, data_edit);
        let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
        let (fit, returned) = crate::api::run_model_with_data(m, Some(d)).expect("fits");
        let prep = crate::api::prepare_run(m, Some(d)).expect("prepares");
        let opts = prep.parsed.fit_options.clone();
        FileFit {
            _dir: dir,
            model_path,
            data_path,
            fit,
            returned: Some(returned),
            prep,
            opts,
        }
    }

    fn iov(rule: &str) -> FileFit {
        let to =
            format!("{rule}\n  covariance = true\n  maxiter = {MAXITER}\n  checkpoint = false");
        run_fit(
            "examples/warfarin_iov.ferx",
            "data/warfarin_iov.csv",
            &[("iov_column = OCC\n  covariance = false", &to)],
            None,
        )
    }

    /// `iov_occasion = dose`.
    fn dose() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| iov("iov_occasion = dose"))
    }

    /// The control: the same occasions from the data's `OCC` column.
    fn column() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| iov("iov_column = OCC"))
    }

    /// The time-window edge of T3. `dose` puts the 72 h and 96 h samples in occasion
    /// 0 (the second dose is at 120 h); `time(60)` puts them in occasion 1.
    const EDGE: f64 = 60.0;

    /// A model file that states **no** occasion rule and no `iov_column`, fitted
    /// through `fit_from_files` with `iov_occasion = time(60)`: the fit's rule is
    /// not the file's (`fit_from_files` ignores the file's `[fit_options]`).
    fn time_window() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| {
            let (dir, model_path, data_path) = write_case(
                "examples/warfarin_iov.ferx",
                "data/warfarin_iov.csv",
                &[(
                    "iov_column = OCC\n  covariance = false",
                    "covariance = true",
                )],
                None,
            );
            let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
            let prep = crate::api::prepare_run(m, Some(d)).expect("prepares");
            let opts = prep.parsed.fit_options.clone();
            assert_eq!(opts.iov_occasion, IovOccasionRule::Column);
            assert!(opts.iov_column.is_none());
            let fit_opts = FitOptions {
                iov_occasion: IovOccasionRule::TimeWindows(vec![EDGE]),
                outer_maxiter: MAXITER,
                run_covariance_step: true,
                ..opts.clone()
            };
            let fit = crate::api::fit_from_files(m, Some(d), None, Some(fit_opts)).expect("fits");
            FileFit {
                _dir: dir,
                model_path,
                data_path,
                fit,
                returned: None,
                prep,
                opts,
            }
        })
    }

    fn ltbs_fit(edits: &[(&str, &str)], data_edit: Option<fn(&str) -> String>) -> FileFit {
        let iters = format!("maxiter = {MAXITER}\n  checkpoint = false");
        let mut all = vec![("maxiter    = 300", iters.as_str())];
        all.extend_from_slice(edits);
        run_fit(
            "examples/warfarin_ltbs.ferx",
            "data/warfarin_ltbs.csv",
            &all,
            data_edit,
        )
    }

    fn ltbs() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| ltbs_fit(&[], None))
    }

    /// `data/warfarin_ltbs.csv` with every observed DV replaced by its natural log,
    /// written in the shortest form that round-trips, so the reader parses exactly
    /// the `f64` that `log_transform_observations` makes of the natural-scale DV.
    fn log_dv(csv: &str) -> String {
        let mut out = String::new();
        for (i, line) in csv.lines().enumerate() {
            let mut cols: Vec<String> = line.split(',').map(str::to_string).collect();
            if i == 0 {
                assert_eq!(&cols[..3], ["ID", "TIME", "DV"]);
            } else if cols[2] != "." {
                let dv: f64 = cols[2].parse().unwrap();
                assert!(dv > 0.0, "the fixture has no DV ≤ 0 to floor");
                cols[2] = format!("{:?}", dv.ln());
            }
            out.push_str(&cols.join(","));
            out.push('\n');
        }
        out
    }

    fn pre_logged() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| {
            ltbs_fit(
                &[("log(DV) ~ additive(ADD_LOG)", "DV ~ log_additive(ADD_LOG)")],
                Some(log_dv),
            )
        })
    }

    /// The covariance, and every SE block, of two runs to the bit.
    fn assert_same_cov(got: &FitResult, want: &FitResult, what: &str) {
        assert_same_covariance(got, want, what);
        let se = |v: &Option<Vec<f64>>| v.as_deref().map(bits);
        assert_eq!(se(&got.se_omega), se(&want.se_omega), "{what}: se_omega");
        assert_eq!(se(&got.se_sigma), se(&want.se_sigma), "{what}: se_sigma");
        assert_eq!(se(&got.se_kappa), se(&want.se_kappa), "{what}: se_kappa");
    }

    /// The population cells of `model = None`: re-read, the population
    /// `run_model_with_data` returned (labelled), and the one `prepare_run` reads
    /// (unlabelled under a derived rule).
    fn cells(f: &FileFit) -> Vec<(Option<&Population>, &'static str)> {
        let mut v = vec![(None, "(None, None)")];
        if let Some(r) = &f.returned {
            v.push((Some(r), "(None, Some(returned))"));
        }
        v.push((Some(&f.prep.population), "(None, Some(read))"));
        v
    }

    fn has_kappa_se(f: &FitResult) -> bool {
        f.se_kappa.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// T1. Under `iov_occasion = dose`, `run_covariance` on every population cell is
    /// the inline covariance step to the bit; the `iov_column` fit is the control
    /// and is unchanged.
    ///
    /// Mutations — drop the derivation from `fitted_population`'s post-hoc call:
    /// `(None, None)` and the read cell are `Failed` (measured on #1783: `Ok`, all
    /// SEs `None`). Prepare only when `population = None`: the read cell is `Failed`.
    #[test]
    fn derived_rule_posthoc_matches_inline() {
        for (f, what) in [(dose(), "dose"), (column(), "column")] {
            assert_eq!(
                f.fit.covariance_status,
                CovarianceStatus::Computed,
                "{what}"
            );
            assert!(
                has_kappa_se(&f.fit),
                "{what}: the inline step has kappa SEs"
            );
            for (p, cell) in cells(f) {
                let got = run_covariance(&f.fit, None, p, &f.opts)
                    .unwrap_or_else(|e| panic!("{what} {cell}: {e}"));
                assert_same_cov(&got, &f.fit, &format!("{what} {cell}"));
            }
        }
        // Live: the read population of the `dose` fit carries no labels, the returned
        // one does, and the control's read population has its column's.
        assert!(dose().prep.population.subjects[0].occasions.is_empty());
        assert!(!dose().returned.as_ref().unwrap().subjects[0]
            .occasions
            .is_empty());
        assert!(!column().prep.population.subjects[0].occasions.is_empty());
        assert_eq!(dose().fit.iov_occasion, Some(IovOccasionRule::PerDose));
        assert_eq!(column().fit.iov_occasion, Some(IovOccasionRule::Column));
    }

    /// T2. `run_sir` weights its draws with the same objective in every cell: the
    /// ESS and the kappa CIs agree to the bit, and the proposal is not degenerate.
    ///
    /// Mutations — those of T1: the unlabelled cells score kappa against one
    /// occasion and the ESS collapses (measured on #1783: 1.0, CI width 0).
    #[test]
    fn derived_rule_run_sir_matches_across_cells() {
        let f = dose();
        let opts = sir_opts(&f.opts);
        let runs: Vec<(FitResult, &str)> = cells(f)
            .into_iter()
            .map(|(p, cell)| {
                let r = run_sir(&f.fit, None, p, &opts).unwrap_or_else(|e| panic!("{cell}: {e}"));
                (r, cell)
            })
            .collect();
        let ess = runs[0].0.sir_ess.expect("ESS");
        assert!(ess > 50.0, "ESS {ess}: the proposal is degenerate");
        let ci = |r: &FitResult| -> Vec<(u64, u64)> {
            r.sir_ci_kappa
                .as_ref()
                .expect("kappa CI")
                .iter()
                .map(|(a, b)| (a.to_bits(), b.to_bits()))
                .collect()
        };
        assert!(!ci(&runs[0].0).is_empty());
        for (r, cell) in &runs[1..] {
            assert_eq!(
                r.sir_ess.map(f64::to_bits),
                Some(ess.to_bits()),
                "{cell}: ESS"
            );
            assert_eq!(ci(r), ci(&runs[0].0), "{cell}: kappa CI");
        }
    }

    /// T3. The rule a post-hoc step derives with is the one the fit recorded, not
    /// the model file's: here the file states none and the fit ran `time(60)`.
    ///
    /// Mutations — hard-code `PerDose` in the post-hoc resolution: the partition
    /// differs (asserted below) and the SEs move. Prefer the file's rule over the
    /// recorded one: the file's is `Column`, so nothing is derived and the step is
    /// `Failed`.
    #[test]
    fn time_window_rule_is_the_recorded_one() {
        let f = time_window();
        assert_eq!(f.fit.covariance_status, CovarianceStatus::Computed);
        assert!(has_kappa_se(&f.fit));
        assert_eq!(
            f.fit.iov_occasion,
            Some(IovOccasionRule::TimeWindows(vec![EDGE]))
        );
        // The precondition of the first mutation: `time(60)` and `dose` partition the
        // fixture differently.
        let derive = |rule: &IovOccasionRule| {
            let mut p = f.prep.population.clone();
            crate::api::apply_iov_occasion_rule(&mut p, rule, false, &mut Vec::new());
            p.subjects[0].occasions.clone()
        };
        assert_ne!(
            derive(&IovOccasionRule::TimeWindows(vec![EDGE])),
            derive(&IovOccasionRule::PerDose)
        );
        for (p, cell) in cells(f) {
            let got =
                run_covariance(&f.fit, None, p, &f.opts).unwrap_or_else(|e| panic!("{cell}: {e}"));
            assert_same_cov(&got, &f.fit, cell);
        }
    }

    /// The caller's options with `inner_tol` pinned to the fit's LTBS tightening,
    /// until #1786 makes the post-hoc steps apply it themselves (without it,
    /// measured on #1783: rel 1.7e-9 on se_theta[0]).
    fn ltbs_opts(f: &FileFit) -> FitOptions {
        FitOptions {
            inner_tol: FitOptions::LTBS_FIT_INNER_TOL,
            ..f.opts.clone()
        }
    }

    /// T4. On a `log(DV) ~ additive` fit the step scores log predictions against the
    /// log-transformed DV, as `fit()` did.
    ///
    /// Mutation — drop the log transform from `fitted_population`'s post-hoc call
    /// (natural DV): se_theta 777× the inline one (measured on #1783: 5.5113 vs
    /// 0.0070974).
    #[test]
    fn ltbs_posthoc_matches_inline() {
        let f = ltbs();
        assert!(f.prep.parsed.model.log_transform && !f.prep.parsed.model.dv_pre_logged);
        assert_eq!(f.fit.covariance_status, CovarianceStatus::Computed);
        for (p, cell) in cells(f) {
            let got = run_covariance(&f.fit, None, p, &ltbs_opts(f))
                .unwrap_or_else(|e| panic!("{cell}: {e}"));
            assert_same_cov(&got, &f.fit, cell);
        }
    }

    /// T5. `DV ~ log_additive` on data already on the log scale is not logged
    /// again: the post-hoc step matches the inline one, and — since the one helper
    /// serves both — the fit itself matches the `log(DV) ~ additive` fit of the
    /// natural-scale data (the same `f64` DVs reach the objective).
    ///
    /// Mutation — `needs_dv_log = model.log_transform` (ignoring `dv_pre_logged`):
    /// the pre-logged fit logs its DV twice and its OFV leaves the LTBS fit's.
    #[test]
    fn pre_logged_dv_is_not_logged_twice() {
        let f = pre_logged();
        assert!(f.prep.parsed.model.dv_pre_logged);
        assert_eq!(f.fit.covariance_status, CovarianceStatus::Computed);
        assert_eq!(f.fit.ofv.to_bits(), ltbs().fit.ofv.to_bits(), "OFV");
        for (p, cell) in cells(f) {
            let got = run_covariance(&f.fit, None, p, &ltbs_opts(f))
                .unwrap_or_else(|e| panic!("{cell}: {e}"));
            assert_same_cov(&got, &f.fit, cell);
        }
    }

    /// T6. The rule survives `save_fit` → `load_fit`, and the loaded fit's
    /// covariance step on the bundled (unlabelled) population is the one on the
    /// labelled population `run_model_with_data` returned. Not the inline step: a
    /// loaded fit rebuilds its Ω factor from `fit.omega` and starts from the EBEs
    /// `ebes.csv` rounds (measured: 4.8e-7 at most), and its covariance differs from
    /// the inline one by up to 5.5e-4 relative (`foce.qmd` documents the
    /// reloaded-fit match as up to FD noise).
    ///
    /// Mutation — drop the wire field (on save or on load): the loaded rule is
    /// `None`, asserted against `Some(PerDose)` first. (The step itself would then
    /// fall back to the model file's `dose` and still match.)
    #[test]
    fn fitrx_roundtrips_the_occasion_rule() {
        let f = dose();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dose.fitrx");
        let src = std::fs::read_to_string(&f.model_path).unwrap();
        let opts = crate::io::fitrx::SaveFitOptions {
            include_data: Some(f.data_path.clone()),
        };
        crate::io::fitrx::save_fit(&f.fit, f.returned.as_ref().unwrap(), &src, &path, opts)
            .unwrap();
        let loaded = crate::io::fitrx::load_fit(&path).unwrap();
        assert_eq!(loaded.fit.iov_occasion, Some(IovOccasionRule::PerDose));
        let pop = loaded.population.as_ref().expect("data bundled");
        assert!(
            pop.subjects[0].occasions.is_empty(),
            "the bundle is unlabelled"
        );
        let got = run_covariance(&loaded.fit, None, Some(pop), &f.opts).unwrap();
        let want = run_covariance(&loaded.fit, None, f.returned.as_ref(), &f.opts).unwrap();
        assert!(has_kappa_se(&want));
        assert_same_cov(&got, &want, "loaded");
    }

    /// The fit as a bundle saved before #1783 reads it: no recorded rule, and with
    /// `keep_fingerprint = false` none from #1685 either.
    fn unrecorded(f: &FileFit, keep_fingerprint: bool) -> FitResult {
        let mut fit = f.fit.clone();
        fit.iov_occasion = None;
        if !keep_fingerprint {
            fit.population_fingerprint = None;
        }
        fit
    }

    /// The note a step carries when it derived the occasions with the model
    /// file's rule (review r1 #1), in full.
    const W_FILE_DOSE: &str = "this fit records no IOV occasion rule (it was saved before \
                               ferx recorded one, #1783), so the occasions were derived with \
                               the model file's `iov_occasion = dose`. A fit made through \
                               `fit_from_files` ran under its caller's rule, not the file's: \
                               if that was another rule, these results are wrong. Set \
                               `fit.iov_occasion` to the rule the fit ran with.";

    fn has_file_note(r: &FitResult) -> bool {
        r.warnings
            .iter()
            .any(|w| w.contains("derived with the model file's"))
    }

    /// T7. A fit that records no rule takes the model file's when the step reads
    /// the file (`model = None`): the `dose` fit's file states `dose`. Right here
    /// (a `run_model_with_data` fit runs under its file's rule), and the result says
    /// which rule it guessed; the recorded fit's says nothing.
    ///
    /// Mutations — skip the file fallback: the unlabelled cells are refused. Drop the
    /// note: the `contains` fails. Push it for a recorded rule too, or for a guessed
    /// `Column`: the controls fail.
    #[test]
    fn unrecorded_rule_takes_the_model_file_rule() {
        let f = dose();
        let fit = unrecorded(f, true);
        for (p, cell) in cells(f) {
            let got =
                run_covariance(&fit, None, p, &f.opts).unwrap_or_else(|e| panic!("{cell}: {e}"));
            assert_same_cov(&got, &f.fit, cell);
            assert!(
                got.warnings.iter().any(|w| w == W_FILE_DOSE),
                "{cell}: {:?}",
                got.warnings
            );
            let recorded = run_covariance(&f.fit, None, p, &f.opts).unwrap();
            assert!(!has_file_note(&recorded), "{cell}: recorded rule");
        }
        // A guessed `Column` derives nothing — the labels are the column's — so it
        // is no guess about the occasions and carries no note.
        let c = column();
        let got = run_covariance(&unrecorded(c, true), None, None, &c.opts).unwrap();
        assert_same_cov(&got, &c.fit, "column, unrecorded");
        assert!(
            !has_file_note(&got),
            "column, unrecorded: {:?}",
            got.warnings
        );
    }

    /// A model file stating `iov_occasion = dose`, fitted through `fit_from_files`
    /// under the caller's `time(60)` (review r1 #1's geometry).
    fn dose_file_time_fit() -> &'static FileFit {
        static F: OnceLock<FileFit> = OnceLock::new();
        F.get_or_init(|| {
            let (dir, model_path, data_path) = write_case(
                "examples/warfarin_iov.ferx",
                "data/warfarin_iov.csv",
                &[(
                    "iov_column = OCC\n  covariance = false",
                    "iov_occasion = dose\n  covariance = true",
                )],
                None,
            );
            let (m, d) = (model_path.to_str().unwrap(), data_path.to_str().unwrap());
            let prep = crate::api::prepare_run(m, Some(d)).expect("prepares");
            let opts = prep.parsed.fit_options.clone();
            assert_eq!(opts.iov_occasion, IovOccasionRule::PerDose);
            let fit_opts = FitOptions {
                iov_occasion: IovOccasionRule::TimeWindows(vec![EDGE]),
                outer_maxiter: MAXITER,
                run_covariance_step: true,
                ..opts.clone()
            };
            let fit = crate::api::fit_from_files(m, Some(d), None, Some(fit_opts)).expect("fits");
            FileFit {
                _dir: dir,
                model_path,
                data_path,
                fit,
                returned: None,
                prep,
                opts,
            }
        })
    }

    /// Review r1 #1. An old bundle of a `fit_from_files` fit whose caller's rule
    /// (`time(60)`) was not its file's (`dose`): the fallback derives with the wrong
    /// rule, the derived labels are not fingerprinted, and the step returns
    /// `Computed` with wrong SEs (measured by the reviewer at `f8a394ad`: se_kappa
    /// 412.47 against the inline 0.10506). It cannot be refused — the common old
    /// bundle, T7's, is right — so the result names the rule it guessed. The
    /// straddle: the same fit with its rule recorded matches inline, with no note.
    ///
    /// Mutation — drop the note: the `contains` fails on both cells.
    #[test]
    fn a_guessed_rule_is_named_on_the_result() {
        let f = dose_file_time_fit();
        assert_eq!(f.fit.covariance_status, CovarianceStatus::Computed);
        assert!(has_kappa_se(&f.fit));
        let fit = unrecorded(f, true);
        for (p, cell) in cells(f) {
            let got = run_covariance(&fit, None, p, &f.opts).unwrap();
            assert!(
                got.warnings.iter().any(|w| w == W_FILE_DOSE),
                "{cell}: {:?}",
                got.warnings
            );
            // The note is not a false alarm here: the guess is wrong.
            assert_ne!(
                got.se_kappa.as_deref().map(bits),
                f.fit.se_kappa.as_deref().map(bits),
                "{cell}: the file's rule reproduced the fit's"
            );
            let recorded = run_covariance(&f.fit, None, p, &f.opts).unwrap();
            assert_same_cov(&recorded, &f.fit, cell);
            assert!(!has_file_note(&recorded), "{cell}: recorded rule");
        }
    }

    const F_LEAD: &str = "run_covariance: this fit records no IOV occasion rule (it was saved \
                          before ferx recorded one, #1783), and the population carries no \
                          occasion labels, so the per-occasion kappas cannot be assigned to \
                          occasions.";
    const F_DERIVED: &str = " Its population fingerprint shows the fit derived its occasions \
                             from a model-side `iov_occasion` rule (`dose` or `time(...)`), \
                             which it did not record.";
    const F_FILE: &str = " The model file's `[fit_options]` sets no `iov_occasion` to fall \
                          back on.";
    const F_FIX: &str = " Pass the population `run_model_with_data` returned, which carries \
                         the occasion labels the fit ran with, set `fit.iov_occasion` to the \
                         rule the fit ran with, or refit so the rule is recorded.";

    /// T8, cell F. A fit that records no rule, on a population with no occasion
    /// labels and no rule to derive them with, is refused rather than run with every
    /// kappa on one occasion. The message's input space is (no fingerprint / one
    /// saying the fit derived its occasions / one saying it did not, reachable only
    /// with another scheme, which is not checked) × (was the model file read,
    /// stating no rule?); every cell is reached here and asserted whole, so deleting a sentence
    /// reddens each cell that carries it, and a sentence leaking into a cell that
    /// must not carry it reddens that cell. The twin: the labelled population
    /// `run_model_with_data` returned runs, and matches the inline step.
    ///
    /// F3 (a recorded `Column` rule, an unlabelled population) is not reachable: the
    /// fingerprint hashes the labels and refuses first, asserted last.
    ///
    /// Mutation — drop the guard: every refused cell returns `Ok` with `Failed`.
    #[test]
    fn unrecorded_rule_without_labels_is_refused() {
        // No file read: `model = Some`, `population = Some(read)`.
        let f = dose();
        let m = &f.prep.parsed.model;
        let read = &f.prep.population;
        for (keep_fp, want) in [
            (false, format!("{F_LEAD}{F_FIX}")),
            (true, format!("{F_LEAD}{F_DERIVED}{F_FIX}")),
        ] {
            let fit = unrecorded(f, keep_fp);
            assert_eq!(
                err_of(run_covariance(&fit, Some(m), Some(read), &f.opts)),
                want,
                "fingerprint kept: {keep_fp}"
            );
            // The twin: the labelled population runs.
            let got = run_covariance(&fit, Some(m), f.returned.as_ref(), &f.opts).unwrap();
            assert_same_cov(&got, &f.fit, "returned");
        }
        // A fingerprint of another scheme is not checked, so it reaches this refusal
        // too. The one that says nothing was derived (the column fit's) must not add
        // the sentence; the `dose` fit's own must. Without this pair a gate on the
        // fingerprint's presence alone passes (#1783 mutation sweep, M12).
        let stale = |fp: &crate::types::PopulationFingerprint| {
            fp.clone()
                .with_scheme(crate::types::POPULATION_FINGERPRINT_SCHEME + 1)
        };
        for (fp, want) in [
            (
                column().fit.population_fingerprint.as_ref().unwrap(),
                format!("{F_LEAD}{F_FIX}"),
            ),
            (
                f.fit.population_fingerprint.as_ref().unwrap(),
                format!("{F_LEAD}{F_DERIVED}{F_FIX}"),
            ),
        ] {
            let mut fit = unrecorded(f, false);
            fit.population_fingerprint = Some(stale(fp));
            assert!(!fit.population_fingerprint.as_ref().unwrap().is_current());
            assert_eq!(
                err_of(run_covariance(&fit, Some(m), Some(read), &f.opts)),
                want,
                "stale fingerprint, derived: {}",
                fp.occasions_derived()
            );
        }
        // The file read (`model = None`), stating no rule: the `time(60)` fit.
        let t = time_window();
        for (keep_fp, want) in [
            (false, format!("{F_LEAD}{F_FILE}{F_FIX}")),
            (true, format!("{F_LEAD}{F_DERIVED}{F_FILE}{F_FIX}")),
        ] {
            let fit = unrecorded(t, keep_fp);
            for p in [None, Some(&t.prep.population)] {
                assert_eq!(
                    err_of(run_covariance(&fit, None, p, &t.opts)),
                    want,
                    "fingerprint kept: {keep_fp}, population supplied: {}",
                    p.is_some()
                );
            }
        }
        // F3: the column fit, given its population with the labels removed.
        let c = column();
        let mut stripped = c.prep.population.clone();
        for s in &mut stripped.subjects {
            s.occasions.clear();
            s.dose_occasions.clear();
        }
        let e = err_of(run_covariance(&c.fit, None, Some(&stripped), &c.opts));
        assert!(e.contains("not the one the fit was given"), "{e}");
    }
}

/// #1773. The resolver's two level writes keep the binder's code through
/// `run_sir` / `run_covariance`: `E_THETA_LEVEL_BINDING` on `parameters`, with the
/// entry point as context and the binder's text unchanged. Both cells are a level
/// the fit never estimated, from each side: the model rebuilt from the file and laid
/// out on the fit, given the design population (`STUDY=4` is unseen); and a lent
/// model bound on the design, given the fit's data re-read (`STUDY=3` is unseen to
/// it). `run_sir` is refused in the resolver, before it reads the covariance matrix
/// this fit does not carry.
///
/// Mutations — `.to_string()` the code away at either write (`bind_from_fit_on` or
/// `write_fitted_level_columns` in `resolve_fit_inputs`): that cell's `code()` is
/// `None`.
#[test]
fn a_level_refusal_in_the_resolver_carries_the_binder_code() {
    let c = case(Kind::Level);
    let design = super::test_fixtures::design(&c);
    // A lent model needs a fit with no recorded bindings to compare against.
    let mut legacy = c.fit.clone();
    legacy.data_bindings = Default::default();
    let lent = &design.parsed.model;
    for entry in ["run_covariance", "run_sir"] {
        let run = |fit: &FitResult, m: Option<&CompiledModel>, p: Option<&Population>| match entry {
            "run_covariance" => run_covariance(fit, m, p, &c.opts),
            _ => run_sir(fit, m, p, &c.opts),
        };
        let cells = [
            (
                "rebuilt",
                run(&c.fit, None, Some(&design.population)),
                "`STUDY=4`",
            ),
            ("lent", run(&legacy, Some(lent), None), "`STUDY=3`"),
        ];
        for (what, r, unseen) in cells {
            let e = r.map(|_| ()).expect_err(what);
            assert_eq!(
                e.code(),
                Some("E_THETA_LEVEL_BINDING"),
                "{entry}/{what}: {e}"
            );
            assert_eq!(e.block(), Some("parameters"), "{entry}/{what}: {e}");
            assert_eq!(e.context(), Some(entry), "{entry}/{what}: {e}");
            assert!(
                e.to_string().starts_with(&format!(
                    "{entry}: theta SHIFT[STUDY]: the design has 1 level(s) the fit estimated \
                     no theta for: {unseen}."
                )),
                "{entry}/{what}: {e}"
            );
        }
    }
}

/// #1791 review r1, row 2. A fit that recorded no data bindings, rebuilt from its
/// model file (`model = None`), is refused by the from-fit binder with a code: the
/// level code on a level model, the statistics code on a stats-only one. This is
/// the class `warnings.qmd` names, run rather than traced.
///
/// Mutation — `.to_string()` the code away at the resolver's `bind_from_fit_on`:
/// both cells' `code()` is `None`.
#[test]
fn a_fit_without_bindings_is_refused_with_the_half_code() {
    for (kind, code) in [
        (Kind::Level, "E_THETA_LEVEL_BINDING"),
        (Kind::Median, "E_COVARIATE_STATS_BINDING"),
    ] {
        let c = case(kind);
        let mut fit = c.fit.clone();
        fit.data_bindings = Default::default();
        let e = run_covariance(&fit, None, Some(&c.prep.population), &c.opts)
            .map(|_| ())
            .expect_err("no bindings to rebuild from");
        assert_eq!(e.code(), Some(code), "{kind:?}: {e}");
        assert_eq!(e.context(), Some("run_covariance"), "{kind:?}: {e}");
        assert!(
            e.to_string()
                .contains("this fit carries no data-derived bindings"),
            "{kind:?}: {e}"
        );
    }
}

// ── #426: the scoring record ───────────────────────────────────────────────

/// A record with every field off its [`FitOptions::default`] value, so each resolve line has
/// a value to take that default caller options do not already hold.
fn off_default_scoring() -> ScoringSettings {
    ScoringSettings {
        inner_maxiter: 17,
        inner_tol: 3e-4,
        inner_restarts: 4,
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

/// #426 T12, the resolve lines. Default caller options take every recorded field, each
/// checked under its own name after its premise (the record differs from the default); a
/// caller's non-default value wins. Both sides of the "caller left the default?" gate in one
/// test. The three `bool`s have one non-default value, which record and caller share, so
/// their `caller` check holds either way; the other ten catch an inverted gate. This is the
/// only test that reaches the `inner_restarts` and `ode_stiff_abort_after` lines: no T9
/// fixture moves the covariance with them (`tests/run_covariance_scoring_record.rs`). The
/// record is destructured without `..`, so a new field does not compile here until it has a
/// row. Mutations: delete any one resolve line (its `record` check dies, naming the field);
/// invert the gate (a `caller` check dies).
#[test]
fn scoring_record_fills_each_field_the_caller_left_default() {
    let rec = off_default_scoring();
    let d = FitOptions::default();
    let resolved = with_scoring_record(Some(&rec), &d);
    let caller = FitOptions {
        inner_maxiter: 18,
        inner_tol: 4e-4,
        inner_restarts: 5,
        mu_referencing: false,
        n_agq: 6,
        inner_optimizer: InnerOptimizer::NelderMead,
        ebe_warm_start: true,
        ode_reltol: 2e-7,
        ode_abstol: 2e-9,
        ode_max_steps: 778,
        ode_method: crate::ode::OdeMethod::Rodas4,
        ode_stiff_abort_after: Some(10),
        ode_auto_switch: false,
        ..FitOptions::default()
    };
    let kept = with_scoring_record(Some(&rec), &caller);
    let ScoringSettings {
        inner_maxiter: _,
        inner_tol: _,
        inner_restarts: _,
        mu_referencing: _,
        n_agq: _,
        inner_optimizer: _,
        ebe_warm_start: _,
        ode_reltol: _,
        ode_abstol: _,
        ode_max_steps: _,
        ode_method: _,
        ode_stiff_abort_after: _,
        ode_auto_switch: _,
    } = rec.clone();
    macro_rules! field {
        ($f:ident) => {
            let name = stringify!($f);
            assert_ne!(rec.$f, d.$f, "{name}: premise — the record is off-default");
            assert_eq!(
                resolved.$f, rec.$f,
                "{name}: record — a default caller takes it"
            );
            assert_eq!(
                kept.$f, caller.$f,
                "{name}: caller — a non-default caller keeps it"
            );
        };
    }
    field!(inner_maxiter);
    field!(inner_tol);
    field!(inner_restarts);
    field!(mu_referencing);
    field!(n_agq);
    field!(inner_optimizer);
    field!(ebe_warm_start);
    field!(ode_reltol);
    field!(ode_abstol);
    field!(ode_max_steps);
    field!(ode_method);
    field!(ode_stiff_abort_after);
    field!(ode_auto_switch);
    // Nothing outside the record moves: a step setting, and the tally the record leaves out.
    assert_eq!(resolved.cov_inner_tol, d.cov_inner_tol);
    assert_eq!(
        resolved.min_obs_for_convergence_check,
        d.min_obs_for_convergence_check
    );
}

/// #426 T12, the fallback order of `resolve_scoring_options`: the fit's own record, else the
/// SIR record's scoring half (a `.fitrx` written between #1758 and #426), else the caller's
/// options unchanged. Mutations: swap the order (the first assertion takes the SIR record);
/// drop the SIR fallback (the second gets the caller's defaults); return anything but
/// `options` when neither exists (the third, whose caller is off-default).
#[test]
fn resolve_scoring_options_prefers_the_fits_record_then_the_sir_records() {
    let fit_rec = off_default_scoring();
    let sir_rec = ScoringSettings {
        inner_maxiter: 99,
        ..off_default_scoring()
    };
    let mut fit = crate::types::test_helpers::minimal_fit_result();
    fit.scoring_settings = Some(fit_rec.clone());
    fit.sir_settings = Some(crate::estimation::sir::SirSettings {
        scoring: sir_rec.clone(),
        ..Default::default()
    });
    let d = FitOptions::default();
    assert_eq!(
        ScoringSettings::from_options(&resolve_scoring_options(&fit, &d)),
        fit_rec
    );
    fit.scoring_settings = None;
    assert_eq!(
        ScoringSettings::from_options(&resolve_scoring_options(&fit, &d)),
        sir_rec
    );
    fit.sir_settings = None;
    let caller = FitOptions {
        inner_tol: 1e-7,
        ode_method: crate::ode::OdeMethod::Rodas4,
        ..FitOptions::default()
    };
    assert_eq!(
        ScoringSettings::from_options(&resolve_scoring_options(&fit, &caller)),
        ScoringSettings::from_options(&caller)
    );
}
