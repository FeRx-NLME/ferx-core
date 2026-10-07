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

    let (n_file, n_fit) = (
        file_read.subjects[0].observations.len(),
        fitted.subjects[0].observations.len(),
    );
    assert_eq!((n_file, n_fit), (10, 8));
    let records = format!(
        "the observation records of subject `1` differ: {n_file} in this population, {n_fit} \
         in the fit's."
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
    // A stale fingerprint's recorded settings are not replayed either: the re-read
    // falls back to the model file's settings, which state the same filter here.
    run_covariance(&stale, None, None, &c.opts).expect("legacy re-read");

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
/// supplied cell is refused ("…the observation records of subject `1` differ…").
#[test]
fn the_population_a_file_fit_returns_with_derived_occasions_is_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("warfarin_iov_dose.ferx");
    let data_path = dir.path().join("warfarin_iov.csv");
    let text = std::fs::read_to_string("examples/warfarin_iov.ferx").unwrap();
    assert!(text.contains("iov_column = OCC"));
    let text = text.replace(
        "iov_column = OCC",
        "iov_occasion = dose\n  maxiter = 2\n  checkpoint = false",
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
    run_covariance(&fit, None, Some(&returned), &opts).expect("the returned population runs");
}
