//! Tier-1 tests for the ruvsearch logic (#1182).
//!
//! Every test drives [`search`] with a *scripted* fitter — a table of OFVs
//! keyed by candidate id — so what is under test is the iteration logic:
//! which candidate an iteration selects, which families it retires, what the
//! proportional base and the final comparison do, what the pre-screen picks
//! and how its estimate seeds the refit. None of that is observable through
//! real fits without putting every test in the slow tier. The path from a
//! `.ferxsearch` file through the edits, the runner and `fit()` is
//! `tests/ruvsearch_end_to_end.rs`.
//!
//! The fixture OFVs are chosen against the `df = 1` cutoff at `p = 0.001`,
//! 10.83: a drop of 15 is significant, a drop of 5 is not.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use ferx_core::edit::{ErrorForm, ErrorSpecText, ModelText};
use ferx_core::{DoseEvent, Population, StrictnessVerdict, Subject};

use super::*;
use crate::search::test_support::converged_fit;
use crate::search::{CandidateError, CandidateResult, RunReport, SearchConfig};

const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method = focei
";

const BASE_N: usize = 7;

/// A population with doses at 0 and 24 and observations spread over both
/// intervals, so TAD has a distribution to cut, and DV values with a known
/// minimum (2.0) for the combined σ init.
fn population() -> Population {
    let subj = |id: &str, shift: f64| Subject {
        id: id.into(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(24.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0 + shift, 4.0, 12.0, 25.0 + shift, 30.0, 47.0],
        observations: vec![10.0, 8.0, 4.0, 9.0, 6.0, 2.0],
        obs_cmts: vec![1; 6],
        cens: vec![0; 6],
        ..Default::default()
    };
    Population {
        subjects: vec![subj("1", 0.0), subj("2", 0.5)],
        covariate_names: vec![],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn space_from(src: &str, options: &RuvsearchOptions, interaction_ok: bool) -> Space {
    let text = ModelText::parse(src).unwrap();
    let spec = ErrorSpecText::read(&text).unwrap().unwrap();
    Space::build(text, spec, population(), interaction_ok, options)
}

fn space(options: &RuvsearchOptions) -> Space {
    space_from(BASE, options, true)
}

/// The scripted fitter. OFVs are keyed by candidate id; a candidate not in
/// the table gets `fallback`. Parameter counts are the base count plus one
/// per feature the candidate's error model carries.
struct Script {
    ofv: HashMap<String, f64>,
    fallback: f64,
    failing: Vec<String>,
    erroring: Vec<String>,
    /// Estimates a screening fit reports, by candidate id:
    /// `(theta_names, theta, eta_names, omega diagonal)`.
    screen_estimates: HashMap<String, (Vec<String>, Vec<f64>, Vec<String>, Vec<f64>)>,
    cancel_after: Option<usize>,
    calls: Mutex<Vec<(String, Vec<Candidate>, Option<usize>)>>,
}

impl Script {
    fn new(table: &[(&str, f64)]) -> Script {
        Script {
            ofv: table.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            fallback: 1000.0,
            failing: Vec::new(),
            erroring: Vec::new(),
            screen_estimates: HashMap::new(),
            cancel_after: None,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn dirs(&self) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .map(|(d, _, _)| d.clone())
            .collect()
    }

    fn fitted_in(&self, dir: &str) -> Vec<String> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _, _)| d == dir)
            .map(|(_, c, _)| c.iter().map(|c| c.id.clone()).collect())
            .unwrap_or_default()
    }

    fn model_of(&self, dir: &str, id: &str) -> ModelText {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _, _)| d == dir)
            .and_then(|(_, c, _)| c.iter().find(|c| c.id == id).map(|c| c.model.clone()))
            .unwrap_or_else(|| panic!("{dir}/{id} was not fitted"))
    }

    /// The number of observations of the dataset a step was fitted on
    /// (`None` for the search's own).
    fn data_of(&self, dir: &str) -> Option<usize> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .find(|(d, _, _)| d == dir)
            .and_then(|(_, _, n)| *n)
    }

    fn respond(
        &self,
        dir: &str,
        candidates: &[Candidate],
        n_obs: Option<usize>,
    ) -> Result<RunReport, String> {
        let n_calls = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((dir.to_string(), candidates.to_vec(), n_obs));
            calls.len()
        };
        let mut results = Vec::new();
        for c in candidates {
            let ofv = self.ofv.get(&c.id).copied().unwrap_or(self.fallback);
            if self.erroring.contains(&c.id) {
                results.push(CandidateResult {
                    id: c.id.clone(),
                    hash: c.hash(),
                    parent: c.parent.clone(),
                    features: c.features.clone(),
                    fit: None,
                    ofv: None,
                    converged: None,
                    verdict: StrictnessVerdict {
                        passed: false,
                        failures: vec!["no fit: does not compile".into()],
                        skipped: vec![],
                    },
                    criterion: f64::NAN,
                    seconds: 0.0,
                    error: Some(CandidateError::model("does not compile")),
                    duplicate_of: None,
                    reused: false,
                });
                continue;
            }
            let mut fit = converged_fit(ofv);
            let spec = ErrorSpecText::read(&c.model).ok().flatten();
            let n_features = spec
                .as_ref()
                .map(|s| {
                    usize::from(s.form == ErrorForm::Combined || s.form == ErrorForm::Power)
                        + usize::from(s.iiv_on_ruv.is_some())
                        + usize::from(s.time_varying.is_some())
                })
                .unwrap_or(0);
            fit.n_parameters = BASE_N + n_features;
            if let Some((tn, t, en, om)) = self.screen_estimates.get(&c.id) {
                fit.theta_names = tn.clone();
                fit.theta = t.clone();
                fit.eta_names = en.clone();
                fit.omega =
                    nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(om.clone()));
            }
            let failing = self.failing.contains(&c.id);
            results.push(CandidateResult {
                id: c.id.clone(),
                hash: c.hash(),
                parent: c.parent.clone(),
                features: c.features.clone(),
                fit: Some(fit),
                ofv: Some(ofv),
                converged: Some(!failing),
                verdict: StrictnessVerdict {
                    passed: !failing,
                    failures: if failing {
                        vec!["stalled at the initial estimates (#751)".into()]
                    } else {
                        vec![]
                    },
                    skipped: vec![],
                },
                criterion: ofv,
                seconds: 0.0,
                error: None,
                duplicate_of: None,
                reused: false,
            });
        }
        Ok(RunReport {
            results,
            cancelled: self.cancel_after.is_some_and(|n| n_calls >= n),
            fitted: candidates.len(),
            reused: 0,
            deduped: 0,
            warnings: vec![],
        })
    }
}

impl StepFitter for Script {
    fn fit_step(&self, dir: &str, candidates: &[Candidate]) -> Result<RunReport, String> {
        self.respond(dir, candidates, None)
    }

    fn fit_step_on(
        &self,
        dir: &str,
        candidates: &[Candidate],
        data: &Population,
    ) -> Result<RunReport, String> {
        self.respond(dir, candidates, Some(data.n_obs()))
    }
}

fn run(script: &Script, space: Space, options: &RuvsearchOptions) -> RuvsearchResult {
    search(script, space, options, None).expect("search")
}

fn selected(result: &RuvsearchResult) -> Vec<(usize, String)> {
    result
        .rows
        .iter()
        .filter(|r| r.selected && !r.screened)
        .map(|r| (r.iteration, r.candidate.clone()))
        .collect()
}

fn config(text: &str) -> Result<SearchConfig, String> {
    SearchConfig::from_str(text, Path::new("."))
}

// ── options ────────────────────────────────────────────────────────────────

#[test]
fn options_take_pharmpys_defaults_and_bounds() {
    let o = RuvsearchOptions::default();
    assert_eq!(o.groups, 4);
    assert_eq!(o.p_value, 0.001);
    assert!(o.skip.is_empty());
    assert_eq!(o.max_iter, 3);
    assert!(!o.cwres_prescreen);
    assert!((o.cutoff() - 10.827_566).abs() < 1e-4);

    let cfg = config(
        "base = \"m.ferx\"\n[ruvsearch]\ngroups = 3\np_value = 0.05\nskip = [\"IIV_on_RUV\", \
         \"time_varying\"]\nmax_iter = 2\ncwres_prescreen = true\n",
    )
    .unwrap();
    let o = RuvsearchOptions::from_config(&cfg).unwrap();
    assert_eq!(o.groups, 3);
    assert_eq!(o.p_value, 0.05);
    assert_eq!(o.skip, vec![Family::IivOnRuv, Family::TimeVarying]);
    assert_eq!(o.max_iter, 2);
    assert!(o.cwres_prescreen);

    for (body, expect) in [
        ("groups = 1", "groups = 1"),
        ("p_value = 0.0", "p_value = 0"),
        ("p_value = 1.5", "p_value = 1.5"),
        ("max_iter = 0", "max_iter = 0"),
        ("max_iter = 4", "max_iter = 4"),
        ("skip = [\"prop\"]", "[ruvsearch]"),
        ("groupz = 2", "[ruvsearch]"),
    ] {
        let e = RuvsearchOptions::from_config(
            &config(&format!("base = \"m.ferx\"\n[ruvsearch]\n{body}\n")).unwrap(),
        )
        .unwrap_err();
        assert!(e.contains(expect), "`{body}`: {e}");
    }
    // A BIC rank, a cutoff or a space does not belong in a ruvsearch file.
    let e = RuvsearchOptions::from_config(
        &config("base = \"m.ferx\"\n[rank]\ntype = \"bic\"\n").unwrap(),
    )
    .unwrap_err();
    assert!(e.contains("likelihood-ratio"), "{e}");
    let e = RuvsearchOptions::from_config(
        &config("base = \"m.ferx\"\n[rank]\ncutoff = 3.84\n").unwrap(),
    )
    .unwrap_err();
    assert!(e.contains("p_value"), "{e}");
    let e = RuvsearchOptions::from_config(
        &config("base = \"m.ferx\"\n[space]\nmfl = \"ABSORPTION(FO)\"\n").unwrap(),
    )
    .unwrap_err();
    assert!(e.contains("no search space to declare"), "{e}");
    // …and the other tools refuse a file without one.
    let e =
        crate::covsearch::CovsearchOptions::from_config(&config("base = \"m.ferx\"\n").unwrap())
            .unwrap_err();
    assert!(e.contains("covsearch needs a [space]"), "{e}");
}

// ── the space ──────────────────────────────────────────────────────────────

#[test]
fn tad_cutoffs_are_pandas_quantiles_over_every_observation() {
    // TADs: subject 1 → 1, 4, 12, 1, 6, 23; subject 2 → 1.5, 4, 12, 1.5, 6, 23.
    // Sorted: 1, 1, 1.5, 1.5, 4, 4, 6, 6, 12, 12, 23, 23 (n = 12).
    let cuts = tad_cutoffs(&population(), 4);
    // Type-7: h = 11·q → 2.75 → 1.5; 5.5 → 5.0; 8.25 → 12.0.
    assert_eq!(cuts.len(), 3);
    assert!((cuts[0] - 1.5).abs() < 1e-12, "{cuts:?}");
    assert!((cuts[1] - 5.0).abs() < 1e-12, "{cuts:?}");
    assert!((cuts[2] - 12.0).abs() < 1e-12, "{cuts:?}");
    // A dataset with no doses has no TAD to cut.
    let mut pop = population();
    for s in &mut pop.subjects {
        s.doses.clear();
    }
    assert!(tad_cutoffs(&pop, 4).is_empty());
    let sp = Space::build(
        ModelText::parse(BASE).unwrap(),
        ErrorSpecText::read(&ModelText::parse(BASE).unwrap())
            .unwrap()
            .unwrap(),
        pop,
        true,
        &RuvsearchOptions::default(),
    );
    assert!(sp
        .notes
        .iter()
        .any(|n| n.contains("time_varying not tested")));
    // The additive σ init: (min DV / 2)² = 1.0.
    assert_eq!(sp.add_sigma_init, 1.0);
}

#[test]
fn candidates_follow_pharmpys_order_and_the_parent_can_take_them() {
    let options = RuvsearchOptions::default();
    let sp = space(&options);
    let parent = Node {
        id: "input".into(),
        model: sp.input_model.clone(),
        spec: sp.input_spec.clone(),
        fit: None,
        ofv: 100.0,
        n_parameters: BASE_N,
        features: vec![],
    };
    let none = BTreeSet::new();
    assert_eq!(
        candidates_for(&parent, &sp, &none),
        vec![
            RuvFeature::IivOnRuv,
            RuvFeature::Power,
            RuvFeature::Combined,
            RuvFeature::TimeVarying(1),
            RuvFeature::TimeVarying(2),
            RuvFeature::TimeVarying(3),
        ]
    );
    // Skipping power alone skips combined too (Pharmpy tests them together).
    let mut skip = BTreeSet::new();
    skip.insert(Family::Power);
    assert_eq!(
        candidates_for(&parent, &sp, &skip),
        vec![
            RuvFeature::IivOnRuv,
            RuvFeature::TimeVarying(1),
            RuvFeature::TimeVarying(2),
            RuvFeature::TimeVarying(3),
        ]
    );
    // A non-interaction method drops IIV_on_RUV.
    let sp_foce = space_from(BASE, &options, false);
    assert!(!candidates_for(&parent, &sp_foce, &none).contains(&RuvFeature::IivOnRuv));
    assert!(sp_foce
        .notes
        .iter()
        .any(|n| n.contains("IIV_on_RUV not tested")));
    // A power parent cannot take power or combined again.
    let mut spec = sp.input_spec.clone();
    spec.form = ErrorForm::Power;
    let power_parent = Node {
        spec,
        ..parent.clone()
    };
    assert_eq!(
        candidates_for(&power_parent, &sp, &none),
        vec![
            RuvFeature::IivOnRuv,
            RuvFeature::TimeVarying(1),
            RuvFeature::TimeVarying(2),
            RuvFeature::TimeVarying(3),
        ]
    );
    // Coinciding cutoffs collapse to one time-varying candidate.
    let mut tied = space(&options);
    tied.tad_cutoffs = vec![4.0, 4.0, 12.0];
    assert_eq!(
        candidates_for(&parent, &tied, &none)
            .into_iter()
            .filter(|f| matches!(f, RuvFeature::TimeVarying(_)))
            .collect::<Vec<_>>(),
        vec![RuvFeature::TimeVarying(1), RuvFeature::TimeVarying(3)]
    );
}

#[test]
fn derive_builds_each_feature_onto_the_parent_with_pharmpys_inits() {
    let options = RuvsearchOptions::default();
    let sp = space(&options);
    let parent = Node {
        id: "input".into(),
        model: sp.input_model.clone(),
        spec: sp.input_spec.clone(),
        fit: None,
        ofv: 100.0,
        n_parameters: BASE_N,
        features: vec![],
    };
    let lines = |f: RuvFeature| -> (Vec<String>, Vec<String>) {
        let c = derive("x", &parent, f, &sp, None).unwrap().unwrap();
        (
            c.model.block_lines("error_model"),
            c.model.block_lines("parameters"),
        )
    };
    let (e, p) = lines(RuvFeature::IivOnRuv);
    assert_eq!(
        e,
        vec!["DV ~ proportional(PROP_ERR)", "iiv_on_ruv = ETA_RUV"]
    );
    assert!(p.contains(&"omega ETA_RUV ~ 0.09".to_string()), "{p:?}");
    let (e, p) = lines(RuvFeature::Power);
    assert_eq!(e, vec!["DV ~ power(PROP_ERR, RUV_POW)"]);
    assert!(
        p.contains(&"theta RUV_POW(1.0, 0.01, 10.0)".to_string()),
        "{p:?}"
    );
    let (e, p) = lines(RuvFeature::Combined);
    assert_eq!(e, vec!["DV ~ combined(PROP_ERR, ADD_ERR)"]);
    assert!(p.contains(&"sigma ADD_ERR ~ 1.0".to_string()), "{p:?}");
    let (e, p) = lines(RuvFeature::TimeVarying(2));
    assert_eq!(
        e,
        vec!["DV ~ proportional(PROP_ERR * (if (TAD < 5.0) RUV_TV else 1.0))"]
    );
    assert!(
        p.contains(&"theta RUV_TV(1.0, 0.01, 10.0)".to_string()),
        "{p:?}"
    );
    // An init from the pre-screen overrides the default.
    let c = derive(
        "x",
        &parent,
        RuvFeature::Power,
        &sp,
        Some(Init { value: 1.37 }),
    )
    .unwrap()
    .unwrap();
    assert!(c
        .model
        .block_lines("parameters")
        .contains(&"theta RUV_POW(1.37, 0.01, 10.0)".to_string()));
    // A taken name gets a suffix.
    let taken = ModelText::parse(&BASE.replace(
        "  sigma PROP_ERR ~ 0.02 (sd)\n",
        "  sigma PROP_ERR ~ 0.02 (sd)\n  theta RUV_POW(2.0, 0.1, 5.0)\n",
    ))
    .unwrap();
    assert_eq!(fresh_name(&taken, "RUV_POW"), "RUV_POW_2");
    assert_eq!(fresh_name(&taken, "ETA_RUV"), "ETA_RUV");
    // Out-of-range time-varying index and a parent that already has the
    // feature give no candidate.
    assert!(derive("x", &parent, RuvFeature::TimeVarying(9), &sp, None)
        .unwrap()
        .is_none());
    let with_iiv = Node {
        spec: parent
            .spec
            .clone()
            .with_iiv_on_ruv(EtaDecl::new("ETA_RUV", 0.09)),
        ..parent.clone()
    };
    assert!(derive("x", &with_iiv, RuvFeature::IivOnRuv, &sp, None)
        .unwrap()
        .is_none());
}

// ── the search ─────────────────────────────────────────────────────────────

#[test]
fn an_iteration_selects_the_largest_significant_drop_and_retires_its_family() {
    let options = RuvsearchOptions::default();
    let script = Script::new(&[
        ("input", 100.0),
        ("IIV_on_RUV-1", 95.0),
        ("power-1", 85.0),
        ("combined-1", 88.0),
        ("time_varying1-1", 99.0),
        ("time_varying2-1", 99.0),
        ("time_varying3-1", 99.0),
        // From the power parent (85): nothing beats it by 10.83.
        ("IIV_on_RUV-2", 80.0),
        ("time_varying1-2", 84.0),
        ("time_varying2-2", 84.0),
        ("time_varying3-2", 84.0),
    ]);
    let result = run(&script, space(&options), &options);
    assert_eq!(selected(&result), vec![(1, "power-1".to_string())]);
    assert_eq!(result.final_id, "power-1");
    assert_eq!(result.features, vec![RuvFeature::Power]);
    assert_eq!(result.final_ofv, 85.0);
    assert_eq!(result.base_id, "input");
    // Iteration 2 tested neither power nor combined again.
    let it2 = script.fitted_in("iteration-2");
    assert_eq!(
        it2,
        vec![
            "IIV_on_RUV-2",
            "time_varying1-2",
            "time_varying2-2",
            "time_varying3-2"
        ]
    );
    assert_eq!(script.dirs(), vec!["input", "iteration-1", "iteration-2"]);
    // The row for IIV_on_RUV-2 says why it lost: not significant.
    let row = result
        .rows
        .iter()
        .find(|r| r.candidate == "IIV_on_RUV-2")
        .unwrap();
    let lrt = row.lrt.unwrap();
    assert_eq!(lrt.df, 1);
    assert!((lrt.dofv - 5.0).abs() < 1e-12);
    assert!(!lrt.significant);
    // The second iteration's candidates carry the power form.
    let m = script.model_of("iteration-2", "time_varying1-2");
    assert_eq!(
        m.block_lines("error_model"),
        vec!["DV ~ power(PROP_ERR * (if (TAD < 1.5) RUV_TV else 1.0), RUV_POW)"]
    );
    assert_eq!(result.n_iterations(), 2);
}

#[test]
fn a_gated_or_failed_candidate_cannot_win() {
    let options = RuvsearchOptions {
        skip: vec![Family::TimeVarying],
        ..RuvsearchOptions::default()
    };
    let mut script = Script::new(&[
        ("input", 100.0),
        ("IIV_on_RUV-1", 50.0),
        ("power-1", 60.0),
        ("combined-1", 85.0),
    ]);
    script.failing.push("IIV_on_RUV-1".into());
    script.erroring.push("power-1".into());
    let result = run(&script, space(&options), &options);
    assert_eq!(selected(&result), vec![(1, "combined-1".to_string())]);
    let iiv = result
        .rows
        .iter()
        .find(|r| r.candidate == "IIV_on_RUV-1")
        .unwrap();
    assert!(!iiv.passed && iiv.lrt.is_some_and(|t| t.significant));
    let power = result
        .rows
        .iter()
        .find(|r| r.candidate == "power-1")
        .unwrap();
    assert!(power.ofv.is_none() && power.note.is_some());
}

#[test]
fn an_input_that_is_not_proportional_gets_a_base_and_the_final_comparison_may_revert() {
    let additive = BASE
        .replace("sigma PROP_ERR ~ 0.02 (sd)", "sigma ADD_ERR ~ 0.5 (sd)")
        .replace("DV ~ proportional(PROP_ERR)", "DV ~ additive(ADD_ERR)");
    let options = RuvsearchOptions {
        skip: vec![Family::TimeVarying, Family::IivOnRuv],
        ..RuvsearchOptions::default()
    };
    // The proportional base is much worse than the additive input; power
    // beats the base but not the input.
    let script = Script::new(&[("input", 100.0), ("base", 130.0), ("power-1", 112.0)]);
    let result = run(&script, space_from(&additive, &options, true), &options);
    assert_eq!(result.base_id, "base");
    assert_eq!(result.base_ofv, 130.0);
    let base_model = script.model_of("base", "base");
    assert_eq!(
        base_model.block_lines("error_model"),
        vec!["DV ~ proportional(PROP_ERR)"]
    );
    assert!(base_model
        .block_lines("parameters")
        .contains(&"sigma PROP_ERR ~ 0.09".to_string()));
    assert!(!base_model.render().contains("ADD_ERR"));
    assert_eq!(selected(&result), vec![(1, "power-1".to_string())]);
    // 100 − 112 < 10.83: the input is returned.
    assert_eq!(result.final_id, "input");
    assert_eq!(result.final_ofv, 100.0);
    assert!(result.features.is_empty());
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("did not beat the input model")));
    assert_eq!(result.rows.iter().filter(|r| r.iteration == 0).count(), 2);
    // The reversion is also an event, so a CLI can say so as it happens.
    let events = Mutex::new(Vec::new());
    let progress = |e: RuvsearchEvent| events.lock().unwrap().push(e);
    let script = Script::new(&[("input", 100.0), ("base", 130.0), ("power-1", 112.0)]);
    search(
        &script,
        space_from(&additive, &options, true),
        &options,
        Some(&progress),
    )
    .unwrap();
    let events = events.into_inner().unwrap();
    assert!(
        events.contains(&RuvsearchEvent::Reverted {
            to: "input".into(),
            ofv: 100.0
        }),
        "{events:?}"
    );
    assert!(events.contains(&RuvsearchEvent::BaseFinished { ofv: 130.0 }));
}

#[test]
fn a_combined_input_keeps_its_proportional_sigma_in_the_base() {
    let combined = BASE
        .replace(
            "  sigma PROP_ERR ~ 0.02 (sd)\n",
            "  sigma PROP_ERR ~ 0.02 (sd)\n  sigma ADD_ERR ~ 0.5 (sd)\n",
        )
        .replace(
            "DV ~ proportional(PROP_ERR)",
            "DV ~ combined(PROP_ERR, ADD_ERR)",
        );
    let options = RuvsearchOptions {
        max_iter: 1,
        skip: vec![Family::TimeVarying, Family::IivOnRuv],
        ..RuvsearchOptions::default()
    };
    // The base beats the input, and combined-1 rediscovers the input's form
    // with a big enough drop to be kept.
    let script = Script::new(&[("input", 100.0), ("base", 80.0), ("combined-1", 60.0)]);
    let result = run(&script, space_from(&combined, &options, true), &options);
    let base_model = script.model_of("base", "base");
    assert!(base_model
        .block_lines("parameters")
        .contains(&"sigma PROP_ERR ~ 0.02 (sd)".to_string()));
    assert_eq!(result.final_id, "combined-1");
    assert_eq!(result.features, vec![RuvFeature::Combined]);
}

#[test]
fn nothing_significant_ends_the_search_on_the_parent() {
    let options = RuvsearchOptions::default();
    let script = Script::new(&[("input", 100.0)]);
    // Every candidate takes the fallback 1000: none is significant.
    let result = run(&script, space(&options), &options);
    assert!(selected(&result).is_empty());
    assert_eq!(result.final_id, "input");
    assert_eq!(result.n_iterations(), 1);
    assert_eq!(script.dirs(), vec!["input", "iteration-1"]);
}

#[test]
fn cancellation_returns_partial_rows_and_the_parent() {
    let options = RuvsearchOptions::default();
    let mut script = Script::new(&[("input", 100.0), ("power-1", 50.0)]);
    script.cancel_after = Some(2);
    let result = run(&script, space(&options), &options);
    assert!(result.cancelled);
    assert_eq!(result.final_id, "input");
    assert!(selected(&result).is_empty());
    assert_eq!(script.dirs().len(), 2);
}

#[test]
fn a_base_that_fails_the_gate_is_an_error() {
    let options = RuvsearchOptions::default();
    let mut script = Script::new(&[("input", 100.0)]);
    script.failing.push("input".into());
    let e = search(&script, space(&options), &options, None).unwrap_err();
    assert!(e.contains("fails the strictness gate"), "{e}");
}

// ── the pre-screen ─────────────────────────────────────────────────────────

fn screen_fit(
    theta: &[(&str, f64)],
    omega: &[(&str, f64)],
) -> (Vec<String>, Vec<f64>, Vec<String>, Vec<f64>) {
    (
        theta.iter().map(|(n, _)| n.to_string()).collect(),
        theta.iter().map(|(_, v)| *v).collect(),
        omega.iter().map(|(n, _)| n.to_string()).collect(),
        omega.iter().map(|(_, v)| *v).collect(),
    )
}

/// The parent's fit has to carry per-subject CWRES for the screen; the
/// fixture fit is the warfarin evaluation, whose subjects do. The population
/// the screen is built from must match it, so this uses the warfarin data.
fn warfarin_space(options: &RuvsearchOptions) -> Space {
    let pop = ferx_core::read_nonmem_csv(Path::new(crate::search::test_support::DATA), None, None)
        .unwrap();
    let text = ModelText::parse(BASE).unwrap();
    let spec = ErrorSpecText::read(&text).unwrap().unwrap();
    Space::build(text, spec, pop, true, options)
}

#[test]
fn the_prescreen_refits_only_the_largest_cwres_drop_seeded_from_the_screen() {
    let options = RuvsearchOptions {
        cwres_prescreen: true,
        skip: vec![Family::TimeVarying],
        max_iter: 2,
        ..RuvsearchOptions::default()
    };
    let mut script = Script::new(&[
        ("input", 100.0),
        ("cwres-base-1", 50.0),
        ("cwres-IIV_on_RUV-1", 45.0), // drop 5: below the cutoff
        ("cwres-power-1", 30.0),      // drop 20: the pick
        ("cwres-combined-1", 35.0),   // drop 15
        ("power-1", 85.0),            // the refit: accepted (drop 15)
        ("cwres-base-2", 50.0),
        ("cwres-IIV_on_RUV-2", 45.0),
    ]);
    script.screen_estimates.insert(
        "cwres-power-1".into(),
        screen_fit(&[("TVB", 0.0), ("RUV_POW", 0.4)], &[("ETA_B", 0.01)]),
    );
    let result = run(&script, warfarin_space(&options), &options);
    assert_eq!(
        script.dirs(),
        vec!["input", "screen-1", "iteration-1", "screen-2"]
    );
    // The screen fitted the base and the three candidates, on the CWRES data.
    assert_eq!(
        script.fitted_in("screen-1"),
        vec![
            "cwres-base-1",
            "cwres-IIV_on_RUV-1",
            "cwres-power-1",
            "cwres-combined-1"
        ]
    );
    let n_cwres = script
        .data_of("screen-1")
        .expect("screened on another dataset");
    assert!(n_cwres > 0);
    // Only the pick was refitted, seeded with `power = θ̂ + 1`.
    assert_eq!(script.fitted_in("iteration-1"), vec!["power-1"]);
    let refit = script.model_of("iteration-1", "power-1");
    assert!(
        refit
            .block_lines("parameters")
            .contains(&"theta RUV_POW(1.4, 0.01, 10.0)".to_string()),
        "{:?}",
        refit.block_lines("parameters")
    );
    assert_eq!(selected(&result), vec![(1, "power-1".to_string())]);
    // Two CWRES bases, three screened features in iteration 1, one in 2.
    let screened: Vec<_> = result.rows.iter().filter(|r| r.screened).collect();
    assert_eq!(screened.len(), 6);
    assert!(screened
        .iter()
        .any(|r| r.candidate == "cwres-base-1" && r.feature.is_none() && r.ofv == Some(50.0)));
    let pick = screened
        .iter()
        .find(|r| r.candidate == "cwres-power-1")
        .unwrap();
    assert!(pick.selected);
    assert_eq!(pick.cwres_dofv, Some(20.0));
    assert!(pick.lrt.is_none(), "a screened row carries no LRT");
    // Iteration 2 screened IIV_on_RUV only (power and combined retired) and
    // found nothing.
    assert_eq!(
        script.fitted_in("screen-2"),
        vec!["cwres-base-2", "cwres-IIV_on_RUV-2"]
    );
    assert_eq!(result.final_id, "power-1");
    // The screening models are on the result, readable and compilable.
    let m = &result.models["cwres-power-1"];
    ferx_core::parser::model_parser::parse_full_model(&m.render())
        .unwrap_or_else(|e| panic!("{e}\n{}", m.render()));
}

#[test]
fn a_screened_pick_whose_refit_is_not_accepted_retires_its_family_and_goes_on() {
    let options = RuvsearchOptions {
        cwres_prescreen: true,
        skip: vec![Family::TimeVarying],
        ..RuvsearchOptions::default()
    };
    let script = Script::new(&[
        ("input", 100.0),
        ("cwres-base-1", 50.0),
        ("cwres-power-1", 30.0),
        ("power-1", 98.0), // the refit: drop 2, not accepted
        ("cwres-base-2", 50.0),
        ("cwres-IIV_on_RUV-2", 20.0),
        ("IIV_on_RUV-2", 80.0), // accepted
        ("cwres-base-3", 50.0),
    ]);
    let result = run(&script, warfarin_space(&options), &options);
    assert_eq!(selected(&result), vec![(2, "IIV_on_RUV-2".to_string())]);
    assert_eq!(result.final_id, "IIV_on_RUV-2");
    assert_eq!(result.features, vec![RuvFeature::IivOnRuv]);
    assert!(result
        .notes
        .iter()
        .any(|n| n.contains("picked power but its refit was not accepted")));
    // The third iteration had nothing left to screen.
    assert!(result.notes.iter().any(|n| n.contains("no candidate left")));
}

#[test]
fn screening_population_is_one_record_per_finite_cwres_with_ipred_and_doses() {
    let fit = converged_fit(100.0);
    let pop = ferx_core::read_nonmem_csv(Path::new(crate::search::test_support::DATA), None, None)
        .unwrap();
    let screen = cwres::screening_population(&fit, &pop).unwrap();
    assert_eq!(screen.covariate_names, vec![cwres::IPRED_COLUMN]);
    let n_finite: usize = fit
        .subjects
        .iter()
        .map(|s| s.cwres.iter().filter(|c| c.is_finite()).count())
        .sum();
    assert_eq!(screen.n_obs(), n_finite);
    for (s, (sr, orig)) in screen
        .subjects
        .iter()
        .zip(fit.subjects.iter().zip(&pop.subjects))
    {
        assert_eq!(s.id, orig.id);
        assert_eq!(s.doses.len(), orig.doses.len(), "doses kept for TAD");
        assert_eq!(s.observations.len(), s.obs_covariates.len());
        for (j, &dv) in s.observations.iter().enumerate() {
            assert!(sr.cwres.contains(&dv));
            let ipred = s.obs_covariates[j][cwres::IPRED_COLUMN];
            assert!(ipred > 0.0 && sr.ipred.iter().any(|&f| f.abs() == ipred));
        }
        assert!(s.time_after_dose(0).is_finite());
    }
    // A mismatched fit is refused, not silently misaligned.
    let mut short = pop.clone();
    short.subjects.pop();
    assert!(cwres::screening_population(&fit, &short).is_err());
}

#[test]
fn screen_inits_map_back_as_pharmpy_does() {
    let mut fit = converged_fit(1.0);
    fit.theta_names = vec!["TVB".into(), "RUV_POW".into(), "RUV_TV".into()];
    fit.theta = vec![0.0, -1.5, 0.7];
    fit.eta_names = vec!["ETA_B".into(), "ETA_RUV".into()];
    fit.omega = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![0.01, 0.2]));
    // power = θ + 1 = −0.5 → floored to 0.02.
    assert_eq!(
        cwres::init_from_screen(RuvFeature::Power, &fit)
            .unwrap()
            .value,
        0.02
    );
    fit.theta[1] = 0.3;
    assert!(
        (cwres::init_from_screen(RuvFeature::Power, &fit)
            .unwrap()
            .value
            - 1.3)
            .abs()
            < 1e-12
    );
    assert_eq!(
        cwres::init_from_screen(RuvFeature::IivOnRuv, &fit)
            .unwrap()
            .value,
        0.2
    );
    assert_eq!(
        cwres::init_from_screen(RuvFeature::TimeVarying(1), &fit)
            .unwrap()
            .value,
        0.7
    );
    assert!(cwres::init_from_screen(RuvFeature::Combined, &fit).is_none());
}

#[test]
fn every_screening_model_compiles() {
    let fit = converged_fit(100.0);
    let pop = ferx_core::read_nonmem_csv(Path::new(crate::search::test_support::DATA), None, None)
        .unwrap();
    let features = [
        RuvFeature::IivOnRuv,
        RuvFeature::Power,
        RuvFeature::Combined,
        RuvFeature::TimeVarying(1),
    ];
    let screen = cwres::Screen::build(&fit, &pop, &features, &[2.0, 6.0, 24.0], 1).unwrap();
    assert_eq!(screen.candidates.len(), 5);
    for c in &screen.candidates {
        ferx_core::parser::model_parser::parse_full_model(&c.model.render())
            .unwrap_or_else(|e| panic!("{}: {e}\n{}", c.id, c.model.render()));
    }
}

// ── the report ─────────────────────────────────────────────────────────────

#[test]
fn the_report_writes_the_step_table_the_models_and_the_final_model() {
    let options = RuvsearchOptions {
        skip: vec![Family::TimeVarying, Family::IivOnRuv],
        ..RuvsearchOptions::default()
    };
    let script = Script::new(&[("input", 100.0), ("power-1", 85.0), ("combined-1", 90.0)]);
    let result = run(&script, space(&options), &options);
    let dir = tempfile::tempdir().unwrap();
    write_report(dir.path(), &result).unwrap();
    let steps = std::fs::read_to_string(steps_path(dir.path())).unwrap();
    let header = steps.lines().next().unwrap();
    assert_eq!(header, STEP_COLUMNS.join(","));
    assert_eq!(steps.lines().count(), 1 + result.rows.len());
    assert!(steps.contains("power-1,power,false,100.000000,85.000000,15.000000,1,"));
    assert!(models_dir(dir.path()).join("power-1.ferx").exists());
    assert!(models_dir(dir.path()).join("input.ferx").exists());
    let final_text = std::fs::read_to_string(final_model_path(dir.path())).unwrap();
    assert!(final_text.contains("DV ~ power(PROP_ERR, RUV_POW)"));
    let summary = render_summary(&result);
    assert!(summary.contains("Input model: OFV 100.000"), "{summary}");
    assert!(summary.contains("SELECTED"), "{summary}");
    assert!(summary.contains("features added: power"), "{summary}");
}
