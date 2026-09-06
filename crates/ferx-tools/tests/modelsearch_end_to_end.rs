//! Tier-2 end-to-end checks for modelsearch (#1181), on a real model and
//! dataset but never to convergence: every fit is an evaluation
//! (`maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/modelsearch/mod_tests.rs` script the fitter, so
//! they test the enumeration and never compile a candidate. This file is the
//! other half: from a `.ferxsearch` file through the structural edits, the
//! runner, `fit()` and the files on disk. Three of the issue's validation
//! bullets live here:
//!
//! * the **degenerate oracle** — a single-point space returns the base fit
//!   *bit for bit*;
//! * **every generated candidate compiles and evaluates** — the role tables
//!   in `structure.rs` agree with the parser's, on every template the search
//!   can reach from a one-compartment oral base;
//! * a generated candidate **is the hand-written model** — the
//!   `tests/covariate_model_equivalence.rs` pattern: bit-identical
//!   predictions and a bit-identical evaluation against the model a
//!   pharmacometrician would have typed.

use std::path::{Path, PathBuf};

use ferx_core::edit::{ModelEdit, ModelText};
use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::{fit, predict};
use ferx_tools::modelsearch::{
    run_modelsearch, Absorption, ModelsearchRun, Structure, TransitCount,
};
use ferx_tools::search::SearchConfig;

const DATA: &str = "../../data/warfarin.csv";

/// The warfarin one-compartment oral model, evaluating rather than fitting.
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
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
";

/// Write the base model and a `.ferxsearch` beside it in a temp dir.
fn write_config(dir: &Path, mfl: &str, extra: &str) -> PathBuf {
    std::fs::write(dir.join("base.ferx"), BASE).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATA);
    let config = format!(
        "base = \"base.ferx\"\ndata = \"{}\"\n\n[space]\nmfl = \"{mfl}\"\n\n\
         [strictness]\nrequire_converged = false\nreject_init_stall = false\n\
         reject_on_boundary = false\n\n[run]\nretries = 0\nthreads = 2\n{extra}",
        data.display()
    );
    let path = dir.join("search.ferxsearch");
    std::fs::write(&path, config).unwrap();
    path
}

fn run(dir: &Path, path: &Path) -> (SearchConfig, ferx_tools::modelsearch::ModelsearchResult) {
    let config = SearchConfig::load(path).unwrap();
    let base = config.load_base().unwrap();
    let result = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
            dir: Some(dir.join("run")),
            ..ModelsearchRun::default()
        },
    )
    .expect("search");
    (config, result)
}

#[test]
fn a_single_point_space_returns_the_base_fit_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "ABSORPTION(FO); PERIPHERALS(0); TRANSITS(0, NODEPOT); LAGTIME(OFF)",
        "",
    );
    let config = SearchConfig::load(&path).unwrap();
    let base = config.load_base().unwrap();
    let direct = fit(
        &base.prepared.parsed.model,
        &base.prepared.population,
        &base.prepared.init_params,
        &{
            let mut o = base.prepared.parsed.fit_options.clone().quiet();
            o.threads = Some(2);
            o
        },
    )
    .expect("direct evaluation");

    let result = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
            dir: Some(dir.path().join("run")),
            ..ModelsearchRun::default()
        },
    )
    .expect("search");

    assert_eq!(result.rows.len(), 1, "{:?}", result.rows);
    assert_eq!(result.final_id, "base");
    assert_eq!(result.base_id, "base");
    let final_fit = result.final_fit.as_ref().expect("the base fit");
    assert_eq!(
        final_fit.ofv.to_bits(),
        direct.ofv.to_bits(),
        "search OFV {} vs direct {}",
        final_fit.ofv,
        direct.ofv
    );
    assert_eq!(final_fit.n_parameters, direct.n_parameters);
    assert_eq!(result.row("base").unwrap().ofv, Some(direct.ofv));
    assert_eq!(result.final_model.render(), base.text.render());
    assert_eq!(result.input_model.render(), base.text.render());
    assert!(result.notes.is_empty(), "{:?}", result.notes);

    let models = std::fs::read_to_string(dir.path().join("run/models.csv")).unwrap();
    assert_eq!(models.lines().count(), 2);
    assert!(dir.path().join("run/base/candidates.csv").exists());
    assert!(dir.path().join("run/models/base.ferx").exists());
    // `final.ferx` is the base with its own (unmoved) estimates written in:
    // it evaluates to the same objective.
    let final_text = std::fs::read_to_string(dir.path().join("run/final.ferx")).unwrap();
    let parsed = parse_full_model(&final_text).expect("final.ferx parses");
    let again = fit(
        &parsed.model,
        &base.prepared.population,
        &parsed.model.default_params,
        &parsed.fit_options.clone().quiet(),
    )
    .expect("re-evaluation of final.ferx");
    assert_eq!(again.ofv.to_bits(), direct.ofv.to_bits());
}

#[test]
fn every_candidate_the_search_can_reach_compiles_and_evaluates() {
    // Every template reachable from a one-compartment oral base: the
    // second and third compartments, a fixed and an estimated transit
    // chain, a lag time, and the pairs Pharmpy allows.
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "ABSORPTION(FO); PERIPHERALS(0..2); TRANSITS([0,3], NODEPOT); TRANSITS(N); \
         LAGTIME([OFF,ON])",
        "[modelsearch]\nalgorithm = \"exhaustive\"\n",
    );
    let (_, result) = run(dir.path(), &path);

    let candidates: Vec<_> = result.layer_rows(1).collect();
    let structures: Vec<String> = candidates
        .iter()
        .map(|r| r.structure.feature_vector().render())
        .collect();
    // Pharmpy's `product` order over (LAGTIME, PERIPHERALS, TRANSITS).
    assert_eq!(
        structures,
        vec![
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;TRANSITS=3",
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;TRANSITS=N",
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=1;TRANSITS=0",
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=1;TRANSITS=3",
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=1;TRANSITS=N",
            "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=2;TRANSITS=0",
            "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=0;TRANSITS=0",
            "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=1;TRANSITS=0",
            "ABSORPTION=FO;LAGTIME=ON;PERIPHERALS=2;TRANSITS=0",
        ],
        "{:?}",
        result.notes
    );
    // Two peripherals with a transit chain has no template, and says so.
    let gaps: Vec<&String> = result
        .notes
        .iter()
        .filter(|n| n.starts_with("not generated"))
        .collect();
    assert_eq!(gaps.len(), 2, "{:?}", result.notes);
    assert!(gaps.iter().all(|n| n.contains("three_cpt_transit")));

    let base_ofv = result.row("base").unwrap().ofv.unwrap();
    for r in &candidates {
        assert!(r.error.is_none(), "{}: {:?}", r.id, r.error);
        let ofv = r.ofv.unwrap_or_else(|| panic!("{}: no OFV", r.id));
        assert!(ofv.is_finite(), "{}: {ofv}", r.id);
        assert!(r.passed, "{}: {:?}", r.id, r.failures);
        assert_eq!(r.converged, Some(false));
        // Every template reached the engine: a new compartment, chain or
        // lag at its default init moves the objective.
        assert_ne!(ofv.to_bits(), base_ofv.to_bits(), "{}", r.id);
        assert!(r.criterion.is_finite());
        assert!(r.rank.is_some());
        assert!(r.seconds > 0.0);
    }
    // Parameter counts follow the templates: a peripheral adds two θ, a
    // lag time one θ and one ω (absorption_delay), a fixed chain swaps KA
    // and its η for MTT and an η on MTT, an estimated chain adds NTR too.
    let n = |i: usize| candidates[i].n_parameters.unwrap();
    let base_n = result.row("base").unwrap().n_parameters.unwrap();
    assert_eq!(base_n, 7);
    assert_eq!(
        n(0),
        base_n,
        "TVKA + ETA_KA out, TVMTT + ETA_MTT in, NTR fixed"
    );
    assert_eq!(n(1), base_n + 1, "…and NTR estimated");
    assert_eq!(n(2), base_n + 2);
    assert_eq!(n(5), base_n + 4);
    assert_eq!(n(6), base_n + 2);
    // The fixed chain's count is a `FIX`ed θ with a name, not a literal.
    let run3 = &result.models[&candidates[0].id];
    let params = run3.block_lines("parameters");
    assert!(
        params.contains(&"theta TVNTR(3.0, 0.0, 64.0) FIX".to_string()),
        "{params:?}"
    );
    assert_eq!(
        run3.block_lines("structural_model"),
        vec!["pk one_cpt_transit(cl=CL, v=V, n=NTR, mtt=MTT)"]
    );
    assert!(!params.iter().any(|l| l.contains("TVKA")), "{params:?}");

    let models = std::fs::read_to_string(dir.path().join("run/models.csv")).unwrap();
    assert_eq!(models.lines().count(), 1 + 1 + 9);
    assert!(dir.path().join("run/candidates/candidates.csv").exists());
}

/// A generated candidate must be the model a pharmacometrician would have
/// typed: same θ and η vectors, bit-identical predictions, bit-identical
/// evaluation. Both sides carry the base's seeded estimates, so the
/// comparison is of the structural spelling alone.
#[test]
fn a_generated_candidate_is_the_hand_written_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "PERIPHERALS(0..1); LAGTIME([OFF,ON])",
        "[modelsearch]\nalgorithm = \"exhaustive\"\n",
    );
    let (config, result) = run(dir.path(), &path);
    let base = config.load_base().unwrap();
    let base_fit = result.row("base").unwrap();
    assert_eq!(base_fit.layer, 0);

    let two_cpt_lag = result
        .rows
        .iter()
        .find(|r| {
            r.structure
                == Structure {
                    absorption: Absorption::Fo,
                    peripherals: 1,
                    transits: None,
                    lagtime: true,
                }
        })
        .expect("the two-compartment lagged candidate");
    let generated = result.models[&two_cpt_lag.id].render();

    // The candidate was seeded from the base's evaluation, and its new
    // parameters scale from those seeded estimates (`Q = CL`, `V2 =
    // 0.05·V`) — so the twin is written from the same fit.
    let base_fit_result = {
        let mut o = base.prepared.parsed.fit_options.clone().quiet();
        o.threads = Some(2);
        fit(
            &base.prepared.parsed.model,
            &base.prepared.population,
            &base.prepared.init_params,
            &o,
        )
        .unwrap()
    };
    let seeded = |name: &str| {
        let i = base_fit_result
            .theta_names
            .iter()
            .position(|n| n == name)
            .unwrap();
        base_fit_result.theta[i]
    };
    let hand_written = format!(
        "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)
  theta TVQ({q}, 0.0, 1000000.0)
  theta TVV2({v2}, 0.0, 1000000.0)
  theta TVALAG(0.25, 0.0, 1000000.0)
  omega ETA_ALAG ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V = TVV * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
  Q = TVQ
  V2 = TVV2
  ALAG = TVALAG * exp(ETA_ALAG)

[structural_model]
  pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA, lagtime=ALAG)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  maxiter    = 0
  covariance = false
  checkpoint = false
",
        q = seeded("TVCL"),
        v2 = 0.05 * seeded("TVV"),
    );
    let mut twin = ModelText::parse(&hand_written).unwrap();
    twin.apply(ModelEdit::SeedInits(&base_fit_result)).unwrap();
    let hand_written = twin.render();

    let gen = parse_full_model(&generated)
        .unwrap_or_else(|e| panic!("the generated model must parse: {e}\n---\n{generated}"));
    let hand = parse_full_model(&hand_written).expect("the hand-written model must parse");
    assert_eq!(gen.model.theta_names, hand.model.theta_names);
    assert_eq!(gen.model.eta_names, hand.model.eta_names);
    assert_eq!(
        gen.model.default_params.theta,
        hand.model.default_params.theta
    );

    let pop = &base.prepared.population;
    let gen_pred = predict(&gen.model, pop, &gen.model.default_params);
    let hand_pred = predict(&hand.model, pop, &hand.model.default_params);
    assert_eq!(gen_pred.len(), hand_pred.len());
    for (g, h) in gen_pred.iter().zip(&hand_pred) {
        assert_eq!(g.id, h.id);
        assert_eq!(g.time, h.time);
        assert_eq!(
            g.pred.to_bits(),
            h.pred.to_bits(),
            "PRED differs for subject {}",
            g.id
        );
    }
    let opts = {
        let mut o = gen.fit_options.clone().quiet();
        o.threads = Some(2);
        o
    };
    let gen_fit = fit(&gen.model, pop, &gen.model.default_params, &opts).unwrap();
    let hand_fit = fit(&hand.model, pop, &hand.model.default_params, &opts).unwrap();
    assert_eq!(
        gen_fit.ofv.to_bits(),
        hand_fit.ofv.to_bits(),
        "generated {} vs hand-written {}",
        gen_fit.ofv,
        hand_fit.ofv
    );
    assert_eq!(gen_fit.n_parameters, hand_fit.n_parameters);
    // …and the search's own row reports that same evaluation.
    assert_eq!(two_cpt_lag.ofv.unwrap().to_bits(), gen_fit.ofv.to_bits());
}

#[test]
fn a_stepwise_search_seeds_each_layer_from_its_parent_and_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "PERIPHERALS(0..1); LAGTIME([OFF,ON])",
        "[modelsearch]\nalgorithm = \"reduced_stepwise\"\n",
    );
    let (config, result) = run(dir.path(), &path);
    assert_eq!(result.n_layers(), 2);
    assert_eq!(result.rows.len(), 1 + 2 + 2);
    assert!(dir.path().join("run/layer-1/candidates.csv").exists());
    assert!(dir.path().join("run/layer-2/candidates.csv").exists());
    // Layer 2's candidates are seeded from their layer-1 parents: the
    // parent's own new parameter is carried, not re-declared.
    for r in result.layer_rows(2) {
        let text = &result.models[&r.id];
        let parent = &result.models[r.parent.as_deref().unwrap()];
        let parent_thetas: Vec<String> = parent
            .block_lines("parameters")
            .iter()
            .filter(|l| l.starts_with("theta "))
            .cloned()
            .collect();
        let thetas = text.block_lines("parameters");
        for t in &parent_thetas {
            assert!(
                thetas.contains(t),
                "{}: parent θ `{t}` not carried: {thetas:?}",
                r.id
            );
        }
        assert_eq!(
            text.block_lines("structural_model"),
            vec!["pk two_cpt_oral(cl=CL, v1=V, q=Q, v2=V2, ka=KA, lagtime=ALAG)"]
        );
    }
    // A second run over the same directory refits nothing.
    let mut config = config;
    config.run.resume = true;
    let base = config.load_base().unwrap();
    let again = run_modelsearch(
        &config,
        &base,
        ModelsearchRun {
            dir: Some(dir.path().join("run")),
            ..ModelsearchRun::default()
        },
    )
    .expect("resumed search");
    assert_eq!(again.rows.len(), result.rows.len());
    assert_eq!(again.final_id, result.final_id);
    for (a, b) in again.rows.iter().zip(&result.rows) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.ofv.map(f64::to_bits), b.ofv.map(f64::to_bits), "{}", a.id);
        assert!(a.reused, "{}: reused, not refitted", a.id);
        assert!(!b.reused, "{}", b.id);
    }
}

#[test]
fn the_transit_count_of_a_fixed_chain_stays_fixed_through_a_resume() {
    // The estimates file of a `TRANSITS(3)` candidate names NTR with SE 0,
    // and `SeedInits` on the final model leaves the `FIX` line alone.
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "TRANSITS([0,3], NODEPOT)",
        "[modelsearch]\nalgorithm = \"exhaustive\"\n[rank]\ntype = \"ofv\"\n",
    );
    let (_, result) = run(dir.path(), &path);
    let chain = result
        .rows
        .iter()
        .find(|r| r.structure.transits == Some(TransitCount::Count(3)))
        .expect("the chain candidate");
    let fit = result.models[&chain.id].clone();
    let mut seeded = fit.clone();
    // Re-seeding from any fit keeps the FIX declaration byte-for-byte.
    if let Some(f) = &result.final_fit {
        seeded.apply(ModelEdit::SeedInits(f)).unwrap();
    }
    let line = |t: &ModelText| {
        t.block_lines("parameters")
            .into_iter()
            .find(|l| l.contains("TVNTR"))
            .unwrap()
    };
    assert_eq!(line(&fit), "theta TVNTR(3.0, 0.0, 64.0) FIX");
    assert_eq!(line(&seeded), line(&fit));
}
