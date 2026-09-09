//! Tier-2 end-to-end checks for the global search (#1185), on a real model
//! and dataset but never to convergence: every fit is an evaluation
//! (`maxiter = 0`, ferx's `MAXEVAL=0`).
//!
//! The unit tests in `src/globalsearch/mod_tests.rs` script the fitter, so
//! they test the grid, the decoding and the fitness and never compile a
//! candidate. This file is the other half: from a `.ferxsearch` file
//! through the structural and covariate edits, the runner, `fit()` and the
//! files on disk. Three of the issue's acceptance bullets have their
//! real-fit half here:
//!
//! * **exhaustive and the GA agree** on a grid small enough to enumerate,
//!   through real evaluations — the same winner, the same fitness;
//! * **every grid point compiles and evaluates**, structural and covariate
//!   axes combined, and a dead gene (a covariate on a parameter the
//!   structure removed) is a duplicate the runner fits once;
//! * **one runner, one cache** — a run resumed from its own directory
//!   refits nothing, because the GA is seeded and proposes the same
//!   genomes.

use std::path::{Path, PathBuf};

use ferx_tools::globalsearch::{run_globalsearch, GlobalsearchResult, GlobalsearchRun};
use ferx_tools::search::SearchConfig;

const DATA: &str = "../../data/two_cpt_oral_cov.csv";

/// A one-compartment oral model with a weight covariate declared, evaluating
/// rather than fitting.
const BASE: &str = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02 (sd)

[covariates]
  WT continuous

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

fn run(dir: &Path, path: &Path, resume: bool) -> GlobalsearchResult {
    let mut config = SearchConfig::load(path).unwrap();
    config.run.resume = resume;
    let base = config.load_base().unwrap();
    run_globalsearch(
        &config,
        &base,
        GlobalsearchRun {
            dir: Some(dir.join("run")),
            ..GlobalsearchRun::default()
        },
    )
    .expect("search")
}

const GRID: &str = "ABSORPTION([INST,FO]); PERIPHERALS(0..1); LAGTIME([OFF,ON]); \
                    COVARIATE?(CL, WT, pow); COVARIATE?(KA, WT, exp)";

#[test]
fn every_grid_point_compiles_and_evaluates_and_a_dead_gene_is_fitted_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        GRID,
        "[globalsearch]\nalgorithm = \"exhaustive\"\n",
    );
    let result = run(dir.path(), &path, false);
    // 2 · 2 · 2 · 2 · 2 = 32 points; the input on top.
    assert_eq!(result.space_size, 32);
    assert_eq!(result.rows.len(), 33, "{:?}", result.notes);

    let input_ofv = result.row("input").unwrap().ofv.unwrap();
    let mut fitted = 0usize;
    for r in &result.rows[1..] {
        let s = r.structure.unwrap();
        let bolus_lag = s.absorption == ferx_tools::modelsearch::Absorption::Inst && s.lagtime;
        if bolus_lag {
            // The one pair no template expresses: a row with the crash
            // value, never submitted.
            assert!(r.error.is_some(), "{}", r.id);
            assert_eq!(r.fitness, result.options.penalties.crash);
            assert!(r.rank.is_none());
            continue;
        }
        assert!(r.error.is_none(), "{}: {:?}", r.id, r.error);
        let ofv = r.ofv.unwrap_or_else(|| panic!("{}: no OFV", r.id));
        assert!(ofv.is_finite(), "{}: {ofv}", r.id);
        assert!(r.passed, "{}: {:?}", r.id, r.failures);
        assert!(r.fitness.is_finite());
        assert!(r.rank.is_some(), "{}", r.id);
        // The KA gene is dead on a bolus: no relation written, the charge
        // applied, and the runner fitted the rendered model once — the
        // second genome is a duplicate of the first.
        let ka_gene = r.description.contains("KA-WT=exponential");
        if s.absorption == ferx_tools::modelsearch::Absorption::Inst && ka_gene {
            assert_eq!(r.non_influential, 1, "{}", r.id);
            assert!(r.duplicate_of.is_some(), "{}: {}", r.id, r.description);
            assert_eq!(r.seconds, 0.0);
            let twin = result.row(r.duplicate_of.as_ref().unwrap()).unwrap();
            assert_eq!(r.ofv, twin.ofv);
            assert_eq!(
                r.fitness,
                twin.fitness + result.options.penalties.non_influential
            );
            assert!(!result.models[&r.id].render().contains("KA ~ WT"));
        } else {
            assert_eq!(r.non_influential, 0, "{}", r.id);
            assert!(r.duplicate_of.is_none(), "{}", r.id);
            assert!(r.seconds > 0.0);
            fitted += 1;
            // A live gene is a real relation the candidate carries.
            let text = result.models[&r.id].render();
            assert_eq!(
                text.contains("CL ~ WT power"),
                r.description.contains("CL-WT=power")
            );
            assert_eq!(text.contains("KA ~ WT exponential"), ka_gene);
            assert_eq!(r.effects.len(), text.matches(" ~ WT ").count());
        }
        // Every point the input's own structure does not occupy moves the
        // objective; its own point re-evaluates the input.
        if r.description == "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;CL-WT=none;KA-WT=none" {
            assert_eq!(r.ofv.unwrap().to_bits(), input_ofv.to_bits(), "{}", r.id);
        }
    }
    // 32 points − 8 bolus-with-lag (INST × ON × 2 × 2 × 2) − 4 dead-gene
    // duplicates (INST × OFF × 2 × 2 × KA gene) = 20 fits.
    assert_eq!(fitted, 20);
    assert_eq!(result.n_fitted(), 21, "the input is fitted too");

    // The penalized criterion charged every candidate for its parameters
    // and for the covariance step it did not run: a second compartment is
    // two θ, so +20 over its one-compartment sibling at equal OFV.
    let p = result.options.penalties;
    let one = result
        .rows
        .iter()
        .find(|r| r.description == "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=0;CL-WT=none;KA-WT=none")
        .unwrap();
    let two = result
        .rows
        .iter()
        .find(|r| r.description == "ABSORPTION=FO;LAGTIME=OFF;PERIPHERALS=1;CL-WT=none;KA-WT=none")
        .unwrap();
    assert_eq!(one.n_parameters, Some(7));
    assert_eq!(two.n_parameters, Some(9));
    assert!(
        ((two.criterion - two.ofv.unwrap()) - (one.criterion - one.ofv.unwrap()) - 2.0 * p.theta)
            .abs()
            < 1e-9
    );
    assert!(
        (one.criterion - one.ofv.unwrap() - (7.0 * 10.0 + p.covariance + p.convergence)).abs()
            < 1e-9,
        "7 parameters, no covariance step, an evaluation is not converged"
    );

    let models = std::fs::read_to_string(dir.path().join("run/models.csv")).unwrap();
    assert_eq!(models.lines().count(), 1 + 33);
    assert!(dir.path().join("run/candidates/candidates.csv").exists());
    assert!(dir.path().join("run/final.ferx").exists());
    assert!(dir.path().join("run/generations.csv").exists());
}

#[test]
fn the_ga_agrees_with_exhaustive_enumeration_and_resumes_without_refitting() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        GRID,
        "[globalsearch]\nalgorithm = \"exhaustive\"\n",
    );
    let exhaustive = run(dir.path(), &path, false);
    let best = exhaustive.row(&exhaustive.final_id).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        GRID,
        "[globalsearch]\nalgorithm = \"ga\"\n[globalsearch.ga]\npopulation_size = 8\n\
         generations = 4\nseed = 7\n",
    );
    let ga = run(dir.path(), &path, false);
    let winner = ga.row(&ga.final_id).unwrap();
    assert_eq!(winner.description, best.description, "{:?}", ga.notes);
    assert_eq!(winner.fitness.to_bits(), best.fitness.to_bits());
    assert!(
        ga.n_fitted() < exhaustive.n_fitted(),
        "{} fits",
        ga.n_fitted()
    );
    assert_eq!(ga.generations.len(), 5);
    let n_fitted = ga.n_fitted();
    let fitted_seconds: f64 = ga.rows.iter().map(|r| r.seconds).sum();
    assert!(fitted_seconds > 0.0);

    // Resume: the seeded GA proposes the same genomes, every batch is in
    // its journal, and nothing is fitted again.
    let resumed = run(dir.path(), &path, true);
    assert_eq!(resumed.final_id, ga.final_id);
    assert_eq!(resumed.n_fitted(), n_fitted);
    assert!(
        resumed
            .rows
            .iter()
            .all(|r| r.reused || r.error.is_some() || r.duplicate_of.is_some()),
        "a resumed run fitted something: {:?}",
        resumed
            .rows
            .iter()
            .filter(|r| !r.reused)
            .map(|r| &r.id)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        resumed.rows.iter().map(|r| r.seconds).sum::<f64>(),
        fitted_seconds
    );
    for (a, b) in ga.rows.iter().zip(&resumed.rows) {
        assert_eq!(a.description, b.description);
        assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "{}", a.id);
    }
}

#[test]
fn a_covariate_only_grid_needs_no_structural_axis() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_config(
        dir.path(),
        "COVARIATE?(@IIV, WT, pow)",
        "[globalsearch]\nalgorithm = \"exhaustive\"\n[rank]\ntype = \"bic\"\n",
    );
    let result = run(dir.path(), &path, false);
    // Three parameters carry an η: 2³ points.
    assert_eq!(result.space_size, 8);
    assert_eq!(result.axes.len(), 3);
    assert!(result.rows[1..].iter().all(|r| r.structure.is_none()));
    assert!(
        result.rows[1..].iter().all(|r| r.error.is_none()),
        "{:?}",
        result.notes
    );
    assert_eq!(result.criterion.label(), "bic_mixed");
    // The axes follow the η order of the base (`@IIV`): CL, V, KA.
    assert_eq!(
        result
            .axes
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>(),
        vec!["CL-WT", "V-WT", "KA-WT"]
    );
    let all = result
        .rows
        .iter()
        .find(|r| r.description == "CL-WT=power;V-WT=power;KA-WT=power")
        .unwrap();
    assert_eq!(all.effects.len(), 3);
    assert_eq!(all.n_parameters, Some(10));
}
