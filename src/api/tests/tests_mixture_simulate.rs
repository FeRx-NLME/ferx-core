//! #2163: a simulated subject of a `[mixture]` model runs under a class drawn from the
//! mixing probabilities, and that class's `MIXNUM`, Ω and Σ, on every simulation path:
//! `simulate*`, `simulate_with_uncertainty`, `simulate_adaptive` and the NPDE reference.
//!
//! The oracle throughout is the **class twin**: the single-population model of one class,
//! the mixture model with `MIXNUM` fixed. At a logit of ±1000 one class has probability 0
//! in `f64`, so the mixture run must reproduce the twin bit for bit. That equality also
//! pins that the class draw consumes nothing from the η stream: the twin draws no class,
//! so a draw taken from the η stream would shift every η after it.

use super::*;
use crate::parser::model_parser::{current_mixture_class, parse_model_string};

/// A model source. `mixl` is the `FIX`ed class-1 logit, `omega` Ω's variance and
/// `overrides` extra `[mixture]` lines. `class = Some(k)` writes the class-`k` twin instead
/// (no `[mixture]`, the class's CL, and `overrides` must be empty); `ode` writes the
/// structural model as an ODE, which the adaptive driver requires.
fn source(mixl: f64, omega: f64, overrides: &str, class: Option<usize>, ode: bool) -> String {
    let cl = match class {
        None => "if (MIXNUM == 1) TVCL * exp(ETA_CL) else 4 * TVCL * exp(ETA_CL)",
        Some(1) => "TVCL * exp(ETA_CL) + 0 * MIXL",
        Some(_) => "4 * TVCL * exp(ETA_CL) + 0 * MIXL",
    };
    let mixture = match class {
        None => format!("[mixture]\n  nsub = 2\n  logit(1) = MIXL\n{overrides}\n"),
        Some(_) => String::new(),
    };
    let structural = if ode {
        "[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n\
         [scaling]\n  y = central / V\n"
    } else {
        "[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n"
    };
    format!(
        "[parameters]\n  theta TVCL(1.0, 0.01, 100.0)\n  theta TVV(10.0, 0.1, 1000.0)\n  \
         theta MIXL({mixl}, -2000.0, 2000.0, FIX)\n  omega ETA_CL ~ {omega}\n  \
         sigma EPS ~ 0.04\n{mixture}[individual_parameters]\n  CL = {cl}\n  V  = TVV\n\
         {structural}[error_model]\n  DV ~ proportional(EPS)\n"
    )
}

fn parse(src: &str) -> CompiledModel {
    parse_model_string(src).unwrap_or_else(|e| panic!("{e}\n{src}"))
}

/// Two classes; class 2 clears 4× faster, so a class mix-up is a 4× difference in CL.
fn mixture_model(mixl: f64) -> CompiledModel {
    parse(&source(mixl, 0.09, "", None, false))
}

/// The single-population model of class `class`, with the same θ, Ω and Σ.
fn class_twin(class: usize, mixl: f64) -> CompiledModel {
    parse(&source(mixl, 0.09, "", Some(class), false))
}

/// A dose of 100 at t = 0, observed at 0.5, 1, 2, 4 and 8, for subjects `1..=n`.
fn population(n: usize) -> Population {
    use std::io::Write;
    let mut csv = String::from("ID,TIME,DV,AMT,EVID,CMT\n");
    for id in 1..=n {
        csv.push_str(&format!("{id},0,0,100,1,1\n"));
        for t in [0.5, 1.0, 2.0, 4.0, 8.0] {
            csv.push_str(&format!("{id},{t},1.0,0,0,1\n"));
        }
    }
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(csv.as_bytes()).unwrap();
    crate::read_nonmem_csv(f.path(), None, None).unwrap()
}

type Row = (String, usize, usize, f64, f64, f64);

/// `(id, draw, sim, time, ipred, value)` of each row.
fn tuples(rows: &[SimulationResult]) -> Vec<Row> {
    rows.iter()
        .map(|r| {
            let SimOutcome::Continuous { value } = r.outcome else {
                panic!("a Gaussian endpoint")
            };
            (r.id.clone(), r.draw, r.sim, r.time, r.ipred, value)
        })
        .collect()
}

fn rows(model: &CompiledModel, pop: &Population) -> Vec<Row> {
    tuples(&simulate_with_seed(model, pop, &model.default_params, 3, 2163).unwrap())
}

/// With `logit(1) = -1000` class 1 has probability `exp(-1000) = 0` in `f64`, so every
/// subject is class 2 and `simulate_with_seed` must reproduce the class-2 single-population
/// model row for row. Before #2163 it reproduced class 1: `MIXNUM` read its default.
#[test]
fn a_certain_class_2_is_simulated_as_the_class_2_model() {
    let pop = population(4);
    let got = rows(&mixture_model(-1000.0), &pop);
    assert_eq!(got, rows(&class_twin(2, -1000.0), &pop));
    assert_ne!(got, rows(&class_twin(1, -1000.0), &pop));
}

/// The other side: at `logit(1) = +1000` every subject is class 1. Without it, a draw
/// that always picked the last class would pass the test above.
#[test]
fn a_certain_class_1_is_simulated_as_the_class_1_model() {
    let pop = population(4);
    let got = rows(&mixture_model(1000.0), &pop);
    assert_eq!(got, rows(&class_twin(1, 1000.0), &pop));
    assert_ne!(got, rows(&class_twin(2, 1000.0), &pop));
}

/// Class 2's own Ω and Σ.
const OVERRIDES: &str = "  omega(2) ETA_CL ~ 0.36\n  sigma(2) EPS ~ 0.09";

/// The class-2 twin of a mixture carrying [`OVERRIDES`]: their Ω and Σ as its only ones.
fn override_twin(ode: bool) -> String {
    source(-1000.0, 0.36, "", Some(2), ode).replace("EPS ~ 0.04", "EPS ~ 0.09")
}

/// A class's `omega(2)` / `sigma(2)` override is what its subjects draw η and ε from. The
/// twin declares the override as its only Ω and Σ, so drawing either from the base would
/// change η (and IPRED) or the residual (and DV) and fail the equality; the straddle with
/// the override-free mixture shows the override is live.
#[test]
fn a_class_draws_eta_and_eps_from_its_own_omega_and_sigma() {
    let pop = population(4);
    let overridden = parse(&source(-1000.0, 0.09, OVERRIDES, None, false));
    let got = rows(&overridden, &pop);
    assert_eq!(got, rows(&parse(&override_twin(false)), &pop));
    assert_ne!(got, rows(&mixture_model(-1000.0), &pop));
}

/// The class 2 IPRED at t = 8 is `10·exp(-3.2) ≈ 0.41` and class 1's `10·exp(-0.8) ≈ 4.5`;
/// at Ω = 1e-4 (η's SD 0.01) the two never overlap, so the IPRED names the class.
/// A fit carries a class's `FIX`-ed zero `omega(2)` as the regularised `1e-8`, and the
/// draw recovers it as exactly 0 from that class's declaration (#1605), which takes its
/// FIX flag from the override and not from the base η. The twin declares the zero as its
/// only Ω, so its η is exactly 0. A carried `1e-8` drawn as is gives η = 1e-4·z instead,
/// which changes IPRED, so the rows are not equal.
///
/// The flag is checked on both sides. A FIX override on an estimated base η is
/// recovered. An estimated override on a FIX base η is not: the base flag must not
/// leak into it. Taking the class's declared Ω, or its FIX flags, from the base breaks
/// one of the two equalities. Both `simulate` and the NPDE reference run.
///
/// NPDE ranks the observation among the reference draws, so a draw that moves by
/// η = 1e-4·z must be able to cross the observation. Each observation therefore sits a
/// relative 1e-5 above the class-2 typical curve 10·exp(−0.4·t), and Σ is negligible
/// (EPS variance 1e-14, a relative SD of 1e-7). With η exactly 0, every draw falls
/// below the observation. With η = 1e-4·z, the relative spread at t = 0.5 is already
/// 2e-5, and some draws land above it.
#[test]
fn a_fit_carried_class_zero_is_recovered_from_the_class_declaration() {
    let mut pop = population(4);
    for s in &mut pop.subjects {
        for (dv, t) in s.observations.iter_mut().zip(&s.obs_times) {
            *dv = 10.0 * (-0.4 * t).exp() * (1.0 + 1e-5);
        }
    }
    let tiny = |src: String| parse(&src.replace("EPS ~ 0.04", "EPS ~ 1e-14"));
    let twin = tiny(
        source(-1000.0, 0.09, "", Some(2), false)
            .replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.0 FIX"),
    );
    let fit_carried = |m: &CompiledModel| {
        let mut p = m.default_params.clone();
        let names = p.omega.eta_names.clone();
        let carried = DMatrix::from_row_slice(1, 1, &[1e-8]);
        p.mixture.as_mut().unwrap().omega[1] = OmegaMatrix::from_matrix(carried, names, true);
        assert!(p.mixture.as_ref().unwrap().omega[1].sim_chol.is_none());
        p
    };
    let sim = |m: &CompiledModel, p: &ModelParameters| {
        tuples(&simulate_with_seed(m, &pop, p, 3, 2163).unwrap())
    };
    let npde = |m: &CompiledModel, p: &ModelParameters| {
        let r = crate::stats::npde::compute_npde_npd(m, &pop, p, 50, Some(7), None).unwrap();
        r.iter()
            .map(|s| {
                s.npd
                    .iter()
                    .chain(&s.npde)
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let want = (
        sim(&twin, &twin.default_params),
        npde(&twin, &twin.default_params),
    );

    let fixed = tiny(source(
        -1000.0,
        0.09,
        "  omega(2) ETA_CL ~ 0.0 FIX",
        None,
        false,
    ));
    let p = fit_carried(&fixed);
    assert!(
        want.0 == sim(&fixed, &p),
        "simulate: a FIX-0 omega(2) drew a nonzero η"
    );
    assert!(
        want.1 == npde(&fixed, &p),
        "NPDE: a FIX-0 omega(2) drew a nonzero η"
    );

    let estimated = tiny(
        source(-1000.0, 0.09, "  omega(2) ETA_CL ~ 0.0", None, false)
            .replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.09 FIX"),
    );
    let p = fit_carried(&estimated);
    assert!(
        want.0 != sim(&estimated, &p),
        "simulate: the base FIX leaked into omega(2)"
    );
    assert!(
        want.1 != npde(&estimated, &p),
        "NPDE: the base FIX leaked into omega(2)"
    );

    // The override's flag is class 2's alone: class 1 keeps the base `0.0 FIX` and still
    // recovers its zero, although `omega(2)` declares the same η estimated.
    let class_1 = |src: String| {
        let m = tiny(src.replace("omega ETA_CL ~ 0.09", "omega ETA_CL ~ 0.0 FIX"));
        let mut p = m.default_params.clone();
        let names = p.omega.eta_names.clone();
        let carried = || {
            OmegaMatrix::from_matrix(DMatrix::from_row_slice(1, 1, &[1e-8]), names.clone(), true)
        };
        p.omega = carried();
        if let Some(mx) = p.mixture.as_mut() {
            mx.omega.iter_mut().for_each(|o| *o = carried());
        }
        (m, p)
    };
    let (twin_1, _) = class_1(source(1000.0, 0.09, "", Some(1), false));
    let (shared, p) = class_1(source(1000.0, 0.09, "  omega(2) ETA_CL ~ 0.0", None, false));
    assert!(
        sim(&twin_1, &twin_1.default_params) == sim(&shared, &p),
        "simulate: omega(2)'s FIX flag leaked into class 1"
    );
}

/// A class copies the base Ω from its regularised matrix, where a declared `~ 0.0` is
/// already `1e-8`. The class must still simulate that η as exactly 0, as the base model
/// does (#1605): at default parameters with no `FIX`, for a class with no `omega(k)` and
/// for one whose override touches only another η. Read as `1e-8`, the zero η draws
/// 1e-4·z and moves IPRED, so the rows differ from the class twin's. With the override
/// present, class 1 must also keep the base ETA_CL: the override is class 2's alone.
#[test]
fn a_declared_base_zero_stays_zero_in_every_class() {
    let pop = population(4);
    let with_v = |src: String| {
        parse(
            &src.replace("sigma EPS", "omega ETA_V ~ 0.0\n  sigma EPS")
                .replace("V  = TVV", "V  = TVV * exp(ETA_V)"),
        )
    };
    let sim = |m: &CompiledModel| {
        tuples(&simulate_with_seed(m, &pop, &m.default_params, 3, 2163).unwrap())
    };
    for (class, mixl) in [(1, 1000.0), (2, -1000.0)] {
        let twin = with_v(source(mixl, 0.09, "", Some(class), false));
        let mixture = with_v(source(mixl, 0.09, "", None, false));
        assert!(
            sim(&twin) == sim(&mixture),
            "class {class} without overrides drew the base zero η from 1e-8"
        );
        let class_omega = if class == 1 { 0.09 } else { 0.36 };
        let twin = with_v(source(mixl, class_omega, "", Some(class), false));
        let mixture = with_v(source(mixl, 0.09, "  omega(2) ETA_CL ~ 0.36", None, false));
        assert!(
            sim(&twin) == sim(&mixture),
            "class {class} under an omega(2) on ETA_CL is not its twin"
        );
    }
}

fn class_at_t8(rows: &[Row]) -> Vec<(String, usize, usize, usize)> {
    rows.iter()
        .filter(|r| r.3 == 8.0)
        .map(|r| (r.0.clone(), r.1, r.2, if r.4 > 1.5 { 1 } else { 2 }))
        .collect()
}

/// The share of class 1 in `classes`.
fn share_of_class_1(classes: &[(String, usize, usize, usize)]) -> f64 {
    classes.iter().filter(|c| c.3 == 1).count() as f64 / classes.len() as f64
}

/// Under `p(1) = 0.3` the class-1 share of 50 subjects × 40 replicates is 0.3: the
/// seeded run measures 0.2985, and the bound of 0.035 is 3.4 binomial SDs (0.0102). The
/// draw is keyed by replicate, so each subject changes class across replicates; a draw
/// keyed by subject only would give each subject one class in all 40. The unseeded
/// `simulate` draws classes too, from an entropy root; its share is random, so its bound
/// is 0.1, 9.8 SDs.
#[test]
fn classes_are_drawn_at_the_mixing_probabilities_per_replicate() {
    let model = parse(&source((0.3f64 / 0.7).ln(), 1e-4, "", None, false));
    let pop = population(50);
    let seeded = class_at_t8(&tuples(
        &simulate_with_seed(&model, &pop, &model.default_params, 40, 2163).unwrap(),
    ));
    assert_eq!(seeded.len(), 2000);
    let share = share_of_class_1(&seeded);
    assert!(
        (share - 0.3).abs() < 0.035,
        "class-1 share {share}, want 0.3"
    );
    for id in 1..=50 {
        let mine: Vec<usize> = seeded
            .iter()
            .filter(|c| c.0 == id.to_string())
            .map(|c| c.3)
            .collect();
        assert!(
            mine.contains(&1) && mine.contains(&2),
            "subject {id} kept one class in all 40 replicates: {mine:?}"
        );
    }
    let unseeded = class_at_t8(&tuples(
        &simulate(&model, &pop, &model.default_params, 40).unwrap(),
    ));
    let share = share_of_class_1(&unseeded);
    assert!((share - 0.3).abs() < 0.1, "unseeded class-1 share {share}");
}

/// A subject (cloned from [`population`]) with id `id` and covariates `covariates`.
fn subject_with(id: &str, covariates: &[(&str, f64)]) -> Subject {
    let mut s = population(1).subjects.remove(0);
    s.id = id.to_string();
    s.covariates = covariates
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect();
    s
}

/// The mixing probabilities are each subject's own, at its baseline covariates: under
/// `logit(1) = 1000·(GRP − 0.5)` a subject of `GRP = 1` is class 1 and one of `GRP = 0`
/// class 2, in every replicate. `MIXNUM` holds the class while the guard lives and is
/// back at its default once it drops.
#[test]
fn a_subject_draws_its_class_at_its_own_covariates() {
    let src = source(0.0, 0.09, "", None, false)
        .replace("logit(1) = MIXL", "logit(1) = MIXL + 1000 * (GRP - 0.5)");
    let model = parse(&src);
    let classes = super::simulate::SimClasses::new(&model, &model.default_params, 7);
    for (grp, want) in [(1.0, 1), (0.0, 2)] {
        let subject = subject_with(&format!("g{grp}"), &[("GRP", grp)]);
        for sim in 1..=20 {
            let class = classes.enter(&subject, 1, sim);
            assert_eq!(
                current_mixture_class(),
                want,
                "GRP = {grp}, replicate {sim}"
            );
            drop(class);
            assert_eq!(current_mixture_class(), 1, "the guard restores MIXNUM");
        }
    }
}

/// The class uniform is a pure function of `(root, id, draw, sim)`: the same key gives
/// the same class whatever was drawn before it, and each coordinate of the key moves it.
#[test]
fn the_class_uniform_is_keyed_by_root_id_draw_and_sim() {
    use super::simulate::mixture_class_uniform as u;
    let base = u(7, "1", 1, 1);
    assert_eq!(base.to_bits(), u(7, "1", 1, 1).to_bits());
    assert!((0.0..1.0).contains(&base));
    for (what, other) in [
        ("root", u(8, "1", 1, 1)),
        ("id", u(7, "2", 1, 1)),
        ("draw", u(7, "1", 2, 1)),
        ("sim", u(7, "1", 1, 2)),
    ] {
        assert_ne!(
            base.to_bits(),
            other.to_bits(),
            "{what} does not key the draw"
        );
    }
    // Draw and sim are not interchangeable.
    assert_ne!(u(7, "1", 1, 2).to_bits(), u(7, "1", 2, 1).to_bits());
}

/// The first class whose cumulative probability exceeds `u`; a class of probability 0 is
/// never selected, at `u = 0` or past rounding.
#[test]
fn pick_mixture_class_selects_by_cumulative_probability() {
    use super::simulate::pick_mixture_class as pick;
    let (a, b) = (0.3f64.ln(), 0.7f64.ln());
    assert_eq!(pick(&[a, b], 0.0), 0);
    assert_eq!(pick(&[a, b], 0.29), 0);
    assert_eq!(pick(&[a, b], 0.31), 1);
    assert_eq!(pick(&[a, b], 0.999), 1);
    // p(1) = 0: `u = 0` is not below a cumulative 0, so class 1 is skipped.
    assert_eq!(pick(&[f64::NEG_INFINITY, 0.0], 0.0), 1);
    // A cumulative sum that rounding left at or below `u` falls back to the last class
    // of positive probability, not to a trailing class of probability 0.
    assert_eq!(pick(&[a, a, f64::NEG_INFINITY], 0.9), 1);
    assert_eq!(pick(&[f64::NEG_INFINITY, a, a], 0.9), 2);
}

/// Propensity matching estimates every subject's posthoc η under class 1, so it is refused
/// on a `[mixture]` model, by code and full text; the class twin is matched as before.
#[test]
fn propensity_matching_is_refused_on_a_mixture_model() {
    let pop = population(3);
    let opts = SimulateOptions {
        seed: Some(1),
        match_method: Some(crate::propensity_match::MatchMethod::Optimal),
        ..Default::default()
    };
    let run = |m: &CompiledModel| simulate_with_options(m, &pop, &m.default_params, 1, &opts);
    let model = mixture_model(0.0);
    let err = run(&model).expect_err("refused");
    assert_eq!(
        err.diagnostic().map(|d| d.code.as_str()),
        Some("E_PROPENSITY_INPUT")
    );
    assert_eq!(
        err.to_string(),
        "propensity-score matching is not supported on a [mixture] model: each subject's \
         posthoc eta would be estimated under class 1 rather than its own class. Simulate \
         without `match_method`; each subject then draws its class from the mixing \
         probabilities"
    );
    assert!(run(&class_twin(1, 0.0)).is_ok(), "the twin is matched");
    // Without matching, the mixture model simulates through `simulate_with_options`.
    let plain = SimulateOptions {
        seed: Some(2163),
        ..Default::default()
    };
    let got = simulate_with_options(&model, &pop, &model.default_params, 3, &plain).unwrap();
    assert_eq!(tuples(&got), rows(&model, &pop));
}

/// `simulate_with_uncertainty`: every draw's subjects run under their class. At a certain
/// class 2 the rows equal the class-2 twin's; at `p(1) = 0.5` with one replicate per draw
/// the draw is the only coordinate that varies, so a subject changing class across the
/// 30 draws pins that the draw keys the class.
#[test]
fn uncertainty_draws_simulate_each_subject_under_its_class() {
    let pop = population(4);
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 3,
        n_sim_per_draw: 2,
        seed: Some(2163),
        ..Default::default()
    };
    let run = |m: &CompiledModel, o: &SimulateUncertaintyOptions| {
        let fit = super::simulate_with_uncertainty_tests::synthetic_fit(&m.default_params);
        tuples(&simulate_with_uncertainty(m, &pop, &fit, o).unwrap())
    };
    let got = run(&mixture_model(-1000.0), &opts);
    assert_eq!(got, run(&class_twin(2, -1000.0), &opts));
    assert_ne!(got, run(&class_twin(1, -1000.0), &opts));

    let even = parse(&source(0.0, 1e-4, "", None, false));
    let opts = SimulateUncertaintyOptions {
        n_uncertainty_draws: 30,
        n_sim_per_draw: 1,
        seed: Some(2163),
        ..Default::default()
    };
    let classes = class_at_t8(&run(&even, &opts));
    for id in 1..=4 {
        let mine: Vec<usize> = classes
            .iter()
            .filter(|c| c.0 == id.to_string())
            .map(|c| c.3)
            .collect();
        assert_eq!(mine.len(), 30);
        assert!(
            mine.contains(&1) && mine.contains(&2),
            "subject {id} kept one class in all 30 draws: {mine:?}"
        );
    }
}

/// `simulate_adaptive`: the reactive driver runs each (subject, replicate) under its
/// class and that class's Ω and Σ, the frozen-schedule replay verifier (on by default)
/// under the same one; and the replicate keys the class.
#[test]
fn adaptive_runs_each_subject_under_its_class() {
    use crate::sim::adaptive::{ControllerCtx, DoseAction, MonitorSpec, ObserveMode};
    let pop = population(3);
    let mut opts = AdaptiveSimulateOptions::default();
    opts.seed = Some(2163);
    // The assay `Dv` read at t = 2 is the only place the driver uses Σ: the trajectories
    // carry IPRED. So the run also returns the bits of every reading the controller saw.
    opts.decision_times = vec![0.0, 2.0];
    opts.monitors = vec![MonitorSpec::new("dv", 1, ObserveMode::Dv)];
    let hold = || |_: &ControllerCtx| vec![DoseAction::Hold];
    let run = |src: String, n_sim: usize| {
        let m = parse(&src);
        let res = simulate_adaptive(&m, &pop, &m.default_params, n_sim, hold, &opts).unwrap();
        let read: Vec<u64> = res
            .decisions
            .iter()
            .flat_map(|d| d.observed_signals.iter().map(|s| s.value.to_bits()))
            .collect();
        assert_eq!(read.len(), 3 * 2 * n_sim);
        (tuples(&res.trajectories), read)
    };
    let got = run(source(-1000.0, 0.09, "", None, true), 2);
    assert_eq!(got, run(source(-1000.0, 0.09, "", Some(2), true), 2));
    assert_ne!(got.0, run(source(-1000.0, 0.09, "", Some(1), true), 2).0);

    // The class's own Ω and Σ: η from `omega(2)`, the assay ε from `sigma(2)`. The
    // σ-only control shows the readings see Σ, so the equality above it does too.
    let overridden = run(source(-1000.0, 0.09, OVERRIDES, None, true), 2);
    assert_eq!(overridden, run(override_twin(true), 2));
    let base_sigma = run(override_twin(true).replace("EPS ~ 0.09", "EPS ~ 0.04"), 2);
    assert_eq!(overridden.0, base_sigma.0);
    assert_ne!(overridden.1, base_sigma.1);

    let classes = class_at_t8(&run(source(0.0, 1e-4, "", None, true), 20).0);
    for id in 1..=3 {
        let mine: Vec<usize> = classes
            .iter()
            .filter(|c| c.0 == id.to_string())
            .map(|c| c.3)
            .collect();
        assert_eq!(mine.len(), 20);
        assert!(
            mine.contains(&1) && mine.contains(&2),
            "subject {id} kept one class in all 20 replicates: {mine:?}"
        );
    }
}

fn npde_bits(model: &CompiledModel, pop: &Population) -> Vec<Vec<u64>> {
    crate::stats::npde::compute_npde_npd(model, pop, &model.default_params, 50, Some(7), None)
        .unwrap()
        .iter()
        .map(|s| s.npd.iter().chain(&s.npde).map(|x| x.to_bits()).collect())
        .collect()
}

/// The NPDE reference simulation draws each replicate's class. At a certain class 2 the
/// scores equal the class-2 twin's, with and without class 2's own Ω and Σ.
///
/// At `p(1) = 0.5`, an observation sitting on the class-1 typical curve (Ω, σ small)
/// has half the class-1 replicates below it and every class-2 one (4× the clearance)
/// below it too: `pd = 0.75`, `npd = Φ⁻¹(0.75) = 0.674`. A reference drawn under one class
/// per subject gives `pd ≈ 0.5` (`npd ≈ 0`, class 1) or `pd ≈ 1` (class 2) instead. Over
/// the 30 observations at 1000 replicates the worst `|npd − 0.674|` measures 0.055; the
/// bound is 0.15.
#[test]
fn the_npde_reference_draws_each_replicates_class() {
    let pop = population(3);
    let got = npde_bits(&mixture_model(-1000.0), &pop);
    assert_eq!(got, npde_bits(&class_twin(2, -1000.0), &pop));
    assert_ne!(got, npde_bits(&class_twin(1, -1000.0), &pop));
    // The class's own Ω and Σ: η from `omega(2)`, the reference ε from `sigma(2)`.
    assert_eq!(
        npde_bits(&parse(&source(-1000.0, 0.09, OVERRIDES, None, false)), &pop),
        npde_bits(&parse(&override_twin(false)), &pop)
    );

    // The class-1 typical concentration 10·exp(-0.1·t) at each observation time.
    let mut on_class_1 = population(6);
    for s in &mut on_class_1.subjects {
        for (dv, t) in s.observations.iter_mut().zip(&s.obs_times) {
            *dv = 10.0 * (-0.1 * t).exp();
        }
    }
    let model = parse(&source(0.0, 1e-4, "", None, false).replace("EPS ~ 0.04", "EPS ~ 0.0001"));
    let scores = crate::stats::npde::compute_npde_npd(
        &model,
        &on_class_1,
        &model.default_params,
        1000,
        Some(7),
        None,
    )
    .unwrap();
    for (i, s) in scores.iter().enumerate() {
        for &npd in &s.npd {
            assert!(
                (npd - 0.674).abs() < 0.15,
                "subject {}: npd {npd}, want Φ⁻¹(0.75) = 0.674",
                i + 1
            );
        }
    }
}
