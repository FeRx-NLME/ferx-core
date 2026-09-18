//! SAEM hot-path optimisations (#1447): the per-subject `EventSchedule` cache,
//! the per-proposal allocation hoist, and the `[individual_parameters]`
//! η-independent prefix hoist.
//!
//! All three ship **default-on** on the strength of one claim: they do not
//! change what the fit computes, only how fast it computes it. That is a claim
//! about behaviour, so the tests here assert it on the raw bits of a whole
//! SAEM run rather than on a tolerance — a tolerance here would be a test that
//! has stopped testing, because SAEM's per-iteration state is the input to the
//! next iteration and a one-ULP divergence at iteration 1 is a visibly
//! different fit by iteration 5.
//!
//! **What each test can see, and how that was established.** An equality A/B
//! is blind to the optimisation simply being absent (both arms then take the
//! same path and agree), so none of these is mutation-tested by deleting the
//! fix. What they exist to catch is a *stale* reuse — a schedule or a memoised
//! prefix served for inputs it was not built at — and each one is mutation-
//! tested against exactly that, by widening the soundness gate that currently
//! keeps the unsound case out. The transcripts are in each test's doc comment.

use super::{mh_steps, run_saem, MhScratch, ScheduleCacheOff};
use crate::estimation::inner_optimizer::build_schedule_cache;
use crate::parser::model_parser::{
    ip_cache_counters, ip_cache_reset, parse_model_string, IpHoistOff,
};
use crate::stats::likelihood::individual_nll;
use crate::types::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

// ─── Fixtures ─────────────────────────────────────────────────────────────

/// Body shared by every fixture: covariate algebra heavy enough to clear the
/// hoist's cost gate (`FFM`, a maturation `powf` pair) sitting in front of the
/// η-dependent statements, plus a `#484` residual-magnitude expression so
/// `IndividualNllPrep` carries a real per-observation multiplier rather than
/// the `None` fast path.
fn model_src(cl_random: &str, extra_params: &str, structural: &str) -> String {
    format!(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  theta MATSLOPE(1.5, 0.1, 10.0)
  theta RUV_LATE(1.4, 0.1, 10.0)
{extra_params}  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP ~ 0.04

[individual_parameters]
  FFM = 9270 * WT / (6680 + 216 * WT / (HT / 100)^2)
  MAT = (AGE^MATSLOPE) / (AGE^MATSLOPE + 2.0^MATSLOPE)
  CL  = TVCL * (FFM / 55)^0.75 * MAT * {cl_random}
  V   = TVV * (FFM / 55) * exp(ETA_V)
{structural}
[error_model]
  DV ~ proportional(PROP * (if (TIME > 12.0) RUV_LATE else 1.0))
"
    )
}

/// Plain analytic event-driven model: closed-form 1-cpt IV, no IOV.
fn analytic_model() -> CompiledModel {
    parse_model_string(&model_src(
        "exp(ETA_CL)",
        "",
        "\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n",
    ))
    .expect("analytic model parses")
}

/// Same model with the `[individual_parameters]` hoist compiled out.
fn analytic_model_unhoisted() -> CompiledModel {
    let _off = IpHoistOff::enter();
    analytic_model()
}

/// IOV variant — the arm that routes through `individual_nll_iov_with_scratch`
/// rather than `individual_nll_prepared`, and so exercises a different half of
/// the allocation hoist.
fn iov_model() -> CompiledModel {
    parse_model_string(&model_src(
        "exp(ETA_CL + KAPPA_CL)",
        "  kappa KAPPA_CL ~ 0.03\n",
        "\n[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n",
    ))
    .expect("IOV model parses")
}

fn iov_model_unhoisted() -> CompiledModel {
    let _off = IpHoistOff::enter();
    iov_model()
}

/// A model whose dose arrival time moves with η. `cacheable_schedule` declines
/// it — a schedule bakes per-dose times in, so reusing one across η would be
/// stale — which is exactly why it is the mutation target: widen that gate and
/// the A/B below must go red.
fn lagtime_model() -> CompiledModel {
    parse_model_string(&model_src(
        "exp(ETA_CL)",
        "  theta TVLAG(0.7, 0.01, 5.0)\n  omega ETA_LAG ~ 0.08\n",
        "  KA = 1.3\n  LAGTIME = TVLAG * exp(ETA_LAG)\n\
         \n[structural_model]\n  pk one_cpt_oral(cl=CL, v=V, ka=KA, lagtime=LAGTIME)\n",
    ))
    .expect("lagtime model parses")
}

fn covmap(wt: f64, ht: f64, age: f64) -> HashMap<String, f64> {
    HashMap::from([
        ("WT".to_string(), wt),
        ("HT".to_string(), ht),
        ("AGE".to_string(), age),
    ])
}

/// Six subjects, all carrying **time-varying** covariates (that is what routes
/// them onto the event-driven analytic walk, and so what makes the schedule
/// cacheable at all) with a different covariate trajectory each, so the
/// per-thread prefix cache is asked for a different answer subject to subject.
fn tv_population(with_occasions: bool) -> Population {
    let subjects = (0..6usize)
        .map(|i| {
            let f = i as f64;
            let wt0 = 58.0 + 7.0 * f;
            Subject {
                id: format!("{}", i + 1),
                doses: vec![
                    DoseEvent::new(0.0, 100.0 + 10.0 * f, 1, 0.0, false, 0.0),
                    DoseEvent::new(24.0, 100.0 + 10.0 * f, 1, 0.0, false, 0.0),
                ],
                dose_occasions: if with_occasions {
                    vec![1, 2]
                } else {
                    Vec::new()
                },
                dose_covariates: vec![
                    covmap(wt0, 160.0 + f, 4.0 + f),
                    covmap(wt0 + 3.0, 160.0 + f, 4.0 + f),
                ],
                obs_times: vec![1.0, 6.0, 25.0, 30.0],
                observations: vec![9.0 + f, 6.0 + 0.5 * f, 11.0 + f, 7.5 + 0.5 * f],
                obs_cmts: vec![1; 4],
                occasions: if with_occasions {
                    vec![1, 1, 2, 2]
                } else {
                    Vec::new()
                },
                obs_covariates: vec![
                    covmap(wt0, 160.0 + f, 4.0 + f),
                    covmap(wt0 + 1.0, 160.0 + f, 4.0 + f),
                    covmap(wt0 + 3.0, 160.0 + f, 4.1 + f),
                    covmap(wt0 + 4.0, 160.0 + f, 4.1 + f),
                ],
                covariates: covmap(wt0, 160.0 + f, 4.0 + f),
                cens: vec![0; 4],
                ..Default::default()
            }
        })
        .collect();
    Population {
        subjects,
        covariate_names: vec!["WT".into(), "HT".into(), "AGE".into()],
        dv_column: "DV".into(),
        input_columns: vec![],
        exclusions: None,
        warnings: vec![],
    }
}

fn saem_opts() -> FitOptions {
    FitOptions {
        saem_n_exploration: 4,
        saem_n_convergence: 3,
        saem_n_mh_steps: 4,
        saem_omega_burnin: 0,
        saem_seed: Some(1_447_000),
        run_covariance_step: false,
        verbose: false,
        ..FitOptions::default()
    }
}

/// Every bit a SAEM run produces that the optimisations could move: the
/// estimates, the objective, and the per-subject posterior summaries. Compared
/// as `u64` so no comparison can silently become a tolerance.
fn run_bits(model: &CompiledModel, pop: &Population) -> Vec<u64> {
    let r = run_saem(model, pop, &model.default_params, &saem_opts()).expect("SAEM run");
    let mut out = vec![r.ofv.to_bits(), r.n_iterations as u64];
    out.extend(r.params.theta.iter().map(|v| v.to_bits()));
    out.extend(r.params.omega.matrix.iter().map(|v| v.to_bits()));
    out.extend(r.params.sigma.values.iter().map(|v| v.to_bits()));
    for e in &r.eta_hats {
        out.extend(e.iter().map(|v| v.to_bits()));
    }
    for subj in &r.kappas {
        for k in subj {
            out.extend(k.iter().map(|v| v.to_bits()));
        }
    }
    out
}

/// How many subjects the shared gate actually hands a schedule. A fixture where
/// this is zero cannot observe anything about the cache, which is the fast-path
/// trap `CLAUDE.md` names: the code short-circuits before the thing under test.
fn n_cached(model: &CompiledModel, pop: &Population) -> usize {
    build_schedule_cache(model, pop)
        .iter()
        .filter(|s| s.is_some())
        .count()
}

// ─── The `EventSchedule` cache ────────────────────────────────────────────

/// Caching the per-subject schedule for the whole fit must be invisible: the
/// same seed, the same model, the same data, the same bits.
///
/// Runs both an analytic and an IOV model, and states the engine each takes:
/// every subject here has time-varying covariates and the model is closed-form
/// 1-cpt IV, so all six route onto the **event-driven analytic** walk — the
/// only path `EventSchedule` exists on — and the assertion below checks that
/// rather than assuming it.
///
/// Mutation check (run, not reasoned): deleting `&& !model.has_lagtime()` from
/// `inner_optimizer::cacheable_schedule` — the smallest edit that makes a
/// *stale* schedule reachable — reddens
/// `schedule_cache_declines_an_eta_dependent_lagtime`, the fixture whose dose
/// arrival moves with η. Deleting the schedule argument from `mh_steps` or from
/// the M-step instead leaves all of these green *by construction*, because both
/// arms then take the uncached path; the non-degeneracy assertion below is what
/// keeps that from being mistaken for a passing test.
#[test]
fn saem_schedule_cache_is_bit_identical() {
    for (name, model, pop) in [
        ("analytic", analytic_model(), tv_population(false)),
        ("iov", iov_model(), tv_population(true)),
    ] {
        assert_eq!(
            n_cached(&model, &pop),
            pop.subjects.len(),
            "{name}: the fixture must actually reach the event-driven analytic \
             path, or the cache under test is never consulted"
        );
        let cached = run_bits(&model, &pop);
        let uncached = {
            let _off = ScheduleCacheOff::enter();
            assert_eq!(
                n_cached(&model, &pop),
                pop.subjects.len(),
                "{name}: the guard is a `run_saem`-local switch and must not \
                 change what the shared builder returns"
            );
            run_bits(&model, &pop)
        };
        assert_eq!(
            cached, uncached,
            "{name}: the cached-schedule fit diverged from the rebuild-per-call fit"
        );
    }
}

/// The gate's job is to decline what cannot be reused. An η-dependent
/// `lagtime` moves every dose's baked-in arrival time as the MH chain walks η,
/// so this subject must get `None` — and the A/B must still agree, which it
/// trivially does while the gate holds and stops doing the moment it does not.
///
/// Mutation check: deleting `&& !model.has_lagtime()` from `cacheable_schedule`
/// reddens both assertions here (the count first, then, with the count check
/// removed as well, the bit comparison), and reddens nothing in
/// `saem_schedule_cache_is_bit_identical`. That is the pair that shows the A/B
/// above can see a stale schedule at all.
#[test]
fn schedule_cache_declines_an_eta_dependent_lagtime() {
    let model = lagtime_model();
    let pop = tv_population(false);
    assert!(
        model.has_lagtime(),
        "the fixture must declare a lagtime, or it is the same case as the \
         analytic model above"
    );
    assert_eq!(
        n_cached(&model, &pop),
        0,
        "a model whose dose arrival moves with η must not get a cached schedule"
    );
    let cached = run_bits(&model, &pop);
    let uncached = {
        let _off = ScheduleCacheOff::enter();
        run_bits(&model, &pop)
    };
    assert_eq!(
        cached, uncached,
        "the lagtime fit must be unaffected by the cache switch, because it \
         never gets a cache"
    );
}

// ─── The `[individual_parameters]` prefix hoist ───────────────────────────

/// The hoist memoises η-independent covariate algebra per thread. Over a whole
/// SAEM fit that cache is asked for six subjects' worth of covariate
/// trajectories on every worker, on both sides of a θ that the prefix reads
/// (`MATSLOPE`) and that the M-step moves every iteration — so a key that
/// omitted any of those would serve a stale prefix and the two runs would part.
///
/// Both models here run on the **event-driven analytic** engine (see
/// `saem_schedule_cache_is_bit_identical`), where `pk_param_fn` is called once
/// per event rather than once per evaluation, which is the access pattern the
/// hoist was written for.
///
/// Mutation check (run, not reasoned): deleting the covariate extend from the
/// cache key in `build_pk_param_fn` reddens both arms of this test; the
/// per-input probes in `model_parser_tests` localise which input was dropped.
#[test]
fn saem_individual_parameter_hoist_is_bit_identical() {
    for (name, hoisted, unhoisted, pop) in [
        (
            "analytic",
            analytic_model(),
            analytic_model_unhoisted(),
            tv_population(false),
        ),
        (
            "iov",
            iov_model(),
            iov_model_unhoisted(),
            tv_population(true),
        ),
    ] {
        assert!(
            closure_hoists(&hoisted),
            "{name}: the fixture's `[individual_parameters]` block has nothing \
             hoistable (or the cost gate declined it), so this comparison is \
             between two identical programs"
        );
        assert!(
            !closure_hoists(&unhoisted),
            "{name}: the unhoisted arm still compiled a split, so the two arms \
             are the same program and the comparison is vacuous"
        );
        assert_eq!(
            run_bits(&hoisted, &pop),
            run_bits(&unhoisted, &pop),
            "{name}: the hoisted fit diverged from the unsplit fit"
        );
    }
}

/// Whether this model's `pk_param_fn` actually compiled a hoisted prefix.
///
/// The split lives inside the closure, not on the `CompiledModel`, and the
/// `IndivParamProgram` snapshot is deliberately the *unsplit* list either way
/// — so the question cannot be answered by inspecting the model. It is
/// answered by observing the thing itself: a hoisted closure consults the
/// per-thread prefix cache (recording a hit or a miss), an unsplit one never
/// touches it.
fn closure_hoists(model: &CompiledModel) -> bool {
    let eta = vec![0.0; model.n_eta + model.n_kappa];
    let cov = covmap(70.0, 170.0, 5.0);
    ip_cache_reset();
    let _ = (model.pk_param_fn)(&model.default_params.theta, &eta, &cov, 1.0);
    let (hits, misses) = ip_cache_counters();
    hits + misses > 0
}

// ─── The per-proposal allocation hoist ────────────────────────────────────

/// `MhScratch` is reused across every subject a rayon worker touches, so every
/// field has to be fully re-established per subject — not just resized. The
/// risky one is `IndividualNllPrep`, which caches the per-observation residual
/// dispatch keys and the `#484` magnitude multipliers: both are subject-static
/// but **not** subject-independent, and the fixture's magnitude expression
/// (`if (TIME > 12.0) …`) differs per observation, so a prep left over from
/// another subject is a wrong likelihood rather than a crash.
///
/// The oracle is a scratch that has never seen another subject.
///
/// Mutation check (run, not reasoned): deleting `self.prep.refresh(...)` from
/// `MhScratch::begin_subject` reddens this test — and, through the E-step, the
/// three SAEM tests above as well. Dropping the `eta_prop.clear()` that
/// precedes the `resize` does **not** redden anything, and that is correct
/// rather than a hole: `n_eta` is fixed for a fit, so the `resize` is a no-op
/// and every element is assigned before it is read. The `clear()` is kept as a
/// statement of the invariant, not as a live guard.
#[test]
fn mh_scratch_reuse_across_subjects_is_bit_identical() {
    let model = analytic_model();
    let pop = tv_population(false);
    let params = &model.default_params;
    let theta = &params.theta;
    let omega = &params.omega;
    let sigma = &params.sigma.values;
    let n_eta = model.n_eta;

    // One scratch walked across every subject, exactly as a rayon worker does.
    let mut shared = MhScratch::default();
    let mut reused = Vec::new();
    for subject in &pop.subjects {
        let mut eta = vec![0.11, -0.07];
        let nll0 = individual_nll(&model, subject, theta, &eta, omega, sigma);
        let mut rng = StdRng::seed_from_u64(9_001);
        shared.begin_subject(&model, subject, theta, n_eta);
        let (acc, nll) = mh_steps(
            &mut eta,
            nll0,
            subject,
            &model,
            theta,
            omega,
            sigma,
            0.3,
            None,
            &mut rng,
            12,
            &mut shared,
            None,
            None,
        );
        reused.push((eta, acc, nll.to_bits()));
    }

    // A fresh scratch per subject — the same computation with no carried state.
    let mut fresh_out = Vec::new();
    for subject in &pop.subjects {
        let mut eta = vec![0.11, -0.07];
        let nll0 = individual_nll(&model, subject, theta, &eta, omega, sigma);
        let mut rng = StdRng::seed_from_u64(9_001);
        let mut scratch = MhScratch::default();
        scratch.begin_subject(&model, subject, theta, n_eta);
        let (acc, nll) = mh_steps(
            &mut eta,
            nll0,
            subject,
            &model,
            theta,
            omega,
            sigma,
            0.3,
            None,
            &mut rng,
            12,
            &mut scratch,
            None,
            None,
        );
        fresh_out.push((eta, acc, nll.to_bits()));
    }

    // Non-degeneracy: the chain has to have moved, and the subjects have to be
    // distinguishable, or a carried-over prep would produce the same numbers
    // anyway.
    assert!(
        reused.iter().any(|(_, acc, _)| *acc > 0),
        "no proposal was ever accepted; the sweep is inert"
    );
    let distinct: std::collections::BTreeSet<u64> = reused.iter().map(|(_, _, nll)| *nll).collect();
    assert_eq!(
        distinct.len(),
        reused.len(),
        "the subjects score identically, so a prep carried from one to the next \
         would be invisible"
    );

    for (i, (a, b)) in reused.iter().zip(fresh_out.iter()).enumerate() {
        assert_eq!(
            a.2, b.2,
            "subject {i}: reused-scratch NLL differs from the fresh-scratch one"
        );
        assert_eq!(a.1, b.1, "subject {i}: acceptance count differs");
        let (ab, bb): (Vec<u64>, Vec<u64>) = (
            a.0.iter().map(|v| v.to_bits()).collect(),
            b.0.iter().map(|v| v.to_bits()).collect(),
        );
        assert_eq!(ab, bb, "subject {i}: final η differs");
    }
}
