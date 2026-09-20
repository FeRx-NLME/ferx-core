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

use super::{cholesky_perturbation_into, mh_steps, run_saem, MhScratch, ScheduleCacheOff};
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
    run_bits_with(model, pop, &saem_opts())
}

/// [`run_bits`] at a caller-chosen iteration budget.
fn run_bits_with(model: &CompiledModel, pop: &Population, opts: &FitOptions) -> Vec<u64> {
    let r = run_saem(model, pop, &model.default_params, opts).expect("SAEM run");
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

// ─── Pre-change oracles (#1452 review, finding 1) ─────────────────────────
//
// Every A/B above runs the *new* code on both sides. That is the right shape
// for "does caching a schedule change the answer", and the wrong shape for
// "does the new likelihood entry point compute the same thing as the one it
// replaced" — dropping `prep.ruv_mult`, or changing the buffered matmul, would
// leave all of them green. The two tests here are the missing oracle: they hold
// the *old* expression next to the new one at identical inputs and compare raw
// bits.

/// A model with a covariate-selected (`if/else`) residual error, so
/// `IndividualNllPrep::err_keys` carries a per-observation key that actually
/// varies within a subject rather than the borrowed constant a `Single` spec
/// hands back.
fn selected_error_model() -> CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  theta MATSLOPE(1.5, 0.1, 10.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP_A ~ 0.04
  sigma PROP_B ~ 0.09
  sigma ADD_C ~ 0.5

[individual_parameters]
  FFM = 9270 * WT / (6680 + 216 * WT / (HT / 100)^2)
  MAT = (AGE^MATSLOPE) / (AGE^MATSLOPE + 2.0^MATSLOPE)
  CL  = TVCL * (FFM / 55)^0.75 * MAT * exp(ETA_CL)
  V   = TVV * (FFM / 55) * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  if (ASSAY == 1) {
    DV ~ proportional(PROP_A)
  } else if (ASSAY == 2) {
    DV ~ proportional(PROP_B)
  } else {
    DV ~ additive(ADD_C)
  }
",
    )
    .expect("selected-error model parses")
}

/// The `tv_population` subjects with a per-observation `ASSAY` flag added, so
/// the selector above resolves to a different endpoint on different rows of the
/// *same* subject.
fn tv_population_with_assay() -> Population {
    let mut pop = tv_population(false);
    for s in &mut pop.subjects {
        for (j, c) in s.obs_covariates.iter_mut().enumerate() {
            c.insert("ASSAY".to_string(), (j % 3) as f64 + 1.0);
        }
        s.covariates.insert("ASSAY".to_string(), 1.0);
        for c in &mut s.dose_covariates {
            c.insert("ASSAY".to_string(), 1.0);
        }
    }
    pop.covariate_names.push("ASSAY".into());
    pop
}

/// `individual_nll_prepared` must equal `individual_nll_into_with_schedule` bit
/// for bit. It is the same arithmetic with the buffers moved to the caller, and
/// this is the only test that says so — the SAEM A/Bs run it on both sides.
///
/// Covered, because each exercises a different branch of what `prep` carries:
/// a plain `Single` spec (`err_keys` borrowed, `ruv_mult` `None`), a `#484`
/// residual-magnitude expression (`ruv_mult` `Some`, one `Vec` per observation),
/// and a covariate-selected spec (`err_keys` genuinely per-observation).
/// Crossed with a cached and an uncached schedule, over six subjects and two η.
///
/// Mutation check (run, not reasoned):
/// * `prep.refresh` computing `ruv_mult` as `None` instead of
///   `model.ruv_obs_mult(..)` → 🔴 on the magnitude arm only, naming it;
/// * `err_keys` filled with `vec![0; n]` instead of `obs_keys(subject)` → 🔴 on
///   the selected arm only, naming it;
/// * passing `&model.residual_correlations` where the wrapper passes the live ρ
///   is not observable here and is not claimed to be — both take ρ from the
///   model at this call site.
#[test]
fn individual_nll_prepared_matches_the_wrapper_it_replaced() {
    use crate::stats::likelihood::{
        individual_nll_into_with_schedule, individual_nll_prepared, IndividualNllPrep,
        IndividualNllScratch,
    };

    let arms: Vec<(&str, CompiledModel, Population)> = vec![
        ("magnitude", analytic_model(), tv_population(false)),
        (
            "selected",
            selected_error_model(),
            tv_population_with_assay(),
        ),
    ];

    for (arm, model, pop) in arms {
        let params = &model.default_params;
        let theta = &params.theta;
        let omega = &params.omega;
        let sigma = &params.sigma.values;
        let schedules = build_schedule_cache(&model, &pop);
        assert_eq!(
            schedules.iter().filter(|s| s.is_some()).count(),
            pop.subjects.len(),
            "{arm}: fixture does not reach the event-driven analytic path"
        );

        // One scratch/prep reused across every subject and every η, exactly as
        // the MH sweep uses them.
        let mut scratch = IndividualNllScratch::default();
        let mut prep = IndividualNllPrep::default();
        let mut wrapper_scratch = crate::pk::EventPkParams::default();
        let mut n_compared = 0usize;

        for (i, subject) in pop.subjects.iter().enumerate() {
            prep.refresh(&model, subject, theta);
            for eta in [vec![0.0, 0.0], vec![0.21, -0.13], vec![-0.4, 0.33]] {
                for (sched_name, sched) in [("cached", schedules[i].as_ref()), ("uncached", None)] {
                    let want = individual_nll_into_with_schedule(
                        &model,
                        subject,
                        theta,
                        &eta,
                        omega,
                        sigma,
                        &model.residual_correlations,
                        &mut wrapper_scratch,
                        sched,
                    );
                    let got = individual_nll_prepared(
                        &model,
                        subject,
                        theta,
                        &eta,
                        omega,
                        sigma,
                        &prep,
                        sched,
                        &mut scratch,
                    );
                    assert!(
                        want.is_finite(),
                        "{arm}/{sched_name}: subject {i} scored {want}, which is \
                         not a live comparison"
                    );
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{arm}/{sched_name}: subject {i}, eta {eta:?}: prepared \
                         form {got} vs wrapper {want}"
                    );
                    n_compared += 1;
                }
            }
        }
        assert_eq!(n_compared, pop.subjects.len() * 3 * 2, "{arm}: coverage");
    }
}

/// The block MH kernel replaced `l * DVector::from_column_slice(&z)` with
/// `perturbation.gemm(1.0, l, &z, 0.0)`. Both reach nalgebra's `gemm` kernel,
/// but only one of them is in the tree now, so this holds the other one next to
/// it on the same seeded draws.
///
/// `beta = 0` is the part that has to be right: it must make `gemm` *write*
/// rather than accumulate, so a reused buffer contributes nothing. The test
/// pre-fills `perturbation` with garbage before every call, which is what makes
/// a wrong `beta` (or a missing overwrite) visible rather than latent.
///
/// Mutation check (run, not reasoned): changing `beta` from `0.0` to `1.0` in
/// `cholesky_perturbation_into` → 🔴 at the first reused buffer. That mutation
/// was **green** until this test stopped reimplementing the `gemm` call and
/// started calling the production helper; dropping the pre-fill makes it green
/// again, which is why the pre-fill is there.
#[test]
fn buffered_mh_proposal_matches_the_allocating_form() {
    use nalgebra::DVector;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    // A diagonal Ω and a correlated block Ω — the second is the one where a
    // wrong `gemm` operand order or a stale buffer would actually show.
    let names: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
    let omegas = [
        OmegaMatrix::from_matrix(
            nalgebra::DMatrix::from_diagonal(&DVector::from_vec(vec![0.09, 0.05, 0.16])),
            names.clone(),
            true,
        ),
        OmegaMatrix::from_matrix(
            nalgebra::DMatrix::from_row_slice(
                3,
                3,
                &[0.09, 0.03, -0.02, 0.03, 0.05, 0.01, -0.02, 0.01, 0.16],
            ),
            names.clone(),
            false,
        ),
    ];

    for (w, omega) in omegas.iter().enumerate() {
        let l = &omega.chol;
        let n = 3usize;
        let mut rng = StdRng::seed_from_u64(1_447_042);
        // The reused buffer, deliberately poisoned before each use.
        let mut perturbation = DVector::zeros(n);
        let mut z = DVector::zeros(n);
        let mut n_nonzero = 0usize;

        for step in 0..200 {
            for slot in z.iter_mut() {
                *slot = rng.sample(StandardNormal);
            }
            // The form that was in the tree before this PR.
            let want = l * DVector::from_column_slice(z.as_slice());
            // Poison the buffer so an accumulate-instead-of-write is fatal.
            perturbation.fill(f64::from(step as u32) * 1e6 + 7.0);
            // The **production** helper, not a copy of it: the first version of
            // this test reimplemented the `gemm` call, so mutating `beta` in
            // `mh_steps` left it green.
            cholesky_perturbation_into(l, &z, &mut perturbation);
            for j in 0..n {
                assert_eq!(
                    perturbation[j].to_bits(),
                    want[j].to_bits(),
                    "omega {w}, step {step}, coord {j}: buffered {} vs allocating {}",
                    perturbation[j],
                    want[j]
                );
                if want[j] != 0.0 {
                    n_nonzero += 1;
                }
            }
        }
        assert!(
            n_nonzero > 500,
            "omega {w}: the draws are degenerate ({n_nonzero} non-zero of 600), \
             so the comparison would hold for a `gemm` that wrote zeros"
        );
    }
}

// ─── Consumption, not just construction (#1452 review, finding 2) ─────────
//
// `n_cached()` says a schedule *can* be built for a subject. It says nothing
// about whether the fit then uses it, and an equality A/B cannot tell the
// difference either — unplug the cache on both sides and both sides still
// agree. The counter below closes that: a cache that is consumed makes
// `EventSchedule::for_subject` stop being called per evaluation, and that is
// directly observable.

/// Run `f` and return its result together with the number of
/// `EventSchedule::for_subject` calls it made **on this thread**.
fn with_build_count<T>(f: impl FnOnce() -> T) -> (T, u64) {
    crate::pk::event_driven::reset_schedule_build_count();
    let out = f();
    (out, crate::pk::event_driven::schedule_build_count())
}

/// Same, for a call that fans out over rayon: everything runs on the single
/// worker of a one-thread pool, and the counter is read from inside that
/// worker, so the thread-local count is complete and uncontaminated.
fn with_build_count_1thread<T: Send>(f: impl FnOnce() -> T + Send) -> (T, u64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(crate::FIT_RAYON_STACK_SIZE)
        .build()
        .expect("thread pool build");
    pool.install(|| with_build_count(f))
}

/// The E-step kernels must *consume* the schedule they are handed: with
/// `Some(..)` the sweep builds none, with `None` it builds one per likelihood
/// evaluation. Asserted per arm, with the arm named in the message, because the
/// IOV branch takes a different predictor and used to ignore the argument
/// entirely (#1452 review) — a single combined assertion would have passed on
/// the strength of the non-IOV arm alone.
///
/// Bit-identity between the two is asserted at the same time: the cache must be
/// invisible as well as effective.
#[test]
fn mh_steps_consumes_the_cached_schedule_on_both_arms() {
    for (arm, model, pop) in [
        ("non-iov", analytic_model(), tv_population(false)),
        ("iov", iov_model(), tv_population(true)),
    ] {
        let params = &model.default_params;
        let theta = &params.theta;
        let omega = &params.omega;
        let sigma = &params.sigma.values;
        let n_eta = model.n_eta;
        let schedules = build_schedule_cache(&model, &pop);
        let subject = &pop.subjects[0];
        assert!(
            schedules[0].is_some(),
            "{arm}: no schedule for the fixture subject"
        );
        // IOV kernels need a κ per occasion group.
        let kappas: Vec<Vec<f64>> = if model.n_kappa > 0 {
            crate::stats::likelihood::iov_occasion_groups(subject)
                .iter()
                .map(|_| vec![0.04; model.n_kappa])
                .collect()
        } else {
            Vec::new()
        };
        let omega_iov = params.omega_iov.clone();
        let kappas_opt = omega_iov
            .as_ref()
            .filter(|_| model.n_kappa > 0)
            .map(|oi| (kappas.as_slice(), oi));

        const N_STEPS: usize = 20;
        // Computed outside the counted region: it is the same call on both arms
        // and it builds a schedule of its own, which would otherwise show up as
        // a phantom leak in the cached count.
        let eta0 = vec![0.11, -0.07];
        let nll0 = individual_nll(&model, subject, theta, &eta0, omega, sigma);
        let run = |sched: Option<&crate::pk::event_driven::EventSchedule>| {
            let mut eta = eta0.clone();
            let mut rng = StdRng::seed_from_u64(4_242);
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
                N_STEPS,
                &mut scratch,
                sched,
                kappas_opt,
            );
            (
                eta.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(),
                acc,
                nll.to_bits(),
            )
        };

        let (cached, builds_cached) = with_build_count(|| run(schedules[0].as_ref()));
        let (uncached, builds_uncached) = with_build_count(|| run(None));

        assert_eq!(
            builds_cached, 0,
            "{arm}: the sweep rebuilt the schedule {builds_cached} times although \
             it was handed a cached one — the argument is being ignored"
        );
        assert!(
            builds_uncached >= N_STEPS as u64,
            "{arm}: the uncached sweep built only {builds_uncached} schedules in \
             {N_STEPS} proposals, so the fixture does not reach the event-driven \
             walk and the count above proves nothing"
        );
        assert_eq!(
            cached, uncached,
            "{arm}: the cached sweep diverged from the rebuild-per-call sweep"
        );
        assert!(
            cached.1 > 0,
            "{arm}: no proposal accepted; the sweep is inert"
        );
    }
}

/// The θ/σ M-step objective, same property, same per-arm split. Tested on the
/// per-subject kernels rather than on `obs_nll_sum*`, because those fan out over
/// rayon and the point of this test is an exact count.
#[test]
fn m_step_objective_consumes_the_cached_schedule_on_both_arms() {
    use crate::estimation::fixed_eta_gradient::obs_nll_subject_into_iov_with_schedule;
    use crate::stats::likelihood::obs_nll_subject_into_with_schedule;

    // Non-IOV kernel.
    {
        let model = analytic_model();
        let pop = tv_population(false);
        let p = &model.default_params;
        let schedules = build_schedule_cache(&model, &pop);
        let subject = &pop.subjects[0];
        let eta = vec![0.05, -0.02];
        let call = |sched| {
            obs_nll_subject_into_with_schedule(
                &model,
                subject,
                &p.theta,
                &p.sigma.values,
                &model.residual_correlations,
                &eta,
                &mut crate::pk::EventPkParams::default(),
                sched,
            )
        };
        let (cached, b_cached) = with_build_count(|| call(schedules[0].as_ref()));
        let (uncached, b_uncached) = with_build_count(|| call(None));
        assert_eq!(b_cached, 0, "non-iov M-step ignored the cached schedule");
        assert_eq!(
            b_uncached, 1,
            "non-iov M-step: the uncached call did not reach the event-driven walk"
        );
        assert_eq!(
            cached.to_bits(),
            uncached.to_bits(),
            "non-iov M-step: cached objective {cached} vs uncached {uncached}"
        );
        assert!(cached.is_finite(), "non-iov M-step objective is not finite");
    }

    // IOV kernel — the one that took no schedule at all before this PR's review
    // round, so `obs_nll_sum_iov` held a cache it could not use.
    {
        let model = iov_model();
        let pop = tv_population(true);
        let p = &model.default_params;
        let schedules = build_schedule_cache(&model, &pop);
        let subject = &pop.subjects[0];
        let eta = vec![0.05, -0.02];
        let kappas: Vec<Vec<f64>> = crate::stats::likelihood::iov_occasion_groups(subject)
            .iter()
            .map(|_| vec![0.04; model.n_kappa])
            .collect();
        assert!(kappas.len() >= 2, "the IOV fixture must have >1 occasion");
        let call = |sched| {
            obs_nll_subject_into_iov_with_schedule(
                &model,
                subject,
                &p.theta,
                &p.sigma.values,
                &eta,
                &kappas,
                &mut crate::pk::EventPkParams::default(),
                sched,
            )
        };
        let (cached, b_cached) = with_build_count(|| call(schedules[0].as_ref()));
        let (uncached, b_uncached) = with_build_count(|| call(None));
        assert_eq!(b_cached, 0, "IOV M-step ignored the cached schedule");
        assert_eq!(
            b_uncached, 1,
            "IOV M-step: the uncached call did not reach the event-driven walk"
        );
        assert_eq!(
            cached.to_bits(),
            uncached.to_bits(),
            "IOV M-step: cached objective {cached} vs uncached {uncached}"
        );
        assert!(cached.is_finite(), "IOV M-step objective is not finite");
    }
}

/// End to end: over a whole SAEM fit, the number of schedule builds must not
/// grow with the number of likelihood evaluations. That is the property a cache
/// *is*, and it is the one the equality A/Bs cannot see — unplug the cache on
/// both sides and they still agree.
///
/// Stated as invariance across two iteration counts rather than as an absolute
/// number, because a fit also builds a handful of schedules **outside** the
/// hot path and pinning a magic total would be pinning those too. Measured,
/// for the analytic fixture: `build_schedule_cache` builds one per subject, and
/// the post-loop final-EBE pass (`run_inner_loop_warm` → `find_ebe` →
/// `sens::provider::run_obs_grad_tvcov`) builds two more per subject. All three
/// are once per fit. The E-step and M-step — the `n_iter`-proportional part —
/// must contribute **zero**, and that is exactly what invariance across
/// `n_iter` asserts, with no magic constant and no knowledge of how many
/// one-off passes exist.
///
/// Mutation check (run, not reasoned): replacing `schedules[i].as_ref()` with
/// `None` at either MH call site, or `&schedules` with `&[]` in the M-step,
/// makes the cached count grow with `n_iter` → 🔴 here, while every equality
/// A/B stays green. That is the gap this test exists to cover. The bound below
/// additionally fails if a future edit adds a new per-subject build to a
/// once-per-fit pass.
#[test]
fn saem_schedule_builds_do_not_grow_with_iteration_count() {
    for (arm, model, pop) in [
        ("analytic", analytic_model(), tv_population(false)),
        ("iov", iov_model(), tv_population(true)),
        ("time-only", time_only_model(), baseline_cov_population()),
    ] {
        let n = pop.subjects.len();
        assert_eq!(
            n_cached(&model, &pop),
            n,
            "{arm}: fixture is not cacheable, so the counts below prove nothing"
        );

        let short = saem_opts();
        let long = FitOptions {
            saem_n_exploration: short.saem_n_exploration * 3,
            saem_n_convergence: short.saem_n_convergence * 3,
            ..short.clone()
        };

        let count = |opts: &FitOptions, off: bool| -> (Vec<u64>, u64) {
            with_build_count_1thread(|| {
                let _guard = off.then(ScheduleCacheOff::enter);
                run_bits_with(&model, &pop, opts)
            })
        };

        let (bits_short, cached_short) = count(&short, false);
        let (_, cached_long) = count(&long, false);
        let (bits_short_off, uncached_short) = count(&short, true);
        let (_, uncached_long) = count(&long, true);

        // The property: tripling the iteration count must not add builds. The
        // bound is not `== 0` because the once-per-fit final-EBE pass is itself
        // a numerical solve whose line-search length depends on where the chain
        // left off, so its own build count drifts between the two budgets.
        //
        // **Measured, not picked**, on this fixture under the #1449 defaults —
        // the drift is `analytic 0, iov 9 (198 → 207), time-only 0`, and the
        // iov number moved from 2 to 9 when the Robbins-Monro scale rule became
        // the default, because the chain now ends somewhere else. A single
        // *per-evaluation* leak of one build per subject per iteration adds
        // `extra_iters · n` = 84. The bound below is a quarter of that: 21, so
        // 4× clear of the leak it must catch and 2.3× clear of the worst drift
        // it must tolerate.
        let extra_iters = (long.saem_n_exploration + long.saem_n_convergence
            - short.saem_n_exploration
            - short.saem_n_convergence) as u64;
        let leak_bound = extra_iters * n as u64 / 4;
        assert!(
            cached_long.saturating_sub(cached_short) < leak_bound,
            "{arm}: schedule builds grew from {cached_short} to {cached_long} over \
             {extra_iters} extra iterations on {n} subjects, past the {leak_bound} \
             bound — the E-step, the M-step or the per-iteration NLL-cache refresh \
             is still rebuilding per evaluation"
        );
        // The counter has to be able to move, or the invariance above is the
        // invariance of a number that is always zero.
        assert!(
            uncached_long > uncached_short && uncached_short > 20 * n as u64,
            "{arm}: uncached builds {uncached_short} → {uncached_long} do not grow \
             with the iteration count, so this fixture barely reaches the \
             event-driven walk and the cached count proves nothing"
        );
        assert_eq!(
            bits_short, bits_short_off,
            "{arm}: the cached fit diverged from the rebuild-per-call fit"
        );
    }
}

/// A gap #1449 found while measuring `score_sa` as a candidate default, pinned
/// with its measured size.
///
/// `mstep_solver = score_sa` computes its score and
/// expected information through
/// `fixed_eta_gradient::obs_nll_subject_grad_fisher`, and that function takes
/// **no cached schedule**: it predicts through `compute_predictions_with_tv_into`
/// / `predict_iov` rather than their `_with_schedule` twins. On a model whose
/// `[individual_parameters]` read `TIME` the event schedule is therefore rebuilt
/// inside every M-step, where the derivative-free solver rebuilds none.
///
/// Measured on this fixture: tripling the iteration count adds **168** builds,
/// i.e. exactly `2 · extra_iters · n` (2 per subject per M-step, and the
/// score-SA M-step runs every iteration), against **0** for the same fixture
/// under `mstep_solver = bobyqa`.
///
/// This is an unrealised saving, not a regression — `score_sa` is 25–54 % less
/// CPU than `bobyqa` across the #1449 benchmark suite *including* the
/// `TIME`-reading pembrolizumab bench, so the cache would make a win larger.
/// The test asserts the leak is exactly the shape described, so that whoever
/// threads the schedule through gets a red test naming this comment rather
/// than a silent no-op, and so that the leak cannot grow unnoticed in the
/// meantime.
#[test]
fn the_score_sa_mstep_rebuilds_the_event_schedule() {
    let (model, pop) = (time_only_model(), baseline_cov_population());
    let n = pop.subjects.len() as u64;
    let short = FitOptions {
        saem_mstep_solver: crate::types::SaemMstepSolver::ScoreSa,
        ..saem_opts()
    };
    let long = FitOptions {
        saem_n_exploration: short.saem_n_exploration * 3,
        saem_n_convergence: short.saem_n_convergence * 3,
        ..short.clone()
    };
    let count = |opts: &FitOptions| -> u64 {
        with_build_count_1thread(|| run_bits_with(&model, &pop, opts)).1
    };
    let (a, b) = (count(&short), count(&long));
    let extra_iters = (long.saem_n_exploration + long.saem_n_convergence
        - short.saem_n_exploration
        - short.saem_n_convergence) as u64;
    let drift = b.saturating_sub(a);
    // Two builds per subject per iteration, on the nose (measured 30 → 198 over
    // 14 extra iterations on 6 subjects). Asserted as an equality rather than a
    // bound: a *smaller* number means someone started threading the cache
    // through and should delete this test, and a larger one is a new leak.
    assert_eq!(
        drift,
        2 * extra_iters * n,
        "score-SA M-step schedule builds {a} → {b} (drift {drift}) over {extra_iters} \
         extra iterations on {n} subjects — expected exactly two rebuilds per subject \
         per iteration; if this is now lower, `obs_nll_subject_grad_fisher` has learnt \
         to take the cached schedule and this test should go"
    );
    // The control: the same fixture and budgets under the derivative-free
    // M-step rebuild nothing, so the number above is the solver's and not the
    // fixture's.
    let bob = |opts: &FitOptions| -> u64 {
        with_build_count_1thread(|| {
            run_bits_with(
                &model,
                &pop,
                &FitOptions {
                    saem_mstep_solver: crate::types::SaemMstepSolver::Bobyqa,
                    ..opts.clone()
                },
            )
        })
        .1
    };
    assert_eq!(
        bob(&long),
        bob(&short),
        "the derivative-free M-step must still add no builds at all"
    );
}

/// A subject with only **baseline** covariates, in a model whose
/// `[individual_parameters]` reads `TIME`. The prediction dispatcher routes it
/// onto the event-driven walk (`has_tv || has_resets || uses_time`), but
/// `cacheable_schedule` used to require the first two, so it rebuilt its
/// schedule on every likelihood call while the gate reported it had no use for
/// one (#1452 review, finding 3).
///
/// Mutation check (run, not reasoned): reverting the gate to
/// `has_tv_covariates() || has_resets()` → 🔴 here at `n_cached`, and with that
/// assertion removed, 🔴 at the build count in
/// `run_saem_builds_each_cacheable_schedule_once`'s `time-only` arm. Nothing
/// else in the suite notices, which is why this fixture exists.
#[test]
fn schedule_cache_covers_a_time_only_baseline_covariate_subject() {
    let model = time_only_model();
    let pop = baseline_cov_population();
    assert!(
        pop.subjects
            .iter()
            .all(|s| !s.has_tv_covariates() && !s.has_resets()),
        "the fixture must have baseline covariates only, or it is the ordinary \
         TV case already covered above"
    );
    assert!(
        crate::pk::model_uses_time_anywhere(&model),
        "the fixture must read the TIME built-in"
    );
    assert_eq!(
        n_cached(&model, &pop),
        pop.subjects.len(),
        "a TIME-reading analytical subject takes the event-driven walk and must \
         get a cached schedule"
    );
    // And the walk is genuinely reached: without a cache, one build per call.
    let p = &model.default_params;
    let subject = &pop.subjects[0];
    let (_, builds) = with_build_count(|| {
        crate::stats::likelihood::obs_nll_subject_into_with_schedule(
            &model,
            subject,
            &p.theta,
            &p.sigma.values,
            &model.residual_correlations,
            &[0.0, 0.0],
            &mut crate::pk::EventPkParams::default(),
            None,
        )
    });
    assert_eq!(
        builds, 1,
        "the TIME-only fixture does not reach the event-driven walk, so caching \
         it would be a no-op and this test would be vacuous"
    );
}

/// `TIME` in `[individual_parameters]`, no time-varying covariate anywhere.
fn time_only_model() -> CompiledModel {
    parse_model_string(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  theta TVDRIFT(0.01, 0.0001, 1.0)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP ~ 0.04

[individual_parameters]
  FFM   = 9270 * WT / (6680 + 216 * WT / (HT / 100)^2)
  DRIFT = exp(-TVDRIFT * TIME)
  CL    = TVCL * (FFM / 55)^0.75 * DRIFT * exp(ETA_CL)
  V     = TVV * (FFM / 55) * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP)
",
    )
    .expect("TIME-only model parses")
}

/// `tv_population` with every per-event covariate map removed, so each subject
/// carries a single baseline snapshot.
fn baseline_cov_population() -> Population {
    let mut pop = tv_population(false);
    for s in &mut pop.subjects {
        s.dose_covariates.clear();
        s.obs_covariates.clear();
    }
    pop
}

// ─── The prep must be built inside the drawn mixture class ───────────────

/// A 2-class mixture whose **residual-magnitude** expression reads `MIXNUM`,
/// with the mixing logit fixed so far negative that every subject is drawn into
/// class 2. `magnitude` is spliced in so the twin below differs in exactly one
/// expression.
fn mixnum_magnitude_model(magnitude: &str) -> CompiledModel {
    parse_model_string(&format!(
        r"
[parameters]
  theta TVCL(3.0, 0.5, 30.0)
  theta TVV(20.0, 2.0, 200.0)
  theta MIXL(-20.0, FIX)
  omega ETA_CL ~ 0.09
  omega ETA_V ~ 0.05
  sigma PROP ~ 0.04

[mixture]
  nsub = 2
  logit(1) = MIXL

[individual_parameters]
  FFM = 9270 * WT / (6680 + 216 * WT / (HT / 100)^2)
  CL  = TVCL * (FFM / 55)^0.75 * exp(ETA_CL)
  V   = TVV * (FFM / 55) * exp(ETA_V)

[structural_model]
  pk one_cpt_iv(cl=CL, v=V)

[error_model]
  DV ~ proportional({magnitude})
"
    ))
    .unwrap_or_else(|e| panic!("mixture/MIXNUM-magnitude model must parse: {e}"))
}

/// `IndividualNllPrep` hoists `model.ruv_obs_mult(..)` and
/// `error_spec.obs_keys(..)` out of the MH proposal loop. Both may read
/// `MIXNUM` — `validate_ruv_expr` rejects η and NN outputs, not the class index
/// — so the hoist is only correct if the caller refreshes the prep **inside**
/// the drawn class's `MixtureClassGuard`. The wrapper it replaced computed them
/// inside the loop and therefore inside the guard; PR #1452 as first posted
/// refreshed one line *above* `MixtureClassGuard::enter`, which served class
/// 1's residual variance to every class-2 subject's acceptance ratio.
///
/// Three things are checked, in the order that makes the last one mean
/// something:
///
/// 1. the fixture's magnitude really does depend on `MIXNUM` (1.0 under class
///    1, 1.5 under class 2) — otherwise nothing below can observe the class it
///    was evaluated at;
/// 2. the seam itself: refreshing under class 2 and consuming under class 2
///    reproduces `individual_nll_into_with_schedule` (which computes the
///    multiplier itself) bit for bit, and refreshing under class 1 does **not**
///    — so the ordering is load-bearing rather than incidental;
/// 3. a whole mixture SAEM fit runs clean. It is the consumption-point
///    assertion inside `individual_nll_prepared` that makes this an assertion at
///    all: every `individual_nll_prepared` call in the E-step compares the class
///    the prep was refreshed at against the class in effect, so a fit that gets
///    the order wrong panics on its first proposal.
///
/// Mutation check (run, not reasoned): moving `mh_scratch.begin_subject(..)`
/// back above `let _class_guard = ..` in `run_saem`'s E-step — the code as
/// posted — reddens part 3 here and, because the check lives at the consumption
/// point rather than in this test, the pre-existing mixture SAEM tests as well.
#[test]
fn e_step_prep_is_built_inside_the_drawn_mixture_class() {
    use crate::parser::model_parser::MixtureClassGuard;
    use crate::stats::likelihood::{
        individual_nll_into_with_schedule, individual_nll_prepared, IndividualNllPrep,
        IndividualNllScratch,
    };

    // `1 + 0.5·(MIXNUM−1)`: 1.0 in class 1, 1.5 in class 2.
    let model = mixnum_magnitude_model("PROP * (1.0 + 0.5 * (MIXNUM - 1))");
    let pop = tv_population(false);
    let subject = &pop.subjects[0];
    let p = &model.default_params;
    let eta = vec![0.12, -0.08];

    // (1) Non-degeneracy: the multiplier must actually move with the class.
    let m1 = {
        let _g = MixtureClassGuard::enter(1);
        model.ruv_obs_mult(subject, &p.theta)
    };
    let m2 = {
        let _g = MixtureClassGuard::enter(2);
        model.ruv_obs_mult(subject, &p.theta)
    };
    assert_ne!(
        m1, m2,
        "the fixture's residual magnitude does not depend on MIXNUM, so nothing          below can observe which class it was evaluated at"
    );

    // (2) The seam. Oracle is the wrapper, which reads the class itself.
    let want = {
        let _g = MixtureClassGuard::enter(2);
        individual_nll_into_with_schedule(
            &model,
            subject,
            &p.theta,
            &eta,
            &p.omega,
            &p.sigma.values,
            &model.residual_correlations,
            &mut crate::pk::EventPkParams::default(),
            None,
        )
    };
    let prepared_under = |refresh_class: usize| -> f64 {
        let mut prep = IndividualNllPrep::default();
        {
            let _g = MixtureClassGuard::enter(refresh_class);
            prep.refresh(&model, subject, &p.theta);
        }
        let _g = MixtureClassGuard::enter(2);
        // The consumption-point check is what this test is ultimately about, so
        // it must not fire here for the deliberately-wrong arm.
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            individual_nll_prepared(
                &model,
                subject,
                &p.theta,
                &eta,
                &p.omega,
                &p.sigma.values,
                &prep,
                None,
                &mut IndividualNllScratch::default(),
            )
        }))
        .unwrap_or(f64::NAN)
    };
    assert_eq!(
        prepared_under(2).to_bits(),
        want.to_bits(),
        "refreshed and consumed under class 2, the prepared form must reproduce          the wrapper exactly"
    );
    let wrong = prepared_under(1);
    assert!(
        wrong.is_nan() || wrong.to_bits() != want.to_bits(),
        "refreshing under class 1 and consuming under class 2 gave the same          answer ({wrong}), so the ordering is not observable and this test          cannot fail"
    );

    // (3) A whole mixture SAEM fit, policed by the consumption-point assertion.
    let r = run_saem(&model, &pop, &model.default_params, &saem_opts())
        .expect("mixture SAEM run must succeed");
    assert!(r.ofv.is_finite(), "mixture SAEM produced a non-finite OFV");
}
