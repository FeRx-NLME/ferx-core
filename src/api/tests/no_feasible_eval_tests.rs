//! Tier-1 tests for #1617: a fit whose **every** outer trial was rejected may not
//! report `converged = true` at the midpoint of its declared bounds.
//!
//! **The regression.** Reported on warfarin (ferx-r 0.4.0, ferx-core `2a6076af`): starve
//! the inner loop and every θ comes back at `sqrt(lower·upper)` to 3e-9, every ω at 1 —
//! the middle of the ±6 log rail — with `converged = true` and an OFV 500 units from the
//! optimum. Re-measured here on the issue's second reproduction, the two-subject /
//! three-observation fixture below (ported from ferx-r `test-theta-levels.R` `tl_data`), at
//! `afe7f52b` with this fix toggled off:
//!
//! ```text
//! converged = true
//! theta     = [0.14142136177403178, 7.0710678480822855]   // sqrt(0.001·20), sqrt(0.1·500)
//! ofv       = 24.578051646640162                          // a real −2LL, at the wrong point
//! ```
//!
//! against the same model and data at the default inner budget: `converged = true`,
//! `theta = [1.0369668486435517, 11.982743248520865]`, `ofv = −10.403143919119280`. Three
//! distinct points — midpoint, optimum, initial estimates `[2.0, 10.0]` — so an assertion
//! on any one of them cannot be satisfied by the other two.
//!
//! **Why every existing gate reads a plausible number.** A guard-rejected trial is
//! scored by `guard_penalty_value` — `1e12 + 50·Σ(xs − c)²`, a quadratic bowl whose
//! minimum sits at the scaled bound midpoint `c` — so L-BFGS, handed nothing else,
//! minimises the *penalty*, walks to the box centre, and NLopt returns `Success`
//! because the penalty genuinely is stationary there. The best-seen tracker then
//! ranks penalties only, so the #59 restore lands on that centre, and the final inner
//! loop re-scores it into an ordinary finite OFV. `gate_converged_on_objective` sees a
//! finite objective; `cold_solve_verdict` compares the 24.6 re-score against `1e12`
//! and calls it a reproduction, so `W_EBE_START_DEPENDENT` cannot fire; and the
//! plateau check never runs, because the status was `Success` and not a bare
//! `Failure`. Nothing anywhere reported that no feasible evaluation took place.
//!
//! **What the fix is.** `no_feasible_eval_warning` reads the feasible-eval count the
//! plateau tracker already keeps, demotes `converged`, and names the guard, the share
//! of unconverged subjects and the remedy; the restore falls back to the point the
//! optimizer started from instead of the penalty's minimum; the #1277 restart and the
//! "increase `maxiter`" advice are both withheld, because neither has anything to act on.
//!
//! **Mutation sweep.** The first six edits were each applied and run (`--lib`, this
//! filter); none of them reddens E2, the feasible control:
//!
//! | Edit | Test that dies |
//! |---|---|
//! | both halves of the fix off | E1 — assertion 2, `converged = true` at θ = `[0.14142136, 7.07106785]` |
//! | drop the `no_feasible_eval_warning` call only | E1 — assertion 2, at θ = `[2.0, 10.0]` |
//! | drop the `optimizer_start_s` fallback only | E1 — assertion 3, θ[0] = `0.14142136177403178` |
//! | restore the θ coordinates only, leaving ω/σ on the best-seen point | E1 — assertion 4, ω = `0.9999999524299102`, the middle of its ±6 log rail |
//! | `no_feasible()` broadened so it claims a feasible run | U1 **and** E2 |
//! | call site passes `cancelled: false` | E3 |
//! | `budget_advice_applies = true` | E4 — "increase maxiter" reappears |
//! | guard arm claims a trace with `n_guard_rejected == 0` | U4 **and** E5 |
//! | `classify_warning`'s token arm off | U7 **and** E1 assertion 5 — `Warning`/`general` |
//! | the non-finite arm's **remedy** sentence deleted whole | U4 |
//! | the mixed trace's second **remedy** dropped | U5 |
//!
//! The last row is the #1405 shape — a diagnostic's advice half that no test can kill is a
//! claim nobody has checked — so every *other* sentence of the warning is covered the same
//! way rather than by control-flow mutation alone: each of U3/U4/U5 asserts a verbatim
//! substring of each sentence it owns (head and both clauses of the guard arm in U3, the
//! non-finite cause and remedy in U4, the mixed-trace clause and its second remedy in
//! U5), so deleting or
//! rewording a sentence necessarily fails a named assertion. The arm that deliberately
//! *withholds* the other's advice asserts its own remedy text, not merely the absence of
//! the other's. E5 is the end-to-end twin of U4: the two can break independently, and the
//! arm-order mutation reddens both.
use super::*;
use crate::api::nonfinite_objective_tests::{subject_with_dose_time, two_state_model};
use crate::estimation::outer_optimizer::{
    no_feasible_eval_warning, RunFeasibility, NO_FEASIBLE_EVAL_TOKEN,
};
use crate::parser::model_parser::parse_model_string;
use crate::types::{Optimizer, WarningCode, WarningSeverity};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
//  The fixture — the issue's second reproduction, ported from ferx-r
//  `tests/testthat/test-theta-levels.R` `tl_data`
// ─────────────────────────────────────────────────────────────────────────────

/// Two subjects, three observations each, one-compartment IV closed form. `CL` has
/// no random effect and `V` carries the only η, which is what makes three inner
/// iterations enough to starve it.
///
/// The bounds are load-bearing: `TVCL(2.0, 0.001, 20.0)` and `TVV(10.0, 0.1, 500.0)`
/// put the geometric midpoints at `sqrt(0.001·20) = 0.141421` and
/// `sqrt(0.1·500) = 7.071068`, both far from the initial estimates *and* far from the
/// optimum (`TVCL ≈ 1.037`, `TVV ≈ 11.983`). All three points are distinct, so an
/// assertion on any one of them cannot be satisfied by the other two.
fn starvable_model() -> CompiledModel {
    parse_model_string(
        r#"
[parameters]
  theta TVCL(2.0, 0.001, 20.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_V ~ 0.04
  sigma PROP_ERR ~ 0.05
[individual_parameters]
  CL = TVCL
  V  = TVV * exp(ETA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#,
    )
    .expect("parse")
}

fn subject(id: &str, observations: Vec<f64>) -> Subject {
    let obs_times = vec![1.0, 4.0, 12.0];
    Subject {
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
        obs_times,
        obs_raw_times: Vec::new(),
        observations,
        obs_cmts: vec![1, 1, 1],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 3],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

fn two_subjects() -> Population {
    Population {
        subjects: vec![
            subject("1", vec![8.1, 6.2, 3.1]),
            subject("2", vec![7.4, 5.5, 2.8]),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// `inner_maxiter` is the only thing that differs between the two arms below.
/// `parse_model_string` drops a `[fit_options]` block, so every option the fit
/// depends on is set here rather than in the model text.
fn opts(inner_maxiter: usize) -> FitOptions {
    FitOptions {
        method: EstimationMethod::FoceI,
        outer_maxiter: 200,
        inner_maxiter,
        run_covariance_step: false,
        threads: Some(2),
        ..Default::default()
    }
}

/// The geometric midpoint of each θ's declared bounds — where the guard penalty's
/// minimum sits, and what the defect reported as the estimate.
const BOUND_MIDPOINT: [f64; 2] = [0.14142135623730953, 7.0710678118654755];

/// ω at the middle of its packed rail, which is the ω half of the same signature. Omega is
/// Cholesky-packed in log space on a ±6 rail, so the box centre is `log L = 0`, i.e.
/// `ω = 1` — the value the issue reports for all three of warfarin's diagonal elements.
/// Declared here is `0.04`, so the two are 25× apart and an assertion on one cannot be
/// satisfied by the other.
const OMEGA_MIDPOINT: f64 = 1.0;

// ─────────────────────────────────────────────────────────────────────────────
//  E1/E2 — the differential pair, end to end through `fit()`
// ─────────────────────────────────────────────────────────────────────────────

/// **E1 — the regression.** With the inner loop starved, the fit must not report
/// convergence, must not report the bound midpoint as its estimate, and must say
/// which guard stopped it.
///
/// Six assertions, each separately load-bearing:
///
/// 1. **the trial trace is still infeasible** — non-degeneracy. The defect needs
///    *every* trial guard-rejected; if a later change lets one through, this fixture
///    stops delivering the defect's input and the rest of the test passes for an
///    unrelated reason. Asserted on `max_unconverged_subjects`, which the run
///    publishes, rather than inferred from the verdict.
/// 2. **`converged` is `false`** — the reported bug.
/// 3. **the estimates are not the bound midpoint** — the *point*, which the issue
///    names as the defect ("the reported OFV is a real objective at the midpoint …
///    the defect is the point, not the score"). Demoting the flag alone would leave
///    a fit whose θ is a number the data never chose.
/// 4. **they are the declared initial estimates, in every packed family** — the positive
///    half of (3), which on its own would pass at any third wrong value, and across θ, ω
///    and σ rather than θ alone, since the issue's signature covers ω too (each at the
///    middle of its ±6 log rail, i.e. `1`). Exact by construction (`2.0`, `10.0`, `0.04`,
///    `sqrt(0.05)` — σ reads back as an SD), so the bound is a round-trip tolerance and not
///    a fit tolerance: the reported
///    values come back through `pack_with_bounds` → `clamp_to_bounds` → `unpack_params`,
///    which costs `10.0` one ULP.
/// 5. **a warning names the guard, the share and the remedy**, carries
///    `W_NO_FEASIBLE_EVAL` and classifies `Critical`/`convergence`.
/// 6. **the warning set for this row** — the generic `Outer optimization did not converge`
///    line is there too, and `W_NONFINITE_OBJECTIVE` is not. That is the per-row claim the
///    docs make; E5 asserts the other row, where the third warning does fire.
///
/// Mutation (run): drop the `no_feasible_eval_warning` call at the NLopt exit →
/// assertion 2 fires (3 and 4 pass, since the estimates *are* the inits). Drop only
/// the `optimizer_start_s` fallback in the best-seen restore → assertion 3 fires
/// first, and 4 with it. Neither mutation reddens E2.
#[test]
fn a_fit_with_no_feasible_evaluation_is_not_reported_converged() {
    let model = starvable_model();
    let pop = two_subjects();
    let result = fit(&model, &pop, &model.default_params, &opts(3)).expect("fit");

    // (1) The fixture still produces the defect's input: the guard really did reject
    //     on unconverged subjects.
    assert!(
        result.max_unconverged_subjects > 0,
        "#1617: this fixture exists to starve the inner loop so that every outer trial is \
         EBE-guard-rejected; it reported 0 unconverged subjects, so the gate below is being \
         tested against the wrong input. Lower `inner_maxiter` or restore the fixture."
    );
    // (2) The regression.
    assert!(
        !result.converged,
        "#1617: fit() reported converged = true with no feasible evaluation (theta = {:?}, \
         ofv = {}). Nothing the optimizer ranked was the objective.",
        result.theta, result.ofv,
    );
    // (3) ...and the estimates are not the guard penalty's minimum.
    for (i, mid) in BOUND_MIDPOINT.iter().enumerate() {
        assert!(
            (result.theta[i] - mid).abs() > 1e-3 * mid,
            "#1617: theta[{i}] = {} is the geometric midpoint of its declared bounds ({mid}) \
             — the minimum of `guard_penalty_value`, not an estimate. The restore must fall \
             back to the point the optimizer started from.",
            result.theta[i],
        );
    }
    // (4) ...they are the values the model declared — in **every** packed family, not just
    //     θ (#1617 review round 4). The restore copies the whole scaled vector, and the
    //     issue reports ω at the midpoint of its own rail alongside the θ midpoints; a
    //     partial restore over the θ coordinates alone would leave ω and σ at the penalty's
    //     minimum with the θ assertions above still green. One ULP of round-trip, not a fit
    //     tolerance: `10.0` comes back as `10.000000000000002` through the packer.
    for (i, init) in [2.0_f64, 10.0].iter().enumerate() {
        assert!(
            (result.theta[i] - init).abs() <= 1e-12 * init,
            "#1617: with nothing feasible to rank, the reported estimates must be the \
             declared initial values; theta[{i}] = {} where {init} was declared.",
            result.theta[i],
        );
    }
    assert!(
        (result.omega[(0, 0)] - 0.04).abs() <= 1e-12 * 0.04,
        "#1617: omega must come back at its declared 0.04, not the middle of its packed \
         rail ({OMEGA_MIDPOINT}); got {}",
        result.omega[(0, 0)],
    );
    assert!(
        (result.omega[(0, 0)] - OMEGA_MIDPOINT).abs() > 0.5,
        "#1617: omega is at the centre of its ±6 log rail — the ω half of the midpoint \
         signature the issue reports; got {}",
        result.omega[(0, 0)],
    );
    // σ is reported on the **standard-deviation** scale: a bare `sigma NAME ~ X` declares a
    // variance and the parser converts it via `sqrt` (#56), so the declared `0.05` reads
    // back as `sqrt(0.05) = 0.223606797749979`. Measured, not assumed — the first version of
    // this assertion compared against `0.05` and failed on exactly that factor.
    let sigma_declared = 0.05_f64.sqrt();
    assert!(
        (result.sigma[0] - sigma_declared).abs() <= 1e-12 * sigma_declared,
        "#1617: sigma must come back at its declared 0.05 (= {sigma_declared} as an SD); \
         got {}",
        result.sigma[0],
    );
    // (5) ...and it says why.
    let entry = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains(NO_FEASIBLE_EVAL_TOKEN))
        .unwrap_or_else(|| {
            panic!(
                "the demotion must carry a typed warning naming the guard; got {:?}",
                result
                    .warnings_structured
                    .iter()
                    .map(|w| &w.message)
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(entry.severity, WarningSeverity::Critical, "{entry:?}");
    assert_eq!(entry.category, WarningCode::Convergence, "{entry:?}");
    assert!(
        entry.message.contains("max_unconverged_frac") && entry.message.contains("inner_maxiter"),
        "the message must name both knobs the user can act on: {}",
        entry.message
    );
    // (6) The warning *set* for this row, which is what `docs/estimation/foce.qmd` claims
    //     and what round 3 of review corrected: the generic non-convergence line always
    //     accompanies a demoted run, and `W_NONFINITE_OBJECTIVE` does **not** fire here —
    //     the re-score at the starting estimates is a real objective (76.85), not a
    //     sentinel. E5 is the same assertion on the other row, where it does fire.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("Outer optimization did not converge")),
        "a demoted run carries the generic line too: {:?}",
        result.warnings
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("W_NONFINITE_OBJECTIVE")),
        "the objective at the reported point is finite on this row ({}), so its own gate \
         must stay quiet: {:?}",
        result.ofv,
        result.warnings
    );
}

/// **E2 — the feasible control, bounded** (#1617 review round 5).
///
/// The other half of the differential pair, and the half that makes E1 mean something:
/// `no_feasible_eval_warning` keyed on anything coarser than the feasible-eval count — "the
/// guard fired at all", say, which it does transiently on plenty of healthy fits — would
/// demote this run too.
///
/// **Bounded on purpose.** The first version ran this to convergence at
/// `outer_maxiter = 200` and asserted `converged`, which is a Tier-3 shape sitting in the
/// always-run `--lib` suite: AGENTS.md reserves convergence fits for the `slow-tests` gate,
/// and on a 2-subject fit whose ω is informed by two subjects the convergence *verdict* is
/// the most trajectory-sensitive thing in the result — the kind of assertion that goes red
/// for a reason unrelated to the guard logic it is meant to protect. `outer_maxiter = 4`
/// gives it `4·(n+1) = 20` evaluations and it comes back `MaxEvalReached`, which is all this
/// test needs. `a_healthy_fit_still_converges_to_its_optimum` keeps the convergence
/// assertion, gated.
///
/// One bound covers both ways the fix could damage a feasible run, which is why it is a
/// bound on the objective and not on θ: the midpoint the defect reported scores **24.578**
/// and the initial estimates score **76.847**, so an objective below 24 proves the run
/// neither reported the penalty's minimum nor got reverted to its start. Measured here:
/// −8.876 at evaluation 20 (macOS arm64), ~33 units of headroom, so no platform's
/// trajectory can reach the bound.
#[test]
fn a_feasible_fit_is_untouched() {
    let model = starvable_model();
    let pop = two_subjects();
    let options = FitOptions {
        outer_maxiter: 4,
        ..opts(200)
    };
    let result = fit(&model, &pop, &model.default_params, &options).expect("fit");

    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains(NO_FEASIBLE_EVAL_TOKEN)),
        "a fit with feasible evaluations must not carry the no-feasible-evaluation warning: \
         {:?}",
        result.warnings
    );
    assert!(
        result.ofv < 24.0,
        "a feasible run must report an objective it actually descended to — below the \
         24.578 the midpoint scores and the 76.847 the initial estimates score, either of \
         which would mean the fix damaged a run it has no business touching; got {}",
        result.ofv
    );
    assert_eq!(
        result.n_iterations, 20,
        "this test is the *bounded* control: it must stop on its 4·(n+1) = 20-evaluation \
         budget rather than run to convergence, which is `a_healthy_fit_still_converges_to_\
         its_optimum`'s job behind the `slow-tests` gate. It made {} evaluations.",
        result.n_iterations
    );
    for (i, mid) in BOUND_MIDPOINT.iter().enumerate() {
        assert!(
            (result.theta[i] - mid).abs() > 1e-3 * mid,
            "a feasible run's estimates must be distinct from the bound midpoint, so E1's \
             assertion (3) discriminates; theta[{i}] = {} vs midpoint {mid}",
            result.theta[i],
        );
    }
}

/// **E2b — Tier 3: a healthy fit still reaches its optimum and says so.**
///
/// The convergence half of E2, in the tier AGENTS.md puts convergence fits in. It is
/// cheap (the same 2-subject closed form, ~36 evaluations) but it *is* a fit to
/// convergence and it asserts the verdict, so it runs nightly via `slow-tests` and on
/// pushes to `main` that touch estimation code rather than on every PR. E2 carries the
/// per-PR half.
///
/// The optimum, measured on this branch at `8b8caf3c` (macOS arm64 — the numbers below are
/// bounds, not digit anchors, per AGENTS.md's reference-platform rule):
/// `ofv = −10.403143919119280`, `theta = [1.0369668486435517, 11.982743248520865]`,
/// `omega = [[0.002407034368112816]]`. That is 35 OFV units below the 24.578 the defect
/// reported — the whole reason #1617 is a defect and not a cosmetic flag.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: opt in with --features slow-tests"
)]
fn a_healthy_fit_still_converges_to_its_optimum() {
    let model = starvable_model();
    let pop = two_subjects();
    let result = fit(&model, &pop, &model.default_params, &opts(200)).expect("fit");

    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains(NO_FEASIBLE_EVAL_TOKEN)),
        "{:?}",
        result.warnings
    );
    assert!(
        result.converged,
        "a fit that reaches its optimum must still report it (ofv = {}, theta = {:?})",
        result.ofv, result.theta,
    );
    assert!(
        result.ofv < 0.0,
        "the converged objective is the −10.4 optimum, not the 24.6 the midpoint scores; \
         got {}",
        result.ofv
    );
}

/// **E3 — a cancelled run is not reported as a defect, at the call site.**
///
/// Not a second copy of U2. U2 pins the *predicate*: handed `cancelled: true` the
/// helper returns `None`. This pins the *wiring* — that the flag reaching the helper is
/// the run's own cancellation state — and the two are independently breakable (the
/// struct literal could pass `false`, and nothing in U2 would notice).
///
/// It also has to go through `optimize_population` rather than `fit()`: `fit()` returns
/// `Err("cancelled by user")` after its stage loop, so a cancelled `OuterResult` only
/// reaches a caller that drives the optimizer directly, which `ferx-tools` does.
///
/// A cancelled run's evaluations all return `1e20` from the top of the objective,
/// *before* the feasible counter advances, so its trace is infeasible by construction —
/// which is the whole reason the guard exists. Without it the run would carry a warning
/// saying every trial was rejected by the EBE guard, which is false.
#[test]
fn a_cancelled_run_carries_no_no_feasible_eval_warning() {
    let model = starvable_model();
    let pop = two_subjects();
    let flag = crate::cancel::CancelFlag::new();
    flag.cancel();
    let options = FitOptions {
        cancel: Some(flag),
        ..opts(200)
    };
    let result = crate::estimation::outer_optimizer::optimize_population(
        &model,
        &pop,
        &model.default_params,
        &options,
    );
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains(NO_FEASIBLE_EVAL_TOKEN)),
        "cancellation is not an EBE-guard finding: {:?}",
        result.warnings
    );
}

/// **E4 — the budget advice is withheld** (#1617 review round 1 #2).
///
/// "Outer optimization hit the evaluation budget (maxiter = N) before converging; increase
/// maxiter for a tighter fit." offers to tighten a fit that does not exist, and a longer
/// *outer* budget buys only more rejected trials: the budget that actually binds is
/// `inner_maxiter`, which `W_NO_FEASIBLE_EVAL` names. Two warnings pointing at different
/// knobs, one of them useless, is how a reader is sent the wrong way.
///
/// Staging `MaxEvalReached` **and** an infeasible trace together took a fixture search,
/// and what did not work is worth recording, because each near-miss looks like it should:
///
/// * `optimizer = bobyqa` never triggers the guard on this model — its opening
///   `rhobeg = 0.5` probes stay near the initial estimates, where the inner loop converges
///   inside its budget. (That is also why the issue's own BOBYQA row reports a real OFV at
///   the initial values rather than a midpoint.)
/// * `nlopt_lbfgs` at `outer_maxiter = 1` is infeasible throughout but stops on NLopt's
///   own criterion: the guard penalty is an exact sphere in scaled space and L-BFGS solves
///   it in 4 evaluations, one inside the `1·(n+1) = 5` budget. Measured by mutation — with
///   the gate removed the budget warning still did not appear, so a fixture that merely
///   *spends* its budget is not enough; the status has to be `MaxEvalReached`.
/// * `slsqp` escapes into feasible territory on the same model (`n_iterations = 5`, no
///   `W_NO_FEASIBLE_EVAL`), so it earns the budget warning legitimately.
///
/// `mma` is the combination: capped like SLSQP, so it crawls rather than solving the
/// sphere, and still infeasible at every trial.
///
/// Mutation (run): drop `&& !feasibility.no_feasible()` from the `max_eval_reached` arm →
/// "increase maxiter for a tighter fit" reappears and assertion (3) fires.
#[test]
fn a_run_with_no_feasible_evaluation_is_not_told_to_raise_maxiter() {
    let model = starvable_model();
    let pop = two_subjects();
    let options = FitOptions {
        optimizer: Optimizer::Mma,
        outer_maxiter: 1,
        ..opts(3)
    };
    let result = fit(&model, &pop, &model.default_params, &options).expect("fit");

    // (1) The fixture still delivers an infeasible trace...
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains(NO_FEASIBLE_EVAL_TOKEN)),
        "the fixture must still produce a run with no feasible evaluation: {:?}",
        result.warnings
    );
    // (2) ...and one that spent its whole outer budget, which is what makes NLopt return
    //     `MaxEvalReached` and the warning below reachable at all. `outer_eval_budget` is
    //     `outer_maxiter · (n + 1)` = 5 here. There is no field carrying NLopt's status,
    //     so the evaluation count is the nearest observable — and it is not sufficient on
    //     its own (see the L-BFGS near-miss above), which is why the mutation is what
    //     establishes that assertion (3) can fail.
    assert_eq!(
        result.n_iterations, 5,
        "this test needs a run that exhausted its 5-evaluation outer budget; it made {}. \
         The `max_eval_reached` arm that assertion (3) guards is then never taken and the \
         assertion is vacuous.",
        result.n_iterations
    );
    // (3) ...so the generic budget advice must be withheld.
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("increase maxiter")),
        "a run that never scored the objective has no fit to tighten, so the \
         \"increase maxiter\" advice must not be offered next to W_NO_FEASIBLE_EVAL: {:?}",
        result.warnings
    );
}

/// **E5 — the non-finite arm, end to end** (#1617 review round 2 #3).
///
/// U4 pins the *message* for a trace of non-finite objectives, but on a synthetic
/// `RunFeasibility`. This pins the *wiring*: that production really does classify such a
/// run into that arm rather than the guard's. The two are independently breakable — the
/// call site could miscount `n_guard_rejected`, or the guard could claim the run — and
/// nothing in U4 would notice.
///
/// The fixture is #1296's / #1303's, reused unchanged: a two-state ODE population one of
/// whose subjects carries a `NaN` dose time, so `timeline_has_non_finite` `NaN`-fills it
/// and the objective is `NaN` for the whole population. That lands in this arm rather than
/// the guard's for a reason worth stating, because it is the arm's own entry condition:
/// `ebe_guard_rejects` needs `raw_ofv.is_finite()` for its fraction trigger and
/// `is_ode_iov(model)` for its unconditional one, and this model is ODE but has no IOV —
/// so the guard never fires, `n_guard_rejected` stays 0, and `guarded` is true purely
/// because the objective is not a number.
///
/// This is also the end-to-end check for the claim
/// [`docs/estimation/foce.qmd`](../../docs/estimation/foce.qmd) makes about this row and
/// round 1 of review corrected: **both** warnings appear, because they are two different
/// facts — `W_NO_FEASIBLE_EVAL` about the trace, `W_NONFINITE_OBJECTIVE` about the number
/// the run finally reported.
///
/// Mutation (run): drop the `no_feasible_eval_warning` call → assertion (2) fires. Swap
/// the arm order in `no_feasible_eval_warning` so the guard arm claims a trace with
/// `n_guard_rejected == 0` → assertion (3) fires.
#[test]
fn an_all_non_finite_run_reports_the_non_finite_arm_end_to_end() {
    let model = two_state_model();
    let pop = Population {
        subjects: vec![
            subject_with_dose_time("bad", f64::NAN, 1.0),
            subject_with_dose_time("2", 0.0, 1.2),
        ],
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    };
    let result = fit(&model, &pop, &model.default_params, &opts(200)).expect("fit");

    // (1) The fixture still delivers this test's input.
    assert!(
        result.ofv.is_nan(),
        "this fixture exists to make every objective non-finite; it reported {}",
        result.ofv
    );
    // (2) The trace is reported as infeasible...
    let entry = result
        .warnings_structured
        .iter()
        .find(|w| w.message.contains(NO_FEASIBLE_EVAL_TOKEN))
        .unwrap_or_else(|| {
            panic!(
                "a run whose every objective was non-finite has no feasible evaluation \
                 either, and must say so; got {:?}",
                result
                    .warnings_structured
                    .iter()
                    .map(|w| &w.message)
                    .collect::<Vec<_>>()
            )
        });
    // (3) ...through the non-finite arm, not the guard's: the guard never fired here, so
    //     naming it would be false, and `inner_maxiter` is not the budget that binds.
    assert!(
        entry
            .message
            .contains("every trial returned a non-finite objective"),
        "{}",
        entry.message
    );
    assert!(
        !entry.message.contains("Raise `inner_maxiter`"),
        "a non-finite objective is not an inner-budget problem: {}",
        entry.message
    );
    // (4) ...and the objective gate speaks for itself. Two facts, two warnings — the
    //     claim the docs make about this row.
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("W_NONFINITE_OBJECTIVE")),
        "the reported objective is `NaN` here, so its own gate must fire too: {:?}",
        result.warnings
    );
    assert!(!result.converged);
}

// ─────────────────────────────────────────────────────────────────────────────
//  U1..U7 — `no_feasible_eval_warning`, every cell of its input space
//
//  The message's input space is (feasible_evals = 0 or > 0) × (cancelled) ×
//  (cause: start rejection / fraction / non-finite) × (guard count = n_evals or
//  fewer). Enumerated here rather than discovered one PR review at a time.
// ─────────────────────────────────────────────────────────────────────────────

/// A trace with nothing feasible, rejected on the fraction trigger.
fn fraction_run() -> RunFeasibility {
    RunFeasibility {
        feasible_evals: 0,
        n_guard_rejected: 7,
        n_nonfinite: 0,
        max_unconverged: 9,
        n_subj: 10,
        max_unconverged_frac: 0.1,
        inner_maxiter: 3,
        cancelled: false,
    }
}

/// **U1 — a run with any feasible evaluation is left alone**, verdict and all.
#[test]
fn a_run_with_one_feasible_evaluation_is_not_demoted() {
    let mut converged = true;
    let run = RunFeasibility {
        feasible_evals: 1,
        ..fraction_run()
    };
    assert_eq!(no_feasible_eval_warning(&mut converged, &run), None);
    assert!(
        converged,
        "one feasible evaluation is enough: the optimizer did score the real objective"
    );
}

/// **U2 — a cancelled run is not a defect.** Its evaluations short-circuit at the top
/// of the objective, before the feasible counter advances, so its trace is infeasible
/// by construction and the verdict must not move.
#[test]
fn a_cancelled_run_is_not_reported_as_infeasible() {
    let mut converged = true;
    let run = RunFeasibility {
        cancelled: true,
        ..fraction_run()
    };
    assert_eq!(no_feasible_eval_warning(&mut converged, &run), None);
    assert!(converged, "cancellation is not a convergence finding");
}

/// **U3 — the fraction arm.** Demotes, names the guard, quotes the share against
/// `max_unconverged_frac`, and offers both knobs that actually move that trigger.
#[test]
fn the_fraction_trigger_names_the_share_and_both_knobs() {
    let mut converged = true;
    let msg = no_feasible_eval_warning(&mut converged, &fraction_run()).expect("a warning");
    assert!(!converged, "the verdict must be demoted");
    assert!(msg.starts_with(NO_FEASIBLE_EVAL_TOKEN), "{msg}");
    // The head sentence: what happened, how many trials, and where the estimates came
    // from. Deleting any clause of it reddens this assertion.
    assert!(
        msg.contains("without a single feasible objective evaluation"),
        "{msg}"
    );
    assert!(msg.contains("all 7 of its trials"), "{msg}");
    assert!(
        msg.contains("the estimates reported are the ones the fit started from"),
        "{msg}"
    );
    // The cause sentence.
    assert!(
        msg.contains("the EBE convergence guard rejected 7 of them"),
        "{msg}"
    );
    assert!(msg.contains("at worst 9 of 10 subjects"), "{msg}");
    // "against", not "above". `ebe_guard_rejects` has a second, unconditional trigger this
    // message does not branch on (see its doc comment), so the sentence must not assert
    // that the fraction threshold is what was exceeded — it reports the share and the
    // threshold and lets them be read together.
    assert!(
        msg.contains("against `max_unconverged_frac` = 0.1"),
        "{msg}"
    );
    assert!(!msg.contains("above `max_unconverged_frac`"), "{msg}");
    // The remedy sentence.
    assert!(msg.contains("Raise `inner_maxiter` (currently 3)"), "{msg}");
    assert!(msg.contains("`1.0` disables the guard"), "{msg}");
    // No mixed-cause clause, and no second remedy, when every trial was the guard's.
    assert!(!msg.contains("The remaining"), "{msg}");
    assert!(!msg.contains("For those"), "{msg}");
}

/// **U4 — the non-finite arm.** When the guard never fired, every trial was infeasible
/// because its objective was not a number. Naming the guard there would be false, and
/// the remedy is the data/model one `W_NONFINITE_OBJECTIVE` gives, not an optimizer
/// setting.
#[test]
fn an_all_non_finite_trace_does_not_blame_the_guard() {
    let mut converged = true;
    let run = RunFeasibility {
        n_guard_rejected: 0,
        n_nonfinite: 7,
        max_unconverged: 0,
        ..fraction_run()
    };
    let msg = no_feasible_eval_warning(&mut converged, &run).expect("a warning");
    assert!(!converged);
    // The cause sentence.
    assert!(
        msg.contains("every trial returned a non-finite objective"),
        "{msg}"
    );
    assert!(msg.contains("never came into it"), "{msg}");
    // The remedy sentence, asserted in its own right — see the note in U4.
    assert!(
        msg.contains("Look for the parameter value or record that poisons the objective"),
        "{msg}"
    );
    assert!(
        msg.contains("at and around the starting estimates"),
        "{msg}"
    );
    assert!(
        !msg.contains("Raise `inner_maxiter`"),
        "a non-finite objective is not an inner-budget problem: {msg}"
    );
}

/// **U5 — a mixed trace reports both halves.** Some trials guard-rejected, the rest
/// non-finite: the counts have to add up to the trial count the head sentence quotes,
/// or a reader subtracts and finds a trial unaccounted for.
#[test]
fn a_mixed_trace_accounts_for_every_trial() {
    let mut converged = true;
    let run = RunFeasibility {
        n_guard_rejected: 5,
        n_nonfinite: 2,
        ..fraction_run()
    };
    let msg = no_feasible_eval_warning(&mut converged, &run).expect("a warning");
    assert!(!converged);
    // The trial count is the sum of the two causes, not a callback count: see
    // `RunFeasibility::n_trials`.
    assert!(msg.contains("all 7 of its trials"), "{msg}");
    assert!(msg.contains("rejected 5 of them"), "{msg}");
    assert!(
        msg.contains("The remaining 2 returned a non-finite objective"),
        "{msg}"
    );
    // ...and *both* remedies, because neither covers the other's trials: `inner_maxiter`
    // does not make a `NaN` finite, and chasing a poisoned record does not widen the inner
    // budget. The guard remedy alone was what round 3 of review caught.
    assert!(msg.contains("Raise `inner_maxiter` (currently 3)"), "{msg}");
    assert!(
        msg.contains("For those, look for the parameter value or record that poisons"),
        "{msg}"
    );
}

/// **U6 — the trial count is the sum of the two causes, never a callback count**
/// (#1617 review round 3 #2).
///
/// The objective closure's callback counter (`n_evals_outer`, which becomes
/// `FitResult::n_iterations`) counts calls, not trials: the latched-stagnation fast path
/// increments it and returns the incumbent objective without producing one. Deriving the
/// non-finite count as "callbacks minus guard rejections" therefore files every post-latch
/// callback under "returned a non-finite objective", which is false.
///
/// `RunFeasibility` no longer carries a callback count at all, so that subtraction is not
/// expressible — which is the fix, and also why no fixture can observe it. Measured, the
/// latent defect is not reachable either: the latch needs 50 evaluations without OFV
/// improvement (the long window, since `last_sig_feasible_eval` is 0 on an infeasible
/// trace), and every optimizer terminates on the guard penalty long before that — BOBYQA
/// stops at 17 evaluations on the flat `1e20` wall, L-BFGS at 4 on the exact quadratic,
/// MMA at 10, over five optimizer/budget combinations. So this pins the contract rather
/// than a reachable failure: reintroducing the bug means reintroducing the field.
#[test]
fn the_trial_count_is_the_sum_of_the_two_causes() {
    let run = RunFeasibility {
        n_guard_rejected: 5,
        n_nonfinite: 2,
        ..fraction_run()
    };
    assert_eq!(run.n_trials(), 7);
    assert_eq!(
        RunFeasibility {
            n_guard_rejected: 0,
            n_nonfinite: 3,
            ..fraction_run()
        }
        .n_trials(),
        3
    );
    let mut converged = true;
    let msg = no_feasible_eval_warning(&mut converged, &run).expect("a warning");
    assert!(msg.contains("all 7 of its trials"), "{msg}");
}

/// **U7 — the token routes to `Critical`/`convergence` on its own.** The message says
/// "did not converge" nowhere, so without the token arm in `classify_warning` it would
/// land in the `general` fallback bucket at `Warning` severity — the opposite of what a
/// consumer branching on `category` needs from a run that produced no estimate.
#[test]
fn the_token_classifies_critical_convergence() {
    let mut converged = true;
    let msg = no_feasible_eval_warning(&mut converged, &fraction_run()).expect("a warning");
    assert!(
        !msg.to_lowercase().contains("did not converge"),
        "if the message ever says this, the prose arm claims it and the token arm stops \
         being load-bearing — which is fine, but then this test is asserting nothing: {msg}"
    );
    let entry = crate::types::classify_warning(&msg);
    assert_eq!(entry.severity, WarningSeverity::Critical, "{entry:?}");
    assert_eq!(entry.category, WarningCode::Convergence, "{entry:?}");
}
