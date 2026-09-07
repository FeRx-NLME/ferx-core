//! NONMEM anchor for a **steady-state dose on a model-time-reading `[odes]` RHS** —
//! issue [#1139](https://github.com/FeRx-NLME/ferx-core/issues/1139), batch-T step T2.
//!
//! ferx: `d/dt(central) = -(CL/V)*central*(1 + 0.03*TAD)`, `obs_scale = V`, one `SS=1`
//! record with `II = 12` at `t = 480`. NONMEM (`nonmem_anchor/ss_tad.ctl`, 7.6.0,
//! `ADVAN13 TOL=9`, `MAXEVAL=0 METHOD=0 POSTHOC`, `$THETA` FIX, `$OMEGA 0 FIX`,
//! `$TABLE FORMAT=,1PE20.13`): the same system with the cycle clock written by hand as
//! `TADX = MOD(T + 120.0, 12.0)`, since **NONMEM has no `TAD` built-in in `$DES`**.
//!
//! Before this change ferx returned `NaN` at every observation and a non-finite objective:
//! the compiled RHS reads `TAD` out of `params[MAX_PK_PARAMS + 1]` and the steady-state
//! run-in handed `solve_ode` a bare `PkParams::values`, exactly `MAX_PK_PARAMS` long, so
//! `.get()` returned `None` and the RHS injected `NaN`. `0.0 * NaN = NaN`, so merely
//! *mentioning* `TAD` broke an otherwise ordinary steady state.
//!
//! # Why this family is anchorable at all, and how the anchor is certified
//!
//! Three independent statements, in the order that makes them trustworthy — none of them
//! is "ferx agrees with ferx":
//!
//! 1. **A periodic steady state exists.** `TAD` is bounded inside one interval, so the
//!    explicit train has a limit: `train_tad_n21.tab` against `train_tad.tab` measures a
//!    3.5e-7 relative change over a doubling. The absolute-clock twin (`train_tabs*`) does
//!    **not** converge — 0.294 per doubling — which is why `T`/`TIME`/`TAFD` under `SS=1`
//!    are a different question and are not anchored here.
//! 2. **NONMEM's own steady-state routine reproduces NONMEM's own train**, to 1.28e-9. So
//!    the `MOD` construction is sound and the `SS=1` record means what it should, measured
//!    rather than assumed — NONMEM raises `WARNING 68` for `MOD` outside a simulation
//!    block, which is about gradients and is inert at `MAXEVAL=0`.
//! 3. **A closed form outside both engines agrees with both.** For
//!    `dA/dt = -k·A·(1 + c·τ)` on a cycle-local `τ`, `Φ(s) = exp(-k(s + c·s²/2))` and the
//!    periodic trough is `D·Φ(II)/(1 - Φ(II))`; the concentration at `τ` is
//!    `(trough + D)·Φ(τ)/V`. That reproduces NONMEM to 7.7e-9 — and ferx's own explicit
//!    train to 4.4e-10, i.e. NONMEM's 8e-9 offset is NONMEM's `TOL=9`/print floor, not
//!    ferx drift. The closed form is therefore the *tighter* reference and is asserted in
//!    its own right, not only through NONMEM.
//!
//! Both ferx prediction engines are asserted **separately**, and the second one has to be
//! reached carefully. `compute_predictions_with_tv` gives `ode_predictions_event_driven`
//! (the model-time reroute, #1124). `compute_predictions_with_states` gives that *same*
//! engine in its first tuple element — `ode_predictions_event_driven_with_states` returns
//! `ipreds` straight from it — and only its **second** element comes from
//! `ode_dense_solve_states`. So the states test reads `.1`, divides the state amount by `V`
//! by hand (`apply_scaling` never touches the state vector), and is the only comparison
//! here that exercises a second integration. Verified, not assumed: `.0` is bit-identical
//! to the objective path at all four times, while `.1/V` differs from it at three of them.
//! A failure names which engine drifted.
//!
//! One test goes further and runs `fit()` end to end against NONMEM's `#OBJV`, because
//! #1139's reported symptom is a non-finite *objective* and every predictor-level test in
//! this file would stay green if the likelihood assembly below it were broken.
//!
//! # Tiering
//!
//! Ungated. Every check is a single ODE evaluation at η = 0 with no convergence loop, and
//! they carry this diff's Codecov patch coverage.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::{compute_predictions_with_states, compute_predictions_with_tv};
use ferx_core::types::CompiledModel;
use ferx_core::{read_nonmem_csv, Population};

fn anchor(name: &str) -> std::path::PathBuf {
    // `CARGO_MANIFEST_DIR`, not a relative path — the sibling anchor suites all resolve
    // this way, and a bare relative path only works while the runner happens to set cwd
    // to the package root.
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("nonmem_anchor");
    p.push(name);
    p
}

fn model(file: &str) -> CompiledModel {
    let src = std::fs::read_to_string(anchor(file)).expect("the ferx twin is committed");
    parse_full_model(&src)
        .expect("the anchor model must parse")
        .model
}

fn population(csv: &str) -> Population {
    read_nonmem_csv(&anchor(csv), None, None).expect("the anchor dataset is committed")
}

/// The `IPRED` column of `results/<stream>.tab`, observation records only (`EVID == 0`),
/// in dataset order.
///
/// `$OMEGA 0 FIX` makes `IPRED == PRED` on every row; `IPRED` is taken because it is what
/// ferx's own predictors return. Dose rows are dropped rather than positionally zipped, so
/// a dataset edit cannot silently misalign the comparison.
fn nonmem_ipred(stream: &str) -> Vec<(f64, f64)> {
    let text = std::fs::read_to_string(anchor(&format!("results/{stream}.tab")))
        .expect("the NONMEM table is committed");
    let mut lines = text.lines();
    lines.next().expect("TABLE NO. banner");
    let header: Vec<&str> = lines
        .next()
        .expect("column header")
        .split(',')
        .map(str::trim)
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|c| *c == name)
            .unwrap_or_else(|| panic!("{name} column in {stream}.tab"))
    };
    let (t_col, e_col, i_col) = (col("TIME"), col("EVID"), col("IPRED"));
    let mut out = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() <= t_col.max(e_col).max(i_col) {
            continue;
        }
        let evid: f64 = f[e_col].parse().expect("numeric EVID");
        if evid != 0.0 {
            continue;
        }
        out.push((
            f[t_col].parse().expect("numeric TIME"),
            f[i_col].parse().expect("numeric IPRED"),
        ));
    }
    assert!(!out.is_empty(), "{stream}.tab has no observation rows");
    out
}

/// ferx predictions at η = 0 from one engine, as `(time, value)`.
fn ferx_rows(
    m: &CompiledModel,
    pop: &Population,
    f: impl Fn(&CompiledModel, &ferx_core::Subject, &[f64], &[f64]) -> Vec<f64>,
) -> Vec<(f64, f64)> {
    let theta = &m.default_params.theta;
    let zero_eta = vec![0.0; m.default_params.omega.dim()];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let v = f(m, s, theta, &zero_eta);
        assert_eq!(v.len(), s.obs_times.len());
        out.extend(s.obs_times.iter().copied().zip(v));
    }
    out
}

/// Compare two `(time, value)` lists, returning the worst relative error.
///
/// `is_finite` is asserted **before** the difference is folded: `f64::max` returns the
/// non-NaN operand, so a `NaN` prediction folded into a running worst-case reads as
/// "in range" and the bound passes on the strength of the rows that worked.
fn worst_rel(got: &[(f64, f64)], want: &[(f64, f64)], what: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    let mut worst = 0.0_f64;
    for (&(tg, g), &(tw, w)) in got.iter().zip(want) {
        assert!(
            (tg - tw).abs() < 1e-9,
            "{what}: time misalignment {tg} vs {tw}"
        );
        assert!(g.is_finite(), "{what}: ferx returned {g} at t={tg}");
        assert!(
            w.is_finite() && w != 0.0,
            "{what}: bad reference {w} at t={tw}"
        );
        worst = worst.max((g - w).abs() / w.abs());
    }
    worst
}

/// [`worst_rel`] for a comparison against an **external** reference, where bit-identity
/// would mean the two sides are not independent — a table that failed to parse into the
/// values it should hold, or a helper wired to compare something with itself.
///
/// Not folded into `worst_rel`, because two ferx engines *may* legitimately agree
/// bit-for-bit and this suite makes that comparison too.
fn worst_rel_external(got: &[(f64, f64)], want: &[(f64, f64)], what: &str) -> f64 {
    let worst = worst_rel(got, want, what);
    assert!(
        worst > 0.0,
        "{what}: bit-identical to an external reference is implausible — check that the \
         table parsed and that the two sides are really different objects"
    );
    worst
}

/// `Φ(s) = exp(-k·(s + c·s²/2))` — the decay factor of `dA/dt = -k·A·(1 + c·τ)` over a
/// cycle-local span `[0, s]`.
fn phi(s: f64, k: f64, c: f64) -> f64 {
    (-k * (s + c * s * s / 2.0)).exp()
}

/// Every constant the closed form and the quoted realised errors depend on, checked
/// against the committed twin and dataset rather than assumed.
///
/// `closed_form` below is the tightest oracle in this file (ferx sits 3.0e-10 from it), and
/// it is written from literals. Editing `ss_tad_fit.ferx`'s `TVV` or its `0.03` coefficient,
/// or `ss_tad.csv`'s `II`, would leave every number here describing a model that no longer
/// exists — and the failure would read as a code regression rather than a fixture edit,
/// which is expensive precisely because the values are quoted to 13 digits and look
/// authoritative. `V` was already guarded in one test; this guards all four, in one place.
fn assert_fixture_constants_unchanged() {
    let m = model("ss_tad_fit.ferx");
    let (cl, v) = (m.default_params.theta[0], m.default_params.theta[1]);
    assert!(
        (cl - 1.0).abs() < 1e-12 && (v - 20.0).abs() < 1e-12,
        "ss_tad_fit.ferx's TVCL/TVV moved to {cl}/{v}; every constant below assumes 1 / 20"
    );
    let src = std::fs::read_to_string(anchor("ss_tad_fit.ferx")).expect("twin is committed");
    assert!(
        src.contains("0.03*TAD"),
        "ss_tad_fit.ferx no longer carries the `0.03*TAD` term the closed form integrates"
    );
    // `II` lives in the dataset, not the model, so it can drift independently of both.
    let pop = population("ss_tad.csv");
    let d = &pop.subjects[0].doses[0];
    assert!(
        d.ss && (d.ii - 12.0).abs() < 1e-12 && (d.amt - 100.0).abs() < 1e-12,
        "ss_tad.csv's SS dose moved to ii={} amt={}; the closed form assumes 12 / 100",
        d.ii,
        d.amt
    );
}

/// The closed-form steady-state concentration of `ss_tad_fit.ferx` at time `t`, computed
/// **outside both engines**: the fixed point of the one-cycle map `A ↦ (A + D)·Φ(II)`,
/// read out `τ = t - 480` after the pulse and scaled by `V`.
///
/// Constants are literals; [`assert_fixture_constants_unchanged`] is what keeps them tied to
/// the committed twin, and every test that uses this calls it first.
fn closed_form(t: f64) -> f64 {
    let (k, c, d, ii, v) = (1.0 / 20.0, 0.03, 100.0, 12.0, 20.0);
    let rho = phi(ii, k, c);
    let peak = d * rho / (1.0 - rho) + d;
    peak * phi(t - 480.0, k, c) / v
}

/// **The objective engine** (`compute_predictions_with_tv` → `ode_predictions_event_driven`
/// via #1124's model-time reroute) against NONMEM's `SS=1` run.
///
/// Before this change every value here was `NaN`.
#[test]
fn ss_tad_objective_engine_matches_nonmem() {
    let m = model("ss_tad_fit.ferx");
    let pop = population("ss_tad.csv");
    let got = ferx_rows(&m, &pop, compute_predictions_with_tv);
    // Realised error, measured: 7.404e-9 relative worst over the four observations
    // (8.8902009676679 against NONMEM 8.8902010334129). Bound at 5e-8, ~7x headroom.
    // It cannot usefully be tighter: NONMEM's own values sit 7.7e-9 *above* the exact
    // closed form, which is `TOL=9` plus the `1PE20.13` print, so this bound measures
    // NONMEM's accuracy floor rather than ferx's.
    let worst = worst_rel_external(
        &got,
        &nonmem_ipred("ss_tad"),
        "objective engine vs NONMEM SS",
    );
    assert!(worst < 5e-8, "worst relative error {worst:.3e}");
}

/// **The states engine** — `ode_dense_solve_states`, reached through
/// `compute_predictions_with_states`'s *second* tuple element.
///
/// **Read `.1`, never `.0`.** `compute_predictions_with_states` on a model-time-reading
/// model calls `ode_predictions_event_driven_with_states`, which returns `ipreds` straight
/// from `ode_predictions_event_driven` (`ode/predictions.rs`) — the very engine the
/// objective test above uses — and only fills the *states* from `ode_dense_solve_states` in
/// a second pass. So a `.0` comparison here would be the same engine twice under two names:
/// two callers, not two geometries. This test read `.0` when it was written, and was
/// therefore a duplicate of its neighbour; it is the exact hazard CLAUDE.md names.
///
/// `ode_dense_solve_states` runs its own steady-state equilibration (`apply_segment_boundary`
/// → `equilibrate_ss_state`), so it genuinely needs anchoring — it is not along for the ride.
///
/// The states are **amounts**: `apply_scaling` is applied to `ipred` only, never to the state
/// vector, so the comparison divides by `V` by hand. That is also why this cannot be folded
/// into the test above.
///
/// `pk_at_obs.first()` is exact here: the model has no time-varying covariates, so every
/// entry of that vector is identical and the second pass's fixed-snapshot approximation is
/// not an approximation.
#[test]
fn ss_tad_states_engine_matches_nonmem() {
    let m = model("ss_tad_fit.ferx");
    let pop = population("ss_tad.csv");
    assert_fixture_constants_unchanged();
    // `V` from the committed twin, so the hand-applied scaling cannot drift from the model —
    // and used directly rather than re-spelled as a literal beside its own assertion.
    let v = m.default_params.theta[1];
    let got = ferx_rows(&m, &pop, |m, s, th, e| {
        let (_ipred, states) = compute_predictions_with_states(m, s, th, e);
        states
            .iter()
            .map(|u| {
                assert!(!u.is_empty(), "the state vector must carry `central`");
                u[0] / v
            })
            .collect()
    });
    // Realised error, measured: **7.956e-9** relative worst — deliberately *not* the
    // objective engine's 7.404e-9. The two differ in the last few bits (identical at t=482,
    // bit-different at 485/488/491 by ~3e-11) because they are two integrations, and that
    // difference is the evidence this test is not a duplicate of its neighbour. If it ever
    // becomes bit-identical, someone has routed both callers to one engine again.
    // Bound at 5e-8, NONMEM's own accuracy floor, as above.
    let worst = worst_rel_external(&got, &nonmem_ipred("ss_tad"), "states engine vs NONMEM SS");
    assert!(worst < 5e-8, "worst relative error {worst:.3e}");
}

/// **The objective, end to end through `fit()`** — #1139's *reported* symptom, which none of
/// the prediction tests above actually touches.
///
/// The issue is titled around `NaN` predictions, but what a user sees is a non-finite
/// objective: measured on this model before the fix, `OFV = NaN` with a
/// `W_ODE_SOLVER_DIAGNOSTICS` warning blaming the solver rather than the anchor. Every other
/// test in this file calls a predictor directly and would stay green if the likelihood
/// assembly downstream of it were broken, so this is the one that pins the thing the issue
/// is about.
///
/// NONMEM's `#OBJV` for `ss_tad.ctl` is **286.890**, printed to six significant figures, and
/// the same value for the explicit 41-dose train — which is itself worth an assertion, since
/// two runs agreeing to the printed digits is how the steady-state record earns the claim
/// that it stands in for the train.
///
/// `maxiter = 0`, so this returns after one objective evaluation: Tier 2, no convergence loop.
#[test]
fn ss_tad_objective_matches_nonmem_end_to_end() {
    let src = std::fs::read_to_string(anchor("ss_tad_fit.ferx")).expect("twin is committed");
    let parsed = parse_full_model(&src).expect("parses");
    let (m, opts) = (parsed.model, parsed.fit_options);
    let ss = ferx_core::fit(&m, &population("ss_tad.csv"), &m.default_params, &opts)
        .expect("the SS fit runs — before #1139 this produced a non-finite objective");
    let train = ferx_core::fit(&m, &population("train_tad.csv"), &m.default_params, &opts)
        .expect("the explicit-train fit runs");

    // `is_finite` first and on its own: that is the regression, and a bound checked against
    // a `NaN` reads as false rather than as a failure, so it must be asserted separately.
    assert!(
        ss.ofv.is_finite(),
        "SS objective is {} — the #1139 symptom is back",
        ss.ofv
    );
    assert!(train.ofv.is_finite(), "train objective is {}", train.ofv);

    // NONMEM `#OBJV` from `results/ss_tad.lst` and `results/train_tad.lst`, both 286.890 —
    // printed to 6 significant figures, so this bound measures NONMEM's *print*, not ferx.
    // Realised, measured: ferx 286.889524669658 (SS) and 286.889524669616 (train), both
    // 1.657e-6 from the printed value — well inside half a unit in its last printed digit.
    // Bound at 1e-5, 6x headroom over that.
    const NM_OBJV: f64 = 286.890;
    for (what, got) in [("SS", ss.ofv), ("train", train.ofv)] {
        let rel = (got - NM_OBJV).abs() / NM_OBJV;
        assert!(
            rel < 1e-5,
            "{what} OFV {got:.5} vs NONMEM {NM_OBJV} (rel {rel:.3e})"
        );
    }
    // And the two ferx runs agree with each other far more tightly than either agrees with
    // NONMEM's printed value — the steady-state record really is standing in for the train,
    // not merely landing in the same print bucket.
    // Realised, measured: 1.474e-13 — five orders tighter than either is from NONMEM's
    // printed value, which is the whole claim. Bound at 1e-9, ~6.8e3 headroom, left that
    // wide only for the adaptive solver's platform-to-platform step drift (#990 / #1241).
    let across = (ss.ofv - train.ofv).abs() / train.ofv;
    assert!(
        across < 1e-9,
        "SS and train objectives differ by {across:.3e}; the run-in is not reproducing the \
         explicit train"
    );
}

/// **The certifying control**: ferx's *explicit* 41-dose train against NONMEM's, on the
/// same model file and the same observations.
///
/// This path has no steady-state machinery in it at all and was already correct, so it is
/// what makes the `SS=1` comparison above trustworthy rather than a coincidence of two
/// engines being wrong together. It is also unchanged by #1139 — the fix touches only the
/// run-in — so if this ever moves, the forward walk did, not the equilibration.
#[test]
fn tad_train_matches_nonmem() {
    let m = model("ss_tad_fit.ferx");
    let pop = population("train_tad.csv");
    let got = ferx_rows(&m, &pop, compute_predictions_with_tv);
    // Realised error, measured: 9.07e-9 relative worst (8.8902009676645 against NONMEM
    // 8.8902010448184). Bound at 5e-8, ~5.5x headroom — again NONMEM's floor, since ferx's
    // train matches the exact closed form to 4.4e-10 (asserted below).
    let worst = worst_rel_external(
        &got,
        &nonmem_ipred("train_tad"),
        "ferx train vs NONMEM train",
    );
    assert!(worst < 5e-8, "worst relative error {worst:.3e}");
}

/// **The anchor's own consistency, read from the committed tables.** NONMEM's steady-state
/// routine must reproduce NONMEM's own explicit train, or the `SS=1` record does not mean
/// what this suite assumes and every comparison above is anchored to nothing.
///
/// In-repo rather than a sentence in a PR description, because that is the claim a reader
/// would otherwise have to take on trust.
#[test]
fn nonmem_ss_reproduces_its_own_train() {
    let worst = worst_rel_external(
        &nonmem_ipred("ss_tad"),
        &nonmem_ipred("train_tad"),
        "NONMEM SS vs NONMEM train",
    );
    // Measured: 1.283e-9. Bound at 1e-8, ~8x headroom.
    assert!(worst < 1e-8, "worst relative error {worst:.3e}");
}

/// **Why `TAD` is anchorable and absolute time is not** — the property that separates
/// #1139's two halves, asserted from the committed N = 21 / N = 41 tables rather than
/// argued.
///
/// A `TAD`-reading train has a periodic limit (`TAD` is bounded inside one interval), so
/// "the steady state" is a thing to compute. An absolute-clock train has none: it keeps
/// moving as the run-in lengthens, so there is nothing for an equilibration to converge to
/// and anchoring it at the run-in's own origin would silently redefine it as `TAD`.
#[test]
fn the_tad_train_converges_and_the_absolute_time_train_does_not() {
    // The N = 21 streams end their train at t = 240 rather than 480, so their samples sit
    // at 242 / 245 / 248 / 251 — the same four *phases* of the cycle, at different absolute
    // times. Compare on the phase, and assert that the phases really do line up, so a
    // future dataset edit cannot silently make this compare unrelated points.
    let by_phase = |stream: &str, last_dose: f64| -> Vec<(f64, f64)> {
        nonmem_ipred(stream)
            .into_iter()
            .map(|(t, v)| (t - last_dose, v))
            .collect()
    };
    let tad = worst_rel_external(
        &by_phase("train_tad_n21", 240.0),
        &by_phase("train_tad", 480.0),
        "TAD train N=21 vs N=41",
    );
    let tabs = worst_rel_external(
        &by_phase("train_tabs_n21", 240.0),
        &by_phase("train_tabs", 480.0),
        "absolute-time train N=21 vs N=41",
    );
    assert_eq!(
        by_phase("train_tad_n21", 240.0)
            .iter()
            .map(|&(p, _)| p)
            .collect::<Vec<_>>(),
        vec![2.0, 5.0, 8.0, 11.0],
        "the short train's samples must sit at the same cycle phases as the long one's"
    );
    // Measured: 3.5e-7 for TAD, 0.294 for absolute time — six orders apart.
    assert!(
        tad < 1e-5,
        "the TAD train must have a periodic limit; N=21 vs N=41 moved {tad:.3e}"
    );
    assert!(
        tabs > 0.1,
        "the absolute-time train must NOT have one, or this test has stopped separating \
         the two halves of #1139; N=21 vs N=41 moved only {tabs:.3e}"
    );
}

/// **The T3 parity guard.** ferx's steady state on an *absolute-clock* RHS matches NONMEM's
/// — both engines equilibrate on a cycle-local clock and then integrate forward on absolute
/// time, so both sit ~67% away from their own (non-convergent) train, together.
///
/// This is committed as a **guard, not a claim of correctness**: the number is not the
/// steady state of a system that has one, and #1139's remaining half is what to do about
/// that. What must not happen is a change to the run-in quietly moving a value that agrees
/// with NONMEM today. Measured before and after #1139's `TAD` fix, this value is
/// bit-identical (`8.5679354785102` at t = 482); the fix leaves the `TAFD` slot `NaN` and
/// never touches the `T` path, and this test is what would notice if that stopped being
/// true.
#[test]
fn absolute_time_ss_still_matches_nonmem() {
    let m = model("ss_tabs_fit.ferx");
    let pop = population("ss_tabs.csv");
    let got = ferx_rows(&m, &pop, compute_predictions_with_tv);
    // Realised error, measured: 1.528e-9 relative worst. Bound at 1e-8, ~6.5x headroom.
    let worst = worst_rel_external(&got, &nonmem_ipred("ss_tabs"), "absolute-time SS vs NONMEM");
    assert!(worst < 1e-8, "worst relative error {worst:.3e}");
}

/// **The two spellings of absolute time agree on the value, not only on the route.**
///
/// `T`/`t` resolve through a slot the compiled program's `uses_time_vars` flag can see; a
/// bare `TIME` compiles to `Op::PushTime` and is visible only to `reads_time_builtin`.
/// #1124 closed that asymmetry on the *gradient* route — before it, `TIME` reached the dual
/// steady-state equilibration while `T` declined, and the gate's own regression test spelled
/// the term `T` to work around it. Nothing checked that the two produce the same **value**
/// under `SS=1`, which is the half that reaches a user's results.
///
/// Measured, they are bit-identical, so this is asserted on the bit pattern rather than to a
/// tolerance: they are the same quantity through the same engine and any difference at all
/// would mean one spelling had acquired its own path.
#[test]
fn both_spellings_of_absolute_time_give_the_same_steady_state() {
    let pop = population("ss_tabs.csv");
    let t_rows = ferx_rows(
        &model("ss_tabs_fit.ferx"),
        &pop,
        compute_predictions_with_tv,
    );
    let time_rows = ferx_rows(
        &model("ss_tabs_time_fit.ferx"),
        &pop,
        compute_predictions_with_tv,
    );
    assert_eq!(t_rows.len(), time_rows.len());
    for (&(t, a), &(_, b)) in t_rows.iter().zip(&time_rows) {
        assert!(a.is_finite() && b.is_finite(), "t={t}: {a} / {b}");
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "`T` and `TIME` disagree at t={t} ({a} vs {b}) — the same quantity under two \
             spellings has acquired two paths, which is #1124 reopening on the value side"
        );
    }
}

/// **The size of the absolute-clock gap, asserted rather than described.**
///
/// A `T`-reading steady state is not the limit of a `T`-reading dose train, on either
/// engine: the run-in equilibrates on a cycle-local clock and then integrates forward on
/// absolute time, so `SS=1` and an explicit 41-dose train disagree — measured **67 %** at
/// t = 482, and by the same factor in NONMEM as in ferx.
///
/// This exists because that number lived only in a doc comment on a *routing* test
/// (`sens/ode_provider_tests.rs`), where it was documenting a defect the routing gate does
/// not fix and could not notice if it changed. The `TAD` half of the same table is now
/// wrong — that is what this PR changed — so the whole table had to become assertions or
/// be deleted. Here it is measured from the committed streams.
#[test]
fn the_absolute_clock_steady_state_is_not_its_own_train_on_either_engine() {
    let nm_gap = worst_rel_external(
        &nonmem_ipred("ss_tabs"),
        &nonmem_ipred("train_tabs"),
        "NONMEM absolute-time SS vs its own train",
    );
    let m = model("ss_tabs_fit.ferx");
    let ferx_ss = ferx_rows(&m, &population("ss_tabs.csv"), compute_predictions_with_tv);
    let ferx_train = ferx_rows(
        &m,
        &population("train_tabs.csv"),
        compute_predictions_with_tv,
    );
    let ferx_gap = worst_rel(
        &ferx_ss,
        &ferx_train,
        "ferx absolute-time SS vs its own train",
    );

    // Measured: 0.673 on both sides at t = 482 (8.5679 against 5.1221).
    assert!(
        nm_gap > 0.5 && ferx_gap > 0.5,
        "the absolute-clock SS/train gap must still be large on both engines (NONMEM \
         {nm_gap:.3}, ferx {ferx_gap:.3}) — if it has closed, `T`/`TIME` under SS is no \
         longer the open question this suite treats it as"
    );
    // …and the two engines must be wrong *together*, which is what makes the parity guard
    // above a parity rather than a coincidence. Measured: 1.4e-9 apart.
    assert!(
        (nm_gap - ferx_gap).abs() < 1e-6,
        "ferx and NONMEM disagree about the size of the gap ({ferx_gap:.9} vs \
         {nm_gap:.9}) — they no longer share the cycle-local convention"
    );

    // The contrast that makes this a statement about absolute clocks and not about steady
    // states in general: the same comparison on the `TAD` model agrees to 3.8e-13.
    let m_tad = model("ss_tad_fit.ferx");
    let tad_gap = worst_rel(
        &ferx_rows(
            &m_tad,
            &population("ss_tad.csv"),
            compute_predictions_with_tv,
        ),
        &ferx_rows(
            &m_tad,
            &population("train_tad.csv"),
            compute_predictions_with_tv,
        ),
        "ferx TAD SS vs its own train",
    );
    assert!(
        tad_gap < 1e-9,
        "the TAD steady state must reproduce its own train ({tad_gap:.3e}) — this is the \
         property #1139 restores, and without it the comparison above says nothing"
    );
}

/// **The third reference.** A closed form computed outside both engines, so ferx and NONMEM
/// cannot agree on a wrong answer between themselves.
///
/// This is the tightest of the three: ferx's `SS=1` values sit 2.98e-10 from it, while
/// NONMEM's sit 7.7e-9 away — the offset is NONMEM's `TOL=9` and `1PE20.13` print, not ferx
/// drift, which is worth knowing before reading the 5e-8 bounds above as "loose".
///
/// The formula is also the reason the run-in's `TAD` anchor is `0`: `Φ` is written on a
/// clock whose origin is the *pulse*, and that is the only clock on which the one-cycle map
/// is the same map every cycle.
#[test]
fn the_closed_form_certifies_both_engines_and_nonmem() {
    assert_fixture_constants_unchanged();
    let m = model("ss_tad_fit.ferx");
    let pop = population("ss_tad.csv");
    let ferx_ss = ferx_rows(&m, &pop, compute_predictions_with_tv);
    let exact: Vec<(f64, f64)> = ferx_ss.iter().map(|&(t, _)| (t, closed_form(t))).collect();

    // Measured: 2.976e-10 worst. Bound at 5e-9, ~17x headroom.
    let ferx_worst = worst_rel_external(&ferx_ss, &exact, "ferx SS vs the closed form");
    assert!(ferx_worst < 5e-9, "ferx vs closed form {ferx_worst:.3e}");

    // Measured: 8.330e-9 worst — NONMEM's own accuracy floor on this problem.
    let nm_worst = worst_rel_external(
        &nonmem_ipred("ss_tad"),
        &exact,
        "NONMEM SS vs the closed form",
    );
    assert!(nm_worst < 5e-8, "NONMEM vs closed form {nm_worst:.3e}");

    // And the ordering itself: ferx is closer to the exact answer than NONMEM is. If that
    // ever inverts, the bounds above are measuring ferx rather than the reference and
    // should be re-derived rather than relaxed.
    assert!(
        ferx_worst < nm_worst,
        "ferx {ferx_worst:.3e} is no longer closer to the closed form than NONMEM \
         {nm_worst:.3e} — re-measure before touching any bound in this file"
    );
}
