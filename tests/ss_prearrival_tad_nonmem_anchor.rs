//! NONMEM anchor for the **pre-arrival `TAD` referent of a lagged steady-state dose** —
//! issue [#1126](https://github.com/FeRx-NLME/ferx-core/issues/1126), batch-T step T4.
//!
//! `d/dt(central) = -(CL/V)*central*(1 + 0.03*TAD)`, `obs_scale = V`, one `SS=1` record with
//! `II = 12` at `t = 480` and `ALAG1 = 3`, so the lagged arrival is at 483 and `[480, 483)` is
//! the window this issue is about. Since #1121 the periodic state is loaded at the **record**
//! and flows to the arrival, so the state in that window is real — the previous cycle's
//! decaying tail — but `TAD` there had no referent pointing at it.
//!
//! Before this change the whole subject was `NaN`, on both ODE predictors and in the sdtab
//! `TAD` column (blank). Two independent causes, and the PR's mutation table shows neither
//! fix alone suffices.
//!
//! # The oracle is the explicit train. NONMEM's own `SS=1` record is a measured negative
//!
//! This is the `ss_chz_r1`/`r2` shape, and it is measured rather than asserted:
//!
//! 1. **The lag machinery is exact.** On an *autonomous* RHS NONMEM's lagged `SS=1` record
//!    reproduces its own lagged 41-dose train to **1.14e-9** (`ss_lag_auto` / `train_lag_auto`),
//!    and both sit within **8.91e-10** of a closed form outside either engine. So nothing about
//!    lag + `SS=1` is in doubt on its own.
//! 2. **Add the `TAD` read and NONMEM's SS routine drifts from its own train** — **1.34e-2**
//!    inside the pre-arrival window and **6.61e-3** after the arrival. It is wrong on *both*
//!    sides, so this is not a window-only artifact. Landing ferx on that number would be
//!    reproducing a defect; `nonmem_ss_record_is_a_measured_negative` pins the distance so a
//!    future change that "fixes" ferx onto it goes red.
//! 3. **A periodic limit exists** for the lagged `TAD` train — `train_tadlag_n21` against
//!    `train_tadlag` moves **7.09e-7** over a doubling — so the run-in has something to
//!    converge to. (`TAFD`/`T`/`TIME` under `SS=1` do not, which is why they are handled
//!    separately — T3 of #1258 reports them, `W_STEADY_STATE_ABSOLUTE_TIME`, and moves no
//!    value. The values they *do* produce stay anchored in
//!    `tests/ss_model_time_nonmem_anchor.rs`; nothing in this file changed.)
//! 4. **A closed form outside both engines** agrees with both. For `dA/dt = -k·A·(1 + c·τ)` on
//!    a cycle-local `τ`, `Φ(s) = exp(-k(s + c·s²/2))` and the periodic trough is
//!    `D·Φ(II)/(1 − Φ(II))`; the concentration at `τ` is `(trough + D)·Φ(τ)/V`. A lagtime only
//!    *shifts* the pulse train, so this is unchanged by it — the pre-arrival window is simply
//!    cycle-local `τ ∈ [II − lag, II)`. It is the **tighter** reference: ferx sits 4.7e-10 from
//!    it, NONMEM 1.73e-8.
//!
//! # The referent is not invented here
//!
//! `pk::predict_concentration` (the analytical superposition) has read this window at elapsed
//! time `t − (dose.time − ss_seed_phase(dose, lagtime))` since #1121, anchored against NONMEM
//! including the `lag ≥ II` clamp (`nonmem_anchor/results/ss_lag_ge_ii`). This change makes the
//! ODE engines' `TAD` agree with the pulse their own analytical sibling already measures from.
//!
//! # Which engine sees what — this is not symmetric
//!
//! `apply_segment_boundary` **overwrites** the flowed state with a fresh `equilibrate_ss_state`
//! at the lagged arrival when `ss_arrival_is_trough`, so the dense states walk discards whatever
//! crossed the pre-arrival window. Measured at `955ccef2`: its post-arrival values were already
//! *correct* (3e-10) while the event-driven walk's were `NaN`. So a post-arrival assertion on
//! the dense engine is **blind** to the walk's anchor, and the tests below put the pre-arrival
//! checks on both engines and the post-arrival ones on the event-driven walk.
//!
//! # Tiering
//!
//! Ungated. Every check is a single ODE evaluation at η = 0 with no convergence loop.

use ferx_core::parser::model_parser::parse_full_model;
use ferx_core::pk::{compute_predictions_with_states, compute_predictions_with_tv};
use ferx_core::types::CompiledModel;
use ferx_core::{read_nonmem_csv, Population};
use std::collections::HashMap;

fn anchor(name: &str) -> std::path::PathBuf {
    // `CARGO_MANIFEST_DIR`, not a relative path — the sibling anchor suites all resolve this
    // way, and a bare relative path only works while the runner happens to set cwd to the
    // package root.
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

/// Parsed once per process and shared. Every test in this file reads two or three datasets
/// and `closed_form_rows` reads one again by name, so the uncached form ran the NONMEM CSV
/// parser a dozen times over the same handful of files.
fn population(csv: &str) -> &'static Population {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<String, &'static Population>>> =
        std::sync::OnceLock::new();
    let mut guard = CACHE
        .get_or_init(|| std::sync::Mutex::new(HashMap::new()))
        .lock()
        .expect("the population cache mutex is never poisoned by these tests");
    guard.entry(csv.to_string()).or_insert_with(|| {
        let p = read_nonmem_csv(&anchor(csv), None, None).expect("the anchor dataset is committed");
        // Leaked deliberately: these live for the whole test binary and handing back a
        // `&'static` keeps every call site's `&Population` borrow trivially valid.
        Box::leak(Box::new(p))
    })
}

/// The `IPRED` column of `results/<stream>.tab`, observation rows only (`EVID == 0`), in
/// dataset order. `$OMEGA 0 FIX` makes `IPRED == PRED` on every row. Dose rows are dropped
/// rather than positionally zipped, so a dataset edit cannot silently misalign the comparison.
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
        if f[e_col].parse::<f64>().expect("numeric EVID") != 0.0 {
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

/// The **event-driven** walk's predictions at η = 0 — the production predictor for this model
/// class (`compute_predictions_with_tv` routes a model-time-reading RHS there, #1124).
fn ferx_event_driven(m: &CompiledModel, pop: &Population) -> Vec<(f64, f64)> {
    let zero = vec![0.0; m.default_params.omega.dim()];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let v = compute_predictions_with_tv(m, s, &m.default_params.theta, &zero);
        assert_eq!(v.len(), s.obs_times.len());
        out.extend(s.obs_times.iter().copied().zip(v));
    }
    out
}

/// The **dense states** walk's predictions at η = 0, read out of the state vector by hand.
///
/// `compute_predictions_with_states`'s first element comes straight from the event-driven walk
/// (`ode_predictions_event_driven_with_states` returns its `ipreds` unchanged), so only `.1` is
/// a second integration. `apply_scaling` never touches the state vector, so the `/ V` is done
/// here. `state_divisor_is_v` pins that this is the right divisor.
fn ferx_dense_states(m: &CompiledModel, pop: &Population) -> Vec<(f64, f64)> {
    let zero = vec![0.0; m.default_params.omega.dim()];
    let v = m.default_params.theta[1];
    let mut out = Vec::new();
    for s in &pop.subjects {
        let (_, states) = compute_predictions_with_states(m, s, &m.default_params.theta, &zero);
        assert_eq!(states.len(), s.obs_times.len());
        // Index 0 is `central` only while these twins stay one-compartment. A fixture that
        // gained a depot or peripheral would make this read the wrong state and compare it
        // against `central`'s reference — a failure that reads as an engine regression
        // rather than the fixture edit it is. `fixture_constants_…` guards every other
        // literal this suite leans on; this is the one it cannot see.
        assert!(
            states.iter().all(|u| u.len() == 1),
            "ferx_dense_states reads u[0] as the observed compartment; this twin is no \
             longer single-state"
        );
        out.extend(
            s.obs_times
                .iter()
                .copied()
                .zip(states.iter().map(|u| u[0] / v)),
        );
    }
    out
}

/// Worst relative error between two `(time, value)` lists, over the rows `keep` selects.
///
/// `is_finite` is asserted **before** the difference is folded, not after: `f64::max` returns
/// the non-NaN operand, so a `NaN` prediction folded into a running worst-case reads as "in
/// range" and the bound passes on the strength of the rows that worked. The entire defect this
/// file anchors is that these predictions used to be `NaN`, so that guard is not decoration.
fn worst_rel(
    got: &[(f64, f64)],
    want: &[(f64, f64)],
    what: &str,
    keep: impl Fn(f64) -> bool,
) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: row count");
    let mut worst = 0.0_f64;
    let mut n = 0usize;
    for (&(tg, g), &(tw, w)) in got.iter().zip(want) {
        assert!(
            (tg - tw).abs() < 1e-9,
            "{what}: time misalignment {tg} vs {tw}"
        );
        if !keep(tg) {
            continue;
        }
        n += 1;
        assert!(g.is_finite(), "{what}: ferx returned {g} at t={tg}");
        assert!(
            w.is_finite() && w != 0.0,
            "{what}: bad reference {w} at t={tw}"
        );
        worst = worst.max((g - w).abs() / w.abs());
    }
    assert!(n > 0, "{what}: the row filter selected nothing to compare");
    worst
}

/// [`worst_rel`] over every row.
fn worst_rel_all(got: &[(f64, f64)], want: &[(f64, f64)], what: &str) -> f64 {
    worst_rel(got, want, what, |_| true)
}

/// [`worst_rel`] for a comparison against an **external** reference, where bit-identity would
/// mean the two sides are not independent — a table that failed to parse, or a helper wired to
/// compare something with itself. Not folded into [`worst_rel`], because two ferx engines may
/// legitimately agree bit-for-bit and this suite makes that comparison too.
fn worst_rel_external(got: &[(f64, f64)], want: &[(f64, f64)], what: &str) -> f64 {
    let worst = worst_rel_all(got, want, what);
    assert!(
        worst > 0.0,
        "{what}: bit-identical to an external reference is implausible — check that the table \
         parsed and that the two sides are really different objects"
    );
    worst
}

// ---------------------------------------------------------------------------
// The closed form, outside both engines.
// ---------------------------------------------------------------------------

/// `Φ(s) = exp(-k·(s + c·s²/2))` — the decay factor of `dA/dt = -k·A·(1 + c·τ)` over a
/// cycle-local span `[0, s]`.
fn phi(s: f64, k: f64, c: f64) -> f64 {
    (-k * (s + c * s * s / 2.0)).exp()
}

/// Periodic pre-pulse trough amount of the bolus train: `D·Φ(II)/(1 − Φ(II))`.
fn trough(k: f64, c: f64, d: f64, ii: f64) -> f64 {
    let p = phi(ii, k, c);
    d * p / (1.0 - p)
}

/// Cycle-local `τ` at `t` for a pulse train with `ALAG = lag` and record `t_rec`.
///
/// This is the geometry the *train* has, and — for `lag ≤ II` — the geometry a correct `SS=1`
/// record must reproduce: pulses at `t_rec + lag + k·II`, so the pre-arrival window
/// `[t_rec, t_rec + lag)` is `τ ∈ [II − lag, II)`.
fn tau_of(t: f64, t_rec: f64, lag: f64, ii: f64) -> f64 {
    (t - (t_rec + lag)).rem_euclid(ii)
}

/// Every constant the closed form and the quoted realised errors depend on, checked against the
/// committed twin and dataset rather than assumed.
///
/// The numbers below are written from literals and quoted to 13 digits, so they look
/// authoritative. Editing `ss_tadlag_fit.ferx`'s `TVV`, its `0.03` coefficient or its `TVLAG`,
/// or `ss_tadlag.csv`'s `II`/`AMT`, would leave every one of them describing a model that no
/// longer exists — and the failure would read as a code regression rather than a fixture edit.
#[test]
fn fixture_constants_are_what_the_closed_form_assumes() {
    let m = model("ss_tadlag_fit.ferx");
    let (cl, v, lag) = (
        m.default_params.theta[0],
        m.default_params.theta[1],
        m.default_params.theta[2],
    );
    assert_eq!((cl, v, lag), (1.0, 20.0, 3.0), "TVCL / TVV / TVLAG");

    let pop = population("ss_tadlag.csv");
    let s = &pop.subjects[0];
    assert_eq!(s.doses.len(), 1, "one SS record");
    let d = &s.doses[0];
    assert!(d.ss, "the record is SS=1");
    assert_eq!((d.time, d.amt, d.ii), (480.0, 100.0, 12.0));
    assert_eq!(
        s.obs_times,
        vec![481.0, 482.0, 485.0, 488.0, 491.0],
        "two observations inside the pre-arrival window [480, 483) and three past the \
         arrival — a fixture that saw only one side could not observe a referent that is \
         wrong on the other"
    );

    // …and the `0.03` coefficient, which lives in the RHS rather than in a theta. Read it off
    // the model source, so an edit there cannot silently drift from the closed form.
    let src = std::fs::read_to_string(anchor("ss_tadlag_fit.ferx")).expect("twin");
    assert!(
        src.contains("(1.0 + 0.03*TAD)"),
        "the closed form's `c` is 0.03; the RHS must still spell it"
    );
}

/// The `/ V` in [`ferx_dense_states`] is the model's own `obs_scale`, not a coincidence.
#[test]
fn state_divisor_is_v() {
    let m = model("ss_tadlag_fit.ferx");
    let src = std::fs::read_to_string(anchor("ss_tadlag_fit.ferx")).expect("twin");
    assert!(src.contains("obs_scale = V"));
    assert_eq!(m.default_params.theta[1], 20.0);
}

const K: f64 = 1.0 / 20.0; // CL / V
const C: f64 = 0.03;
const D: f64 = 100.0;
const II: f64 = 12.0;
const V: f64 = 20.0;

/// Closed-form concentration of the bolus train at absolute time `t`, for lagtime `lag < II`.
fn closed_form(t: f64, lag: f64, c: f64) -> f64 {
    (trough(K, c, D, II) + D) * phi(tau_of(t, 480.0, lag, II), K, c) / V
}

/// Reference rows for a stream, from the closed form, aligned to that dataset's own times.
fn closed_form_rows(csv: &str, lag: f64, c: f64) -> Vec<(f64, f64)> {
    population(csv).subjects[0]
        .obs_times
        .iter()
        .map(|&t| (t, closed_form(t, lag, c)))
        .collect()
}

// ---------------------------------------------------------------------------
// The anchor itself.
// ---------------------------------------------------------------------------

/// **The exit condition.** ferx's lagged `SS=1` record must land on the explicit lagged train,
/// inside the pre-arrival window and after the arrival, on the production predictor.
///
/// Bounds are set from the realised errors measured 2026-09-06, with the headroom stated:
///
/// | comparison | realised | bound | headroom |
/// |---|---|---|---|
/// | vs the closed form | 4.70e-10 | 5e-9 | 10× |
/// | vs NONMEM's explicit lagged train | 1.69e-8 | 1e-7 | 6× |
/// | vs ferx's *own* explicit lagged train | 2.50e-11 | 1e-9 | 40× |
///
/// The ordering is asserted too — the closed form is the tighter reference, and NONMEM's ~1.7e-8
/// offset is its `TOL=9` / `1PE20.13` print floor rather than ferx drift. A future relaxation
/// then has to re-measure rather than loosen.
#[test]
fn ferx_lagged_ss_lands_on_the_explicit_lagged_train() {
    let m = model("ss_tadlag_fit.ferx");
    let ss = ferx_event_driven(&m, &population("ss_tadlag.csv"));
    let train = ferx_event_driven(&m, &population("train_tadlag.csv"));

    let vs_closed = worst_rel_external(&ss, &closed_form_rows("ss_tadlag.csv", 3.0, C), "SS vs cf");
    let vs_nonmem = worst_rel_external(&ss, &nonmem_ipred("train_tadlag"), "SS vs NM train");
    let vs_own = worst_rel_all(&ss, &train, "SS vs ferx's own train");

    assert!(vs_closed < 5e-9, "SS vs closed form: {vs_closed:.3e}");
    assert!(
        vs_nonmem < 1e-7,
        "SS vs NONMEM's lagged train: {vs_nonmem:.3e}"
    );
    assert!(vs_own < 1e-9, "SS vs ferx's own lagged train: {vs_own:.3e}");
    assert!(
        vs_closed < vs_nonmem,
        "the closed form must be the tighter reference (cf {vs_closed:.3e} vs NM \
         {vs_nonmem:.3e}); if that has inverted, ferx has drifted and the bounds above are \
         measuring NONMEM's print floor instead"
    );

    // The pre-arrival window on its own, so a regression confined to it cannot hide behind
    // three correct post-arrival rows.
    let pre = worst_rel(
        &ss,
        &closed_form_rows("ss_tadlag.csv", 3.0, C),
        "pre-arrival window",
        |t| t < 483.0,
    );
    assert!(pre < 5e-9, "inside [480, 483): {pre:.3e}");
}

/// The **dense states** walk, separately — and, per the module doc, only where it can see the
/// defect. It re-equilibrates at the lagged arrival, so its post-arrival rows were already
/// right at `955ccef2` while the walk's were `NaN`; the pre-arrival rows are where it is a real
/// second geometry for this fix.
///
/// Asserted as an **agreement** between the two engines rather than only against a number: the
/// slice-extension-only tree state makes them disagree by 2.73 %, and an agreement assertion
/// keeps catching that even if the target value is ever re-derived.
#[test]
fn both_ode_engines_agree_across_the_pre_arrival_boundary() {
    let m = model("ss_tadlag_fit.ferx");
    let pop = population("ss_tadlag.csv");
    let walk = ferx_event_driven(&m, &pop);
    let dense = ferx_dense_states(&m, &pop);

    let agree = worst_rel_all(&dense, &walk, "dense states vs event-driven walk");
    assert!(
        agree < 1e-9,
        "the two ODE predictors must agree; realised 6.61e-11, bound 1e-9. Got {agree:.3e} — \
         2.7e-2 is the signature of the phase advance being anchored while the walk is not"
    );

    // And the dense engine independently against the closed form, inside the window, where it
    // is not merely echoing the walk.
    let pre = worst_rel(
        &dense,
        &closed_form_rows("ss_tadlag.csv", 3.0, C),
        "dense states, pre-arrival",
        |t| t < 483.0,
    );
    assert!(pre < 5e-9, "dense states inside [480, 483): {pre:.3e}");
}

/// NONMEM's own `SS=1` record is **not** the target, and this pins how far away it is.
///
/// Landing ferx on 5.4709357327409 at t = 481 would be reproducing a defect. The distances are
/// bracketed from both sides so the test fails whether ferx drifts toward NONMEM's SS answer or
/// NONMEM's tables are silently replaced with something closer.
#[test]
fn nonmem_ss_record_is_a_measured_negative() {
    let nm_ss = nonmem_ipred("ss_tadlag");
    let nm_train = nonmem_ipred("train_tadlag");
    let worst = worst_rel_all(&nm_ss, &nm_train, "NONMEM SS vs NONMEM train");
    assert!(
        (1.2e-2..1.5e-2).contains(&worst),
        "NONMEM's lagged SS record sits 1.34e-2 from its own lagged train (1.34e-2 inside the \
         window, 6.61e-3 after the arrival). Got {worst:.4e} — if this has shrunk, the streams \
         changed and the 'train is the oracle' argument needs re-deriving"
    );

    // …and ferx must be on the *other* side of that gap by orders of magnitude.
    let m = model("ss_tadlag_fit.ferx");
    let ferx_ss = ferx_event_driven(&m, &population("ss_tadlag.csv"));
    let ferx_vs_nm_ss = worst_rel_external(&ferx_ss, &nm_ss, "ferx SS vs NONMEM SS");
    assert!(
        ferx_vs_nm_ss > 1e-3,
        "ferx must NOT reproduce NONMEM's steady-state record; got {ferx_vs_nm_ss:.3e}"
    );
}

/// The **certifying control**: with the `TAD` term removed, lag + `SS=1` is exact on both
/// engines and in both tools. Without this, a reader cannot tell whether the `TAD` arm's
/// agreement is about the referent or about the #1121 seed geometry underneath it.
///
/// One-variable experiment: `ss_lag_auto_fit.ferx` differs from `ss_tadlag_fit.ferx` by the
/// single factor `(1.0 + 0.03*TAD)`, and the datasets are byte-identical apart from their names.
#[test]
fn the_autonomous_lagged_pair_certifies_the_lag_machinery() {
    let m = model("ss_lag_auto_fit.ferx");
    let ss = ferx_event_driven(&m, &population("ss_lag_auto.csv"));
    let train = ferx_event_driven(&m, &population("train_lag_auto.csv"));

    // NONMEM's SS record agrees with NONMEM's train here — which is exactly what fails once
    // the RHS reads TAD.
    let nm = worst_rel_all(
        &nonmem_ipred("ss_lag_auto"),
        &nonmem_ipred("train_lag_auto"),
        "NONMEM autonomous SS vs train",
    );
    assert!(
        nm < 1e-8,
        "on an autonomous RHS NONMEM's own lagged SS and lagged train agree to 1.14e-9; got \
         {nm:.3e}. If this fails, the lag geometry itself is in question and no conclusion \
         about the TAD referent follows"
    );

    let vs_own = worst_rel_all(&ss, &train, "ferx autonomous SS vs its own train");
    assert!(vs_own < 1e-9, "realised 2.74e-11; got {vs_own:.3e}");
    let vs_closed = worst_rel_external(
        &ss,
        &closed_form_rows("ss_lag_auto.csv", 3.0, 0.0),
        "autonomous vs closed form",
    );
    assert!(vs_closed < 5e-9, "realised 2.39e-10; got {vs_closed:.3e}");
}

/// A periodic limit exists for the **lagged** `TAD` train, so the run-in has a target.
///
/// `train_tadlag_n21` is a 21-dose train ending at 240, observed at the same cycle phases
/// (241/242/245/248/251 ≡ 481/482/485/488/491 mod 12), so this measures convergence and not a
/// change of observation point.
#[test]
fn a_periodic_limit_exists_for_the_lagged_tad_train() {
    let short = nonmem_ipred("train_tadlag_n21");
    let long = nonmem_ipred("train_tadlag");
    assert_eq!(short.len(), long.len());
    let mut worst = 0.0_f64;
    for (&(ts, s), &(tl, l)) in short.iter().zip(&long) {
        assert!(
            ((ts - 240.0) - (tl - 480.0)).abs() < 1e-9,
            "phase misalignment: {ts} against {tl}"
        );
        assert!(s.is_finite() && l.is_finite() && l != 0.0);
        worst = worst.max((s - l).abs() / l.abs());
    }
    assert!(
        worst < 1e-6,
        "the lagged TAD train must converge over a doubling; realised 7.09e-7, got {worst:.3e}"
    );
    assert!(
        worst > 0.0,
        "bit-identical would mean the two tables are the same file"
    );
}

/// The **discriminator**: an inert `0.0*TAD` must return the autonomous steady state exactly.
///
/// It proves the anchor is genuinely plumbed rather than accidentally cancelling — the same
/// check T2 uses for the un-lagged case, and the reason `0.0 * NaN = NaN` made merely
/// *mentioning* `TAD` break the model before #1139. Run on all three geometries, because each
/// reaches a different set of `solve_ode` windows.
#[test]
fn a_zero_coefficient_tad_term_returns_the_autonomous_steady_state() {
    for (twin, csv, tag) in [
        ("ss_tadlag_fit.ferx", "ss_tadlag.csv", "bolus lag=3"),
        (
            "ss_inf_tadlag_fit.ferx",
            "ss_inf_tadlag.csv",
            "infusion lag=3 (active + quiet windows)",
        ),
        (
            "ss_inf_tadlag_resid_fit.ferx",
            "ss_inf_tadlag_resid.csv",
            "infusion lag=10 (residual infusion at the record)",
        ),
    ] {
        let src = std::fs::read_to_string(anchor(twin)).expect("twin");
        // Assert both substitutions actually matched. If a fixture edit made BOTH of them
        // no-ops, `inert` and `autonomous` would be the same unmodified model, `worst` would
        // be 0.0, and the assertion below would pass while comparing a model with itself —
        // and this is the file's only plumbing discriminator, so nothing else would notice.
        let inert_src = src.replace("0.03*TAD", "0.0*TAD");
        let autonomous_src = src.replace(" * (1.0 + 0.03*TAD)", "");
        assert_ne!(inert_src, src, "{tag}: `0.03*TAD` not found in {twin}");
        assert_ne!(
            autonomous_src, src,
            "{tag}: ` * (1.0 + 0.03*TAD)` not found in {twin}"
        );
        let inert = parse_full_model(&inert_src).expect("parses").model;
        let autonomous = parse_full_model(&autonomous_src).expect("parses").model;
        let pop = population(csv);
        let a = ferx_event_driven(&inert, &pop);
        let b = ferx_event_driven(&autonomous, &pop);
        let worst = worst_rel_all(&a, &b, tag);
        assert!(
            worst < 1e-10,
            "{tag}: `0.0*TAD` must equal the same model with the term removed; got {worst:.3e}. \
             A NaN anchor here would make both sides NaN and `is_finite` inside `worst_rel` is \
             what catches that"
        );
    }
}

/// The **infusion** arms, which a bolus fixture structurally cannot reach.
///
/// `ss_state_at_phase_pk` has three `solve_ode` windows and a bolus fixture exercises exactly
/// one. `lag = 3` (`phase = 9 > T_inf = 4`) reaches the **active** window and the **quiet** one
/// that must be anchored at `−T_inf`; `lag = 10` (`phase = 2 < T_inf`) reaches the active window
/// with the previous cycle's infusion still running across the record.
///
/// The `0.0*TAD` discriminator above cannot see the quiet window's `−T_inf` — mutate it to `0`
/// and both sides stay TAD-independent — so this arm needs a **value** oracle, and it has one:
/// `lag < II`, so the explicit lagged infusion train reproduces the steady-state geometry.
#[test]
fn lagged_ss_infusions_land_on_their_explicit_infusion_trains() {
    for (twin, ss_csv, tr_csv, nm_train, arrival, bound, realised) in [
        (
            "ss_inf_tadlag_fit.ferx",
            "ss_inf_tadlag.csv",
            "train_inf_tadlag.csv",
            "train_inf_tadlag",
            483.0,
            2e-8,
            "5.8287e-9",
        ),
        (
            "ss_inf_tadlag_resid_fit.ferx",
            "ss_inf_tadlag_resid.csv",
            "train_inf_tadlag_resid.csv",
            "train_inf_tadlag_resid",
            490.0,
            1e-8,
            "2.4196e-9",
        ),
    ] {
        let m = model(twin);
        let ss = ferx_event_driven(&m, &population(ss_csv));
        let own_train = ferx_event_driven(&m, &population(tr_csv));

        let vs_nm = worst_rel_external(&ss, &nonmem_ipred(nm_train), twin);
        assert!(
            vs_nm < bound,
            "{twin}: ferx SS vs NONMEM's explicit lagged infusion train {vs_nm:.3e} \
             (realised {realised}, bound {bound:.0e} — between 3x and 5x headroom, set \
             from the realised error rather than picked round)"
        );
        let vs_own = worst_rel_all(&ss, &own_train, twin);
        assert!(vs_own < 1e-8, "{twin}: vs ferx's own train {vs_own:.3e}");

        // The pre-arrival window on its own: for `lag = 3` that is the quiet window's
        // contribution, for `lag = 10` the residual infusion's.
        let pre = worst_rel(&ss, &nonmem_ipred(nm_train), twin, |t| t < arrival);
        assert!(
            pre < bound,
            "{twin}: inside the pre-arrival window {pre:.3e}"
        );
    }
}

/// The `lag ≥ II` **clamp**, which has no train twin — and that is measured, not a shortcut.
///
/// `ss_seed_phase` clamps the phase to `0` there (the pulse lands on the record and nothing
/// intervenes before the real arrival), while an explicit train's pulses keep their 12 h
/// spacing and put an arrival at 483. The two geometries genuinely differ, which is #1121's
/// finding, so the train cannot be the oracle here. The **closed form** can: the compartment
/// carries `trough + D` at the record and decays under `TAD = t − 480` for a full `lag`, with
/// no pulse in between.
///
/// The phase itself is already NONMEM-anchored by #1121
/// (`nonmem_anchor/results/ss_lag_ge_ii.tab`); what is new here is that `TAD` follows it.
#[test]
fn a_lagtime_of_a_full_interval_or_more_follows_the_clamped_phase() {
    let src = std::fs::read_to_string(anchor("ss_tadlag_fit.ferx")).expect("twin");
    let m = parse_full_model(&src.replace("TVLAG(3.0, FIX)", "TVLAG(15.0, FIX)"))
        .expect("parses")
        .model;
    let pop = population("ss_tadlag.csv");
    let got = ferx_event_driven(&m, &pop);

    // Closed form for the clamped geometry: `τ = t − 480` on `[480, 495)`.
    let a0 = trough(K, C, D, II) + D;
    let want: Vec<(f64, f64)> = pop.subjects[0]
        .obs_times
        .iter()
        .map(|&t| {
            assert!(t < 495.0, "every observation must sit inside [480, 495)");
            (t, a0 * phi(t - 480.0, K, C) / V)
        })
        .collect();
    let worst = worst_rel_external(&got, &want, "lag >= II clamp vs closed form");
    assert!(
        worst < 5e-9,
        "realised 4.0869e-10 across 481/482/485/488/491 — 12x headroom; got {worst:.3e}"
    );

    // And it is genuinely a different answer from the un-clamped wrap — otherwise this test
    // would pass against an implementation that ignored the clamp entirely.
    let wrapped = closed_form(481.0, 15.0, C);
    let clamped = got[0].1;
    assert!(
        (clamped - wrapped).abs() / wrapped > 0.1,
        "the clamped and wrapped geometries must differ materially at t = 481 ({clamped:.6} vs \
         {wrapped:.6}); if they agree, this fixture cannot tell `ss_seed_phase` from `rem_euclid`"
    );
}

/// `TAFD` under a lagged `SS=1` dose reads `NaN`, and that is now a **decision**, not an open
/// question. An absolute clock has no periodic limit for a run-in to converge to (measured:
/// 0.294 per doubling, against `TAD`'s 3.5e-7), so a finite plausible number there would be
/// `TAD` under another name.
///
/// Written on T4 as a scope pin against a change quietly extending the referent to `TAFD`
/// without doing T3's work. T3 (#1139) has since landed and chose to move no value: the
/// `NaN` stays and `W_STEADY_STATE_ABSOLUTE_TIME` names it, so this test kept its behaviour
/// and gained a reason. The un-lagged twin lives in `tests/ss_model_time_nonmem_anchor.rs`
/// (`tafd_under_an_unlagged_ss_dose_stays_nan_while_tad_does_not`), which also asserts the
/// asymmetry against `TAD`.
#[test]
fn tafd_stays_nan_under_a_lagged_ss_dose() {
    let src = std::fs::read_to_string(anchor("ss_tadlag_fit.ferx")).expect("twin");
    let m = parse_full_model(&src.replace("0.03*TAD", "0.003*TAFD"))
        .expect("parses")
        .model;
    let got = ferx_event_driven(&m, &population("ss_tadlag.csv"));
    // `all()` on an empty iterator is `true`: without this the test passes for a fixture
    // that produced no predictions at all, which is not the property it claims to pin.
    assert!(
        !got.is_empty(),
        "the lagged SS fixture produced no predictions to check"
    );
    assert!(
        got.iter().all(|(_, v)| v.is_nan()),
        "TAFD under SS reads NaN by decision (#1139 T3); it must still. Got {got:?}"
    );
}

/// `simulate()` and `predict()` reach the fix too — measured, not inferred (#1126 review).
///
/// Both are separate public entry points from the objective path this file otherwise
/// exercises, and the CHANGELOG makes a user-facing claim about them. T2 (#1270) measured
/// `simulate()` on the un-lagged case rather than asserting it, for the same reason: the
/// steady-state family has several entry points and "it plausibly routes through the fixed
/// predictor" is a code read, not a result. Before #1126 both returned `NaN` here.
///
/// `predict()` is at η = 0 by construction; `simulate()` is seeded and draws residual error,
/// so its `ipred` — the individual prediction *before* error — is what is comparable, and
/// `omega` is empty in this twin so its η is zero too.
#[test]
fn simulate_and_predict_reach_the_pre_arrival_referent() {
    let m = model("ss_tadlag_fit.ferx");
    let pop = population("ss_tadlag.csv");
    let want = nonmem_ipred("train_tadlag");

    let preds = ferx_core::api::predict(&m, pop, &m.default_params);
    assert_eq!(preds.len(), want.len(), "predict() row count");
    let mut worst_pred = 0.0_f64;
    for (p, &(t, w)) in preds.iter().zip(&want) {
        assert!(
            (p.time - t).abs() < 1e-9,
            "predict() time {} vs {t}",
            p.time
        );
        assert!(p.pred.is_finite(), "predict() returned {} at t={t}", p.pred);
        worst_pred = worst_pred.max((p.pred - w).abs() / w.abs());
    }
    assert!(
        worst_pred < 1e-7,
        "predict() vs NONMEM's lagged train: {worst_pred:.3e} (realised 1.69e-8, 6x headroom)"
    );

    let sims = ferx_core::api::simulate_with_seed(&m, pop, &m.default_params, 1, 7);
    let ipreds: Vec<f64> = sims.iter().map(|r| r.ipred).collect();
    assert_eq!(ipreds.len(), want.len(), "simulate() row count");
    let mut worst_sim = 0.0_f64;
    for (&got, &(t, w)) in ipreds.iter().zip(&want) {
        assert!(got.is_finite(), "simulate() ipred is {got} at t={t}");
        worst_sim = worst_sim.max((got - w).abs() / w.abs());
    }
    assert!(
        worst_sim < 1e-7,
        "simulate() ipred vs NONMEM's lagged train: {worst_sim:.3e}"
    );
}

/// The sdtab `TAD` column reads the same referent the integrator does.
///
/// Before #1126 this cell was **blank** — `tad_at_time` had no candidate inside the window and
/// folded to `NaN` — while the predictors were integrating under an anchor of their own. Both
/// now come from `crate::dosing::tad_referent`, so the reported column and the injected value
/// cannot disagree, and the `SS=1` record reports the same column as the explicit train.
#[test]
fn the_sdtab_tad_column_reads_the_pre_arrival_referent() {
    let want = [
        (481.0, 10.0), // inside the window: the previous cycle's pulse is at 480 + 3 - 12 = 471
        (482.0, 11.0),
        (485.0, 2.0), // past the arrival at 483
        (488.0, 5.0),
        (491.0, 8.0),
    ];
    for csv in ["ss_tadlag.csv", "train_tadlag.csv"] {
        let pop = population(csv);
        let s = &pop.subjects[0];
        let lags = vec![3.0; s.doses.len()];
        for (j, (t, w)) in want.iter().enumerate() {
            let (_, tad) = ferx_core::api::tafd_tad_for_subject(s, j, &lags);
            assert_eq!(s.obs_times[j], *t, "{csv}: observation order");
            assert!(
                (tad - w).abs() < 1e-9,
                "{csv} t={t}: TAD column reads {tad}, want {w}"
            );
        }
    }
}
