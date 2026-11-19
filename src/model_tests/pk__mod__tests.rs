#[allow(unused_imports)]
use ferx_model::internal::*;
// --- the imports of the original parent module (relocated test, #3041) ---
use ferx_model::pk::one_compartment::*;
use ferx_model::pk::three_compartment::*;
use ferx_model::pk::two_compartment::*;
use ferx_model::types::CompiledModel;
use ferx_model::types::DoseEvent;
use ferx_model::types::PkModel;
use ferx_model::types::PkParams;
use ferx_model::types::ScalingSpec;
use ferx_model::types::Subject;

/// #1564: the IOV record memo reuses a parameter evaluation exactly when every input
/// the program reads repeats — occasion, covariate map (by address), and the time unless
/// the program cannot read `TIME` — and evaluates otherwise.
#[test]
fn record_param_memo_reuses_only_identical_inputs() {
    use std::cell::Cell;
    use std::collections::HashMap;
    let calls = Cell::new(0usize);
    let eval = |v: f64| {
        calls.set(calls.get() + 1);
        let mut p = PkParams::default();
        p.values[0] = v;
        p
    };
    let (a, b): (HashMap<String, f64>, HashMap<String, f64>) = (HashMap::new(), HashMap::new());
    let mut memo = RecordParamMemo { time_free: true, last: None, value: PkParams::default() };
    assert_eq!(memo.get_ref(0, &a, 1.0, || eval(1.0)).values[0], 1.0);
    // Same occasion and map, another time, on a time-free program: reused.
    assert_eq!(memo.get_ref(0, &a, 2.0, || eval(9.0)).values[0], 1.0);
    assert_eq!(calls.get(), 1);
    // Another occasion, another map (even with equal contents): evaluated.
    assert_eq!(memo.get_ref(1, &a, 2.0, || eval(2.0)).values[0], 2.0);
    assert_eq!(memo.get_ref(1, &b, 2.0, || eval(3.0)).values[0], 3.0);
    assert_eq!(calls.get(), 3);
    // A program that reads `TIME` keys on the time too.
    let mut timed = RecordParamMemo { time_free: false, last: None, value: PkParams::default() };
    timed.get_ref(0, &a, 1.0, || eval(4.0));
    assert_eq!(timed.get_ref(0, &a, 2.0, || eval(5.0)).values[0], 5.0);
    assert_eq!(timed.get_ref(0, &a, 2.0, || eval(6.0)).values[0], 5.0);
    assert_eq!(calls.get(), 5);
}

use approx::assert_relative_eq;
use ferx_model::pk::*;
use std::collections::HashMap;

fn bolus_dose(time: f64, amt: f64) -> DoseEvent {
    DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
}

fn make_pk_params(cl: f64, v: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[0] = cl;
    p.values[1] = v;
    p
}

// ── records with the same `$PK` inputs share one evaluation ──

fn covs(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
}

/// Each row names the regression it catches: a key that leaves out a covariate the
/// program reads, its group or its sign of zero shares two records that evaluate
/// differently. A key that reads the map's unreferenced entries, or compares
/// `NaN` by value, only loses sharing, and the rows for those pin that too.
#[test]
fn records_share_an_evaluation_only_when_every_pk_input_matches_bitwise() {
    let model = crate::parser::model_parser::parse_model_string(crate::test_models::ONE_CPT_IV_21)
        .expect("parses");
    assert_eq!(model.referenced_covariates, ["AGE", "WT"]);
    let base = covs(&[("WT", 70.0), ("AGE", 40.0)]);
    let maps = [
        /* 0 */ base.clone(),
        /* 1 equal values, another map */ base.clone(),
        /* 2 an unreferenced column differs */
        covs(&[("WT", 70.0), ("AGE", 40.0), ("FOO", 1.0)]),
        /* 3 */ covs(&[("WT", 95.0), ("AGE", 40.0)]),
        /* 4 back to the first, not adjacent */ base.clone(),
        /* 5 */ covs(&[("WT", f64::NAN), ("AGE", 40.0)]),
        /* 6 NaN matches NaN by bits */ covs(&[("WT", f64::NAN), ("AGE", 40.0)]),
        /* 7 */ covs(&[("WT", 0.0), ("AGE", 40.0)]),
        /* 8 -0 is not 0 */ covs(&[("WT", -0.0), ("AGE", 40.0)]),
        /* 9 WT missing is not WT = 0 */ covs(&[("AGE", 40.0)]),
        /* 10 */ covs(&[("AGE", 40.0)]),
        /* 11 the other covariate differs */ covs(&[("WT", 70.0), ("AGE", 41.0)]),
    ];
    let mut records: Vec<(Option<usize>, &HashMap<String, f64>)> =
        maps.iter().map(|m| (None, m)).collect();
    // 12, 13: the first record's inputs in another κ group; 14: in no group again.
    records.extend([(Some(1), &maps[0]), (Some(1), &maps[1])]);
    records.push((None, &maps[1]));
    // 15, 16: one map borrowed by consecutive records, as `subject.covariates` is.
    records.extend([(None, &maps[3]), (None, &maps[3])]);
    assert_eq!(
        first_record_with_same_pk_inputs(&model, records.iter().copied()),
        Some(vec![0, 0, 0, 3, 0, 5, 5, 7, 8, 9, 9, 11, 12, 12, 0, 3, 3])
    );
}

/// The straddle: one program with and without its `TIME` read, on records whose
/// covariates and times all agree. Without `TIME` they share; with it there is no
/// index at all, so each is its own, even the two at the same time.
#[test]
fn a_model_reading_time_shares_nothing() {
    let src = |cl: &str| {
        format!(
            r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[individual_parameters]
  CL = {cl}
  V  = 10 * (WT/70)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(EPS)
"#
        )
    };
    let parse =
        |cl: &str| crate::parser::model_parser::parse_model_string(&src(cl)).expect("parses");
    let with_time = parse("TVCL * exp(ETA_CL) * (1 + 0.01 * TIME)");
    let without = parse("TVCL * exp(ETA_CL)");
    assert!(crate::parser::model_parser::compiled_model_uses_time_builtin(&with_time));
    assert!(!crate::parser::model_parser::compiled_model_uses_time_builtin(&without));
    let m = covs(&[("WT", 70.0)]);
    let records = [(None, &m), (None, &m), (Some(0), &m), (Some(0), &m)];
    assert_eq!(
        first_record_with_same_pk_inputs(&without, records.iter().copied()),
        Some(vec![0, 0, 2, 2])
    );
    assert_eq!(first_record_with_same_pk_inputs(&with_time, records.iter().copied()), None);
}

/// A hand-built `pk_param_fn` has no program whose inputs the key could list, so it
/// is evaluated at every record however alike they look.
#[test]
fn a_model_without_an_individual_parameter_program_shares_nothing() {
    let model = cl_from_cr_model();
    assert!(model.indiv_param_partials.indiv_param_program().is_none());
    let m = covs(&[("CR", 1.0)]);
    let records = [(None, &m), (None, &m), (Some(0), &m)];
    assert_eq!(first_record_with_same_pk_inputs(&model, records.iter().copied()), None);
}

/// The per-event materializer copies a snapshot from the record it shares with, in
/// whichever list that record sits. Every snapshot must still be the bits of `$PK`
/// evaluated at that record's own inputs, which this compares against directly. The
/// fixture shares within a list, across lists (a reset and an EVID=2 row reusing an
/// observation's), and non-adjacently, and asserts that it does.
#[test]
fn shared_per_event_snapshots_are_each_records_own_evaluation_bitwise() {
    let model = crate::parser::model_parser::parse_model_string(crate::test_models::ONE_CPT_IV_21)
        .expect("parses");
    let w = |wt: f64| covs(&[("WT", wt), ("AGE", 40.0)]);
    let mut s = make_subject_with_tv(w(70.0), vec![w(70.0), w(95.0)], Vec::new(), 2, 5);
    s.obs_covariates = vec![w(70.0), w(95.0), w(95.0), w(80.0), w(70.0)];
    s.pk_only_times = vec![3.5, 4.5];
    s.pk_only_covariates = vec![w(80.0), w(60.0)];
    s.reset_times = vec![2.5];
    s.reset_covariates = vec![w(95.0)];
    assert!(subject_needs_per_event_pk(&model, &s));
    let theta = model.default_params.theta.clone();
    let eta = [0.2];

    let ev = compute_event_pk_params(&model, &s, &theta, &eta);
    let own = |cov: &HashMap<String, f64>, t: f64| (model.pk_param_fn)(&theta, &eta, cov, t);
    let lists: [(&str, &[PkParams], Vec<PkParams>); 4] = [
        ("dose", &ev.dose, (0..2).map(|k| own(s.dose_cov(k), s.dose_param_time(k))).collect()),
        ("obs", &ev.obs, (0..5).map(|j| own(s.obs_cov(j), s.param_time_obs(j))).collect()),
        (
            "pk_only",
            &ev.pk_only,
            (0..2).map(|m| own(s.pk_only_cov(m), s.param_time_pk_only(m))).collect(),
        ),
        ("reset", &ev.reset, vec![own(s.reset_cov(0), s.param_time_at(s.reset_times[0]))]),
    ];
    for (name, got, want) in &lists {
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.values.map(f64::to_bits),
                w.values.map(f64::to_bits),
                "{name} {i}: the snapshot is not its own record's evaluation"
            );
        }
    }
    // The sharing the comparison relies on: across the obs, EVID=2 and reset rows
    // the materializer passes in one call, obs 4 reuses obs 0, EVID=2 row 0 reuses
    // obs 3 and the reset reuses obs 1. A distinct WT per record reduces this test
    // to the unshared one.
    let records: Vec<_> = (0..5)
        .map(|j| (None, s.obs_cov(j)))
        .chain((0..2).map(|m| (None, s.pk_only_cov(m))))
        .chain([(None, s.reset_cov(0))])
        .collect();
    assert_eq!(
        first_record_with_same_pk_inputs(&model, records),
        Some(vec![0, 1, 1, 3, 0, 3, 6, 1])
    );
    // And the shared snapshots differ, so a copy from the wrong record shows.
    assert_ne!(ev.obs[0].cl(), ev.obs[1].cl());
    assert_ne!(ev.obs[3].cl(), ev.pk_only[1].cl());
}

// ── the sharing index is built once per scope (#2558) ──

/// `CL` reads `WT` and `V` reads `AGE` when `wt`; otherwise `WT` is no input at all.
fn wt_age_model(wt: bool) -> CompiledModel {
    let cl = if wt { "TVCL * exp(ETA_CL) * (WT/70)" } else { "TVCL * exp(ETA_CL)" };
    crate::parser::model_parser::parse_model_string(&format!(
        r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma EPS ~ 0.1
[individual_parameters]
  CL = {cl}
  V  = TVV * (AGE/40)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ additive(EPS)
"#
    ))
    .expect("parses")
}

/// A subject whose `WT` changes from record to record over a constant `AGE`, with as
/// many resets as doses, so its dose list and its reset list are the same length.
fn wt_varying_subject(wts: [f64; 5]) -> Subject {
    let w = |wt: f64| covs(&[("WT", wt), ("AGE", 40.0)]);
    let mut s = make_subject_with_tv(
        w(wts[0]),
        vec![w(wts[0]), w(wts[1])],
        wts.iter().map(|&x| w(x)).collect(),
        2,
        5,
    );
    s.reset_times = vec![2.5, 3.5];
    s.reset_covariates = vec![w(wts[2]), w(wts[3])];
    s
}

/// Every snapshot of `ev`, as bits.
fn snapshot_bits(ev: &EventPkParams) -> Vec<Vec<u64>> {
    [&ev.dose, &ev.obs, &ev.pk_only, &ev.reset]
        .iter()
        .flat_map(|l| l.iter().map(|p| p.values.map(f64::to_bits).to_vec()))
        .collect()
}

/// Outside a scope each evaluation builds both of its indices; inside one the first
/// evaluation builds them and every later one reuses them, at any η, and the
/// snapshots are the same bits either way. Catches an EBE search whose scope stops
/// holding the index, and a reused index that is not the one a fresh build gives.
#[test]
fn a_sharing_scope_builds_each_index_once_and_changes_no_snapshot() {
    let model = wt_age_model(true);
    let s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
    assert!(subject_needs_per_event_pk(&model, &s));
    let theta = model.default_params.theta.clone();
    let builds = sharing_scope::builds;
    let eval = |eta: f64| snapshot_bits(&compute_event_pk_params(&model, &s, &theta, &[eta]));

    let b0 = builds();
    let unscoped = [eval(0.2), eval(-0.3)];
    assert_eq!(builds() - b0, 4, "outside a scope: dose + rest index per evaluation");
    let scoped = {
        let _share = sharing_scope::SharingScope::enter(&model, &s);
        let b1 = builds();
        let scoped = [eval(0.2), eval(-0.3), eval(0.2)];
        assert_eq!(builds() - b1, 2, "inside a scope: dose + rest index, once");
        scoped
    };
    assert_eq!(scoped[0], unscoped[0], "η = 0.2: a reused index moved a snapshot");
    assert_eq!(scoped[1], unscoped[1], "η = -0.3: a reused index moved a snapshot");
    assert_eq!(scoped[2], unscoped[0]);
    // Closed, the scope keeps nothing.
    let b2 = builds();
    eval(0.2);
    assert_eq!(builds() - b2, 2, "a closed scope must not keep serving its index");
}

/// The scope is for one model, one subject, one site and one set of κ groups, and an
/// index for anything else is built and handed back but not kept, so the scope's own
/// still answers afterwards. Each pair below differs in one of those and has a
/// different index, asserted, so a key that drops it hands back the wrong one.
#[test]
fn a_sharing_scope_rebuilds_for_another_model_subject_site_or_grouping() {
    use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite};
    let (with_wt, age_only) = (wt_age_model(true), wt_age_model(false));
    let s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
    let other = wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]);
    let builds = sharing_scope::builds;
    fn none(s: &Subject) -> Vec<(Option<usize>, &HashMap<String, f64>)> {
        (0..s.obs_times.len()).map(|j| (None, s.obs_cov(j))).collect()
    }
    let direct = |m: &CompiledModel, r: &[(Option<usize>, &HashMap<String, f64>)]| {
        first_record_with_same_pk_inputs(m, r.iter().copied()).expect("parsed, no TIME")
    };
    let get = |m: &CompiledModel, sub: &Subject, site, r: &[_]| {
        scoped(m, sub, site, r.iter().copied()).expect("parsed, no TIME").to_vec()
    };
    let (rs, ro) = (none(&s), none(&other));
    // The same records in other κ groups: obs 1 and 2 now share only within a group.
    let grouped: Vec<_> = rs.iter().enumerate().map(|(j, &(_, c))| (Some(j % 2), c)).collect();
    let want = direct(&with_wt, &rs);
    assert_eq!(want, [0, 1, 1, 3, 0]);
    assert_ne!(direct(&age_only, &rs), want, "the models must index s differently");
    assert_ne!(direct(&with_wt, &ro), want, "the subjects must index differently");
    assert_ne!(direct(&with_wt, &grouped), want, "the groupings must index differently");

    let _share = sharing_scope::SharingScope::enter(&with_wt, &s);
    assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
    let b = builds();
    assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
    assert_eq!(builds(), b, "the scope's own index is reused");

    assert_eq!(
        get(&age_only, &s, ShareSite::Recorder, &rs),
        direct(&age_only, &rs),
        "another model (as the evaluation twin is) builds its own index"
    );
    assert_eq!(
        get(&with_wt, &other, ShareSite::Recorder, &ro),
        direct(&with_wt, &ro),
        "another subject builds its own index"
    );
    assert_eq!(builds(), b + 2);
    let b = builds();
    assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
    assert_eq!(builds(), b, "neither of those replaced the scope's index");

    // Another site's list, the same length with the same (absent) groups, as the
    // recorder's and the f64 materializer's lists can be on a model without IOV.
    assert_eq!(rs.len(), ro.len());
    assert_eq!(
        get(&with_wt, &s, ShareSite::ObsPkOnlyReset, &ro),
        direct(&with_wt, &ro),
        "another site keeps its own index"
    );
    let b = builds();
    assert_eq!(get(&with_wt, &s, ShareSite::Recorder, &rs), want);
    assert_eq!(builds(), b, "and leaves the recorder's alone");

    assert_eq!(
        get(&with_wt, &s, ShareSite::Recorder, &grouped),
        direct(&with_wt, &grouped),
        "other κ groups rebuild"
    );
}

/// A scope opened inside another hands the outer one back when it closes; one opened
/// for the pair already open changes nothing. Catches a guard that leaves its own
/// state behind, which costs the outer search its index for the rest of the search.
#[test]
fn a_nested_sharing_scope_restores_the_outer_one() {
    use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite, SharingScope};
    let model = wt_age_model(true);
    let (s, other) = (
        wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]),
        wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]),
    );
    let builds = sharing_scope::builds;
    let ask = |sub: &Subject| {
        let r: Vec<_> = (0..5).map(|j| (None, sub.obs_cov(j))).collect();
        scoped(&model, sub, ShareSite::Dose, r.into_iter());
    };
    let _outer = SharingScope::enter(&model, &s);
    ask(&s);
    let b = builds();
    {
        let _inner = SharingScope::enter(&model, &other);
        ask(&other);
        ask(&other);
        assert_eq!(builds(), b + 1, "the inner scope keeps its own index");
    }
    let b = builds();
    ask(&s);
    assert_eq!(builds(), b, "the outer scope's index survives the inner one");
    {
        let _again = SharingScope::enter(&model, &s);
        ask(&s);
    }
    ask(&s);
    assert_eq!(builds(), b, "re-entering the open pair keeps its index");
}

/// A guard that closes while one opened after it is still open must not hand back its
/// own predecessor: the later guard would then reinstall a scope whose borrows have
/// ended. A debug build fails at the out-of-order drop, and every scope is closed, so a
/// release build only loses the index. Catches a `Drop` that restores unconditionally.
#[test]
#[cfg(debug_assertions)]
fn an_out_of_order_sharing_scope_closes_every_scope() {
    use sharing_scope::{first_record_with_same_pk_inputs as scoped, ShareSite, SharingScope};
    let model = wt_age_model(true);
    let (s, other) = (
        wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]),
        wt_varying_subject([70.0, 70.0, 95.0, 95.0, 95.0]),
    );
    let ask = |sub: &Subject| {
        let r: Vec<_> = (0..5).map(|j| (None, sub.obs_cov(j))).collect();
        scoped(&model, sub, ShareSite::Dose, r.into_iter());
    };
    let closed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let first = SharingScope::enter(&model, &s);
        let second = SharingScope::enter(&model, &other);
        drop(first);
        drop(second);
    }));
    let msg = closed.expect_err("an out-of-order drop must fail a debug build");
    assert!(
        msg.downcast_ref::<&str>().is_some_and(|m| m.contains("closed before one opened after it")),
        "the panic names the out-of-order close"
    );
    // Nothing is left open: neither subject's index is kept.
    for sub in [&s, &other] {
        let b = sharing_scope::builds();
        ask(sub);
        ask(sub);
        assert_eq!(sharing_scope::builds(), b + 2, "no scope may outlive its guard");
    }
}

/// An EBE search opens the scope: across all its evaluations each site builds once.
/// Without it a search of two evaluations already builds four f64 indices.
#[test]
fn an_ebe_search_builds_each_sharing_index_once() {
    let model = wt_age_model(true);
    let mut s = wt_varying_subject([70.0, 95.0, 95.0, 80.0, 70.0]);
    s.observations = vec![9.0, 6.0, 4.0, 3.0, 2.0];
    let b = sharing_scope::builds();
    let ebe = crate::estimation::inner_optimizer::find_ebe(
        &model,
        &s,
        &model.default_params,
        200,
        1e-8,
        None,
        None,
        0,
    );
    assert!(ebe.nll.is_finite());
    // Exactly one per site, so a site the search stops reaching fails as surely as a site
    // that rebuilds. This closed form's objective values come from the compiled plan
    // (`plan_value_route`), not the f64 predictor, so the f64 dose and obs/PK-only/reset
    // indices are never built: the one left is the recorder's (measured: 3 before the
    // plan served the value-only objective, 1 after).
    let built = sharing_scope::builds() - b;
    assert_eq!(
        built, 1,
        "indices built across one EBE search: want the recorder's index only, once \
         (the plan serves the objective's values, so no f64 index is built)"
    );
}

// ── the doses-only materializer is the full one's `dose` vector (#1235) ──

/// `compute_dose_pk_params_into` must be **bit-identical** to
/// `compute_event_pk_params(..).dose` — it exists only to skip the observation
/// snapshots a doses-only caller never reads, never to answer differently.
///
/// Both arms of the gate are exercised, and the subjects below straddle it —
/// **asserted, not assumed**, so this cannot silently become a one-arm test.
///
/// **What this can and cannot see, measured rather than assumed.** The two callers
/// share `fill_dose_pk_params`, so a mutation *inside the shared body* moves both
/// sides of the comparison together and a bit-equality test is blind to it by
/// construction — the tautological-twin shape. Running the three mutations:
///
/// | mutation | this test | `tests/typical_value_snapshot_sets.rs` |
/// |---|---|---|
/// | shared body reads `&subject.covariates` | **dies** — on the `assert_ne!` below, not the bit compare | arm A + both `D`/`R` arms |
/// | shared body reads `0.0` for the time | **survives** — a true tautology on that axis | arm B |
/// | the two callers diverge (doses-only always freezes) | **dies** | arms A, B + both `D`/`R` |
///
/// So the `assert_ne!` non-degeneracy check is not decoration: it is the only reason
/// row 1 is visible here at all. And row 2 is the honest limit — the time axis is
/// covered by arm B, not by this test. This is not the sole guard against row 3
/// either; it is the Tier-1 statement of the contract at the code's own site, where
/// `cargo test --lib` reaches it.
#[test]
fn dose_only_snapshots_match_the_full_event_builder() {
    let src = r#"
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP_ERR ~ 0.15 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL) * (WT/70)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP_ERR)
"#;
    let model = crate::parser::model_parser::parse_full_model(src)
        .expect("the parity fixture parses")
        .model;
    let theta = model.default_params.theta.clone();
    let eta = vec![0.1_f64; model.n_eta + model.n_kappa];

    let base = |doses: Vec<DoseEvent>| Subject {
        id: "1".into(),
        doses,
        obs_times: vec![1.0, 4.0, 26.0],
        observations: vec![1.0; 3],
        obs_cmts: vec![1; 3],
        covariates: HashMap::from([("WT".to_string(), 70.0)]),
        cens: vec![0; 3],
        ..Default::default()
    };

    let static_subject = base(vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)]);
    let mut tv_subject = base(vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)]);
    tv_subject.dose_covariates =
        vec![HashMap::from([("WT".to_string(), 70.0)]), HashMap::from([("WT".to_string(), 95.0)])];
    tv_subject.obs_covariates = vec![
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 70.0)]),
        HashMap::from([("WT".to_string(), 95.0)]),
    ];

    // The straddle itself: without it a later change to the gate could quietly put
    // both subjects on one arm and this would still pass.
    assert!(
        !subject_needs_per_event_pk(&model, &static_subject),
        "the static subject must take the constant arm"
    );
    assert!(
        subject_needs_per_event_pk(&model, &tv_subject),
        "the TV-covariate subject must take the per-event arm"
    );

    let mut doses_only = Vec::new();
    for (arm, subject) in [("static", &static_subject), ("per-event", &tv_subject)] {
        let full = compute_event_pk_params(&model, subject, &theta, &eta);
        compute_dose_pk_params_into(&model, subject, &theta, &eta, &mut doses_only);
        assert_eq!(
            doses_only.len(),
            full.dose.len(),
            "{arm} arm: dose snapshot count must match the full builder"
        );
        for (k, (a, b)) in doses_only.iter().zip(full.dose.iter()).enumerate() {
            assert_eq!(
                a.values.map(f64::to_bits),
                b.values.map(f64::to_bits),
                "{arm} arm: dose {k} snapshot differs from the full builder's"
            );
        }
    }

    // Non-degeneracy: on the per-event arm the two doses must actually *differ*,
    // or every bit comparison above is satisfied by a builder that froze the
    // snapshot — the exact defect #1235 is about.
    compute_dose_pk_params_into(&model, &tv_subject, &theta, &eta, &mut doses_only);
    assert_ne!(
        doses_only[0].values.map(f64::to_bits),
        doses_only[1].values.map(f64::to_bits),
        "the per-event fixture must resolve different params at its two doses"
    );
}

// ── analytical [initial_conditions] (issue #521) ──

#[test]
fn analytical_init_central_one_cpt_iv_is_bolus_decay() {
    // An initial amount A₀ in the central compartment of a 1-cpt model
    // decays as the IV-bolus impulse: C(t) = (A₀/V)·exp(-(CL/V)·t).
    let (cl, v, a0) = (2.0_f64, 10.0_f64, 500.0_f64);
    let p = make_pk_params(cl, v);
    for &t in &[0.0_f64, 1.0, 5.0, 24.0] {
        let got = analytical_init_concentration(PkModel::OneCptIv, 1, a0, t, &p);
        let expected = (a0 / v) * (-(cl / v) * t).exp();
        assert_relative_eq!(got, expected, max_relative = 1e-12);
    }
}

#[test]
fn analytical_init_central_oral_uses_cmt2_and_ignores_ka() {
    // For an oral model, `central` is cmt 2 and the init is a direct IV
    // bolus into central — KA must not enter (the amount is already there,
    // not absorbed through the depot). So it matches the 1-cpt decay and is
    // independent of KA.
    let (cl, v, a0) = (3.0_f64, 20.0_f64, 250.0_f64);
    let mut p = make_pk_params(cl, v);
    p.values[crate::types::PK_IDX_KA] = 1.7;
    let mut p_other_ka = p;
    p_other_ka.values[crate::types::PK_IDX_KA] = 0.3;
    let t = 4.0;
    let got = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p);
    let got_other = analytical_init_concentration(PkModel::OneCptOral, 2, a0, t, &p_other_ka);
    let expected = (a0 / v) * (-(cl / v) * t).exp();
    assert_relative_eq!(got, expected, max_relative = 1e-12);
    assert_relative_eq!(got, got_other, max_relative = 1e-12);
}

#[test]
fn analytical_init_depot_oral_absorbs_through_ka() {
    // An init amount pre-loaded in the depot (cmt 1) of an oral model is
    // absorbed first-order with F=1, i.e. the oral closed form with D=A₀.
    let (cl, v, ka, a0) = (3.0_f64, 20.0_f64, 1.1_f64, 100.0_f64);
    let mut p = make_pk_params(cl, v);
    p.values[crate::types::PK_IDX_KA] = ka;
    let t = 2.0;
    let got = analytical_init_concentration(PkModel::OneCptOral, 1, a0, t, &p);
    let expected = crate::sens::one_cpt::one_cpt_oral_g::<f64>(a0, t, cl, v, ka, 1.0);
    assert_relative_eq!(got, expected, max_relative = 1e-12);
    // At t=0 nothing has absorbed yet → zero central concentration.
    let at0 = analytical_init_concentration(PkModel::OneCptOral, 1, a0, 0.0, &p);
    assert_relative_eq!(at0, 0.0, epsilon = 1e-12);
}

#[test]
fn iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form() {
    // NONMEM anchor for #327/#419. With bioavailability F1 on an IV dose,
    // NONMEM delivers F1·AMT, so for a 1-cpt model (ADVAN1/TRANS2,
    // `F1 = THETA(3)` in $PK) the closed forms are:
    //   bolus:    C(t) = F·Dose/V · exp(-k·t)
    //   infusion: rate held at R, duration scaled to T_F = F·AMT/R (#419):
    //             C(t) = R/CL · (1 − exp(-k·t))                   (t ≤ T_F)
    //             C(t) = R/CL · (1 − exp(-k·T_F)) · exp(-k·(t−T_F)) (t > T_F)
    // For a *rate-defined* infusion NONMEM keeps the rate and scales the
    // duration (total exposure still F·AMT), so there is no amplitude factor
    // on the rate. The analytical superposition path (`predict_concentration`)
    // must reproduce these.
    let (cl, v, f) = (5.0_f64, 50.0_f64, 0.4_f64);
    let k = cl / v;
    let mut pk = make_pk_params(cl, v);
    pk.values[crate::types::PK_IDX_F] = f;

    // IV bolus: F scales the amount (unchanged by #419).
    let amt = 100.0;
    let doses = vec![bolus_dose(0.0, amt)];
    for &t in &[0.25_f64, 1.0, 4.0, 12.0] {
        let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
        let want = f * amt / v * (-k * t).exp();
        assert_relative_eq!(got, want, max_relative = 1e-12);
    }

    // IV infusion: R=25, raw T = AMT/R = 4 h; under F the duration scales to
    // T_F = F·AMT/R = 1.6 h with the rate held at R.
    let rate = 25.0;
    let t_inf = f * amt / rate;
    let doses = vec![DoseEvent::new(0.0, amt, 1, rate, false, 0.0)];
    for &t in &[1.0_f64, 4.0, 8.0] {
        let got = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
        let want = if t <= t_inf {
            rate / cl * (1.0 - (-k * t).exp())
        } else {
            rate / cl * (1.0 - (-k * t_inf).exp()) * (-k * (t - t_inf)).exp()
        };
        assert_relative_eq!(got, want, max_relative = 1e-12);
    }
}

#[test]
fn consumes_pk_slot_matches_solver() {
    // Pin `PkModel::consumes_pk_slot` — the source of truth for the parser's
    // unused-param warning (#309) — to what the analytical solvers actually
    // read: perturbing a *consumed* slot must change the predicted
    // concentration, and perturbing a *non-consumed* slot must leave it
    // bit-identical. A future variant whose closed form reads a new slot but
    // isn't reflected in `consumes_pk_slot`/`required_pk_params` fails here,
    // closing the drift gap between the parser table and the solvers.
    use crate::types::{
        PK_IDX_CL, PK_IDX_F, PK_IDX_KA, PK_IDX_LAGTIME, PK_IDX_Q, PK_IDX_Q3, PK_IDX_V, PK_IDX_V2,
        PK_IDX_V3,
    };
    let named_slots = [
        PK_IDX_CL,
        PK_IDX_V,
        PK_IDX_Q,
        PK_IDX_V2,
        PK_IDX_KA,
        PK_IDX_F,
        PK_IDX_Q3,
        PK_IDX_V3,
        PK_IDX_LAGTIME,
    ];
    let baseline = || {
        let mut p = PkParams::default();
        p.values[PK_IDX_CL] = 1.0;
        p.values[PK_IDX_V] = 10.0;
        p.values[PK_IDX_Q] = 0.7;
        p.values[PK_IDX_V2] = 20.0;
        p.values[PK_IDX_KA] = 1.3;
        p.values[PK_IDX_F] = 0.6;
        p.values[PK_IDX_Q3] = 0.4;
        p.values[PK_IDX_V3] = 30.0;
        p.values[PK_IDX_LAGTIME] = 0.5;
        p
    };
    let doses = vec![bolus_dose(0.0, 100.0)];
    let t = 2.0;
    for model in [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ] {
        let c0 = predict_concentration(model, &doses, t, &baseline());
        assert!(c0 > 0.0, "{model:?}: baseline concentration should be positive");
        for &slot in &named_slots {
            let mut perturbed = baseline();
            perturbed.values[slot] *= 5.0;
            let c1 = predict_concentration(model, &doses, t, &perturbed);
            if model.consumes_pk_slot(slot) {
                assert!(
                    (c1 - c0).abs() > 1e-9,
                    "{model:?}: slot {slot} is marked consumed but perturbing it \
                     did not change the prediction (c0={c0}, c1={c1})"
                );
            } else {
                assert_eq!(
                    c1, c0,
                    "{model:?}: slot {slot} is marked unused but perturbing it \
                     changed the prediction"
                );
            }
        }
    }
}

#[test]
fn test_superposition_single_dose() {
    let doses = vec![bolus_dose(0.0, 1000.0)];
    let pk = make_pk_params(10.0, 100.0);
    let c = predict_concentration(PkModel::OneCptIv, &doses, 0.0, &pk);
    assert_relative_eq!(c, 10.0, epsilon = 1e-10);
}

#[test]
fn test_superposition_two_doses() {
    let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(10.0, 1000.0)];
    let pk = make_pk_params(10.0, 100.0);
    let k: f64 = 10.0 / 100.0;

    // At t=10, first dose has decayed, second dose just given
    let c = predict_concentration(PkModel::OneCptIv, &doses, 10.0, &pk);
    let expected = (1000.0_f64 / 100.0) * (-k * 10.0).exp() + 1000.0 / 100.0;
    assert_relative_eq!(c, expected, epsilon = 1e-10);
}

#[test]
fn test_superposition_ignores_future_doses() {
    let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(100.0, 1000.0)];
    let pk = make_pk_params(10.0, 100.0);

    // At t=5, second dose hasn't happened yet
    let c_single = predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 1000.0)], 5.0, &pk);
    let c_two = predict_concentration(PkModel::OneCptIv, &doses, 5.0, &pk);
    assert_relative_eq!(c_single, c_two, epsilon = 1e-12);
}

/// #1133, the occasion-label route. NONMEM anchor J (`nonmem_anchor/reset_init_snapshot_J.ctl`):
/// with `WT` flat and only `OCC` moving, `$PK` at the reset row runs under **that row's own
/// occasion** (42.0, occasion 2), not the preceding record's (14.0, occasion 1). The ferx twin
/// of that anchor reaches `OCC` as a covariate (κ is estimated); this test drives `predict_iov`
/// with explicit κ so the occasion LABEL selects the seed. The record before the reset is
/// `OCC = 1` and the reset row `OCC = 2`, so any rule inferring the occasion from the PRECEDING
/// record gives 1 (what ferx did before) and only reading `reset_occasions` gives 2. The
/// candidate seeds carry distinct κ (occasion 2: ln 3, occasion 1: ln 2, none: 0).
#[test]
fn iov_reset_takes_its_own_rows_occasion() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
        "
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(50.0, 0.1, 1000.0)
  theta TVBASE(10.0, 0.01, 1000.0)
  omega ETA_CL ~ 0.0
  kappa KAPPA_B ~ 0.04
  sigma PROP ~ 0.01
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVBASE * exp(KAPPA_B)
[structural_model]
  ode(obs_cmt=central, states=[central])
[odes]
  init(central)  = BASE
  d/dt(central)  = -(CL/V) * central
[scaling]
  obs_scale = V
[error_model]
  DV ~ proportional(PROP)
",
    )
    .expect("fixture parses");

    let mut subj = Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 9.0],
        observations: vec![0.0; 2],
        obs_cmts: vec![1; 2],
        reset_times: vec![8.0],
        // The reset row itself is occasion 2 — the occasion it STARTS. The record
        // before it is occasion 1, which is what every neighbour-inferring rule
        // resolves to, and what ferx used before this was measured.
        reset_occasions: vec![2],
        cens: vec![0; 2],
        // t=1 is occasion 1, t=9 (after the reset) is occasion 2, so both are kappa
        // groups through an observation. (An occasion named only on a reset row is a
        // group too since #1153 — the end of this test.)
        occasions: vec![1, 2],
        ..Default::default()
    };

    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.len(), 2, "occasions 1 and 2 must both be groups");
    let theta = model.default_params.theta.clone();
    let eta_bsv = vec![0.0];
    // Occasion 1 gets κ = ln 2, occasion 2 gets κ = ln 3.
    let k_of = |occ: u32| groups.iter().position(|(id, _)| *id == occ).expect("group");
    let mut kappas = vec![vec![0.0]; groups.len()];
    kappas[k_of(1)] = vec![2.0f64.ln()];
    kappas[k_of(2)] = vec![3.0f64.ln()];

    let own = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];

    // Drop the row's own label: an unlabelled reset seeds under κ = 0 (#1153), not
    // under the preceding record's occasion 1 — a guess from a neighbour would give
    // `own / unlabelled = 1.5` here.
    subj.reset_occasions.clear();
    let unlabelled = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];

    assert_relative_eq!(own / unlabelled, 3.0, epsilon = 1e-9);

    // An occasion ONLY the reset row carries (#1153) is a κ group of its own, so the
    // seed reads its κ: `BASE = 10·exp(κ₇)` and nothing else moves, so
    // `y(9) = 10·5·e^{−0.1}/50` exactly, against `10·e^{−0.1}/50` under κ = 0.
    subj.reset_occasions = vec![7];
    let groups = crate::stats::likelihood::iov_occasion_groups(&subj);
    assert_eq!(groups.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1, 2, 7]);
    let mut kappas = vec![vec![0.0]; 3];
    kappas[0] = vec![2.0f64.ln()];
    kappas[2] = vec![5.0f64.ln()];
    let reset_only = predict_iov(&model, &subj, &theta, &eta_bsv, &kappas)[1];
    assert_relative_eq!(reset_only, 10.0 * 5.0 * (-0.1f64).exp() / 50.0, epsilon = 1e-9);
}

/// Records at 0 (dose, occasion 1), 2 (pk-only), 5 (obs, occasion 1) and 8 (obs,
/// occasion 2). Occasion 2 is a κ group through the t = 8 observation.
fn pk_only_occasion_subject(pk_only_occasions: Vec<u32>) -> Subject {
    Subject {
        id: "1".into(),
        doses: vec![bolus_dose(0.0, 1000.0)],
        dose_occasions: vec![1],
        obs_times: vec![5.0, 8.0],
        observations: vec![1.0; 2],
        obs_cmts: vec![1; 2],
        occasions: vec![1, 2],
        cens: vec![0; 2],
        pk_only_times: vec![2.0],
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: vec![HashMap::new()],
        pk_only_occasions,
        ..Default::default()
    }
}

/// #1876 / #1153: a pk-only row resolves to its own label, and to `None` (κ = 0)
/// without one — never to a neighbouring record's, though the t = 0 dose (occasion 4)
/// sits right before it.
#[test]
fn pk_only_row_occasion_is_its_own_label_or_none() {
    assert_eq!(pk_only_row_occasion(&pk_only_occasion_subject(vec![7]), 0), Some(7));
    let mut s = pk_only_occasion_subject(Vec::new());
    s.dose_occasions = vec![4];
    assert_eq!(pk_only_row_occasion(&s, 0), None);
    // A label is not a group. `None` is the κ = 0 row in every provider.
    let occ_to_k = HashMap::from([(1u32, 0usize), (2, 1)]);
    let s = pk_only_occasion_subject(vec![2]);
    assert_eq!(pk_only_row_group(&s, 0, &occ_to_k), Some(1));
    let s = pk_only_occasion_subject(vec![9]);
    assert_eq!(pk_only_row_group(&s, 0, &occ_to_k), None);
}

/// #1153: a reset row and a dose resolve to their own label, and to `None` (κ = 0)
/// without one. The reset at 3 sits right after a pk-only row labelled 3 and the dose
/// is labelled 1, so a neighbour guess would give `Some(3)` or `Some(1)`; a dose with no
/// label used to fall back to occasion `0`, which here names a real group.
#[test]
fn reset_and_dose_occasion_are_their_own_label_or_none() {
    let mut s = pk_only_occasion_subject(vec![3]);
    s.reset_times = vec![3.0];
    s.reset_occasions = vec![5];
    assert_eq!(reset_row_occasion(&s, 0), Some(5));
    let occ_to_k = HashMap::from([(0u32, 0usize), (1, 1), (3, 2), (5, 3)]);
    assert_eq!(reset_row_group(&s, 0, &occ_to_k), Some(3));
    s.reset_occasions.clear();
    assert_eq!(reset_row_occasion(&s, 0), None);
    assert_eq!(reset_row_group(&s, 0, &occ_to_k), None);

    assert_eq!(dose_row_occasion(&s, 0), Some(1));
    assert_eq!(dose_row_group(&s, 0, &occ_to_k), Some(1));
    s.dose_occasions.clear();
    assert_eq!(dose_row_occasion(&s, 0), None);
    assert_eq!(dose_row_group(&s, 0, &occ_to_k), None);
}

/// #1876 on the value: the pk-only row at `t_p = 2` governs `(0, 2]` under ITS
/// occasion, so the t = 5 observation is `D/V·exp(−k₂·t_p − k₁·(5 − t_p))`, with
/// `k_i = CL·exp(κ_i)/V`. κ₁ = ln 2 and κ₂ = ln 3, so the three conventions differ:
///
/// | pk-only row under | `(0, 2]` rate | pred(5) |
/// |---|---|---|
/// | its own occasion: 2, or 9 that only it carries (κ₉ = ln 3) | 0.3 | 20·e^−1.2 |
/// | κ = 0 (ferx before #1876) | 0.1 | 20·e^−0.8 |
/// | the record before it (occasion 1) | 0.2 | 20·e^−1.0 |
///
/// Run on the closed form and on an `[odes]` twin, which share
/// `fill_iov_record_params` and nothing downstream of it.
#[test]
fn iov_pk_only_row_runs_under_its_own_occasion() {
    use crate::parser::model_parser::parse_model_string;
    let header = "
[parameters]
  theta TVCL(5.0, 0.01, 100.0)
  theta TVV(50.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.04
  kappa KAPPA_CL ~ 0.04
  sigma PROP ~ 0.01
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
";
    let tail = "
[error_model]
  DV ~ proportional(PROP)
";
    let analytical = parse_model_string(&format!("{header}  pk one_cpt_iv(cl=CL, v=V)\n{tail}"))
        .expect("analytical fixture parses");
    let ode = parse_model_string(&format!(
        "{header}  ode(obs_cmt=central, states=[central])
[odes]
  d/dt(central) = -(CL/V) * central
[scaling]
  obs_scale = V
{tail}"
    ))
    .expect("ODE fixture parses");
    let own = 20.0 * (-1.2f64).exp();
    let kappa0 = 20.0 * (-0.8f64).exp();
    let previous = 20.0 * (-1.0f64).exp();
    // Measured worst relative errors (macOS arm64): analytical 2.0e-16; ODE 1.03e-5,
    // the same order in all three conventions, i.e. RK45 at the default `reltol = 1e-4`
    // and not the switch at t = 2. 1e-4 keeps ~10x headroom; the straddle needs 0.22.
    for (name, model, tol) in [("analytical", &analytical, 1e-12), ("ode", &ode, 1e-4)] {
        let theta = model.default_params.theta.clone();
        let pred = |labels: Vec<u32>| {
            let s = pk_only_occasion_subject(labels);
            let groups = crate::stats::likelihood::iov_occasion_groups(&s);
            let mut kappas = vec![vec![0.0]; groups.len()];
            // Occasion 9 is carried by the pk-only row alone, when it is so labelled.
            for (occ, kappa) in [(1, 2.0f64.ln()), (2, 3.0f64.ln()), (9, 3.0f64.ln())] {
                if let Some(k) = groups.iter().position(|(id, _)| *id == occ) {
                    kappas[k] = vec![kappa];
                }
            }
            predict_iov(model, &s, &theta, &[0.0], &kappas)[0]
        };
        let got = pred(vec![2]);
        assert!(got.is_finite(), "{name}: {got}");
        eprintln!("{name}: pred(5) relative error {:e}", (got - own).abs() / own);
        assert!((got - own).abs() / own < tol, "{name}: pred(5) = {got}, own-occasion value {own}");
        // The straddle: the conventions differ by 22% and 49% of `own`, far above `tol`.
        for (wrong, what) in [(kappa0, "kappa = 0"), (previous, "previous record")] {
            assert!(
                (got - wrong).abs() / own > 0.1,
                "{name}: indistinguishable from the {what} convention"
            );
        }
        // A label no dose or observation carries gets a κ group of its own, so the
        // row runs under κ₉ = ln 3, not κ = 0.
        let only_pk_only = pred(vec![9]);
        assert!(
            (only_pk_only - own).abs() / own < tol,
            "{name}: an occasion only the pk-only row carries gave {only_pk_only}, \
             its own kappa gives {own} (kappa = 0 is {kappa0})"
        );
        // No label (an in-memory subject): κ = 0 (#1153), not the occasion-1 dose before
        // it — the straddle above keeps the two conventions apart.
        let unlabelled = pred(Vec::new());
        assert!(
            (unlabelled - kappa0).abs() / kappa0 < tol,
            "{name}: unlabelled row gave {unlabelled}, kappa = 0 is {kappa0} \
             (previous record is {previous})"
        );
    }
}

/// #1133 on the adaptive (decision-window) IOV materialiser: a reset row's snapshot is *its own
/// covariates* under the κ of the decision window active at its time, like every other record.
///
/// With `CL = TVCL·(WT/100)·exp(ETA_CL + KAPPA_CL)`, decisions at 0 and 24, resets at `t = 5`
/// (window 0, κ = 0) and `t = 30` (window 1, κ = ln 2), each candidate convention differs:
///
/// | reset | own row | previous record | next record |
/// |---|---|---|---|
/// | `t = 5`  | **0.5** (WT 50) | 1.0 (WT 100) | 0.1 (WT 10) |
/// | `t = 30` | **1.6** (WT 80, κ = ln 2) | 2.0 (WT 100) | 0.2 (WT 10) |
///
/// and the baseline κ instead of the window's gives 0.8 for the second.
#[test]
fn iov_event_pk_seeds_each_reset_from_its_own_row_and_window() {
    use crate::parser::model_parser::parse_model_string;
    let model = parse_model_string(
        "
[parameters]
  theta TVCL(1.0, 0.01, 100.0)
  theta TVV(10.0, 0.1, 1000.0)
  omega ETA_CL ~ 0.0
  kappa KAPPA_CL ~ 0.0
  sigma PROP ~ 0.01
[individual_parameters]
  CL = TVCL * (WT / 100.0) * exp(ETA_CL + KAPPA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[covariates]
  WT continuous
[error_model]
  DV ~ proportional(PROP)
",
    )
    .expect("fixture parses");

    let wt = |w: f64| HashMap::from([("WT".to_string(), w)]);
    let subj = Subject {
        id: "1".to_string(),
        obs_times: vec![1.0, 40.0],
        observations: vec![0.0; 2],
        obs_cmts: vec![1; 2],
        covariates: wt(100.0),
        obs_covariates: vec![wt(100.0), wt(10.0)],
        reset_times: vec![5.0, 30.0],
        // Each reset row carries a `WT` shared with no neighbouring record, so reading
        // one row early or late shows up in the assertions below.
        reset_covariates: vec![wt(50.0), wt(80.0)],
        cens: vec![0; 2],
        ..Default::default()
    };

    let theta = model.default_params.theta.clone();
    let eta_baseline = vec![0.0, 0.0]; // [ETA_CL, KAPPA_CL], κ = 0
    let eta_occ = vec![
        vec![0.0, 0.0],                    // window 0 (decision at t=0): κ = 0
        vec![0.0, std::f64::consts::LN_2], // window 1 (decision at t=24): κ = ln 2
    ];
    let decisions = vec![0.0, 24.0];
    let ev =
        compute_event_pk_params_iov(&model, &subj, &theta, &eta_baseline, &eta_occ, &decisions);

    assert_eq!(ev.reset.len(), 2, "one snapshot per reset row");
    assert_relative_eq!(ev.reset[0].values[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(ev.reset[1].values[0], 1.6, epsilon = 1e-12);
}

#[test]
fn compute_predictions_routes_reset_subject_to_event_driven() {
    // A subject with a reset (EVID=3) must NOT use superposition:
    // `compute_predictions` should zero the state at the reset. Dose at
    // t=0, reset at t=5, obs at t=1 (positive) and t=6 (zero).
    let subj = Subject {
        id: "1".to_string(),
        doses: vec![bolus_dose(0.0, 1000.0)],
        obs_times: vec![1.0, 6.0],
        observations: vec![0.0; 2],
        obs_cmts: vec![1; 2],
        reset_times: vec![5.0],
        cens: vec![0; 2],
        ..Default::default()
    };
    let pk = make_pk_params(10.0, 100.0);
    let preds = compute_predictions(PkModel::OneCptIv, &subj, &pk);
    assert!(preds[0] > 0.0);
    assert_relative_eq!(preds[1], 0.0, epsilon = 1e-12);
}

#[test]
#[should_panic(expected = "modeled-RATE dose reached the analytical predictor")]
fn compute_predictions_panics_on_modeled_dose() {
    // #324 / review #2: the analytical superposition predictor must reject a
    // modeled-RATE (RATE=-2) dose loudly *in release too* — a real `assert!`,
    // not the old `debug_assert!` that compiled out and let a 0-rate "infusion"
    // silently produce 0/NaN. Direct call (bypassing the public entrypoints'
    // gates) proves the predictor itself fails fast on an unvalidated subject.
    use crate::types::RateMode;
    let mut subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 0, 1);
    subj.doses = vec![DoseEvent::modeled(0.0, 100.0, 1, false, 0.0, RateMode::ModeledDuration)];
    let pk = make_pk_params(10.0, 100.0);
    let _ = compute_predictions(PkModel::OneCptIv, &subj, &pk);
}

fn make_subject_with_tv(
    cov_const: HashMap<String, f64>, dose_cov: Vec<HashMap<String, f64>>,
    obs_cov: Vec<HashMap<String, f64>>, n_doses: usize, n_obs: usize,
) -> Subject {
    Subject {
        id: "1".to_string(),
        doses: (0..n_doses).map(|i| bolus_dose(i as f64, 100.0)).collect(),
        obs_times: (0..n_obs).map(|i| 1.0 + i as f64).collect(),
        observations: vec![0.0; n_obs],
        obs_cmts: vec![1; n_obs],
        covariates: cov_const,
        dose_covariates: dose_cov,
        obs_covariates: obs_cov,
        cens: vec![0; n_obs],
        ..Default::default()
    }
}

/// Build a tiny analytical CompiledModel where CL = covariate `CR` *
/// theta[0] (so we can prove pk_param_fn was called with the right
/// covariate snapshot per event).
fn cl_from_cr_model() -> crate::types::CompiledModel {
    use crate::types::{CompiledModel, ErrorModel, ModelParameters, OmegaMatrix, SigmaVector};
    CompiledModel {
        name: "cl_from_cr".into(),
        error_model: ErrorModel::Additive,
        error_spec: crate::types::ErrorSpec::Single(ErrorModel::Additive),
        pk_param_fn: Box::new(|theta, _eta, cov, _t: f64| {
            let mut p = PkParams::default();
            let cr = cov.get("CR").copied().unwrap_or(1.0);
            p.values[crate::types::PK_IDX_CL] = theta[0] * cr;
            p.values[crate::types::PK_IDX_V] = 50.0;
            p
        }),
        n_theta: 1,
        n_eta: 0,
        theta_names: vec!["TVCL".into()],
        eta_names: Vec::new(),
        omega_init_as_sd: Vec::new(),
        pk_indices: vec![],
        eta_map: vec![],
        pk_idx_f64: vec![],
        sel_flat: vec![],
        referenced_covariates: vec!["CR".into()],
        indiv_param_names: Vec::new(),
        ..crate::types::test_helpers::bare_model(ModelParameters {
            theta: vec![1.0],
            theta_names: vec!["TVCL".into()],
            theta_lower: vec![0.0],
            theta_upper: vec![f64::INFINITY],
            theta_fixed: vec![false],
            omega_fixed: Vec::new(),
            sigma: SigmaVector { values: vec![0.1], names: vec!["EPS".into()] },
            ..crate::types::test_helpers::bare_params(OmegaMatrix::from_diagonal(&[], Vec::new()))
        })
    }
}

#[test]
fn test_event_pk_params_no_tv_evaluates_once() {
    // No TV covariates → uses subject.covariates; result is identical
    // for every event.
    let mut covs = HashMap::new();
    covs.insert("CR".to_string(), 2.0);
    let subj = make_subject_with_tv(covs, Vec::new(), Vec::new(), 2, 3);
    let model = cl_from_cr_model();
    let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
    assert_eq!(ev.dose.len(), 2);
    assert_eq!(ev.obs.len(), 3);
    // CL = theta * CR = 10 * 2 = 20 everywhere.
    for p in ev.dose.iter().chain(ev.obs.iter()) {
        assert_relative_eq!(p.cl(), 20.0, epsilon = 1e-12);
    }
}

/// The parameter-static arm must still emit one `pk_only` snapshot per EVID=2 record. Both
/// event-driven walkers (`ode/predictions.rs`, `pk/event_driven.rs`) open with the same three
/// unconditional length asserts (`assert_eq!`, so release aborts too), so a short `pk_only`
/// aborts the run. The arm used to leave `pk_only` empty while filling `dose` and `obs`. It is
/// reachable when a *parameter-static* subject is routed to an event-driven walker: by
/// `subject.has_resets()` (EVID 3/4, predating #1124) or, since #1124, an `[odes]` RHS that
/// reads model time. Both need only EVID=2 rows, because `prune_irrelevant_tv_covariates`
/// clears `pk_only_covariates` but leaves `pk_only_times` populated, the state where
/// `subject_needs_per_event_pk` is false and `pk_only_times` is not.
#[test]
fn event_pk_params_fills_pk_only_on_the_parameter_static_arm() {
    let mut covs = HashMap::new();
    covs.insert("CR".to_string(), 2.0);
    let mut subj = make_subject_with_tv(covs, Vec::new(), Vec::new(), 2, 3);
    // An EVID=2 row that survived covariate pruning: a time, but no snapshot.
    subj.pk_only_times = vec![0.5, 2.5];
    subj.pk_only_covariates = Vec::new();
    let model = cl_from_cr_model();
    // Precondition: this is the static arm, not the per-event one. Without
    // this the test would still pass via the `if` branch and prove nothing.
    assert!(
        !subject_needs_per_event_pk(&model, &subj),
        "fixture must exercise the parameter-static arm"
    );
    let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
    assert_eq!(ev.dose.len(), subj.doses.len());
    assert_eq!(ev.obs.len(), subj.obs_times.len());
    assert_eq!(
        ev.pk_only.len(),
        subj.pk_only_times.len(),
        "pk_only must carry one snapshot per EVID=2 record; both event-driven \
         walkers assert this length unconditionally"
    );
    // CL = theta * CR = 10 * 2 = 20 on every row, pk_only included.
    for p in ev.dose.iter().chain(&ev.obs).chain(&ev.pk_only) {
        assert_relative_eq!(p.cl(), 20.0, epsilon = 1e-12);
    }
}

#[test]
fn test_event_pk_params_tv_uses_per_event_snapshot() {
    // Per-event snapshots drive different CL at each event.
    let mk = |cr: f64| {
        let mut h = HashMap::new();
        h.insert("CR".to_string(), cr);
        h
    };
    let dose_cov = vec![mk(1.0), mk(1.5)];
    let obs_cov = vec![mk(1.0), mk(2.0)];
    let subj = make_subject_with_tv(mk(1.0), dose_cov, obs_cov, 2, 2);
    let model = cl_from_cr_model();
    let ev = compute_event_pk_params(&model, &subj, &[10.0], &[]);
    assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12);
    assert_relative_eq!(ev.dose[1].cl(), 15.0, epsilon = 1e-12);
    assert_relative_eq!(ev.obs[0].cl(), 10.0, epsilon = 1e-12);
    assert_relative_eq!(ev.obs[1].cl(), 20.0, epsilon = 1e-12);
}

#[test]
fn test_event_pk_params_time_builtin_uses_event_times_without_tv_covariates() {
    let model_str = "
[parameters]
  theta TVCL(10.0, 0.001, 100.0)
  omega ETA_CL ~ 0.1
  sigma EPS ~ 0.1

[individual_parameters]
  if (TIME > 0.5) {
CL = (TVCL + TIME) * exp(ETA_CL)
  } else {
CL = TVCL * exp(ETA_CL)
  }

[structural_model]
  pk one_cpt_iv(cl=CL, v=time)

[error_model]
  DV ~ additive(EPS)
";
    let model = crate::parser::model_parser::parse_model_string(model_str)
        .expect("TIME/time built-ins parse");
    assert!(
        !model.referenced_covariates.iter().any(|c| c == "TIME" || c == "time"),
        "TIME/time must not be tracked as data covariates: {:?}",
        model.referenced_covariates
    );

    let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 2, 2);
    assert!(!subj.has_tv_covariates(), "fixture must stay on no-TV path");

    let ev = compute_event_pk_params(&model, &subj, &model.default_params.theta, &[0.0]);

    assert_relative_eq!(ev.dose[0].cl(), 10.0, epsilon = 1e-12); // TIME = 0
    assert_relative_eq!(ev.dose[1].cl(), 11.0, epsilon = 1e-12); // TIME = 1
    assert_relative_eq!(ev.obs[0].cl(), 11.0, epsilon = 1e-12); // TIME = 1
    assert_relative_eq!(ev.obs[1].cl(), 12.0, epsilon = 1e-12); // TIME = 2
    assert_relative_eq!(ev.dose[0].v(), 0.0, epsilon = 1e-12); // time = 0
    assert_relative_eq!(ev.obs[1].v(), 2.0, epsilon = 1e-12); // time = 2
}

#[test]
fn test_compute_predictions_length() {
    let subject = Subject {
        id: "1".to_string(),
        doses: vec![bolus_dose(0.0, 1000.0)],
        obs_times: vec![1.0, 2.0, 4.0, 8.0],
        observations: vec![0.0; 4],
        obs_cmts: vec![1; 4],
        cens: vec![0; 4],
        ..Default::default()
    };
    let pk = make_pk_params(10.0, 100.0);
    let preds = compute_predictions(PkModel::OneCptIv, &subject, &pk);
    assert_eq!(preds.len(), 4);
    // Predictions should be monotonically decreasing for IV bolus
    for i in 1..preds.len() {
        assert!(preds[i] < preds[i - 1]);
    }
}

fn oral_pk_params(cl: f64, v: f64, ka: f64) -> PkParams {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = cl;
    p.values[crate::types::PK_IDX_V] = v;
    p.values[crate::types::PK_IDX_KA] = ka;
    p
}

#[test]
fn test_predict_concentration_lagtime_zero_matches_baseline() {
    // With lagtime = 0, predict_concentration must agree with the
    // pre-feature behavior at any observation time.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let pk = oral_pk_params(2.0, 20.0, 1.5); // CL/V = 0.1, KA = 1.5
    let c0 = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk);
    // Bateman: C(t) = (F*Dose*KA) / (V*(KA-k)) * [exp(-k*t) - exp(-KA*t)]
    let k = 2.0_f64 / 20.0;
    let expected =
        (1.0 * 100.0 * 1.5) / (20.0 * (1.5 - k)) * ((-k * 2.0).exp() - (-1.5 * 2.0_f64).exp());
    assert_relative_eq!(c0, expected, epsilon = 1e-10);
}

#[test]
fn test_predict_concentration_lagtime_shifts_curve() {
    // Concentration at t with lagtime=L should equal the unlagged
    // concentration evaluated at (t - L).
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let mut pk_lag = oral_pk_params(2.0, 20.0, 1.5);
    pk_lag.values[crate::types::PK_IDX_LAGTIME] = 1.5;
    let pk_nolag = oral_pk_params(2.0, 20.0, 1.5);

    // At t = 3.5 with lag=1.5, effective elapsed time is 2.0.
    let c_lag = predict_concentration(PkModel::OneCptOral, &doses, 3.5, &pk_lag);
    let c_no = predict_concentration(PkModel::OneCptOral, &doses, 2.0, &pk_nolag);
    assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
}

#[test]
fn test_predict_concentration_lagtime_before_first_obs_returns_zero() {
    // Lagtime longer than the observation window: nothing has reached
    // the system yet, so concentration is exactly 0.
    let doses = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let mut pk = oral_pk_params(2.0, 20.0, 1.5);
    pk.values[crate::types::PK_IDX_LAGTIME] = 5.0;
    let c = predict_concentration(PkModel::OneCptOral, &doses, 1.0, &pk);
    assert_eq!(c, 0.0);
}

/// A non-finite dose arrival must come back `NaN`, never a drug-free `0.0` (#1189's silent-drop
/// shape, closed on this engine by #1284).
///
/// `predict_concentration` sums closed forms over doses instead of walking a timeline, so it
/// has no `times_have_non_finite` to run. Both its arms test `t_eff` against `t`, every
/// comparison against `NaN` is false, so the dose was skipped and the remaining trajectory
/// returned as valid. Also the trailing `conc.max(0.0)` is an `f64::max`, which discards `NaN`
/// (`0.0f64.max(NaN) == 0.0`, verified below), so even a dose that contributed `NaN` was floored
/// to `0.0`: two independent ways to read finite and wrong.
///
/// Both inputs to `t_eff` are exercised (a `NaN` `lagtime` from an overflowing `$PK`
/// expression; a `NaN` `dose.time` from `DoseEvent::non_finite`, a modeled `D{n}`/`R{n}` out of
/// domain); the finite control pins that the guard did not start rejecting everything.
///
/// Mutation (run): deleting the `superposition_arrival_non_finite` early return in
/// `predict_concentration` makes both non-finite arms return `0.0`; in
/// `analytical_state_at_times`, the state half fires instead.
#[test]
fn non_finite_arrival_gives_nan_not_a_drug_free_zero() {
    // The floor that would have hidden it — measured here rather than recalled.
    assert_eq!(f64::NAN.max(0.0), 0.0, "f64::max discards NaN");

    let pk = make_pk_params(1.0, 10.0);
    let finite = predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 100.0)], 4.0, &pk);
    assert!(finite.is_finite() && finite > 0.0, "control: a finite dose still predicts ({finite})");

    let mut nan_lag = pk;
    nan_lag.values[crate::types::PK_IDX_LAGTIME] = f64::NAN;
    let via_lag =
        predict_concentration(PkModel::OneCptIv, &[bolus_dose(0.0, 100.0)], 4.0, &nan_lag);
    assert!(via_lag.is_nan(), "NaN lagtime: got {via_lag}, want NaN");

    let via_dose_time =
        predict_concentration(PkModel::OneCptIv, &[bolus_dose(f64::NAN, 100.0)], 4.0, &pk);
    assert!(via_dose_time.is_nan(), "NaN dose time: got {via_dose_time}, want NaN");

    // The *other* finite dose in the same subject must not rescue it: the
    // subject is non-finite, not "finite where it happens to have data".
    let mixed = predict_concentration(
        PkModel::OneCptIv,
        &[bolus_dose(0.0, 100.0), bolus_dose(f64::NAN, 100.0)],
        4.0,
        &pk,
    );
    assert!(mixed.is_nan(), "one bad dose poisons the sum: got {mixed}");

    // …and the **state** twin must say the same thing about the same subject.
    // It ran its own copy of this loop and had its own zero floor, so with the
    // guard on one side only it reported a drug-free `[[0.0]]` next to the
    // `NaN` above — a valid-looking compartment amount for a repelled subject
    // (#1399 review finding 1).
    let subj = Subject {
        id: "nan-arrival".into(),
        doses: vec![bolus_dose(f64::NAN, 100.0)],
        obs_times: vec![4.0, 8.0],
        ..Default::default()
    };
    let states = analytical_state_at_times(PkModel::OneCptIv, &subj, &pk, &subj.obs_times);
    assert_eq!(states.len(), 2, "one state vector per time");
    for (k, s) in states.iter().enumerate() {
        assert!(s.iter().all(|v| v.is_nan()), "state twin at time {k}: got {s:?}, want all NaN");
    }
    // Straddle: the same walk on a finite dose still returns real amounts.
    let ok_subj = Subject {
        id: "finite".into(),
        doses: vec![bolus_dose(0.0, 100.0)],
        obs_times: vec![4.0],
        ..Default::default()
    };
    let ok_states = analytical_state_at_times(PkModel::OneCptIv, &ok_subj, &pk, &ok_subj.obs_times);
    assert!(
        ok_states[0].iter().all(|v| v.is_finite()) && ok_states[0][0] > 0.0,
        "control state: {:?}",
        ok_states[0]
    );
}

/// A `NaN` that enters the superposition **through the closed form's own inputs**, not the
/// arrival, must survive the trailing zero floor (#1399 review). `conc.max(0.0)` discarded the
/// `NaN` and reported a confident `0.0`. A non-finite bioavailability `F` is the reachable route:
/// the arrival is finite, so `superposition_arrival_non_finite` does **not** cover it (asserted
/// below: two gates rejecting the same inputs would be a test hole). Before the fix an analytical
/// `one_cpt_iv(.., f=F)` with `F` overflowed at typical values returned `predict() = [0.0]` while
/// the state twin (whose `if *s < 0.0` floor keeps `NaN`) returned `[[NaN]]`.
///
/// Mutation (run): restoring the bare `conc.max(0.0)` makes the `F` arm return `0.0`.
#[test]
fn a_nan_from_the_closed_form_survives_the_zero_floor() {
    let doses = [bolus_dose(0.0, 100.0)];
    let mut pk = make_pk_params(1.0, 10.0);
    pk.values[crate::types::PK_IDX_F] = f64::NAN;

    // The arrival is finite, so the other guard cannot be what catches this.
    assert!(
        !superposition_arrival_non_finite(&doses, pk.lagtime()),
        "the arrival guard must NOT cover the F case — otherwise this test is \
         passing on the strength of a different gate"
    );

    let got = predict_concentration(PkModel::OneCptIv, &doses, 4.0, &pk);
    assert!(got.is_nan(), "NaN F: got {got}, want NaN");

    // A negative concentration is still floored — the fix preserves `NaN`, it does
    // not remove the floor.
    let mut neg = make_pk_params(1.0, 10.0);
    neg.values[crate::types::PK_IDX_F] = -1.0;
    let floored = predict_concentration(PkModel::OneCptIv, &doses, 4.0, &neg);
    assert_eq!(floored, 0.0, "a negative concentration still floors to 0");
}

#[test]
fn test_infusion_with_lagtime_shifts_window() {
    // 1-cpt infusion, amt=100, rate=100 → duration = amt/rate = 1.0
    // (the `DoseEvent::new` signature is `(time, amt, cmt, rate, ss, ii)`
    // — duration is auto-computed from amt/rate; the trailing 0.0 is `ii`).
    // Starting at t_dose=2, with lagtime=0.5, the infusion effectively
    // runs 2.5..3.5.
    let dose = DoseEvent::new(2.0, 100.0, 1, 100.0, false, 0.0);
    debug_assert!(
        dose.is_infusion() && dose.duration > 0.0,
        "test must exercise the infusion branch"
    );
    let doses = vec![dose.clone()];
    let mut pk = make_pk_params(10.0, 100.0);
    pk.values[crate::types::PK_IDX_LAGTIME] = 0.5;

    // (a) Before lagged start: still zero.
    let c_pre = predict_concentration(PkModel::OneCptIv, &doses, 0.6, &pk);
    assert_eq!(c_pre, 0.0);

    // (b) Mid-infusion (lagged): pre-lag conc at tau = t - (dose.time + lag) = t - 2.5.
    // Compare against unlagged infusion at tau via the same formula.
    let pk_nolag = make_pk_params(10.0, 100.0);
    let c_lag = predict_concentration(PkModel::OneCptIv, &doses, 2.6, &pk);
    let c_no = predict_concentration(
        PkModel::OneCptIv,
        &[DoseEvent::new(0.1, 100.0, 1, 100.0, false, 0.0)],
        0.2,
        &pk_nolag,
    );
    // Both probe an elapsed time of 0.1 into a 1h infusion of rate=100.
    assert_relative_eq!(c_lag, c_no, epsilon = 1e-10);

    // (c) Post-infusion (lagged window ends at 3.5).
    let c_post = predict_concentration(PkModel::OneCptIv, &doses, 3.6, &pk);
    let c_post_nolag = predict_concentration(
        PkModel::OneCptIv,
        &[DoseEvent::new(0.0, 100.0, 1, 100.0, false, 0.0)],
        1.1,
        &pk_nolag,
    );
    assert_relative_eq!(c_post, c_post_nolag, epsilon = 1e-10);
}

// --- Steady-state (SS=1) integration with predict_concentration ---

#[test]
fn test_ss_iv_bolus_via_predict_matches_closed_form() {
    // Sanity check: predict_concentration with an SS=1 dose dispatches to
    // the closed-form SS formula (not the single-dose response).
    let ii = 12.0_f64;
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];
    let pk = make_pk_params(10.0, 100.0);
    let k = 10.0_f64 / 100.0;
    for &t in &[0.0, 3.0, 11.9, 24.0] {
        let c = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
        let expected = (1000.0 / 100.0) * (-k * t).exp() / (1.0 - (-k * ii).exp());
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }
}

#[test]
fn test_ss_with_lagtime_identity_c_t_l_equals_c_t_minus_l() {
    // Acceptance criterion from issue #15: with SS+lagtime, the curve at
    // time t with lagtime L equals the unlagged SS curve at t - L. This
    // is the geometric-series identity under linear superposition.
    let ii = 12.0;
    let lagtime = 1.5;
    let doses = vec![DoseEvent::new(0.0, 1000.0, 1, 0.0, true, ii)];

    let mut pk_lag = make_pk_params(10.0, 100.0);
    pk_lag.values[crate::types::PK_IDX_LAGTIME] = lagtime;
    let pk_nolag = make_pk_params(10.0, 100.0);

    // Sample several t > lagtime so both curves are defined.
    for &t in &[2.0, 5.0, 11.0, 13.5] {
        let c_lag = predict_concentration(PkModel::OneCptIv, &doses, t, &pk_lag);
        let c_no = predict_concentration(PkModel::OneCptIv, &doses, t - lagtime, &pk_nolag);
        assert_relative_eq!(c_lag, c_no, epsilon = 1e-12);
    }
}

/// NONMEM cross-check for SS + ALAG1 (issue #15). Reference PRED values
/// were generated with NONMEM 7.5.1 (`ADVAN2 TRANS2`, `$ESTIMATION
/// MAXEVAL=0`) for a 1-cpt oral model: CL=2, V=20, KA=1.5, ALAG1=1.5,
/// a single `SS=1, II=24, AMT=100` dose into the depot, observed in the
/// central compartment (`S2=V`). Control file + dataset documented in
/// `tests/ss_lagtime_nonmem.rs`.
///
/// The two earliest samples (t=0.5, 1.0 < ALAG1) exercise the
/// previous-interval steady-state tail: NONMEM reports ~0.59 / ~0.56
/// there, not 0. This is the analytical-path coverage of the wrap added
/// to `predict_concentration`.
#[test]
fn test_ss_oral_with_lagtime_matches_nonmem() {
    let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
    let doses = vec![DoseEvent::new(0.0, amt, 1, 0.0, true, ii)];
    let mut pk = oral_pk_params(cl, v, ka);
    pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

    // (observation time, NONMEM PRED). t < 1.5 is the previous-interval
    // tail; t >= 1.5 is the current interval; t > II decays (naked SS).
    let nonmem: &[(f64, f64)] = &[
        (0.5, 0.59069),
        (1.0, 0.56188),
        (2.0, 3.07370),
        (4.0, 4.46240),
        (8.0, 3.07540),
        (12.0, 2.06170),
        (18.0, 1.13150),
        (23.0, 0.68628),
        (25.0, 0.56188),
        (30.0, 0.34080),
    ];
    for &(t, pred) in nonmem {
        let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
        assert_relative_eq!(c, pred, max_relative = 1e-4);
    }
}

#[test]
fn test_ss_dose_does_not_contribute_before_its_record_time() {
    // SS=1 establishes steady state *at* the dose record time, so an SS
    // dose must not contribute to observations earlier than its own
    // record (the pre-arrival wrap only applies in [dose.time, t_eff)).
    // Regression guard: before the `t >= dose.time` gate, a future SS
    // record leaked a non-zero steady-state tail into earlier times.
    let (cl, v, ka, lagtime, ii, amt) = (2.0, 20.0, 1.5, 1.5, 24.0, 100.0);
    // SS dose recorded at t = 10 (not at the origin).
    let doses = vec![DoseEvent::new(10.0, amt, 1, 0.0, true, ii)];
    let mut pk = oral_pk_params(cl, v, ka);
    pk.values[crate::types::PK_IDX_LAGTIME] = lagtime;

    // Strictly before the record time → no contribution.
    for &t in &[0.0, 5.0, 9.9] {
        let c = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
        assert_eq!(c, 0.0, "future SS dose leaked into t={t}");
    }
    // In the record-to-arrival window [10, 11.5) → previous-interval tail.
    let c_pre = predict_concentration(PkModel::OneCptOral, &doses, 10.5, &pk);
    let dose0 = DoseEvent::new(0.0, amt, 1, 0.0, true, ii);
    // phase = (10.5 - 11.5) + II = 23.0
    let expected = one_cpt_oral_ss(&dose0, 23.0, cl, v, ka);
    assert_relative_eq!(c_pre, expected, max_relative = 1e-9);
}

#[test]
fn test_ss_oral_via_predict_dispatches_to_ss_formula() {
    // With SS=1 and II large enough that exp(-k·II) is small, the SS
    // value should be close to but strictly above the single-dose value
    // at the same τ (the "+1" geometric-tail extra contribution).
    let ii = 24.0_f64;
    let cl = 2.0;
    let v = 20.0;
    let ka = 1.5;

    let doses_ss = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, true, ii)];
    let doses_single = vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)];
    let pk = oral_pk_params(cl, v, ka);

    let t = 4.0;
    let c_ss = predict_concentration(PkModel::OneCptOral, &doses_ss, t, &pk);
    let c_single = predict_concentration(PkModel::OneCptOral, &doses_single, t, &pk);
    assert!(
        c_ss > c_single,
        "SS conc {} should be greater than single-dose conc {}",
        c_ss,
        c_single
    );
    // With II = 24 and k = 0.1, KA = 1.5: exp(-k·II) ≈ 0.091, so the SS
    // tail adds ~10% to the slow term. Sanity: under 50%.
    assert!((c_ss / c_single) < 1.5);
}

// ── SS=1 resets the system at its record (#1576) ─────────────────────────
//
// The oracle is the closed form, written out here rather than read from the
// engine: `C_single(τ) = D·ka/(V(ka−k))·(e^{−kτ} − e^{−ka·τ})` and
// `C_ss(τ) = D·ka/(V(ka−k))·(e^{−kτ}/(1−e^{−k·II}) − e^{−ka·τ}/(1−e^{−ka·II}))`.
// NONMEM ADVAN2 equals `C_ss` to 7.8e-16 on every reset row
// (`nonmem_anchor/ss_reset`); these pin the analytic superposition to it.
const SS_RESET_P: (f64, f64, f64, f64, f64) = (2.0, 20.0, 0.15, 100.0, 12.0); // CL V KA D II

fn ss_reset_c_single(tau: f64) -> f64 {
    let (cl, v, ka, d, _) = SS_RESET_P;
    let k = cl / v;
    d * ka / (v * (ka - k)) * ((-k * tau).exp() - (-ka * tau).exp())
}
fn ss_reset_c_ss(tau: f64) -> f64 {
    let (cl, v, ka, d, ii) = SS_RESET_P;
    let k = cl / v;
    d * ka / (v * (ka - k))
        * ((-k * tau).exp() / (1.0 - (-k * ii).exp())
            - (-ka * tau).exp() / (1.0 - (-ka * ii).exp()))
}
fn ss_reset_bolus(t: f64) -> DoseEvent {
    DoseEvent::new(t, SS_RESET_P.3, 1, 0.0, false, 0.0)
}
fn ss_reset_ss(t: f64) -> DoseEvent {
    DoseEvent::new(t, SS_RESET_P.3, 1, 0.0, true, SS_RESET_P.4)
}
fn ss_reset_pk() -> PkParams {
    oral_pk_params(SS_RESET_P.0, SS_RESET_P.1, SS_RESET_P.2)
}
fn ss_reset_check(label: &str, doses: &[DoseEvent], cases: &[(f64, f64)]) {
    let pk = ss_reset_pk();
    for &(t, want) in cases {
        let got = predict_concentration(PkModel::OneCptOral, doses, t, &pk);
        assert!(got.is_finite(), "{label}: non-finite at t={t}");
        let rel = ((got - want) / want).abs();
        assert!(rel < 1e-11, "{label}: t={t} got {got} want {want} (rel {rel:e})");
    }
}

/// Row 5: a bolus before a mid-timeline `SS=1` record is dead at and after it
/// (was +10 … +24 %), and fully live before it.
#[test]
fn an_ss_record_wipes_an_earlier_bolus_in_analytic_superposition() {
    let doses = [ss_reset_bolus(0.0), ss_reset_ss(24.0)];
    ss_reset_check(
        "row 5",
        &doses,
        &[
            (10.0, ss_reset_c_single(10.0)),
            (24.5, ss_reset_c_ss(0.5)),
            (30.0, ss_reset_c_ss(6.0)),
            (35.0, ss_reset_c_ss(11.0)),
        ],
    );
}

/// Row 41, the TDM "SS on every visit" idiom (was +91 % at t = 12.5): only the
/// latest reached record is live, so every interval reads `C_ss` exactly.
#[test]
fn ss_on_every_dose_record_reads_the_steady_state_in_every_interval() {
    let doses: Vec<_> = (0..4).map(|i| ss_reset_ss(12.0 * i as f64)).collect();
    ss_reset_check(
        "row 41",
        &doses,
        &[
            (6.0, ss_reset_c_ss(6.0)),
            (12.5, ss_reset_c_ss(0.5)),
            (30.0, ss_reset_c_ss(6.0)),
            (40.0, ss_reset_c_ss(4.0)),
            (60.0, ss_reset_c_ss(24.0)),
        ],
    );
}

/// Rows 51/52 in one test, both sides of the tie: a bolus row before a co-timed
/// `SS=1` row is wiped, one after it superposes (NONMEM-measured row order).
#[test]
fn a_co_timed_bolus_is_wiped_before_the_ss_row_and_superposes_after_it() {
    ss_reset_check(
        "row 51 (bolus row, then SS row)",
        &[ss_reset_bolus(10.0), ss_reset_ss(10.0)],
        &[(10.5, ss_reset_c_ss(0.5)), (15.0, ss_reset_c_ss(5.0))],
    );
    ss_reset_check(
        "row 52 (SS row, then bolus row)",
        &[ss_reset_ss(10.0), ss_reset_bolus(10.0)],
        &[
            (10.5, ss_reset_c_ss(0.5) + ss_reset_c_single(0.5)),
            (15.0, ss_reset_c_ss(5.0) + ss_reset_c_single(5.0)),
        ],
    );
}

/// Row 4, the control: a non-SS dose after an `SS=1` record still superposes.
#[test]
fn a_non_ss_dose_after_an_ss_record_still_superposes() {
    ss_reset_check(
        "row 4",
        &[ss_reset_ss(0.0), ss_reset_bolus(120.0)],
        &[(100.0, ss_reset_c_ss(100.0)), (125.0, ss_reset_c_ss(125.0) + ss_reset_c_single(5.0))],
    );
}

/// Superposition keys the reset on the dose **record**, so a lagged dose recorded before
/// an `SS=1` record but arriving after it is cancelled, as NONMEM does (#1587's measured
/// row: dose at 9 with `ALAG1 = 2`, `SS=1` at 10; NONMEM 3.5948 at t = 11.5, `main` +9.8 %).
/// The event-driven walkers still key on the arrival (#1587). With `lag < II` the `SS=1`
/// dose reads its pre-arrival tail `C_ss(t − rec + II − lag)` before its own arrival.
#[test]
fn a_pending_lagged_dose_is_cancelled_by_an_ss_record_in_superposition() {
    let mut pk = ss_reset_pk();
    pk.values[crate::types::PK_IDX_LAGTIME] = 2.0;
    let doses = [ss_reset_bolus(9.0), ss_reset_ss(10.0)];
    let cases = [
        (11.5, ss_reset_c_ss(11.5)), // pre-arrival: phase II − lag = 10
        (13.0, ss_reset_c_ss(1.0)),  // one hour after the SS arrival at 12
    ];
    for (t, want) in cases {
        let got = predict_concentration(PkModel::OneCptOral, &doses, t, &pk);
        let rel = ((got - want) / want).abs();
        assert!(got.is_finite() && rel < 1e-11, "t={t}: got {got} want {want} (rel {rel:e})");
    }
    // Differential: without the `SS=1` row the pending dose is live at 13.
    let alone = predict_concentration(PkModel::OneCptOral, &doses[..1], 13.0, &pk);
    assert!(alone > 0.1 * ss_reset_c_ss(1.0), "fixture degenerate: {alone}");
}

/// A central infusion running across an `SS=1` record into the depot contributes nothing
/// after the record on the superposition path (#1586's "infusion into central" row: +110 %
/// before #1576). An infusion superposition cannot express (into the depot) routes to the
/// event-driven walk, which stops it the same way since #1586
/// (`an_ss_record_stops_a_running_infusion_in_the_cached_and_the_scanned_rates`).
#[test]
fn a_central_infusion_running_across_an_ss_record_stops_in_superposition() {
    let (cl, _, _, _, _) = SS_RESET_P;
    let infusion = DoseEvent::new(0.0, 200.0, 2, 10.0, false, 0.0); // 0–20 h into central
    let doses = [infusion, ss_reset_ss(10.0)];
    let k = cl / SS_RESET_P.1;
    ss_reset_check(
        "central infusion 0–20 h, SS=1 into the depot at 10",
        &doses,
        &[
            (5.0, 10.0 / cl * (1.0 - (-k * 5.0).exp())), // live before the record
            (10.5, ss_reset_c_ss(0.5)),
            (15.0, ss_reset_c_ss(5.0)),
            (21.0, ss_reset_c_ss(11.0)),
        ],
    );
}

/// The state twin (`[derived]`, sdtab states, depot readout) on row 41: the skip in
/// `predict_concentration` alone leaves this wrong.
#[test]
fn analytical_states_reset_at_every_ss_record_too() {
    let (_, _, ka, d, ii) = SS_RESET_P;
    let subj = Subject {
        doses: (0..4).map(|i| ss_reset_ss(12.0 * i as f64)).collect(),
        ..make_subject_with_tv(HashMap::new(), vec![], vec![], 0, 0)
    };
    let times = [6.0, 12.5, 30.0, 40.0];
    let states = analytical_state_at_times(PkModel::OneCptOral, &subj, &ss_reset_pk(), &times);
    for (&t, st) in times.iter().zip(&states) {
        // Elapsed time since the latest reached record (records at 0, 12, 24, 36).
        let tau = t - (12.0 * (t / 12.0).floor()).min(36.0);
        let depot = d * (-ka * tau).exp() / (1.0 - (-ka * ii).exp());
        for (name, got, want) in [("depot", st[0], depot), ("central", st[1], ss_reset_c_ss(tau))] {
            assert!(got.is_finite(), "{name} non-finite at t={t}");
            let rel = ((got - want) / want).abs();
            assert!(rel < 1e-11, "{name} at t={t}: got {got} want {want} (rel {rel:e})");
        }
    }
}

// ── apply_scaling ───────────────────────────────────────────────────────

/// Build the same `cl_from_cr_model` test fixture but with the
/// requested `ScalingSpec` swapped in. Used by all apply_scaling tests
/// so they exercise the production signature
/// (`apply_scaling(&model, &subject, ...)`).
fn model_with_scaling(scaling: crate::types::ScalingSpec) -> crate::types::CompiledModel {
    let mut m = cl_from_cr_model();
    m.scaling = scaling;
    m
}

fn one_subject_for_scaling() -> Subject {
    let mut cov = HashMap::new();
    cov.insert("CR".to_string(), 1.0);
    make_subject_with_tv(cov, Vec::new(), Vec::new(), 1, 3)
}

#[test]
fn test_apply_scaling_none_is_noop() {
    let model = model_with_scaling(ScalingSpec::None);
    let subj = one_subject_for_scaling();
    let mut preds = vec![10.0, 20.0, 30.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    assert_eq!(preds, vec![10.0, 20.0, 30.0]);
}

// ── analytic Form C readout (#650) ───────────────────────────────────────

/// A degenerate `y = central / V` readout must reproduce the built-in
/// concentration bit-for-bit (central amount = conc × V, so `/V` recovers
/// conc), and an additive term must shift each prediction by exactly that
/// amount — proving the readout replaces the concentration output correctly.
#[test]
fn analytic_form_c_readout_matches_and_shifts_concentration() {
    use crate::parser::model_parser::parse_model_string;
    let base = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
";
    let plain = parse_model_string(base).expect("plain parses");
    let ident = parse_model_string(&format!("{base}[scaling]\n  y = central / V\n"))
        .expect("identity readout parses");
    let shifted = parse_model_string(&format!("{base}[scaling]\n  y = central / V + 2.0\n"))
        .expect("shifted readout parses");
    assert!(ident.analytic_readout.is_some());

    let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
    let theta = [1.0, 50.0];
    let c_plain = compute_predictions_with_tv(&plain, &subj, &theta, &[0.0]);
    let c_ident = compute_predictions_with_tv(&ident, &subj, &theta, &[0.0]);
    let c_shift = compute_predictions_with_tv(&shifted, &subj, &theta, &[0.0]);

    assert_eq!(c_plain.len(), 6);
    for i in 0..c_plain.len() {
        assert_relative_eq!(c_ident[i], c_plain[i], max_relative = 1e-12);
        assert_relative_eq!(c_shift[i], c_plain[i] + 2.0, epsilon = 1e-9);
        assert!(c_plain[i] > 0.0, "sanity: positive concentration");
    }
}

/// The readout's covariate is read from the **per-observation** snapshot, so a
/// per-row flag (here `SEL`) selects the branch on that row — the analytic
/// analogue of NONMEM's per-record `$ERROR`. Rows with `SEL==0` read `central/V`
/// (concentration); rows with `SEL==1` read twice that.
#[test]
fn analytic_form_c_readout_uses_per_obs_covariate() {
    use crate::parser::model_parser::parse_model_string;
    let src = "\
[parameters]
  theta TVCL(1.0, 0.001, 100.0)
  theta TVV(50.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[scaling]
  y = if (SEL == 0) central / V else 2 * central / V
[error_model]
  DV ~ proportional(PROP)
";
    let model = parse_model_string(src).expect("parses");
    // 4 observations, alternating SEL flag per row.
    let obs_cov = vec![
        HashMap::from([("SEL".to_string(), 0.0)]),
        HashMap::from([("SEL".to_string(), 1.0)]),
        HashMap::from([("SEL".to_string(), 0.0)]),
        HashMap::from([("SEL".to_string(), 1.0)]),
    ];
    let subj = make_subject_with_tv(HashMap::new(), Vec::new(), obs_cov, 1, 4);
    let theta = [1.0, 50.0];
    let preds = compute_predictions_with_tv(&model, &subj, &theta, &[0.0]);
    // Odd rows (SEL==1) are exactly double the even rows' readout only if the
    // underlying concentration were equal — instead compare each row to its own
    // concentration via the ratio of the SEL==1 formula (2×) applied on its conc.
    assert_relative_eq!(
        preds[1] / preds[0],
        2.0 * (preds_conc(&subj, &theta, 1) / preds_conc(&subj, &theta, 0)),
        max_relative = 1e-9
    );
    assert_relative_eq!(
        preds[3] / preds[2],
        2.0 * (preds_conc(&subj, &theta, 3) / preds_conc(&subj, &theta, 2)),
        max_relative = 1e-9
    );
}

/// Helper: the plain built-in concentration at observation `i` for the fixture
/// above (CL=1, V=50, single 100 mg IV bolus at t=0).
fn preds_conc(subj: &Subject, theta: &[f64], i: usize) -> f64 {
    let mut pk = PkParams::default();
    pk.values[crate::types::PK_IDX_CL] = theta[0];
    pk.values[crate::types::PK_IDX_V] = theta[1];
    predict_concentration(PkModel::OneCptIv, &subj.doses, subj.obs_times[i], &pk)
}

/// **NONMEM-validation cross-check (#650).** An analytic Form C readout
/// computes the same output as the *ODE* Form C readout, which is validated
/// against NONMEM's per-record `$ERROR` (free/total protein-binding model,
/// `docs/model-file/scaling.qmd`). On an IV 1-cpt model the analytic central
/// amount (`concentration × V`) equals the ODE `central` state amount, so a
/// saturable-binding total-vs-free readout `y = C + BMAX·C/(KD+C)` must give
/// bit-for-bit-equal predictions on the analytic and ODE paths at matched
/// parameters — inheriting the ODE path's NONMEM anchor.
#[test]
fn analytic_form_c_matches_ode_form_c_binding_readout() {
    use crate::parser::model_parser::parse_model_string;
    let readout = "  y = central / V + BMAX * (central / V) / (KD + central / V)\n";
    let params = "\
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVBMAX(3.0, 0.01, 100.0)
  theta TVKD(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  sigma PROP ~ 0.02 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV  * exp(ETA_V)
  BMAX = TVBMAX
  KD   = TVKD
";
    let analytic = parse_model_string(&format!(
        "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
    ))
    .expect("analytic parses");
    // Equivalent ODE: `central` is the amount, disposition -(CL/V)·central,
    // IV bolus into central (cmt 1). Same readout replaces the state readout.
    let ode = parse_model_string(&format!(
        "{params}[structural_model]\n  ode(states=[central])\n[odes]\n  d/dt(central) = -(CL / V) * central\n[scaling]\n{readout}[error_model]\n  DV ~ proportional(PROP)\n"
    ))
    .expect("ode parses");
    assert!(analytic.analytic_readout.is_some());
    assert!(ode.ode_spec.is_some());

    let subj = make_subject_with_tv(HashMap::new(), Vec::new(), Vec::new(), 1, 6);
    let theta = [0.2, 10.0, 3.0, 2.0];
    let eta = [0.1, -0.05];
    let a = compute_predictions_with_tv(&analytic, &subj, &theta, &eta);
    let o = compute_predictions_with_tv(&ode, &subj, &theta, &eta);
    assert_eq!(a.len(), o.len());
    for (av, ov) in a.iter().zip(o.iter()) {
        assert_relative_eq!(av, ov, max_relative = 1e-6, epsilon = 1e-9);
    }
}

/// Two-occasion IOV subject for the Form C readout cross-engine checks (#1079):
/// a dose at t = 0 (occasion 1) and at t = 24 (occasion 2), two samples inside
/// each occasion. No washout, so the occasion-2 samples carry occasion-1 drug.
fn iov_readout_subject() -> Subject {
    let obs_times = vec![1.0, 3.0, 25.0, 27.0];
    let n = obs_times.len();
    Subject {
        id: "1".to_string(),
        doses: vec![bolus_dose(0.0, 100.0), bolus_dose(24.0, 100.0)],
        obs_times,
        observations: vec![0.0; n],
        obs_cmts: vec![1; n],
        cens: vec![0; n],
        occasions: vec![1, 1, 2, 2],
        dose_occasions: vec![1, 2],
        ..Default::default()
    }
}

/// **Regression (#1079): a Form C readout under IOV must be evaluated with the observation's
/// own occasion κ.** `apply_analytic_readout` builds the readout's individual parameters from a
/// `pk_param_fn` snapshot; under IOV `predict_iov` called it once with `eta_bsv` (κ held at 0).
/// The concentration did carry the occasion's κ, but every individual parameter the readout
/// read was frozen at its typical value, so an additive κ-carrying baseline was silently
/// dropped from the prediction and objective (12-20% on the reproduction below).
///
/// The oracle is cross-engine: the hand-written ODE twin applies the same readout inside the
/// integrator against the per-observation parameter snapshot (also NONMEM's per-record
/// `$ERROR` convention). A `Dual2`-vs-FD parity test passes against the wrong function, because
/// the analytic path was seeded to match it. The decomposition is checked too: `y − central/V`
/// must equal `BASE` evaluated at *that observation's* κ, pinning the fix to the per-occasion
/// snapshot rather than a compensating error elsewhere.
#[test]
fn analytic_form_c_iov_readout_uses_occasion_kappa_matching_ode_twin() {
    use crate::parser::model_parser::parse_model_string;
    let params = "\
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  theta TVBASE(2.0, 0.01, 100.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_B ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL   = TVCL * exp(ETA_CL)
  V    = TVV
  BASE = TVBASE * exp(KAPPA_B)
";
    let tail = "\
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
  ode_reltol = 1e-12
  ode_abstol = 1e-12
";
    // Closed-form and its hand-written ODE twin, same `[scaling]` readout.
    let analytic = parse_model_string(&format!(
        "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
         [scaling]\n  y = central / V + BASE\n{tail}"
    ))
    .expect("analytic parses");
    let ode = parse_model_string(&format!(
        "{params}[structural_model]\n  ode(states=[central])\n\
         [odes]\n  d/dt(central) = -(CL / V) * central\n\
         [scaling]\n  y = central / V + BASE\n{tail}"
    ))
    .expect("ode twin parses");
    // The readout's `V` divisor cancels the `conc × V` amount reconstruction, so
    // this (natural) readout cannot show the bug — pinned below so the fix does
    // not break the load-bearing cancellation.
    let conc_only = parse_model_string(&format!(
        "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n\
         [scaling]\n  y = central / V\n{tail}"
    ))
    .expect("identity readout parses");
    // Same model with no readout at all — the built-in concentration output.
    let plain = parse_model_string(&format!(
        "{params}[structural_model]\n  pk one_cpt_iv(cl=CL, v=V)\n{tail}"
    ))
    .expect("plain parses");
    assert_eq!(analytic.n_kappa, 1, "one occasion random effect");
    assert!(analytic.analytic_readout.is_some() && analytic.ode_spec.is_none());
    assert!(ode.ode_spec.is_some());

    let subj = iov_readout_subject();
    let theta = [3.0, 20.0, 2.0];
    let eta_bsv = [0.1];
    let kappas = vec![vec![0.5], vec![-0.4]];

    let a = predict_iov(&analytic, &subj, &theta, &eta_bsv, &kappas);
    let o = predict_iov(&ode, &subj, &theta, &eta_bsv, &kappas);
    assert_eq!(a.len(), subj.obs_times.len());
    for (j, (av, ov)) in a.iter().zip(o.iter()).enumerate() {
        assert_relative_eq!(av, ov, max_relative = 1e-8, epsilon = 1e-12);
        assert!(av.is_finite(), "obs {j} finite");
    }

    // Decomposition: the readout output minus the concentration is `BASE` at the
    // observation's own occasion κ (`2·e^0.5` in occasion 1, `2·e^−0.4` in
    // occasion 2), NOT the typical value `2` the κ = 0 snapshot produced.
    let c = predict_iov(&plain, &subj, &theta, &eta_bsv, &kappas);
    let expected_base =
        [2.0 * 0.5_f64.exp(), 2.0 * 0.5_f64.exp(), 2.0 * (-0.4_f64).exp(), 2.0 * (-0.4_f64).exp()];
    for (j, (av, cv)) in a.iter().zip(c.iter()).enumerate() {
        assert_relative_eq!(av - cv, expected_base[j], max_relative = 1e-10);
        assert!(
            (av - cv - 2.0).abs() > 0.5,
            "obs {j}: baseline must not collapse to the κ = 0 typical value"
        );
    }

    // The `y = central / V` cancellation still holds: the amount is rebuilt as
    // `conc × V` with the *same* `V` the readout divides by, so the per-occasion
    // evaluation must not perturb it (the round-trip is exact to rounding).
    let ident = predict_iov(&conc_only, &subj, &theta, &eta_bsv, &kappas);
    for (iv, cv) in ident.iter().zip(c.iter()) {
        assert_relative_eq!(iv, cv, max_relative = 1e-14);
    }
}

/// The `y = central / V` cancellation with a κ **on `V` itself** (#1079 fix note
/// 3). The predictor rebuilds the central amount as `conc × V` and the readout
/// divides by the same `V`; both now come from the observation's occasion, so the
/// two must still cancel (exactly, up to the multiply/divide round-trip). Reading
/// `V` from a κ = 0 snapshot on one side only would break this by `exp(κ)`.
#[test]
fn analytic_form_c_iov_identity_readout_cancels_kappa_on_v() {
    use crate::parser::model_parser::parse_model_string;
    let params = "\
[parameters]
  theta TVCL(3.0, 0.01, 100.0)
  theta TVV(20.0, 1.0, 500.0)
  omega ETA_CL ~ 0.09
  kappa KAPPA_V ~ 0.04
  sigma PROP_ERR ~ 0.04 (sd)
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV * exp(KAPPA_V)
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
";
    let tail = "\
[error_model]
  DV ~ proportional(PROP_ERR)
[fit_options]
  iov_column = OCC
";
    let ident =
        parse_model_string(&format!("{params}[scaling]\n  y = central / V\n{tail}")).unwrap();
    let plain = parse_model_string(&format!("{params}{tail}")).unwrap();

    let subj = iov_readout_subject();
    let theta = [3.0, 20.0];
    let eta_bsv = [0.1];
    let kappas = vec![vec![0.5], vec![-0.4]];
    let y = predict_iov(&ident, &subj, &theta, &eta_bsv, &kappas);
    let c = predict_iov(&plain, &subj, &theta, &eta_bsv, &kappas);
    assert_eq!(y.len(), c.len());
    for (yv, cv) in y.iter().zip(c.iter()) {
        assert_relative_eq!(yv, cv, max_relative = 1e-14);
    }
}

#[test]
fn test_apply_log_transform_noop_when_disabled() {
    let mut model = cl_from_cr_model();
    model.log_transform = false;
    let mut preds = vec![1.0, 2.0, 4.0];
    apply_log_transform(&model, &mut preds);
    assert_eq!(preds, vec![1.0, 2.0, 4.0]);
}

#[test]
fn test_apply_log_transform_logs_with_floor() {
    let mut model = cl_from_cr_model();
    model.log_transform = true;
    let mut preds = vec![1.0, std::f64::consts::E, 0.0, -5.0];
    apply_log_transform(&model, &mut preds);
    assert_relative_eq!(preds[0], 0.0, epsilon = 1e-12);
    assert_relative_eq!(preds[1], 1.0, epsilon = 1e-12);
    // Non-positive predictions are floored, not turned into -inf/NaN.
    assert_relative_eq!(preds[2], LTBS_FLOOR.ln(), epsilon = 1e-9);
    assert_relative_eq!(preds[3], LTBS_FLOOR.ln(), epsilon = 1e-9);
    assert!(preds.iter().all(|p| p.is_finite()));
}

#[test]
fn test_ltbs_prediction_is_log_of_natural() {
    // The whole prediction pipeline (compute_predictions_with_tv) returns
    // ln(f) when LTBS is active, so log-scale predictions flow into every
    // downstream consumer (likelihood, IWRES/CWRES, IPRED, simulation).
    let subj = one_subject_for_scaling();
    let theta = [1.0];

    let mut natural_model = cl_from_cr_model();
    natural_model.log_transform = false;
    let natural = compute_predictions_with_tv(&natural_model, &subj, &theta, &[]);

    let mut ltbs_model = cl_from_cr_model();
    ltbs_model.log_transform = true;
    let logged = compute_predictions_with_tv(&ltbs_model, &subj, &theta, &[]);

    assert_eq!(natural.len(), logged.len());
    for (n, l) in natural.iter().zip(logged.iter()) {
        assert_relative_eq!(*l, n.max(LTBS_FLOOR).ln(), epsilon = 1e-9);
    }
}

#[test]
fn test_ltbs_frem_predictions_are_raw_not_logged() {
    // Regression: FREM covariate predictions must be raw (theta + eta),
    // NOT log-transformed, even when the PK error model uses log_additive.
    // The override must happen AFTER apply_log_transform so the log does
    // not corrupt covariate predictions.
    let mut model = cl_from_cr_model();
    model.log_transform = true; // LTBS (log-additive)
    model.n_eta = 2;
    model.eta_names = vec!["ETA_CL".into(), "ETA_COV".into()];
    model.default_params.omega = crate::types::OmegaMatrix::from_diagonal(
        &[0.1, 100.0],
        vec!["ETA_CL".into(), "ETA_COV".into()],
    );
    // FREM config: FREMTYPE 100 -> (theta_idx=0 [TVCL], eta_idx=1 [ETA_COV])
    let mut map = std::collections::HashMap::new();
    map.insert(100u16, (0usize, 1usize));
    model.frem_config =
        Some(crate::types::FremConfig { fremtype_to_indices: map, covariate_sigma_index: 0 });

    let theta = [1.0]; // TVCL = 1.0
    let eta = [0.0, 5.0]; // ETA_COV = 5.0
                          // Expected FREM prediction: theta[0] + eta[1] = 1.0 + 5.0 = 6.0 (raw, NOT logged)

    // Subject with 2 obs: FREMTYPE 0 (PK) and FREMTYPE 100 (covariate)
    let mut subj = one_subject_for_scaling(); // has 3 obs
    subj.fremtype = vec![0, 100, 0];

    let preds = compute_predictions_with_tv(&model, &subj, &theta, &eta);
    // PK rows (0, 2) should be log-transformed
    assert!(preds[0].is_finite());
    // FREM row (1) must be raw: theta[0] + eta[1] = 6.0, not ln(6.0)
    assert_relative_eq!(preds[1], 6.0, epsilon = 1e-10);
}

#[test]
fn test_apply_scaling_scalar_divides_in_place() {
    let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
    let subj = one_subject_for_scaling();
    let mut preds = vec![1000.0, 2000.0, 3000.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    for (got, exp) in preds.iter().zip([1.0, 2.0, 3.0].iter()) {
        assert_relative_eq!(got, exp, epsilon = 1e-12);
    }
}

/// Regression (#769 review): `predict_iov` must apply `[scaling]` to a subject
/// that carries *dose* occasions but no *observation* occasions. In that state
/// `iov_occasion_groups` is non-empty — it appends the dose-only occasion as a
/// group with an EMPTY obs-index list — yet no observation is owned by any
/// group, so the per-occasion scaling loop writes nothing. The base pass is
/// what scales those observations; guarding it on `occ_groups.is_empty()`
/// (rather than `subject.occasions.is_empty()`) skips it here and returns the
/// predictions UNSCALED, off by the full scale factor — the #723 bug.
///
/// The current data readers never build this state (they gate observation- and
/// dose-occasion labels on the same column), so it is constructed directly to
/// lock the guard against a future path that produces it.
#[test]
fn predict_iov_scales_subject_with_dose_only_occasions() {
    let scale = 1000.0;
    let mut model = cl_from_cr_model();
    // A consistent IOV model — `dose_occasions` are only populated for IOV
    // models. The κ block stays zero here, which is all this guard needs.
    model.n_kappa = 1;
    model.kappa_names = vec!["KAPPA_CL".to_string()];

    // Observations present + a dose occasion, but NO observation occasions.
    let mut dose_only = one_subject_for_scaling();
    dose_only.dose_occasions = vec![1; dose_only.doses.len()];
    assert!(
        dose_only.occasions.is_empty() && !dose_only.dose_occasions.is_empty(),
        "trigger state: dose occasions present, observation occasions absent"
    );

    // Unscaled reference (scaling off) — the predictions the scale divides.
    model.scaling = ScalingSpec::None;
    let raw = predict_iov(&model, &dose_only, &[1.0], &[], &[]);
    assert!(
        !raw.is_empty() && raw.iter().all(|v| v.is_finite() && *v > 0.0),
        "sanity: positive concentrations to scale, got {raw:?}"
    );

    // Scaling on: every prediction must be divided by `scale`, not left as-is.
    // Fails on the `occ_groups.is_empty()` guard (returns `raw`, `scale`× too big).
    model.scaling = ScalingSpec::ScalarScale(scale);
    let got = predict_iov(&model, &dose_only, &[1.0], &[], &[]);
    assert_eq!(got.len(), raw.len());
    for (g, r) in got.iter().zip(raw.iter()) {
        assert_relative_eq!(*g, r / scale, epsilon = 1e-12);
    }

    // And identical to the fully occasion-less subject (the known-correct #723
    // base-pass path, unchanged by #769): dose-only occasions must not move it.
    let occasionless = one_subject_for_scaling(); // occasions=[] AND dose_occasions=[]
    let reference = predict_iov(&model, &occasionless, &[1.0], &[], &[]);
    assert_eq!(got.len(), reference.len());
    for (g, r) in got.iter().zip(reference.iter()) {
        assert_relative_eq!(*g, r, epsilon = 1e-12);
    }

    // Mixed subject — observations DO carry occasions, plus a second dose in an
    // occasion with no co-timed sample. This takes the per-occasion (`else`)
    // branch and exercises the dose-only-group `continue`: the empty group is
    // skipped while every observation is still scaled per-occasion.
    let mut mixed = one_subject_for_scaling();
    mixed.doses.push(bolus_dose(0.0, 100.0));
    mixed.occasions = vec![1, 1, 1]; // all observations in occasion 1
    mixed.dose_occasions = vec![1, 2]; // second dose in occasion 2 (no observations)
    model.scaling = ScalingSpec::None;
    let raw_mixed = predict_iov(&model, &mixed, &[1.0], &[], &[]);
    assert!(raw_mixed.iter().all(|v| v.is_finite() && *v > 0.0));
    model.scaling = ScalingSpec::ScalarScale(scale);
    let got_mixed = predict_iov(&model, &mixed, &[1.0], &[], &[]);
    assert_eq!(got_mixed.len(), raw_mixed.len());
    for (g, r) in got_mixed.iter().zip(raw_mixed.iter()) {
        assert_relative_eq!(*g, r / scale, epsilon = 1e-12);
    }
}

#[test]
fn test_apply_scaling_marks_bad_scale_as_nan() {
    // When a ScaleFn returns 0 / NaN / inf at runtime, apply_scaling
    // replaces every prediction with NaN so the outer NLL goes NaN
    // and the optimizer rejects the step. Loud failure mode (vs the
    // earlier "skip silently" policy, which hid mis-scaled fits).
    let bad_returns = [0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY];
    for &val in &bad_returns {
        let scale_fn: crate::types::ScaleFn = Box::new(move |_, _, _, _| val);
        let model = model_with_scaling(ScalingSpec::ExpressionScale { scale_fn, deriv: None });
        let subj = one_subject_for_scaling();
        let mut preds = vec![1.0, 2.0, 3.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        for p in &preds {
            assert!(p.is_nan(), "bad scale ({}) must produce NaN preds, got {}", val, p);
        }
    }
}

#[test]
fn test_apply_scaling_scalar_bad_value_marks_nan() {
    // Defensive guard on the ScalarScale branch (hand-constructed
    // models bypass the parser's > 0 check).
    for &k in &[0.0_f64, -1.0_f64, f64::NAN, f64::INFINITY] {
        let model = model_with_scaling(ScalingSpec::ScalarScale(k));
        let subj = one_subject_for_scaling();
        let mut preds = vec![1.0, 2.0, 3.0];
        apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
        for p in &preds {
            assert!(p.is_nan(), "bad scalar scale ({}) must produce NaN preds, got {}", k, p);
        }
    }
}

#[test]
fn test_apply_scaling_expression_uses_covariate() {
    // scale_fn = WT / 70 — divisive convention so preds / (WT/70).
    let scale_fn: crate::types::ScaleFn =
        Box::new(|_theta, _eta, cov: &HashMap<String, f64>, _pk: &PkParams| {
            cov.get("WT").copied().unwrap_or(70.0) / 70.0
        });
    let model = model_with_scaling(ScalingSpec::ExpressionScale { scale_fn, deriv: None });
    let mut subj = one_subject_for_scaling();
    subj.covariates.insert("WT".to_string(), 84.0); // scale = 84/70 = 1.2
    let mut preds = vec![12.0, 24.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    assert_relative_eq!(preds[0], 10.0, epsilon = 1e-12); // 12 / 1.2
    assert_relative_eq!(preds[1], 20.0, epsilon = 1e-12); // 24 / 1.2
}

#[test]
fn test_apply_scaling_expression_uses_pk_param() {
    // Regression for Phase 1.5 lift: scale_fn can now read individual
    // parameters from `pk`. Here `scale = V / 10` — V is at PK slot 1
    // (PK_IDX_V) for the cl_from_cr model fixture (CL=slot 0, V=slot 1).
    // The model sets V = 50 so scale = 5 and preds get divided by 5.
    let scale_fn: crate::types::ScaleFn =
        Box::new(|_theta, _eta, _cov: &HashMap<String, f64>, pk: &PkParams| {
            pk.values[crate::types::PK_IDX_V] / 10.0
        });
    let model = model_with_scaling(ScalingSpec::ExpressionScale { scale_fn, deriv: None });
    let subj = one_subject_for_scaling();
    let mut preds = vec![100.0, 50.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    // cl_from_cr_model sets V = 50, so scale = 5.
    assert_relative_eq!(preds[0], 20.0, epsilon = 1e-12);
    assert_relative_eq!(preds[1], 10.0, epsilon = 1e-12);
}

// ── PerCmt dispatch ────────────────────────────────────────────────────

#[test]
fn test_apply_scaling_per_cmt_dispatches_per_observation() {
    // Two observations on different CMTs get scaled by different
    // factors. CMT=1 → /1000, CMT=2 → /1.
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    map.insert(2, ScalingSpec::ScalarScale(1.0));
    let model = model_with_scaling(ScalingSpec::PerCmt(map));

    let mut subj = one_subject_for_scaling();
    // The helper produces 3 obs all on CMT=1; override one to CMT=2.
    assert!(subj.obs_cmts.len() >= 2);
    subj.obs_cmts[0] = 1;
    subj.obs_cmts[1] = 2;
    subj.obs_cmts[2] = 1;

    let mut preds = vec![1000.0, 50.0, 2000.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12); // 1000 / 1000
    assert_relative_eq!(preds[1], 50.0, epsilon = 1e-12); // 50 / 1
    assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12); // 2000 / 1000
}

#[test]
fn test_apply_scaling_per_cmt_missing_cmt_yields_nan() {
    // PerCmt map covers CMT=1 only, but obs[1] is on CMT=2.
    // Pre-fit validation catches this at fit() entry, but apply_scaling
    // is the defensive last line — must produce NaN, not silent zero.
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    let model = model_with_scaling(ScalingSpec::PerCmt(map));

    let mut subj = one_subject_for_scaling();
    subj.obs_cmts[0] = 1;
    subj.obs_cmts[1] = 2; // not in map → NaN
    subj.obs_cmts[2] = 1;

    let mut preds = vec![1000.0, 50.0, 2000.0];
    apply_scaling(&model, &subj, &[10.0], &[], &mut preds);
    assert_relative_eq!(preds[0], 1.0, epsilon = 1e-12);
    assert!(preds[1].is_nan(), "missing CMT must produce NaN");
    assert_relative_eq!(preds[2], 2.0, epsilon = 1e-12);
}

#[test]
fn test_validate_per_cmt_scaling_errors_on_missing_entry() {
    // The pre-fit validation should produce a clear error naming the
    // missing CMTs rather than relying on the runtime NaN fallback.
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    let model = model_with_scaling(ScalingSpec::PerCmt(map));

    let mut subj = one_subject_for_scaling();
    subj.obs_cmts = vec![1, 2, 3]; // 2 and 3 missing from map

    let err = validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
        .expect_err("missing CMTs must error");
    assert!(err.contains("[2, 3]"), "error should list missing CMTs explicitly, got: {}", err);
}

#[test]
fn test_validate_per_cmt_scaling_passes_when_complete() {
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    map.insert(2, ScalingSpec::ScalarScale(1.0));
    let model = model_with_scaling(ScalingSpec::PerCmt(map));

    let mut subj = one_subject_for_scaling();
    subj.obs_cmts = vec![1, 2, 1];

    validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
        .expect("complete PerCmt coverage must pass validation");
}

#[test]
fn test_validate_per_cmt_scaling_noop_for_non_per_cmt() {
    // Non-PerCmt scaling never errors regardless of observed CMTs.
    let model = model_with_scaling(ScalingSpec::ScalarScale(1000.0));
    let subj = one_subject_for_scaling();
    validate_per_cmt_scaling(&model, std::slice::from_ref(&subj))
        .expect("non-PerCmt scaling never errors");
}

// ── build_obs_scale_array (Phase 2.5: shared FD/AD materialiser) ───────

#[test]
fn test_build_obs_scale_array_none() {
    let spec = ScalingSpec::None;
    let pk = PkParams::default();
    let cov = HashMap::new();
    let obs_cmts = vec![1, 1, 2];
    let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
    assert_eq!(arr, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_build_obs_scale_array_scalar() {
    let spec = ScalingSpec::ScalarScale(1000.0);
    let pk = PkParams::default();
    let cov = HashMap::new();
    let obs_cmts = vec![1; 4];
    let arr = spec.build_obs_scale_array(&[], &[], &cov, &pk, &obs_cmts);
    assert_eq!(arr, vec![1000.0; 4]);
}

#[test]
fn test_build_obs_scale_array_scalar_bad_value_yields_nan() {
    // Defensive: hand-constructed ScalarScale with invalid k materialises
    // NaN — feeds the FD path's loud-failure semantic and the AD path
    // gets a Const NaN that propagates to NaN gradient too.
    for &k in &[0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        let spec = ScalingSpec::ScalarScale(k);
        let pk = PkParams::default();
        let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
        assert!(arr.iter().all(|s| s.is_nan()), "bad k={} → all NaN", k);
    }
}

#[test]
fn test_build_obs_scale_array_expression() {
    // Closure returns 5.0; array should be 5.0 for every obs.
    let scale_fn: crate::types::ScaleFn = Box::new(|_, _, _, _| 5.0);
    let spec = ScalingSpec::ExpressionScale { scale_fn, deriv: None };
    let pk = PkParams::default();
    let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &[1; 3]);
    assert_eq!(arr, vec![5.0; 3]);
}

#[test]
fn test_build_obs_scale_array_per_cmt_dispatches() {
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    map.insert(2, ScalingSpec::ScalarScale(2.0));
    let spec = ScalingSpec::PerCmt(map);
    let pk = PkParams::default();
    let obs_cmts = vec![1, 2, 1, 2];
    let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
    assert_eq!(arr, vec![1000.0, 2.0, 1000.0, 2.0]);
}

#[test]
fn test_build_obs_scale_array_per_cmt_missing_cmt_is_nan() {
    // CMT=3 is observed but not in the map. The runtime fit-time
    // validation should catch this earlier; if it doesn't (hand-built
    // CompiledModel), the array materialises NaN at the missing slot.
    let mut map: HashMap<usize, ScalingSpec> = HashMap::new();
    map.insert(1, ScalingSpec::ScalarScale(1000.0));
    let spec = ScalingSpec::PerCmt(map);
    let pk = PkParams::default();
    let obs_cmts = vec![1, 3, 1];
    let arr = spec.build_obs_scale_array(&[], &[], &HashMap::new(), &pk, &obs_cmts);
    assert_eq!(arr[0], 1000.0);
    assert!(arr[1].is_nan());
    assert_eq!(arr[2], 1000.0);
}

// ── Mixed bolus + infusion within a single subject (issue #176) ─────────
//
// Regression guard for the silent-wrong-answer bug on the superposition
// path: before #176, `single_dose_concentration` branched on the model
// enum (`OneCptIvBolus` vs `OneCptInfusion`), so an infusion event under
// an IV-bolus model was treated as an instantaneous bolus of `AMT`,
// ignoring the duration entirely. After #176 the dispatch picks per
// dose from `dose.is_infusion()`, so both routes coexist in one record.

#[test]
fn test_mixed_bolus_and_infusion_in_single_subject_superposition() {
    // Same `OneCptIv` model, two doses with different administration:
    //   t=0:  bolus     AMT=100, RATE=0
    //   t=4:  infusion  AMT=500, RATE=50 → duration=10
    // The superposition prediction must equal the sum of the two
    // single-dose closed forms (one bolus, one infusion).
    let bolus = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let infusion = DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0);
    debug_assert!(!bolus.is_infusion() && infusion.is_infusion());
    debug_assert!((infusion.duration - 10.0).abs() < 1e-12);
    let doses = vec![bolus.clone(), infusion.clone()];
    let pk = make_pk_params(10.0, 100.0);

    // Probe times that fall in three regimes:
    //   t = 2.0  → only the bolus has been given
    //   t = 8.0  → both active, infusion still running
    //   t = 20.0 → infusion has ended, post-infusion decay
    for &t in &[2.0, 8.0, 20.0] {
        let combined = predict_concentration(PkModel::OneCptIv, &doses, t, &pk);
        let from_bolus = predict_concentration(PkModel::OneCptIv, &[bolus.clone()], t, &pk);
        let from_infusion = predict_concentration(PkModel::OneCptIv, &[infusion.clone()], t, &pk);
        assert_relative_eq!(combined, from_bolus + from_infusion, epsilon = 1e-12);
    }
}

#[test]
fn test_mixed_dose_superposition_matches_event_driven_path() {
    // Cross-check: the analytical superposition path and the
    // event-driven analytical path must agree to roundoff on a
    // mixed-administration record. Before #176 the superposition path
    // was silently wrong (bolus formula applied to the infusion dose),
    // while the event-driven path was already correct — so these two
    // would have disagreed materially.
    use crate::pk::event_driven::event_driven_predictions;
    use crate::types::Subject;

    let doses = vec![
        DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0), // bolus
        DoseEvent::new(4.0, 500.0, 1, 50.0, false, 0.0), // infusion (dur=10)
    ];
    let obs_times = vec![1.0_f64, 3.0, 6.0, 10.0, 14.0, 20.0];
    let pk = make_pk_params(10.0, 100.0);

    let analytical: Vec<f64> = obs_times
        .iter()
        .map(|&t| predict_concentration(PkModel::OneCptIv, &doses, t, &pk))
        .collect();

    let n_obs = obs_times.len();
    let subject = Subject {
        id: "mixed".into(),
        doses: doses.clone(),
        obs_times: obs_times.clone(),
        observations: vec![0.0; n_obs],
        obs_cmts: vec![1; n_obs],
        cens: vec![0; n_obs],
        ..Default::default()
    };
    let pk_dose: Vec<PkParams> = vec![pk.clone(); doses.len()];
    let pk_obs: Vec<PkParams> = vec![pk.clone(); obs_times.len()];
    let event_driven =
        event_driven_predictions(PkModel::OneCptIv, &subject, &pk_dose, &pk_obs, &[]);

    for (i, (&a, &e)) in analytical.iter().zip(event_driven.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < 1e-10,
            "mismatch at obs[{i}] t={}: superposition {a} vs event-driven {e} (|Δ|={diff})",
            obs_times[i]
        );
    }
}

// -----------------------------------------------------------------------
// Regression tests for single_dose_states SS dispatch
// -----------------------------------------------------------------------

/// Bug regression: single_dose_states must use the SS formula for the
/// central compartment when dose.ss = true.  Before the fix, oral and
/// IV-infusion SS doses fell through to single-dose formulas, returning
/// concentrations that were typically 5-15× too low at trough.
///
/// For each model the test checks that the SS central state matches
/// single_dose_concentration (the long-standing, validated scalar path).
#[test]
fn single_dose_states_central_matches_concentration_for_ss_doses() {
    use crate::pk::{
        one_cpt_infusion_ss, one_cpt_iv_bolus_ss, one_cpt_oral_f_ss, two_cpt_infusion_ss,
        two_cpt_iv_bolus_ss, two_cpt_oral_f_ss,
    };
    // --- helpers ---
    fn bolus_ss(amt: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, true, ii)
    }
    fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    let cl = 5.0_f64;
    let v = 80.0_f64;
    let q = 2.0_f64;
    let v2 = 40.0_f64;
    let ka = 1.0_f64;
    let f_bio = 1.0_f64;
    let ii = 24.0_f64;
    let amt = 100.0_f64;
    let rate = 50.0_f64; // 2 h infusion
    let tau = 6.0_f64;

    let mut p1iv = PkParams::default();
    p1iv.values[crate::types::PK_IDX_CL] = cl;
    p1iv.values[crate::types::PK_IDX_V] = v;

    let mut p1oral = PkParams::default();
    p1oral.values[crate::types::PK_IDX_CL] = cl;
    p1oral.values[crate::types::PK_IDX_V] = v;
    p1oral.values[crate::types::PK_IDX_KA] = ka;
    p1oral.values[crate::types::PK_IDX_F] = f_bio;

    let mut p2iv = PkParams::default();
    p2iv.values[crate::types::PK_IDX_CL] = cl;
    p2iv.values[crate::types::PK_IDX_V] = v;
    p2iv.values[crate::types::PK_IDX_Q] = q;
    p2iv.values[crate::types::PK_IDX_V2] = v2;

    let mut p2oral = PkParams::default();
    p2oral.values[crate::types::PK_IDX_CL] = cl;
    p2oral.values[crate::types::PK_IDX_V] = v;
    p2oral.values[crate::types::PK_IDX_Q] = q;
    p2oral.values[crate::types::PK_IDX_V2] = v2;
    p2oral.values[crate::types::PK_IDX_KA] = ka;
    p2oral.values[crate::types::PK_IDX_F] = f_bio;

    // OneCptIv SS bolus — central index 0
    let d = bolus_ss(amt, ii);
    let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
    let expected = one_cpt_iv_bolus_ss(&d, tau, cl, v);
    assert!(
        approx::relative_eq!(states[0], expected, max_relative = 1e-9),
        "OneCptIv SS bolus: central mismatch"
    );

    // OneCptIv SS infusion — was using non-SS formula before fix
    let d = infusion_ss(amt, rate, ii);
    let states = single_dose_states(PkModel::OneCptIv, &d, tau, &p1iv);
    let expected = one_cpt_infusion_ss(&d, tau, cl, v);
    assert!(
        approx::relative_eq!(states[0], expected, max_relative = 1e-9),
        "OneCptIv SS infusion: central mismatch"
    );

    // OneCptOral SS bolus — was using one_cpt_oral_f (non-SS) before fix
    let d = bolus_ss(amt, ii);
    let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p1oral);
    let expected = one_cpt_oral_f_ss(&d, tau, cl, v, ka, f_bio);
    assert!(
        approx::relative_eq!(states[1], expected, max_relative = 1e-9),
        "OneCptOral SS: central (index 1) mismatch"
    );

    // TwoCptIv SS bolus
    let d = bolus_ss(amt, ii);
    let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
    let expected = two_cpt_iv_bolus_ss(&d, tau, cl, v, q, v2);
    assert!(
        approx::relative_eq!(states[0], expected, max_relative = 1e-9),
        "TwoCptIv SS bolus: central mismatch"
    );

    // TwoCptIv SS infusion — was using two_cpt_infusion (non-SS) before fix
    let d = infusion_ss(amt, rate, ii);
    let states = single_dose_states(PkModel::TwoCptIv, &d, tau, &p2iv);
    let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
    assert!(
        approx::relative_eq!(states[0], expected, max_relative = 1e-9),
        "TwoCptIv SS infusion: central mismatch"
    );

    // TwoCptOral SS bolus — was using two_cpt_oral_f (non-SS) before fix
    let d = bolus_ss(amt, ii);
    let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
    let expected = two_cpt_oral_f_ss(&d, tau, cl, v, q, v2, ka, f_bio);
    assert!(
        approx::relative_eq!(states[1], expected, max_relative = 1e-9),
        "TwoCptOral SS bolus: central (index 1) mismatch"
    );

    // TwoCptOral SS infusion central — was using two_cpt_infusion (non-SS) before fix
    let d = infusion_ss(amt, rate, ii);
    let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p2oral);
    let expected = two_cpt_infusion_ss(&d, tau, cl, v, q, v2);
    assert!(
        approx::relative_eq!(states[1], expected, max_relative = 1e-9),
        "TwoCptOral SS infusion: central (index 1) mismatch"
    );

    // SS states must exceed single-dose states (accumulation guard).
    let d_single = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
    let states_ss = single_dose_states(PkModel::OneCptOral, &bolus_ss(amt, ii), tau, &p1oral);
    let states_sd = single_dose_states(PkModel::OneCptOral, &d_single, tau, &p1oral);
    assert!(
        states_ss[1] > states_sd[1],
        "SS central must be > single-dose central (accumulation): ss={} sd={}",
        states_ss[1],
        states_sd[1]
    );
}

/// Transit is a 2-state `[depot, central]` model: `single_dose_states` must return the
/// lumped unabsorbed-chain "depot" amount and the central concentration for a bolus
/// (the `[derived]` / compartment-amount path), #386.
#[test]
fn single_dose_states_transit_returns_depot_and_central() {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = 0.5;
    p.values[crate::types::PK_IDX_V] = 10.0;
    p.values[crate::types::PK_IDX_N] = 3.0;
    p.values[crate::types::PK_IDX_MTT] = 1.5;
    let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let states = single_dose_states(PkModel::OneCptTransit, &d, 2.0, &p);
    assert_eq!(states.len(), 2, "transit exposes [depot, central]");
    assert!(states[0] > 0.0 && states[1] > 0.0, "both amounts positive mid-absorption: {states:?}");
}

/// 2-cpt transit is a 3-state `[depot, central, peripheral]` model:
/// `single_dose_states` must return the lumped unabsorbed-chain depot, the central
/// concentration (matching `single_dose_concentration`), and the peripheral
/// concentration — covering `two_cpt_transit_depot` / `_peripheral` (#386 PR D).
#[test]
fn single_dose_states_two_cpt_transit_returns_depot_central_periph() {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = 0.5;
    p.values[crate::types::PK_IDX_V] = 10.0; // V1
    p.values[crate::types::PK_IDX_Q] = 1.0;
    p.values[crate::types::PK_IDX_V2] = 20.0;
    p.values[crate::types::PK_IDX_N] = 3.0;
    p.values[crate::types::PK_IDX_MTT] = 1.5;
    let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let states = single_dose_states(PkModel::TwoCptTransit, &d, 2.0, &p);
    assert_eq!(states.len(), 3, "2-cpt transit exposes [depot, central, periph]");
    assert!(
        states.iter().all(|&x| x > 0.0),
        "all three amounts positive mid-absorption: {states:?}"
    );
    // Central state must equal the standalone concentration entry point.
    let c = single_dose_concentration(PkModel::TwoCptTransit, &d, 2.0, &p);
    assert!(
        (states[1] - c).abs() <= 1e-9 * c.abs().max(1.0),
        "central state {} vs single_dose_concentration {}",
        states[1],
        c
    );
}

/// IG is a 2-state `[depot, central]` model: `single_dose_states` must return the
/// unabsorbed-IG "depot" amount and the central concentration for a bolus, covering
/// the `single_dose_states` IG arm and `one_cpt_ig_depot` / `one_cpt_ig_f` (#790).
/// Params in-domain (`ke = 0.05 < 1/(2·MAT·CV²) = 1.111`).
#[test]
fn single_dose_states_ig_returns_depot_and_central() {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = 0.5;
    p.values[crate::types::PK_IDX_V] = 10.0;
    p.values[crate::types::PK_IDX_MAT] = 1.5;
    p.values[crate::types::PK_IDX_CV2] = 0.3;
    let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let states = single_dose_states(PkModel::OneCptIg, &d, 2.0, &p);
    assert_eq!(states.len(), 2, "IG exposes [depot, central]");
    assert!(states[0] > 0.0 && states[1] > 0.0, "both amounts positive mid-absorption: {states:?}");
    // Central state must equal the standalone concentration entry point.
    let c = single_dose_concentration(PkModel::OneCptIg, &d, 2.0, &p);
    assert!((states[1] - c).abs() <= 1e-9 * c.abs().max(1.0));
}

/// 2-cpt IG is a 3-state `[depot, central, peripheral]` model, covering
/// `two_cpt_ig_depot` / `two_cpt_ig_f` / `two_cpt_ig_peripheral` (#790).
#[test]
fn single_dose_states_two_cpt_ig_returns_depot_central_periph() {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = 0.5;
    p.values[crate::types::PK_IDX_V] = 10.0; // V1
    p.values[crate::types::PK_IDX_Q] = 1.0;
    p.values[crate::types::PK_IDX_V2] = 20.0;
    p.values[crate::types::PK_IDX_MAT] = 1.5;
    p.values[crate::types::PK_IDX_CV2] = 0.3;
    let d = DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0);
    let states = single_dose_states(PkModel::TwoCptIg, &d, 2.0, &p);
    assert_eq!(states.len(), 3, "2-cpt IG exposes [depot, central, periph]");
    assert!(
        states.iter().all(|&x| x > 0.0),
        "all three amounts positive mid-absorption: {states:?}"
    );
    let c = single_dose_concentration(PkModel::TwoCptIg, &d, 2.0, &p);
    assert!((states[1] - c).abs() <= 1e-9 * c.abs().max(1.0));
}

/// A `PkParams` whose `CL` and `V` pass the degenerate-parameter early return (#2104),
/// so a direct call reaches the dispatch below it.
fn pk_valid_cl_v() -> PkParams {
    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = 5.0;
    p.values[crate::types::PK_IDX_V] = 50.0;
    p
}

/// SS doses reroute to the ODE twin before reaching the closed forms (`effective_for`,
/// #719), so the SS arms of `single_dose_concentration` / `single_dose_states` are
/// `unreachable!` — calling them directly with an SS dose must hit those guards (covers the
/// defensive arms).
#[test]
#[should_panic(expected = "one_cpt_ig SS doses reroute")]
fn single_dose_concentration_one_cpt_ig_ss_unreachable() {
    let p = pk_valid_cl_v();
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let _ = single_dose_concentration(PkModel::OneCptIg, &ss, 2.0, &p);
}

#[test]
#[should_panic(expected = "two_cpt_ig SS doses reroute")]
fn single_dose_states_two_cpt_ig_ss_unreachable() {
    let p = pk_valid_cl_v();
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let _ = single_dose_states(PkModel::TwoCptIg, &ss, 2.0, &p);
}

/// SS doses reroute to the ODE twin (`effective_for`, #719) before reaching the transit
/// closed form, so the SS arm of `single_dose_concentration` is `unreachable!`; calling it
/// directly with an SS dose must hit that guard (covers the defensive arm + its message).
#[test]
#[should_panic(expected = "two_cpt_transit SS doses reroute")]
fn single_dose_concentration_two_cpt_transit_ss_unreachable() {
    let p = pk_valid_cl_v();
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let _ = single_dose_concentration(PkModel::TwoCptTransit, &ss, 2.0, &p);
}

/// The mirror SS `unreachable!` arm in `single_dose_states`.
#[test]
#[should_panic(expected = "two_cpt_transit SS doses reroute")]
fn single_dose_states_two_cpt_transit_ss_unreachable() {
    let p = pk_valid_cl_v();
    let ss = DoseEvent::new(0.0, 100.0, 1, 0.0, true, 12.0);
    let _ = single_dose_states(PkModel::TwoCptTransit, &ss, 2.0, &p);
}

/// Regression: `single_dose_concentration` and `single_dose_states[central]` must
/// agree for OneCptOral infusion doses.  Before the fix both functions used the
/// oral Bateman formula for infusions (which bypass the depot), so both returned
/// a consistent but wrong value.  After the fix both route to the IV infusion
/// formula, keeping ipred == compartment_states[central] and matching NONMEM.
#[test]
fn single_dose_concentration_matches_states_for_one_cpt_oral_infusion() {
    fn infusion(amt: f64, rate: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
    }
    fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    let cl = 3.0_f64;
    let v = 50.0_f64;
    let ka = 1.2_f64; // ka present but must NOT affect infusion formula
    let f_bio = 0.9_f64;
    let ii = 12.0_f64;
    let amt = 100.0_f64;
    let rate = 50.0_f64; // 2 h infusion
    let tau = 5.0_f64;

    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = cl;
    p.values[crate::types::PK_IDX_V] = v;
    p.values[crate::types::PK_IDX_KA] = ka;
    p.values[crate::types::PK_IDX_F] = f_bio;

    // Non-SS infusion: central (index 1) must equal single_dose_concentration
    let d = infusion(amt, rate);
    let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
    let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "OneCptOral non-SS infusion: concentration={conc} ≠ central state={} \
         (depot is index 0, central is index 1)",
        states[1]
    );
    // Depot must be zero for an infusion (bypasses depot)
    assert_eq!(states[0], 0.0, "OneCptOral infusion: depot must be 0.0");

    // SS infusion
    let d = infusion_ss(amt, rate, ii);
    let conc = single_dose_concentration(PkModel::OneCptOral, &d, tau, &p);
    let states = single_dose_states(PkModel::OneCptOral, &d, tau, &p);
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "OneCptOral SS infusion: concentration={conc} ≠ central state={}",
        states[1]
    );
    assert_eq!(states[0], 0.0, "OneCptOral SS infusion: depot must be 0.0");

    // Bolus must be unchanged (oral formula correct for bolus)
    let d_bolus = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
    let conc = single_dose_concentration(PkModel::OneCptOral, &d_bolus, tau, &p);
    let states = single_dose_states(PkModel::OneCptOral, &d_bolus, tau, &p);
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "OneCptOral bolus: concentration={conc} ≠ central state={}",
        states[1]
    );
    assert!(states[0] > 0.0, "OneCptOral bolus: depot must be > 0");
}

/// Regression test: `single_dose_concentration` and `single_dose_states[central]`
/// must agree for TwoCptOral infusion doses (both SS and non-SS).
///
/// Before the fix, `single_dose_concentration` used the oral formula
/// (`two_cpt_oral_f_ss` / `two_cpt_oral_f`) for infusion doses, while
/// `single_dose_states` correctly used the IV infusion formula (since infusions
/// bypass the depot).  This caused `ipred ≠ compartment_states[central]` for
/// any TwoCptOral model with infusion doses.
#[test]
fn single_dose_concentration_matches_states_for_two_cpt_oral_infusion() {
    fn infusion(amt: f64, rate: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
    }
    fn infusion_ss(amt: f64, rate: f64, ii: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, true, ii)
    }

    let cl = 5.0_f64;
    let v = 80.0_f64;
    let q = 2.0_f64;
    let v2 = 40.0_f64;
    let ka = 1.2_f64;
    let f_bio = 1.0_f64;
    let ii = 24.0_f64;
    let amt = 200.0_f64;
    let rate = 100.0_f64; // 2 h infusion
    let tau = 8.0_f64;

    let mut p = PkParams::default();
    p.values[crate::types::PK_IDX_CL] = cl;
    p.values[crate::types::PK_IDX_V] = v;
    p.values[crate::types::PK_IDX_Q] = q;
    p.values[crate::types::PK_IDX_V2] = v2;
    p.values[crate::types::PK_IDX_KA] = ka;
    p.values[crate::types::PK_IDX_F] = f_bio;

    // Non-SS infusion: single_dose_concentration must equal central state
    let d = infusion(amt, rate);
    let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
    let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
    // central is index 1 for TwoCptOral: [depot=0, central=1, periph=2]
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "TwoCptOral non-SS infusion: concentration={conc} ≠ central state={}",
        states[1]
    );

    // SS infusion
    let d = infusion_ss(amt, rate, ii);
    let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
    let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "TwoCptOral SS infusion: concentration={conc} ≠ central state={}",
        states[1]
    );

    // Bolus should be unchanged (oral formula is correct for bolus)
    let d = DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0);
    let conc = single_dose_concentration(PkModel::TwoCptOral, &d, tau, &p);
    let states = single_dose_states(PkModel::TwoCptOral, &d, tau, &p);
    assert!(
        approx::relative_eq!(conc, states[1], max_relative = 1e-9),
        "TwoCptOral bolus: concentration={conc} ≠ central state={}",
        states[1]
    );
}

/// Closes PR #327 review finding #5: bioavailability `F` must scale **every**
/// element of the `single_dose_states` vector — depot, central, and all
/// peripheral compartments — on every route. The analytical states are linear
/// in the dose, so the invariant is `states(F) == F · states(F=1)`
/// element-wise. This pins down what the central-only
/// `single_dose_concentration` checks cannot see: a depot wrongly scaled, a
/// scale missed on a peripheral arm, or `F` applied twice. `compartment_states`
/// feeds `[derived]`/`[output]` expressions, so a silent error here would
/// surface in user output.
#[test]
fn single_dose_states_scales_every_compartment_by_f() {
    let f = 0.4_f64;
    let tau = 6.0_f64;
    let ii = 12.0_f64;
    let amt = 100.0_f64;

    // Full structural params for up to three compartments; F set per call.
    let params = |f_bio: f64| {
        let mut p = PkParams::default();
        p.values[crate::types::PK_IDX_CL] = 3.0;
        p.values[crate::types::PK_IDX_V] = 50.0;
        p.values[crate::types::PK_IDX_Q] = 2.0;
        p.values[crate::types::PK_IDX_V2] = 40.0;
        p.values[crate::types::PK_IDX_KA] = 1.1;
        p.values[crate::types::PK_IDX_Q3] = 1.5;
        p.values[crate::types::PK_IDX_V3] = 30.0;
        p.values[crate::types::PK_IDX_F] = f_bio;
        p
    };
    let p1 = params(1.0);
    let pf = params(f);

    let models = [
        PkModel::OneCptIv,
        PkModel::OneCptOral,
        PkModel::TwoCptIv,
        PkModel::TwoCptOral,
        PkModel::ThreeCptIv,
        PkModel::ThreeCptOral,
    ];
    // Bolus into compartment 1, non-SS and SS. `F` scales the bioavailable
    // AMOUNT (`F·AMT` into central, or the oral depot load), so it scales
    // every compartment's state uniformly. Infusions are excluded here: for a
    // rate-defined infusion `F` scales the *duration* (#419), which reshapes
    // the profile rather than scaling it - that path is anchored by
    // `iv_bolus_and_infusion_apply_f_matching_nonmem_closed_form` and the
    // cross-path equality tests instead.
    let doses = [
        ("bolus", DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)),
        ("SS bolus", DoseEvent::new(0.0, amt, 1, 0.0, true, ii)),
    ];

    for model in models {
        for (label, dose) in &doses {
            let s1 = single_dose_states(model, dose, tau, &p1);
            let sf = single_dose_states(model, dose, tau, &pf);
            assert_eq!(
                s1.len(),
                sf.len(),
                "{model:?} {label}: state-vector length must not depend on F"
            );
            // Guard against a vacuous pass (all-zero states scale trivially).
            assert!(
                s1.iter().any(|&x| x.abs() > 1e-9),
                "{model:?} {label}: all states ~0 at F=1 — test would be vacuous"
            );
            for (i, (&a, &b)) in s1.iter().zip(sf.iter()).enumerate() {
                assert!(
                    approx::relative_eq!(b, f * a, max_relative = 1e-9, epsilon = 1e-12),
                    "{model:?} {label}: state[{i}] = {b} at F={f}, expected {} = F·{a} \
                     — F must scale every compartment exactly once",
                    f * a
                );
            }
        }
    }
}
