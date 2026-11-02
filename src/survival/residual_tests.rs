//! Per-record survival terms and the residuals built from them (#901).

use super::*;

fn ev(time: f64, event_type: EventType, entry_time: f64) -> ObsRecord {
    ObsRecord::Event {
        time,
        event_type,
        entry_time,
        cmt: 2,
    }
}

fn exact(t: f64, entry: f64) -> ObsRecord {
    ev(t, EventType::Exact, entry)
}

fn cens(t: f64, entry: f64) -> ObsRecord {
    ev(t, EventType::RightCensored, entry)
}

fn interval(left: f64, right: f64, entry: f64) -> ObsRecord {
    ev(right, EventType::IntervalCensored { left, right }, entry)
}

/// The record sets of the T2 golden matrix: the shapes of the existing `tte_data_term_*` and
/// `rtte_*` fixtures (single exact / censored / left-truncated / interval, a three-record
/// stream with and without delayed entry), each scored under every recurrence.
fn golden_record_sets() -> Vec<(&'static str, Vec<ObsRecord>)> {
    vec![
        ("exact", vec![exact(10.0, 0.0)]),
        ("censored", vec![cens(10.0, 0.0)]),
        ("censored_entry", vec![cens(10.0, 5.0)]),
        ("exact_entry", vec![exact(7.5, 3.0)]),
        ("interval", vec![interval(3.0, 5.0, 0.0)]),
        ("interval_tight_entry", vec![interval(10.0, 10.0001, 2.0)]),
        ("two_records", vec![exact(4.0, 0.0), cens(9.0, 1.0)]),
        (
            "stream",
            vec![exact(5.0, 0.0), exact(10.0, 0.0), cens(30.0, 0.0)],
        ),
        (
            "stream_entry",
            vec![
                exact(5.0, 2.0),
                exact(6.5, 2.0),
                exact(11.0, 2.0),
                cens(20.0, 2.0),
            ],
        ),
    ]
}

fn golden_families() -> Vec<(&'static str, HazardFamily, Vec<f64>)> {
    vec![
        ("exp", HazardFamily::Exponential, vec![0.1]),
        ("exp_hr", HazardFamily::Exponential, vec![0.07, 0.3]),
        ("weib", HazardFamily::Weibull, vec![8.0, 1.6]),
        ("weib_hr", HazardFamily::Weibull, vec![12.0, 0.7, -0.2]),
        ("gomp", HazardFamily::Gompertz, vec![0.05, 0.08, 0.1]),
    ]
}

fn golden_recurrences() -> [(&'static str, TteRecurrence); 3] {
    [
        ("single", TteRecurrence::Single),
        (
            "forward",
            TteRecurrence::Repeated {
                clock: RtteClock::Forward,
            },
        ),
        (
            "reset",
            TteRecurrence::Repeated {
                clock: RtteClock::Reset,
            },
        ),
    ]
}

/// Every cell of the golden matrix, in a fixed order: `(label, nll)`.
fn golden_nlls() -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for (fname, family, params) in golden_families() {
        for (rname, records) in golden_record_sets() {
            for (cname, recurrence) in golden_recurrences() {
                let nll = analytic_tte_nll_g(&records, family, recurrence, &params);
                out.push((format!("{fname}/{rname}/{cname}"), nll));
            }
        }
    }
    out
}

/// `to_bits()` of every golden cell, captured from the loops **before** the #901 visitor
/// was threaded through them (at `4ccb9f1a`). The 20 `0x4415af1d78b58c40` cells are the
/// `1e20` sentinel: interval censoring under either RTTE clock.
const GOLDEN_NLL_BITS: &[(&str, u64)] = &[
    ("exp/exact/single", 0x400a6bb1bbb55515), // 3.3025850929940455e0
    ("exp/exact/forward", 0x400a6bb1bbb55515), // 3.3025850929940455e0
    ("exp/exact/reset", 0x400a6bb1bbb55515),  // 3.3025850929940455e0
    ("exp/censored/single", 0x3ff0000000000000), // 1e0
    ("exp/censored/forward", 0x3ff0000000000000), // 1e0
    ("exp/censored/reset", 0x3ff0000000000000), // 1e0
    ("exp/censored_entry/single", 0x3fe0000000000000), // 5e-1
    ("exp/censored_entry/forward", 0x3fe0000000000000), // 5e-1
    ("exp/censored_entry/reset", 0x3fe0000000000000), // 5e-1
    ("exp/exact_entry/single", 0x4006054b554eeeae), // 2.752585092994045e0
    ("exp/exact_entry/forward", 0x4006054b554eeeae), // 2.752585092994045e0
    ("exp/exact_entry/reset", 0x4006054b554eeeae), // 2.752585092994045e0
    ("exp/interval/single", 0x40000feaa977fec8), // 2.00777180097052e0
    ("exp/interval/forward", 0x4415af1d78b58c40), // 1e20
    ("exp/interval/reset", 0x4415af1d78b58c40), // 1e20
    ("exp/interval_tight_entry/single", 0x4028a0386c01d8aa), // 1.231293046495951e1
    ("exp/interval_tight_entry/forward", 0x4415af1d78b58c40), // 1e20
    ("exp/interval_tight_entry/reset", 0x4415af1d78b58c40), // 1e20
    ("exp/two_records/single", 0x400c054b554eeeae), // 3.502585092994045e0
    ("exp/two_records/forward", 0x40099ee4eee88848), // 3.2025850929940454e0
    ("exp/two_records/reset", 0x40099ee4eee88848), // 3.2025850929940454e0
    ("exp/stream/single", 0x402235d8dddaaa8a), // 9.10517018598809e0
    ("exp/stream/forward", 0x401e6bb1bbb55515), // 7.605170185988091e0
    ("exp/stream/reset", 0x401e6bb1bbb55515), // 7.605170185988091e0
    ("exp/stream_entry/single", 0x4024b72bb32e6636), // 1.0357755278982136e1
    ("exp/stream_entry/forward", 0x40216a5ee661996a), // 8.707755278982138e0
    ("exp/stream_entry/reset", 0x40216a5ee661996a), // 8.707755278982138e0
    ("exp_hr/exact/single", 0x400a6eec1182864e), // 3.3041612022359805e0
    ("exp_hr/exact/forward", 0x400a6eec1182864e), // 3.3041612022359805e0
    ("exp_hr/exact/reset", 0x400a6eec1182864e), // 3.3041612022359805e0
    ("exp_hr/censored/single", 0x3fee3ca15e5dbed3), // 9.449011653032023e-1
    ("exp_hr/censored/forward", 0x3fee3ca15e5dbed3), // 9.449011653032023e-1
    ("exp_hr/censored/reset", 0x3fee3ca15e5dbed3), // 9.449011653032023e-1
    ("exp_hr/censored_entry/single", 0x3fde3ca15e5dbed3), // 4.7245058265160117e-1
    ("exp_hr/censored_entry/forward", 0x3fde3ca15e5dbed3), // 4.7245058265160117e-1
    ("exp_hr/censored_entry/reset", 0x3fde3ca15e5dbed3), // 4.7245058265160117e-1
    ("exp_hr/exact_entry/single", 0x40064695e155a277), // 2.784465561319219e0
    ("exp_hr/exact_entry/forward", 0x40064695e155a277), // 2.784465561319219e0
    ("exp_hr/exact_entry/reset", 0x40064695e155a277), // 2.784465561319219e0
    ("exp_hr/interval/single", 0x400057372c171c3f), // 2.0425857014229503e0
    ("exp_hr/interval/forward", 0x4415af1d78b58c40), // 1e20
    ("exp_hr/interval/reset", 0x4415af1d78b58c40), // 1e20
    ("exp_hr/interval_tight_entry/single", 0x4028a6ab5a3bfa98), // 1.232552606565632e1
    ("exp_hr/interval_tight_entry/forward", 0x4415af1d78b58c40), // 1e20
    ("exp_hr/interval_tight_entry/reset", 0x4415af1d78b58c40), // 1e20
    ("exp_hr/two_records/single", 0x400bf1f42307363f), // 3.493141435296621e0
    ("exp_hr/two_records/forward", 0x4009ad6808c02e55), // 3.20967108570566e0
    ("exp_hr/two_records/reset", 0x4009ad6808c02e55), // 3.20967108570566e0
    ("exp_hr/stream/single", 0x4021f0ef3f7fe8f8), // 8.970575317729967e0
    ("exp_hr/stream/forward", 0x401e36803d4e3e28), // 7.553223569775163e0
    ("exp_hr/stream/reset", 0x401e36803d4e3e28), // 7.553223569775163e0
    ("exp_hr/stream_entry/single", 0x4024ace596fc87b1), // 1.033768913109438e1
    ("exp_hr/stream_entry/forward", 0x40218ea4f2dadcd1), // 8.778602208344099e0
    ("exp_hr/stream_entry/reset", 0x40218ea4f2dadcd1), // 8.778602208344099e0
    ("weib/exact/single", 0x40073caeb087dd4a), // 2.904630068918469e0
    ("weib/exact/forward", 0x40073caeb087dd4a), // 2.904630068918469e0
    ("weib/exact/reset", 0x40073caeb087dd4a), // 2.904630068918469e0
    ("weib/censored/single", 0x3ff6dd8131b42c24), // 1.4290782872728949e0
    ("weib/censored/forward", 0x3ff6dd8131b42c24), // 1.4290782872728949e0
    ("weib/censored/reset", 0x3ff6dd8131b42c24), // 1.4290782872728949e0
    ("weib/censored_entry/single", 0x3feea522eff4f2cd), // 9.57658260980304e-1
    ("weib/censored_entry/forward", 0x3feea522eff4f2cd), // 9.57658260980304e-1
    ("weib/censored_entry/reset", 0x3feea522eff4f2cd), // 9.57658260980304e-1
    ("weib/exact_entry/single", 0x4002bc249e2bf1af), // 2.3418667180214565e0
    ("weib/exact_entry/forward", 0x4002bc249e2bf1af), // 2.3418667180214565e0
    ("weib/exact_entry/reset", 0x4002bc249e2bf1af), // 2.3418667180214565e0
    ("weib/interval/single", 0x3ffabefb23580df5), // 1.6716262226699403e0
    ("weib/interval/forward", 0x4415af1d78b58c40), // 1e20
    ("weib/interval/reset", 0x4415af1d78b58c40), // 1e20
    ("weib/interval_tight_entry/single", 0x4028032769157b49), // 1.2006160053131355e1
    ("weib/interval_tight_entry/forward", 0x4415af1d78b58c40), // 1e20
    ("weib/interval_tight_entry/reset", 0x4415af1d78b58c40), // 1e20
    ("weib/two_records/single", 0x400c36a76e5f0fb0), // 3.5266865370521927e0
    ("weib/two_records/forward", 0x4009dc952a02d0dc), // 3.2327063829526264e0
    ("weib/two_records/reset", 0x40069ceca3238649), // 2.8266232247558816e0
    ("weib/stream/single", 0x402b1c69ecc2ec83), // 1.355549564246599e1
    ("weib/stream/forward", 0x40274f5bcf5530a7), // 1.1654997328900505e1
    ("weib/stream/reset", 0x40221da1db9af5ee), // 9.05787550226793e0
    ("weib/stream_entry/single", 0x402796817c48aa3f), // 1.179395664584183e1
    ("weib/stream_entry/forward", 0x402288c9afb15914), // 9.267163744353219e0
    ("weib/stream_entry/reset", 0x4020fe64f8075930), // 8.496864081284542e0
    ("weib_hr/exact/single", 0x400da8ff56845018), // 3.7075182685021097e0
    ("weib_hr/exact/forward", 0x400da8ff56845018), // 3.7075182685021097e0
    ("weib_hr/exact/reset", 0x400da8ff56845018), // 3.7075182685021097e0
    ("weib_hr/censored/single", 0x3fe70f6d3c101874), // 7.206331418135634e-1
    ("weib_hr/censored/forward", 0x3fe70f6d3c101874), // 7.206331418135634e-1
    ("weib_hr/censored/reset", 0x3fe70f6d3c101874), // 7.206331418135634e-1
    ("weib_hr/censored_entry/single", 0x3fd1bae1f1dfe665), // 2.770314085060817e-1
    ("weib_hr/censored_entry/forward", 0x3fd1bae1f1dfe665), // 2.770314085060817e-1
    ("weib_hr/censored_entry/reset", 0x3fd1bae1f1dfe665), // 2.770314085060817e-1
    ("weib_hr/exact_entry/single", 0x40096fae7082d6e2), // 3.1795319356850635e0
    ("weib_hr/exact_entry/forward", 0x40096fae7082d6e2), // 3.1795319356850635e0
    ("weib_hr/exact_entry/reset", 0x40096fae7082d6e2), // 3.1795319356850635e0
    ("weib_hr/interval/single", 0x4003208462921861), // 2.3908775044283606e0
    ("weib_hr/interval/forward", 0x4415af1d78b58c40), // 1e20
    ("weib_hr/interval/reset", 0x4415af1d78b58c40), // 1e20
    ("weib_hr/interval_tight_entry/single", 0x40295e5a53c1e53e), // 1.2684282891672975e1
    ("weib_hr/interval_tight_entry/forward", 0x4415af1d78b58c40), // 1e20
    ("weib_hr/interval_tight_entry/reset", 0x4415af1d78b58c40), // 1e20
    ("weib_hr/two_records/single", 0x400cefbdf05960ae), // 3.617061498382177e0
    ("weib_hr/two_records/forward", 0x400b0d18f4b9c346), // 3.381395255965262e0
    ("weib_hr/two_records/reset", 0x400c47c8deba55c8), // 3.5350510979285197e0
    ("weib_hr/stream/single", 0x4020f84b3b1f25a2), // 8.484948966552853e0
    ("weib_hr/stream/forward", 0x401d48694658438e), // 7.3207140914318085e0
    ("weib_hr/stream/reset", 0x401e76891be74a92), // 7.615757404319668e0
    ("weib_hr/stream_entry/single", 0x402545543a40d49c), // 1.0635408230222588e1
    ("weib_hr/stream_entry/forward", 0x40232da697bd1f7b), // 9.58916162665968e0
    ("weib_hr/stream_entry/reset", 0x4022da5ba1760469), // 9.426480336815262e0
    ("gomp/exact/single", 0x400789bb9f8e5b92), // 2.942252394229805e0
    ("gomp/exact/forward", 0x400789bb9f8e5b92), // 2.942252394229805e0
    ("gomp/exact/reset", 0x400789bb9f8e5b92), // 2.942252394229805e0
    ("gomp/censored/single", 0x3feb16b15d36acce), // 8.46520120675814e-1
    ("gomp/censored/forward", 0x3feb16b15d36acce), // 8.46520120675814e-1
    ("gomp/censored/reset", 0x3feb16b15d36acce), // 8.46520120675814e-1
    ("gomp/censored_entry/single", 0x3fe037b70bc1a594), // 5.068011502855136e-1
    ("gomp/censored_entry/forward", 0x3fe037b70bc1a594), // 5.068011502855136e-1
    ("gomp/censored_entry/reset", 0x3fe037b70bc1a594), // 5.068011502855136e-1
    ("gomp/exact_entry/single", 0x400568ee24980d31), // 2.676235471620793e0
    ("gomp/exact_entry/forward", 0x400568ee24980d31), // 2.676235471620793e0
    ("gomp/exact_entry/reset", 0x400568ee24980d31), // 2.676235471620793e0
    ("gomp/interval/single", 0x4001271959ee9cad), // 2.144091322512017e0
    ("gomp/interval/forward", 0x4415af1d78b58c40), // 1e20
    ("gomp/interval/reset", 0x4415af1d78b58c40), // 1e20
    ("gomp/interval_tight_entry/single", 0x402810c40277c632), // 1.2032745434876322e1
    ("gomp/interval_tight_entry/forward", 0x4415af1d78b58c40), // 1e20
    ("gomp/interval_tight_entry/reset", 0x4415af1d78b58c40), // 1e20
    ("gomp/two_records/single", 0x400c0e64adf7c6df), // 3.5070279685249797e0
    ("gomp/two_records/forward", 0x400a6eb880a813f2), // 3.304062848214465e0
    ("gomp/two_records/reset", 0x400968560ed23772), // 3.175945392408658e0
    ("gomp/stream/single", 0x402966ed799404a1), // 1.2701030539816488e1
    ("gomp/stream/forward", 0x40270792bea94960), // 1.1514791448750373e1
    ("gomp/stream/reset", 0x4020cd82b319e854), // 8.401387783920804e0
    ("gomp/stream_entry/single", 0x4025d8e944aadc44), // 1.0923654695374474e1
    ("gomp/stream_entry/forward", 0x4022fee3eb57cc3e), // 9.497832636325366e0
    ("gomp/stream_entry/reset", 0x4022491a1fbbc2bb), // 9.142777434988025e0
];

/// T2: the visitor refactor moved no bit. Every golden cell's NLL is the captured
/// pre-change value, so the refactored add order is the old one; the #2437 dual instance
/// of the same loops is pinned by its own parity tests (`eta_hessian_step_tests`).
///
/// Dies on: any reassociation of a record's term (e.g. `dh + (−log h)`, or adding
/// `log h` to `nll` separately), a term visited but not added, or a changed sentinel path.
#[test]
fn visitor_refactor_keeps_every_golden_nll_bit() {
    let got = golden_nlls();
    assert_eq!(got.len(), GOLDEN_NLL_BITS.len());
    for ((label, nll), (want_label, want_bits)) in got.iter().zip(GOLDEN_NLL_BITS) {
        assert_eq!(label, want_label);
        assert_eq!(
            nll.to_bits(),
            *want_bits,
            "{label}: {nll:e} vs golden {:e}",
            f64::from_bits(*want_bits)
        );
    }
}

/// Collect a record set's terms and its NLL under one analytic family and recurrence.
fn terms_and_nll(
    records: &[ObsRecord],
    family: HazardFamily,
    recurrence: TteRecurrence,
    params: &[f64],
) -> (Vec<RecordTerm<f64>>, f64) {
    let mut terms = Vec::new();
    let nll = analytic_tte_nll_visit_g(records, family, recurrence, params, &mut |t| terms.push(t));
    (terms, nll)
}

/// Fold terms the way the loops do: `dh − log h` per point, `−log_prob` per interval.
fn fold(terms: &[RecordTerm<f64>]) -> f64 {
    let mut nll = 0.0_f64;
    for t in terms {
        nll = match *t {
            RecordTerm::Point { dh, log_h, .. } => match log_h {
                Some(l) => nll + (dh - l),
                None => nll + dh,
            },
            RecordTerm::Interval { log_prob } => nll - log_prob,
        };
    }
    nll
}

/// T1: on every non-sentinel golden cell — Single / Forward / Reset, with and without a
/// delayed entry, five families — the visited terms fold **bit-identically** to the
/// NLL the loop returned, one term per `Event` record.
///
/// Dies on: a loop visiting a `dh` other than the one it adds (e.g. the forward loop
/// handing `h_t` or the reset loop `H(t)` to the visitor), a record visited twice or not
/// at all.
#[test]
fn record_terms_fold_to_the_scored_nll() {
    let mut checked = 0;
    for (fname, family, params) in golden_families() {
        for (rname, records) in golden_record_sets() {
            for (cname, recurrence) in golden_recurrences() {
                let (terms, nll) = terms_and_nll(&records, family, recurrence, &params);
                if nll >= SUBJECT_SENTINEL_NLL {
                    continue;
                }
                assert_eq!(terms.len(), records.len(), "{fname}/{rname}/{cname}");
                assert_eq!(
                    fold(&terms).to_bits(),
                    nll.to_bits(),
                    "{fname}/{rname}/{cname}"
                );
                checked += 1;
            }
        }
    }
    // 135 cells less the 20 RTTE-interval sentinels.
    assert_eq!(checked, 115);
}

fn single_rows(
    records: &[ObsRecord],
    family: HazardFamily,
    recurrence: TteRecurrence,
    params: &[f64],
) -> Vec<crate::types::TteObsDiagnostic> {
    let (terms, nll) = terms_and_nll(records, family, recurrence, params);
    assert!(nll < SUBJECT_SENTINEL_NLL);
    tte_record_diagnostics(records, Some(&terms), |t| t)
}

/// T3: one exponential subject, `T = 5`, `entry = 2`, `λ = 0.1`, an event: the record's
/// increment is `ΔH = λ·(T − entry) = 0.3`, so `M = 0.7`, `PIT = e^{−0.3}` and
/// `D = √(2·(0.3 − 1 − ln 0.3))`, under all three recurrences (the first record of an
/// RTTE stream conditions on entry the same way, #740).
///
/// Dies on: `H(T)` scored without the entry lower limit (`ΔH = 0.5`).
#[test]
fn residuals_of_a_left_truncated_exponential_event_are_closed_form() {
    let records = vec![exact(5.0, 2.0)];
    let dh = 0.1 * 5.0 - 0.1 * 2.0;
    let d = (2.0 * (dh - 1.0 - dh.ln())).sqrt();
    for (cname, recurrence) in golden_recurrences() {
        let rows = single_rows(&records, HazardFamily::Exponential, recurrence, &[0.1]);
        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!((r.cmt, r.time, r.dv), (2, 5.0, 1.0), "{cname}");
        assert!((r.coxsnell - 0.3).abs() < 1e-15, "{cname}: {}", r.coxsnell);
        assert!(
            (r.martingale - 0.7).abs() < 1e-15,
            "{cname}: {}",
            r.martingale
        );
        assert!(
            (r.pit - (-0.3_f64).exp()).abs() < 1e-15,
            "{cname}: {}",
            r.pit
        );
        assert!(
            (r.deviance - d).abs() < 1e-14,
            "{cname}: {} vs {d}",
            r.deviance
        );
    }
}

/// T4: a Weibull stream (events at 5 and 10, censor at 30). Clock-reset scores each
/// record on its gap, `M_k = δ_k − H(Δ_k)` with `Δ = 5, 5, 20`; clock-forward on the
/// increment `H(t_k) − H(t_{k−1})`. The two clocks differ for a Weibull, so the test
/// tells them apart.
///
/// Dies on: the reset path reporting `δ − H(t_k)` (absolute time) or the forward path
/// reporting `δ − H(t_k)` without the previous record's lower limit.
#[test]
fn rtte_residuals_use_the_gap_on_reset_and_the_increment_on_forward() {
    let (scale, shape) = (8.0_f64, 1.6_f64);
    let h = |t: f64| (t / scale).powf(shape);
    let records = vec![exact(5.0, 0.0), exact(10.0, 0.0), cens(30.0, 0.0)];
    let reset = single_rows(
        &records,
        HazardFamily::Weibull,
        TteRecurrence::Repeated {
            clock: RtteClock::Reset,
        },
        &[scale, shape],
    );
    let forward = single_rows(
        &records,
        HazardFamily::Weibull,
        TteRecurrence::Repeated {
            clock: RtteClock::Forward,
        },
        &[scale, shape],
    );
    let want_reset = [1.0 - h(5.0), 1.0 - h(5.0), -h(20.0)];
    let want_forward = [1.0 - h(5.0), 1.0 - (h(10.0) - h(5.0)), -(h(30.0) - h(10.0))];
    for k in 0..3 {
        assert!(
            (reset[k].martingale - want_reset[k]).abs() < 1e-13,
            "reset record {k}: {} vs {}",
            reset[k].martingale,
            want_reset[k]
        );
        assert!(
            (forward[k].martingale - want_forward[k]).abs() < 1e-13,
            "forward record {k}: {} vs {}",
            forward[k].martingale,
            want_forward[k]
        );
    }
    // The forward stream's martingales telescope to N(T) − H(T).
    let sum: f64 = forward.iter().map(|r| r.martingale).sum();
    assert!((sum - (2.0 - h(30.0))).abs() < 1e-13);
}

/// T5: the deviance residual on its awkward inputs. Finite on every cell but an event
/// with no hazard mass (`δ = 1, ΔH = 0`), which is `+∞` — the limit, and R's value; its
/// sign follows `M`; it is exactly 0 where `M = 0` (`δ = 1, ΔH = 1`) and at `δ = 0,
/// ΔH = 0`; and a `ΔH` a round-off band below zero is clamped, not `NaN`.
///
/// Dies on: dropping the `δ = 0` branch (`0·ln 0 = NaN` at `ΔH = 0`), dropping the
/// `ΔH ≥ 0` clamp (`NaN` at `ΔH = −1e-18`), a sign taken from anything but `M`. The
/// grid's `1 ± 1e-12` cells also pin that the unclamped radicand stays non-negative
/// next to `ΔH = 1` on this platform's `ln`.
#[test]
fn deviance_residual_is_finite_signed_and_zero_where_m_is() {
    let grid = [0.0, 1e-300, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 50.0];
    for &dh in &grid {
        for delta in [0.0, 1.0] {
            let d = deviance_residual(delta, dh);
            if delta == 1.0 && dh == 0.0 {
                assert_eq!(d, f64::INFINITY);
                continue;
            }
            assert!(d.is_finite(), "δ={delta} ΔH={dh}: {d}");
            let m = delta - dh;
            if m != 0.0 && d != 0.0 {
                assert_eq!(d.signum(), m.signum(), "δ={delta} ΔH={dh}: {d}");
            }
        }
    }
    assert_eq!(deviance_residual(1.0, 1.0), 0.0);
    assert_eq!(deviance_residual(0.0, 0.0), 0.0);
    assert_eq!(deviance_residual(0.0, 2.0), -2.0);
    assert_eq!(deviance_residual(0.0, -1e-18), 0.0);
    assert_eq!(deviance_residual(1.0, -1e-18), f64::INFINITY);
}

/// The row builder's own branches: an interval-censored record has no single `ΔH`
/// (`DV` and every residual NaN); `terms = None` (the subject on the `1e20` rail) keeps
/// the rows and their `DV` but blanks every residual; a non-`Event` record makes no row;
/// `TIME` goes through the written-time map.
///
/// Dies on: an interval row carrying a residual, the rail dropping rows or keeping
/// values, the time map skipped.
#[test]
fn tte_rows_blank_intervals_and_the_sentinel_rail() {
    let records = vec![
        exact(4.0, 0.0),
        ObsRecord::DiscreteState {
            time: 5.0,
            raw_time: 5.0,
            state: 1,
            cmt: 3,
        },
        interval(6.0, 8.0, 0.0),
        cens(9.0, 0.0),
    ];
    let (terms, nll) = terms_and_nll(
        &records,
        HazardFamily::Exponential,
        TteRecurrence::Single,
        &[0.1],
    );
    assert!(nll < SUBJECT_SENTINEL_NLL);
    let rows = tte_record_diagnostics(&records, Some(&terms), |t| t + 100.0);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter().map(|r| r.time).collect::<Vec<_>>(),
        [104.0, 108.0, 109.0]
    );
    assert_eq!((rows[0].dv, rows[2].dv), (1.0, 0.0));
    assert!((rows[0].coxsnell - 0.4).abs() < 1e-15);
    assert!((rows[2].coxsnell - 0.9).abs() < 1e-15);
    let iv = &rows[1];
    assert!(iv.dv.is_nan() && iv.coxsnell.is_nan() && iv.martingale.is_nan());
    assert!(iv.deviance.is_nan() && iv.pit.is_nan());

    let rail = tte_record_diagnostics(&records, None, |t| t);
    assert_eq!(rail.len(), 3);
    assert_eq!((rail[0].dv, rail[2].dv), (1.0, 0.0));
    for r in &rail {
        assert!(r.coxsnell.is_nan() && r.martingale.is_nan());
        assert!(r.deviance.is_nan() && r.pit.is_nan());
    }
}

/// #901 review round 1, row 7: a censor scored a round-off band **below** 0 (a flat ODE
/// CHZ inside the `MonoTol` tolerance) reports the scored `ΔH` unclamped, as documented on
/// `TteObsDiagnostic`: `COXSNELL < 0`, `MARTINGALE = δ − ΔH` exactly, `PIT > 1`, and only
/// `DEVIANCE` clamped, to `−0`.
///
/// Dies on: clamping `ΔH` for the other three columns, which would make `MARTINGALE`
/// stop being `δ −` the scored increment.
#[test]
fn a_round_off_negative_increment_is_reported_as_scored() {
    let records = vec![cens(9.0, 0.0)];
    let dh = -1e-15;
    let terms = [RecordTerm::Point {
        index: 0,
        dh,
        log_h: None,
    }];
    let rows = tte_record_diagnostics(&records, Some(&terms), |t| t);
    let r = &rows[0];
    assert_eq!(r.coxsnell, dh);
    assert_eq!(r.martingale, -dh);
    assert!(r.pit > 1.0, "{}", r.pit);
    assert!(r.deviance == 0.0 && r.deviance.is_sign_negative());
}
