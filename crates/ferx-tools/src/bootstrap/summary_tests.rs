use super::*;

fn replicate(index: usize, estimates: Vec<f64>) -> ReplicateResult {
    ReplicateResult {
        index,
        estimates,
        standard_errors: None,
        ofv: 100.0,
        converged: true,
        estimate_near_boundary: false,
        covariance_step_successful: true,
        covariance_step_warnings: false,
        seconds: 1.0,
        error: None,
        delta_ofv: None,
    }
}

// ── the PsN percentile estimator ────────────────────────────────────────────

#[test]
fn percentile_matches_the_hand_computed_weighted_average() {
    // The guide's formula, worked by hand on [1,2,3,4]:
    //   n = 4+1 = 5
    //   p = 0.50 -> n*p = 2.5 -> i=2, f=0.5 -> 0.5*x_2 + 0.5*x_3 = 2.5
    //   p = 0.25 -> n*p = 1.25 -> i=1, f=0.25 -> 0.75*x_1 + 0.25*x_2 = 1.25
    let x = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(percentile(&x, 50.0), Some(2.5));
    assert_eq!(percentile(&x, 25.0), Some(1.25));
    assert_eq!(percentile(&x, 75.0), Some(3.75));
    // Landing exactly on an order statistic: n*p = 5*0.2 = 1.0, f = 0.
    assert_eq!(percentile(&x, 20.0), Some(1.0));
    assert_eq!(percentile(&x, 80.0), Some(4.0));
}

#[test]
fn percentile_is_none_on_an_empty_sample() {
    assert_eq!(percentile(&[], 50.0), None);
}

#[test]
fn minimum_sample_sizes_are_exactly_the_ones_psn_documents() {
    // The guide tabulates four thresholds. They are not four rules — they all
    // fall out of "(n+1)p >= 1", i.e. there must be an order statistic at or
    // below the requested quantile rather than an extrapolation past the
    // smallest observation. Pin all four, from both sides.
    for (interval, n_min) in [
        ((5.0, 95.0), 19usize),
        ((2.5, 97.5), 39),
        ((0.5, 99.5), 199),
        ((0.05, 99.95), 1999),
    ] {
        let (lo_pct, hi_pct) = interval;

        let enough: Vec<f64> = (0..n_min).map(|i| i as f64).collect();
        assert!(
            percentile(&enough, lo_pct).is_some(),
            "{n_min} samples should resolve the {lo_pct}% tail"
        );
        assert!(
            percentile(&enough, hi_pct).is_some(),
            "{n_min} samples should resolve the {hi_pct}% tail"
        );

        let too_few: Vec<f64> = (0..n_min - 1).map(|i| i as f64).collect();
        assert!(
            percentile(&too_few, lo_pct).is_none(),
            "{} samples must not resolve the {lo_pct}% tail",
            n_min - 1
        );
        assert!(
            percentile(&too_few, hi_pct).is_none(),
            "{} samples must not resolve the {hi_pct}% tail",
            n_min - 1
        );
    }
}

#[test]
fn at_the_threshold_the_interval_is_the_extreme_order_statistics() {
    // n = 19, p = 0.05: n*p = 20*0.05 = 1 exactly, so the 5th percentile is the
    // minimum and the 95th is the maximum — the widest the data can support.
    let x: Vec<f64> = (1..=19).map(|i| i as f64).collect();
    assert_eq!(percentile(&x, 5.0), Some(1.0));
    assert_eq!(percentile(&x, 95.0), Some(19.0));
}

// ── the normal quantile ─────────────────────────────────────────────────────

#[test]
fn normal_quantile_matches_known_values() {
    let cases = [
        (0.975, 1.959_963_984_540_054),
        (0.95, 1.644_853_626_951_472),
        (0.995, 2.575_829_303_548_901),
        (0.5, 0.0),
    ];
    for (p, expected) in cases {
        let got = normal_quantile(p);
        assert!(
            (got - expected).abs() < 1e-8,
            "normal_quantile({p}) = {got}, expected {expected}"
        );
    }
    // Symmetric tails come out exactly symmetric: the central branch is odd in
    // `p - 0.5`, so this is equality, not a tolerance.
    assert_eq!(normal_quantile(0.025), -normal_quantile(0.975));
    // The far tails take the other branch, and must still be usable.
    assert!((normal_quantile(0.0005) + 3.290_526_731_491_896).abs() < 1e-8);
}

// ── exclusion filters ───────────────────────────────────────────────────────

#[test]
fn default_filters_drop_terminated_and_boundary_replicates() {
    let options = BootstrapOptions::default();
    assert!(options.skip_minimization_terminated);
    assert!(options.skip_estimate_near_boundary);

    let mut r = replicate(1, vec![1.0]);
    assert_eq!(exclusion_reason(&r, &options), None);

    r.converged = false;
    assert_eq!(
        exclusion_reason(&r, &options),
        Some("minimization terminated")
    );

    r.converged = true;
    r.estimate_near_boundary = true;
    assert_eq!(
        exclusion_reason(&r, &options),
        Some("estimate near boundary")
    );

    // A fit that returned Err is excluded regardless of the filters — there is
    // nothing to include.
    let mut failed = replicate(2, vec![]);
    failed.error = Some("boom".to_string());
    let permissive = BootstrapOptions {
        skip_minimization_terminated: false,
        skip_estimate_near_boundary: false,
        ..BootstrapOptions::default()
    };
    assert_eq!(exclusion_reason(&failed, &permissive), Some("fit failed"));
}

#[test]
fn disabling_a_filter_keeps_the_replicate() {
    let options = BootstrapOptions {
        skip_minimization_terminated: false,
        ..BootstrapOptions::default()
    };
    let mut r = replicate(1, vec![1.0]);
    r.converged = false;
    assert_eq!(exclusion_reason(&r, &options), None);
}

#[test]
fn covariance_filters_only_fire_when_asked_for() {
    let mut r = replicate(1, vec![1.0]);
    r.covariance_step_successful = false;
    r.covariance_step_warnings = true;
    assert_eq!(exclusion_reason(&r, &BootstrapOptions::default()), None);

    let strict = BootstrapOptions {
        keep_covariance: true,
        skip_covariance_step_terminated: true,
        ..BootstrapOptions::default()
    };
    assert_eq!(
        exclusion_reason(&r, &strict),
        Some("covariance step terminated")
    );

    let warn = BootstrapOptions {
        keep_covariance: true,
        skip_with_covstep_warnings: true,
        ..BootstrapOptions::default()
    };
    r.covariance_step_successful = true;
    assert_eq!(
        exclusion_reason(&r, &warn),
        Some("covariance step warnings")
    );
}

// ── the summary table ───────────────────────────────────────────────────────

#[test]
fn summary_reports_mean_bias_sd_and_median() {
    let names = vec!["CL".to_string()];
    let original = replicate(0, vec![10.0]);
    let replicates: Vec<ReplicateResult> = [8.0, 9.0, 10.0, 11.0, 14.0]
        .iter()
        .enumerate()
        .map(|(i, v)| replicate(i + 1, vec![*v]))
        .collect();

    let s = summarize(
        &names,
        Some(&original),
        &replicates,
        &BootstrapOptions::default(),
    );
    let p = &s.parameters[0];

    assert_eq!(p.original, Some(10.0));
    assert!((p.mean - 10.4).abs() < 1e-12);
    // Bias is the bootstrap mean minus the original estimate.
    assert!((p.bias.unwrap() - 0.4).abs() < 1e-12);
    // Sample SD (n-1): values 8,9,10,11,14 about 10.4.
    let expected_sd = (((8.0f64 - 10.4).powi(2)
        + (9.0f64 - 10.4).powi(2)
        + (10.0f64 - 10.4).powi(2)
        + (11.0f64 - 10.4).powi(2)
        + (14.0f64 - 10.4).powi(2))
        / 4.0)
        .sqrt();
    assert!((p.standard_error - expected_sd).abs() < 1e-12);
    assert!((p.median - 10.0).abs() < 1e-12);

    // Five samples cannot resolve a 2.5% tail (39 needed), so no percentile CI —
    // but the normal-approximation interval is still available.
    assert!(p.ci_percentile.is_none());
    // Against `normal_quantile` itself: this pins the wiring (which quantile,
    // applied to which centre and spread), while the quantile's own accuracy is
    // pinned by `normal_quantile_matches_known_values`.
    let z = normal_quantile(0.975);
    let (lo, hi) = p.ci_standard_error.unwrap();
    assert!((lo - (10.0 - z * expected_sd)).abs() < 1e-12);
    assert!((hi - (10.0 + z * expected_sd)).abs() < 1e-12);

    assert_eq!(s.n_completed, 5);
    assert_eq!(s.n_included, 5);
}

#[test]
fn excluded_replicates_do_not_enter_the_statistics() {
    let names = vec!["CL".to_string()];
    let original = replicate(0, vec![10.0]);
    let mut replicates: Vec<ReplicateResult> = (1..=4).map(|i| replicate(i, vec![10.0])).collect();
    // One wild non-converged replicate, which would move the mean by 22.5 if it
    // were counted.
    replicates.push({
        let mut r = replicate(5, vec![100.0]);
        r.converged = false;
        r
    });

    let s = summarize(
        &names,
        Some(&original),
        &replicates,
        &BootstrapOptions::default(),
    );
    assert_eq!(s.n_completed, 5);
    assert_eq!(s.n_included, 4);
    assert!((s.parameters[0].mean - 10.0).abs() < 1e-12);
    assert_eq!(
        s.excluded_by,
        vec![("minimization terminated".to_string(), 1)]
    );

    // The diagnostic means describe the whole run, including what was filtered.
    let converged = s
        .diagnostic_means
        .iter()
        .find(|(k, _)| k == "minimization_successful")
        .unwrap()
        .1;
    assert!((converged - 0.8).abs() < 1e-12);
}

#[test]
fn a_run_with_no_base_fit_reports_no_bias() {
    let names = vec!["CL".to_string()];
    let replicates: Vec<ReplicateResult> = (1..=3).map(|i| replicate(i, vec![i as f64])).collect();
    let s = summarize(&names, None, &replicates, &BootstrapOptions::default());
    assert_eq!(s.parameters[0].original, None);
    assert_eq!(s.parameters[0].bias, None);
    assert_eq!(s.parameters[0].ci_standard_error, None);
    assert!((s.parameters[0].mean - 2.0).abs() < 1e-12);
}
