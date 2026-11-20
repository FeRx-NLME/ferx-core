use super::*;
use crate::types::FitOptions;

/// Deterministic standard normal stream (xorshift64* + Box-Muller).
struct Gauss(u64);
impl Gauss {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
}

/// AR(1) noise with lag-1 correlation `rho` and marginal sd `sd`, plus `drift` per step.
fn series(seed: u64, n: usize, rho: f64, sd: f64, drift: f64) -> Vec<f64> {
    let mut g = Gauss(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut e = 0.0;
    let innov = sd * (1.0 - rho * rho).sqrt();
    (0..n)
        .map(|i| {
            e = rho * e + innov * g.n();
            1.0 + drift * i as f64 + e
        })
        .collect()
}

/// Share of `trials` AR(1) series of length `n` the test does *not* call stationary
/// (`Some(true)` or not yet testable), with a linear drift totalling `drift_sd`
/// marginal standard deviations over the series.
fn flag_rate(n: usize, rho: f64, drift_sd: f64, trials: u64) -> f64 {
    let sd = 0.1;
    let per_step = drift_sd * sd / n as f64;
    let hits = (1..=trials)
        .filter(|&s| trend_test(&series(s, n, rho, sd, per_step)) != Some(false))
        .count();
    hits as f64 / trials as f64
}

/// Regression this exists to catch: a test whose points are closer together than
/// the series' autocorrelation time reads wander as trend. The previous rule (five
/// 20-iteration batch means) flagged 12–23 % of stationary SAEM coordinates. With
/// the spacing taken from the autocorrelation time the null rate stays near the
/// nominal 5 % (NONMEM `CALPHA=0.05`) up to rho = 0.97 (autocorrelation time ~66,
/// clamped to 50). Bounds: measured value plus a ~4-sigma binomial band.
#[test]
fn stationary_autocorrelated_noise_is_not_flagged_beyond_nominal() {
    let r0 = flag_rate(2000, 0.0, 0.0, 400);
    let r9 = flag_rate(2000, 0.9, 0.0, 400);
    let r97 = flag_rate(2000, 0.97, 0.0, 400);
    eprintln!("null rates: rho 0 {r0}, 0.9 {r9}, 0.97 {r97}");
    assert!(r0 < 0.09, "iid null rate {r0}");
    assert!(r9 < 0.11, "rho 0.9 null rate {r9}");
    assert!(r97 < 0.17, "rho 0.97 null rate {r97}");
}

/// The other half: a real drift must be seen, more often the larger it is. The test
/// reads only its latest ten points (about 170 iterations at rho = 0.9, spacing ~19),
/// so a drift of `D` sd over the 600-iteration series is ~0.29·D sd across them:
/// 15 sd there is ~4.3 sd in the window, 30 sd ~8.6.
#[test]
fn drift_is_detected_with_power_growing_in_its_size() {
    let weak = flag_rate(600, 0.9, 5.0, 300);
    let mid = flag_rate(600, 0.9, 15.0, 300);
    let strong = flag_rate(600, 0.9, 30.0, 200);
    eprintln!("power: 5 sd {weak}, 15 sd {mid}, 30 sd {strong}");
    assert!(weak < mid && mid <= strong, "{weak} {mid} {strong}");
    assert!(mid > 0.5, "15-sd drift power {mid}");
    assert!(strong > 0.95, "30-sd drift power {strong}");
}

#[test]
fn decaying_variance_trajectory_is_a_trend_and_its_plateau_is_not() {
    // log omega decaying as exp(-k/150) towards a plateau, plus rho = 0.9 noise: the
    // slow decay of a weakly identified variance this rule exists for.
    let still = (1..=40u64)
        .filter(|&s| {
            let noise = series(s, 300, 0.9, 0.05, 0.0);
            let y: Vec<f64> =
                (0..300).map(|k| -2.0 - 3.0 * (-(k as f64) / 150.0).exp() + noise[k]).collect();
            trend_test(&y) != Some(false)
        })
        .count();
    let settled = (1..=40u64)
        .filter(|&s| {
            let noise = series(s, 2000, 0.9, 0.05, 0.0);
            let y: Vec<f64> =
                (0..2000).map(|k| -2.0 - 3.0 * (-(k as f64) / 150.0).exp() + noise[k]).collect();
            trend_test(&y) == Some(false)
        })
        .count();
    eprintln!("decay: still flagged at 300 in {still}/40, settled at 2000 in {settled}/40");
    assert!(still >= 38, "still decaying at 300: {still}/40");
    assert!(settled >= 33, "settled by 2000: {settled}/40");
}

#[test]
fn constant_short_and_nonfinite_series() {
    assert_eq!(trend_test(&vec![0.3; 200]), Some(false), "fixed parameter");
    assert_eq!(trend_test(&vec![0.0; 200]), Some(false), "structural zero");
    assert_eq!(trend_test(&[1.0, 2.0, 3.0]), None, "shorter than a window");
    assert_eq!(
        trend_test(&vec![0.3; MIN_TEST_WINDOW - 1]),
        None,
        "one short of the shortest window"
    );
    let mut y = series(1, 600, 0.0, 0.1, 0.0);
    let last = y.len() - 1;
    y[last] = f64::NAN;
    assert_eq!(trend_test(&y), Some(true), "a diverged coordinate is not converged");
    y[last] = f64::INFINITY;
    assert_eq!(trend_test(&y), Some(true));
}

/// A parameter that is fixed but wobbles in the last bits (a value round-tripped through
/// exp/ln each iteration) has a numerically zero residual, so its slope/se ratio is
/// huge: the constant guard, not the t-test, must call it constant.
#[test]
fn rounding_level_ramp_is_constant_not_a_trend() {
    // 600 long: a noiseless ramp's detrended residual is rounding noise, which reads as
    // maximally autocorrelated (spacing 50), so it needs the longest window (500).
    let y: Vec<f64> = (0..600).map(|i| 0.3 + 1e-15 * i as f64).collect();
    assert_eq!(trend_test(&y), Some(false));
    // The same ramp at a visible scale is a trend (so the guard is what decides above).
    let y: Vec<f64> = (0..600).map(|i| 0.3 + 1e-3 * i as f64).collect();
    assert_eq!(trend_test(&y), Some(true));
}

#[test]
fn only_the_latest_points_are_tested() {
    // A long steep history followed by a long flat stretch: not a trend.
    let mut y: Vec<f64> = (0..300).map(|k| 10.0 - 0.05 * k as f64).collect();
    y.extend(series(3, 1200, 0.0, 0.01, 0.0).iter().map(|v| v - 6.0));
    assert_eq!(trend_test(&y), Some(false));
}

#[test]
fn spacing_follows_the_autocorrelation_time_within_its_bounds() {
    // iid: autocorrelation time 1, raised to the minimum spacing.
    assert_eq!(spacing(&series(5, 2000, 0.0, 0.1, 0.0)), SPACING_MIN);
    // rho = 0.8: time (1 + 0.8)/(1 - 0.8) = 9, so around the minimum.
    let s8 = spacing(&series(6, 2000, 0.8, 0.1, 0.0));
    assert!((SPACING_MIN..=16).contains(&s8), "rho 0.8 spacing {s8}");
    // rho = 0.97: time ~66, clamped to the maximum.
    assert_eq!(spacing(&series(7, 2000, 0.97, 0.1, 0.0)), SPACING_MAX);
    // A linear trend is removed before the estimate: it does not inflate the spacing.
    assert_eq!(spacing(&series(5, 2000, 0.0, 0.1, 0.001)), SPACING_MIN);
}

fn push_all(m: &mut ExplorationMonitor, cols: &[Vec<f64>], upto: usize) {
    for k in m.len()..upto {
        let row: Vec<f64> = cols.iter().map(|c| c[k]).collect();
        m.push(&row);
    }
}

#[test]
fn monitor_stationary_stops_on_a_check_after_the_minimum_and_names_nothing() {
    let cols: Vec<Vec<f64>> = (0..6).map(|c| series(100 + c, 800, 0.0, 0.05, 0.0)).collect();
    let mut m = ExplorationMonitor::default();
    let mut stop = None;
    for k in 1..=800 {
        push_all(&mut m, &cols, k);
        if m.may_stop_at(k, 150) {
            stop = Some(k);
            break;
        }
    }
    let stop = stop.expect("stationary coordinates must pass the test");
    assert!(stop >= 150, "never before the minimum, got {stop}");
    assert_eq!(stop % CHECK_EVERY, 0, "only on a check iteration, got {stop}");
    assert!(m.drifting().is_empty());
    // Before the shortest window nothing can be declared converged.
    let mut early = ExplorationMonitor::default();
    push_all(&mut early, &cols, MIN_TEST_WINDOW - 1);
    assert!(!early.converged());
    assert!(!early.may_stop_at(MIN_TEST_WINDOW - 1, 0));
    assert_eq!(early.drifting().len(), cols.len(), "untestable coordinates are listed");
}

#[test]
fn monitor_one_drifting_coordinate_blocks_the_stop_and_is_named() {
    let mut cols: Vec<Vec<f64>> = (0..4).map(|c| series(200 + c, 600, 0.0, 0.05, 0.0)).collect();
    // Coordinate 2 decays steadily: -0.01 per iteration against sd 0.05 noise (20 sd
    // across its 100-iteration test window).
    cols[2] = series(9, 600, 0.0, 0.05, -0.01);
    let mut m = ExplorationMonitor::default();
    push_all(&mut m, &cols, 600);
    assert!(!m.converged());
    assert_eq!(m.drifting(), vec![2]);
    assert!(!m.may_stop_at(600, 150));
}

#[test]
fn bounds_precedence_explicit_length_overrides_the_automatic_stop() {
    let mut o = FitOptions::default();
    assert_eq!(exploration_bounds(&o), (true, 150, 4000), "default is automatic");
    o.saem_exploration_min = 100;
    o.saem_exploration_max = 300;
    assert_eq!(exploration_bounds(&o), (true, 100, 300));
    // A maximum below the minimum is raised to it.
    o.saem_exploration_max = 50;
    assert_eq!(exploration_bounds(&o), (true, 100, 100));
    // An explicit length is fixed and ignores min/max.
    o.saem_n_exploration = Some(40);
    o.saem_exploration_min = 100;
    o.saem_exploration_max = 300;
    assert_eq!(exploration_bounds(&o), (false, 40, 40));
}

#[test]
fn tracked_coordinates_are_log_variance_correlation_and_log_sigma() {
    let omega = DMatrix::from_row_slice(2, 2, &[0.04, 0.012, 0.012, 0.09]);
    let c = exploration_coords(-812.5, &[1.5, -2.0], &omega, None, &[0.2]);
    let corr = 0.012 / (0.04_f64 * 0.09).sqrt();
    assert_eq!(c.len(), 1 + 2 + 2 + 1 + 1);
    // The complete-data −log L leads (Monolix's convergence indicator).
    assert_eq!(c[0], -812.5);
    assert_eq!(&c[1..3], &[1.5, -2.0]);
    assert!((c[3] - 0.04_f64.ln()).abs() < 1e-14);
    assert!((c[4] - 0.09_f64.ln()).abs() < 1e-14);
    assert!((c[5] - corr).abs() < 1e-14);
    assert!((c[6] - 0.2_f64.ln()).abs() < 1e-14);
    // IOV block appended after the BSV block, before sigma.
    let iov = DMatrix::from_row_slice(1, 1, &[0.5]);
    let c = exploration_coords(0.0, &[], &omega, Some(&iov), &[0.2]);
    assert!((c[c.len() - 2] - 0.5_f64.ln()).abs() < 1e-14);
}

#[test]
fn coordinate_names_follow_the_coordinate_order() {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let theta = s(&["TVCL", "TVV"]);
    let eta = s(&["ETA_CL", "ETA_V", "ETA_KA"]);
    let kappa = s(&["KAPPA_CL"]);
    let sigma = s(&["PROP"]);
    let names = exploration_coord_names(&theta, &eta, Some(&kappa), &sigma);
    let omega = DMatrix::from_diagonal_element(3, 3, 0.1);
    let iov = DMatrix::from_diagonal_element(1, 1, 0.1);
    let coords = exploration_coords(0.0, &[1.0, 2.0], &omega, Some(&iov), &[0.1]);
    assert_eq!(names.len(), coords.len());
    assert_eq!(
        names,
        s(&[
            "complete-data -logL",
            "TVCL",
            "TVV",
            "OMEGA(ETA_CL)",
            "OMEGA(ETA_V)",
            "OMEGA(ETA_KA)",
            "corr OMEGA(ETA_CL,ETA_V)",
            "corr OMEGA(ETA_CL,ETA_KA)",
            "corr OMEGA(ETA_V,ETA_KA)",
            "OMEGA_IOV(KAPPA_CL)",
            "SIGMA(PROP)",
        ])
    );
}

#[test]
fn drifting_summary_names_and_caps() {
    let names: Vec<String> = (0..12).map(|i| format!("p{i}")).collect();
    assert_eq!(drifting_summary(&[3, 7], &names, 8), "p3, p7");
    assert_eq!(drifting_summary(&[0, 1, 2], &names, 2), "p0, p1, and 1 more");
    assert_eq!(drifting_summary(&[20], &names, 8), "#20");
    assert_eq!(drifting_summary(&[], &names, 8), "none");
}
