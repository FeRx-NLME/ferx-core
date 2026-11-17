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

fn detection_rate(rho: f64, drift: f64, trials: u64) -> f64 {
    let hits = (1..=trials)
        .filter(|&s| trend_detected(&series(s, TREND_WINDOW, rho, 0.1, drift)))
        .count();
    hits as f64 / trials as f64
}

/// Regression this exists to catch: an OLS slope test that ignores
/// autocorrelation flags stationary SAEM trajectories (measured 20-40 % of
/// windows on real log-Ω traces), so the stop never fires. Measured here over
/// 2000 AR(1) windows: 0.051 / 0.058 / 0.087 at rho = 0 / 0.5 / 0.9 against the
/// nominal 0.05; the 0.12 bound has the headroom of a 4-sigma binomial band.
#[test]
fn stationary_autocorrelated_noise_is_not_flagged_beyond_nominal() {
    let r0 = detection_rate(0.0, 0.0, 2000);
    let r5 = detection_rate(0.5, 0.0, 2000);
    let r9 = detection_rate(0.9, 0.0, 2000);
    assert!((0.03..0.08).contains(&r0), "iid null rate {r0}");
    assert!(r5 < 0.09, "rho 0.5 null rate {r5}");
    assert!(r9 < 0.12, "rho 0.9 null rate {r9}");
}

/// The other half: a real drift must be seen. Total drift over the window of
/// 4 marginal sds (0.004 * 100 / 0.1) is detected 75 % of the time at rho = 0.9
/// (measured 0.751), and a 10 sd drift essentially always (the bounds are the
/// measured value minus a binomial band).
#[test]
fn drift_is_detected_with_power_growing_in_its_size() {
    let weak = detection_rate(0.9, 0.004, 2000);
    let strong = detection_rate(0.9, 0.01, 500);
    assert!(weak > 0.65, "4-sd drift power {weak}");
    assert!(strong > 0.99, "10-sd drift power {strong}");
    // Monotone in the drift: a smaller one is flagged less often.
    assert!(detection_rate(0.9, 0.001, 2000) < weak);
}

#[test]
fn decaying_variance_trajectory_is_a_trend_and_its_plateau_is_not() {
    // log omega decaying as exp(-k/80) towards a plateau, plus rho = 0.9 noise.
    let noise = series(7, 400, 0.9, 0.05, 0.0);
    let y: Vec<f64> = (0..400)
        .map(|k| -2.0 - 3.0 * (-(k as f64) / 80.0).exp() + noise[k])
        .collect();
    assert!(trend_detected(&y[..120]), "still decaying at 120");
    assert!(!trend_detected(&y[300..]), "settled by 300");
}

#[test]
fn constant_short_and_nonfinite_series() {
    assert!(!trend_detected(&vec![0.3; TREND_WINDOW]), "fixed parameter");
    assert!(!trend_detected(&vec![0.0; TREND_WINDOW]), "structural zero");
    assert!(!trend_detected(&[1.0, 2.0, 3.0]), "shorter than a window");
    let mut y = series(1, TREND_WINDOW, 0.5, 0.1, 0.0);
    y[10] = f64::NAN;
    assert!(trend_detected(&y), "a diverged coordinate is not converged");
    y[10] = f64::INFINITY;
    assert!(trend_detected(&y));
}

/// A parameter that is fixed but wobbles in the last bits (a value round-tripped through
/// exp/ln each iteration) has a numerically zero residual, so its slope/se ratio is
/// huge: the constant guard, not the t-test, must call it constant.
#[test]
fn rounding_level_ramp_is_constant_not_a_trend() {
    let y: Vec<f64> = (0..TREND_WINDOW).map(|i| 0.3 + 1e-15 * i as f64).collect();
    assert!(!trend_detected(&y));
    // The same ramp at a visible scale is a trend (so the guard is what decides above).
    let y: Vec<f64> = (0..TREND_WINDOW).map(|i| 0.3 + 1e-3 * i as f64).collect();
    assert!(trend_detected(&y));
}

#[test]
fn only_the_trailing_window_is_read() {
    // A long steep history followed by a flat window: not a trend.
    let mut y: Vec<f64> = (0..200).map(|k| 10.0 - 0.05 * k as f64).collect();
    y.extend(series(3, TREND_WINDOW, 0.5, 0.01, 0.0));
    assert!(!trend_detected(&y));
}

fn push_all(m: &mut ExplorationMonitor, cols: &[Vec<f64>], upto: usize) {
    for k in m.len()..upto {
        let row: Vec<f64> = cols.iter().map(|c| c[k]).collect();
        m.push(&row);
    }
}

#[test]
fn monitor_stationary_stops_at_the_minimum_and_names_nothing() {
    let cols: Vec<Vec<f64>> = (0..6)
        .map(|c| series(100 + c, 400, 0.5, 0.05, 0.0))
        .collect();
    // Find the first stop for a given seed set by replaying the loop.
    let mut m = ExplorationMonitor::default();
    let mut stop = None;
    for k in 1..=400 {
        push_all(&mut m, &cols, k);
        if m.may_stop_at(k, 150) {
            stop = Some(k);
            break;
        }
    }
    let stop = stop.expect("stationary coordinates must pass the test");
    assert!(stop >= 150, "never before the minimum, got {stop}");
    assert!(m.drifting().is_empty());
    // Before a full window nothing can be declared converged.
    let mut early = ExplorationMonitor::default();
    push_all(&mut early, &cols, TREND_WINDOW - 1);
    assert!(!early.converged());
    assert!(!early.may_stop_at(TREND_WINDOW - 1, 0));
}

#[test]
fn monitor_one_drifting_coordinate_blocks_the_stop_and_is_named() {
    let mut cols: Vec<Vec<f64>> = (0..4)
        .map(|c| series(200 + c, 300, 0.5, 0.05, 0.0))
        .collect();
    // Coordinate 2 decays steadily: -0.01 per iteration against sd 0.05 noise.
    cols[2] = series(9, 300, 0.5, 0.05, -0.01);
    let mut m = ExplorationMonitor::default();
    push_all(&mut m, &cols, 300);
    assert!(!m.converged());
    assert_eq!(m.drifting(), vec![2]);
    assert!(!m.may_stop_at(300, 150));
}

#[test]
fn bounds_precedence_explicit_length_overrides_the_automatic_stop() {
    let mut o = FitOptions::default();
    assert_eq!(
        exploration_bounds(&o),
        (true, 150, 500),
        "default is automatic"
    );
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
    let c = exploration_coords(&[1.5, -2.0], &omega, None, &[0.2]);
    let corr = 0.012 / (0.04_f64 * 0.09).sqrt();
    assert_eq!(c.len(), 2 + 2 + 1 + 1);
    assert_eq!(&c[..2], &[1.5, -2.0]);
    assert!((c[2] - 0.04_f64.ln()).abs() < 1e-14);
    assert!((c[3] - 0.09_f64.ln()).abs() < 1e-14);
    assert!((c[4] - corr).abs() < 1e-14);
    assert!((c[5] - 0.2_f64.ln()).abs() < 1e-14);
    // IOV block appended after the BSV block, before sigma.
    let iov = DMatrix::from_row_slice(1, 1, &[0.5]);
    let c = exploration_coords(&[], &omega, Some(&iov), &[0.2]);
    assert!((c[c.len() - 2] - 0.5_f64.ln()).abs() < 1e-14);
}
