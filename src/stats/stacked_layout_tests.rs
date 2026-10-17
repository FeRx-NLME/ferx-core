use super::*;
use crate::types::DoseEvent;

const N_ETA: usize = 2;
const N_KAPPA: usize = 3;

/// Observation occasions out of label order (first-seen order is the stacked order), two
/// **dose-only** occasions (7 and 9, no observation) appended after them, and an EVID=2
/// pk-only row, which does not create a group of its own.
fn subject() -> Subject {
    Subject {
        obs_rgroup: Vec::new(),
        id: "1".to_string(),
        doses: vec![
            DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(2.0, 100.0, 1, 0.0, false, 0.0),
            DoseEvent::new(4.0, 100.0, 1, 0.0, false, 0.0),
        ],
        obs_times: vec![1.0, 2.5, 3.0, 5.0, 6.0],
        obs_raw_times: Vec::new(),
        observations: vec![50.0, 40.0, 30.0, 45.0, 35.0],
        obs_cmts: vec![1; 5],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: vec![0.5],
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cens: vec![0; 5],
        occasions: vec![2, 2, 5, 5, 1],
        obs_l2: Vec::new(),
        dose_occasions: vec![7, 2, 9],
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
    }
}

/// The prior `Ω ⊕ (I_K ⊗ Ω_iov)`: `Ω` top-left, one `Ω_iov` per group on the diagonal at
/// the group's offset, and exact zeros everywhere else (no cross-group, no η–κ entry).
#[test]
fn prior_places_one_omega_iov_per_group_and_zeros_elsewhere() {
    let s = subject();
    let layout = StackedLayout::new(N_ETA, N_KAPPA, &s);
    let omega = DMatrix::from_fn(N_ETA, N_ETA, |i, j| 1.0 + (i * N_ETA + j) as f64);
    let omega_iov = DMatrix::from_fn(N_KAPPA, N_KAPPA, |i, j| 100.0 + (i * N_KAPPA + j) as f64);
    let prior = layout.prior(&omega, &omega_iov);
    let n = N_ETA + layout.n_groups() * N_KAPPA;
    assert_eq!(prior.shape(), (n, n));
    for r in 0..n {
        for c in 0..n {
            let want = if r < N_ETA && c < N_ETA {
                omega[(r, c)]
            } else if r >= N_ETA && c >= N_ETA && (r - N_ETA) / N_KAPPA == (c - N_ETA) / N_KAPPA {
                omega_iov[((r - N_ETA) % N_KAPPA, (c - N_ETA) % N_KAPPA)]
            } else {
                0.0
            };
            assert_eq!(prior[(r, c)].to_bits(), want.to_bits(), "entry ({r}, {c})");
        }
    }
}
