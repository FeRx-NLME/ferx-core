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

/// A stacked vector whose every entry is distinct, so a slice taken at the wrong offset
/// cannot coincide with the right one.
fn distinct(n: usize) -> Vec<f64> {
    (0..n).map(|i| 10.0 + i as f64).collect()
}

/// #1082 PR1 is a pure refactor: the layout must reproduce, index for index, the stacked
/// convention every caller wrote out by hand before it — `[η_bsv, κ_0, …, κ_{K−1}]` with
/// group `g` at `n_eta + g·n_kappa`. The formula is spelled out literally here, not through
/// the module's own helpers, so a change to `kappa_offset` cannot move both sides.
#[test]
fn layout_reproduces_the_pre_1082_stacked_convention_index_for_index() {
    let s = subject();
    let layout = StackedLayout::new(N_ETA, N_KAPPA, &s);

    // Group order: observation occasions first-seen (2, 5, 1), then dose-only (7, 9).
    let labels: Vec<u32> = layout.groups().iter().map(|(occ, _)| *occ).collect();
    assert_eq!(labels, vec![2, 5, 1, 7, 9]);
    assert_eq!(
        layout.groups()[3].1,
        Vec::<usize>::new(),
        "dose-only group owns no obs"
    );
    let k = labels.len();
    assert_eq!(layout.n_groups(), k);
    assert_eq!(group_count(&s), k);
    assert_eq!(layout.n_stacked(), N_ETA + k * N_KAPPA);
    assert_eq!(stacked_width(N_ETA, N_KAPPA, k), N_ETA + k * N_KAPPA);

    for (g, &occ) in labels.iter().enumerate() {
        assert_eq!(layout.group_of(occ), Some(g), "occasion {occ}");
        assert_eq!(layout.kappa_offset(g), N_ETA + g * N_KAPPA, "group {g}");
        assert_eq!(
            kappa_offset(N_ETA, N_KAPPA, g),
            N_ETA + g * N_KAPPA,
            "group {g}"
        );
        assert_eq!(
            layout.kappa_range(g),
            N_ETA + g * N_KAPPA..N_ETA + (g + 1) * N_KAPPA
        );
    }
    // An occasion with no group (production asks with `u32::MAX` on a miss), and a label
    // no record carries, both map to no group.
    assert_eq!(layout.group_of(u32::MAX), None);
    assert_eq!(layout.group_of(3), None);

    let stacked = distinct(layout.n_stacked());
    for g in 0..k {
        let base = N_ETA + g * N_KAPPA;
        let mut want = stacked[..N_ETA].to_vec();
        want.extend_from_slice(&stacked[base..base + N_KAPPA]);
        assert_eq!(layout.combined_effect(&stacked, g), want, "group {g}");
        assert_eq!(layout.kappa(&stacked, g), &stacked[base..base + N_KAPPA]);
    }
    let mut pk_only = stacked[..N_ETA].to_vec();
    pk_only.extend([0.0; N_KAPPA]);
    assert_eq!(layout.combined_pk_only(&stacked), pk_only);
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

/// `kappa_chunks` serves the route decision, which must not panic on a vector of the
/// wrong width: it cuts whatever κ tail there is, and nothing when there is none.
#[test]
fn kappa_chunks_cut_the_tail_and_tolerate_a_malformed_width() {
    let s = subject();
    let layout = StackedLayout::new(N_ETA, N_KAPPA, &s);
    let stacked = distinct(layout.n_stacked());
    let chunks = layout.kappa_chunks(&stacked);
    assert_eq!(chunks.len(), layout.n_groups());
    for (g, c) in chunks.iter().enumerate() {
        assert_eq!(*c, layout.kappa(&stacked, g));
    }
    // One short: the last block is short rather than a panic.
    let short = layout.kappa_chunks(&stacked[..stacked.len() - 1]);
    assert_eq!(short.len(), layout.n_groups());
    assert_eq!(short.last().unwrap().len(), N_KAPPA - 1);
    // Shorter than η: no κ tail at all.
    assert!(layout.kappa_chunks(&stacked[..N_ETA - 1]).is_empty());
    // No κ declared: nothing to cut, whatever the vector.
    assert!(StackedLayout::new(N_ETA, 0, &s)
        .kappa_chunks(&stacked)
        .is_empty());
}
