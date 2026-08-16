//! Monomorphised dual-width buckets for the const-generic dispatch ladders (#971).
//!
//! Every analytic sensitivity walk is generic over a compile-time dual width
//! (`Dual1<N>` / `Dual2<M>`), and the entry points pick the instantiation with a
//! `match` over the runtime axis count. Enumerating *every* width `1..=cap`
//! monomorphises the whole ODE-integration + sensitivity stack once per width:
//! at the `MAX_ODE_IOV_AXES = 96` cap that ladder alone was 38.6 % of the crate's
//! LLVM IR, and ~93 % of the lib compile is LLVM (#969/#970).
//!
//! The fix is to **round the runtime width up to the next bucket** and leave the
//! extra lanes zero. Padding is semantically inert: the dual arithmetic chains
//! every lane, and a zero lane stays zero; every seeder guards its axis writes
//! (`ax < N`) and every readout indexes by the *runtime* axis count
//! (`n_theta` / `n_stacked`), never by `N`. Padding is not free at *runtime* —
//! `Dual1` costs `O(N)` per op and `Dual2` costs `O(M²)` — so the small widths,
//! which are the common case, stay exact and only the tail is bucketed.
//!
//! A width past the largest bucket must still route to FD **loudly**, through the
//! per-subject gate (`ode_iov_subject_supported`) rather than through a dispatch
//! `_ => None`; [`bucket_for`] returning `None` is the backstop, and the
//! [`buckets_well_formed`] compile-time check keeps the ladder's last entry pinned
//! to the cap the gate enforces.

/// Smallest width in `buckets` that can hold `dim` live axes, or `None` when
/// `dim` is zero or wider than the largest bucket (caller routes to FD).
///
/// `buckets` must be strictly ascending — [`buckets_well_formed`] is the
/// compile-time guard for that at each ladder's definition site.
pub(crate) const fn bucket_for(dim: usize, buckets: &[usize]) -> Option<usize> {
    if dim == 0 {
        return None;
    }
    let mut i = 0;
    while i < buckets.len() {
        if dim <= buckets[i] {
            return Some(buckets[i]);
        }
        i += 1;
    }
    None
}

/// Whether `buckets` is non-empty, strictly ascending, and ends exactly at `cap`.
///
/// Used in a `const _: () = assert!(…)` next to each ladder so a widened cap, a
/// reordered ladder, or a bucket list that no longer reaches the cap fails to
/// compile instead of silently sending in-scope subjects to the `_ => None` FD
/// arm (the #438/#466/#534 tripwire convention, carried over to the bucketed
/// form).
pub(crate) const fn buckets_well_formed(buckets: &[usize], cap: usize) -> bool {
    if buckets.is_empty() {
        return false;
    }
    let mut i = 1;
    while i < buckets.len() {
        if buckets[i - 1] >= buckets[i] {
            return false;
        }
        i += 1;
    }
    buckets[buckets.len() - 1] == cap
}

/// Slice equality for `const` contexts — the guard that a dispatch macro's literal
/// arm list still spells exactly the ladder const it looks buckets up in. Without it
/// a bucket added to the const (but not to the arms) would fall through to the
/// `_ => None` FD arm at runtime with nothing complaining at compile time.
pub(crate) const fn slices_eq(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: [usize; 6] = [1, 2, 4, 8, 16, 32];

    #[test]
    fn bucket_for_rounds_up_to_the_next_width() {
        assert_eq!(bucket_for(1, &B), Some(1));
        assert_eq!(bucket_for(2, &B), Some(2));
        assert_eq!(bucket_for(3, &B), Some(4));
        assert_eq!(bucket_for(5, &B), Some(8));
        assert_eq!(bucket_for(9, &B), Some(16));
        assert_eq!(bucket_for(17, &B), Some(32));
        assert_eq!(bucket_for(32, &B), Some(32));
    }

    #[test]
    fn bucket_for_declines_zero_and_past_the_cap() {
        assert_eq!(bucket_for(0, &B), None, "no live axes → FD");
        assert_eq!(bucket_for(33, &B), None, "past the last bucket → FD");
        assert_eq!(bucket_for(usize::MAX, &B), None);
    }

    #[test]
    fn bucket_for_is_idempotent_on_bucket_boundaries() {
        for &b in &B {
            assert_eq!(bucket_for(b, &B), Some(b));
        }
    }

    #[test]
    fn well_formed_accepts_ascending_ladder_ending_at_cap() {
        assert!(buckets_well_formed(&B, 32));
    }

    #[test]
    fn well_formed_rejects_bad_ladders() {
        assert!(
            !buckets_well_formed(&B, 64),
            "last bucket must equal the cap"
        );
        assert!(!buckets_well_formed(&[], 0), "empty ladder");
        assert!(!buckets_well_formed(&[1, 4, 2], 2), "not ascending");
        assert!(
            !buckets_well_formed(&[1, 4, 4], 4),
            "not strictly ascending"
        );
    }

    #[test]
    fn slices_eq_compares_elementwise() {
        assert!(slices_eq(&B, &[1, 2, 4, 8, 16, 32]));
        assert!(slices_eq(&[], &[]));
        assert!(!slices_eq(&B, &[1, 2, 4, 8, 16]), "length differs");
        assert!(!slices_eq(&B, &[1, 2, 4, 8, 16, 64]), "element differs");
    }
}
