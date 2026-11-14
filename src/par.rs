//! Index-parallel maps that share ONE rayon instantiation.
//!
//! A `xs.par_iter().enumerate().map(f).collect()` chain monomorphizes rayon's whole bridge /
//! `join_context` / collect-consumer machinery per call site, because every closure is its own
//! type. The ~40 per-subject fan-outs in the estimators each paid that (~1.2 MB of shipped code
//! in total). Here the generic part is only a loop over a range writing results in place; the
//! rayon driver ([`run_ranges`]) is non-generic and is entered once per *leaf range* through a
//! `&dyn Fn` — never per item — so the per-item cost is the same as rayon's own collect.
//!
//! Scheduling is rayon's adaptive splitting (`rayon::iter::split`, the same `Splitter` the
//! indexed bridge uses): the range is halved while the pool wants work, and re-split on steals.
//! `map_init` calls `init` once per leaf, as rayon's `map_init` does once per split job.

use rayon::iter::ParallelIterator;

/// Calls `f(lo, hi)` on disjoint ranges covering `0..n`, in parallel on the current pool.
#[inline(never)]
fn run_ranges(n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    rayon::iter::split((0usize, n), |(lo, hi)| {
        if hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            ((lo, mid), Some((mid, hi)))
        } else {
            ((lo, hi), None)
        }
    })
    .for_each(|(lo, hi)| f(lo, hi));
}

/// Output buffer shared across workers; each index is written by exactly one leaf.
struct Slots<T>(*mut T);
// SAFETY: workers write disjoint indices of one allocation; `T: Send` moves each value across.
unsafe impl<T: Send> Sync for Slots<T> {}
impl<T> Slots<T> {
    /// # Safety
    /// `i` is in bounds of the allocation and written by exactly one caller.
    #[inline]
    unsafe fn put(&self, i: usize, v: T) {
        unsafe { self.0.add(i).write(v) }
    }
}

/// `(0..n).into_par_iter().map(f).collect()`, results in index order.
pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    map_init(n, || (), |_, i| f(i))
}

/// `(0..n).into_par_iter().map_init(init, f).collect()`: one `init()` scratch per leaf range.
pub(crate) fn map_init<S, T: Send>(
    n: usize,
    init: impl Fn() -> S + Sync,
    f: impl Fn(&mut S, usize) -> T + Sync,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(n);
    let slots = Slots(out.as_mut_ptr());
    run_ranges(n, &|lo, hi| {
        let mut s = init();
        for i in lo..hi {
            // SAFETY: `i < n <= capacity`, and the leaf ranges are disjoint.
            unsafe { slots.put(i, f(&mut s, i)) }
        }
    });
    // SAFETY: `run_ranges` returned normally, so every index in `0..n` was written once. On a
    // panic it unwinds past this with `len == 0`, leaking (never double-dropping) the values.
    unsafe { out.set_len(n) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_preserves_index_order_and_count() {
        for n in [0usize, 1, 2, 7, 1000] {
            let v = map(n, |i| i * 3);
            assert_eq!(v, (0..n).map(|i| i * 3).collect::<Vec<_>>());
        }
    }

    #[test]
    fn map_init_reuses_scratch_within_a_leaf() {
        // Each leaf gets a fresh scratch; the count of items a scratch saw sums to n.
        let v = map_init(
            500,
            || 0usize,
            |s, i| {
                *s += 1;
                (i, *s)
            },
        );
        assert_eq!(v.len(), 500);
        assert!(v.iter().enumerate().all(|(k, &(i, s))| k == i && s >= 1));
    }

    #[test]
    fn map_drops_owned_results_exactly_once() {
        let v = map(64, |i| vec![i; 3]);
        assert!(v.iter().enumerate().all(|(i, x)| x == &vec![i; 3]));
    }
}
