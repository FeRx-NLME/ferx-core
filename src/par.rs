//! Index-parallel maps that share ONE rayon instantiation.
//!
//! A `xs.par_iter().enumerate().map(f).collect()` chain monomorphizes rayon's whole bridge /
//! `join_context` / collect-consumer machinery per call site, because every closure is its own
//! type. The ~40 per-subject fan-outs in the estimators each paid that (~1.2 MB of shipped code
//! in total). Here the generic part is only a loop over a range writing results in place; the
//! rayon driver ([`run_ranges`]) is non-generic and is entered once per *leaf range* through a
//! `&dyn Fn` — never per item — so the per-item cost is the same as rayon's own collect.
//!
//! Scheduling is fixed-grain: `0..n` is cut into about [`LEAVES_PER_THREAD`] contiguous leaves
//! per worker, and each leaf is its own stealable job. Rayon's adaptive splitting
//! (`rayon::iter::split`) was used before #3216, but it makes a leaf only as fine as the pool
//! looked hungry when the leaf was cut, and a leaf its own worker pops is never re-split, so on
//! few workers one long leaf could keep running after the rest ran dry. FOCEI on
//! `busulfan_dominic` (1,518 subjects, ODE+IOV) on 2 threads measured 77% busy (summed leaf
//! time / (wall × workers), first 60 s of the fit) against 98% with fixed grain. `map_init`
//! calls `init` once per leaf.

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Leaves per worker. A started leaf cannot be handed off, so at the end of a pass the other
/// workers wait on at most one leaf: about `1 / (workers × LEAVES_PER_THREAD)` of the items.
const LEAVES_PER_THREAD: usize = 16;

/// The leaf length [`run_ranges`] cuts `0..n` into on a pool of `workers`.
fn grain(n: usize, workers: usize) -> usize {
    n.div_ceil(workers.max(1) * LEAVES_PER_THREAD).max(1)
}

/// Calls `f(lo, hi)` on disjoint ranges covering `0..n`, in parallel on the current pool.
#[inline(never)]
fn run_ranges(n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    // A leaf may run on a worker another fit shares: it reads the plans of the fit that spawned
    // it (#2786).
    let fit = crate::parser::model_parser::rhs_regcode::kernel_jit::current_fit();
    let grain = grain(n, rayon::current_num_threads());
    // `with_max_len(1)`: every leaf is its own job, so an idle worker can always steal one.
    (0..n.div_ceil(grain)).into_par_iter().with_max_len(1).for_each(|leaf| {
        let lo = leaf * grain;
        let hi = (lo + grain).min(n);
        crate::parser::model_parser::rhs_regcode::kernel_jit::in_fit(fit, || f(lo, hi))
    });
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
    n: usize, init: impl Fn() -> S + Sync, f: impl Fn(&mut S, usize) -> T + Sync,
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

    /// #3216: no leaf is longer than the fixed grain, however the pool happens to steal. Under
    /// the old adaptive split a 1-worker pool cut 1000 items into two leaves of 500, and a
    /// 2-worker pool left an un-stolen leaf of about 250; the grain is 63 and 32 here.
    #[test]
    fn leaves_are_never_longer_than_the_grain() {
        let n = 1000;
        for workers in [1usize, 2] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
            let v = pool.install(|| {
                map_init(
                    n,
                    || 0usize,
                    |s, _| {
                        *s += 1;
                        *s
                    },
                )
            });
            let longest = v.iter().copied().max().unwrap();
            let g = grain(n, workers);
            assert_eq!(g, n.div_ceil(workers * LEAVES_PER_THREAD));
            assert!(longest <= g, "workers = {workers}: a leaf of {longest} > grain {g}");
            // Every leaf but the last is exactly `g` long: count the leaves that reached `g`.
            assert_eq!(v.iter().filter(|&&s| s == g).count(), n / g, "workers = {workers}");
        }
    }

    #[test]
    fn grain_is_at_least_one() {
        assert_eq!(grain(0, 8), 1);
        assert_eq!(grain(3, 8), 1);
        assert_eq!(grain(1000, 0), grain(1000, 1));
    }

    #[test]
    fn map_drops_owned_results_exactly_once() {
        let v = map(64, |i| vec![i; 3]);
        assert!(v.iter().enumerate().all(|(i, x)| x == &vec![i; 3]));
    }
}
