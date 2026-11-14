//! Stable sorts for cold paths, through a `dyn` comparator (the same helper as
//! `ferx_core`'s crate-private `cold_sort`; ten lines are not worth a public API item).
//!
//! `slice.sort_by(closure)` instantiates std's whole stable sort (~5 KB of driftsort +
//! quicksort) per call site, because every closure is its own type. Here the comparator is a
//! `&dyn Fn`, so the sort is instantiated once per *element type* however many sites use it.
//! The price is an indirect call per comparison — fine for parsing, reporting and post-fit
//! summaries, wrong for anything in an objective evaluation (those keep `sort_by`).
//! Same algorithm, same comparator, so the order is identical to `sort_by`'s.

use std::cmp::Ordering;

pub(crate) trait ColdSort<T> {
    /// `sort_by` with the comparator behind a `dyn`; stable.
    fn cold_sort_by(&mut self, cmp: &dyn Fn(&T, &T) -> Ordering);
    /// `sort_by_key` with the key behind a `dyn`; stable.
    fn cold_sort_by_key<K: Ord>(&mut self, key: &dyn Fn(&T) -> K) {
        self.cold_sort_by(&|a, b| key(a).cmp(&key(b)))
    }
}

impl<T> ColdSort<T> for [T] {
    fn cold_sort_by(&mut self, cmp: &dyn Fn(&T, &T) -> Ordering) {
        self.sort_by(|a, b| cmp(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_sorts_match_std_including_tie_order() {
        let v: Vec<(u8, usize)> = (0..200).map(|i| (((i * 37) % 11) as u8, i)).collect();
        let mut a = v.clone();
        let mut b = v.clone();
        a.cold_sort_by(&|x, y| x.0.cmp(&y.0));
        b.sort_by(|x, y| x.0.cmp(&y.0));
        assert_eq!(a, b);
        let mut c = v.clone();
        c.cold_sort_by_key(&|x| std::cmp::Reverse(x.0));
        let mut d = v;
        d.sort_by_key(|x| std::cmp::Reverse(x.0));
        assert_eq!(c, d);
    }
}
