//! Per-thread reuse of the short-lived buffers a fit's hot paths build on every
//! evaluation (#2549 W1).
//!
//! An estimation evaluates each subject hundreds of thousands of times, and most of those
//! evaluations used to allocate their working vectors afresh: the predictions of a
//! recording, a solution point's state, the parameters of every record. Measured on main
//! `cec23d46`, swapping only the global allocator for mimalloc made fits 1.13–1.93× faster,
//! so the allocations themselves, not the arithmetic, were a first-order cost — and the
//! library cannot choose its embedder's allocator (ferx-r runs on the system one).
//!
//! [`take`] hands out an empty `Vec<T>` from this thread's spare list for `T` (or a new
//! one), and [`give`] returns it, cleared, for the next caller; a buffer may cross a call
//! boundary (a function's result) and be given back by its consumer. A taken buffer is out of the pool, so
//! a nested caller of the same type gets its own; nothing is shared while in use, and a
//! buffer's contents never outlive its holder (it is cleared on return).
//!
//! The spare list per type is bounded ([`MAX_SPARE`]), and a buffer that grew past
//! [`MAX_SPARE_BYTES`] is dropped rather than kept, so a single huge recording does not
//! pin its memory on a worker for the rest of a long-lived process.

use std::any::Any;
use std::cell::RefCell;

/// Most spare buffers kept per element type and thread: above what one evaluation holds at
/// once, so a steady state allocates nothing.
const MAX_SPARE: usize = 32;

/// A returned buffer whose capacity exceeds this many bytes is freed, not kept.
const MAX_SPARE_BYTES: usize = 1 << 22;

thread_local! {
    /// One `Vec<Vec<T>>` of spare buffers per element type `T`.
    static SPARE: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// An empty `Vec<T>` from this thread's spare list, or a new one.
#[inline]
pub(crate) fn take<T: 'static>() -> Vec<T> {
    if !crate::devflags::on(crate::devflags::POOL) {
        return Vec::new();
    }
    SPARE
        .try_with(|p| {
            p.borrow_mut()
                .iter_mut()
                .find_map(|b| b.downcast_mut::<Vec<Vec<T>>>())
                .and_then(Vec::pop)
        })
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// An empty `Vec<T>` with room for at least `n` elements.
#[inline]
pub(crate) fn take_with_capacity<T: 'static>(n: usize) -> Vec<T> {
    let mut v = take();
    v.reserve(n);
    v
}

/// Return a buffer from [`take`] to this thread's spare list, emptied.
#[inline]
pub(crate) fn give<T: 'static>(mut v: Vec<T>) {
    if v.capacity() == 0
        || v.capacity().saturating_mul(std::mem::size_of::<T>()) > MAX_SPARE_BYTES
        || !crate::devflags::on(crate::devflags::POOL)
    {
        return;
    }
    v.clear();
    // `try_with`: a buffer dropped while the thread-local is being torn down is freed.
    let _ = SPARE.try_with(|p| {
        let Ok(mut p) = p.try_borrow_mut() else {
            return;
        };
        match p.iter_mut().find_map(|b| b.downcast_mut::<Vec<Vec<T>>>()) {
            Some(spare) if spare.len() < MAX_SPARE => spare.push(v),
            Some(_) => {}
            None => p.push(Box::new(vec![v])),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A returned buffer is handed out again, empty, with its capacity; a taken one is out
    /// of the pool, so two live takes are two buffers.
    #[test]
    fn a_returned_buffer_is_reused_empty_and_live_ones_are_distinct() {
        let mut a: Vec<u64> = take();
        a.extend(0..100);
        let ptr = a.as_ptr();
        give(a);
        let b: Vec<u64> = take();
        assert!(b.is_empty(), "a reused buffer is empty");
        assert_eq!(b.as_ptr(), ptr, "the returned buffer is the one handed out");
        assert!(b.capacity() >= 100);
        let c: Vec<u64> = take();
        assert_ne!(
            c.as_ptr(),
            b.as_ptr(),
            "a live buffer is not handed out twice"
        );
        give(b);
        give(c);
    }

    /// A buffer grown past the byte cap is freed, not kept.
    #[test]
    fn a_huge_buffer_is_not_kept() {
        let big: Vec<u8> = vec![0; MAX_SPARE_BYTES + 1];
        let big_ptr = big.as_ptr();
        give(big);
        let r: Vec<u8> = take();
        assert_ne!(r.as_ptr(), big_ptr, "a huge buffer is not kept");
        give(r);
    }
}

/// Whether a recording over `T` draws its buffers from the pool: the value and first-order
/// types of the inner loop (`f64`, the η-gradient's forward duals), where measured; a
/// higher-order or taped recording allocates as before (#2549: pooling its 80-byte values
/// measured slower on the AGQ fit).
#[inline(always)]
pub(crate) fn pools<T: crate::sens::num::PkNum>() -> bool {
    T::JET_ORDER < 2
}

/// [`take_with_capacity`] where [`pools`] says so, else a fresh `Vec`.
#[inline(always)]
pub(crate) fn take_for<T: crate::sens::num::PkNum>(n: usize) -> Vec<T> {
    if pools::<T>() {
        take_with_capacity(n)
    } else {
        Vec::with_capacity(n)
    }
}

/// [`give`] where [`pools`] says so, else drop.
#[inline(always)]
pub(crate) fn give_for<T: crate::sens::num::PkNum>(v: Vec<T>) {
    if pools::<T>() {
        give(v);
    }
}

/// [`take_for`] for a `Vec<Option<T>>` (a recording's per-slot parameters).
#[inline(always)]
pub(crate) fn take_for_opt<T: crate::sens::num::PkNum>(n: usize) -> Vec<Option<T>> {
    if pools::<T>() {
        take_with_capacity(n)
    } else {
        Vec::with_capacity(n)
    }
}

/// [`give_for`] for a `Vec<Option<T>>`.
#[inline(always)]
pub(crate) fn give_opt_for<T: crate::sens::num::PkNum>(v: Vec<Option<T>>) {
    if pools::<T>() {
        give(v);
    }
}
