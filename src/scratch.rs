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
//! one), and [`give`] returns it, cleared, for the next caller. [`Pooled`] does the return
//! on drop, so a buffer can be handed across a call boundary — a function's result — and
//! still go back when its consumer is done with it. A taken buffer is out of the pool, so
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

/// A `Vec<T>` that goes back to this thread's spare list when dropped: a pooled buffer that
/// can be returned from a function. Dereferences to the `Vec`.
pub(crate) struct Pooled<T: 'static>(Vec<T>);

impl<T: 'static> Pooled<T> {
    /// An empty pooled buffer.
    #[inline]
    pub(crate) fn new() -> Self {
        Pooled(take())
    }

    /// An empty pooled buffer with room for at least `n` elements.
    #[inline]
    pub(crate) fn with_capacity(n: usize) -> Self {
        Pooled(take_with_capacity(n))
    }

    /// The buffer, out of the pool's care (it will be freed, not returned, when dropped).
    #[inline]
    pub(crate) fn into_vec(mut self) -> Vec<T> {
        std::mem::take(&mut self.0)
    }
}

impl<T: 'static> Default for Pooled<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Drop for Pooled<T> {
    #[inline]
    fn drop(&mut self) {
        give(std::mem::take(&mut self.0));
    }
}

impl<T: 'static> std::ops::Deref for Pooled<T> {
    type Target = Vec<T>;
    #[inline]
    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T: 'static> std::ops::DerefMut for Pooled<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.0
    }
}

impl<T: 'static> FromIterator<T> for Pooled<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut p = Pooled::new();
        p.extend(iter);
        p
    }
}

impl<T: 'static + std::fmt::Debug> std::fmt::Debug for Pooled<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: 'static + Clone> Clone for Pooled<T> {
    fn clone(&self) -> Self {
        let mut p = Pooled::with_capacity(self.len());
        p.extend_from_slice(self);
        p
    }
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

    /// `Pooled` returns its buffer on drop; one grown past the byte cap is freed instead.
    #[test]
    fn pooled_returns_on_drop_and_drops_a_huge_buffer() {
        let mut p: Pooled<u8> = Pooled::new();
        p.extend_from_slice(&[1, 2, 3]);
        let ptr = p.as_ptr();
        drop(p);
        let q: Pooled<u8> = Pooled::new();
        assert_eq!(q.as_ptr(), ptr);
        drop(q);
        let big: Vec<u8> = vec![0; MAX_SPARE_BYTES + 1];
        let big_ptr = big.as_ptr();
        give(big);
        let r: Vec<u8> = take();
        assert_ne!(r.as_ptr(), big_ptr, "a huge buffer is not kept");
        give(r);
    }
}
