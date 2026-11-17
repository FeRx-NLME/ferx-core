//! Per-solve memo of a subject's predictions for the numerical θ/σ M-step.
//!
//! Predictions are a function of `(θ, η[, κ])` and the subject alone — never of σ — and the
//! M-step solve holds every η and κ fixed. So every evaluation of one solve that sees the same
//! θ sees the same predictions: for a σ-only M-step (every θ owned by the closed-form steps,
//! the common case) that is *every* evaluation, and the solve is one prediction pass plus
//! cheap residual arithmetic instead of one pass per evaluation. A trial with a different θ
//! (a free θ moving) recomputes and replaces the entry.
//!
//! The memo returns exactly what the caller's own predictor returned the first time, so a
//! solve that uses it scores bit-for-bit what the same solve without it does. One slot per
//! `(draw, subject)` is owned by one worker at a time (the objective maps over subjects), so
//! the per-slot lock is never contended.

use std::sync::Mutex;

/// One `(θ, predictions)` entry per `(draw, subject)` of a solve.
pub(crate) struct PredMemo {
    slots: Vec<Mutex<(Vec<f64>, Vec<f64>)>>,
}

impl PredMemo {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            slots: (0..n)
                .map(|_| Mutex::new((Vec::new(), Vec::new())))
                .collect(),
        }
    }
}

/// A [`PredMemo`] entry, addressed once by the caller that knows which draw and subject it is.
#[derive(Clone, Copy)]
pub(crate) struct MemoSlot<'a> {
    pub(crate) memo: &'a PredMemo,
    pub(crate) idx: usize,
}

/// The predictions at `theta`: the memoised ones when the slot holds them for this exact `θ`,
/// else `compute()`'s, which then replace the slot's. Without a slot, just `compute()`.
pub(crate) fn memoized(
    slot: Option<MemoSlot<'_>>,
    theta: &[f64],
    compute: impl FnOnce() -> Vec<f64>,
) -> Vec<f64> {
    let Some(s) = slot else {
        return compute();
    };
    let mut g = s.memo.slots[s.idx]
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if !g.1.is_empty() && g.0 == theta {
        return g.1.clone();
    }
    let preds = compute();
    g.0.clear();
    g.0.extend_from_slice(theta);
    g.1.clear();
    g.1.extend_from_slice(&preds);
    preds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_serves_the_same_theta_from_memory_and_recomputes_on_another() {
        let memo = PredMemo::new(2);
        let slot = Some(MemoSlot {
            memo: &memo,
            idx: 1,
        });
        let calls = std::cell::Cell::new(0u32);
        let get = |theta: &[f64]| {
            memoized(slot, theta, || {
                calls.set(calls.get() + 1);
                vec![theta[0] * 2.0, theta[0] * 3.0]
            })
        };
        let a = get(&[1.0, 5.0]);
        assert_eq!(get(&[1.0, 5.0]), a);
        assert_eq!(calls.get(), 1, "same θ: served from the memo");
        // A different θ (a free coordinate moved) recomputes — the straddle of the gate —
        // and replaces the entry, so the old θ recomputes again.
        assert_eq!(get(&[2.0, 5.0]), vec![4.0, 6.0]);
        assert_eq!(calls.get(), 2);
        assert_eq!(get(&[1.0, 5.0]), a);
        assert_eq!(calls.get(), 3);
        // The other slot is independent, and no slot is the caller's own predictor.
        let other = Some(MemoSlot {
            memo: &memo,
            idx: 0,
        });
        assert_eq!(memoized(other, &[1.0, 5.0], || vec![9.0]), vec![9.0]);
        assert_eq!(memoized(None, &[1.0], || vec![7.0]), vec![7.0]);
        assert_eq!(memoized(None, &[1.0], || vec![8.0]), vec![8.0]);
    }
}
