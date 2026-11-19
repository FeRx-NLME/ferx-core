//! Per-solve memo of a subject's predictions for the numerical θ/σ M-step.
//!
//! Predictions are a function of `(θ, η[, κ])` and the subject alone — never of σ — and the
//! M-step solve holds every η and κ fixed. So evaluations of one solve that see the same θ
//! see the same predictions: for a σ-only M-step (every θ owned by the closed-form steps, the
//! common case) that is *every* evaluation, and the solve is one prediction pass plus cheap
//! residual arithmetic instead of one pass per evaluation.
//!
//! What is kept is decided by the solve's structure, never by a size:
//!
//! * [`MemoMode::Fixed`] — every θ is pinned for the solve, so the solve has *one* θ and every
//!   evaluation reads and fills the same predictions.
//! * [`MemoMode::Track`] — θ is free and the solver (`mstep_optimizer = newton`) evaluates the
//!   value at a trial point and then the value, score and information at the *same* point once
//!   the trial is accepted: the memo follows the latest θ, so that second pass is served.
//! * Anything else (a derivative-free or L-BFGS solve with free θ never sees one θ twice) has
//!   no memo at all — [`PredMemo::for_solve`] returns `None` and nothing is stored.
//!
//! The θ is compared once per evaluation ([`PredMemo::begin`]), not once per slot, and the
//! predictions of one subject's samples live in one buffer behind one lock (the objective maps
//! over subjects, so a subject is owned by one worker at a time and the lock is never
//! contended). A hit hands the caller a borrow of the stored predictions — no copy.
//!
//! The memo returns exactly what the caller's own predictor returned the first time, so a
//! solve that uses it scores bit-for-bit what the same solve without it does.
//!
//! What it holds is `n_obs` numbers per *scored* `(subject, sample)` — a sample nothing asks
//! for (a zero importance weight) takes no space — which for SAEM (one η draw per subject,
//! `mstep_draws` at most) is the size of the data, and for IMP/IMPMAP (`imp_samples` per
//! subject, 1000 by default) is several times what the estimator itself holds. Which solves
//! pay that is the caller's decision: IMP/IMPMAP open a memo under the gradient solvers
//! (`mstep_optimizer = newton | lbfgs`; the default `auto` is `newton` wherever its scope
//! holds) only, never under BOBYQA.

use std::ops::{Deref, Range};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Which evaluations of a solve share predictions (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MemoMode {
    /// θ is pinned for the whole solve: one θ, one set of predictions.
    Fixed,
    /// θ is free: the memo follows the latest evaluation's θ.
    Track,
}

/// Whether two θ are the same *numbers*, bit for bit. `==` would call `-0.0` and `0.0` the same,
/// and a model that reads the sign of a zero (`1/θ`, `atan2`, `copysign`) predicts differently at
/// each: the memo must hand back exactly what the predictor returns at *this* θ.
fn same_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The solve's θ key and its generation (bumped when the key moves).
struct ThetaKey {
    theta: Option<Vec<f64>>,
    generation: u64,
}

#[cfg(test)]
thread_local! {
    /// Memos opened on this thread (test builds): what a test reads to see whether a solve
    /// kept predictions at all.
    pub(crate) static OPENED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// `(evaluations opened, evaluations declined)` of the memo most recently dropped on this
    /// thread (test builds): what a solve-level test reads to see how many of a solve's
    /// evaluations the memo took part in.
    pub(crate) static LAST_SOLVE: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
}

/// Marks a sample whose predictions are not stored for the current generation.
const NOT_STORED: u32 = u32::MAX;

/// One subject's predictions for each of its samples, in one buffer.
pub(crate) struct SubjectPreds {
    /// The [`ThetaKey`] generation `start` and `data` belong to.
    generation: u64,
    /// Predictions per sample, known once the first one is stored.
    n_obs: Option<usize>,
    /// Where each sample's predictions start in `data` for the current generation, or
    /// [`NOT_STORED`].
    start: Vec<u32>,
    /// The predictions of the samples scored so far, `n_obs` each, in the order they were first
    /// scored. A sample nothing scores (a zero importance weight) takes no space, and the
    /// buffer's size is the scored samples' — never the sample count's.
    data: Vec<f64>,
}

/// The predictions of one solve: one buffer per subject, and the solve's θ key.
pub(crate) struct PredMemo {
    mode: MemoMode,
    /// Set by [`PredMemo::continue_with_bobyqa`] on a [`MemoMode::Track`] memo: nothing is
    /// stored from then on.
    retired: AtomicBool,
    /// Evaluations opened and declined (test builds), reported on drop to [`LAST_SOLVE`].
    #[cfg(test)]
    seen: [std::sync::atomic::AtomicU32; 2],
    key: Mutex<ThetaKey>,
    subjects: Vec<Mutex<SubjectPreds>>,
}

impl PredMemo {
    /// The memo of a solve whose subject `i` has `samples[i]` samples (draws), or `None` where
    /// the solve's structure gives nothing to reuse: θ pinned throughout is
    /// [`MemoMode::Fixed`]; θ free under `newton` is [`MemoMode::Track`]; free θ under a
    /// solver that never revisits a point has no memo.
    pub(crate) fn for_solve(
        samples: impl IntoIterator<Item = usize>,
        theta_free: bool,
        newton: bool,
    ) -> Option<Self> {
        let mode = match (theta_free, newton) {
            (false, _) => MemoMode::Fixed,
            (true, true) => MemoMode::Track,
            (true, false) => return None,
        };
        #[cfg(test)]
        OPENED.with(|c| c.set(c.get() + 1));
        Some(Self {
            mode,
            retired: AtomicBool::new(false),
            #[cfg(test)]
            seen: Default::default(),
            key: Mutex::new(ThetaKey {
                theta: None,
                generation: 0,
            }),
            subjects: samples
                .into_iter()
                .map(|k| {
                    Mutex::new(SubjectPreds {
                        generation: 0,
                        n_obs: None,
                        start: vec![NOT_STORED; k],
                        data: Vec::new(),
                    })
                })
                .collect(),
        })
    }

    /// Samples of `subject` whose predictions are stored (test builds): what a short-circuit
    /// test reads to see which samples were scored.
    #[cfg(test)]
    pub(crate) fn scored(&self, subject: usize) -> usize {
        let g = self.subjects[subject]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        g.start.iter().filter(|&&s| s != NOT_STORED).count()
    }

    /// The solve hands over to a derivative-free optimizer (a Newton solve that stalled).
    ///
    /// * [`MemoMode::Track`]: that optimizer evaluates every point once, so the memo can never
    ///   hit; each evaluation would only pay a copy of every prediction and the clearing of
    ///   every subject's sample table. The memo is retired — its buffers freed, nothing
    ///   stored again.
    /// * [`MemoMode::Fixed`]: kept. The continuation sees the same pinned θ, bit for bit — the
    ///   optimizer's `scale_params` packing divides a pinned coordinate by a scale taken from
    ///   that very coordinate (`|x|`, or 1), so `(x / scale) · scale` returns `x` exactly
    ///   (pinned by `the_scaled_packing_returns_a_pinned_coordinate_bit_for_bit`) — and the
    ///   memo's remaining evaluations are served as before.
    pub(crate) fn continue_with_bobyqa(&self) {
        if self.mode == MemoMode::Track {
            self.retired.store(true, Ordering::Relaxed);
            for s in &self.subjects {
                let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
                g.start = Vec::new();
                g.data = Vec::new();
            }
        }
    }

    /// Open one evaluation at `theta`: the one place the θ key is read. `None` — evaluate
    /// without the memo — when this θ is not the one a [`MemoMode::Fixed`] solve holds, or
    /// when the memo is retired ([`PredMemo::continue_with_bobyqa`]).
    pub(crate) fn begin(&self, theta: &[f64]) -> Option<EvalMemo<'_>> {
        let opened = self.begin_inner(theta);
        #[cfg(test)]
        self.seen[usize::from(opened.is_none())].fetch_add(1, Ordering::Relaxed);
        opened
    }

    fn begin_inner(&self, theta: &[f64]) -> Option<EvalMemo<'_>> {
        if self.retired.load(Ordering::Relaxed) {
            return None;
        }
        let mut key = self.key.lock().unwrap_or_else(|e| e.into_inner());
        let same = key.theta.as_deref().is_some_and(|k| same_bits(k, theta));
        match (self.mode, same) {
            (_, true) => {}
            (MemoMode::Fixed, false) if key.theta.is_some() => return None,
            _ => {
                key.theta = Some(theta.to_vec());
                key.generation += 1;
            }
        }
        Some(EvalMemo {
            memo: self,
            generation: key.generation,
        })
    }
}

#[cfg(test)]
impl Drop for PredMemo {
    fn drop(&mut self) {
        let [opened, declined] = &self.seen;
        LAST_SOLVE.with(|c| {
            c.set((
                opened.load(Ordering::Relaxed),
                declined.load(Ordering::Relaxed),
            ))
        });
    }
}

/// A [`PredMemo`] opened for one evaluation at one θ.
#[derive(Clone, Copy)]
pub(crate) struct EvalMemo<'a> {
    memo: &'a PredMemo,
    generation: u64,
}

impl<'a> EvalMemo<'a> {
    /// The slot of `subject`'s `sample`-th sample.
    pub(crate) fn slot(&self, subject: usize, sample: usize) -> MemoSlot<'a> {
        MemoSlot {
            eval: *self,
            subject,
            sample,
        }
    }
}

/// One `(subject, sample)` of an [`EvalMemo`], addressed by the caller that knows which it is.
#[derive(Clone, Copy)]
pub(crate) struct MemoSlot<'a> {
    eval: EvalMemo<'a>,
    subject: usize,
    sample: usize,
}

/// Predictions: the caller's own, or a borrow of the memo's (the subject's lock is held for
/// the borrow's lifetime — a subject has one user at a time, so nothing waits on it).
pub(crate) enum Preds<'a> {
    Owned(Vec<f64>),
    Stored {
        guard: MutexGuard<'a, SubjectPreds>,
        range: Range<usize>,
    },
}

impl Deref for Preds<'_> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        match self {
            Preds::Owned(v) => v,
            Preds::Stored { guard, range } => &guard.data[range.clone()],
        }
    }
}

/// The predictions of `slot` at the θ its evaluation was opened at: the stored ones when the
/// slot holds them, else `compute()`'s, which are then stored. Without a slot, just
/// `compute()`.
pub(crate) fn memoized<'a>(
    slot: Option<MemoSlot<'a>>,
    compute: impl FnOnce() -> Vec<f64>,
) -> Preds<'a> {
    let Some(s) = slot else {
        return Preds::Owned(compute());
    };
    let mut g = s.eval.memo.subjects[s.subject]
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if g.generation != s.eval.generation {
        g.generation = s.eval.generation;
        g.start.fill(NOT_STORED);
        g.data.clear();
    }
    if g.start[s.sample] == NOT_STORED {
        let preds = compute();
        let n_obs = *g.n_obs.get_or_insert(preds.len());
        let at = g.data.len();
        if n_obs != preds.len() || at + n_obs >= NOT_STORED as usize {
            // Predictions of another length than the subject's first, or a buffer past what
            // a start can address: not storable.
            return Preds::Owned(preds);
        }
        // Grown by what is scored, amortised (`Vec`'s doubling): nothing is reserved for the
        // samples nothing asks for, so a subject whose samples mostly carry a zero importance
        // weight holds the scored samples' predictions (at most twice that while it grows) and
        // not `K · n_obs` numbers each.
        g.data.extend_from_slice(&preds);
        g.start[s.sample] = at as u32;
    }
    let n_obs = g.n_obs.unwrap_or(0);
    let at = g.start[s.sample] as usize;
    Preds::Stored {
        range: at..at + n_obs,
        guard: g,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(calls: &std::cell::Cell<u32>, v: Vec<f64>) -> Vec<f64> {
        calls.set(calls.get() + 1);
        v
    }

    /// Mode selection is structural: pinned θ is `Fixed` (whether or not the solver is Newton),
    /// free θ is `Track` only under Newton, and free θ under any other solver has no memo —
    /// nothing is allocated for a solve that can never hit.
    #[test]
    fn the_mode_follows_the_solve_structure_and_a_missless_solve_has_no_memo() {
        let mode = |free, newton| PredMemo::for_solve([2usize], free, newton).map(|m| m.mode);
        assert_eq!(mode(false, false), Some(MemoMode::Fixed));
        assert_eq!(mode(false, true), Some(MemoMode::Fixed));
        assert_eq!(mode(true, true), Some(MemoMode::Track));
        assert_eq!(mode(true, false), None);
    }

    /// `Fixed`: every evaluation at the solve's θ is served after the first fill; a different
    /// θ (which a pinned-θ solve never produces) is not served from, nor does it disturb, the
    /// stored predictions.
    #[test]
    fn a_fixed_memo_serves_its_theta_and_declines_another() {
        let memo = PredMemo::for_solve([2usize, 1], false, false).unwrap();
        let calls = std::cell::Cell::new(0u32);
        let get = |theta: &[f64], subject: usize, sample: usize| {
            let ev = memo.begin(theta)?;
            let p = memoized(Some(ev.slot(subject, sample)), || {
                count(&calls, vec![theta[0] * 2.0, theta[0] * 3.0])
            });
            Some(p.to_vec())
        };
        let a = get(&[1.0, 5.0], 0, 1).unwrap();
        assert_eq!(a, vec![2.0, 3.0]);
        assert_eq!(get(&[1.0, 5.0], 0, 1).unwrap(), a);
        assert_eq!(calls.get(), 1, "same θ: served");
        // A sibling sample and another subject are their own slots.
        assert_eq!(get(&[1.0, 5.0], 0, 0).unwrap(), a);
        assert_eq!(get(&[1.0, 5.0], 1, 0).unwrap(), a);
        assert_eq!(calls.get(), 3);
        // Another θ is declined (no memo for that evaluation) and leaves the store alone.
        assert!(get(&[2.0, 5.0], 0, 1).is_none());
        assert_eq!(get(&[1.0, 5.0], 0, 1).unwrap(), a);
        assert_eq!(calls.get(), 3);
    }

    /// `Track`: an unchanged θ is served (the value pass at a trial, then the fused pass at the
    /// accepted trial); a moved θ recomputes and replaces, so the old θ recomputes again.
    #[test]
    fn a_tracking_memo_serves_an_unchanged_theta_and_replaces_on_a_move() {
        let memo = PredMemo::for_solve([1usize], true, true).unwrap();
        let calls = std::cell::Cell::new(0u32);
        let get = |theta: &[f64]| {
            let ev = memo.begin(theta).expect("a tracking memo always opens");
            memoized(Some(ev.slot(0, 0)), || {
                count(&calls, vec![theta[0] * 2.0, theta[0] * 3.0])
            })
            .to_vec()
        };
        let a = get(&[1.0, 5.0]);
        assert_eq!(get(&[1.0, 5.0]), a);
        assert_eq!(calls.get(), 1, "same θ: served");
        assert_eq!(get(&[2.0, 5.0]), vec![4.0, 6.0]);
        assert_eq!(calls.get(), 2);
        assert_eq!(get(&[1.0, 5.0]), a);
        assert_eq!(calls.get(), 3, "the replaced θ recomputes");
    }

    /// No slot is the caller's own predictor, every time; predictions of another length than
    /// the subject's first are returned but never stored over its buffer.
    #[test]
    fn no_slot_computes_and_a_length_mismatch_is_not_stored() {
        let calls = std::cell::Cell::new(0u32);
        for _ in 0..2 {
            assert_eq!(memoized(None, || count(&calls, vec![7.0])).to_vec(), [7.0]);
        }
        assert_eq!(calls.get(), 2);
        let memo = PredMemo::for_solve([2usize], false, false).unwrap();
        let ev = memo.begin(&[1.0]).unwrap();
        let first = memoized(Some(ev.slot(0, 0)), || vec![1.0, 2.0]).to_vec();
        assert_eq!(first, [1.0, 2.0]);
        let odd = memoized(Some(ev.slot(0, 1)), || vec![9.0]);
        assert!(matches!(odd, Preds::Owned(_)));
        assert_eq!(odd.to_vec(), [9.0]);
        drop(odd);
        // The first sample is intact and sample 1 was not marked filled.
        let kept = memoized(Some(ev.slot(0, 0)), || unreachable!());
        assert_eq!(kept.to_vec(), [1.0, 2.0]);
        drop(kept);
        let fresh = memoized(Some(ev.slot(0, 1)), || vec![3.0, 4.0]);
        assert_eq!(fresh.to_vec(), [3.0, 4.0]);
    }

    /// What the memo holds is the scored samples', not the sample count's: a subject with 1000
    /// samples of which two are scored (the rest carry a zero weight and nothing asks for
    /// them) holds two samples of predictions, and a θ move empties it again. Each sample is
    /// served from its own place whatever order they were first scored in.
    #[test]
    fn only_the_scored_samples_take_space() {
        let memo = PredMemo::for_solve([1000usize], true, true).unwrap();
        let held = || memo.subjects[0].lock().unwrap().data.len();
        let ev = memo.begin(&[1.0]).unwrap();
        assert_eq!(held(), 0, "nothing is allocated before a sample is scored");
        memoized(Some(ev.slot(0, 700)), || vec![7.0, 7.5, 8.0]);
        memoized(Some(ev.slot(0, 3)), || vec![3.0, 3.5, 4.0]);
        assert_eq!(held(), 2 * 3);
        let capacity = memo.subjects[0].lock().unwrap().data.capacity();
        assert!(
            capacity < 1000 * 3 / 10,
            "no room is reserved for the 998 samples nobody scored: {capacity}"
        );
        assert_eq!(
            memoized(Some(ev.slot(0, 700)), || unreachable!()).to_vec(),
            [7.0, 7.5, 8.0]
        );
        assert_eq!(
            memoized(Some(ev.slot(0, 3)), || unreachable!()).to_vec(),
            [3.0, 3.5, 4.0]
        );
        assert_eq!(held(), 2 * 3, "a hit stores nothing");
        // A moved θ starts the buffer over: one sample scored, one sample held.
        let moved = memo.begin(&[2.0]).unwrap();
        memoized(Some(moved.slot(0, 5)), || vec![5.0, 5.5, 6.0]);
        assert_eq!(held(), 3);
    }

    /// The buffer follows the scored samples as they arrive — amortised growth, the footprint
    /// within twice what is held — and never the sample count: 100 of 10 000 samples scored in
    /// one subject, with a reallocation count logarithmic in what is held. A reserve of
    /// `K · n_obs` at the first store (10 000 · 20 numbers per subject) fails the bound at once.
    #[test]
    fn the_buffer_grows_with_the_scored_samples_not_the_sample_count() {
        let (k, n_obs, scored) = (10_000usize, 20usize, 100usize);
        let memo = PredMemo::for_solve([k], true, true).unwrap();
        let ev = memo.begin(&[1.0]).unwrap();
        let (mut reallocations, mut last) = (0u32, 0usize);
        for (i, sample) in (0..scored).map(|i| i * 97 % k).enumerate() {
            memoized(Some(ev.slot(0, sample)), || vec![i as f64; n_obs]);
            let g = memo.subjects[0].lock().unwrap();
            assert!(
                g.data.capacity() <= 2 * g.data.len().max(n_obs),
                "capacity {} holds {} numbers",
                g.data.capacity(),
                g.data.len()
            );
            if g.data.capacity() != last {
                last = g.data.capacity();
                reallocations += 1;
            }
        }
        assert_eq!(memo.scored(0), scored);
        assert!(
            reallocations <= 2 * (usize::BITS - (scored * n_obs).leading_zeros()),
            "{reallocations} reallocations for {} numbers",
            scored * n_obs
        );
    }

    /// The θ key is the numbers' bits: `-0.0` is not `0.0` (a model reading the sign of a zero
    /// predicts differently at each), in both modes, and an identical θ still hits. `NaN`
    /// compares equal to itself bit for bit — it is the same key, not a never-hit one.
    #[test]
    fn a_negative_zero_is_not_the_theta_a_positive_zero_stored() {
        for (free, newton) in [(false, false), (true, true)] {
            let memo = PredMemo::for_solve([1usize], free, newton).unwrap();
            let calls = std::cell::Cell::new(0u32);
            let get = |theta: &[f64]| {
                let ev = memo.begin(theta)?;
                Some(memoized(Some(ev.slot(0, 0)), || count(&calls, vec![1.0 / theta[0]])).to_vec())
            };
            assert_eq!(get(&[0.0]).unwrap(), [f64::INFINITY]);
            assert_eq!(get(&[0.0]).unwrap(), [f64::INFINITY]);
            assert_eq!(calls.get(), 1, "the same bits are served");
            if free {
                assert_eq!(
                    get(&[-0.0]).unwrap(),
                    [f64::NEG_INFINITY],
                    "Track recomputes"
                );
                assert_eq!(calls.get(), 2);
            } else {
                assert!(
                    get(&[-0.0]).is_none(),
                    "Fixed declines another θ, -0.0 included"
                );
                assert_eq!(calls.get(), 1);
            }
            let nan = f64::NAN;
            let before = calls.get();
            let _ = get(&[nan]);
            let n_first = calls.get();
            let _ = get(&[nan]);
            assert_eq!(calls.get(), n_first, "a NaN θ is its own key");
            assert!(n_first <= before + 1);
        }
    }

    /// The hand-over to a derivative-free continuation. `Track`: retired — nothing served,
    /// nothing stored, the buffers freed (a continuation never revisits a θ, so a memo left on
    /// would copy every prediction into it for no hit). `Fixed`: untouched, still serving its
    /// θ.
    #[test]
    fn a_continuation_retires_a_tracking_memo_and_leaves_a_fixed_one() {
        let track = PredMemo::for_solve([3usize], true, true).unwrap();
        let ev = track.begin(&[1.0]).unwrap();
        memoized(Some(ev.slot(0, 1)), || vec![1.0, 2.0]);
        assert_eq!(track.scored(0), 1);
        track.continue_with_bobyqa();
        assert!(track.begin(&[1.0]).is_none(), "retired: never serves");
        assert!(track.begin(&[2.0]).is_none());
        {
            let g = track.subjects[0].lock().unwrap();
            assert_eq!((g.data.capacity(), g.start.capacity()), (0, 0), "freed");
        }

        let fixed = PredMemo::for_solve([1usize], false, true).unwrap();
        let calls = std::cell::Cell::new(0u32);
        let get = |theta: f64| {
            let ev = fixed.begin(&[theta])?;
            Some(memoized(Some(ev.slot(0, 0)), || count(&calls, vec![theta])).to_vec())
        };
        assert_eq!(get(1.5).unwrap(), [1.5]);
        fixed.continue_with_bobyqa();
        assert_eq!(get(1.5).unwrap(), [1.5]);
        assert_eq!(calls.get(), 1, "still served after the hand-over");
        assert!(get(2.5).is_none(), "and still pinned to its θ");
    }
}
