//! Bounded admission for the many-fit tools (#1329): run `n` jobs over exactly
//! `width` concurrent lanes, and no more.
//!
//! # Why a parallel iterator is not enough
//!
//! Every tool in this crate has the same shape — `N` independent fits over a
//! thread budget of `width` — and the obvious spelling is an outer pool of
//! `width` workers running a `par_iter` over the `N` jobs. That spelling does
//! **not** bound how many fits are live at once, and the reason is documented
//! Rayon behaviour rather than a bug: each job calls [`ferx_core::fit`], which
//! `install`s its own inner pool, and a worker blocked on a nested `install`
//! **keeps stealing work from its own pool**.[^install] With `N ≫ width` there
//! are always pending jobs to steal, so every blocked worker picks up another
//! fit, whose `install` blocks it again, and so on. A synthetic probe of the
//! old shape peaked at 13 / 24 / 29 live jobs for requested widths 1 / 2 / 4.
//!
//! A blocking permit *inside* the parallel iterator is the wrong fix: the
//! worker that blocks on the permit is the same worker the run needs to make
//! progress, so the permit can occupy the pool it was meant to protect.
//!
//! # What this does instead
//!
//! Exactly `width` long-lived lane closures, each of which claims one job index
//! from a shared atomic counter, runs it to completion, records the result, and
//! only then claims another. Two properties follow:
//!
//! * **The bound is exact.** A lane runs one job at a time and there are
//!   `width` lanes, so at most `width` fits are ever live — and therefore at
//!   most `width` fit stacks, which is what the memory pressure is.
//! * **A blocked lane steals nothing.** The lanes are plain OS threads, not
//!   Rayon workers, so a lane blocked on its inner pool's `install` waits on a
//!   latch instead of looking for more work. Spawning the lanes as `width`
//!   tasks on a `width`-wide Rayon pool would *not* give this: the lane tasks
//!   are themselves stealable, so a lane blocked before its siblings had been
//!   picked up could run two of them nested.
//!
//! The lanes carry [`FIT_RAYON_STACK_SIZE`], not the 2 MiB platform default —
//! the same reason [`ferx_core::PoolPlan::install`] does. Wide ODE+IOV
//! analytic-gradient models overflow a stock worker stack, and the bootstrap's
//! hand-built `ThreadPoolBuilder` inherited that default.
//!
//! # What it deliberately does not change
//!
//! Jobs are claimed in index order and results come back **sorted by job
//! index**, which is what `par_iter().filter_map().collect()` returned, so no
//! caller's assembly, journalling order or seed mapping changes. A job that
//! returns `None` is dropped, as `filter_map` dropped it. A panic in a job is
//! resumed on the calling thread, so it still takes the run with it.
//!
//! [^install]: Rayon, [`ThreadPool::install`](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html#method.install).

use std::sync::atomic::{AtomicUsize, Ordering};

use ferx_core::FIT_RAYON_STACK_SIZE;

/// Run `job` for every index in `0..n_jobs` on at most `width` concurrent
/// lanes, and return the `Some` results sorted by job index.
///
/// `width` is clamped to `1..=n_jobs`: a wider budget than there is work spawns
/// idle lanes for nothing, and `0` would mean "let the runtime decide", which is
/// the ambiguity this function exists to remove. `n_jobs == 0` runs no lanes.
///
/// `job` is called exactly once per index, from one lane, and must therefore be
/// `Sync`. It is free to block — blocking a lane is the point.
///
/// # Errors
///
/// Only if a lane thread cannot be spawned (a resource limit), in which case the
/// queue is retired and the results of whatever had already run are discarded.
/// Jobs that fail report it through `T`, as they did under `filter_map`.
///
/// # Panics
///
/// Re-raises a job's panic on the calling thread, after the other lanes have
/// been joined.
pub(crate) fn run_in_lanes<T, F>(width: usize, n_jobs: usize, job: F) -> Result<Vec<T>, String>
where
    F: Fn(usize) -> Option<T> + Sync,
    T: Send,
{
    if n_jobs == 0 {
        return Ok(Vec::new());
    }
    let width = width.clamp(1, n_jobs);
    let next = AtomicUsize::new(0);

    // Each lane accumulates locally and is merged after the join, so the lanes
    // never contend on a shared sink and the merge order is the lanes' spawn
    // order rather than a completion order that would vary run to run.
    let mut claimed: Vec<(usize, T)> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(width);
        let mut spawn_error = None;
        for lane in 0..width {
            let lane_fn = || {
                let mut mine = Vec::new();
                loop {
                    // `Relaxed` is enough: the counter's only job is to hand
                    // each index to one lane, and the results are published by
                    // the join, which is itself the synchronisation edge.
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n_jobs {
                        return mine;
                    }
                    if let Some(result) = job(i) {
                        mine.push((i, result));
                    }
                }
            };
            match std::thread::Builder::new()
                .name(format!("ferx-lane-{lane}"))
                .stack_size(FIT_RAYON_STACK_SIZE)
                .spawn_scoped(scope, lane_fn)
            {
                Ok(handle) => handles.push(handle),
                // The lanes already spawned are joined by the scope either way,
                // but draining the queue first would run every remaining fit and
                // then discard it with the error. Retiring the counter stops them
                // at their next claim instead; whatever had already finished is
                // still journalled by the caller's job closure, so a resume picks
                // those up.
                Err(e) => {
                    next.store(n_jobs, Ordering::Relaxed);
                    spawn_error = Some(format!(
                        "could not spawn lane {lane} of {width} for the fit queue: {e}"
                    ));
                    break;
                }
            }
        }
        let mut all = Vec::new();
        for handle in handles {
            match handle.join() {
                Ok(mine) => all.extend(mine),
                // A job's panic used to cross the `par_iter` boundary and take
                // the run with it. Resuming it here keeps that. The lanes this
                // loop has not reached yet are *still* joined, by the scope's
                // own drop glue on the way out — `thread::scope` guarantees that
                // on an unwind as well as on a return, which is why the borrows
                // the lanes hold stay valid.
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        match spawn_error {
            Some(e) => Err(e),
            None => Ok(all),
        }
    })?;

    claimed.sort_by_key(|(i, _)| *i);
    Ok(claimed.into_iter().map(|(_, result)| result).collect())
}

#[cfg(test)]
#[path = "lanes_tests.rs"]
mod tests;
