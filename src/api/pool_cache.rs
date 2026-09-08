//! Persistent shared pools and exclusive leases for fit worker reuse.
use super::fit_thread_pool_builder;
use crate::ode::solver::OdeSolverOverride;
use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

struct CachedPool {
    pool: rayon::ThreadPool,
    ov: OdeSolverOverride,
}

struct SharedCachedPool {
    pool: Arc<rayon::ThreadPool>,
    ov: OdeSolverOverride,
}

fn build_pool(threads: usize, ov: OdeSolverOverride) -> Result<rayon::ThreadPool, String> {
    fit_thread_pool_builder()
        .num_threads(threads)
        .start_handler(move |_| {
            if !ov.is_empty() {
                crate::ode::solver::install_worker_ode_override(ov);
            }
        })
        .build()
        .map_err(|e| format!("failed to build rayon pool with {threads} threads: {e}"))
}

/// Only idle workers count towards this cache's ordinary limit. Active callers
/// retain their requested widths; acquisition never waits for another fit to
/// finish. One most-recent oversized pool may remain idle so explicit wide
/// configurations still benefit from reuse.
pub(super) struct PoolCache {
    idle: Mutex<VecDeque<CachedPool>>,
    max_idle_workers: usize,
}

impl PoolCache {
    pub(super) fn new(max_idle_workers: usize) -> Self {
        Self {
            idle: Mutex::new(VecDeque::new()),
            max_idle_workers,
        }
    }

    pub(super) fn acquire(
        &self,
        threads: usize,
        ov: OdeSolverOverride,
    ) -> Result<FitPoolLease<'_>, String> {
        if threads == 0 {
            return Err("thread count must be positive".to_string());
        }
        ov.validate()?;
        let cached = {
            let mut idle = self
                .idle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Search newest first to keep recently used workers hot.
            idle.iter()
                .rposition(|p| p.pool.current_num_threads() == threads && p.ov.same_pool_key(&ov))
                .and_then(|i| idle.remove(i))
        };
        let entry = match cached {
            Some(p) => p,
            None => CachedPool {
                // Build outside the lock: unrelated fits can acquire/return
                // their leases while OS threads are being started.
                pool: build_pool(threads, ov)?,
                ov,
            },
        };
        Ok(FitPoolLease {
            entry: Some(entry),
            cache: self,
        })
    }

    fn release(&self, entry: CachedPool) {
        let mut evicted = Vec::new();
        {
            let mut idle = self
                .idle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            idle.push_back(entry);
            let mut workers: usize = idle.iter().map(|p| p.pool.current_num_threads()).sum();
            // Keep at least the newest pool even when it alone exceeds the
            // ordinary idle-worker budget. This preserves reuse for an
            // explicitly requested wide pool while retaining only one such
            // reservation and evicting it as soon as a newer key is used.
            while workers > self.max_idle_workers && idle.len() > 1 {
                let p = idle.pop_front().expect("workers counted an idle pool");
                workers -= p.pool.current_num_threads();
                evicted.push(p);
            }
        }
        // Dropping a pool signals worker shutdown; do so without the cache lock.
        drop(evicted);
    }
}

/// Shared pools for unpinned calls. Such calls did not request an independent
/// worker budget, so callers with identical ODE settings may use one pool
/// concurrently instead of multiplying the process worker count.
pub(super) struct SharedPoolCache {
    pools: Mutex<VecDeque<SharedCachedPool>>,
    max_cached_workers: usize,
}

impl SharedPoolCache {
    pub(super) fn new(max_cached_workers: usize) -> Self {
        Self {
            pools: Mutex::new(VecDeque::new()),
            max_cached_workers,
        }
    }

    pub(super) fn acquire(
        &self,
        threads: usize,
        ov: OdeSolverOverride,
    ) -> Result<Arc<rayon::ThreadPool>, String> {
        if threads == 0 {
            return Err("thread count must be positive".to_string());
        }
        ov.validate()?;
        {
            let mut pools = self
                .pools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(index) = pools
                .iter()
                .rposition(|p| p.pool.current_num_threads() == threads && p.ov.same_pool_key(&ov))
            {
                let entry = pools.remove(index).expect("matching shared pool");
                let pool = Arc::clone(&entry.pool);
                pools.push_back(entry);
                return Ok(pool);
            }
        }

        let built = Arc::new(build_pool(threads, ov)?);
        let mut evicted = Vec::new();
        {
            let mut pools = self
                .pools
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Another caller may have built the same key while this build ran.
            // Prefer the pool it published so concurrent unpinned calls still
            // converge on one shared budget.
            if let Some(index) = pools
                .iter()
                .rposition(|p| p.pool.current_num_threads() == threads && p.ov.same_pool_key(&ov))
            {
                let entry = pools.remove(index).expect("matching shared pool");
                let pool = Arc::clone(&entry.pool);
                pools.push_back(entry);
                return Ok(pool);
            }
            pools.push_back(SharedCachedPool {
                pool: Arc::clone(&built),
                ov,
            });
            let mut workers: usize = pools.iter().map(|p| p.pool.current_num_threads()).sum();
            while workers > self.max_cached_workers && pools.len() > 1 {
                let old = pools.pop_front().expect("workers counted a shared pool");
                workers -= old.pool.current_num_threads();
                evicted.push(old);
            }
        }
        drop(evicted);
        Ok(built)
    }
}

/// Owns one pool until the caller's scoped work has completed. Returning
/// it on unwind is safe too: Rayon propagates a panic after its scoped work joins.
pub(crate) struct FitPoolLease<'a> {
    entry: Option<CachedPool>,
    cache: &'a PoolCache,
}

impl Deref for FitPoolLease<'_> {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &Self::Target {
        &self.entry.as_ref().expect("live pool lease").pool
    }
}

impl Drop for FitPoolLease<'_> {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            self.cache.release(entry);
        }
    }
}

#[cfg(test)]
#[path = "tests/pool_cache_tests.rs"]
mod tests;
