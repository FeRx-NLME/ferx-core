use super::*;
use crate::ode::solver::{effective_solver_options, OdeMethod, OdeSolverOptions};
use std::thread;

fn worker(pool: &rayon::ThreadPool) -> thread::ThreadId {
    pool.install(|| thread::current().id())
}

#[test]
fn completed_leases_reuse_workers_but_live_leases_are_exclusive() {
    let cache = PoolCache::new(4);
    let first = cache.acquire(1, Default::default()).unwrap();
    let id = worker(&first);
    let concurrent = cache.acquire(1, Default::default()).unwrap();
    assert_ne!(id, worker(&concurrent));
    drop(first);
    let reused = cache.acquire(1, Default::default()).unwrap();
    assert_eq!(id, worker(&reused));
}

#[test]
fn settings_and_width_are_part_of_the_key() {
    let cache = PoolCache::new(16);
    let base = OdeSolverOverride {
        reltol: Some(1e-9),
        ..Default::default()
    };
    let id = {
        let p = cache.acquire(1, base).unwrap();
        worker(&p)
    };
    for ov in [
        OdeSolverOverride::default(),
        OdeSolverOverride {
            reltol: Some(1e-10),
            ..base
        },
        OdeSolverOverride {
            abstol: Some(1e-12),
            ..base
        },
        OdeSolverOverride {
            max_steps: Some(13),
            ..base
        },
        OdeSolverOverride {
            method: Some(OdeMethod::Vern7),
            ..base
        },
        OdeSolverOverride {
            stiff_abort_after: Some(None),
            ..base
        },
        OdeSolverOverride {
            stiff_abort_after: Some(Some(0)),
            ..base
        },
        OdeSolverOverride {
            auto_switch: Some(false),
            ..base
        },
    ] {
        let p = cache.acquire(1, ov).unwrap();
        assert_ne!(id, worker(&p));
        let baked = OdeSolverOptions::default();
        p.install(|| {
            let actual = effective_solver_options(baked);
            let expected = ov.apply_to(baked);
            assert_eq!(actual.reltol, expected.reltol);
            assert_eq!(actual.abstol, expected.abstol);
            assert_eq!(actual.max_steps, expected.max_steps);
            assert_eq!(actual.method, expected.method);
            assert_eq!(actual.stiff_abort_after, expected.stiff_abort_after);
            assert_eq!(actual.auto_switch, expected.auto_switch);
        });
    }
    let wide = cache.acquire(2, base).unwrap();
    assert_eq!(wide.current_num_threads(), 2);
    assert_ne!(worker(&wide), id);
    assert_eq!(worker(&cache.acquire(1, base).unwrap()), id);
}

#[test]
fn idle_worker_budget_evicts_oldest_settings_and_rejects_oversized_pools() {
    let cache = PoolCache::new(2);
    for n in 1..=5 {
        let ov = OdeSolverOverride {
            max_steps: Some(n),
            ..Default::default()
        };
        drop(cache.acquire(1, ov).unwrap());
    }
    {
        let idle = cache.idle.lock().unwrap();
        assert_eq!(idle.len(), 2);
        assert_eq!(idle[0].ov.max_steps, Some(4));
        assert_eq!(idle[1].ov.max_steps, Some(5));
    }
    drop(cache.acquire(3, Default::default()).unwrap());
    assert_eq!(cache.idle.lock().unwrap().len(), 2);
    drop(cache.acquire(2, Default::default()).unwrap());
    let idle = cache.idle.lock().unwrap();
    assert_eq!(idle.len(), 1);
    assert_eq!(idle[0].pool.current_num_threads(), 2);
}

#[test]
fn panic_returns_a_usable_lease_and_nested_acquisition_never_waits() {
    let cache = PoolCache::new(4);
    let first = cache.acquire(1, Default::default()).unwrap();
    let id = worker(&first);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _lease = first;
        _lease.install(|| panic!("intentional worker panic"));
    }));
    assert!(panic.is_err());
    let reused = cache.acquire(1, Default::default()).unwrap();
    assert_eq!(worker(&reused), id);
    reused.install(|| {
        let nested = cache.acquire(1, Default::default()).unwrap();
        assert_ne!(worker(&nested), id);
    });
}

#[test]
fn poisoned_cache_lock_is_recovered() {
    let cache = PoolCache::new(1);
    let _ = std::panic::catch_unwind(|| {
        let _guard = cache.idle.lock().unwrap();
        panic!("poison the idle list");
    });
    drop(cache.acquire(1, Default::default()).unwrap());
    assert_eq!(
        cache
            .idle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len(),
        1
    );
}
