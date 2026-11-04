//! The sharing index of [`super::first_record_with_same_pk_inputs`], built once per EBE
//! search instead of once per θ/η evaluation (#2558).
//!
//! The index is a function of the model (its referenced covariates, its program and
//! whether that reads `TIME`), the subject's covariate snapshots and each record's κ
//! group. None of those moves with θ or η, yet both index sites rebuilt it on every
//! evaluation: the f64 materializer (`fill_per_record_pk_params`, once for the doses and
//! once for the other records) and the closed-form recorder (`sens::ode_adjoint`). On a
//! 100-subject time-varying-covariate fit that rebuild was 13–15% of the fit, and about
//! 97% of the builds ran inside an EBE search, where one subject is evaluated at many η
//! under one model.
//!
//! So an EBE search opens a [`SharingScope`] for its `(model, subject)`, and each site
//! builds its index the first time it is asked inside it and reuses it after that. A
//! request for any other model or subject, as from the `effective_model_for_eval` twin or
//! a cloned subject, misses and builds its own index as before, without storing it.
//! Outside a scope nothing is stored and every request builds.
//!
//! **Why an address is a sound key.** The guard borrows the model and the subject for as
//! long as it is open, so neither can be mutated (no field the index reads has interior
//! mutability) or freed, and no other live value can sit at either address. Rayon may run
//! a stolen task on a thread whose scope is open for another subject: that task's
//! addresses do not match, so it builds as it did before. A task that opens its own scope
//! saves the one it found and restores it on drop, unwinding included.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::types::{CompiledModel, Subject};

/// Which record list an index was built over: each site lists different records, so each
/// keeps its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShareSite {
    /// The f64 materializer's dose records.
    Dose,
    /// The f64 materializer's observation, PK-only and reset records, in one list.
    ObsPkOnlyReset,
    /// The recorder's dose, observation and PK-only records ([`super::record_pk_inputs`]).
    Recorder,
}

impl ShareSite {
    const COUNT: usize = 3;

    fn slot(self) -> usize {
        match self {
            ShareSite::Dose => 0,
            ShareSite::ObsPkOnlyReset => 1,
            ShareSite::Recorder => 2,
        }
    }
}

/// One site's index and the κ groups it was built with. The groups come from the
/// recorder's `lay` argument rather than from the model and subject alone, so a hit
/// checks them.
struct Slot {
    groups: Vec<Option<usize>>,
    first: Option<Rc<[usize]>>,
}

struct Scope {
    /// Which guard installed it, so a guard closing out of order is caught.
    id: u64,
    model: *const CompiledModel,
    subject: *const Subject,
    slots: [Option<Slot>; ShareSite::COUNT],
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
    static NEXT_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Indices built on this thread, for the tests that count reuse.
    static BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The number of indices built on this thread so far.
#[cfg(test)]
pub(crate) fn builds() -> usize {
    BUILDS.with(|b| b.get())
}

/// While this lives, the sharing indices of `(model, subject)` are built once and reused.
/// Bind it to a local (`let _share = SharingScope::enter(..)`) so scopes close in the
/// order they opened; `let _ = ..` closes it at once. Not `Send`: the state it restores
/// is this thread's.
#[must_use = "bind to a local; dropping closes the scope"]
pub(crate) struct SharingScope<'a> {
    /// The id of the scope this guard installed and the scope it replaced, restored on
    /// drop; `None` when this guard re-entered the scope already open for the same pair
    /// and changes nothing.
    installed: Option<(u64, Option<Scope>)>,
    _borrow: PhantomData<(&'a CompiledModel, &'a Subject, *const ())>,
}

impl<'a> SharingScope<'a> {
    #[must_use = "bind to a local; dropping closes the scope"]
    pub(crate) fn enter(model: &'a CompiledModel, subject: &'a Subject) -> Self {
        let (model, subject) = (model as *const _, subject as *const _);
        let installed = SCOPE.with(|s| {
            let mut s = s.borrow_mut();
            if s.as_ref()
                .is_some_and(|sc| sc.model == model && sc.subject == subject)
            {
                return None;
            }
            let id = NEXT_ID.with(|n| n.replace(n.get() + 1));
            let prev = s.replace(Scope {
                id,
                model,
                subject,
                slots: Default::default(),
            });
            Some((id, prev))
        });
        Self {
            installed,
            _borrow: PhantomData,
        }
    }
}

impl Drop for SharingScope<'_> {
    /// Restores the scope this guard replaced, when the open one is still this guard's.
    /// When it is not, a guard opened later is still open, and restoring would reinstall
    /// a scope whose borrows may have ended, so a later subject at a freed address would
    /// hit its index. That closes every scope instead, which costs only speed, and fails
    /// a debug build.
    fn drop(&mut self) {
        let Some((id, prev)) = self.installed.take() else {
            return;
        };
        let in_order = SCOPE.with(|s| {
            let mut s = s.borrow_mut();
            let in_order = s.as_ref().is_some_and(|sc| sc.id == id);
            *s = if in_order { prev } else { None };
            in_order
        });
        // Not while unwinding: the guards a panic drops may be out of order already.
        if !std::thread::panicking() {
            debug_assert!(in_order, "a SharingScope closed before one opened after it");
        }
    }
}

/// [`super::first_record_with_same_pk_inputs`] over `records`, `subject`'s records for
/// `site`: the index built earlier in the open scope when it is for this site, model,
/// subject and groups, otherwise a new one, kept for the scope when the scope is for this
/// model and subject.
pub(crate) fn first_record_with_same_pk_inputs<'r>(
    model: &CompiledModel,
    subject: &Subject,
    site: ShareSite,
    records: impl Iterator<Item = (Option<usize>, &'r HashMap<String, f64>)> + Clone,
) -> Option<Rc<[usize]>> {
    let (m, s) = (model as *const CompiledModel, subject as *const Subject);
    let lookup = SCOPE.with(|sc| {
        let sc = sc.borrow();
        let Some(sc) = sc.as_ref().filter(|sc| sc.model == m && sc.subject == s) else {
            return Lookup::Unscoped;
        };
        match &sc.slots[site.slot()] {
            Some(slot)
                if records
                    .clone()
                    .map(|(g, _)| g)
                    .eq(slot.groups.iter().copied()) =>
            {
                Lookup::Hit(slot.first.clone())
            }
            _ => Lookup::Miss,
        }
    });
    if let Lookup::Hit(first) = lookup {
        return first;
    }
    #[cfg(test)]
    BUILDS.with(|b| b.set(b.get() + 1));
    let first: Option<Rc<[usize]>> =
        super::first_record_with_same_pk_inputs(model, records.clone()).map(Rc::from);
    if let Lookup::Miss = lookup {
        SCOPE.with(|sc| {
            let mut sc = sc.borrow_mut();
            // The build opens no scope, so the one the lookup found is still open.
            let sc = sc.as_mut().expect("the scope a miss found is still open");
            sc.slots[site.slot()] = Some(Slot {
                groups: records.map(|(g, _)| g).collect(),
                first: first.clone(),
            });
        });
    }
    first
}

/// What the open scope holds for one request.
enum Lookup {
    /// No scope is open for this model and subject: build, keep nothing.
    Unscoped,
    /// The scope's index for this site and groups.
    Hit(Option<Rc<[usize]>>),
    /// The scope is for this pair but has no index for this site and groups: build, keep.
    Miss,
}
