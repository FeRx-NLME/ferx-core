//! The stacked random-effect layout of one subject under inter-occasion variability.
//!
//! A subject with IOV is integrated over one stacked vector
//! `b = [η_bsv, κ_group0, κ_group1, …]`: the between-subject effects first, then one κ
//! block per occasion group, in [`crate::stats::likelihood::iov_occasion_groups`] order.
//! Each record sees the **combined** vector `[η_bsv, κ_g]` of its own group `g`, of width
//! `n_eta + n_kappa` whatever the group count; only the stacked vector grows with it.
//!
//! [`StackedLayout`] is the single owner of that arithmetic — the group count, the
//! occasion → group map, every κ block's offset, the combined-vector slices, and the
//! block-diagonal prior `Ω ⊕ (I_K ⊗ Ω_iov)`. Before #1082 the offset `n_eta + g·n_kappa`
//! was written out at some thirty sites across the likelihood, the inner optimizer, the
//! six sensitivity walks and the IS / AGQ / VI / covariance preparations; #1082 generalises
//! the layout to nested levels, which changes the offset, so it must change in one place.
//!
//! Callers that only know a group **count** (a per-subject `K` precomputed elsewhere) use
//! [`stacked_width`]; callers with the subject build a [`StackedLayout`].

use std::collections::HashMap;
use std::ops::Range;

use nalgebra::DMatrix;

use crate::types::{CompiledModel, Subject};

/// The stacked-vector index of κ block `g`'s first entry. The one place the offset is
/// computed; everything else (including the legacy `pub` helpers in
/// `stats::likelihood`) goes through it.
#[inline]
pub(crate) fn kappa_offset(n_eta: usize, n_kappa: usize, g: usize) -> usize {
    n_eta + g * n_kappa
}

/// The stacked-vector index range of κ block `g`.
#[inline]
pub(crate) fn kappa_range(n_eta: usize, n_kappa: usize, g: usize) -> Range<usize> {
    kappa_offset(n_eta, n_kappa, g)..kappa_offset(n_eta, n_kappa, g + 1)
}

/// Width of the stacked vector for `n_groups` κ blocks.
#[inline]
pub(crate) fn stacked_width(n_eta: usize, n_kappa: usize, n_groups: usize) -> usize {
    kappa_offset(n_eta, n_kappa, n_groups)
}

/// Number of κ groups `subject` stacks, without building the occasion → group map.
pub(crate) fn group_count(subject: &Subject) -> usize {
    crate::stats::likelihood::iov_occasion_groups(subject).len()
}

/// Block-diagonal `a ⊕ (I_K ⊗ b)`: `a` on the leading block, then `n_groups` copies of
/// `b` at the κ offsets. Serves both the prior covariance (`Ω`, `Ω_iov`) and the prior
/// precision (`Ω⁻¹`, `Ω_iov⁻¹`), which have the same block structure.
pub(crate) fn block_diag_stack(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    n_groups: usize,
) -> DMatrix<f64> {
    let n_a = a.nrows();
    let n_b = b.nrows();
    let n_total = stacked_width(n_a, n_b, n_groups);
    let mut m = DMatrix::zeros(n_total, n_total);
    for i in 0..n_a {
        for j in 0..n_a {
            m[(i, j)] = a[(i, j)];
        }
    }
    for g in 0..n_groups {
        let off = kappa_offset(n_a, n_b, g);
        for i in 0..n_b {
            for j in 0..n_b {
                m[(off + i, off + j)] = b[(i, j)];
            }
        }
    }
    m
}

/// One subject's stacked `[η_bsv, κ…]` layout. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct StackedLayout {
    n_eta: usize,
    n_kappa: usize,
    /// `(occasion label, observation indices)` per group, in stacked order.
    groups: Vec<(u32, Vec<usize>)>,
    occ_to_group: HashMap<u32, usize>,
}

impl StackedLayout {
    /// The layout of `subject` for `n_eta` between-subject and `n_kappa` per-group effects.
    pub(crate) fn new(n_eta: usize, n_kappa: usize, subject: &Subject) -> Self {
        let groups = crate::stats::likelihood::iov_occasion_groups(subject);
        let occ_to_group = crate::stats::likelihood::iov_occ_to_k(&groups);
        Self {
            n_eta,
            n_kappa,
            groups,
            occ_to_group,
        }
    }

    /// The layout of `subject` under `model`'s `n_eta` / `n_kappa`.
    pub(crate) fn of(model: &CompiledModel, subject: &Subject) -> Self {
        Self::new(model.n_eta, model.n_kappa, subject)
    }

    /// Width of one κ block.
    pub(crate) fn n_kappa(&self) -> usize {
        self.n_kappa
    }

    /// Number of κ groups.
    pub(crate) fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Width of the stacked vector.
    pub(crate) fn n_stacked(&self) -> usize {
        stacked_width(self.n_eta, self.n_kappa, self.groups.len())
    }

    /// `(occasion label, observation indices)` per group, in stacked order.
    pub(crate) fn groups(&self) -> &[(u32, Vec<usize>)] {
        &self.groups
    }

    /// Occasion label → group index.
    pub(crate) fn occ_to_group(&self) -> &HashMap<u32, usize> {
        &self.occ_to_group
    }

    /// The group of occasion label `occ`, or `None` for a label no group carries.
    pub(crate) fn group_of(&self, occ: u32) -> Option<usize> {
        self.occ_to_group.get(&occ).copied()
    }

    /// Stacked index of group `g`'s first κ entry.
    pub(crate) fn kappa_offset(&self, g: usize) -> usize {
        kappa_offset(self.n_eta, self.n_kappa, g)
    }

    /// Stacked index range of group `g`'s κ block.
    pub(crate) fn kappa_range(&self, g: usize) -> Range<usize> {
        kappa_range(self.n_eta, self.n_kappa, g)
    }

    /// Group `g`'s κ block, sliced from `stacked`.
    pub(crate) fn kappa<'a>(&self, stacked: &'a [f64], g: usize) -> &'a [f64] {
        &stacked[self.kappa_range(g)]
    }

    /// The κ tail of `stacked` cut into `n_kappa`-wide blocks, tolerating a vector whose
    /// width does not match this subject (the last block is then short, or there is none).
    /// For the route decisions that must not panic on a malformed vector; a caller that
    /// has checked `stacked.len() == n_stacked()` uses [`Self::kappa`].
    pub(crate) fn kappa_chunks<'a>(&self, stacked: &'a [f64]) -> Vec<&'a [f64]> {
        if self.n_kappa == 0 {
            return Vec::new();
        }
        stacked[self.n_eta.min(stacked.len())..]
            .chunks(self.n_kappa)
            .collect()
    }

    /// The combined `[η_bsv, κ_g]` a record of group `g` sees.
    pub(crate) fn combined_effect(&self, stacked: &[f64], g: usize) -> Vec<f64> {
        crate::stats::likelihood::iov_combined_effect(stacked, self.n_eta, self.n_kappa, g)
    }

    /// The combined `[η_bsv, 0…0]` a record with no κ group sees (κ held at zero) — e.g. a
    /// pk-only row whose occasion has no group (#1876).
    pub(crate) fn combined_pk_only(&self, stacked: &[f64]) -> Vec<f64> {
        crate::stats::likelihood::iov_combined_pk_only(stacked, self.n_eta, self.n_kappa)
    }

    /// The stacked prior covariance `Ω ⊕ (I_K ⊗ Ω_iov)`.
    pub(crate) fn prior(&self, omega: &DMatrix<f64>, omega_iov: &DMatrix<f64>) -> DMatrix<f64> {
        block_diag_stack(omega, omega_iov, self.n_groups())
    }
}

#[cfg(test)]
#[path = "stacked_layout_tests.rs"]
mod tests;
