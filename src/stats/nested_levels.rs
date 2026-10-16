//! Nested variability levels below `ID` (#1082).
//!
//! A model with more than one `level` integrates, per subject, one random effect per
//! **group** of each level: a κ at `STUDY` is drawn once per study and shared by every
//! arm of it, a κ at `ARM` once per arm. The records see that through the occasion
//! machinery that already exists. When the population is prepared, every distinct key
//! path — (study label, arm label) — becomes one **cell**, and the cell's id is written
//! into the occasion labels ([`assign_level_cells`]). Every prediction path, which reads
//! one κ vector per occasion group, therefore runs unchanged on per-cell κ vectors.
//!
//! What changes is the vector the subject is integrated over. Cells that share a study
//! share that study's κ, so the per-cell vectors are an *expansion* of a shorter
//! **compact** vector `b = [η, level-0 blocks…, level-1 blocks…, …]`, where level `L`'s
//! group `g` starts at `offset_L + g·n_κ,L`. [`NestedLayout`] owns that map: compact →
//! cells ([`NestedLayout::expand`]), cells → compact ([`NestedLayout::compact`]), and
//! the compact prior `Ω ⊕_L (I_{G_L} ⊗ Ω_L)`. The inner optimizer searches the compact
//! vector, and the FOCE marginal linearizes over it. Nothing else needs it.
//!
//! **Nesting is by key.** A group is identified by its whole label path, so `ARM = 1` in
//! study 1 and `ARM = 1` in study 2 are two groups; no label needs to be unique on its
//! own.

use std::collections::HashMap;

use nalgebra::DMatrix;

use crate::types::{CompiledModel, NestedLevels, Population, Subject};

/// A record's level labels, outermost first, as the bits of each level column's value
/// in `cov`; `None` when any of them is missing (absent, or `NaN`).
pub(crate) fn label_path(levels: &NestedLevels, cov: &HashMap<String, f64>) -> Option<Vec<u64>> {
    levels
        .levels
        .iter()
        .map(|l| {
            cov.get(&l.column)
                .copied()
                .filter(|v| !v.is_nan())
                .map(f64::to_bits)
        })
        .collect()
}

/// Which kind of record an occasion label belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordKind {
    Observation,
    Dose,
    PkOnly,
    Reset,
}

/// Every record of `subject` that carries an occasion label, with its label path:
/// observations, doses, pk-only rows and resets, in that order.
pub(crate) fn record_paths(
    levels: &NestedLevels,
    subject: &Subject,
) -> Vec<(RecordKind, usize, Option<Vec<u64>>)> {
    let mut out = Vec::new();
    for j in 0..subject.obs_times.len() {
        out.push((
            RecordKind::Observation,
            j,
            label_path(levels, subject.obs_cov(j)),
        ));
    }
    for k in 0..subject.doses.len() {
        out.push((RecordKind::Dose, k, label_path(levels, subject.dose_cov(k))));
    }
    for m in 0..subject.pk_only_times.len() {
        out.push((
            RecordKind::PkOnly,
            m,
            label_path(levels, subject.pk_only_cov(m)),
        ));
    }
    for r in 0..subject.reset_times.len() {
        out.push((
            RecordKind::Reset,
            r,
            label_path(levels, subject.reset_cov(r)),
        ));
    }
    out
}

/// Give every record of every subject the id of its cell — its full label path,
/// numbered from 1 in [`record_paths`] order — in the occasion label fields, for a model
/// with nested levels; a no-op for any other model. A record with a missing label gets
/// cell 0: `fit()`'s data check refuses such a record first (`E_LEVEL_MISSING_LABEL`).
pub(crate) fn assign_level_cells(model: &CompiledModel, population: &mut Population) {
    let Some(levels) = model.nested_levels() else {
        return;
    };
    for s in &mut population.subjects {
        let mut cells: Vec<Vec<u64>> = Vec::new();
        let mut occasions = vec![0u32; s.obs_times.len()];
        let mut dose_occasions = vec![0u32; s.doses.len()];
        let mut pk_only_occasions = vec![0u32; s.pk_only_times.len()];
        let mut reset_occasions = vec![0u32; s.reset_times.len()];
        for (kind, i, path) in record_paths(levels, s) {
            let id = match path {
                None => 0,
                Some(p) => match cells.iter().position(|c| *c == p) {
                    Some(c) => c as u32 + 1,
                    None => {
                        cells.push(p);
                        cells.len() as u32
                    }
                },
            };
            match kind {
                RecordKind::Observation => occasions[i] = id,
                RecordKind::Dose => dose_occasions[i] = id,
                RecordKind::PkOnly => pk_only_occasions[i] = id,
                RecordKind::Reset => reset_occasions[i] = id,
            }
        }
        s.occasions = occasions;
        s.dose_occasions = dose_occasions;
        s.pk_only_occasions = pk_only_occasions;
        s.reset_occasions = reset_occasions;
    }
}

/// The observation rows of `subject` per group of level `l`, as `(label, rows)` in
/// first-row order. The label is the group's whole path, `STUDY 1 / ARM 2`, since under
/// nesting by key the value at `l` alone (`ARM 2`) names a group in every study. Rows with
/// a missing label are left out (the data check refuses them).
pub(crate) fn obs_groups_at(
    levels: &NestedLevels,
    subject: &Subject,
    l: usize,
) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(Vec<u64>, Vec<usize>)> = Vec::new();
    for j in 0..subject.obs_times.len() {
        let Some(path) = label_path(levels, subject.obs_cov(j)) else {
            continue;
        };
        let key = path[..=l].to_vec();
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, rows)) => rows.push(j),
            None => groups.push((key, vec![j])),
        }
    }
    groups
        .into_iter()
        .map(|(key, rows)| {
            let label = levels
                .levels
                .iter()
                .zip(&key)
                .map(|(level, &v)| format!("{} {}", level.name, f64::from_bits(v)))
                .collect::<Vec<_>>()
                .join(" / ");
            (label, rows)
        })
        .collect()
}

/// The groups each level has in `subject`, keyed by label path prefix, in the order
/// of their first cell: `groups[L]` maps the path `path[..=L]` to the group's index.
///
/// The key is the **prefix**, not the label at `L` alone — that is what makes `ARM = 1`
/// in two studies two groups.
fn level_groups(levels: &NestedLevels, cell_paths: &[Vec<u64>]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n_levels = levels.levels.len();
    let mut keys: Vec<HashMap<&[u64], usize>> = vec![HashMap::new(); n_levels];
    let mut cell_groups = Vec::with_capacity(cell_paths.len());
    for path in cell_paths {
        let mut row = Vec::with_capacity(n_levels);
        for (l, key) in keys.iter_mut().enumerate() {
            let n = key.len();
            row.push(*key.entry(&path[..=l]).or_insert(n));
        }
        cell_groups.push(row);
    }
    let counts = keys.iter().map(HashMap::len).collect();
    (cell_groups, counts)
}

/// One subject's compact random-effect layout under nested levels. See the module docs.
#[derive(Debug, Clone)]
pub(crate) struct NestedLayout {
    n_eta: usize,
    /// Per κ (model order): its level and its position among that level's κs.
    kappa_slot: Vec<(usize, usize)>,
    /// Per level: the width of one group's κ block, the group count, and the compact
    /// index of its first block.
    level_width: Vec<usize>,
    level_count: Vec<usize>,
    level_offset: Vec<usize>,
    /// Per cell, in `StackedLayout` group order: its group index at each level.
    cell_groups: Vec<Vec<usize>>,
    n_compact: usize,
}

impl NestedLayout {
    /// The layout of `subject` under `model`, or `None` for a model without nested
    /// levels (whose compact vector *is* the per-occasion stack) or a subject whose
    /// occasion labels were never given cells (a cell with no labelled record).
    pub(crate) fn of(model: &CompiledModel, subject: &Subject) -> Option<Self> {
        let levels = model.nested_levels()?;
        let mut path_of: HashMap<u32, Vec<u64>> = HashMap::new();
        for (kind, i, path) in record_paths(levels, subject) {
            let occ = match kind {
                RecordKind::Observation => subject.occasions.get(i),
                RecordKind::Dose => subject.dose_occasions.get(i),
                RecordKind::PkOnly => subject.pk_only_occasions.get(i),
                RecordKind::Reset => subject.reset_occasions.get(i),
            };
            if let (Some(&occ), Some(path)) = (occ, path) {
                path_of.entry(occ).or_insert(path);
            }
        }
        let cells = crate::stats::likelihood::iov_occasion_groups(subject);
        let cell_paths: Option<Vec<Vec<u64>>> = cells
            .iter()
            .map(|(occ, _)| path_of.get(occ).cloned())
            .collect();
        let (cell_groups, level_count) = level_groups(levels, &cell_paths?);

        let n_levels = levels.levels.len();
        let mut level_width = vec![0usize; n_levels];
        let kappa_slot: Vec<(usize, usize)> = levels
            .kappa_level
            .iter()
            .map(|&l| {
                level_width[l] += 1;
                (l, level_width[l] - 1)
            })
            .collect();
        let mut level_offset = Vec::with_capacity(n_levels);
        let mut next = model.n_eta;
        for l in 0..n_levels {
            level_offset.push(next);
            next += level_count[l] * level_width[l];
        }
        Some(Self {
            n_eta: model.n_eta,
            kappa_slot,
            level_width,
            level_count,
            level_offset,
            cell_groups,
            n_compact: next,
        })
    }

    /// Width of the compact vector `[η, κ blocks…]`.
    pub(crate) fn n_compact(&self) -> usize {
        self.n_compact
    }

    /// Number of cells (the per-occasion groups the records are labelled with).
    pub(crate) fn n_cells(&self) -> usize {
        self.cell_groups.len()
    }

    /// The compact index cell `g`'s κ `k` reads.
    pub(crate) fn compact_index(&self, g: usize, k: usize) -> usize {
        let (l, pos) = self.kappa_slot[k];
        self.level_offset[l] + self.cell_groups[g][l] * self.level_width[l] + pos
    }

    /// The per-cell κ vectors of the compact vector `b`.
    pub(crate) fn expand(&self, b: &[f64]) -> Vec<Vec<f64>> {
        (0..self.n_cells())
            .map(|g| {
                (0..self.kappa_slot.len())
                    .map(|k| b[self.compact_index(g, k)])
                    .collect()
            })
            .collect()
    }

    /// The compact vector of `eta` and the per-cell κ vectors `kappas`, which must be an
    /// expansion (cells of one group agree on that group's κ). `None` when `kappas` does
    /// not hold one full-width vector per cell.
    pub(crate) fn compact<K: AsRef<[f64]>>(&self, eta: &[f64], kappas: &[K]) -> Option<Vec<f64>> {
        if kappas.len() != self.n_cells()
            || kappas
                .iter()
                .any(|k| k.as_ref().len() != self.kappa_slot.len())
        {
            return None;
        }
        let mut b = vec![0.0; self.n_compact];
        b[..self.n_eta].copy_from_slice(&eta[..self.n_eta]);
        for (g, kap) in kappas.iter().enumerate() {
            for (k, &v) in kap.as_ref().iter().enumerate() {
                b[self.compact_index(g, k)] = v;
            }
        }
        Some(b)
    }

    /// The model-order κ indices of level `l`, in their in-level position order.
    fn level_kappas(&self, l: usize) -> Vec<usize> {
        let mut ks: Vec<(usize, usize)> = self
            .kappa_slot
            .iter()
            .enumerate()
            .filter(|(_, &(lk, _))| lk == l)
            .map(|(k, &(_, pos))| (pos, k))
            .collect();
        ks.sort_unstable();
        ks.into_iter().map(|(_, k)| k).collect()
    }

    /// The compact prior covariance `Ω ⊕_L (I_{G_L} ⊗ Ω_L)`, with `Ω_L` the rows and
    /// columns of `omega_iov` at level `L`'s κs.
    pub(crate) fn prior(&self, omega: &DMatrix<f64>, omega_iov: &DMatrix<f64>) -> DMatrix<f64> {
        let mut m = DMatrix::zeros(self.n_compact, self.n_compact);
        m.view_mut((0, 0), (self.n_eta, self.n_eta))
            .copy_from(&omega.view((0, 0), (self.n_eta, self.n_eta)));
        for l in 0..self.level_width.len() {
            let ks = self.level_kappas(l);
            let w = self.level_width[l];
            for g in 0..self.level_count[l] {
                let off = self.level_offset[l] + g * w;
                for (a, &ka) in ks.iter().enumerate() {
                    for (c, &kc) in ks.iter().enumerate() {
                        m[(off + a, off + c)] = omega_iov[(ka, kc)];
                    }
                }
            }
        }
        m
    }

    /// The κ part of the prior's `−2 log` density at the compact vector `b`, without the
    /// `2π` constant: `Σ_L Σ_g κ_gᵀ Ω_L⁻¹ κ_g + G_L·log|Ω_L|`.
    ///
    /// `omega_iov_inv` and `omega_iov_chol` are `Ω_IOV`'s inverse and lower Cholesky
    /// factor. Because no `block_kappa` spans two levels, `Ω_IOV` is block-diagonal by
    /// level up to a permutation, so its inverse restricted to a level is `Ω_L⁻¹` and its
    /// Cholesky diagonal on a level is that of `Ω_L` (elimination never fills in between
    /// unconnected indices). With one level this is today's `Σ_k κ_kᵀΩ⁻¹κ_k + K·log|Ω|`.
    pub(crate) fn kappa_prior(
        &self,
        b: &[f64],
        omega_iov_inv: &DMatrix<f64>,
        omega_iov_chol: &DMatrix<f64>,
    ) -> f64 {
        let mut total = 0.0;
        for l in 0..self.level_width.len() {
            let ks = self.level_kappas(l);
            let w = self.level_width[l];
            let mut log_det = 0.0;
            for &k in &ks {
                let d = omega_iov_chol[(k, k)];
                if d <= 0.0 {
                    return 1e20;
                }
                log_det += 2.0 * d.ln();
            }
            for g in 0..self.level_count[l] {
                let off = self.level_offset[l] + g * w;
                let mut quad = 0.0;
                for (a, &ka) in ks.iter().enumerate() {
                    for (c, &kc) in ks.iter().enumerate() {
                        quad += b[off + a] * omega_iov_inv[(ka, kc)] * b[off + c];
                    }
                }
                total += quad + log_det;
            }
        }
        total
    }

    /// The compact vector's κ part, `b[n_eta..]`, cut into the groups of each level, as
    /// `(level, group, block)`, for the callers that walk the blocks (the runaway guard).
    pub(crate) fn blocks<'a>(&self, b: &'a [f64]) -> Vec<(usize, &'a [f64])> {
        let mut out = Vec::new();
        for l in 0..self.level_width.len() {
            let w = self.level_width[l];
            for g in 0..self.level_count[l] {
                let off = self.level_offset[l] + g * w;
                out.push((l, &b[off..off + w]));
            }
        }
        out
    }

    /// The model-order κ indices of level `l`, for a caller indexing `Ω_IOV` by level.
    pub(crate) fn kappas_at(&self, l: usize) -> Vec<usize> {
        self.level_kappas(l)
    }
}

#[cfg(test)]
#[path = "nested_levels_tests.rs"]
mod tests;
