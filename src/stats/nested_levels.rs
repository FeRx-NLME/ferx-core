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
use rand::RngExt;

use crate::stats::likelihood::SUBJECT_SENTINEL_NLL;
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
        let [occasions, dose_occasions, pk_only_occasions, reset_occasions] = cell_ids(levels, s);
        s.occasions = occasions;
        s.dose_occasions = dose_occasions;
        s.pk_only_occasions = pk_only_occasions;
        s.reset_occasions = reset_occasions;
    }
}

/// The cell ids [`assign_level_cells`] gives `subject`'s records: observation, dose,
/// pk-only and reset labels, in that order.
fn cell_ids(levels: &NestedLevels, s: &Subject) -> [Vec<u32>; 4] {
    let mut cells: Vec<Vec<u64>> = Vec::new();
    let mut ids = [
        vec![0u32; s.obs_times.len()],
        vec![0u32; s.doses.len()],
        vec![0u32; s.pk_only_times.len()],
        vec![0u32; s.reset_times.len()],
    ];
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
        let slot = match kind {
            RecordKind::Observation => 0,
            RecordKind::Dose => 1,
            RecordKind::PkOnly => 2,
            RecordKind::Reset => 3,
        };
        ids[slot][i] = id;
    }
    ids
}

/// The lower factor a κ draw multiplies by; built only by [`KappaDraw::factor`].
pub(crate) struct KappaFactor<'a>(std::borrow::Cow<'a, DMatrix<f64>>);

/// How one subject's κ is drawn when a model is simulated: one κ vector per occasion
/// group, or — under nested levels — one block per group of each κ's own level
/// ([`NestedLayout::draw`]). Built once per subject; [`Self::draw`] returns the per-group
/// κ vectors in the order `predict_iov` indexes its `kappas` by. The one draw
/// `simulate()` and the NPDE reference share (#1082 PR4), so the two cannot disagree on
/// what a level is.
#[derive(Debug, Clone)]
pub(crate) enum KappaDraw {
    /// Today's IOV: `n_groups` independent `chol·z`, `z` of width `n_kappa`.
    OneLevel { n_groups: usize, n_kappa: usize },
    /// Nested levels.
    Nested(NestedLayout),
}

impl KappaDraw {
    /// The draw of `subject` under `model`. A nested model's subject must have its
    /// cells assigned ([`with_level_cells`]) and a label on every record
    /// (`check_level_labels`): one that has not is an `Err`, never a one-level draw,
    /// which would draw every outer-level κ once per cell (review r1 #1 on #1082 PR4).
    pub(crate) fn of(
        model: &CompiledModel,
        subject: &Subject,
    ) -> Result<Self, crate::diagnostics::EngineError> {
        match NestedLayout::of(model, subject) {
            Some(layout) => Ok(Self::Nested(layout)),
            None if model.has_nested_levels() => {
                Err(crate::diagnostics::EngineError::uncoded(format!(
                    "subject `{}` of a model with nested levels reached the kappa draw without \
                     its level cells; every simulate entry point assigns them after refusing a \
                     record with no label (`E_LEVEL_MISSING_LABEL`), so this is an engine bug",
                    subject.id
                )))
            }
            None => Ok(Self::OneLevel {
                n_groups: crate::stats::stacked_layout::group_count(subject),
                n_kappa: model.n_kappa,
            }),
        }
    }

    /// The factor every κ draw multiplies by: the declared Ω_IOV's
    /// [`crate::types::OmegaMatrix::sim_draw_factor`], so a zero κ draws exactly 0
    /// (#1605). The only way to build a [`KappaFactor`], so no draw site can hand
    /// [`Self::draw`] the regularised `chol` instead (review r2 on #1082 PR4: under nested
    /// levels the NPDE reference cannot observe that difference, so a type has to).
    pub(crate) fn factor<'a>(
        model: &CompiledModel,
        omega_iov: &'a crate::types::OmegaMatrix,
    ) -> KappaFactor<'a> {
        KappaFactor(omega_iov.sim_draw_factor(
            model.default_params.omega_iov.as_ref(),
            &model.default_params.kappa_fixed,
        ))
    }

    /// One draw of the per-group κ vectors from `factor` ([`Self::factor`]).
    pub(crate) fn draw<R: rand::Rng>(
        &self,
        factor: &KappaFactor<'_>,
        normal: rand_distr::Normal<f64>,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let omega_iov_chol: &DMatrix<f64> = &factor.0;
        match self {
            Self::Nested(layout) => layout.draw(omega_iov_chol, normal, rng),
            Self::OneLevel { n_groups, n_kappa } => (0..*n_groups)
                .map(|_| {
                    let z: Vec<f64> = (0..*n_kappa).map(|_| rng.sample(normal)).collect();
                    (omega_iov_chol * nalgebra::DVector::from_column_slice(&z))
                        .iter()
                        .copied()
                        .collect()
                })
                .collect(),
        }
    }
}

/// `E_PARAM_SHAPE` for an Ω_IOV that correlates two kappas at different levels of a
/// nested model (#1082 PR4, review r1 #2). A κ is drawn per group of its own level from
/// that level's block of the Cholesky factor, which is `chol(Ω_L)` only when nothing
/// couples two levels; a fitted Ω_IOV holds those entries at structural zeros, but a
/// caller-built one could set them, and the draw would then be silently wrong.
pub(crate) fn check_cross_level_omega_iov(
    model: &CompiledModel,
    omega_iov: &DMatrix<f64>,
) -> Result<(), crate::diagnostics::EngineError> {
    let Some(levels) = model.nested_levels() else {
        return Ok(());
    };
    let n = levels.kappa_level.len().min(omega_iov.nrows());
    for i in 0..n {
        for j in 0..i {
            let (li, lj) = (levels.kappa_level[i], levels.kappa_level[j]);
            if li != lj && omega_iov[(i, j)] != 0.0 {
                return Err(crate::api::param_shape(format!(
                    "omega_iov[{i}, {j}] = {} correlates kappa `{}` at level `{}` with `{}` at \
                     level `{}`, but kappas at two levels are independent (a `block_kappa` lives \
                     at one level). Set the entry to 0.",
                    omega_iov[(i, j)],
                    model.kappa_names[i],
                    levels.levels[li].name,
                    model.kappa_names[j],
                    levels.levels[lj].name
                )));
            }
        }
    }
    Ok(())
}

/// `population` with its level cells assigned ([`assign_level_cells`]): a copy for a
/// model with nested levels whose population does not carry them yet, the population
/// itself otherwise. For the entry points that take a population as read (`simulate()`,
/// #1082 PR4), where `fitted_population` never ran; a population an earlier step
/// prepared (`with_derived_occasions`, or each draw of `simulate_with_uncertainty`) is
/// borrowed rather than copied again.
pub(crate) fn with_level_cells<'a>(
    model: &CompiledModel,
    population: &'a Population,
) -> std::borrow::Cow<'a, Population> {
    let Some(levels) = model.nested_levels() else {
        return std::borrow::Cow::Borrowed(population);
    };
    if cells_ready(levels, population) {
        return std::borrow::Cow::Borrowed(population);
    }
    let mut owned = population.clone();
    assign_level_cells(model, &mut owned);
    std::borrow::Cow::Owned(owned)
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

/// Whether every subject already carries exactly the cell ids [`assign_level_cells`]
/// would write. Not merely non-zero ids: occasion labels read from a data column, or
/// another model's cells, are non-zero and group the records differently, and a
/// population carrying them must be reassigned rather than borrowed (review r2 on
/// #1082 PR4).
fn cells_ready(levels: &NestedLevels, population: &Population) -> bool {
    population.subjects.iter().all(|s| {
        let [o, d, p, r] = cell_ids(levels, s);
        s.occasions == o
            && s.dose_occasions == d
            && s.pk_only_occasions == p
            && s.reset_occasions == r
    })
}

/// Whether every record of `subject` carries an occasion label — what
/// [`assign_level_cells`] writes, for every record kind.
fn cells_assigned(subject: &Subject) -> bool {
    subject.occasions.len() == subject.obs_times.len()
        && subject.dose_occasions.len() == subject.doses.len()
        && subject.pk_only_occasions.len() == subject.pk_only_times.len()
        && subject.reset_occasions.len() == subject.reset_times.len()
}

/// A nested-level model with a subject that has no [`NestedLayout`]: the population was
/// not prepared for the fit (`fitted_population` never ran, as on the lower-level entry
/// points `optimize_population` / `run_inner_loop` handed a population as read), or a
/// record kept a missing label. The estimator must not fall back to one-level IOV for it,
/// which would integrate κ per record kind as read, or drop κ altogether; each site
/// refuses the subject with its own sentinel instead (review r1 #6 on #2028).
pub(crate) fn unprepared(model: &CompiledModel, subject: &Subject) -> bool {
    model.has_nested_levels() && NestedLayout::of(model, subject).is_none()
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
    /// Per cell: its label path, outermost level first (the bits of each label).
    cell_paths: Vec<Vec<u64>>,
    n_compact: usize,
}

impl NestedLayout {
    /// The layout of `subject` under `model`, or `None` for a model without nested
    /// levels (whose compact vector *is* the per-occasion stack) or for a subject that was
    /// never given its cells: occasion labels not parallel to its records (a population
    /// that skipped `fitted_population`), or a cell with no labelled record.
    /// [`unprepared`] tells the second case apart from the first.
    pub(crate) fn of(model: &CompiledModel, subject: &Subject) -> Option<Self> {
        let levels = model.nested_levels()?;
        crate::stats::stacked_layout::count_build();
        if !cells_assigned(subject) {
            return None;
        }
        let mut path_of: HashMap<u32, Vec<u64>> = HashMap::new();
        for (kind, i, path) in record_paths(levels, subject) {
            let occ = match kind {
                RecordKind::Observation => subject.occasions.get(i),
                RecordKind::Dose => subject.dose_occasions.get(i),
                RecordKind::PkOnly => subject.pk_only_occasions.get(i),
                RecordKind::Reset => subject.reset_occasions.get(i),
            };
            if let (Some(&occ), Some(path)) = (occ, path) {
                // One cell id, one label path: an id two paths share is not this
                // model's cells (labels from elsewhere), and grouping by it would merge
                // two groups — the subject is unprepared (review r2 on #1082 PR4).
                match path_of.get(&occ) {
                    Some(p) if *p != path => return None,
                    Some(_) => {}
                    None => {
                        path_of.insert(occ, path);
                    }
                }
            }
        }
        let cells = crate::stats::likelihood::iov_occasion_groups(subject);
        let cell_paths: Option<Vec<Vec<u64>>> = cells
            .iter()
            .map(|(occ, _)| path_of.get(occ).cloned())
            .collect();
        let cell_paths = cell_paths?;
        let (cell_groups, level_count) = level_groups(levels, &cell_paths);

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
            cell_paths,
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

    /// Number of κ each cell carries: `CompiledModel::n_kappa`, since `kappa_level` is
    /// parallel to `kappa_names`.
    pub(crate) fn n_kappa(&self) -> usize {
        self.kappa_slot.len()
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
                    return SUBJECT_SENTINEL_NLL;
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

    /// Every compact pair `(row, col)` at which `Ω_IOV[(i, j)]` appears in [`Self::prior`]:
    /// once per group of the level κ `i` and κ `j` both live at, in group order; none when
    /// they live at two levels (a structural zero of `Ω_IOV`).
    pub(crate) fn iov_entry_axes(&self, i: usize, j: usize) -> Vec<(usize, usize)> {
        let ((li, pi), (lj, pj)) = (self.kappa_slot[i], self.kappa_slot[j]);
        if li != lj {
            return Vec::new();
        }
        let w = self.level_width[li];
        (0..self.level_count[li])
            .map(|g| {
                let off = self.level_offset[li] + g * w;
                (off + pi, off + pj)
            })
            .collect()
    }

    /// Number of groups of level `l` in this subject.
    pub(crate) fn level_count(&self, l: usize) -> usize {
        self.level_count[l]
    }

    /// Cell `g`'s group index at level `l`.
    pub(crate) fn cell_group(&self, g: usize, l: usize) -> usize {
        self.cell_groups[g][l]
    }

    /// Per group of level `l`, in group order: the first cell of it (in cell order).
    /// Every group has one, since a group is made of the cells that name it.
    pub(crate) fn first_cells(&self, l: usize) -> Vec<usize> {
        let mut first = vec![usize::MAX; self.level_count[l]];
        for (g, groups) in self.cell_groups.iter().enumerate().rev() {
            first[groups[l]] = g;
        }
        first
    }

    /// Cell `g`'s labels from the outermost level down to level `l`, as values.
    pub(crate) fn labels(&self, g: usize, l: usize) -> Vec<f64> {
        self.cell_paths[g][..=l]
            .iter()
            .map(|&b| f64::from_bits(b))
            .collect()
    }

    /// One κ draw for simulation (#1082 PR4): per level, outermost first, one block
    /// `L_L·z` per group in group order, `z` standard normal of the level's width, and
    /// `L_L` the rows and columns of `omega_iov_chol` (Ω_IOV's lower Cholesky factor) at
    /// the level's κs. Returned per cell, as [`Self::expand`] lays it out, so a study κ
    /// is drawn once and read by every arm of that study.
    ///
    /// `L_L` is `chol(Ω_L)`: no `block_kappa` spans two levels, so Ω_IOV is
    /// block-diagonal by level up to a permutation, and elimination fills in nothing
    /// between two levels. With every κ at one level this is one `chol·z` per group,
    /// the one-level draw `simulate()` makes per occasion.
    pub(crate) fn draw<R: rand::Rng>(
        &self,
        omega_iov_chol: &DMatrix<f64>,
        normal: rand_distr::Normal<f64>,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let mut b = vec![0.0; self.n_compact];
        for l in 0..self.level_width.len() {
            let ks = self.level_kappas(l);
            let chol_l = omega_iov_chol.select_rows(&ks).select_columns(&ks);
            let w = self.level_width[l];
            for g in 0..self.level_count[l] {
                let z: Vec<f64> = (0..w).map(|_| rng.sample(normal)).collect();
                let kappa = &chol_l * nalgebra::DVector::from_column_slice(&z);
                let off = self.level_offset[l] + g * w;
                b[off..off + w].copy_from_slice(kappa.as_slice());
            }
        }
        self.expand(&b)
    }

    /// The κ blocks of the compact vector, as `(model κ indices, first compact index)`:
    /// one per group of each level, its κs in in-level order.
    pub(crate) fn kappa_blocks(&self) -> Vec<(Vec<usize>, usize)> {
        let mut out = Vec::new();
        for l in 0..self.level_width.len() {
            let ks = self.level_kappas(l);
            for g in 0..self.level_count[l] {
                out.push((ks.clone(), self.level_offset[l] + g * self.level_width[l]));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "nested_levels_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "nested_levels_test_helpers.rs"]
pub(crate) mod test_helpers;
