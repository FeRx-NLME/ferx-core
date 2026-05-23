use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// A single dose event (bolus, infusion, or oral)
#[derive(Debug, Clone)]
pub struct DoseEvent {
    pub time: f64,
    pub amt: f64,
    pub cmt: usize,
    pub rate: f64,
    pub duration: f64,
    pub ss: bool,
    pub ii: f64,
}

impl DoseEvent {
    pub fn new(time: f64, amt: f64, cmt: usize, rate: f64, ss: bool, ii: f64) -> Self {
        let duration = if rate > 0.0 { amt / rate } else { 0.0 };
        Self {
            time,
            amt,
            cmt,
            rate,
            duration,
            ss,
            ii,
        }
    }

    pub fn is_infusion(&self) -> bool {
        self.rate > 0.0
    }
}

/// Fixed-layout PK parameters — replaces HashMap<String, f64> for AD compatibility.
///
/// Index convention:
///   0: CL      (clearance)
///   1: V       (volume, or V1 for 2-cmt)
///   2: Q/Q2    (intercompartmental clearance, central ↔ peripheral 1; 2-cmt and 3-cmt)
///   3: V2      (peripheral volume 1; 2-cmt and 3-cmt)
///   4: KA      (absorption rate constant, oral only)
///   5: F       (bioavailability, default 1.0)
///   6: Q3      (intercompartmental clearance, 3-cmt: central ↔ peripheral 2)
///   7: V3      (peripheral volume 2, 3-cmt only)
///   8: LAGTIME (dose/absorption lagtime, default 0.0; equivalent to NONMEM ALAG)
pub const MAX_PK_PARAMS: usize = 9;

pub const PK_IDX_CL: usize = 0;
pub const PK_IDX_V: usize = 1;
pub const PK_IDX_Q: usize = 2;
pub const PK_IDX_V2: usize = 3;
pub const PK_IDX_KA: usize = 4;
pub const PK_IDX_F: usize = 5;
pub const PK_IDX_Q3: usize = 6;
pub const PK_IDX_V3: usize = 7;
pub const PK_IDX_LAGTIME: usize = 8;

#[derive(Debug, Clone, Copy)]
pub struct PkParams {
    pub values: [f64; MAX_PK_PARAMS],
}

impl Default for PkParams {
    fn default() -> Self {
        let mut v = [0.0; MAX_PK_PARAMS];
        v[PK_IDX_F] = 1.0; // bioavailability defaults to 1
        Self { values: v }
    }
}

impl PkParams {
    pub fn cl(&self) -> f64 {
        self.values[PK_IDX_CL]
    }
    pub fn v(&self) -> f64 {
        self.values[PK_IDX_V]
    }
    pub fn q(&self) -> f64 {
        self.values[PK_IDX_Q]
    }
    pub fn v2(&self) -> f64 {
        self.values[PK_IDX_V2]
    }
    pub fn ka(&self) -> f64 {
        self.values[PK_IDX_KA]
    }
    pub fn f_bio(&self) -> f64 {
        self.values[PK_IDX_F]
    }
    pub fn q3(&self) -> f64 {
        self.values[PK_IDX_Q3]
    }
    pub fn v3(&self) -> f64 {
        self.values[PK_IDX_V3]
    }
    pub fn lagtime(&self) -> f64 {
        self.values[PK_IDX_LAGTIME]
    }

    /// Map a PK parameter name to its index in the fixed-size array.
    ///
    /// `"alag"` is accepted as an alias for `"lagtime"` for NONMEM familiarity.
    pub fn name_to_index(name: &str) -> Option<usize> {
        match name {
            "cl" => Some(PK_IDX_CL),
            "v" | "v1" => Some(PK_IDX_V),
            "q" | "q2" => Some(PK_IDX_Q),
            "v2" => Some(PK_IDX_V2),
            "ka" => Some(PK_IDX_KA),
            "f" => Some(PK_IDX_F),
            "q3" => Some(PK_IDX_Q3),
            "v3" => Some(PK_IDX_V3),
            "lagtime" | "alag" => Some(PK_IDX_LAGTIME),
            _ => None,
        }
    }

    /// Build from named HashMap (bridge for parser compatibility)
    pub fn from_hashmap(map: &HashMap<String, f64>) -> Self {
        let mut p = Self::default();
        if let Some(&v) = map.get("cl") {
            p.values[PK_IDX_CL] = v;
        }
        if let Some(&v) = map.get("v") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("v1") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("q") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("q2") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("v2") {
            p.values[PK_IDX_V2] = v;
        }
        if let Some(&v) = map.get("ka") {
            p.values[PK_IDX_KA] = v;
        }
        if let Some(&v) = map.get("f") {
            p.values[PK_IDX_F] = v;
        }
        if let Some(&v) = map.get("q3") {
            p.values[PK_IDX_Q3] = v;
        }
        if let Some(&v) = map.get("v3") {
            p.values[PK_IDX_V3] = v;
        }
        if let Some(&v) = map.get("lagtime").or_else(|| map.get("alag")) {
            p.values[PK_IDX_LAGTIME] = v;
        }
        p
    }
}

/// A single subject with dosing and observation data
#[derive(Debug, Clone)]
pub struct Subject {
    pub id: String,
    pub doses: Vec<DoseEvent>,
    pub obs_times: Vec<f64>,
    pub observations: Vec<f64>,
    pub obs_cmts: Vec<usize>,
    /// Subject-representative covariate values (first non-missing value per
    /// covariate). Used by the AD fast path and as a fallback when neither
    /// `dose_covariates` nor `obs_covariates` is populated.
    pub covariates: HashMap<String, f64>,
    /// Per-dose covariate snapshot (LOCF), parallel to `doses`. When the
    /// dataset has no time-varying covariates, this is empty and consumers
    /// fall back to `covariates`. NONMEM-equivalent: the value of `$PK`
    /// inputs at each dose record.
    pub dose_covariates: Vec<HashMap<String, f64>>,
    /// Per-observation covariate snapshot (LOCF), parallel to `obs_times`.
    /// Same fallback semantics as `dose_covariates`.
    pub obs_covariates: Vec<HashMap<String, f64>>,
    /// Times of EVID=2 "other event" rows (typically covariate-change
    /// markers). Only populated when the subject has time-varying
    /// covariates — for time-constant covariates these rows are no-ops
    /// (NONMEM-equivalent: $PK runs but with the same values).
    /// The event-driven propagators see them as a third event kind that
    /// does not mutate compartment amounts but does refresh the
    /// piecewise-constant rate matrix from the row's covariate values.
    pub pk_only_times: Vec<f64>,
    /// Per-EVID-2 covariate snapshot (LOCF), parallel to `pk_only_times`.
    /// Empty when no TV covariates.
    pub pk_only_covariates: Vec<HashMap<String, f64>>,
    /// Censoring flag per observation (0 = quantified, 1 = below LLOQ).
    /// When `cens[j] == 1`, `observations[j]` holds the LLOQ value (NONMEM convention).
    pub cens: Vec<u8>,
    /// Occasion index per observation row (parallel to `obs_times`).
    /// Empty when no IOV column is present in the data.
    pub occasions: Vec<u32>,
    /// Occasion index per dose event (parallel to `doses`).
    /// Empty when no IOV column is present in the data.
    pub dose_occasions: Vec<u32>,
}

impl Subject {
    pub fn has_bloq(&self) -> bool {
        self.cens.iter().any(|&c| c != 0)
    }

    /// True when the subject carries per-event covariate snapshots (i.e. at
    /// least one covariate was time-varying in the source data). When false,
    /// callers can use `covariates` directly and skip the per-event evaluation
    /// loop.
    pub fn has_tv_covariates(&self) -> bool {
        !self.dose_covariates.is_empty() || !self.obs_covariates.is_empty()
    }

    /// True when any dose record on this subject is flagged steady-state
    /// (SS=1). Used to gate paths that don't yet honour SS (event-driven,
    /// AD propagators, ODE) — see `estimation/inner_optimizer.rs` and the
    /// SS warning in `api.rs`.
    pub fn has_ss_doses(&self) -> bool {
        self.doses.iter().any(|d| d.ss)
    }

    /// Covariate snapshot at observation index `j`. Falls back to the
    /// subject-static `covariates` map when per-event snapshots aren't present.
    pub fn obs_cov(&self, j: usize) -> &HashMap<String, f64> {
        self.obs_covariates.get(j).unwrap_or(&self.covariates)
    }

    /// Covariate snapshot at dose index `k`. Same fallback as `obs_cov`.
    pub fn dose_cov(&self, k: usize) -> &HashMap<String, f64> {
        self.dose_covariates.get(k).unwrap_or(&self.covariates)
    }

    /// Covariate snapshot at EVID=2 row index `m`. Same fallback as
    /// the others — for time-constant covariates this returns the
    /// subject-static map.
    pub fn pk_only_cov(&self, m: usize) -> &HashMap<String, f64> {
        self.pk_only_covariates.get(m).unwrap_or(&self.covariates)
    }
}

/// A collection of subjects
#[derive(Debug, Clone)]
pub struct Population {
    pub subjects: Vec<Subject>,
    pub covariate_names: Vec<String>,
    pub dv_column: String,
}

impl Population {
    pub fn n_obs(&self) -> usize {
        self.subjects.iter().map(|s| s.observations.len()).sum()
    }

    /// Drop per-event covariate snapshots that don't carry any
    /// variation in covariates the model actually references.
    ///
    /// The data reader populates `dose_covariates` / `obs_covariates` /
    /// `pk_only_covariates` whenever *any* non-standard CSV column
    /// varies within a subject — including pure time-tracker columns
    /// like `DAY` or `STIME` that no model expression touches. The
    /// downstream prediction path then takes the heavy event-driven
    /// route (one `pk_param_fn` call per event, plus state propagation
    /// across each interval) instead of the analytical superposition
    /// fast path that runs `pk_param_fn` once per subject. This pre-fit
    /// pass clears the snapshots for any subject whose model-referenced
    /// covariates are all time-constant, so the existing
    /// `has_tv_covariates()`-based dispatcher routes those subjects
    /// through the fast path automatically.
    ///
    /// Returns the number of subjects whose snapshots were cleared
    /// (for diagnostic / warning purposes).
    pub fn prune_irrelevant_tv_covariates(&mut self, referenced: &[String]) -> usize {
        let mut pruned = 0;
        for subj in &mut self.subjects {
            if subj.dose_covariates.is_empty()
                && subj.obs_covariates.is_empty()
                && subj.pk_only_covariates.is_empty()
            {
                continue; // already on the fast path
            }
            let mut any_relevant_tv = false;
            'covs: for cov in referenced {
                let base = subj.covariates.get(cov).copied();
                for snap in subj
                    .dose_covariates
                    .iter()
                    .chain(subj.obs_covariates.iter())
                    .chain(subj.pk_only_covariates.iter())
                {
                    if snap.get(cov).copied() != base {
                        any_relevant_tv = true;
                        break 'covs;
                    }
                }
            }
            if !any_relevant_tv {
                subj.dose_covariates.clear();
                subj.obs_covariates.clear();
                subj.pk_only_covariates.clear();
                pruned += 1;
            }
        }
        pruned
    }
}

/// Between-subject variability matrix (Omega)
#[derive(Debug, Clone)]
pub struct OmegaMatrix {
    pub matrix: DMatrix<f64>,
    pub chol: DMatrix<f64>,
    pub eta_names: Vec<String>,
    pub diagonal: bool,
    /// Which (i,j) entries are free parameters (not structural zeros).
    /// Diagonal entries are always free. Off-diagonals are free only when
    /// both etas belong to the same `block_omega` declaration; cross-block
    /// and standalone-vs-block entries are structural zeros and stay false.
    /// Used by the SAEM M-step to zero sampling correlations that bleed into
    /// structurally-absent entries via `(1/N) Σ ηη^T`.
    pub free_mask: DMatrix<bool>,
    /// Pre-computed Ω⁻¹. Cached at construction so per-call code paths
    /// (`individual_nll_into`, SAEM MH proposals) don't have to clone the
    /// matrix, run Cholesky, and invert on every evaluation.
    pub inv: DMatrix<f64>,
    /// Pre-computed `log|Ω| = 2·Σᵢ log(L_ii)`. Same motivation as `inv`.
    pub log_det: f64,
}

impl OmegaMatrix {
    pub fn from_matrix_with_mask(
        m: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = m.nrows();
        // If the input matrix is PD we use it as-is. If Cholesky fails we
        // regularise (eigenvalue floor) and switch to the regularised
        // matrix from here on, so `matrix`, `chol`, `inv`, and `log_det`
        // all describe the *same* matrix downstream — otherwise the
        // FOCE inner loop's eta_prior (read from `matrix`) would be
        // inconsistent with the cached `inv` (computed from `m_reg`).
        let (matrix, chol, inv) = match m.clone().cholesky() {
            Some(c) => (m, c.l(), c.inverse()),
            None => {
                let eig = m.clone().symmetric_eigen();
                let min_eig = eig.eigenvalues.min();
                let reg = if min_eig < 0.0 { -min_eig + 1e-8 } else { 1e-8 };
                let m_reg = &m + DMatrix::identity(n, n) * reg;
                let c = m_reg
                    .clone()
                    .cholesky()
                    .expect("Regularized matrix must be PD");
                (m_reg, c.l(), c.inverse())
            }
        };
        // log|Ω| = 2·Σᵢ log(L_ii). Negative or zero diagonals shouldn't
        // happen post-regularisation but we fall back to f64::INFINITY so
        // downstream NLL code can short-circuit cleanly instead of NaNing.
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = chol[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn from_matrix(m: DMatrix<f64>, names: Vec<String>, diagonal: bool) -> Self {
        let n = m.nrows();
        // Infer free_mask: diagonal entries always free; for non-diagonal
        // matrices, off-diagonals are free iff non-zero. This is the correct
        // inference when reconstructing an OmegaMatrix from a final estimate
        // matrix where the original block structure has already been imposed.
        // For initial parsing of `block_omega` declarations, use
        // `from_matrix_with_mask` directly so cross-block zeros are preserved.
        let mut free_mask = DMatrix::from_element(n, n, false);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    free_mask[(i, j)] = true;
                } else if !diagonal && m[(i, j)] != 0.0 {
                    free_mask[(i, j)] = true;
                }
            }
        }
        Self::from_matrix_with_mask(m, names, diagonal, free_mask)
    }

    pub fn from_diagonal(variances: &[f64], names: Vec<String>) -> Self {
        let n = variances.len();
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = variances[i];
        }
        Self::from_matrix(m, names, true)
    }

    /// Construct from a known lower-Cholesky factor `L` such that Ω = L Lᵀ.
    /// Avoids the Cholesky factorisation that `from_matrix*` runs, which the
    /// SAEM/FOCEI hot paths hit on every NLopt M-step and every outer
    /// iteration via `unpack_params`. Ω⁻¹ is computed from `L` directly:
    /// Ω⁻¹ = L⁻ᵀ L⁻¹, where L⁻¹ comes from one lower-triangular solve
    /// against the identity.
    pub fn from_chol_factor(
        l: DMatrix<f64>,
        names: Vec<String>,
        diagonal: bool,
        free_mask: DMatrix<bool>,
    ) -> Self {
        let n = l.nrows();
        let matrix = &l * l.transpose();
        // L⁻¹ via lower-triangular solve: L · X = I ⇒ X = L⁻¹.
        // `solve_lower_triangular(&I)` returns Some(_) iff L is non-singular;
        // L Lᵀ is PD by construction so a positive-diagonal L is guaranteed
        // unless the caller hands us a degenerate factor — fall back to a
        // full Cholesky on the materialised matrix in that degenerate case
        // so the cache is at least populated rather than panicking.
        let identity = DMatrix::<f64>::identity(n, n);
        let inv = match l.solve_lower_triangular(&identity) {
            Some(l_inv) => &l_inv.transpose() * &l_inv,
            None => matrix
                .clone()
                .cholesky()
                .expect("L Lᵀ should be PD by construction")
                .inverse(),
        };
        let mut log_det = 0.0;
        for i in 0..n {
            let lii = l[(i, i)];
            if lii > 0.0 {
                log_det += lii.ln();
            } else {
                log_det = f64::INFINITY;
                break;
            }
        }
        log_det *= 2.0;
        Self {
            matrix,
            chol: l,
            eta_names: names,
            diagonal,
            free_mask,
            inv,
            log_det,
        }
    }

    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Residual error parameters (Sigma)
#[derive(Debug, Clone)]
pub struct SigmaVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

/// Full set of model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    pub theta_lower: Vec<f64>,
    pub theta_upper: Vec<f64>,
    /// Per-theta FIX flags (NONMEM-style). Fixed thetas are not estimated;
    /// they retain their initial value and receive SE = 0 in the cov step.
    pub theta_fixed: Vec<bool>,
    pub omega: OmegaMatrix,
    /// Per-eta FIX flags. For diagonal omegas: flag fixes the variance.
    /// For block omegas: all etas within a fixed block share `true`, and
    /// every Cholesky element whose row *or* column is flagged is held
    /// constant during optimization. Parser rejects fixing an eta that
    /// belongs to a block unless the whole block is fixed.
    pub omega_fixed: Vec<bool>,
    pub sigma: SigmaVector,
    /// Per-sigma FIX flags.
    pub sigma_fixed: Vec<bool>,
    /// Inter-occasion variability matrix (Omega_IOV). `None` when no `kappa`
    /// declarations appear in the model file.  Always diagonal for Option A.
    pub omega_iov: Option<OmegaMatrix>,
    /// Per-kappa FIX flags (parallel to `omega_iov` diagonal).
    pub kappa_fixed: Vec<bool>,
}

impl ModelParameters {
    /// Return `true` if any parameter is marked FIX.
    pub fn has_any_fixed(&self) -> bool {
        self.theta_fixed.iter().any(|&b| b)
            || self.omega_fixed.iter().any(|&b| b)
            || self.sigma_fixed.iter().any(|&b| b)
            || self.kappa_fixed.iter().any(|&b| b)
    }
}

/// Supported PK structural models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkModel {
    OneCptIvBolus,
    OneCptOral,
    OneCptInfusion,
    TwoCptIvBolus,
    TwoCptOral,
    TwoCptInfusion,
    ThreeCptIvBolus,
    ThreeCptOral,
    ThreeCptInfusion,
}

/// Supported residual error models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    Additive,
    Proportional,
    Combined,
}

/// How a sigma parameter enters the residual error model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaType {
    Proportional,
    Additive,
}

impl ErrorModel {
    /// Return the `SigmaType` for each sigma, in the order they appear in `FitResult.sigma`.
    pub fn sigma_types(self) -> Vec<SigmaType> {
        match self {
            ErrorModel::Proportional => vec![SigmaType::Proportional],
            ErrorModel::Additive => vec![SigmaType::Additive],
            ErrorModel::Combined => vec![SigmaType::Proportional, SigmaType::Additive],
        }
    }
}

/// Transformation applied to a theta on the natural scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThetaTransform {
    /// Theta is on the natural scale (no transformation).
    Identity,
    /// Theta is on the log scale; back-transform = exp(theta).
    Log,
    /// Theta is on the logit scale: `inv_logit(THETA + ETA)`. User sets THETA
    /// on the logit scale (e.g. logit(0.7) ≈ 0.847).
    Logit,
    /// Theta is on the probability scale: `inv_logit(logit(THETA) + ETA)`.
    /// User sets THETA directly in (0,1) (e.g. 0.70 for 70% bioavailability).
    LogitProbability,
}

/// Distribution / parameterisation of an ETA random effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtaParamType {
    /// `TVCL * exp(ETA)` or `exp(THETA + ETA)` — log-normal.
    LogNormal,
    /// `TVCL + ETA` — normal (additive).
    Additive,
    /// `inv_logit(THETA + ETA)` — logit-normal; THETA on the logit scale.
    Logit,
    /// `inv_logit(logit(THETA) + ETA)` — logit-normal; THETA on the (0,1) scale.
    LogitProbability,
    /// Pattern not automatically recognised.
    Custom,
}

/// Per-ETA transformation metadata, carried in `FitResult`.
#[derive(Debug, Clone)]
pub struct EtaParamInfo {
    pub eta_name: String,
    pub param_type: EtaParamType,
    /// Theta paired with this ETA. Set only when the ETA is added directly to a single THETA in
    /// the same expression (e.g. `THETA * exp(ETA)` or `inv_logit(THETA + ETA)`).
    /// Not set for mu-ref patterns like `TVCL * exp(ETA)` where the THETA is a scale factor.
    pub linked_theta: Option<String>,
    /// Name of the individual parameter this ETA appears in (e.g. `"CL"`).
    pub individual_param_name: String,
}

/// PK parameter function: maps (theta, eta, covariates) -> PkParams
pub type PkParamFn = Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> PkParams + Send + Sync>;

/// Closure signature for `[scaling] obs_scale = <expr>` (Form B). Receives
/// `(theta, eta, covariates, pk_params)` and returns the per-subject scale
/// factor used to divide the raw prediction. `pk_params` is the subject-
/// static evaluation of `model.pk_param_fn`, so the scale expression can
/// reference individual parameters (e.g. `obs_scale = 1000 / V`) — the
/// closure looks up V via its PK slot in `pk_params.values`.
pub type ScaleFn =
    Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>, &PkParams) -> f64 + Send + Sync>;

/// How the structural model's raw output is mapped to the observed `DV`.
///
/// Set by the `[scaling]` block in `.ferx` model files. The convention is
/// **divisive**: `pred_scaled = pred_raw / scale`. This matches the natural
/// reading of `obs_scale = V/1000` as "divide amount by V/1000 to get
/// concentration in the user's units."
///
/// Forms A/B (this enum) post-multiply analytical and ODE predictions
/// uniformly at the end of the prediction dispatcher. Form C (ODE-only
/// `y = <expr>`) is handled inside the ODE timeline loop via
/// `OdeSpec::output_fn` instead — it replaces the state readout entirely,
/// so it doesn't share the post-multiply path.
pub enum ScalingSpec {
    /// No scaling: prediction is returned as-is.
    None,
    /// Constant divisor applied to every prediction.
    ScalarScale(f64),
    /// Per-subject divisor evaluated from `(theta, eta, covariates, pk)`.
    /// Used for expressions like `obs_scale = 1000 / V`.
    ExpressionScale { scale_fn: ScaleFn },
    /// Per-CMT dispatch for multi-analyte models (parent+metabolite,
    /// sum-of-moieties, free vs total, ...). Key is the 1-based CMT
    /// index from the data file's CMT column (matches
    /// `subject.obs_cmts[i]`, which is `usize`). Each entry is one of
    /// `None` / `ScalarScale` / `ExpressionScale` (no nested `PerCmt`
    /// — parser enforces).
    ///
    /// Fit-time validation requires every observed CMT in the population
    /// to have an entry; missing entries fall through to NaN predictions
    /// at runtime as a defensive guard against hand-constructed
    /// CompiledModels.
    PerCmt(HashMap<usize, ScalingSpec>),
}

impl Default for ScalingSpec {
    fn default() -> Self {
        Self::None
    }
}

impl ScalingSpec {
    /// Returns the per-subject scalar divisor for the AD path.
    /// `None` → 1.0 (no-op). `ScalarScale(k)` → `k`. The remaining
    /// variants are rejected at parse time when the model resolves
    /// `gradient_method == Ad` (ExpressionScale because the closure
    /// needs theta/eta; PerCmt because per-observation scale arrays
    /// aren't yet threaded through the AD entry points) — they return
    /// `1.0` with a debug assertion rather than producing incorrect
    /// gradients.
    #[inline]
    pub fn scalar_for_ad(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::ScalarScale(k) => *k,
            Self::ExpressionScale { .. } => {
                debug_assert!(
                    false,
                    "ExpressionScale with AD must be rejected at parse time"
                );
                1.0
            }
            Self::PerCmt(_) => {
                debug_assert!(
                    false,
                    "PerCmt scaling with AD must be rejected at parse time"
                );
                1.0
            }
        }
    }

    /// Returns true if this spec is `PerCmt` or contains an
    /// `ExpressionScale` (directly or in any PerCmt entry). Used by the
    /// parser to decide whether the gradient method must be FD.
    pub fn requires_fd(&self) -> bool {
        match self {
            Self::None | Self::ScalarScale(_) => false,
            Self::ExpressionScale { .. } => true,
            // PerCmt always forces FD (Phase 2 limitation), independent
            // of whether the inner variants need it.
            Self::PerCmt(_) => true,
        }
    }
}

impl std::fmt::Debug for ScalingSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "ScalingSpec::None"),
            Self::ScalarScale(k) => write!(f, "ScalingSpec::ScalarScale({})", k),
            Self::ExpressionScale { .. } => write!(f, "ScalingSpec::ExpressionScale {{ .. }}"),
            Self::PerCmt(map) => {
                let cmts: Vec<usize> = {
                    let mut v: Vec<usize> = map.keys().copied().collect();
                    v.sort();
                    v
                };
                write!(f, "ScalingSpec::PerCmt({:?})", cmts)
            }
        }
    }
}

/// Associates an ETA with its mu-referencing anchor theta.
#[derive(Debug, Clone)]
pub struct MuRef {
    pub theta_name: String,
    /// true for patterns THETA*exp(ETA) or exp(log(THETA)+ETA); false for THETA+ETA
    pub log_transformed: bool,
}

/// A compiled model ready for estimation
pub struct CompiledModel {
    pub name: String,
    pub pk_model: PkModel,
    pub error_model: ErrorModel,
    pub pk_param_fn: PkParamFn,
    pub n_theta: usize,
    /// Number of between-subject variability (BSV) ETAs.
    pub n_eta: usize,
    /// Number of inter-occasion variability (IOV) kappa parameters.
    /// Zero when no `kappa` declarations are present.
    pub n_kappa: usize,
    pub n_epsilon: usize,
    pub theta_names: Vec<String>,
    /// BSV ETA names only (length == n_eta).
    pub eta_names: Vec<String>,
    /// IOV kappa names (length == n_kappa). Empty when no IOV.
    pub kappa_names: Vec<String>,
    /// Names of the individual parameters declared at the top level of the
    /// `[individual_parameters]` block, in declaration order. Parallel to
    /// `pk_indices`; for analytical models the i-th name is the variable
    /// whose value lands in `PkParams.values[pk_indices[i]]`. For ODE
    /// models the i-th name is written sequentially into slot `i` by
    /// `pk_param_fn`. Used by the FFI to label per-subject EBE individual
    /// parameter values (e.g. `CL`, `V`, `Ka`).
    ///
    /// Bound: `pk_param_fn` writes at most `MAX_PK_PARAMS` slots (the size
    /// of the fixed `PkParams.values` array). For analytical models the
    /// parser already routes assignments through that fixed slot table, so
    /// excess names are not possible. For ODE models with more than
    /// `MAX_PK_PARAMS` top-level `[individual_parameters]` assignments,
    /// names beyond index `MAX_PK_PARAMS - 1` will appear in this list but
    /// `pk_param_fn` won't store their values — downstream consumers will
    /// read either zero or NaN for those slots. In practice no PK model
    /// approaches this limit.
    pub indiv_param_names: Vec<String>,
    pub default_params: ModelParameters,
    /// Per-eta flag (parallel to `eta_names` / omega diagonal): `true` when
    /// the user wrote `omega NAME ~ X (sd)` and the parser squared the value.
    /// Pure display metadata — the stored `default_params.omega` is always on
    /// the variance scale. Always `false` for etas declared inside a
    /// `block_omega`, which is variance-only.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma flag (parallel to `default_params.sigma.values`): `true` when
    /// the user wrote `sigma NAME ~ X (sd)`. Since #56 the .ferx default for
    /// sigma is variance — the parser `sqrt`s the variance-scale input into
    /// the internal SD representation. With `(sd)` the value is stored as-is.
    pub sigma_init_as_sd: Vec<bool>,
    /// Per-kappa flag (parallel to `kappa_names`): `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Empty when no kappa declarations are present.
    pub kappa_init_as_sd: Vec<bool>,
    /// Detected mu-referencing relationships: eta_name → (theta_name, log_transformed).
    /// Populated by the parser; empty map means no mu-referencing detected.
    pub mu_refs: HashMap<String, MuRef>,
    /// Same as `mu_refs` but for IOV kappa parameters (kappa_name → MuRef).
    pub kappa_mu_refs: HashMap<String, MuRef>,
    /// Computes covariate-adjusted typical values per subject for AD.
    /// Returns one value per `[individual_parameters]` assignment (in
    /// declaration order), evaluated with eta = 0. Covariates and theta are
    /// folded in; only eta is differentiated. The AD inner loop then
    /// computes `pk[pk_indices[i]] = tv[i] * exp(dot(sel_flat[i,:], eta))`,
    /// so `tv.len() == pk_indices.len() == eta_map.len() == sel_flat.len() / n_eta`,
    /// and the eta application is driven by `sel_flat` rather than being
    /// positional. When `Some`, enables AD gradient computation in the
    /// inner loop; when `None` (e.g. ODE models), falls back to FD.
    pub tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>>,
    /// Maps each `[individual_parameters]` assignment (by declaration order)
    /// to its PK parameter slot. E.g. for a model with CL, V, KA:
    /// `[PK_IDX_CL, PK_IDX_V, PK_IDX_KA] = [0, 1, 4]`. Parallel to the
    /// output of `tv_fn` and to `eta_map`; used by AD to route each tv
    /// value to the correct PK slot. Note: the index here is the
    /// assignment/tv index, *not* the eta index — see `eta_map` for the
    /// latter (they diverge when some params are eta-free).
    pub pk_indices: Vec<usize>,
    /// Per-tv eta index: `eta_map[i]` is the eta index referenced by the
    /// i-th [individual_parameters] assignment, or -1 if the assignment
    /// references no eta (e.g. `Q = TVQ`). Parallel to `pk_indices` and the
    /// output of `tv_fn`; used by the AD path to correctly combine eta
    /// with each tv slot. Before this field existed the AD loop assumed
    /// `pk_indices.len() == n_eta` with 1:1 positional correspondence,
    /// which silently misaligned eta and produced NaN gradients for models
    /// with eta-free PK parameters like 2-cpt where `Q` is fixed.
    pub eta_map: Vec<i32>,
    /// Precomputed `pk_indices` as `Vec<f64>` — the form the AD functions
    /// actually want. Cached here so each BFGS gradient call doesn't
    /// reallocate and recast a tiny vector; on a 110k-find_ebe fit that
    /// saves several million allocations.
    pub pk_idx_f64: Vec<f64>,
    /// Precomputed one-hot eta selector (row-major, n_tv × n_eta) derived
    /// from `eta_map`. Same motivation as `pk_idx_f64`: built once, reused
    /// for every AD gradient evaluation.
    pub sel_flat: Vec<f64>,
    /// ODE specification. When `Some`, predictions use ODE integration instead of
    /// analytical PK equations. The `pk_param_fn` output is flattened and passed
    /// to the ODE RHS function as the parameter vector.
    pub ode_spec: Option<crate::ode::OdeSpec>,
    /// Index of the first diffusion theta in the theta vector, and the parallel
    /// mapping from diffusion-theta index to ODE state index.
    /// `None` when no `[diffusion]` block is present.
    /// Used by `ekf_p_obs` to read current diffusion variances from `theta`
    /// without requiring mutation of `ode_spec` during estimation.
    pub diffusion_theta_start: Option<usize>,
    /// For each diffusion theta (offset from `diffusion_theta_start`),
    /// the index of the ODE state it applies to. Parallel to the diffusion
    /// theta slice of `theta`. Empty when `diffusion_theta_start` is `None`.
    pub diffusion_state_indices: Vec<usize>,
    /// Mirror of [`FitOptions::bloq_method`] so likelihood/AD paths can read
    /// it without threading the options struct through every call site.
    /// Set by [`fit_from_files`](crate::fit_from_files) automatically;
    /// callers invoking [`fit`](crate::fit) with a hand-built `CompiledModel`
    /// must set this field to match `options.bloq_method` themselves.
    pub bloq_method: BloqMethod,
    /// Covariate names referenced by any expression in the model (preserved
    /// in the case the modeller wrote). Validated against the data's covariate
    /// columns before a fit so that a missing/misspelt covariate fails loudly
    /// instead of silently evaluating to zero.
    pub referenced_covariates: Vec<String>,
    /// Mirror of [`FitOptions::gradient_method`] so the inner loop can
    /// dispatch at runtime without threading the options struct through
    /// every call site. Set by [`fit_from_files`](crate::fit_from_files)
    /// automatically; callers invoking [`fit`](crate::fit) with a
    /// hand-built `CompiledModel` must set this field to match
    /// `options.gradient_method` themselves. A mismatch is not detected —
    /// `find_ebe` reads this field, not `options`.
    pub gradient_method: GradientMethod,
    /// Warnings generated at parse time (e.g. mu-referencing disabled for
    /// conditional parameters).  Prepended to `FitResult.warnings` by `fit()`.
    pub parse_warnings: Vec<String>,
    /// Per-ETA transformation metadata derived from the `[individual_parameters]`
    /// expressions at parse time. Length ≤ n_eta (only ETAs whose expression was
    /// classified are present). Forwarded into `FitResult`.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation: `theta_transform[i]` describes whether theta i
    /// is used on the natural (Identity), log, or logit scale. Length == n_theta.
    pub theta_transform: Vec<ThetaTransform>,
    /// Parsed `[covariate_nn NAME]` blocks (one entry per block in the model
    /// file). Empty when the `nn` feature is off or no block is present.
    ///
    /// Consumed by `build_pk_param_fn` and `tv_fn`: each NN's forward output is
    /// pre-computed once per call and looked up by `Expression::NnOutput` via
    /// the `NAME.OUTPUT` dot-access syntax. Weights live in `theta` starting at
    /// `weights_offset` for `mapper.n_weights()` slots, so they participate in
    /// the optimizer vector like any other theta.
    #[cfg(feature = "nn")]
    pub covariate_nns: Vec<crate::nn::CovariateNn>,
    /// How the structural model's raw output is mapped to the observed `DV`.
    /// Default `ScalingSpec::None` preserves the historical behaviour where
    /// the prediction is returned unchanged. Forms A/B from the `[scaling]`
    /// block populate this field; Form C lives on `ode_spec.output_fn`.
    pub scaling: ScalingSpec,
}

/// Inner-loop (per-subject EBE) gradient method.
///
/// The inner optimizer is BFGS; what differs across variants is how the
/// gradient of the individual NLL w.r.t. ETA is computed.
///
/// - `Ad`: reverse-mode automatic differentiation via Enzyme. One forward
///   pass + one reverse pass per gradient, regardless of `n_eta`. Requires
///   the crate to be compiled with the `autodiff` feature and the model to
///   have an analytical PK path (`tv_fn` populated). Falls back to `Fd`
///   automatically when either condition isn't met (e.g. ODE models, which
///   currently have no AD path).
/// - `Fd`: central finite differences on the forward NLL. Performs `2·n_eta`
///   forward evaluations per gradient, so cost scales linearly with the
///   number of random effects.
/// - `Auto` (default): pick `Ad` whenever it is available, else `Fd`.
///
/// ## When each wins
///
/// AD's relative advantage over FD grows with:
/// 1. **Number of etas.** FD cost scales as `O(n_eta)`; AD stays roughly
///    flat. For `n_eta ≥ 3` AD is already faster per gradient call on every
///    analytical PK model tested.
/// 2. **Forward-pass cost.** Many observations per subject, many doses per
///    subject, 2- or 3-compartment analytical formulas, and (when
///    implemented) ODE-based models all amortize AD's fixed reverse-pass
///    overhead and make the per-gradient gap wider.
///
/// On small analytical problems (`n_eta ≈ 3`, few observations, 1-cpt PK)
/// the wall-clock difference can be small because gradient work is only a
/// fraction of total fit time — NLopt, population NLL reduction, and
/// parallel scheduling dominate. Relative gradient-call speedups we have
/// measured range from ~1.5× (3-cpt infusion) to ~5× (1-cpt oral).
///
/// ## Numerical equivalence
///
/// For well-conditioned problems both methods converge to the same OFV
/// within line-search tolerance. FD introduces `O(1e-9)` noise per
/// component; AD is exact up to floating-point roundoff. Rare disagreements
/// at the 2nd-decimal level of OFV usually reflect different trajectories
/// to the same optimum rather than a correctness gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientMethod {
    Auto,
    Ad,
    Fd,
}

impl Default for GradientMethod {
    fn default() -> Self {
        Self::Auto
    }
}

impl CompiledModel {
    /// Returns true when this model uses ODE integration; false for analytical PK.
    pub fn is_ode_based(&self) -> bool {
        self.ode_spec.is_some()
    }

    /// Returns true when the model has a `[diffusion]` block (SDE / EKF path).
    pub fn is_sde(&self) -> bool {
        self.diffusion_theta_start.is_some()
    }

    /// Returns true when `[individual_parameters]` declares `LAGTIME` (or its
    /// `ALAG` alias). Used by the prediction dispatcher and inner optimizer
    /// to choose between cached-schedule / AD fast paths and the lagtime-
    /// aware slow paths.
    ///
    /// Checks both routes by which lagtime can be wired in:
    ///   1. Analytical PK: `pk_indices` contains `PK_IDX_LAGTIME` when the
    ///      `[structural_model]` line includes `lagtime=` / `alag=`.
    ///   2. ODE: the LAGTIME/ALAG slot is populated by name in
    ///      `build_pk_param_fn`'s ODE branch (sequential pk_indices do not
    ///      reflect this), so we fall back to scanning `indiv_param_names`.
    pub fn has_lagtime(&self) -> bool {
        if self.pk_indices.iter().any(|&i| i == PK_IDX_LAGTIME) {
            return true;
        }
        self.indiv_param_names.iter().any(|n| {
            let u = n.to_uppercase();
            u == "LAGTIME" || u == "ALAG"
        })
    }
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("name", &self.name)
            .field("pk_model", &self.pk_model)
            .field("error_model", &self.error_model)
            .field("n_theta", &self.n_theta)
            .field("n_eta", &self.n_eta)
            .field("n_kappa", &self.n_kappa)
            .finish()
    }
}

/// Per-subject estimation results
#[derive(Debug, Clone)]
pub struct SubjectResult {
    pub id: String,
    pub eta: DVector<f64>,
    pub ipred: Vec<f64>,
    pub pred: Vec<f64>,
    pub iwres: Vec<f64>,
    pub cwres: Vec<f64>,
    pub ofv_contribution: f64,
    pub cens: Vec<u8>,
    /// Number of observations for this subject (MDV=0 rows).
    pub n_obs: usize,
}

/// How per-occasion kappa random effects are treated in the IS marginal likelihood.
///
/// `Marginalized` — kappa is jointly sampled with eta (planned v2; not yet implemented).
/// `FixedAtMode` — kappa is held at its EBE (κ̂) and only eta is sampled. This is a
/// *partial* marginal likelihood that ignores κ uncertainty, so the reported `-2LL`
/// is biased downward relative to the true marginal and is not directly comparable
/// to NONMEM's `$EST METHOD=IMP LAPLACIAN=1` on κ.
/// `NotApplicable` — model has no kappa declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaTreatment {
    NotApplicable,
    FixedAtMode,
    // Reserved for joint (eta, kappa) sampling; see TODO(imp-iov-v2) in
    // `estimation/importance_sampling.rs`. Never constructed by v1 IMP so
    // `dead_code` would otherwise fire on a strict clippy run.
    #[allow(dead_code)]
    Marginalized,
}

/// Result of the importance-sampling marginal log-likelihood step.
///
/// Produced by the `Imp` stage in a method chain (`methods = [..., imp]`).
/// Surfaced on `FitResult.importance_sampling`.
#[derive(Debug, Clone)]
pub struct ImportanceSamplingResult {
    /// `−2 · Σᵢ log p(yᵢ | θ)` estimated by importance sampling. Lower-bias
    /// alternative to the FOCE/Laplace OFV when subject posteriors are
    /// non-Gaussian (sparse data, strong nonlinearity).
    pub minus2_log_likelihood: f64,
    /// Monte-Carlo standard error on `minus2_log_likelihood`. Scales with
    /// `1/sqrt(n_samples)`; halve by quadrupling `is_samples`.
    pub mc_standard_error: f64,
    /// `(subject_id, ESS/K)` for every subject whose normalized effective sample
    /// size fraction fell below `FitOptions::is_low_ess_threshold`. Empty list
    /// means every subject's proposal matched its posterior well.
    pub low_ess_subjects: Vec<(String, f64)>,
    /// Number of importance samples drawn per subject (`FitOptions::is_samples`).
    pub n_samples: usize,
    /// Student-t proposal degrees of freedom (`FitOptions::is_proposal_df`).
    pub proposal_df: f64,
    /// Minimum across-subject normalized ESS fraction (ESS / K). 1.0 = ideal,
    /// near 0 = degenerate proposal for at least one subject.
    pub ess_min: f64,
    /// Median across-subject normalized ESS fraction.
    pub ess_median: f64,
    /// Treatment of per-occasion kappa random effects.  See [`KappaTreatment`].
    pub kappa_treatment: KappaTreatment,
}

/// Outcome of the post-estimation covariance step.
#[derive(Debug, Clone, PartialEq)]
pub enum CovarianceStatus {
    /// User set `covariance = false`; step was not attempted.
    NotRequested,
    /// Covariance matrix was successfully computed.
    Computed,
    /// Step was attempted but failed (e.g. singular Hessian).
    Failed,
}

/// Full fit result
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Final method in the chain (same as `method_chain.last()`).
    pub method: EstimationMethod,
    /// Full sequence of methods executed, in order. Always has at least one entry.
    pub method_chain: Vec<EstimationMethod>,
    pub converged: bool,
    pub ofv: f64,
    pub aic: f64,
    pub bic: f64,
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    /// Names of the random effects (etas), parallel to the omega diagonal.
    pub eta_names: Vec<String>,
    pub omega: DMatrix<f64>,
    pub sigma: Vec<f64>,
    /// Names of the sigma parameters, parallel to `sigma`.
    pub sigma_names: Vec<String>,
    /// Residual error model (additive, proportional, combined).
    pub error_model: ErrorModel,
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub se_theta: Option<Vec<f64>>,
    pub se_omega: Option<Vec<f64>>,
    pub se_sigma: Option<Vec<f64>>,
    /// FIX flags carried through from the model so the output layer can
    /// render `FIXED` for SE columns rather than the (meaningless) zero
    /// they acquire from the reduced-Hessian covariance step.
    pub theta_fixed: Vec<bool>,
    pub omega_fixed: Vec<bool>,
    pub sigma_fixed: Vec<bool>,
    /// Per-eta SD-init flag (parallel to the omega diagonal): `true` when the
    /// initial value was written as `omega NAME ~ X (sd)`. Lets downstream
    /// printers annotate the estimate with `[initial specified as SD]`.
    /// Always `false` for block-omega entries.
    pub omega_init_as_sd: Vec<bool>,
    /// Per-sigma SD-init flag (parallel to `sigma`): `true` when the initial
    /// value was written as `sigma NAME ~ X (sd)`, `false` for the variance
    /// default.
    pub sigma_init_as_sd: Vec<bool>,
    pub subjects: Vec<SubjectResult>,
    pub n_obs: usize,
    pub n_subjects: usize,
    pub n_parameters: usize,
    pub n_iterations: usize,
    pub interaction: bool,
    pub warnings: Vec<String>,
    // SIR results (optional)
    pub sir_ci_theta: Option<Vec<(f64, f64)>>,
    pub sir_ci_omega: Option<Vec<(f64, f64)>>,
    pub sir_ci_sigma: Option<Vec<(f64, f64)>>,
    pub sir_ess: Option<f64>,
    /// Resampled packed parameter vectors retained from the SIR step, available
    /// when `FitOptions.sir_keep_samples = true`. Each `Vec<f64>` is a draw in
    /// the packed parameter space — same layout as `pack_params`:
    /// `[log-theta, Cholesky-omega, log-sigma]`, with the IOV Cholesky block
    /// appended when the model has kappa declarations.
    /// Consumed by `simulate_with_uncertainty()` with `UncertaintyMethod::Sir`.
    pub sir_resamples_packed: Option<Vec<Vec<f64>>>,
    /// Importance-sampling marginal log-likelihood result. `Some` when an
    /// `Imp` stage ran in the method chain (`methods = [..., imp]`). The
    /// `−2 log L_IS` value is lower-bias than the FOCE/Laplace `ofv` for
    /// sparsely-sampled subjects and is the preferred quantity for AIC/BIC
    /// model comparison in those settings. See [`ImportanceSamplingResult`].
    pub importance_sampling: Option<ImportanceSamplingResult>,
    // IOV results (present when kappa declarations exist in the model)
    pub omega_iov: Option<DMatrix<f64>>,
    pub kappa_names: Vec<String>,
    pub kappa_fixed: Vec<bool>,
    /// Per-kappa SD-init flag (parallel to the `omega_iov` diagonal). Same
    /// semantics as `omega_init_as_sd` — `true` when the user wrote
    /// `kappa NAME ~ X (sd)`. Always `false` for block_kappa entries.
    pub kappa_init_as_sd: Vec<bool>,
    pub se_kappa: Option<Vec<f64>>,
    pub shrinkage_kappa: Vec<f64>,
    /// Per-subject, per-occasion kappa EBEs.
    /// `ebe_kappas[i][k]` is the kappa vector for subject i, occasion k.
    /// Outer vec is empty when `n_kappa == 0`.
    pub ebe_kappas: Vec<Vec<DVector<f64>>>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    /// Number of subjects that used HMC at least once during the SAEM E-step.
    /// `None` when `n_leapfrog = 0` (MH-only) or for non-SAEM methods.
    pub saem_n_subjects_hmc: Option<usize>,
    /// Gradient method used in the inner (per-subject EBE) BFGS loop.
    pub gradient_method_inner: String,
    /// Gradient method used in the outer (population parameter) optimizer.
    pub gradient_method_outer: String,
    /// True when the model uses ODE integration; false for analytical PK.
    pub uses_ode_solver: bool,
    /// True when the model has a `[diffusion]` block (SDE / EKF likelihood).
    pub uses_sde: bool,
    /// Number of Rayon worker threads used during this fit.
    pub n_threads_used: usize,
    /// NLopt algorithms requested but not available in this platform build.
    pub nlopt_missing_algorithms: Vec<String>,
    /// Estimated OFV evaluations for the covariance step (n_params²), set
    /// when `run_covariance_step = true` and `n_parameters > 30`.
    pub covariance_n_evals_estimated: Option<usize>,
    /// Path to the per-iteration optimizer trace CSV, present when
    /// `FitOptions::optimizer_trace = true`.
    pub trace_path: Option<String>,
    /// Number of outer iterations in which at least one subject had an
    /// unconverged EBE.  Always `0` for SAEM (which uses MH sampling).
    pub ebe_convergence_warnings: u32,
    /// Worst-case number of unconverged subjects in a single outer iteration.
    pub max_unconverged_subjects: u32,
    /// Total number of times the Nelder-Mead fallback was invoked across all
    /// subjects and all outer iterations.  Always `0` for SAEM.
    pub total_ebe_fallbacks: u32,
    /// Outcome of the post-estimation covariance step.
    pub covariance_status: CovarianceStatus,
    /// ETA shrinkage per random effect: `1 - SD(eta_hat_k) / sqrt(omega_kk)`.
    /// `NaN` when `omega_kk` is zero.
    pub shrinkage_eta: Vec<f64>,
    /// EPS shrinkage: `1 - SD(IWRES)`.  `NaN` when fewer than 2 valid residuals.
    pub shrinkage_eps: f64,
    /// Pooled lag-1 Pearson correlation of IWRES across subjects.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub iwres_lag1_r: f64,
    /// Pooled Durbin-Watson statistic for IWRES within subjects.
    /// 2.0 = no autocorrelation; < 1.5 = positive; > 2.5 = negative.
    /// `NaN` when no subject has ≥ 2 valid IWRES values.
    pub dw_statistic: f64,
    /// Wall-clock time for the complete fit in seconds.
    pub wall_time_secs: f64,
    /// Model name (from the `.ferx` file or "Unnamed").
    pub model_name: String,
    /// ferx-core library version (from Cargo.toml at compile time).
    pub ferx_version: String,
    /// Per-ETA transformation metadata (see `EtaParamInfo`). Used by the R
    /// layer to pick the correct CI / CV% formula for each random effect.
    pub eta_param_info: Vec<EtaParamInfo>,
    /// Per-theta transformation (Identity / Log / Logit), parallel to `theta`.
    /// Tells the R layer whether a theta must be back-transformed before display.
    pub theta_transform: Vec<ThetaTransform>,
    /// Per-sigma type (Proportional / Additive), parallel to `sigma`.
    pub sigma_types: Vec<SigmaType>,
    /// Eigenvalues of the correlation matrix of free (non-fixed) parameters,
    /// sorted descending. `None` when the covariance step was not run, failed,
    /// or fewer than two free parameters exist.
    pub cov_eigenvalues: Option<Vec<f64>>,
    /// Ratio of the largest to smallest eigenvalue of the correlation matrix of
    /// free parameters. `f64::INFINITY` when the smallest eigenvalue is
    /// non-positive (signals a near-singular parameter space). `None` when
    /// `cov_eigenvalues` is `None`.
    pub cov_condition_number: Option<f64>,
    /// Whether each BSV eta is lognormally parameterised (`true`) or
    /// additive/unknown (`false`). Parallel to `eta_names` / omega diagonal.
    pub eta_log_transformed: Vec<bool>,
    /// Parameter-level correlation matrix for BSV omega.  Entry `[i,j]` uses
    /// the lognormal formula `(exp(ω_ij)−1)/√((exp(ω_ii)−1)(exp(ω_jj)−1))`
    /// when both etas are lognormal, otherwise falls back to
    /// `ω_ij/√(ω_ii·ω_jj)`.  `None` when omega is diagonal (no off-diagonals).
    pub omega_param_corr: Option<DMatrix<f64>>,
    /// Parameter-level correlation matrix for IOV block kappa, analogous to
    /// `omega_param_corr`.  `None` when `omega_iov` is absent or diagonal.
    pub omega_iov_param_corr: Option<DMatrix<f64>>,
    /// Path to the `.ferx` model file used for this fit, as supplied by the
    /// caller. `Some` when the fit was launched via `fit_from_files` or
    /// `run_model_with_data`; `None` when `fit()` was called with an in-memory
    /// `CompiledModel`. Stored verbatim (no canonicalisation) so paths don't
    /// leak the runner's home directory into shared `.fitrx` bundles.
    pub model_path: Option<String>,
    /// Path to the NONMEM-format CSV data file used for this fit, as supplied
    /// by the caller. `Some` / `None` follows the same rules as `model_path`.
    pub data_path: Option<String>,
    /// SHA-256 hex digest (64 chars, lowercase) of the model file bytes at
    /// fit time. Used by `run_sir` to refuse stale data when the caller
    /// re-supplies a model or asks the function to re-read from `model_path`.
    /// Computed only when the fit was launched from a file path.
    pub model_hash: Option<String>,
    /// SHA-256 hex digest of the data file bytes at fit time. Same semantics
    /// as `model_hash`.
    pub data_hash: Option<String>,
    /// One entry per `[covariate_nn NAME]` block in the model, populated by
    /// `fit()` from `CompiledModel.covariate_nns`. Empty when the `nn`
    /// feature is off or no block is declared. Output writers
    /// (`write_estimates_yaml`, `print_results`, `.fitrx`) use this to
    /// collapse the wall of per-weight thetas (`W_NN_l_i_j`, `B_NN_l_i`)
    /// into a single readable `neural_networks:` summary section — see
    /// `plans/dcm-and-low-dim-node.md` "Option E".
    #[cfg(feature = "nn")]
    pub neural_networks: Vec<NeuralNetworkInfo>,
}

/// Minimal per-NN metadata carried on `FitResult` so output writers can
/// summarise NN weights without re-walking `theta_names` to detect them.
#[cfg(feature = "nn")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuralNetworkInfo {
    /// Block name from the model file (e.g. `TYPICAL_PK`).
    pub name: String,
    /// Layer shape including input and output dimensions
    /// (e.g. `[2, 16, 5]` for 2 inputs → 16 hidden → 5 outputs).
    pub shape: Vec<usize>,
    /// Hidden-layer activation name (e.g. `"tanh"`, `"relu"`).
    pub hidden_activation: String,
    /// Output-layer activation name.
    pub output_activation: String,
    /// Total weight + bias count for this NN.
    pub n_weights: usize,
    /// Index into `FitResult.theta` (and `FitResult.theta_names`) where
    /// this NN's contiguous weight block starts.
    pub weights_offset: usize,
    /// Input covariate names in declaration order.
    pub input_names: Vec<String>,
    /// PK output names in declaration order.
    pub output_names: Vec<String>,
}

/// Options for fit()
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Primary estimation method (used when `methods` is empty).
    /// When `methods` is non-empty, `method` is ignored for execution and
    /// is set to the final method in the chain for backwards-compatible reporting.
    pub method: EstimationMethod,
    /// Sequence of estimation methods to run. Each stage's converged parameters
    /// are used as the initial values for the next stage. The final stage
    /// produces the reported fit (covariance, diagnostics, OFV). Leave empty
    /// to run a single stage using `method`.
    pub methods: Vec<EstimationMethod>,
    pub outer_maxiter: usize,
    pub outer_gtol: f64,
    pub inner_maxiter: usize,
    pub inner_tol: f64,
    pub run_covariance_step: bool,
    pub interaction: bool,
    pub verbose: bool,
    pub optimizer: Optimizer,
    pub lbfgs_memory: usize,
    /// Run a gradient-free global pre-search (NLopt GN_CRS2_LM) before local optimization.
    pub global_search: bool,
    /// Max evaluations for the global pre-search (0 = auto).
    pub global_maxeval: usize,
    // SAEM-specific options
    pub saem_n_exploration: usize,
    pub saem_n_convergence: usize,
    pub saem_n_mh_steps: usize,
    pub saem_adapt_interval: usize,
    pub saem_seed: Option<u64>,
    /// Number of leapfrog steps per HMC proposal in the SAEM E-step.
    /// `0` (default) uses the Metropolis-Hastings random-walk sampler.
    /// A positive value (e.g. `3`) enables HMC; requires the `autodiff`
    /// feature and an analytical PK model — falls back to MH otherwise.
    pub saem_n_leapfrog: usize,
    /// Levenberg-Marquardt damping factor for Gauss-Newton (0 = pure GN).
    pub gn_lambda: f64,
    // SIR options
    pub sir: bool,
    pub sir_samples: usize,
    pub sir_resamples: usize,
    pub sir_seed: Option<u64>,
    /// When `true` and SIR is enabled, the resampled packed parameter vectors
    /// are retained on `FitResult.sir_resamples_packed` for downstream use by
    /// `simulate_with_uncertainty()`. Adds `n_resamples * n_packed * 8` bytes
    /// to the result; default `false`.
    pub sir_keep_samples: bool,
    /// Degrees of freedom for the Student-t SIR proposal distribution.
    /// Heavier tails than the normal improve ESS for parameters near boundaries
    /// (omega variances, constrained thetas). Default 5.0 follows Dosne (2017).
    /// Set to a large value (e.g. 100.0) to recover near-normal behaviour.
    pub sir_df: f64,
    // Importance-sampling marginal log-likelihood options (consumed by the
    // `Imp` chain stage; ignored otherwise). The Imp stage estimates
    // `−2 log L = −2 Σᵢ log ∫ p(yᵢ|η,θ)p(η|θ) dη` by Monte Carlo with a
    // Student-t proposal centred on each subject's EBE.
    /// Number of importance samples per subject. Default 1000. Recommended
    /// 2000–5000 for publication-quality MC SE (cost scales linearly).
    pub is_samples: usize,
    /// Degrees of freedom for the Student-t proposal. Default 5.0 (heavy-tailed
    /// — robust to mild proposal misspecification). Must be ≥ 1.
    pub is_proposal_df: f64,
    /// RNG seed for the IS sampling. `None` falls back to a fixed default so
    /// runs are reproducible across invocations.
    pub is_seed: Option<u64>,
    /// Subjects with normalized effective sample size below this fraction
    /// (ESS / K) are flagged in the result. Default 0.1. Set to 0 to silence
    /// the flag entirely.
    pub is_low_ess_threshold: f64,
    /// How BLOQ (Below Limit of Quantification) observations are handled.
    /// See [`BloqMethod`]. Defaults to `Drop` (backward-compatible: no effect
    /// when the data has no CENS column).
    pub bloq_method: BloqMethod,
    /// Maximum CG iterations for the Steihaug subproblem solver (trust-region only).
    /// `None` (default) uses a size-adaptive budget of `ceil(sqrt(n_params)).clamp(5, n_params)`,
    /// which is 5 for typical NLME problems (n_params ≈ 7–15) and grows with model size.
    /// `Some(n)` pins the budget to `n` — set to `50` to recover the previous fixed behaviour.
    pub steihaug_max_iters: Option<usize>,
    /// If true (default), use automatically detected mu-referencing to centre
    /// ETA starting points on the current population mean at each outer step.
    /// Set to false to disable for comparison purposes.
    pub mu_referencing: bool,
    /// Number of rayon worker threads used for the per-subject parallel loops
    /// (inner EBE search, SAEM MH steps, SIR weighting, likelihood reductions).
    /// `None` (default) leaves rayon's global pool alone, which means one
    /// worker per logical CPU. `Some(n)` runs the fit inside a scoped local
    /// pool of `n` threads — so the setting is per-call, not process-wide,
    /// and different fits can use different thread counts.
    pub threads: Option<usize>,
    /// Number of independent optimizations to run from perturbed starting values.
    /// `1` (default) is a single run — no behaviour change. When `> 1`, runs are
    /// launched in parallel via rayon; the result with the lowest OFV among
    /// converged runs is returned. Start 0 always uses the exact user initials;
    /// starts 1..n are log-space or additive perturbations of size `start_sigma`.
    /// Useful for models with local minima (nonlinear elimination, full-block omega,
    /// many covariates). Nested rayon parallelism (multi-start × per-subject) is
    /// safe — rayon's work-stealing pool handles it without oversubscription.
    pub n_starts: usize,
    /// Log-space standard deviation of the perturbation applied to initial theta
    /// values for starts 1..n_starts. Log-packed thetas are multiplied by
    /// `exp(N(0, start_sigma))`; identity-packed thetas (negative lower bound)
    /// are shifted by `start_sigma * N(0,1)`. Default `0.3` (≈ 30% CV).
    pub start_sigma: f64,
    /// RNG seed for the multi-start theta perturbations. Independent of
    /// `saem_seed` so that changing the SAEM seed for SAEM convergence does
    /// not silently alter which perturbed starts are tried for FOCE multi-start
    /// runs. Default `None` falls back to `42`.
    pub multi_start_seed: Option<u64>,
    /// Name of the column in the dataset that identifies the occasion for each row.
    /// When `Some`, `read_nonmem_csv` populates `Subject::occasions` / `dose_occasions`
    /// and the inner loop estimates per-occasion kappas alongside the BSV etas.
    /// Requires at least one `kappa` declaration in the model's `[parameters]` block.
    pub iov_column: Option<String>,
    /// Optional cooperative cancellation token. When present and flipped by
    /// another thread, the outer/inner/SAEM/GN loops exit at the next safe
    /// point and `fit()` returns `Err("cancelled by user")`. Default `None`.
    pub cancel: Option<crate::cancel::CancelFlag>,
    /// Keys the user explicitly set, in the order they were applied. Populated
    /// by `parse_fit_options` / `apply_fit_option`. Used by `fit()` to warn
    /// when a key is set that the selected estimation method does not consume.
    pub user_set_keys: Vec<String>,
    /// Inner-loop gradient method. Default [`GradientMethod::Auto`] prefers
    /// AD whenever the crate was built with the `autodiff` feature and the
    /// model has an analytical PK path (`tv_fn` populated); otherwise falls
    /// back to FD. See [`GradientMethod`] for the full contract.
    pub gradient_method: GradientMethod,
    /// When `true`, write a per-iteration optimizer trace CSV to a temp file
    /// and store its path in `FitResult::trace_path`. Default: `false`.
    pub optimizer_trace: bool,
    /// Apply an additional scaling layer on top of the existing log/Cholesky
    /// parameterization so that all transformed parameters are O(1) when
    /// passed to the outer optimizer.  Scaling is mathematically transparent
    /// (identical OFV and estimates by design); it only changes the internal
    /// coordinate system seen by NLopt / BFGS / GN.  Default: `true`.
    pub scale_params: bool,
    /// Fraction of subjects allowed to have unconverged EBEs before the outer
    /// optimizer rejects the current parameter step (returns OFV = ∞).  Set to
    /// `1.0` to disable the guard (old behaviour).  Default: `0.1`.
    pub max_unconverged_frac: f64,
    /// Minimum number of observations a subject must have for its EBE to count
    /// toward `max_unconverged_frac`.  Subjects below this threshold are
    /// excluded from the convergence fraction but still run normally.
    /// Default: `2`.
    pub min_obs_for_convergence_check: u32,
    /// Enable the outer-loop stagnation guard. When `true` (default), the
    /// NLopt-based outer optimizers short-circuit once recent evals show
    /// no OFV improvement above 1e-3 over a window of `3*(n+1).max(50)`
    /// evals — letting SLSQP / L-BFGS terminate in microseconds via their
    /// own xtol/ftol instead of burning through the remaining maxeval
    /// budget at full inner-loop cost. Set to `false` to disable when
    /// you want the optimizer to run to its natural termination criterion
    /// (or to `outer_maxiter`), e.g. for debugging or for problems with
    /// very slow-but-real OFV improvements below the stagnation threshold.
    pub stagnation_guard: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: EstimationMethod::FoceI,
            methods: Vec::new(),
            outer_maxiter: 500,
            outer_gtol: 1e-6,
            inner_maxiter: 200,
            // 1e-4 matches typical NLME engines (NONMEM's default inner-loop
            // SIGDIGITS is ~3, equivalent to ~1e-3). Tighter tolerances
            // (1e-6 or 1e-8) over-converge the EBE relative to the
            // Sheiner–Beal linearisation error and force BFGS to do many
            // extra iterations per find_ebe — measured ~15x slowdown on
            // a 100-subject 2-cpt FOCEI fit when set to 1e-8 vs 1e-4,
            // with no measurable change in the final OFV. Override via
            // `inner_tol = ...` in `[fit_options]` for studies that need
            // tighter EBEs (e.g. very-small-data simulation work).
            inner_tol: 1e-4,
            run_covariance_step: true,
            interaction: true,
            verbose: true,
            optimizer: Optimizer::Slsqp,
            lbfgs_memory: 5,
            global_search: false,
            global_maxeval: 0,
            saem_n_exploration: 150,
            saem_n_convergence: 250,
            saem_n_mh_steps: 3,
            saem_adapt_interval: 50,
            saem_seed: None,
            saem_n_leapfrog: 0,
            gn_lambda: 0.01,
            sir: false,
            sir_samples: 1000,
            sir_resamples: 250,
            sir_seed: None,
            sir_keep_samples: false,
            sir_df: 5.0,
            is_samples: 1000,
            is_proposal_df: 5.0,
            is_seed: None,
            is_low_ess_threshold: 0.1,
            bloq_method: BloqMethod::Drop,
            steihaug_max_iters: None,
            mu_referencing: true,
            threads: None,
            n_starts: 1,
            start_sigma: 0.3,
            multi_start_seed: None,
            iov_column: None,
            cancel: None,
            user_set_keys: Vec::new(),
            gradient_method: GradientMethod::default(),
            optimizer_trace: false,
            scale_params: true,
            max_unconverged_frac: 0.1,
            min_obs_for_convergence_check: 2,
            stagnation_guard: true,
        }
    }
}

/// BLOQ (Below Limit of Quantification) handling.
///
/// `Drop` — CENS rows are kept as ordinary observations (no special treatment). If
/// the dataset has no CENS column, every row is treated as quantified and this is
/// equivalent to the pre-M3 behavior.
///
/// `M3` — Beal's M3 method: each BLOQ observation contributes
/// `P(y < LLOQ | θ,η) = Φ((LLOQ - f)/√V)` to the likelihood instead of a
/// Gaussian residual term. LLOQ is read from DV on CENS=1 rows (NONMEM convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BloqMethod {
    Drop,
    M3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Bfgs,
    Lbfgs,
    /// NLopt LD_SLSQP — Sequential Least Squares Programming. Gradient-based,
    /// default outer optimizer. Robust on the FOCE/FOCEI objective surface in
    /// practice: FD-gradient noise from EBE re-estimation is small relative
    /// to the OFV moves the optimizer cares about.
    Slsqp,
    /// NLopt LD_LBFGS
    NloptLbfgs,
    /// NLopt LD_MMA — Method of Moving Asymptotes
    Mma,
    /// NLopt LN_BOBYQA — derivative-free quadratic interpolation. Useful when
    /// FD gradients are unreliable (e.g. small datasets where EBE-loop noise
    /// dominates per-eval signal). Needs more outer evaluations than SLSQP
    /// because it must triangulate a quadratic from scratch.
    Bobyqa,
    /// Newton trust-region with Steihaug CG subproblem (via argmin)
    TrustRegion,
}

/// Estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    Foce,
    FoceI,
    FoceGn,
    FoceGnHybrid,
    Saem,
    /// Importance-sampling marginal log-likelihood. Not an estimator — does not
    /// update parameters. Must follow another stage in `methods` (consumes that
    /// stage's params + EBEs + per-subject Hessians as the proposal centre /
    /// scale) and is typically terminal. Reports `−2 log L_IS` on
    /// `FitResult.importance_sampling`, distinct from the Laplace OFV on `ofv`.
    Imp,
}

impl EstimationMethod {
    pub fn label(self) -> &'static str {
        match self {
            EstimationMethod::Foce => "FOCE",
            EstimationMethod::FoceI => "FOCEI",
            EstimationMethod::FoceGn => "FOCE-GN",
            EstimationMethod::FoceGnHybrid => "FOCE-GN-Hybrid",
            EstimationMethod::Saem => "SAEM",
            EstimationMethod::Imp => "IMP",
        }
    }
}

impl FitOptions {
    /// Returns the sequence of methods to execute. If `methods` is non-empty it
    /// is returned as-is; otherwise a single-element chain wrapping `method`.
    pub fn method_chain(&self) -> Vec<EstimationMethod> {
        if self.methods.is_empty() {
            vec![self.method]
        } else {
            self.methods.clone()
        }
    }

    /// Check `user_set_keys` against the selected method chain. Returns one
    /// warning per key that isn't consumed by any method in the chain, listing
    /// the method-specific keys that *are* applicable so the user can correct
    /// the mistake. Framework-level keys (covariance/verbose/sir/bloq/threads/
    /// mu_referencing) are omitted from the suggestion list — they apply to
    /// every method and are exposed as top-level arguments in the wrappers.
    pub fn unsupported_keys_warnings(&self) -> Vec<String> {
        if self.user_set_keys.is_empty() {
            return Vec::new();
        }
        let chain = self.method_chain();
        // Applicability = framework keys ∪ (method-specific keys for each
        // stage in the chain). A key is legit as long as *some* stage
        // consumes it.
        let mut applicable: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        applicable.extend(framework_keys().iter().copied());
        for &m in &chain {
            applicable.extend(method_specific_keys(m).iter().copied());
        }
        // Only method-specific keys get surfaced as "available" — listing
        // framework keys here would conflate the two layers.
        let mut method_only: std::collections::BTreeSet<&'static str> =
            std::collections::BTreeSet::new();
        for &m in &chain {
            method_only.extend(method_specific_keys(m).iter().copied());
        }
        let chain_label: String = if chain.len() == 1 {
            chain[0].label().to_string()
        } else {
            chain
                .iter()
                .map(|m| m.label())
                .collect::<Vec<_>>()
                .join(" → ")
        };
        let available: Vec<&'static str> = method_only.iter().copied().collect();

        let mut seen = std::collections::HashSet::new();
        let mut warnings = Vec::new();
        for key in &self.user_set_keys {
            // `method` / `methods` select the chain itself — they can't be
            // "wrong for the method" in the way other options can.
            if key == "method" || key == "methods" {
                continue;
            }
            if applicable.contains(key.as_str()) {
                continue;
            }
            if !seen.insert(key.clone()) {
                continue;
            }
            warnings.push(format!(
                "fit option `{}` is not used by method `{}` and will be ignored. \
                 Method-specific options for `{}`: {}",
                key,
                chain_label,
                chain_label,
                available.join(", ")
            ));
        }
        warnings
    }
}

/// Framework-level fit-option keys: consumed by every method and typically
/// exposed as dedicated top-level arguments in the language wrappers
/// (`covariance`, `verbose`, `bloq_method`, `threads`, `sir`, ...). Kept
/// separate from `method_specific_keys` so the "unsupported option" warning
/// can list only method-specific suggestions without conflating the layers.
pub fn framework_keys() -> &'static [&'static str] {
    &[
        "covariance",
        "verbose",
        "sir",
        "sir_samples",
        "sir_resamples",
        "sir_seed",
        "sir_keep_samples",
        "sir_df",
        "bloq_method",
        "bloq",
        "mu_referencing",
        "threads",
        "n_starts",
        "start_sigma",
        "multi_start_seed",
        "gradient",
        "gradient_method",
        "iov_column",
        "optimizer_trace",
        "scale_params",
        "max_unconverged_frac",
        "min_obs_for_convergence_check",
    ]
}

/// Fit-option keys that are meaningful only for a particular estimation
/// method (or family of methods). `method` / `methods` are omitted — those
/// select the chain itself and can't be "wrong for the method". Framework-
/// wide keys live in `framework_keys`.
pub fn method_specific_keys(m: EstimationMethod) -> &'static [&'static str] {
    match m {
        EstimationMethod::Foce | EstimationMethod::FoceI => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "optimizer",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
        ],
        EstimationMethod::FoceGn => &["maxiter", "inner_maxiter", "inner_tol", "gn_lambda"],
        EstimationMethod::FoceGnHybrid => &[
            "maxiter",
            "inner_maxiter",
            "inner_tol",
            "optimizer",
            "steihaug_max_iters",
            "global_search",
            "global_maxeval",
            "stagnation_guard",
            "gn_lambda",
        ],
        EstimationMethod::Saem => &[
            "inner_maxiter",
            "inner_tol",
            "n_exploration",
            "n_convergence",
            "n_mh_steps",
            "n_leapfrog",
            "saem_n_leapfrog",
            "adapt_interval",
            "seed",
            "saem_seed",
        ],
        EstimationMethod::Imp => &[
            "is_samples",
            "is_proposal_df",
            "is_seed",
            "is_low_ess_threshold",
        ],
    }
}

/// Trial design specification parsed from [simulation] block
#[derive(Debug, Clone)]
pub struct SimulationSpec {
    pub n_subjects: usize,
    pub dose_amt: f64,
    pub dose_cmt: usize,
    pub obs_times: Vec<f64>,
    pub seed: u64,
    /// Optional per-subject covariates: (name, values) — length must equal n_subjects
    pub covariates: Vec<(String, Vec<f64>)>,
}

/// Full parsed model including simulation spec and fit options
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    pub fit_options: FitOptions,
}

/// Factories that build minimal `CompiledModel` instances for unit tests.
/// Exposed `pub(crate)` (gated on `#[cfg(test)]`) so other modules' tests
/// can construct models without duplicating the boilerplate.
#[cfg(test)]
pub(crate) mod test_helpers {
    use super::*;
    use std::collections::HashMap;

    /// Build an analytical-PK model (`tv_fn = Some`, `ode_spec = None`).
    pub(crate) fn analytical_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(false, gradient_method)
    }

    /// Build an ODE-backed model (`tv_fn = None`, `ode_spec = Some`).
    pub(crate) fn ode_model(gradient_method: GradientMethod) -> CompiledModel {
        make_compiled_model(true, gradient_method)
    }

    fn make_compiled_model(with_ode: bool, gradient_method: GradientMethod) -> CompiledModel {
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptOral,
            error_model: ErrorModel::Additive,
            pk_param_fn: Box::new(|_, _, _| PkParams::default()),
            n_theta: 1,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 0,
            theta_names: vec!["CL".into()],
            eta_names: vec!["ETA_CL".into()],
            kappa_names: Vec::new(),
            indiv_param_names: vec!["CL".into()],
            default_params: ModelParameters {
                theta: vec![1.0],
                theta_names: vec!["CL".into()],
                theta_lower: vec![0.0],
                theta_upper: vec![f64::INFINITY],
                theta_fixed: vec![false],
                omega: OmegaMatrix::from_diagonal(&[0.1], vec!["ETA_CL".into()]),
                omega_fixed: vec![false],
                sigma: SigmaVector {
                    values: vec![0.1],
                    names: vec!["EPS".into()],
                },
                sigma_fixed: vec![false],
                omega_iov: None,
                kappa_fixed: Vec::new(),
            },
            omega_init_as_sd: vec![false],
            sigma_init_as_sd: vec![false],
            kappa_init_as_sd: Vec::new(),
            mu_refs: HashMap::new(),
            kappa_mu_refs: HashMap::new(),
            // Analytical models populate tv_fn; ODE models leave it None.
            tv_fn: if with_ode {
                None
            } else {
                Some(Box::new(|_t, _c| vec![1.0]))
            },
            pk_indices: vec![],
            eta_map: vec![],
            pk_idx_f64: vec![],
            sel_flat: vec![],
            diffusion_theta_start: None,
            diffusion_state_indices: Vec::new(),
            ode_spec: if with_ode {
                Some(crate::ode::OdeSpec {
                    rhs: Box::new(|_y, _p, _t, _dy| {}),
                    n_states: 2,
                    state_names: vec!["depot".into(), "central".into()],
                    readout: crate::ode::OdeReadout::ObsCmt(0),
                    diffusion_var: Vec::new(),
                })
            } else {
                None
            },
            bloq_method: BloqMethod::Drop,
            referenced_covariates: vec![],
            gradient_method,
            parse_warnings: Vec::new(),
            eta_param_info: Vec::new(),
            theta_transform: Vec::new(),
            #[cfg(feature = "nn")]
            covariate_nns: Vec::new(),
            scaling: ScalingSpec::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_ode_based_false_for_analytical() {
        let m = test_helpers::analytical_model(GradientMethod::Auto);
        assert!(!m.is_ode_based());
    }

    #[test]
    fn is_ode_based_true_for_ode() {
        let m = test_helpers::ode_model(GradientMethod::Auto);
        assert!(m.is_ode_based());
    }

    #[test]
    fn test_lagtime_name_to_index_and_default() {
        assert_eq!(PkParams::name_to_index("lagtime"), Some(PK_IDX_LAGTIME));
        // NONMEM-style alias maps to the same slot.
        assert_eq!(PkParams::name_to_index("alag"), Some(PK_IDX_LAGTIME));
        assert_eq!(PK_IDX_LAGTIME, 8);
        assert_eq!(MAX_PK_PARAMS, 9);

        let default = PkParams::default();
        assert_eq!(default.lagtime(), 0.0);
        // F still defaults to 1.0 (unchanged).
        assert_eq!(default.f_bio(), 1.0);
    }

    #[test]
    fn test_lagtime_from_hashmap_primary_and_alias() {
        let mut m = HashMap::new();
        m.insert("lagtime".to_string(), 1.5);
        let p = PkParams::from_hashmap(&m);
        assert_eq!(p.lagtime(), 1.5);

        let mut m_alias = HashMap::new();
        m_alias.insert("alag".to_string(), 2.0);
        let p_alias = PkParams::from_hashmap(&m_alias);
        assert_eq!(p_alias.lagtime(), 2.0);
    }
}
