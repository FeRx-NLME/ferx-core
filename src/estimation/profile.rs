//! Profiled θ level blocks (#1065).
//!
//! **The rule.** A `profile`d block's levels are, at every outer evaluation, the exact
//! minimiser over the block of the same FOCE/FOCEI objective the unprofiled model would
//! minimise. The minimiser is found by Newton on the **total** gradient — the EBEs
//! re-solved at every point, so `dη̂/dβ` is in it — and never by a frozen-EBE shortcut:
//! "solve the levels at the current η̂, re-solve η̂, repeat" converges to a point 0.05 OFV
//! per study above the minimum once the prediction is nonlinear in η (the M3 probe on
//! #1065). So the OFV, the structural θ and their standard errors are those of the
//! unprofiled fit, and the levels never enter the optimizer's vector.
//!
//! Three pieces, all engine-agnostic (no new derivative code):
//!
//! - [`ProfileCtx::solve`]: the Newton solve. Its matrix is the subject objective's
//!   Hessian in the levels **with η̂, H and κ̂ held** — exactly `XᵀC̃⁻¹X`, since a profiled
//!   block enters the prediction affinely with a data-only coefficient (the parser's
//!   `E_PROFILE_NOT_LINEAR`) and the residual variance does not read it
//!   (`E_PROFILE_SCOPE`). `C̃` therefore comes from the objective's own builder
//!   (`subject_nll`), never a re-derivation. The gradient is a central difference of the
//!   subject objective over the subject's own levels ([`crate::api::active_levels`]), the
//!   EBE re-solved at each point. On a model linear in (η, κ) the Newton step is the exact
//!   GLS solution and the solve takes one step; otherwise a few.
//! - [`optimize_profiled`]: the outer loop over the structural parameters only. Its
//!   gradient is the production outer gradient at `β̂`, with the levels held the way
//!   `FIX` holds a θ: by the envelope theorem `dO_p/dθ = ∂O/∂θ |_{β = β̂}`, because
//!   `∇_β O = 0` there.
//! - [`ProfileCtx::covariance`]: `covariance_method = r` as the FD Hessian of the
//!   **profile** objective over the structural parameters, re-profiling the levels at
//!   every stencil point — the Schur complement `A − BD⁻¹Bᵀ` of the full Hessian — and
//!   the level blocks of the full `R⁻¹` from `D = ∇²_ββ O` and `B = ∇²_θβ O`, both FD per
//!   subject over its active levels.
//!
//! The levels are held through a **driver-local** template whose `theta_fixed` marks
//! them, never through the fit's: `n_parameters`, BIC and `fixed:` in the output read the
//! fit's template, where the levels are estimated parameters (#2103 review #9).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

use crate::estimation::covariance::{fd_ofv_hessian, CovStepOutcome};
use crate::estimation::inner_optimizer::{
    find_ebe_for_stage, run_inner_loop_warm_seeded, InnerHessianSeed, InnerLoopStats,
};
use crate::estimation::outer_optimizer::{
    drive_polish_legs, ebe_guard_rejects, population_gradient, resolve_outer_ftol,
    runs_outer_optimizer, subject_nll, OuterFdDeclineLog, OuterResult, OuterTrial, PolishEnd,
    MAX_POLISH_LEGS,
};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_mu_k, fd_step, fd_step_floor, fd_step_floors, pack_with_bounds,
    packed_fixed_mask, theta_packs_log, unpack_params, PackedBounds, PackedStart,
};
use crate::parser::model_parser::LevelRule;
use crate::types::{
    CompiledModel, CovarianceMethod, FitOptions, ModelParameters, Optimizer, Population,
};

/// Newton steps a profile solve may take before the evaluation is declared failed. A
/// model linear in (η, κ) takes one; the nonlinear F2 fixture of #1065 takes three.
pub(crate) const MAX_PROFILE_ITER: usize = 50;

/// Stop when every level's total gradient, in units of its own conditional standard
/// error (`|g_k| / sqrt(M_kk)`, the distance to the optimum measured in SEs), is below
/// this. Dimensionless, so it means the same on every data scale.
///
/// Set from the measured floor of the FD gradient, not below it. Forcing the solve past
/// convergence on F1 (#1065) and watching the criterion at the optimum, it wanders between
/// 3e-8 and 9.6e-7 from step to step (the re-solved EBEs' noise at
/// [`PROFILE_INNER_TOL`]); a tolerance below that makes the loop spin to
/// [`MAX_PROFILE_ITER`]. `1e-5` is 10× the worst. At that residual the OFV is off by
/// ~`½·L·1e-10` and the envelope gradient by a first-order term measured at 3e-5 relative
/// (`profile_tests`, G1).
pub(crate) const PROFILE_TOL: f64 = 1e-5;

/// The inner EBE tolerance the profile solve uses when the fit's `inner_tol` is looser.
/// The total gradient differences the re-solved objective at a step of `FD_STEP_SE`
/// standard errors; a warm-started inner solve that already met a loose tolerance would
/// barely move η̂, and the difference would drift toward the frozen-EBE partial the
/// profile exists to avoid. Measured on F1: at `1e-9` 37% of the re-solves ended in the
/// Nelder–Mead fallback (the BFGS stop is unreachable there), which was most of the
/// solve's time; at `1e-7` 0.4% did, with the same gradient floor.
pub(crate) const PROFILE_INNER_TOL: f64 = 1e-7;

/// The central-difference step of the total gradient, in conditional standard errors.
const FD_STEP_SE: f64 = 1e-2;

/// The FD step of the covariance step's `D` and `B`, in conditional standard errors for
/// a level and the covariance step's own `fd_hessian_step · (1 + |x|)` for a structural
/// coordinate.
const COV_STEP_SE: f64 = 1e-1;

/// The relative noise of a subject's re-solved objective, below which a backtracking
/// step is not asked to show a decrease. On F1 (#1065) the objective of a group at its
/// optimum wanders by ~1e-9 between re-solves at [`PROFILE_INNER_TOL`]; a 1e-12 floor
/// rejected every step from a point within noise of the optimum.
const OFV_NOISE: f64 = 1e-9;

/// Halvings of a Newton step before the solve gives up.
const MAX_BACKTRACK: usize = 30;

/// The solved coordinates and how the subjects share them.
#[derive(Debug, Clone)]
pub(crate) struct ProfileLayout {
    /// θ index of every coordinate the solve sets: each free θ of a `profile`d block,
    /// less any level ferx froze (#1891), which is solved around as an offset. Sorted.
    pub(crate) coords: Vec<usize>,
    /// Per subject, in population order, the positions into `coords` its rows read.
    pub(crate) active: Vec<Vec<usize>>,
    /// Coordinates no two subjects couple fall into independent groups, each solved on
    /// its own: a study's levels under `sum_to_zero_within`, the whole block under a
    /// global `sum_to_zero`, whose dependent level every subject reads.
    pub(crate) groups: Vec<ProfileGroup>,
    /// `(group, index within the group)` per position of `coords`.
    local: Vec<(usize, usize)>,
}

/// One independent set of coordinates and the subjects that read them.
#[derive(Debug, Clone)]
pub(crate) struct ProfileGroup {
    /// Positions into [`ProfileLayout::coords`], sorted.
    pub(crate) coords: Vec<usize>,
    /// Subject indices, sorted.
    pub(crate) subjects: Vec<usize>,
}

impl ProfileLayout {
    /// The layout of `model`'s profiled blocks on `population` (level columns written),
    /// or `None` when the model profiles nothing. A level `template` FIXes is left out.
    pub(crate) fn build(
        model: &CompiledModel,
        population: &Population,
        template: &ModelParameters,
    ) -> Option<Self> {
        let blocks = &model.indiv_param_partials.theta_blocks;
        let active_levels = crate::api::active_levels(model, population);
        if active_levels.is_empty() {
            return None;
        }
        let frozen = |k: usize| template.theta_fixed.get(k).copied().unwrap_or(false);
        let n_subj = population.subjects.len();
        let mut theta_sets: Vec<Vec<usize>> = vec![Vec::new(); n_subj];
        for al in &active_levels {
            let decl = blocks.decls.iter().find(|d| d.name == al.block)?;
            for (i, levels) in al.per_subject.iter().enumerate() {
                for &l in levels {
                    let range = match decl.spec.levels[l] {
                        LevelRule::Free(k) => k as usize..k as usize + 1,
                        LevelRule::NegSum(a, b) => a as usize..b as usize,
                    };
                    theta_sets[i].extend(range.filter(|&k| !frozen(k)));
                }
            }
        }
        let mut coords: Vec<usize> = theta_sets.iter().flatten().copied().collect();
        coords.sort_unstable();
        coords.dedup();
        let position = |k: usize| coords.binary_search(&k).expect("a collected θ index");
        let active: Vec<Vec<usize>> = theta_sets
            .iter()
            .map(|s| {
                let mut p: Vec<usize> = s.iter().map(|&k| position(k)).collect();
                p.sort_unstable();
                p.dedup();
                p
            })
            .collect();

        // Union-find over positions: a subject couples every coordinate it reads.
        let mut parent: Vec<usize> = (0..coords.len()).collect();
        fn find(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for set in &active {
            for w in set.windows(2) {
                let (ra, rb) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut root_group: std::collections::HashMap<usize, usize> = Default::default();
        let mut groups: Vec<ProfileGroup> = Vec::new();
        let mut local = vec![(0, 0); coords.len()];
        for (p, slot) in local.iter_mut().enumerate() {
            let r = find(&mut parent, p);
            let g = *root_group.entry(r).or_insert_with(|| {
                groups.push(ProfileGroup {
                    coords: Vec::new(),
                    subjects: Vec::new(),
                });
                groups.len() - 1
            });
            *slot = (g, groups[g].coords.len());
            groups[g].coords.push(p);
        }
        for (i, set) in active.iter().enumerate() {
            if let Some(&p) = set.first() {
                groups[local[p].0].subjects.push(i);
            }
        }
        Some(Self {
            coords,
            active,
            groups,
            local,
        })
    }
}

/// Warm start for the next solve: the levels and each subject's η̂.
#[derive(Debug, Clone)]
pub(crate) struct ProfileWarm {
    pub(crate) beta: Vec<f64>,
    pub(crate) etas: Option<Vec<DVector<f64>>>,
}

impl ProfileWarm {
    /// The template's own level values, and a cold inner start.
    pub(crate) fn from_template(layout: &ProfileLayout, template: &ModelParameters) -> Self {
        Self {
            beta: layout.coords.iter().map(|&k| template.theta[k]).collect(),
            etas: None,
        }
    }

    fn from_solve(layout: &ProfileLayout, s: &ProfileSolve) -> Self {
        Self {
            beta: layout.coords.iter().map(|&k| s.params.theta[k]).collect(),
            etas: Some(s.etas.clone()),
        }
    }
}

/// A converged profile solve at one structural point.
#[derive(Debug, Clone)]
pub(crate) struct ProfileSolve {
    /// The structural parameters with the solved levels written into `theta`.
    pub(crate) params: ModelParameters,
    pub(crate) etas: Vec<DVector<f64>>,
    pub(crate) h_matrices: Vec<DMatrix<f64>>,
    pub(crate) kappas: Vec<Vec<DVector<f64>>>,
    /// `2·Σ nllᵢ` at `params`, the same sum `pop_nll` takes.
    pub(crate) ofv: f64,
    /// The most Newton steps any group took.
    pub(crate) iterations: usize,
    /// The largest stopping criterion over the groups at exit (≤ [`PROFILE_TOL`]).
    pub(crate) criterion: f64,
    /// Per coordinate, `1 / sqrt(M_kk)`: the conditional standard error the step sizes
    /// are measured in.
    pub(crate) se_scale: Vec<f64>,
    /// The EBE convergence of this evaluation, for the outer loop's guard and published
    /// counts (#2130 review #2): the population inner loop at the start levels, plus every
    /// subject whose final mode came from a re-solve that did not converge or fell back to
    /// Nelder–Mead.
    pub(crate) ebe: InnerLoopStats,
}

/// Why a profile solve did not return a point. The outer evaluation is then
/// non-finite, the convention a failed inner solve follows: never silently accepted.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProfileFailure {
    /// [`MAX_PROFILE_ITER`] steps without meeting [`PROFILE_TOL`].
    NotConverged { criterion: f64 },
    /// The Newton matrix of a group is not positive definite: some level is not
    /// identified at this point.
    Singular,
    /// No step length decreased the objective.
    LineSearch,
    /// The objective was not finite at the start point.
    NonFinite,
}

/// One subject's converged inner state.
#[derive(Debug, Clone)]
struct SubjectState {
    eta: DVector<f64>,
    h: DMatrix<f64>,
    kappas: Vec<DVector<f64>>,
    nll: f64,
    /// The re-solve that produced this mode did not converge, or fell back to
    /// Nelder–Mead. `false` for a mode from the population inner loop, whose own counts
    /// are carried separately.
    unconverged: bool,
    fallback: bool,
}

/// Everything a profile solve reads.
pub(crate) struct ProfileCtx<'a> {
    pub(crate) model: &'a CompiledModel,
    pub(crate) population: &'a Population,
    pub(crate) layout: &'a ProfileLayout,
    pub(crate) options: &'a FitOptions,
    /// The inner EBE tolerance of this solve: [`PROFILE_INNER_TOL`], or tighter when the
    /// covariance step asks for it.
    pub(crate) inner_tol: f64,
}

impl ProfileCtx<'_> {
    /// Subject `i`'s objective term at `params`, its EBE re-solved from `warm`.
    fn resolve(&self, i: usize, params: &ModelParameters, warm: &DVector<f64>) -> SubjectState {
        let subject = &self.population.subjects[i];
        let mu_k = compute_mu_k(self.model, &params.theta, self.options.mu_referencing);
        let ebe = find_ebe_for_stage(
            self.model,
            subject,
            params,
            self.options.inner_maxiter,
            self.inner_tol,
            Some(warm.as_slice()),
            Some(&mu_k),
            0,
            self.options,
        );
        let nll = subject_nll(
            self.model,
            subject,
            params,
            &ebe.eta,
            &ebe.h_matrix,
            &ebe.kappas,
            self.options.interaction,
        );
        SubjectState {
            eta: ebe.eta,
            h: ebe.h_matrix,
            kappas: ebe.kappas,
            nll,
            unconverged: !ebe.converged,
            fallback: ebe.used_fallback,
        }
    }

    /// Subject `i`'s objective term at `params` with its EBE, H and κ̂ held at `s`, under
    /// `interaction` (the fit's own for the objective; `false` for the Newton matrix).
    fn held(&self, i: usize, params: &ModelParameters, s: &SubjectState, interaction: bool) -> f64 {
        subject_nll(
            self.model,
            &self.population.subjects[i],
            params,
            &s.eta,
            &s.h,
            &s.kappas,
            interaction,
        )
    }

    /// The Newton matrix of subject `i` over the positions `pos` (a subset of its active
    /// set): `XᵀC̃⁻¹X`, the Hessian in those levels of the subject's **FOCE** term
    /// (`interaction = false`) with η̂, H and κ̂ held. That term is the Sheiner–Beal
    /// marginal `(y − f(η̂) + Hη̂)ᵀ C̃⁻¹ (·) + log|C̃|`, `C̃ = HΩHᵀ + R`, so it is quadratic
    /// in the levels and a forward second difference is exact up to rounding at any step
    /// (`1 + 2k + k(k−1)/2` evaluations, where a central stencil takes `2k²`); `steps`
    /// sets it per coordinate.
    ///
    /// Not the fit's own term under FOCEI: that one is the conditional (Laplace) form, whose
    /// held-η̂ Hessian is `XᵀR⁻¹X` — the curvature with the random effects *not* free to
    /// absorb a level shift, larger than the total one. Measured on F1 (#1065): Newton on
    /// it converges linearly, ~1/30 per step, where `XᵀC̃⁻¹X` lands in one.
    fn held_hessian(
        &self,
        i: usize,
        params: &ModelParameters,
        s: &SubjectState,
        pos: &[usize],
        steps: &[f64],
    ) -> DMatrix<f64> {
        let k = pos.len();
        let coords = &self.layout.coords;
        let at = |moves: &[(usize, f64)]| {
            let mut p = params.clone();
            for &(j, d) in moves {
                p.theta[coords[pos[j]]] += d;
            }
            self.held(i, &p, s, false)
        };
        let f0 = at(&[]);
        let f1: Vec<f64> = (0..k).map(|a| at(&[(a, steps[a])])).collect();
        let mut m = DMatrix::zeros(k, k);
        for a in 0..k {
            let h = steps[a];
            m[(a, a)] = (at(&[(a, 2.0 * h)]) - 2.0 * f1[a] + f0) / (h * h);
            for b in 0..a {
                let g = steps[b];
                let v = (at(&[(a, h), (b, g)]) - f1[a] - f1[b] + f0) / (h * g);
                m[(a, b)] = v;
                m[(b, a)] = v;
            }
        }
        m
    }

    /// The total gradient of subject `i`'s term over `pos`: central differences with the
    /// EBE re-solved at every point, warm-started from the subject's current mode.
    fn total_gradient(
        &self,
        i: usize,
        params: &ModelParameters,
        s: &SubjectState,
        pos: &[usize],
        steps: &[f64],
    ) -> Vec<f64> {
        let coords = &self.layout.coords;
        pos.iter()
            .zip(steps)
            .map(|(&p, &h)| {
                let at = |d: f64| {
                    let mut q = params.clone();
                    q.theta[coords[p]] += d;
                    self.resolve(i, &q, &s.eta).nll
                };
                (at(h) - at(-h)) / (2.0 * h)
            })
            .collect()
    }

    /// Solve the profiled levels at the structural point `params` (whose level values are
    /// ignored), from `warm`.
    pub(crate) fn solve(
        &self,
        params: &ModelParameters,
        warm: &ProfileWarm,
    ) -> Result<ProfileSolve, ProfileFailure> {
        let layout = self.layout;
        let mut p = params.clone();
        for (&k, &b) in layout.coords.iter().zip(&warm.beta) {
            p.theta[k] = b;
        }
        let mu_k = compute_mu_k(self.model, &p.theta, self.options.mu_referencing);
        // The start point's EBEs through the population inner loop, so the guarded
        // multi-start (`inner_restarts`) sees every evaluation as it does unprofiled.
        let (etas, hms, start_stats, kaps) = run_inner_loop_warm_seeded(
            self.model,
            self.population,
            &p,
            self.options.inner_maxiter,
            self.inner_tol,
            warm.etas.as_deref(),
            Some(&mu_k),
            self.options.min_obs_for_convergence_check as usize,
            self.options.inner_restarts,
            InnerHessianSeed::for_options(self.options),
            self.options,
        );
        let mut states: Vec<SubjectState> = (0..self.population.subjects.len())
            .into_par_iter()
            .map(|i| {
                let mut s = SubjectState {
                    eta: etas[i].clone(),
                    h: hms[i].clone(),
                    kappas: kaps.get(i).cloned().unwrap_or_default(),
                    nll: 0.0,
                    unconverged: false,
                    fallback: false,
                };
                s.nll = self.held(i, &p, &s, self.options.interaction);
                s
            })
            .collect();
        if !states.iter().all(|s| s.nll.is_finite()) {
            return Err(ProfileFailure::NonFinite);
        }

        // Each group is independent: its subjects read only its coordinates.
        let theta0 = p.theta.clone();
        let outcomes: Vec<Result<GroupOutcome, ProfileFailure>> = layout
            .groups
            .par_iter()
            .map(|g| {
                let sub: Vec<SubjectState> =
                    g.subjects.iter().map(|&i| states[i].clone()).collect();
                self.solve_group(g, &p, &theta0, sub)
            })
            .collect();
        let mut iterations = 0;
        let mut criterion = 0.0_f64;
        let mut se_scale = vec![f64::NAN; layout.coords.len()];
        for (g, out) in layout.groups.iter().zip(outcomes) {
            let out = out?;
            iterations = iterations.max(out.iterations);
            criterion = criterion.max(out.criterion);
            for (j, &pos) in g.coords.iter().enumerate() {
                p.theta[layout.coords[pos]] = out.beta[j];
                se_scale[pos] = out.se[j];
            }
            for (&i, s) in g.subjects.iter().zip(out.states) {
                states[i] = s;
            }
        }
        let ofv = 2.0 * states.iter().map(|s| s.nll).sum::<f64>();
        let ebe = InnerLoopStats {
            n_unconverged: start_stats
                .n_unconverged
                .max(states.iter().filter(|s| s.unconverged).count()),
            n_fallback: start_stats.n_fallback + states.iter().filter(|s| s.fallback).count(),
            n_start_rejected: start_stats.n_start_rejected,
        };
        let (mut etas, mut h_matrices, mut kappas) = (Vec::new(), Vec::new(), Vec::new());
        for s in states {
            etas.push(s.eta);
            h_matrices.push(s.h);
            kappas.push(s.kappas);
        }
        Ok(ProfileSolve {
            params: p,
            etas,
            h_matrices,
            kappas,
            ofv,
            iterations,
            criterion,
            se_scale,
            ebe,
        })
    }

    /// Newton with backtracking on one group, from the levels in `theta0`.
    fn solve_group(
        &self,
        g: &ProfileGroup,
        params: &ModelParameters,
        theta0: &[f64],
        mut states: Vec<SubjectState>,
    ) -> Result<GroupOutcome, ProfileFailure> {
        let layout = self.layout;
        let n = g.coords.len();
        let mut p = params.clone();
        let mut beta: Vec<f64> = g.coords.iter().map(|&c| theta0[layout.coords[c]]).collect();
        // Each subject's active positions, and their indices within the group.
        let subj_pos: Vec<(&[usize], Vec<usize>)> = g
            .subjects
            .iter()
            .map(|&i| {
                let pos = layout.active[i].as_slice();
                (pos, pos.iter().map(|&c| layout.local[c].1).collect())
            })
            .collect();
        let mut iterations = 0;
        // The step of the Newton matrix's difference: one SE once a pass has measured
        // it, a unit step before. The term is quadratic, so either is exact up to
        // rounding; the SE step only keeps the differences on the objective's own scale.
        let mut m_steps = vec![1.0; n];
        loop {
            // The Newton matrix first: its diagonal sizes the gradient's steps.
            let m_at = |steps: &[f64], states: &[SubjectState], p: &ModelParameters| {
                let parts: Vec<DMatrix<f64>> = g
                    .subjects
                    .par_iter()
                    .zip(states.par_iter())
                    .zip(subj_pos.par_iter())
                    .map(|((&i, s), (pos, loc))| {
                        let st: Vec<f64> = loc.iter().map(|&l| steps[l]).collect();
                        self.held_hessian(i, p, s, pos, &st)
                    })
                    .collect();
                let mut m = DMatrix::zeros(n, n);
                for (part, (_, loc)) in parts.iter().zip(&subj_pos) {
                    for (a, &la) in loc.iter().enumerate() {
                        for (b, &lb) in loc.iter().enumerate() {
                            m[(la, lb)] += part[(a, b)];
                        }
                    }
                }
                m
            };
            let m = m_at(&m_steps, &states, &p);
            let se_of = |m: &DMatrix<f64>| -> Option<Vec<f64>> {
                (0..n)
                    .map(|a| {
                        let d = m[(a, a)];
                        (d.is_finite() && d > 0.0).then(|| 1.0 / d.sqrt())
                    })
                    .collect()
            };
            let se = se_of(&m).ok_or(ProfileFailure::Singular)?;
            m_steps.clone_from(&se);
            let steps: Vec<f64> = se.iter().map(|s| FD_STEP_SE * s).collect();
            let parts: Vec<Vec<f64>> = g
                .subjects
                .par_iter()
                .zip(states.par_iter())
                .zip(subj_pos.par_iter())
                .map(|((&i, s), (pos, loc))| {
                    let st: Vec<f64> = loc.iter().map(|&l| steps[l]).collect();
                    self.total_gradient(i, &p, s, pos, &st)
                })
                .collect();
            let mut grad = vec![0.0; n];
            for (part, (_, loc)) in parts.iter().zip(&subj_pos) {
                for (v, &l) in part.iter().zip(loc) {
                    grad[l] += v;
                }
            }
            if !grad.iter().all(|v| v.is_finite()) {
                return Err(ProfileFailure::NonFinite);
            }
            let criterion = grad
                .iter()
                .zip(&se)
                .map(|(g, s)| (g * s).abs())
                .fold(0.0_f64, f64::max);
            if criterion <= PROFILE_TOL {
                return Ok(GroupOutcome {
                    beta,
                    states,
                    iterations,
                    criterion,
                    se,
                });
            }
            if iterations == MAX_PROFILE_ITER {
                return Err(ProfileFailure::NotConverged { criterion });
            }
            let chol = m.clone().cholesky().ok_or(ProfileFailure::Singular)?;
            let step = -chol.solve(&DVector::from_vec(grad.clone()));
            let slope: f64 = grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();
            let o0: f64 = states.iter().map(|s| s.nll).sum();
            let mut alpha = 1.0;
            let mut accepted = None;
            for _ in 0..MAX_BACKTRACK {
                let trial: Vec<f64> = beta
                    .iter()
                    .zip(step.iter())
                    .map(|(b, s)| b + alpha * s)
                    .collect();
                let mut q = p.clone();
                for (&c, &b) in g.coords.iter().zip(&trial) {
                    q.theta[layout.coords[c]] = b;
                }
                let next: Vec<SubjectState> = g
                    .subjects
                    .par_iter()
                    .zip(states.par_iter())
                    .map(|(&i, s)| self.resolve(i, &q, &s.eta))
                    .collect();
                let o: f64 = next.iter().map(|s| s.nll).sum();
                // Armijo, with a floor at the objective's own noise: within it, the step
                // from a point already at the optimum cannot show a decrease.
                if o.is_finite() && o - o0 <= 1e-4 * alpha * slope + OFV_NOISE * (1.0 + o0.abs()) {
                    accepted = Some((trial, q, next));
                    break;
                }
                alpha *= 0.5;
            }
            let (trial, q, next) = accepted.ok_or(ProfileFailure::LineSearch)?;
            beta = trial;
            p = q;
            states = next;
            iterations += 1;
        }
    }
}

struct GroupOutcome {
    beta: Vec<f64>,
    states: Vec<SubjectState>,
    iterations: usize,
    criterion: f64,
    se: Vec<f64>,
}

/// The packed value of θ `k` at `value`, as `pack_params` would write it.
fn packed_theta(template: &ModelParameters, k: usize, value: f64) -> f64 {
    if theta_packs_log(template.theta_lower[k]) {
        value.max(1e-10).ln()
    } else {
        value
    }
}

/// The driver-local template: the fit's, with the solved levels held.
pub(crate) fn hold_template(layout: &ProfileLayout, template: &ModelParameters) -> ModelParameters {
    let mut hold = template.clone();
    for &k in &layout.coords {
        hold.theta_fixed[k] = true;
    }
    hold
}

/// The outer gradient at a solved point: the production gradient, the levels held at
/// `β̂` in `hold` (envelope theorem). Indexed over the whole packed vector; the held
/// coordinates' entries are not read.
pub(crate) fn envelope_gradient(
    ctx: &ProfileCtx<'_>,
    hold: &ModelParameters,
    x: &[f64],
    solve: &ProfileSolve,
    bounds: &PackedBounds,
    grad_idx: &mut usize,
    declines: &OuterFdDeclineLog,
) -> Vec<f64> {
    let mut x_hat = x.to_vec();
    for &k in &ctx.layout.coords {
        x_hat[k] = packed_theta(hold, k, solve.params.theta[k]);
    }
    // Off the analytic route, every evaluation's gradient reconverges the EBEs. The FD
    // route's default is the fixed-EBE gradient, which drops `dη̂/dθ`: an approximation the
    // unprofiled fit's optimizer tolerates, but the profile's L-BFGS stopped on it 0.42
    // OFV above the optimum on S1 (#1065, 100 × 8), where the reconverged gradient reaches
    // it. The envelope theorem removes `dβ̂/dθ`, not `dη̂/dθ`. The analytic gradient is
    // exact and is left alone (the schedule would replace it), and the IOV FD route
    // reconverges already.
    let analytic = crate::sens::provider::analytic_outer_gradient_for_interaction(
        ctx.model,
        ctx.options.interaction,
    );
    let opts = FitOptions {
        reconverge_gradient_interval: if analytic {
            ctx.options.reconverge_gradient_interval
        } else {
            1
        },
        ..ctx.options.clone()
    };
    population_gradient(
        &x_hat,
        ctx.population.subjects.len(),
        hold,
        ctx.model,
        ctx.population,
        &solve.etas,
        &solve.h_matrices,
        &solve.kappas,
        bounds,
        &opts,
        grad_idx,
        OuterTrial::unknown(),
        declines,
    )
}

/// The outer algorithm of a profiled fit: `auto` is L-BFGS on the envelope gradient,
/// whichever engine serves it, and the explicit NLopt choices are honoured;
/// `check_profiled_levels` refuses the rest.
///
/// Not the unprofiled rule (BOBYQA when the outer gradient is FD). That rule exists
/// because an FD gradient through a re-solved inner loop is noisy, and BOBYQA never
/// differentiates. Here the objective is a smooth, exactly re-solved profile, and its
/// gradient is checked against FD of that profile (G1, 5.5e-5 relative on the FD
/// engine). Measured on F2 (#1065): BOBYQA stopped 0.0185 OFV above NONMEM's optimum
/// after 110 evaluations, L-BFGS 1.5e-8 above it after 26 (7.9 s against 22.9 s).
///
/// `Optimizer::Bfgs` and `Optimizer::Lbfgs` — the hand-rolled loops, reachable only by
/// constructing the variant in Rust, since the parser reads `bfgs` and `lbfgs` as
/// `nlopt_lbfgs` (#483) — also run NLopt L-BFGS here, the profiled loop having no
/// hand-rolled arm (#2130 review #6). The docs say so.
fn profiled_algorithm(options: &FitOptions) -> nlopt::Algorithm {
    match options.optimizer {
        Optimizer::Bobyqa => nlopt::Algorithm::Bobyqa,
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::Auto | Optimizer::NloptLbfgs | Optimizer::Bfgs | Optimizer::Lbfgs => {
            nlopt::Algorithm::Lbfgs
        }
        // Refused by `check_profiled_levels` before a fit gets here.
        Optimizer::TrustRegion | Optimizer::Mma => nlopt::Algorithm::Lbfgs,
    }
}

/// `ftol_rel` of a profiled fit's gradient-based outer optimizer (L-BFGS, SLSQP) when
/// the user set no `outer_ftol` (#2285).
///
/// The unprofiled default, `1e-6`, is relative to the OFV, and on the F2 anchor
/// (`nonmem_anchor/profile_mbma_fit.ferx`, OFV ≈ 102) it stops L-BFGS on any step under
/// 1e-4. L-BFGS met it mid-descent: from the start (45, 7.5, 2) the trace's last steps
/// were 3.1e-4, 2.1e-5, 5.0e-5 and the run stopped `converged: true` 3.9e-4 above
/// NONMEM's optimum. #2280's start stopped 1.6e-3 above it after a 1.2e-5 step. That
/// default is set for the unprofiled objective, whose re-solved inner loop carries
/// ~1e-3 of noise. The profile objective is re-solved to [`OFV_NOISE`] (1e-9 relative),
/// so a stop it can reach sits much lower. `1e-8` is 10× that floor.
///
/// It is still a test on one iteration's change, so it does not by itself put the run
/// on the optimum: over a 27-start grid around F2's start it stopped up to 8.5e-5 short
/// (#2285's plan, M4), and #2280's start then stopped 1.6e-3 short on `xtol_rel`
/// instead (M1). Whether a stop is a minimum is decided by the verification leg
/// ([`PROFILED_VERIFY_GAIN`]), not by this tolerance.
///
/// BOBYQA keeps the unprofiled rule: it is derivative-free, and the tight stop would
/// only spend its budget.
pub(crate) const PROFILED_GRADIENT_FTOL_REL: f64 = 1e-8;

/// Ceiling on `xtol_rel` for a profiled fit's gradient-based outer optimizer (L-BFGS,
/// SLSQP), applied to every leg (#2285). NLopt's L-BFGS tests `xtol_rel` on a single
/// iteration's step, so at the unprofiled `outer_xtol = 1e-4` one short line-search
/// step ends the run: #2280's own A2 start stopped `XtolReached` 1.584e-3 above
/// NONMEM's optimum. Verification legs at that tolerance do not help either: from
/// (45, 10, 2) every leg ended on a short step and gained under
/// [`PROFILED_VERIFY_GAIN`] while still 2.5e-5 short (M9). A tighter user `outer_xtol`
/// wins, as for the #1782 polish legs. Not lower than this: at 1e-12 `main`'s L-BFGS
/// ends five of seven starts on NLopt `Failure` (M5).
pub(crate) const PROFILED_GRADIENT_XTOL_REL: f64 = 1e-6;

/// A profiled gradient-based fit is converged when a fresh L-BFGS / SLSQP leg from its
/// best point lowers the OFV by no more than this (#2285). Absolute OFV, like A2's band:
/// on F2 a leg from a stop short of the optimum gained at least 2.14e-5, and one from
/// a genuine optimum at most 6.0e-8 (the earlier S1 fixture, |OFV| 3579), so 1e-6 sits
/// 21× below the one and 17× above the other (#2285's plan, M8/M10). A leg that gains
/// more is followed by another, up to `MAX_POLISH_LEGS`.
pub(crate) const PROFILED_VERIFY_GAIN: f64 = 1e-6;

/// The `xtol_rel` [`optimize_profiled`] sets for `algo`: `outer_xtol` capped at
/// [`PROFILED_GRADIENT_XTOL_REL`] for a gradient method, `outer_xtol` for BOBYQA.
pub(crate) fn profiled_xtol(algo: nlopt::Algorithm, options: &FitOptions) -> f64 {
    match algo {
        nlopt::Algorithm::Bobyqa => options.outer_xtol,
        _ => options.outer_xtol.min(PROFILED_GRADIENT_XTOL_REL),
    }
}

#[cfg(test)]
thread_local! {
    /// Verification legs [`optimize_profiled`] ran on this thread, for the tests.
    static VERIFY_LEGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

type NloptReturn = Result<(nlopt::SuccessState, f64), (nlopt::FailState, f64)>;

/// Does a profiled run that returned `result` need a verification leg before it may be
/// called converged? Every stop of a gradient method that could be a minimum: any
/// tolerance or success status, and `Failure` / `RoundoffLimited`, which `main`'s L-BFGS
/// returns at the optimum under tight stops (M5). Not a spent budget or time, not a
/// cancelled run, not a bad argument, and not BOBYQA (#2336).
pub(crate) fn profiled_stop_needs_verify(
    algo: nlopt::Algorithm,
    result: &NloptReturn,
    cancelled: bool,
) -> bool {
    !matches!(algo, nlopt::Algorithm::Bobyqa)
        && !cancelled
        && match result {
            Ok((status, _)) => !matches!(
                status,
                nlopt::SuccessState::MaxEvalReached | nlopt::SuccessState::MaxTimeReached
            ),
            Err((fail, _)) => matches!(
                fail,
                nlopt::FailState::Failure | nlopt::FailState::RoundoffLimited
            ),
        }
}

/// The NLopt status alone as a verdict: the rule for a run no leg verified (BOBYQA, a
/// spent budget, a cancelled run).
fn status_is_converged(result: &NloptReturn) -> bool {
    matches!(
        result,
        Ok((
            nlopt::SuccessState::Success
                | nlopt::SuccessState::FtolReached
                | nlopt::SuccessState::XtolReached
                | nlopt::SuccessState::StopValReached,
            _
        ))
    )
}

/// `converged` of a profiled fit from how its verification legs ended and the last
/// leg's return. `Settled` on a verifiable stop means the last leg gained no more than
/// [`PROFILED_VERIFY_GAIN`]: converged, whatever NLopt's status, `Failure` included.
/// A leg cap, a spent budget or a leg that could not start is not converged. Otherwise
/// (nothing to verify, or the last leg ended on a budget) the status decides.
pub(crate) fn profiled_converged(
    algo: nlopt::Algorithm,
    end: PolishEnd,
    last: &NloptReturn,
    cancelled: bool,
) -> bool {
    match end {
        PolishEnd::Settled if profiled_stop_needs_verify(algo, last, cancelled) => true,
        PolishEnd::Settled | PolishEnd::NotNeeded => status_is_converged(last),
        PolishEnd::LegCapReached { .. } | PolishEnd::BudgetSpent | PolishEnd::Skipped => false,
    }
}

/// The `ftol_rel` [`optimize_profiled`] sets for `algo`: the user's `outer_ftol` when
/// set, else [`PROFILED_GRADIENT_FTOL_REL`] for a gradient method and the unprofiled
/// `resolve_outer_ftol` for BOBYQA.
pub(crate) fn profiled_ftol(
    algo: nlopt::Algorithm,
    model: &CompiledModel,
    options: &FitOptions,
) -> f64 {
    match (options.outer_ftol, algo) {
        (Some(user), _) => user,
        (None, nlopt::Algorithm::Bobyqa) => {
            resolve_outer_ftol(model.has_non_gaussian(), model.is_ode_based(), None)
        }
        (None, _) => PROFILED_GRADIENT_FTOL_REL,
    }
}

/// The outer loop of a fit with `profile`d blocks: NLopt over the structural packed
/// coordinates, every evaluation a profile solve. Returns the same `OuterResult` the
/// unprofiled path does, its `params` carrying the solved levels and the fit's own
/// `theta_fixed`.
pub(crate) fn optimize_profiled(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let layout = ProfileLayout::build(model, population, init_params)
        .expect("optimize_profiled is called for a model with a profiled block");
    let hold = hold_template(&layout, init_params);
    let PackedStart {
        packed: mut x0,
        bounds,
        fixed,
        ..
    } = pack_with_bounds(&hold);
    clamp_to_bounds(&mut x0, &bounds);
    let free: Vec<usize> = (0..x0.len()).filter(|&i| !fixed[i]).collect();
    let ctx = ProfileCtx {
        model,
        population,
        layout: &layout,
        options,
        inner_tol: options.inner_tol.min(PROFILE_INNER_TOL),
    };
    let declines = OuterFdDeclineLog::new(population.subjects.len());
    let mut warnings: Vec<String> = Vec::new();

    // The incumbent: its solve warm-starts the next evaluation (#1290: a warm start from
    // a rejected trial can stall a line search).
    struct Incumbent {
        x: Vec<f64>,
        solve: Option<ProfileSolve>,
        warm: ProfileWarm,
    }
    let best = Mutex::new(Incumbent {
        x: x0.clone(),
        solve: None,
        warm: ProfileWarm::from_template(&layout, init_params),
    });
    let n_evals = AtomicUsize::new(0);
    let grad_idx = Mutex::new(0usize);
    let full = |xs: &[f64]| {
        let mut x = x0.clone();
        for (&i, &v) in free.iter().zip(xs) {
            x[i] = v;
        }
        x
    };
    let ebe_accum = Mutex::new(EbeAccum::default());
    let evaluate = |x: &[f64]| -> Option<ProfileSolve> {
        n_evals.fetch_add(1, Ordering::Relaxed);
        let params = unpack_params(x, &hold);
        let warm = best.lock().unwrap().warm.clone();
        let solve = ctx.solve(&params, &warm).ok()?;
        {
            let mut acc = ebe_accum.lock().unwrap();
            acc.max_unconverged = acc.max_unconverged.max(solve.ebe.n_unconverged);
            acc.total_fallback += solve.ebe.n_fallback;
            // The unprofiled loop's EBE guard, on the same counts: too many unconverged
            // subjects (or one hard-rejected at its start) and the trial is a wall.
            if ebe_guard_rejects(
                &solve.ebe,
                population.subjects.len(),
                solve.ofv,
                options.max_unconverged_frac,
            ) {
                acc.n_guarded += 1;
                return None;
            }
        }
        let mut b = best.lock().unwrap();
        if b.solve.as_ref().is_none_or(|s| solve.ofv < s.ofv) {
            b.x = x.to_vec();
            b.warm = ProfileWarm::from_solve(&layout, &solve);
            b.solve = Some(solve.clone());
        }
        Some(solve)
    };

    let mut converged = false;
    if runs_outer_optimizer(options) && !free.is_empty() {
        let algo = profiled_algorithm(options);
        let lower: Vec<f64> = free.iter().map(|&i| bounds.lower[i]).collect();
        let upper: Vec<f64> = free.iter().map(|&i| bounds.upper[i]).collect();
        let objective = |xs: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
            if crate::cancel::is_cancelled(&options.cancel) {
                if let Some(g) = grad {
                    g.fill(0.0);
                }
                return 1e20;
            }
            let x = full(xs);
            match evaluate(&x) {
                Some(solve) => {
                    if let Some(g) = grad {
                        let mut idx = grad_idx.lock().unwrap();
                        let gr = envelope_gradient(
                            &ctx, &hold, &x, &solve, &bounds, &mut idx, &declines,
                        );
                        for (gi, &i) in g.iter_mut().zip(&free) {
                            *gi = if gr[i].is_finite() { gr[i] } else { 0.0 };
                        }
                    }
                    if options.verbose {
                        eprintln!(
                            "Eval {:>4}: OFV = {:.6} (profile: {} Newton step{}, gradient {:.1e} SE)",
                            n_evals.load(Ordering::Relaxed),
                            solve.ofv,
                            solve.iterations,
                            if solve.iterations == 1 { "" } else { "s" },
                            solve.criterion
                        );
                    }
                    solve.ofv
                }
                None => {
                    // A failed profile solve is a wall, as a failed inner solve is; the
                    // gradient pushes back toward the box centre.
                    if let Some(g) = grad {
                        for (k, gi) in g.iter_mut().enumerate() {
                            *gi = 100.0 * (xs[k] - 0.5 * (lower[k] + upper[k]));
                        }
                    }
                    1e20
                }
            }
        };
        let mut opt = nlopt::Nlopt::new(algo, free.len(), &objective, nlopt::Target::Minimize, ());
        opt.set_lower_bounds(&lower).unwrap();
        opt.set_upper_bounds(&upper).unwrap();
        let per_iter = free.len() as u32 + 1;
        let mut budget = (options.outer_maxiter as u32).saturating_mul(per_iter);
        if matches!(algo, nlopt::Algorithm::Bobyqa) {
            budget = budget.saturating_add(40 * per_iter);
            let step: Vec<f64> = (0..free.len())
                .map(|k| 0.5_f64.min(((upper[k] - lower[k]).abs() * 0.5).max(1e-6)))
                .collect();
            opt.set_initial_step(&step).unwrap();
        }
        opt.set_maxeval(budget.max(1)).unwrap();
        let xtol = profiled_xtol(algo, options);
        let ftol = profiled_ftol(algo, model, options);
        opt.set_xtol_rel(xtol).unwrap();
        opt.set_ftol_rel(ftol).unwrap();
        let mut xs: Vec<f64> = free.iter().map(|&i| x0[i]).collect();
        let first = opt.optimize(&mut xs);
        if options.verbose {
            eprintln!("Profiled outer optimizer finished: {first:?}");
        }
        let cancelled = || crate::cancel::is_cancelled(&options.cancel);
        // #2285: a gradient method's stop is not evidence of a minimum — `ftol_rel` and
        // `xtol_rel` are tests on one iteration, which L-BFGS meets mid-descent, and
        // `main`'s L-BFGS returns `Failure` at a genuine optimum. So run a fresh leg from
        // the best point at the same stops; the fit is converged when a leg gains no
        // more than `PROFILED_VERIFY_GAIN`. Every leg feeds the same incumbent and shares
        // the budget, the #1782 loop.
        let (result, end) = drive_polish_legs(
            first,
            PROFILED_VERIFY_GAIN,
            |r| profiled_stop_needs_verify(algo, r, cancelled()),
            || budget.saturating_sub(n_evals.load(Ordering::Relaxed) as u32),
            |left| {
                let (bx, before) = {
                    let b = best.lock().unwrap();
                    (b.x.clone(), b.solve.as_ref()?.ofv)
                };
                let mut leg =
                    nlopt::Nlopt::new(algo, free.len(), &objective, nlopt::Target::Minimize, ());
                leg.set_lower_bounds(&lower).unwrap();
                leg.set_upper_bounds(&upper).unwrap();
                leg.set_maxeval(left).unwrap();
                leg.set_xtol_rel(xtol).unwrap();
                leg.set_ftol_rel(ftol).unwrap();
                let mut lx: Vec<f64> = free.iter().map(|&i| bx[i]).collect();
                let r = leg.optimize(&mut lx);
                #[cfg(test)]
                VERIFY_LEGS.with(|c| c.set(c.get() + 1));
                let after = best
                    .lock()
                    .unwrap()
                    .solve
                    .as_ref()
                    .map_or(before, |s| s.ofv);
                if options.verbose {
                    eprintln!(
                        "Profiled verification leg from OFV {before:.9}: {r:?}, gain {:.3e}",
                        before - after
                    );
                }
                Some((r, before - after))
            },
        );
        converged = profiled_converged(algo, end, &result, cancelled());
        match end {
            PolishEnd::LegCapReached { last_gain } => warnings.push(format!(
                "Profiled outer optimizer was still improving after {MAX_POLISH_LEGS} \
                 verification legs (the last lowered the objective by {last_gain:.3e}), so \
                 it is not reported as converged; re-run from these estimates."
            )),
            PolishEnd::BudgetSpent => warnings.push(
                "Profiled outer optimizer spent its evaluation budget before its stop was \
                 verified; increase maxiter."
                    .to_string(),
            ),
            _ if !converged => {
                let status = match &result {
                    Ok((s, _)) => format!("{s:?}"),
                    Err((f, _)) => format!("{f:?}"),
                };
                warnings.push(format!(
                    "Profiled outer optimizer stopped without converging ({status})."
                ));
            }
            _ => {}
        }
    } else {
        // Evaluation only (`maxiter = 0`, NONMEM `MAXEVAL=0`), or nothing to search: the
        // levels are still solved at the initial structural point.
        evaluate(&x0);
    }

    let (x_best, solve) = {
        let b = best.lock().unwrap();
        (b.x.clone(), b.solve.clone())
    };
    let Some(solve) = solve else {
        warnings.push(
            "Profile solve failed at every evaluated point: no objective is available.".to_string(),
        );
        let params = unpack_params(&x0, init_params);
        let ebe = ebe_accum.into_inner().unwrap();
        return failed_result(params, population, model, warnings, &ebe);
    };
    let mut x_hat = x_best.clone();
    for &k in &layout.coords {
        x_hat[k] = packed_theta(&hold, k, solve.params.theta[k]);
    }

    let cov = if options.run_covariance_step {
        let out = ctx.covariance_at_estimate(&hold, &x_hat, &free, &solve.params, &solve.etas);
        warnings.extend(out.warnings.iter().cloned());
        Some(out)
    } else {
        None
    };
    // #1303: the verdict a consumer keys on is gated on the objective it describes.
    if let Some(w) =
        crate::estimation::outer_optimizer::gate_converged_on_objective(&mut converged, solve.ofv)
    {
        warnings.push(w);
    }
    let mut params = solve.params.clone();
    params.theta_fixed = init_params.theta_fixed.clone();
    let ebe = ebe_accum.into_inner().unwrap();
    OuterResult {
        params,
        ofv: solve.ofv,
        converged,
        n_iterations: n_evals.load(Ordering::Relaxed),
        eta_hats: solve.etas,
        h_matrices: solve.h_matrices,
        kappas: solve.kappas,
        covariance_matrix: cov.as_ref().and_then(|c| c.matrix.clone()),
        covariance_method: cov.as_ref().and_then(|c| c.method),
        covariance_wall_time_secs: cov.as_ref().map_or(0.0, |c| c.wall_time_secs),
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: ebe.n_guarded as u32,
        max_unconverged_subjects: ebe.max_unconverged as u32,
        total_ebe_fallbacks: ebe.total_fallback as u32,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal: None,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: Some(x_hat),
        left_init: None,
        mixture_posteriors: None,
        vi: None,
    }
}

/// The warning `covariance_at_estimate` returns when the profile solve at the estimates
/// fails, on the inline `fit()` step and on `replay_covariance` alike.
/// A function rather than an inline `format!` so the `classify_warning` table test
/// (`types_tests.rs`) classifies the text this site emits, not a copy of it (#2158).
pub(crate) fn profile_solve_failed_msg(e: &ProfileFailure) -> String {
    format!(
        "Covariance step failed: the profile solve did not converge at the \
         estimates ({e:?}). SE estimates not available."
    )
}

/// The covariance step of a profiled fit, re-run on its estimates (`run_covariance`), or
/// `None` when `model` profiles nothing. The levels are re-profiled from the fit's own
/// values first, so a reloaded fit gets the step the fit ran — not the unprofiled `R`
/// over every level, which at a few hundred levels is the stencil profiling exists to
/// avoid. As in `fit()`, only `r` is computed (#2108).
///
/// `x_hat` is the packed point the step runs at — the fit's own packed vector when
/// `run_covariance` can reuse it — not a repack of `params`, which can differ in the last
/// bits through the log and Cholesky transforms.
pub(crate) fn replay_covariance(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    x_hat: &[f64],
    eta_hats: &[DVector<f64>],
    options: &FitOptions,
) -> Option<CovStepOutcome> {
    let layout = ProfileLayout::build(model, population, params)?;
    if options.covariance_method != CovarianceMethod::Hessian {
        return Some(CovStepOutcome {
            matrix: None,
            wall_time_secs: 0.0,
            warnings: vec![format!(
                "Covariance step not run: a profiled θ level block's covariance step \
                 computes `r` only for now (#2108); this call asks for \
                 `covariance_method = {}`.",
                options.covariance_method.label()
            )],
            sir_fallback_proposal: None,
            method: None,
        });
    }
    let hold = hold_template(&layout, params);
    let fixed = packed_fixed_mask(&hold);
    let free: Vec<usize> = (0..x_hat.len()).filter(|&i| !fixed[i]).collect();
    let ctx = ProfileCtx {
        model,
        population,
        layout: &layout,
        options,
        inner_tol: options.inner_tol.min(PROFILE_INNER_TOL),
    };
    Some(ctx.covariance_at_estimate(&hold, x_hat, &free, params, eta_hats))
}

/// EBE convergence over a profiled run, as the unprofiled NLopt loop keeps it: the worst
/// unconverged count, the fallbacks, and the evaluations the EBE guard rejected.
#[derive(Default)]
struct EbeAccum {
    max_unconverged: usize,
    total_fallback: usize,
    n_guarded: usize,
}

/// An `OuterResult` for a fit whose profile solve never succeeded.
fn failed_result(
    params: ModelParameters,
    population: &Population,
    model: &CompiledModel,
    warnings: Vec<String>,
    ebe: &EbeAccum,
) -> OuterResult {
    let n = population.subjects.len();
    OuterResult {
        params,
        ofv: f64::INFINITY,
        converged: false,
        n_iterations: 0,
        eta_hats: vec![DVector::zeros(model.n_eta); n],
        h_matrices: population
            .subjects
            .iter()
            .map(|s| DMatrix::zeros(s.obs_times.len(), model.n_eta))
            .collect(),
        kappas: vec![Vec::new(); n],
        covariance_matrix: None,
        covariance_method: None,
        covariance_wall_time_secs: 0.0,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        saem_n_subjects_hmc: None,
        saem_mh_accept_tail: None,
        ebe_convergence_warnings: ebe.n_guarded as u32,
        max_unconverged_subjects: ebe.max_unconverged as u32,
        total_ebe_fallbacks: ebe.total_fallback as u32,
        final_gradient: None,
        final_gradient_source: None,
        sir_fallback_proposal: None,
        impmap_trace: None,
        bayes: None,
        cond_dist: None,
        packed_estimate: None,
        left_init: None,
        mixture_posteriors: None,
        vi: None,
    }
}

impl ProfileCtx<'_> {
    /// The profile objective at the packed point `x` (levels ignored), re-profiled from
    /// `warm`; `NaN` when the solve fails.
    pub(crate) fn profile_ofv(&self, hold: &ModelParameters, x: &[f64], warm: &ProfileWarm) -> f64 {
        self.solve(&unpack_params(x, hold), warm)
            .map_or(f64::NAN, |s| s.ofv)
    }

    /// The covariance step at an estimate: re-profile the levels from `params`' own values
    /// and `eta_hats`, then [`Self::covariance`] at that solve. `fit()` and
    /// `replay_covariance` both go through here, so a replay handed the fit's estimates
    /// and EBEs repeats the inline step bit for bit (#2173).
    pub(crate) fn covariance_at_estimate(
        &self,
        hold: &ModelParameters,
        x: &[f64],
        free: &[usize],
        params: &ModelParameters,
        eta_hats: &[DVector<f64>],
    ) -> CovStepOutcome {
        let layout = self.layout;
        let warm = ProfileWarm {
            beta: layout.coords.iter().map(|&k| params.theta[k]).collect(),
            etas: (eta_hats.len() == self.population.subjects.len()).then(|| eta_hats.to_vec()),
        };
        let solve = match self.solve(params, &warm) {
            Ok(s) => s,
            Err(e) => {
                return CovStepOutcome {
                    matrix: None,
                    wall_time_secs: 0.0,
                    warnings: vec![profile_solve_failed_msg(&e)],
                    sir_fallback_proposal: None,
                    method: None,
                }
            }
        };
        // On a converged start `solve_group` takes no Newton step, so β̂ is unchanged and this
        // overwrite is a bitwise no-op; it matters only when the re-solve moves the levels.
        let mut x_hat = x.to_vec();
        for &k in &layout.coords {
            x_hat[k] = packed_theta(hold, k, solve.params.theta[k]);
        }
        self.covariance(hold, &x_hat, free, &solve)
    }

    /// `covariance_method = r` for a profiled fit, as the full packed covariance the
    /// unprofiled step would publish (levels included), so every downstream reader —
    /// `se_theta`, the level SEs and their delta method for a `NegSum` dependent, SIR's
    /// proposal — reads it unchanged.
    ///
    /// With `H = [[A, B], [Bᵀ, D]]` the OFV Hessian over (structural, levels):
    /// - `Cov(θ) = 2·A_p⁻¹`, `A_p = A − BD⁻¹Bᵀ` the Hessian of the profile objective,
    ///   differenced with the levels re-profiled at every stencil point. Holding them
    ///   fixed would difference `A` instead, and `2·A⁻¹` is too small.
    /// - `Cov(β) = 2·D⁻¹ + D⁻¹Bᵀ Cov(θ) B D⁻¹`, `Cov(θ, β) = −Cov(θ) B D⁻¹`: the level
    ///   blocks of `2·H⁻¹`. `D` is the exact Hessian (EBE re-solved), not the Newton
    ///   matrix; the two differ when the model is nonlinear in η.
    pub(crate) fn covariance(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
    ) -> CovStepOutcome {
        let start = Instant::now();
        let fail = |reason: String| CovStepOutcome {
            matrix: None,
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![reason],
            sir_fallback_proposal: None,
            method: None,
        };
        let options = self.options;
        let cov_ctx = ProfileCtx {
            inner_tol: options
                .effective_cov_inner_tol(self.model.uses_closed_form_ltbs_inner())
                .min(self.inner_tol),
            ..*self
        };
        let warm = ProfileWarm::from_solve(self.layout, solve);
        let f = |xv: &[f64]| cov_ctx.profile_ofv(hold, xv, &warm);
        let n = x_hat.len();
        let f0 = f(x_hat);
        if !f0.is_finite() {
            return fail(
                "Covariance step failed: the profile objective is not finite at the estimate. \
                 SE estimates not available."
                    .to_string(),
            );
        }
        let floor = fd_step_floors(hold, &self.model.default_params);
        let Some(stencil) = fd_ofv_hessian(
            n,
            x_hat,
            free,
            &floor,
            options.fd_hessian_step,
            f0,
            &f,
            options,
        ) else {
            return fail(crate::estimation::covariance::COV_CANCELLED_MSG.to_string());
        };
        if !stencil.diag_nan.is_empty() || !stencil.offdiag_nan.is_empty() {
            return fail(
                "Covariance step failed: the profile objective is not finite at some \
                 finite-difference point. SE estimates not available."
                    .to_string(),
            );
        }
        let nf = free.len();
        let a_p = DMatrix::from_fn(nf, nf, |a, b| stencil.hess[(free[a], free[b])]);
        let Some(a_chol) = a_p.clone().cholesky() else {
            return fail(
                "Covariance step failed: the Hessian of the profile objective is not positive \
                 definite. SE estimates not available."
                    .to_string(),
            );
        };
        let cov_s = a_chol.inverse() * 2.0;

        // D (per group) and B, per subject over its own levels, with the EBE re-solved.
        let (d_groups, b) = match cov_ctx.level_blocks(hold, x_hat, free, solve) {
            Ok(v) => v,
            Err(reason) => return fail(reason),
        };
        let layout = self.layout;
        let l = layout.coords.len();
        // D⁻¹ per group, and D⁻¹Bᵀ (L × nf).
        let mut d_inv_bt = DMatrix::zeros(l, nf);
        let mut d_inv: Vec<DMatrix<f64>> = Vec::with_capacity(layout.groups.len());
        for (g, d) in layout.groups.iter().zip(&d_groups) {
            let Some(ch) = d.clone().cholesky() else {
                return fail(
                    "Covariance step failed: the Hessian of the objective in the profiled \
                     levels is not positive definite. SE estimates not available."
                        .to_string(),
                );
            };
            let inv = ch.inverse();
            for (a, &pa) in g.coords.iter().enumerate() {
                for s in 0..nf {
                    d_inv_bt[(pa, s)] = (0..g.coords.len())
                        .map(|c| inv[(a, c)] * b[(s, g.coords[c])])
                        .sum();
                }
            }
            d_inv.push(inv);
        }
        // Cov(θ, β) = −Cov(θ)·B·D⁻¹ = −Cov(θ)·(D⁻¹Bᵀ)ᵀ.
        let cov_sb = -(&cov_s * d_inv_bt.transpose());
        // Cov(β) = 2·D⁻¹ + (D⁻¹Bᵀ)·Cov(θ)·(D⁻¹Bᵀ)ᵀ.
        let mut cov_bb = &d_inv_bt * &cov_s * d_inv_bt.transpose();
        for (g, inv) in layout.groups.iter().zip(&d_inv) {
            for (a, &pa) in g.coords.iter().enumerate() {
                for (c, &pc) in g.coords.iter().enumerate() {
                    cov_bb[(pa, pc)] += 2.0 * inv[(a, c)];
                }
            }
        }
        let mut full = DMatrix::zeros(n, n);
        for a in 0..nf {
            for c in 0..nf {
                full[(free[a], free[c])] = cov_s[(a, c)];
            }
            for (p, &k) in layout.coords.iter().enumerate() {
                full[(free[a], k)] = cov_sb[(a, p)];
                full[(k, free[a])] = cov_sb[(a, p)];
            }
        }
        for (p, &k) in layout.coords.iter().enumerate() {
            for (q, &j) in layout.coords.iter().enumerate() {
                full[(k, j)] = cov_bb[(p, q)];
            }
        }
        if options.verbose {
            eprintln!("  Covariance step successful (profile)");
        }
        CovStepOutcome {
            matrix: Some(full),
            wall_time_secs: start.elapsed().as_secs_f64(),
            warnings: Vec::new(),
            sir_fallback_proposal: None,
            method: Some(CovarianceMethod::Hessian),
        }
    }

    /// `D` per group (OFV scale, over the group's coordinates) and `B` (`nf × L`, OFV
    /// scale), by central differences of each subject's term over the structural free
    /// coordinates and its own levels, the EBE re-solved at every point.
    #[allow(clippy::type_complexity)]
    fn level_blocks(
        &self,
        hold: &ModelParameters,
        x_hat: &[f64],
        free: &[usize],
        solve: &ProfileSolve,
    ) -> Result<(Vec<DMatrix<f64>>, DMatrix<f64>), String> {
        let layout = self.layout;
        let nf = free.len();
        let l = layout.coords.len();
        let theta_steps: Vec<f64> = free
            .iter()
            .map(|&i| {
                let floor = fd_step_floor(hold, &self.model.default_params, i);
                fd_step(self.options.fd_hessian_step, x_hat[i], floor)
            })
            .collect();
        let level_steps: Vec<f64> = solve.se_scale.iter().map(|s| COV_STEP_SE * s).collect();
        if !level_steps.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(
                "Covariance step failed: a profiled level has no curvature at the estimate. \
                 SE estimates not available."
                    .to_string(),
            );
        }
        // Per subject: (D_i over its positions, B_i nf × its positions).
        let parts: Vec<(DMatrix<f64>, DMatrix<f64>)> = (0..self.population.subjects.len())
            .into_par_iter()
            .map(|i| {
                let pos = &layout.active[i];
                let k = pos.len();
                let warm = &solve.etas[i];
                // A packed point moved along structural coordinates `ds` and level
                // positions `dl`, scored as subject `i`'s term (OFV scale). `x_hat`
                // already carries `β̂` in the level coordinates.
                let at = |ds: &[(usize, f64)], dl: &[(usize, f64)]| -> f64 {
                    let mut x = x_hat.to_vec();
                    for &(a, d) in ds {
                        x[free[a]] += d;
                    }
                    let mut p = unpack_params(&x, hold);
                    for &(j, d) in dl {
                        p.theta[layout.coords[pos[j]]] += d;
                    }
                    2.0 * self.resolve(i, &p, warm).nll
                };
                let f0 = at(&[], &[]);
                let mut d = DMatrix::zeros(k, k);
                for a in 0..k {
                    let h = level_steps[pos[a]];
                    d[(a, a)] = (at(&[], &[(a, h)]) - 2.0 * f0 + at(&[], &[(a, -h)])) / (h * h);
                    for c in 0..a {
                        let g = level_steps[pos[c]];
                        let v = (at(&[], &[(a, h), (c, g)])
                            - at(&[], &[(a, h), (c, -g)])
                            - at(&[], &[(a, -h), (c, g)])
                            + at(&[], &[(a, -h), (c, -g)]))
                            / (4.0 * h * g);
                        d[(a, c)] = v;
                        d[(c, a)] = v;
                    }
                }
                let mut b = DMatrix::zeros(nf, k);
                for s in 0..nf {
                    let hs = theta_steps[s];
                    for a in 0..k {
                        let h = level_steps[pos[a]];
                        b[(s, a)] = (at(&[(s, hs)], &[(a, h)])
                            - at(&[(s, hs)], &[(a, -h)])
                            - at(&[(s, -hs)], &[(a, h)])
                            + at(&[(s, -hs)], &[(a, -h)]))
                            / (4.0 * hs * h);
                    }
                }
                (d, b)
            })
            .collect();
        let mut d_groups: Vec<DMatrix<f64>> = layout
            .groups
            .iter()
            .map(|g| DMatrix::zeros(g.coords.len(), g.coords.len()))
            .collect();
        let mut b: DMatrix<f64> = DMatrix::zeros(nf, l);
        for (i, (di, bi)) in parts.iter().enumerate() {
            let pos = &layout.active[i];
            for (a, &pa) in pos.iter().enumerate() {
                let (g, la) = layout.local[pa];
                for (c, &pc) in pos.iter().enumerate() {
                    d_groups[g][(la, layout.local[pc].1)] += di[(a, c)];
                }
                for s in 0..nf {
                    b[(s, pa)] += bi[(s, a)];
                }
            }
        }
        if !d_groups.iter().all(|d| d.iter().all(|v| v.is_finite()))
            || !b.iter().all(|v| v.is_finite())
        {
            return Err(
                "Covariance step failed: the objective is not finite at some \
                 finite-difference point of the profiled levels. SE estimates not available."
                    .to_string(),
            );
        }
        Ok((d_groups, b))
    }
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
