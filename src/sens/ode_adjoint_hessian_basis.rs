//! The covariance Hessian's plans along a basis of their inputs (#2980).
//!
//! A pass sends its directions through the kernels and every plan of the composition (the
//! predictions, an SDE's variance rows, FOCE's population predictions, the non-Gaussian term,
//! AGQ's node plans). A plan is linear in its inputs' tangents: along direction `a`, whose
//! inputs move by `v_a = B·c_a` over a basis `B` of the inputs any direction moves, its
//! outputs' tangents are `J·B·c_a`, and its forward over reverse, seeded with the kernel's
//! value adjoint `ū` (the same in every pass) and its tangent `u̇̄_a`, returns the inputs'
//! adjoint tangents `∇²(ūᵀP)·B·c_a + Jᵀ·u̇̄_a`. So a plan runs **once per subject** along the
//! basis — its forward kept, its forward over reverse seeded with `ū` on the first pass — and
//! every pass assembles its directions from those: the plan's work is the basis's size instead
//! of the number of directions. Exact: the same derivatives, assembled by linearity.
//!
//! The basis is the plan's η slots (every direction's `η′` moves them all) and the parameter
//! slots some direction moves. Ω, σ outside an SDE's variance plan, and ρ never reach a plan
//! but through η, so they cost a plan nothing. A θ that reaches every prediction only through
//! exclusive mu lines `P = g⁻¹(μ(θ) + η)` ([`MuFold`]) moves the predictions as the η-direction
//! `∂μ/∂θ` does, so it needs no axis either: the basis of a mu-referenced model's prediction
//! plan is its η slots alone.

use super::*;
use crate::types::MuTransform;

/// A plan input: η slot `k`, or parameter slot `i` (θ, then the network outputs under the
/// reduced basis, then an SDE variance plan's σ).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Input {
    Eta(usize),
    Param(usize),
}

/// Which plan of the composition a basis belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum PlanKey {
    Pred,
    Var,
    Pop,
    Term,
    /// AGQ node `j`'s predictions, its variance rows, its term.
    Node(usize),
    NodeVar(usize),
    NodeTerm(usize),
}

impl PlanKey {
    /// Whether the plan reads θ the way the predictions do, so the mu fold holds for it: the
    /// predictions at η̂ and at the nodes, and FOCE's population predictions — the same plan at
    /// `η = 0`, where `g⁻¹(μ(θ) + η)` folds with the same coefficients — when they are the
    /// model's own (`pop_is_model`: no IOV, whose effective model is another). The other plans
    /// keep an axis per moved θ.
    fn folds(self, pop_is_model: bool) -> bool {
        match self {
            PlanKey::Pred | PlanKey::Node(_) => true,
            PlanKey::Pop => pop_is_model,
            _ => false,
        }
    }
}

/// What a plan's tangent lanes carry: the pass's directions, or input axes.
#[derive(Clone, Copy)]
pub(super) enum Lanes<'d> {
    Dirs(&'d [Dir]),
    Axes(&'d [Input]),
}

/// The packed coordinates and network outputs some direction of the Hessian moves.
pub(super) struct Moved {
    x: Vec<bool>,
    z: Vec<bool>,
    n_dirs: usize,
}

/// The mu fold at one subject (#2982): θ that reach every plan only through exclusive mu lines
/// `P_k = g_k⁻¹(μ_k(θ) + η_{e(k)})` ([`crate::parser::model_parser::MuFoldSpec`]) move the
/// predictions exactly as the η-direction `Σ_k ∂μ_k/∂θ·e_{e(k)}` does.
pub(super) struct MuFold {
    /// Per θ: `(η, ∂μ/∂θ)` over the lines it reaches; empty for a θ that is not folded.
    d1: Vec<Vec<(usize, f64)>>,
    /// Per `(θ_i, θ_j)`, `i` folded: `(η, ∂²μ/∂θ_i∂θ_j)` — what the fold, which uses `∂μ/∂θ_i`
    /// at this point, leaves out of the curvature along θ_i. Formed on first read
    /// ([`Self::d2`]): a pair of θ per program run, and only the covariance Hessian's curvature
    /// reads it — the mode response (the outer optimizer's predictor) does not.
    d2: std::cell::OnceCell<Option<Vec<Vec<Vec<(usize, f64)>>>>>,
}

/// The mu lines' `μ_k(θ) = g_k(P_k(θ, η = 0))` on a forward second-order dual seeded with lane
/// 0 along `θ_i` and lane 1 along `θ_j` (`usize::MAX`: no lane), from one program run: a mu
/// line is a few operations, so this is cheaper than recording one.
fn mu_dual(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    n_eta: usize,
    i: usize,
    j: usize,
) -> Option<Vec<crate::sens::dual2::Dual2<2>>> {
    use crate::sens::dual2::Dual2;
    let spec = &model.indiv_param_partials.mu_fold;
    let prog = model.indiv_param_partials.indiv_param_program.as_ref()?;
    let slots: Vec<usize> = spec.lines.iter().map(|l| l.1).collect();
    let th: Vec<Dual2<2>> = theta
        .iter()
        .enumerate()
        .map(|(m, &v)| {
            let mut d = Dual2::<2>::constant(v);
            if m == i {
                d.grad[0] = 1.0;
            }
            if m == j {
                d.grad[1] = 1.0;
            }
            d
        })
        .collect();
    let z = vec![Dual2::<2>::constant(0.0); n_eta];
    let vals = prog.eval_var_slots_g(&th, &z, &subject.covariates, &slots);
    Some(
        spec.lines
            .iter()
            .zip(vals)
            .map(|(&(_, _, t), v)| match t {
                MuTransform::Log => v.ln(),
                MuTransform::Logit => (v / (Dual2::<2>::constant(1.0) - v)).ln(),
                _ => v,
            })
            .collect(),
    )
}

impl MuFold {
    /// The fold at `(θ, η)`: `None` where the model has no foldable θ, where its plans read θ
    /// another way the parse cannot see (covariate networks, mixtures, IOV), where the
    /// subject's covariates vary over its records (`μ` would be per record), or where the
    /// program does not reproduce `g⁻¹(μ + η)` at this `η`. One program run per folded θ.
    pub(super) fn at(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        eta: &[f64],
    ) -> Option<Self> {
        let spec = &model.indiv_param_partials.mu_fold;
        if spec.thetas.is_empty()
            || model.has_covariate_nn()
            || model.mixture.is_some()
            || model.n_kappa > 0
            || subject.has_tv_covariates()
        {
            return None;
        }
        let prog = model.indiv_param_partials.indiv_param_program.as_ref()?;
        let cov = &subject.covariates;
        let n_th = theta.len();
        let inv = |t: MuTransform, u: f64| match t {
            MuTransform::Log => u.exp(),
            MuTransform::Logit => 1.0 / (1.0 + (-u).exp()),
            _ => u,
        };
        let slots: Vec<usize> = spec.lines.iter().map(|l| l.1).collect();
        let mut mu: Vec<f64> = Vec::new();
        let mut d1 = vec![Vec::new(); n_th];
        for &i in spec.thetas.iter().filter(|&&i| i < n_th) {
            let d = mu_dual(model, subject, theta, eta.len(), i, usize::MAX)?;
            if mu.is_empty() {
                mu = d.iter().map(|x| x.value).collect();
            }
            for (k, x) in d.iter().enumerate() {
                if !(x.value.is_finite() && x.grad[0].is_finite()) {
                    return None;
                }
                if x.grad[0] != 0.0 {
                    d1[i].push((spec.lines[k].0, x.grad[0]));
                }
            }
        }
        if mu.is_empty() {
            return None;
        }
        // The structure: each line's value is `g⁻¹(μ + η)` at the current η and at η moved by a
        // half on every effect — one point alone says nothing where η̂ is (near) zero, and AGQ's
        // plans run at nodes away from it. The two sides are one identity evaluated two ways
        // (through `exp`/`log` and back), so they differ by rounding, ~1e-15 relative; a line
        // that is not of that form misses by O(1) at the moved point.
        let moved: Vec<f64> = eta.iter().map(|v| v + 0.5).collect();
        for at in [eta, moved.as_slice()] {
            let vals = prog.eval_var_slots_g(theta, at, cov, &slots);
            for (k, (&(e, _, t), &p)) in spec.lines.iter().zip(&vals).enumerate() {
                let want = inv(t, mu[k] + at.get(e).copied()?);
                if !(p.is_finite() && (p - want).abs() <= 1e-10 * (1.0 + want.abs())) {
                    return None;
                }
            }
        }
        (d1.iter().any(|d| !d.is_empty())).then_some(MuFold {
            d1,
            d2: std::cell::OnceCell::new(),
        })
    }

    /// θ `i`'s lines, if the fold carries it.
    pub(super) fn of(&self, i: usize) -> Option<&[(usize, f64)]> {
        self.d1.get(i).map(Vec::as_slice).filter(|d| !d.is_empty())
    }

    /// `∂²μ/∂θ_i∂θ_j` per folded `i` (as the field), formed on the first read: one program run
    /// per pair, a pair of folded θ taking one run for both entries. `None`: a non-finite
    /// derivative, where the caller declines.
    pub(super) fn d2(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        n_eta: usize,
    ) -> Option<&Vec<Vec<Vec<(usize, f64)>>>> {
        self.d2
            .get_or_init(|| {
                let spec = &model.indiv_param_partials.mu_fold;
                let n_th = theta.len();
                let folded = |m: usize| self.of(m).is_some();
                // Only θ that reach a common line can have a nonzero cross derivative there.
                let share = |i: usize, j: usize| {
                    spec.reach.iter().any(|r| r.contains(&i) && r.contains(&j))
                };
                let mut d2 = vec![vec![Vec::new(); n_th]; n_th];
                for i in (0..n_th).filter(|&i| folded(i)) {
                    for j in (0..n_th).filter(|&j| !(folded(j) && j < i) && share(i, j)) {
                        let d = mu_dual(model, subject, theta, n_eta, i, j)?;
                        for (k, x) in d.iter().enumerate() {
                            let h = x.hess[0][1];
                            if !h.is_finite() {
                                return None;
                            }
                            if h != 0.0 {
                                let e = spec.lines[k].0;
                                d2[i][j].push((e, h));
                                if folded(j) && j != i {
                                    d2[j][i].push((e, h));
                                }
                            }
                        }
                    }
                }
                Some(d2)
            })
            .as_ref()
    }
}

/// One plan run along its input axes at a subject, and its adjoints once seeded.
pub(super) struct Basis {
    axes: Vec<Input>,
    n_eta: usize,
    n_par: usize,
    /// Chunk `c` carries axes `c·w ..`.
    runs: Vec<Kept>,
    w: usize,
    /// The outputs' base lanes, row `o·lb + x`: their values, and their Jacobian over the axes
    /// (row-major, a row per output lane) — read by every pass without touching the runs.
    lb: usize,
    vals: Vec<f64>,
    jac: Vec<f64>,
    /// The plan's fold, if it carries folded θ on its η axes.
    folded: bool,
    adj: Option<BasisAdj>,
}

/// A basis plan's forward over reverse seeded with `ū`: per input (η slots, then parameter
/// slots) its value adjoint and its adjoint's tangent along each axis (`∇²(ūᵀP)·B`).
struct BasisAdj {
    g: Vec<f64>,
    t: Vec<Vec<f64>>,
}

impl Basis {
    /// `(axes, base lanes per output, values, Jacobian)` of the prediction basis.
    pub(super) fn parts(&self) -> (&[Input], usize, &[f64], &[f64]) {
        (&self.axes, self.lb, &self.vals, &self.jac)
    }

    fn idx(&self, inp: Input) -> usize {
        match inp {
            Input::Eta(k) => k,
            Input::Param(i) => self.n_eta + i,
        }
    }

    fn axis_of(&self, inp: Input) -> Option<usize> {
        self.axes.iter().position(|&a| a == inp)
    }

    /// The runs' outputs as [`Self::vals`] and [`Self::jac`].
    fn dense(runs: &[Kept], w: usize, n_ax: usize) -> (usize, Vec<f64>, Vec<f64>) {
        let k0 = &runs[0];
        let (l, lb) = (k0.layout.lanes(), k0.layout.base_lanes());
        let n_out = (k0.du.len() / l).saturating_sub(1);
        let rows = n_out * lb;
        let mut vals = vec![0.0; rows];
        let mut jac = vec![0.0; rows * n_ax];
        for o in 0..n_out {
            for x in 0..lb {
                let r = o * lb + x;
                vals[r] = k0.out(o, x, None);
                for ax in 0..n_ax {
                    jac[r * n_ax + ax] = runs[ax / w].out(o, x, Some(ax % w));
                }
            }
        }
        (lb, vals, jac)
    }
}

/// A pass's use of a plan's basis: per direction of the pass, its coefficient on each axis, and
/// the folded θ it moves with that θ's tangent.
pub(super) struct BasisUse {
    key: PlanKey,
    coef: Vec<Vec<f64>>,
    folded: Vec<Vec<(usize, f64)>>,
    /// The outputs' base lanes (row `o·lb + x`): values, and per direction its tangent
    /// (`J·c_a`, row-major over the pass's directions).
    lb: usize,
    vals: Vec<f64>,
    tan: Vec<f64>,
}

/// A plan of a pass: run along the pass's directions, or assembled from its basis.
pub(super) enum PlanRun {
    Direct(Kept),
    Basis(BasisUse),
}

#[cfg(test)]
thread_local! {
    /// Test switch (#2980): every plan along the pass's directions — the basis's oracle.
    pub(crate) static PLAN_PER_DIRECTION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Test switch (#2982): no θ carried on the η axes — the fold's oracle.
    pub(crate) static MU_FOLD_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Plans this thread assembled from a basis; the tests read it to know which engine ran.
    pub(crate) static BASIS_PLANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The worst relative gap the fold's adjoint check has seen on this thread.
    pub(crate) static FOLD_CHECK_WORST: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn plan_per_direction() -> bool {
    #[cfg(test)]
    return PLAN_PER_DIRECTION.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

pub(super) fn mu_fold_off() -> bool {
    #[cfg(test)]
    return MU_FOLD_OFF.with(std::cell::Cell::get);
    #[cfg(not(test))]
    false
}

impl CovPlan<'_> {
    /// Before the passes: the axes some direction moves, so each plan's basis covers every axis
    /// a row is read along — not only the current pass's.
    pub(super) fn prepare(&self, dirs: &[Dir]) {
        let mut m = Moved {
            x: vec![false; self.nx],
            z: vec![false; self.nz],
            n_dirs: dirs.len(),
        };
        for d in dirs {
            if let Some(i) = d.x.filter(|&i| i < self.nx) {
                m.x[i] = true;
            }
            if let Some(z) = d.z.filter(|&z| z < self.nz) {
                m.z[z] = true;
            }
        }
        let same = self
            .moved
            .borrow()
            .as_ref()
            .is_some_and(|o| o.x == m.x && o.z == m.z && o.n_dirs == m.n_dirs);
        if !same {
            *self.moved.borrow_mut() = Some(m);
            self.bases.borrow_mut().clear();
        }
    }

    /// Parameter slot `i`'s tangent on lane `a` (θ, then network outputs, then σ).
    pub(super) fn param_t(&self, lanes: Lanes<'_>, i: usize, a: usize) -> f64 {
        match lanes {
            Lanes::Axes(ax) => f64::from(u8::from(ax.get(a) == Some(&Input::Param(i)))),
            Lanes::Dirs(dirs) => {
                let n_th = self.th.len();
                if i < n_th {
                    self.theta_t(dirs, i, a)
                } else if i < n_th + self.nz {
                    f64::from(u8::from(dirs.get(a).and_then(|d| d.z) == Some(i - n_th)))
                } else {
                    let k = i - n_th - self.nz;
                    let s0 = packed_segments(self.template).sigma_start();
                    if dirs.get(a).and_then(|d| d.x) == Some(s0 + k) {
                        self.sc.params.sigma.values[k]
                    } else {
                        0.0
                    }
                }
            }
        }
    }

    /// Whether parameter slot `i` is one some direction moves.
    fn moved_param(&self, m: &Moved, i: usize) -> bool {
        let n_th = self.th.len();
        if i < n_th {
            m.x.get(i).copied().unwrap_or(false)
        } else if i < n_th + self.nz {
            m.z[i - n_th]
        } else {
            let s0 = packed_segments(self.template).sigma_start();
            m.x.get(s0 + i - n_th - self.nz).copied().unwrap_or(false)
        }
    }

    /// Plan `key` for a pass along `dirs`: from its basis — run along the axes on the first
    /// pass that needs it, `run` the plan along given lanes — where the basis is narrower than
    /// the directions; else directly along `dirs`. `eta_t(k, a)` is direction `a`'s tangent on
    /// η slot `k`, `eta_up` whether the plan's η adjoints are read (its η slots then axes).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_run(
        &self,
        key: PlanKey,
        dirs: &[Dir],
        n_eta: usize,
        n_par: usize,
        eta_up: bool,
        eta_t: &dyn Fn(usize, usize) -> f64,
        width: usize,
        run: &dyn Fn(Lanes<'_>, &dyn Fn(usize, usize) -> f64, usize) -> Option<Kept>,
    ) -> Option<PlanRun> {
        let direct = || run(Lanes::Dirs(dirs), eta_t, width).map(PlanRun::Direct);
        let moved = self.moved.borrow();
        let Some(m) = moved.as_ref().filter(|_| {
            !dirs.is_empty() && !plan_per_direction() && !super::reverse_per_direction()
        }) else {
            return direct();
        };
        let pop_is_model = std::ptr::eq(self.sc.pop_model, self.sc.model);
        let fold = self.fold.as_ref().filter(|_| key.folds(pop_is_model));
        let folds = |i: usize| fold.and_then(|f| f.of(i));
        // The axes: the η slots where read (or where a folded θ lands), and the parameter slots
        // some direction moves that the fold does not carry.
        let mut eta_ax = vec![eta_up; n_eta];
        if let Some(f) = fold {
            for i in 0..self.th.len().min(n_par) {
                if self.moved_param(m, i) {
                    for &(k, _) in f.of(i).unwrap_or(&[]) {
                        if k < n_eta {
                            eta_ax[k] = true;
                        }
                    }
                }
            }
        }
        let axes: Vec<Input> = (0..n_eta)
            .filter(|&k| eta_ax[k])
            .map(Input::Eta)
            .chain(
                (0..n_par)
                    .filter(|&i| self.moved_param(m, i) && folds(i).is_none())
                    .map(Input::Param),
            )
            .collect();
        if axes.is_empty() || axes.len() >= m.n_dirs {
            return direct();
        }
        // Per direction of the pass, its tangent on every input, a folded θ's moved onto η.
        let lanes = Lanes::Dirs(dirs);
        let mut coef = Vec::with_capacity(dirs.len());
        let mut folded = vec![Vec::new(); dirs.len()];
        for (a, fa) in folded.iter_mut().enumerate() {
            let mut ve: Vec<f64> = (0..n_eta).map(|k| eta_t(k, a)).collect();
            let mut vp: Vec<f64> = (0..n_par).map(|i| self.param_t(lanes, i, a)).collect();
            for i in 0..n_par {
                if let Some(d) = folds(i).filter(|_| vp[i] != 0.0) {
                    for &(k, mk) in d {
                        if k < n_eta {
                            ve[k] += vp[i] * mk;
                        }
                    }
                    fa.push((i, vp[i]));
                    vp[i] = 0.0;
                }
            }
            let c: Vec<f64> = axes
                .iter()
                .map(|&ax| match ax {
                    Input::Eta(k) => ve[k],
                    Input::Param(i) => vp[i],
                })
                .collect();
            // Every moved input must be an axis; one that is not (a direction this basis was not
            // prepared for) takes the direct route.
            let off_axis = (0..n_eta).any(|k| ve[k] != 0.0 && !eta_ax[k])
                || (0..n_par).any(|i| vp[i] != 0.0 && !axes.contains(&Input::Param(i)));
            if off_axis {
                return direct();
            }
            coef.push(c);
        }
        drop(moved);
        if !self.bases.borrow().contains_key(&key) {
            // One axis per run. A plan's kernels follow its subject's event sequence, so subjects
            // with different designs compile their own: the compiled code grows with the tangent
            // width times the number of distinct designs, while the arithmetic is the same however
            // the axes are split. Measured (#2980) on 300 subjects with an ODE: one run of every
            // axis compiled 2.7× the code of this.
            let bw = 1;
            let runs = axes
                .chunks(bw)
                .map(|ch| {
                    let t =
                        |k: usize, b: usize| f64::from(u8::from(ch.get(b) == Some(&Input::Eta(k))));
                    run(Lanes::Axes(ch), &t, bw)
                })
                .collect::<Option<Vec<Kept>>>()?;
            let (lb, vals, jac) = Basis::dense(&runs, bw, axes.len());
            self.bases.borrow_mut().insert(
                key,
                Basis {
                    axes: axes.clone(),
                    n_eta,
                    n_par,
                    runs,
                    w: bw,
                    lb,
                    vals,
                    jac,
                    folded: fold.is_some(),
                    adj: None,
                },
            );
        }
        // The pass's tangents, `J·c_a` per output lane, once.
        let bases = self.bases.borrow();
        let b = &bases[&key];
        let (n_ax, nd) = (b.axes.len(), coef.len());
        let rows = b.vals.len();
        let mut tan = vec![0.0; rows * nd];
        for r in 0..rows {
            let jr = &b.jac[r * n_ax..(r + 1) * n_ax];
            for (a, c) in coef.iter().enumerate() {
                tan[r * nd + a] = jr.iter().zip(c).map(|(j, c)| j * c).sum();
            }
        }
        let (lb, vals) = (b.lb, b.vals.clone());
        drop(bases);
        #[cfg(test)]
        BASIS_PLANS.with(|c| c.set(c.get() + 1));
        Some(PlanRun::Basis(BasisUse {
            key,
            coef,
            folded,
            lb,
            vals,
            tan,
        }))
    }

    /// Output `o`'s base lane `x` of a pass's plan: its value, or its tangent along the pass's
    /// direction `a`.
    pub(super) fn plan_out(&self, pr: &PlanRun, o: usize, x: usize, a: Option<usize>) -> f64 {
        match pr {
            PlanRun::Direct(k) => k.out(o, x, a),
            PlanRun::Basis(u) => {
                let r = o * u.lb + x;
                match a {
                    None => u.vals[r],
                    Some(a) if a < u.coef.len() => u.tan[r * u.coef.len() + a],
                    Some(_) => 0.0,
                }
            }
        }
    }

    /// [`Self::plan_reverse_lanes`] of a plan assembled from its basis, under forward over
    /// reverse: the inputs' adjoint tangents along each of the pass's directions, `M·c_a`
    /// (the basis's forward over reverse, formed with `ū` on the first pass) plus the
    /// Gauss–Newton term `Jᵀ·u̇̄_a` from the basis's forward tangents, plus — for a folded θ — the
    /// curvature of `μ` the fold froze. Returns the η adjoints (value, then each direction's
    /// tangent) as [`Self::plan_reverse_lanes`] does.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn basis_reverse(
        &self,
        u: &BasisUse,
        r: &CovRun,
        adj: &[Vec<f64>],
        slot: &dyn Fn(usize, usize) -> usize,
        n_out: usize,
        nl: usize,
        rows: &mut [Vec<f64>],
    ) -> Option<Vec<Vec<f64>>> {
        let (kt, lk) = (r.kt, 1 + r.kt);
        let a0 = adj.first()?;
        let mut bases = self.bases.borrow_mut();
        let b = bases.get_mut(&u.key)?;
        let n_in = b.n_eta + b.n_par;
        if b.adj.is_none() {
            // `∇²(ūᵀP)·B`: the basis's forward over reverse, `ū` on the value lanes.
            let mut g = vec![0.0; n_in];
            let mut t = vec![vec![0.0; b.axes.len()]; n_in];
            for (c, kept) in b.runs.iter().enumerate() {
                let (l, lb) = (kept.layout.lanes(), kept.layout.base_lanes());
                let w = n_out * l;
                let mut du_bar = kj::Buf::new(w);
                for j in 0..n_out {
                    for x in 0..nl {
                        let s = slot(j, x);
                        if s != usize::MAX {
                            du_bar[j * l + x] = a0[s * lk];
                        }
                    }
                }
                let (ne, np) = (kept.plan.n_eta(), kept.plan.n_theta());
                let (mut ub, mut pb) = (kj::Buf::new(ne * l), kj::Buf::new(np * l));
                if !kept
                    .plan
                    .reverse_tangent_from(&kept.fwd, 1, &mut ub, &mut pb, &du_bar)
                {
                    return None;
                }
                for tl in 0..b.w.min(b.axes.len() - c * b.w) {
                    let ax = c * b.w + tl;
                    for k in 0..ne.min(b.n_eta) {
                        t[k][ax] = ub[k * l + lb + tl];
                    }
                    for i in 0..np.min(b.n_par) {
                        t[b.n_eta + i][ax] = pb[i * l + lb + tl];
                    }
                }
                if c == 0 {
                    for k in 0..ne.min(b.n_eta) {
                        g[k] = ub[k * l];
                    }
                    for i in 0..np.min(b.n_par) {
                        g[b.n_eta + i] = pb[i * l];
                    }
                }
            }
            // The fold's own check, on what the plan computed: a θ it carries reaches the plan
            // only through its mu lines, so the plan's value adjoint of it is the η adjoints
            // along `∂μ/∂θ` — one sum two ways, equal to rounding. Any other route (a name the
            // parse did not see) shows as an O(1) gap, and the subject declines rather than
            // returning a Hessian without that route.
            if let Some(f) = self.fold.as_ref().filter(|_| b.folded) {
                for i in 0..b.n_par.min(self.th.len()) {
                    let Some(d) = f.of(i) else { continue };
                    let gi = g[b.n_eta + i];
                    let (sum, size) = d
                        .iter()
                        .filter(|&&(k, _)| k < b.n_eta)
                        .map(|&(k, mk)| mk * g[k])
                        .fold((0.0, gi.abs()), |(s, z), v| (s + v, z + v.abs()));
                    let gap = (gi - sum).abs();
                    #[cfg(test)]
                    FOLD_CHECK_WORST.with(|c| {
                        let rel = gap / size.max(1e-300);
                        if !(rel <= c.get()) {
                            c.set(rel);
                        }
                    });
                    if !(gap <= 1e-8 * size + f64::MIN_POSITIVE) {
                        return None;
                    }
                }
            }
            b.adj = Some(BasisAdj { g, t });
        }
        let b = &*b;
        let ba = b.adj.as_ref()?;
        // `Jᵀ·u̇̄_a` on each axis: the kernel's adjoint tangents of the outputs against the
        // basis's forward tangents.
        let n_ax = b.axes.len();
        let mut gn = vec![vec![0.0; n_ax]; kt];
        for j in 0..n_out {
            for x in 0..nl.min(b.lb) {
                let at = slot(j, x);
                if at == usize::MAX {
                    continue;
                }
                let jr = &b.jac[(j * b.lb + x) * n_ax..(j * b.lb + x + 1) * n_ax];
                for (a, g) in gn.iter_mut().enumerate() {
                    let ud = a0[at * lk + 1 + a];
                    if ud != 0.0 {
                        for (gv, jv) in g.iter_mut().zip(jr) {
                            *gv += ud * jv;
                        }
                    }
                }
            }
        }
        let fold = self.fold.as_ref().filter(|_| b.folded);
        // The fold's second derivatives, formed here on first use (the curvature below).
        let fold_d2 = match fold {
            Some(f) => Some(f.d2(self.sc.model, self.subject, &self.th, self.pn())?),
            None => None,
        };
        // Input `inp`'s adjoint tangent along direction `a`.
        let tan = |a: usize, inp: Input| -> f64 {
            let Some(c) = u.coef.get(a) else { return 0.0 };
            let gnv = match (b.axis_of(inp), inp) {
                (Some(ax), _) => gn[a][ax],
                (None, Input::Param(i)) => match fold.and_then(|f| f.of(i)) {
                    Some(d) => d
                        .iter()
                        .filter_map(|&(k, mk)| b.axis_of(Input::Eta(k)).map(|ax| mk * gn[a][ax]))
                        .sum(),
                    None => return 0.0,
                },
                (None, Input::Eta(_)) => return 0.0,
            };
            let ti = &ba.t[b.idx(inp)];
            let curv: f64 = c
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != 0.0)
                .map(|(ax, &v)| v * ti[ax])
                .sum();
            // The fold moved θ_i along `∂μ/∂θ_i` frozen at this point; the true direction's
            // curvature also carries that vector's own θ-derivative, read off the η value
            // adjoints (the plan sees θ_i only through those lines).
            let corr: f64 = match (inp, fold_d2) {
                (Input::Param(j), Some(d2)) if j < d2.len() => u.folded[a]
                    .iter()
                    .map(|&(i, ti)| {
                        ti * d2[i][j]
                            .iter()
                            .filter(|&&(k, _)| k < b.n_eta)
                            .map(|&(k, h)| h * ba.g[k])
                            .sum::<f64>()
                    })
                    .sum(),
                _ => 0.0,
            };
            curv + gnv + corr
        };
        let n_th = self.th.len();
        let s0 = packed_segments(self.template).sigma_start();
        let sigma = self.sc.params.sigma.values.as_slice();
        for (a, row) in rows.iter_mut().enumerate().take(kt) {
            for i in 0..b.n_par {
                let (pv, pa) = (ba.g[b.n_eta + i], tan(a, Input::Param(i)));
                if i < n_th {
                    row[i] += pa * self.d1[i];
                    if r.dir_x[a] == Some(i) {
                        row[i] += pv * self.d2[i];
                    }
                } else if i - n_th < self.nz {
                    row[self.nx + (i - n_th)] += pa;
                } else {
                    let k = i - n_th - self.nz;
                    row[s0 + k] += pa * sigma[k];
                    if r.dir_x[a] == Some(s0 + k) {
                        row[s0 + k] += pv * sigma[k];
                    }
                }
            }
        }
        let mut e = vec![0.0; b.n_eta * lk];
        for k in 0..b.n_eta {
            e[k * lk] = ba.g[k];
            for a in 0..kt {
                e[k * lk + 1 + a] = tan(a, Input::Eta(k));
            }
        }
        Some(vec![e])
    }
}
