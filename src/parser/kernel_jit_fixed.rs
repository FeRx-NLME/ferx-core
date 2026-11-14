//! A periodic steady state as one call of a compiled plan (#2786): `u* = P(u*)` solved at run
//! time on the values, its derivatives by the implicit function theorem.
//!
//! A steady-state dose's state is the fixed point of the one-cycle map `P` (apply the dose,
//! integrate one interval). How the value path finds it — the linear closed form, Anderson
//! acceleration, or the pulse train to a tolerance — is a run-time search whose length depends
//! on the values, so it cannot be unrolled into a recording. Its derivatives do not need the
//! search: with `M = (I − J)⁻¹`, `J = ∂P/∂u` at `u*`, the step `Φ(u) = u + M(P(u) − u)` has
//! `∂Φ/∂u = I − M(I − J) = 0` there, so starting from `u*` with no derivative the derivative
//! lanes of `Φ(…Φ(u*))` are the exact implicit derivatives `du* = M·∂P` to the order of the
//! steps (each step makes one more order exact), and the reverse of one step more is the
//! exact adjoint `λ = Mᵀū`, `ū_caps = ∂P/∂capsᵀ·λ`.
//!
//! So the call holds `P` (and the maps the value search uses) as sub-plans over `[u, captures]`
//! — the outer values the cycle reads — and at run time:
//!
//! 1. the value search runs on the sub-plans' value layout: the very code the value path runs
//!    (`solve`), so the value lane agrees with it bit for bit;
//! 2. `J` from the defining map's forward with `u`'s tangents one-hot;
//! 3. steps `Φ` over the call's layout — its derivative order, one more before a reverse —
//!    the value lane then set to the search's `u*`.
use super::*;

/// One map of `u` (and the captures) the value search calls, as the value path's `f64` closure.
pub(crate) type F64Map<'a> = &'a dyn Fn(&[f64]) -> Option<Vec<f64>>;

/// What the value search found, which decides how the call differentiates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Found {
    /// A fixed point of map `k`: derivatives by the implicit function theorem through it.
    Fixed(usize),
    /// No fixed point reached: map `map` applied `count` times from the zero state (a capped
    /// pulse train). The value is that iterate and not a fixed point, so its derivatives are
    /// the iteration's own — the implicit ones would belong to a state it never computed.
    Iterated { map: usize, count: usize },
}

/// The value search: the value path's own steady-state solve over the given maps, returning
/// `u*` and what it is ([`Found`]).
pub(crate) type SolveFn = dyn Fn(&[F64Map<'_>]) -> Option<(Vec<f64>, Found)> + Send + Sync;

/// A steady-state call: its maps as sub-plans over `[u (n), captures]`, and the value search.
pub(crate) struct FixedSpec {
    pub(super) n: usize,
    /// The coordinates the search holds at their start value (see [`record_fixed`]).
    pub(super) held: Vec<usize>,
    pub(super) n_caps: usize,
    pub(super) maps: Vec<Arc<Plan>>,
    pub(super) solve: Box<SolveFn>,
}

/// What a fixed-point call's forward keeps for its reverse: the steps' forwards and `M` (empty
/// for an iterated search, whose steps are the map's own applications).
pub(crate) struct FixedLog {
    steps: Vec<Fwd>,
    map: usize,
    m: Vec<f64>,
}

impl FixedSpec {
    /// Map `k` at `layout` from `u`'s and the captures' lanes; `None` when it fails or a
    /// guard of its own does not hold.
    fn eval(&self, k: usize, layout: Layout, u: &[f64], caps: &[f64]) -> Option<(Vec<f64>, Fwd)> {
        let l = layout.lanes();
        let plan = &self.maps[k];
        let mut p = Vec::with_capacity(u.len() + caps.len());
        p.extend_from_slice(u);
        p.extend_from_slice(caps);
        let mut du = vec![0.0; (plan.n_out() + 1) * l];
        let f = plan.forward_layout(layout, &[], &p, &mut du)?;
        if du[plan.n_out() * l] != 1.0 {
            return None;
        }
        du.truncate(self.n * l);
        Some((du, f))
    }

    /// The call over `layout`: `inputs` the captures' lanes, `out` the state's (`n × lanes`).
    pub(super) fn run(
        &self,
        layout: Layout,
        inputs: &[f64],
        out: &mut [f64],
        log: Option<&mut Option<FixedLog>>,
    ) -> bool {
        self.run_inner(layout, inputs, out, log).is_some()
    }

    fn run_inner(
        &self,
        layout: Layout,
        inputs: &[f64],
        out: &mut [f64],
        log: Option<&mut Option<FixedLog>>,
    ) -> Option<()> {
        let (n, l) = (self.n, layout.lanes());
        // 1. The value search, on the captures' values.
        let caps0: Vec<f64> = (0..self.n_caps).map(|c| inputs[c * l]).collect();
        let maps: Vec<Box<dyn Fn(&[f64]) -> Option<Vec<f64>>>> = (0..self.maps.len())
            .map(|k| {
                let caps0 = caps0.clone();
                Box::new(move |u: &[f64]| self.eval(k, Layout::F64, u, &caps0).map(|(y, _)| y))
                    as Box<dyn Fn(&[f64]) -> Option<Vec<f64>>>
            })
            .collect();
        let refs: Vec<F64Map<'_>> = maps.iter().map(|b| b.as_ref() as F64Map<'_>).collect();
        let (u_star, found) = (self.solve)(&refs)?;
        let k = match found {
            Found::Fixed(k) => k,
            Found::Iterated { map, count } => {
                // The iteration itself over the layout, from the zero state.
                let mut u = vec![0.0; n * l];
                let mut steps = Vec::new();
                for _ in 0..count {
                    let (p, f) = self.eval(map, layout, &u, inputs)?;
                    u = p;
                    if log.is_some() {
                        steps.push(f);
                    }
                }
                if !u.iter().all(|x| x.is_finite()) {
                    return None;
                }
                for i in 0..n {
                    u[i * l] = u_star[i];
                }
                out[..n * l].copy_from_slice(&u);
                if let Some(lg) = log {
                    *lg = Some(FixedLog {
                        steps,
                        map,
                        m: Vec::new(),
                    });
                }
                return Some(());
            }
        };
        // One Newton step per derivative order the layout carries, and one more when a
        // reverse follows: it holds `M` fixed, which leaves an error one order past the steps.
        let n_steps = layout.order() + log.is_some() as usize;
        if n_steps == 0 {
            out[..n].copy_from_slice(&u_star);
            return Some(());
        }
        // 2. `M = (I − J)⁻¹` through the defining map, `u`'s tangents one-hot.
        let lj = Layout {
            n: n as u16,
            m: 0,
            th: false,
            wide: false,
            tan: 0,
        };
        let w = lj.lanes();
        let mut uj = vec![0.0; n * w];
        for i in 0..n {
            uj[i * w] = u_star[i];
            uj[i * w + 1 + i] = 1.0;
        }
        let capsj: Vec<f64> = caps0
            .iter()
            .flat_map(|&c| {
                let mut v = vec![0.0; w];
                v[0] = c;
                v
            })
            .collect();
        let (pj, _) = self.eval(k, lj, &uj, &capsj)?;
        let mut a = vec![0.0; n * n];
        for r in 0..n {
            for c in 0..n {
                a[r * n + c] = if r == c { 1.0 } else { 0.0 } - pj[r * w + 1 + c];
            }
        }
        // `M` over the coordinates the search solves for; a held one (a hazard accumulator
        // the run-in keeps still, `P_h(u) = u_h`) has a zero row and column, so its output
        // keeps the search's value and no derivative, and no step moves it.
        let free: Vec<usize> = (0..n).filter(|i| !self.held.contains(i)).collect();
        let nf = free.len();
        let af: Vec<f64> = free
            .iter()
            .flat_map(|&r| free.iter().map(move |&c| (r, c)))
            .map(|(r, c)| a[r * n + c])
            .collect();
        let mut m = vec![0.0; n * n];
        for (ci, &c) in free.iter().enumerate() {
            let mut e = vec![0.0; nf];
            e[ci] = 1.0;
            let x = crate::sens::linsolve::solve_linear_system_g::<f64>(&af, &e, nf)?;
            for (ri, &r) in free.iter().enumerate() {
                m[r * n + c] = x[ri];
            }
        }
        // 3. The steps over the call's layout, from `u*` with no derivative.
        let mut u = vec![0.0; n * l];
        for i in 0..n {
            u[i * l] = u_star[i];
        }
        let mut steps = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let (p, f) = self.eval(k, layout, &u, inputs)?;
            let mut next = u.clone();
            for lane in 0..l {
                for r in 0..n {
                    let mut s = 0.0;
                    for c in 0..n {
                        s += m[r * n + c] * (p[c * l + lane] - u[c * l + lane]);
                    }
                    next[r * l + lane] += s;
                }
            }
            u = next;
            steps.push(f);
        }
        if !u.iter().all(|x| x.is_finite()) {
            return None;
        }
        for i in 0..n {
            u[i * l] = u_star[i];
        }
        out[..n * l].copy_from_slice(&u);
        if let Some(lg) = log {
            *lg = Some(FixedLog { steps, map: k, m });
        }
        Some(())
    }

    /// The reverse over `layout` from the forward's `log`, for `seeds` blocks: `out_adj` the
    /// state's adjoints (`seeds × n × lanes`), adding the captures' into `in_adj`
    /// (`seeds × n_caps × lanes`). The state the search started from takes none.
    pub(super) fn run_rev(
        &self,
        layout: Layout,
        log: &FixedLog,
        seeds: usize,
        out_adj: &[f64],
        in_adj: &mut [f64],
    ) -> bool {
        let (n, l, nc) = (self.n, layout.lanes(), self.n_caps);
        if log.map == usize::MAX {
            // A constant state: no adjoint reaches the captures.
            return true;
        }
        let plan = &self.maps[log.map];
        if log.m.is_empty() {
            // An iterated search: back through each application of the map.
            for sd in 0..seeds {
                let mut ub: Vec<f64> = out_adj[sd * n * l..(sd + 1) * n * l].to_vec();
                for f in log.steps.iter().rev() {
                    let mut pb = vec![0.0; (n + nc) * l];
                    if !plan.reverse_from(f, layout, 1, &mut [], &mut pb, &ub) {
                        return false;
                    }
                    ub.copy_from_slice(&pb[..n * l]);
                    let ia = &mut in_adj[sd * nc * l..(sd + 1) * nc * l];
                    for (x, y) in ia.iter_mut().zip(&pb[n * l..]) {
                        *x += y;
                    }
                }
            }
            return true;
        }
        for sd in 0..seeds {
            let mut ub: Vec<f64> = out_adj[sd * n * l..(sd + 1) * n * l].to_vec();
            for f in log.steps.iter().rev() {
                // `u ← u + M(P(u) − u)`: `w = Mᵀū` reaches `P`'s output; `ū − w` stays on `u`.
                let mut w = vec![0.0; n * l];
                for lane in 0..l {
                    for c in 0..n {
                        let mut s = 0.0;
                        for r in 0..n {
                            s += log.m[r * n + c] * ub[r * l + lane];
                        }
                        w[c * l + lane] = s;
                    }
                }
                let mut pb = vec![0.0; (n + nc) * l];
                if !plan.reverse_from(f, layout, 1, &mut [], &mut pb, &w) {
                    return false;
                }
                for i in 0..n * l {
                    ub[i] = ub[i] - w[i] + pb[i];
                }
                let ia = &mut in_adj[sd * nc * l..(sd + 1) * nc * l];
                for (x, y) in ia.iter_mut().zip(&pb[n * l..]) {
                    *x += y;
                }
            }
        }
        true
    }
}

/// Record a steady-state call (see the module docs): each of `maps` recorded once over fresh
/// variables `u` (record-time values `u0`) as a sub-plan whose other inputs are the outer
/// values it reads, then the call in the open recording, its outputs at the value search's
/// `u*`. `held` are the coordinates every map leaves where they are (a run-in's hazard
/// accumulators): the search holds them, so they keep its value with no derivative and `M`
/// is taken over the rest. `None` — the recording fails — if a map cannot be recorded or the
/// search fails.
pub(crate) fn record_fixed(
    n: usize,
    held: &[usize],
    u0: &[f64],
    maps: &[&dyn Fn(&[Tr]) -> Option<Vec<Tr>>],
    solve: Box<SolveFn>,
) -> Option<Vec<Tr>> {
    let mut caps: Vec<u32> = Vec::new();
    let mut subs: Vec<Recording> = Vec::with_capacity(maps.len());
    for f in maps {
        subs.push(record_sub(n, u0, &mut caps, f)?);
    }
    // Every map reads every capture (an unread one is an input it ignores).
    let n_caps = caps.len();
    let plans: Vec<Arc<Plan>> = subs
        .into_iter()
        .map(|r| {
            r.with_caps(n, &caps)
                .plan(Layout::F64, 0, n + n_caps)
                .map(Arc::new)
        })
        .collect::<Option<_>>()?;
    let spec = Arc::new(FixedSpec {
        n,
        held: held.to_vec(),
        n_caps,
        maps: plans,
        solve,
    });
    let cap_vals: Vec<f64> = rec(|r| caps.iter().map(|&c| r.vals[c as usize]).collect());
    let mut out = vec![0.0; n];
    if !spec.run(Layout::F64, &cap_vals, &mut out, None) {
        fail_here("a steady-state call's record-time search failed");
        return None;
    }
    let id = rec(|r| {
        r.loops.push(CallKind::Fixed(spec));
        (r.loops.len() - 1) as u32
    });
    for &c in &caps {
        let a = c as u32;
        rec(|r| r.code.push(Flat::LoopArg { a }));
    }
    rec(|r| r.code.push(Flat::Loop { id }));
    Some(
        out.iter()
            .enumerate()
            .map(|(k, &v)| {
                let r = node(None, |d| Flat::LoopOut { d, k: k as u32 });
                Tr::mk(v, r, true)
            })
            .collect(),
    )
}

/// Record `f` over fresh variables (values `u0`) cut out of the open recording as a recording
/// of its own: its inputs the variables, then the outer registers it reads (added to `caps`).
fn record_sub(
    n: usize,
    u0: &[f64],
    caps: &mut Vec<u32>,
    f: &dyn Fn(&[Tr]) -> Option<Vec<Tr>>,
) -> Option<Recording> {
    let (start_code, start_reg, outer_guard, outer_guarded, outer_marks) = rec(|r| {
        r.sub_depth += 1;
        (
            r.code.len(),
            r.n_regs,
            r.guard.take(),
            std::mem::take(&mut r.guarded),
            r.marks.len(),
        )
    });
    let vars: Vec<Tr> = u0
        .iter()
        .map(|&v| Tr::mk(v, node(None, |d| Flat::LoadK { d, c: v }), true))
        .collect();
    let y = f(&vars);
    let (region, guard, marks) = rec(|r| {
        r.sub_depth -= 1;
        let region = r.code.split_off(start_code);
        let g = r.guard.take();
        r.guard = outer_guard;
        r.guarded = outer_guarded;
        let marks: Vec<usize> = r
            .marks
            .split_off(outer_marks)
            .into_iter()
            .map(|m| m - start_code)
            .collect();
        r.cse.retain(|_, v| *v < start_reg);
        (region, g, marks)
    });
    let y = y?;
    if y.len() != n {
        return None;
    }
    let var_regs: Vec<u32> = vars.iter().map(|v| v.r).collect();
    let mut seen: std::collections::HashSet<u32, FxBuild> = var_regs.iter().copied().collect();
    let body: Vec<Flat> = region
        .into_iter()
        .filter(|f| !matches!(f, Flat::LoadK { d, .. } if var_regs.contains(&(*d as u32))))
        .collect();
    let mut outer: Vec<u32> = Vec::new();
    for g in &body {
        for u in uses(g).into_iter().flatten() {
            if seen.insert(u as u32) {
                outer.push(u as u32);
            }
        }
        if let Some(d) = def(g) {
            seen.insert(d as u32);
        }
    }
    let mut outs: Vec<u32> = y.iter().map(|o| o.r).collect();
    // An output (or a guard condition, shared by hash-consing) recorded before the map is an
    // input too.
    for o in outs.iter().chain(guard.iter()) {
        if !seen.contains(o) {
            outer.push(*o);
            seen.insert(*o);
        }
    }
    for c in outer {
        if !caps.contains(&c) {
            caps.push(c);
        }
    }
    // The loads are laid down by `with_caps`, once every map's captures are known.
    let (n_regs, pool, vals, loops) =
        rec(|r| (r.n_regs, r.pool.clone(), r.vals.clone(), r.loops.clone()));
    let _ = &mut outs;
    Some(Recording {
        code: body,
        n_regs,
        outs,
        guard,
        pool,
        vals,
        marks,
        eta_regs: Vec::new(),
        theta_regs: var_regs.iter().enumerate().map(|(i, &r)| (r, i)).collect(),
        loops,
        ode: true,
        dense: true,
        declined: 0,
        tie: Default::default(),
    })
}

impl Recording {
    /// The sub-recording's inputs laid down: the variables, then every capture, as θ slots.
    fn with_caps(mut self, n: usize, caps: &[u32]) -> Recording {
        let mut code: Vec<Flat> = Vec::with_capacity(n + caps.len() + self.code.len());
        for &(r, i) in &self.theta_regs {
            code.push(Flat::LoadParam {
                d: r as u32,
                slot: i as u32,
            });
        }
        for (c, &r) in caps.iter().enumerate() {
            code.push(Flat::LoadParam {
                d: r as u32,
                slot: (n + c) as u32,
            });
            self.theta_regs.push((r, n + c));
        }
        let shift = code.len();
        code.extend(self.code);
        self.code = code;
        self.marks = self.marks.into_iter().map(|m| m + shift).collect();
        self
    }
}

/// A sub-plan applied `count` times to its state (`count` an input whose value is a whole
/// number, read at run time): Padé's squarings, whose count follows the matrix norm. The
/// recording holds one application; the run repeats it exactly as often as the values ask,
/// and the reverse walks the kept applications back.
pub(crate) struct RepeatSpec {
    pub(super) n: usize,
    pub(super) n_caps: usize,
    pub(super) body: Arc<Plan>,
}

impl RepeatSpec {
    /// The call over `layout`: `inputs` the state's lanes, the count's, then the captures';
    /// `out` the state's after `count` applications.
    pub(super) fn run(
        &self,
        layout: Layout,
        inputs: &[f64],
        out: &mut [f64],
        log: Option<&mut Option<FixedLog>>,
    ) -> bool {
        let (n, l) = (self.n, layout.lanes());
        let count = inputs[n * l];
        if !(count >= 0.0 && count <= 64.0 && count.fract() == 0.0) {
            return false;
        }
        let caps = &inputs[(n + 1) * l..];
        let mut x = inputs[..n * l].to_vec();
        let mut steps = Vec::new();
        for _ in 0..count as usize {
            let mut p = Vec::with_capacity(x.len() + caps.len());
            p.extend_from_slice(&x);
            p.extend_from_slice(caps);
            let mut du = vec![0.0; (self.body.n_out() + 1) * l];
            let Some(f) = self.body.forward_layout(layout, &[], &p, &mut du) else {
                return false;
            };
            if du[self.body.n_out() * l] != 1.0 {
                return false;
            }
            x.copy_from_slice(&du[..n * l]);
            if log.is_some() {
                steps.push(f);
            }
        }
        out[..n * l].copy_from_slice(&x);
        if let Some(lg) = log {
            *lg = Some(FixedLog {
                steps,
                map: 0,
                m: Vec::new(),
            });
        }
        true
    }

    /// The reverse: `out_adj` the final state's adjoints per seed, into `in_adj` per seed the
    /// initial state's, none for the count, and the captures'.
    pub(super) fn run_rev(
        &self,
        layout: Layout,
        log: &FixedLog,
        seeds: usize,
        out_adj: &[f64],
        in_adj: &mut [f64],
    ) -> bool {
        let (n, l, nc) = (self.n, layout.lanes(), self.n_caps);
        let w = (n + 1 + nc) * l;
        for sd in 0..seeds {
            let mut ub: Vec<f64> = out_adj[sd * n * l..(sd + 1) * n * l].to_vec();
            let ia = &mut in_adj[sd * w..(sd + 1) * w];
            for f in log.steps.iter().rev() {
                let mut pb = vec![0.0; (n + nc) * l];
                if !self.body.reverse_from(f, layout, 1, &mut [], &mut pb, &ub) {
                    return false;
                }
                ub.copy_from_slice(&pb[..n * l]);
                for (x, y) in ia[(n + 1) * l..].iter_mut().zip(&pb[n * l..]) {
                    *x += y;
                }
            }
            for (x, y) in ia[..n * l].iter_mut().zip(&ub) {
                *x += y;
            }
        }
        true
    }
}

/// Record `body` applied `count` times to `x0` (see [`RepeatSpec`]): the body once, as a
/// sub-plan over fresh variables and the outer values it reads, then the call.
pub(crate) fn record_repeat(
    x0: &[Tr],
    count: Tr,
    body: &dyn Fn(&[Tr]) -> Vec<Tr>,
) -> Option<Vec<Tr>> {
    let n = x0.len();
    let mut caps: Vec<u32> = Vec::new();
    let vals: Vec<f64> = x0.iter().map(|v| v.v).collect();
    let f = |v: &[Tr]| Some(body(v));
    let sub = record_sub(n, &vals, &mut caps, &f)?;
    let n_caps = caps.len();
    let plan = Arc::new(sub.with_caps(n, &caps).plan(Layout::F64, 0, n + n_caps)?);
    let spec = Arc::new(RepeatSpec {
        n,
        n_caps,
        body: plan,
    });
    let mut inputs = vals.clone();
    inputs.push(count.v);
    inputs.extend(rec(|r| {
        caps.iter()
            .map(|&c| r.vals[c as usize])
            .collect::<Vec<f64>>()
    }));
    let mut out = vec![0.0; n];
    if !spec.run(Layout::F64, &inputs, &mut out, None) {
        fail_here("a repeat call's record-time run failed");
        return None;
    }
    let id = rec(|r| {
        r.loops.push(CallKind::Repeat(spec));
        (r.loops.len() - 1) as u32
    });
    for a in x0
        .iter()
        .map(|v| v.r)
        .chain(std::iter::once(count.r))
        .chain(caps.iter().copied())
    {
        let a = a as u32;
        rec(|r| r.code.push(Flat::LoopArg { a }));
    }
    rec(|r| r.code.push(Flat::Loop { id }));
    Some(
        out.iter()
            .enumerate()
            .map(|(k, &v)| {
                let r = node(None, |d| Flat::LoopOut { d, k: k as u32 });
                Tr::mk(v, r, true)
            })
            .collect(),
    )
}
