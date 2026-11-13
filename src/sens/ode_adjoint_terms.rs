//! A subject term's compiled plan (#2786): the non-Gaussian likelihood term (TTE, RTTE,
//! binary, CTMM) recorded once per subject on the recording type, θ and η its inputs, and
//! served by the plan's forward and reverse at any layout.
//!
//! One forward at the second-order layout gives `(q, ∇q, ∇²q)`. Its reverse gives every
//! contraction of them at once: seeded with `(a_q, a_g, a_h)` on the output lanes, η's value
//! lanes carry `∂/∂η` of the contraction (so the third-order tensor, `N(N+1)/2` seeds of one
//! multi-seed reverse, is never formed by a third-order type) and θ's value lanes its
//! θ-gradient (no taped recording of the term).
use super::*;
use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
use crate::sens::num::vm::Layout;
use std::sync::Arc;

/// The second-order layout over `n` random effects (`n = 0`: the value alone).
pub(super) fn layout2(n: usize) -> Layout {
    if n == 0 {
        return Layout::F64;
    }
    Layout {
        n: n as u16,
        m: (n * (n + 1) / 2) as u16,
        th: false,
        wide: false,
        tan: 0,
    }
}

/// The first-order layout over `n` random effects.
pub(super) fn layout1(n: usize) -> Layout {
    Layout {
        n: n as u16,
        m: 0,
        th: false,
        wide: false,
        tan: 0,
    }
}

/// `layout`'s inputs at `(theta, eta)`: η seeded one-hot, θ constants.
fn inputs(layout: Layout, theta: &[f64], eta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let l = layout.lanes();
    let mut u = vec![0.0; eta.len() * l];
    for (k, &v) in eta.iter().enumerate() {
        u[k * l] = v;
        if layout.n > 0 {
            u[k * l + 1 + k] = 1.0;
        }
    }
    let mut p = vec![0.0; theta.len() * l];
    for (j, &v) in theta.iter().enumerate() {
        p[j * l] = v;
    }
    (u, p)
}

/// The lane of `∂²/∂η_k∂η_l` (`l ≤ k`) at `n` random effects.
pub(super) fn h_lane(n: usize, k: usize, l: usize) -> usize {
    let (k, l) = if l <= k { (k, l) } else { (l, k) };
    1 + n + k * (k + 1) / 2 + l
}

/// The plan (the first of the subject's variants whose guard holds) and its forward at
/// `layout`: the output's lanes, and the forward kept for the plan's reverse passes.
fn forward(
    plans: &[Arc<kj::Plan>],
    layout: Layout,
    theta: &[f64],
    eta: &[f64],
) -> Option<(Arc<kj::Plan>, Vec<f64>, kj::Fwd)> {
    let (u, p) = inputs(layout, theta, eta);
    let l = layout.lanes();
    let mut du = vec![0.0; 2 * l];
    plans.iter().find_map(|pl| {
        let f = pl.forward_layout(layout, &u, &p, &mut du)?;
        (du[l] == 1.0).then(|| (pl.clone(), du[..l].to_vec(), f))
    })
}

/// `(q, ∇q, ∇²q)` from a second-order forward's lanes.
fn jets2(n: usize, lanes: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
    let g = (0..n).map(|k| lanes[1 + k]).collect();
    let h = (0..n)
        .map(|k| (0..n).map(|l| lanes[h_lane(n, k, l)]).collect())
        .collect();
    (lanes[0], g, h)
}

/// A subject term's plans over `n = eta.len()` random effects (`kind` keys the term, see
/// [`TermKind`]), one of them a variant whose guard holds at `(theta, eta)`: recorded there
/// when none of the cached ones does. `None` when the term does not record.
#[cfg(feature = "survival")]
pub(crate) fn term_plans<S: crate::survival::SubjectTerm>(
    model: &CompiledModel,
    subject: &Subject,
    kind: TermKind,
    term: &S,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<Arc<kj::Plan>>> {
    use std::sync::atomic::Ordering::Relaxed;
    if !kj::enabled() || subject.obs_records.is_empty() {
        return None;
    }
    let n = eta.len();
    let key = kj::SubjectKey::new(model, subject, n, theta.len()).with_kind(kind.key());
    match kj::lookup(&key) {
        kj::Lookup::Bad => return None,
        kj::Lookup::Found(ks) => {
            // A variant whose guard holds here: the guards read values only.
            if forward(&ks, Layout::F64, theta, eta).is_some() {
                return Some(ks.as_ref().clone());
            }
            kj::STATS.guard_fail.fetch_add(1, Relaxed);
        }
        kj::Lookup::Missing => {}
    }
    kj::begin();
    let th: Vec<kj::Tr> = theta
        .iter()
        .enumerate()
        .map(|(j, &v)| kj::Tr::theta(j, v))
        .collect();
    let et: Vec<kj::Tr> = eta
        .iter()
        .enumerate()
        .map(|(k, &v)| kj::Tr::eta(k, v))
        .collect();
    let y = term.eval(&th, &et);
    kj::guard_log_diff(format!("{key:?}"));
    let rec = match kj::end(y.map(|v| vec![v.reg()])) {
        Ok(r) => r,
        Err(why) => {
            if kj::debug() {
                eprintln!("kjit: term not recorded: {why}");
            }
            kj::mark_bad(&key);
            return None;
        }
    };
    let base = if n == 0 { Layout::F64 } else { layout1(n) };
    let plan = Arc::new(rec.plan(base, n, theta.len())?);
    kj::add(&key, plan.clone());
    let kj::Lookup::Found(ks) = kj::lookup(&key) else {
        return None;
    };
    Some(ks.as_ref().clone())
}

#[cfg(not(feature = "survival"))]
pub(crate) fn term_plans<S>(
    _: &CompiledModel,
    _: &Subject,
    _: TermKind,
    _: &S,
    _: &[f64],
    _: &[f64],
) -> Option<Vec<Arc<kj::Plan>>> {
    None
}

/// Which subject term a plan computes: its key kind.
#[derive(Clone, Copy)]
pub(crate) enum TermKind {
    /// Every non-Gaussian endpoint summed (`NonGaussianTerm`).
    NonGaussian,
    Tte,
    Ctmm,
    Discrete,
    #[cfg(test)]
    Dtmm,
}

impl TermKind {
    fn key(self) -> u8 {
        match self {
            TermKind::NonGaussian => 1,
            TermKind::Tte => 2,
            TermKind::Ctmm => 3,
            TermKind::Discrete => 4,
            #[cfg(test)]
            TermKind::Dtmm => 5,
        }
    }
}

/// The subject's non-Gaussian term plans.
#[cfg(feature = "survival")]
pub(super) fn non_gaussian_plans(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<Vec<Arc<kj::Plan>>> {
    let term = crate::survival::NonGaussianTerm::new(model, subject);
    term_plans(model, subject, TermKind::NonGaussian, &term, theta, eta)
}

#[cfg(not(feature = "survival"))]
pub(super) fn non_gaussian_plans(
    _: &CompiledModel,
    _: &Subject,
    _: &[f64],
    _: &[f64],
) -> Option<Vec<Arc<kj::Plan>>> {
    None
}

/// `(q, ∇q, ∇²q)` of a subject term from its plan's second-order forward.
#[cfg(test)]
pub(crate) fn term_jets2_planned<S>(
    model: &CompiledModel,
    subject: &Subject,
    kind: TermKind,
    term: &S,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>, Vec<Vec<f64>>)>
where
    S: TermEval,
{
    let plans = term.plans(model, subject, kind, theta, eta)?;
    let n = eta.len();
    let (_, lanes, _) = forward(&plans, layout2(n), theta, eta)?;
    Some(jets2(n, &lanes))
}

/// `(q, ∇q)` of a subject term from its plan's first-order forward.
pub(crate) fn term_gradient_planned<S>(
    model: &CompiledModel,
    subject: &Subject,
    kind: TermKind,
    term: &S,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)>
where
    S: TermEval,
{
    let plans = term.plans(model, subject, kind, theta, eta)?;
    let n = eta.len();
    let layout = if n == 0 { Layout::F64 } else { layout1(n) };
    let (_, lanes, _) = forward(&plans, layout, theta, eta)?;
    Some((lanes[0], (0..n).map(|k| lanes[1 + k]).collect()))
}

/// A term the plans can serve: every `SubjectTerm` with `survival`, none without.
pub(crate) trait TermEval {
    fn plans(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        kind: TermKind,
        theta: &[f64],
        eta: &[f64],
    ) -> Option<Vec<Arc<kj::Plan>>>;
}

#[cfg(feature = "survival")]
impl<S: crate::survival::SubjectTerm> TermEval for S {
    fn plans(
        &self,
        model: &CompiledModel,
        subject: &Subject,
        kind: TermKind,
        theta: &[f64],
        eta: &[f64],
    ) -> Option<Vec<Arc<kj::Plan>>> {
        term_plans(model, subject, kind, self, theta, eta)
    }
}

/// `(q, ∇q, ∇²q)` of the subject's non-Gaussian term at `(theta, eta)` from its plan.
pub(crate) fn non_gaussian_jets2_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>, Vec<Vec<f64>>)> {
    let plans = non_gaussian_plans(model, subject, theta, eta)?;
    let n = eta.len();
    let (_, lanes, _) = forward(&plans, layout2(n), theta, eta)?;
    Some(jets2(n, &lanes))
}

/// `(q, ∇q)` of the subject's non-Gaussian term from its plan's first-order forward.
pub(crate) fn non_gaussian_gradient_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let plans = non_gaussian_plans(model, subject, theta, eta)?;
    let n = eta.len();
    let layout = if n == 0 { Layout::F64 } else { layout1(n) };
    let (_, lanes, _) = forward(&plans, layout, theta, eta)?;
    Some((lanes[0], (0..n).map(|k| lanes[1 + k]).collect()))
}

/// `(q, ∇q, ∇²q)` from the plan's second-order forward, kept with the plan: the third order
/// is only contracted ([`super::NgJets::contract_t`], [`contract_hessian_lanes`]).
pub(super) fn non_gaussian_jets3_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<NgJets> {
    let plans = non_gaussian_plans(model, subject, theta, eta)?;
    let n = eta.len();
    let layout = layout2(n);
    let (plan, lanes, fw) = forward(&plans, layout, theta, eta)?;
    let (q, g, h) = jets2(n, &lanes);
    let t = super::NgThird::Planned(plan, fw);
    let jets = NgJets { q, g, h, t };
    (jets.q.is_finite() && jets.q < 1e19).then_some(jets)
}

/// `Σ_{k≥l} a_h[k][l]·∂³q/∂η_k∂η_l∂η_m` for every `m`: one reverse of the kept second-order
/// forward, seeded with `a_h` on the `∇²q` lanes (its η value lanes).
pub(super) fn contract_hessian_lanes(
    plan: &kj::Plan,
    fwd: &kj::Fwd,
    n: usize,
    a_h: &[Vec<f64>],
) -> Option<Vec<f64>> {
    let layout = fwd.layout();
    let l = layout.lanes();
    let mut du_bar = vec![0.0; l];
    for k in 0..n {
        for m in 0..=k {
            du_bar[h_lane(n, k, m)] = a_h[k][m];
        }
    }
    let mut u_bar = vec![0.0; n * l];
    let mut p_bar = vec![0.0; plan.n_theta() * l];
    if !plan.reverse_from(fwd, layout, 1, &mut u_bar, &mut p_bar, &du_bar) {
        return None;
    }
    let out: Vec<f64> = (0..n).map(|m| u_bar[m * l]).collect();
    out.iter().all(|v| v.is_finite()).then_some(out)
}

/// `Σ_parts ∂/∂θ [a_q·q + Σ_k a_g,k·∂_k q + Σ_{k≥l} a_h,kl·∂²_kl q]` at each part's η, from
/// the plan's reverse (one per part, θ's value lanes): [`ng_theta_adjoint`]'s quantity.
pub(super) fn non_gaussian_theta_adjoint_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    parts: &[NgSeed<'_>],
) -> Option<Vec<f64>> {
    let mut out = vec![0.0; theta.len()];
    for part in parts {
        let d = non_gaussian_theta_seeds_planned(
            model,
            subject,
            theta,
            part.eta,
            &[(part.a_q, part.a_g, part.a_h)],
        )?;
        for (o, v) in out.iter_mut().zip(&d[0]) {
            *o += v;
        }
    }
    Some(out)
}

/// Per seed `(a_q, a_g, a_h)` at one `eta`, `∂/∂θ [a_q·q + Σ a_g,k·∂_k q + Σ_{k≥l} a_h,kl·∂²_kl q]`:
/// one forward and one reverse for every seed, at the value layout when every seed weights
/// the value alone (an AGQ node's) and the second-order one otherwise.
pub(super) fn non_gaussian_theta_seeds_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    seeds: &[(f64, &[f64], &[Vec<f64>])],
) -> Option<Vec<Vec<f64>>> {
    let n = eta.len();
    let plans = non_gaussian_plans(model, subject, theta, eta)?;
    let value_only = seeds.iter().all(|(_, a_g, a_h)| {
        a_g.iter().all(|&v| v == 0.0) && a_h.iter().flatten().all(|&v| v == 0.0)
    });
    let layout = if value_only || n == 0 {
        Layout::F64
    } else {
        layout2(n)
    };
    let (plan, _, fw) = forward(&plans, layout, theta, eta)?;
    let l = layout.lanes();
    let mut du_bar = vec![0.0; seeds.len() * l];
    for (s, (a_q, a_g, a_h)) in seeds.iter().enumerate() {
        let du = &mut du_bar[s * l..(s + 1) * l];
        du[0] = *a_q;
        if l > 1 {
            for k in 0..n {
                if let Some(&a) = a_g.get(k) {
                    du[1 + k] = a;
                }
                for m in 0..=k {
                    if let Some(&a) = a_h.get(k).and_then(|r| r.get(m)) {
                        du[h_lane(n, k, m)] = a;
                    }
                }
            }
        }
    }
    let (u, p) = inputs(layout, theta, eta);
    let mut u_bar = vec![0.0; seeds.len() * u.len()];
    let mut p_bar = vec![0.0; seeds.len() * p.len()];
    if !plan.reverse_from(&fw, layout, seeds.len(), &mut u_bar, &mut p_bar, &du_bar) {
        return None;
    }
    let out: Vec<Vec<f64>> = (0..seeds.len())
        .map(|s| {
            (0..theta.len())
                .map(|j| p_bar[s * p.len() + j * l])
                .collect()
        })
        .collect();
    out.iter().flatten().all(|v| v.is_finite()).then_some(out)
}

/// The term's value and θ-gradient at fixed `eta`, from the plan: the value-layout forward and
/// its reverse seeded on the value ([`non_gaussian_theta_score`]'s quantity).
pub(super) fn non_gaussian_theta_score_planned(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> Option<(f64, Vec<f64>)> {
    let plans = non_gaussian_plans(model, subject, theta, eta)?;
    let layout = Layout::F64;
    let (plan, lanes, fw) = forward(&plans, layout, theta, eta)?;
    let (u, p) = inputs(layout, theta, eta);
    let (mut u_bar, mut p_bar) = (vec![0.0; u.len()], vec![0.0; p.len()]);
    if !plan.reverse_from(&fw, layout, 1, &mut u_bar, &mut p_bar, &[1.0]) {
        return None;
    }
    let q = lanes[0];
    (q.is_finite() && p_bar.iter().all(|v| v.is_finite())).then_some((q, p_bar))
}
