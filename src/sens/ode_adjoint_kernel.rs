//! A subject's likelihood kernel as a compiled plan (#2786).
//!
//! The kernel maps its leaves — the predictions with their η-jets, η, the population
//! parameters the likelihood reads directly (Ω's Cholesky factor, σ, ρ, θ) and a
//! non-Gaussian term's jets — to its outputs (the objective `F` and `∇_η l`). It is recorded
//! once per subject and configuration on the recording type, every leaf a parameter slot of
//! the plan, so a value forward replays the kernel at new leaves and one multi-seed reverse
//! gives every output's adjoint at every leaf: what a scalar tape's recording and sweeps gave
//! per call, without the recording. A forward along tangent lanes and its reverse give the
//! covariance step's Hessian-vector products the same way.
use super::*;
use crate::parser::model_parser::rhs_regcode::kernel_jit as kj;
use crate::sens::num::vm::Layout;
use std::sync::Arc;

/// What a kernel plan computes: its [`kj::SubjectKey`] kind.
pub(super) const KERNEL_KIND: u8 = 8;

/// A kernel plan's forward at one point: its outputs, and what its reverse passes read.
pub(super) struct KernelRun {
    plan: Arc<kj::Plan>,
    fwd: kj::Fwd,
    layout: Layout,
    n_in: usize,
    n_out: usize,
    /// Per output, its lanes (`layout.lanes()` each).
    pub(super) out: kj::Buf,
}

impl KernelRun {
    /// Output `o`'s value.
    pub(super) fn value(&self, o: usize) -> f64 {
        self.out[o * self.layout.lanes()]
    }

    /// One reverse seeded with `seeds` (`rows · n_out · lanes` entries, laid out like the
    /// outputs): per row, the adjoint of every leaf's every lane, `n_in · lanes` entries,
    /// row after row. `None` where the plan's reverse fails.
    pub(super) fn reverse_flat(&self, seeds: &[f64], rows: usize) -> Option<kj::Buf> {
        let nt = self.n_in * self.layout.lanes();
        let mut p_bar = kj::Buf::new(rows * nt);
        if seeds.len() != rows * self.n_out * self.layout.lanes()
            || !self
                .plan
                .reverse_from(&self.fwd, self.layout, rows, &mut [], &mut p_bar, seeds)
        {
            return None;
        }
        p_bar.iter().all(|v| v.is_finite()).then_some(p_bar)
    }

    /// [`Self::reverse_flat`] for one seed.
    pub(super) fn reverse_one(&self, seed: &[f64]) -> Option<kj::Buf> {
        self.reverse_flat(seed, 1)
    }

    /// One reverse per row of `seeds`, each an owned row ([`Self::reverse_flat`]).
    pub(super) fn reverse(&self, seeds: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let w = self.n_out * self.layout.lanes();
        let mut flat = kj::Buf::new(seeds.len() * w);
        for (r, s) in seeds.iter().enumerate() {
            if s.len() != w {
                return None;
            }
            flat[r * w..(r + 1) * w].copy_from_slice(s);
        }
        let p = self.reverse_flat(&flat, seeds.len())?;
        let nt = (self.n_in * self.layout.lanes()).max(1);
        Some(
            p.chunks_exact(nt)
                .take(seeds.len())
                .map(<[f64]>::to_vec)
                .collect(),
        )
    }

    /// [`Self::reverse`] at the value layout with one seed per output combination: per row
    /// `c`, `Σ_o c[o]·∂out_o/∂leaf` for every leaf.
    pub(super) fn reverse_values(&self, coefs: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        debug_assert_eq!(self.layout.lanes(), 1);
        self.reverse(coefs)
    }
}

/// The subject's kernel plan for configuration `variant` run forward at `layout` on `inputs`
/// (per leaf, `layout.lanes()` entries: the value, then its tangents), recording it from `f`
/// at the leaves' values where no cached variant's guard holds. `f` maps the leaves to the
/// `n_out` outputs; `None` from it — a point outside the kernel's domain, a factorization
/// that fails — declines this call only, never the subject.
#[allow(clippy::too_many_arguments)]
pub(super) fn kernel_run(
    model: &CompiledModel,
    subject: &Subject,
    variant: kj::Desc,
    layout: Layout,
    inputs: &[f64],
    n_out: usize,
    f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>,
) -> Option<KernelRun> {
    use std::sync::atomic::Ordering::Relaxed;
    let l = layout.lanes();
    if inputs.len() % l != 0 {
        return None;
    }
    let n_in = inputs.len() / l;
    // A kernel also bakes in the observed values: subjects of one design key apart by them.
    let key = kj::SubjectKey::new(model, subject, 0, n_in)
        .with_kind(KERNEL_KIND)
        .with_variant(variant)
        .with_observed(subject);
    let run = |plan: &Arc<kj::Plan>| -> Option<KernelRun> {
        let mut out = kj::Buf::new((n_out + 1) * l);
        let fwd = plan.forward_layout(layout, &[], inputs, &mut out)?;
        if out[n_out * l] != 1.0 {
            return None;
        }
        Some(KernelRun {
            plan: plan.clone(),
            fwd,
            layout,
            n_in,
            n_out,
            out,
        })
    };
    match kj::lookup(&key) {
        kj::Lookup::Bad => return None,
        kj::Lookup::Found(ks) => {
            if let Some((i, r)) = ks.order().find_map(|(i, p)| Some((i, run(p)?))) {
                ks.hit(i);
                return Some(r);
            }
            kj::STATS.guard_fail.fetch_add(1, Relaxed);
        }
        kj::Lookup::Missing => {}
    }
    // Record at the leaves' values: how many outputs `f` gave (none: outside the kernel's
    // domain here, nothing to keep and nothing to condemn).
    let given = std::cell::Cell::new(None);
    let rec = kj::recorded(&key.family(), || {
        let leaves: Vec<kj::Tr> = (0..n_in).map(|j| kj::Tr::theta(j, inputs[j * l])).collect();
        let y = f(&leaves);
        given.set(y.as_ref().map(Vec::len));
        y.filter(|y| y.len() == n_out)
            .map(|y| y.iter().map(|x| x.reg()).collect())
    });
    let rec = match (given.get(), rec) {
        (None, _) => return None,
        (Some(n), _) if n != n_out => {
            kj::mark_bad(&key, "the kernel recorded the wrong number of outputs");
            return None;
        }
        (_, Ok(r)) => r,
        (_, Err((why, _))) => {
            kj::mark_bad(&key, &why);
            return None;
        }
    };
    let Some(plan) = rec.plan(Layout::F64, 0, n_in).map(Arc::new) else {
        kj::mark_bad(&key, "the recording could not be planned");
        return None;
    };
    kj::add(&key, plan.clone());
    run(&plan)
}

/// A kernel configuration's [`kj::SubjectKey`] variant: everything that shapes the kernel's
/// code and is not one of its leaves, as its canonical bytes (compared exactly).
pub(super) fn variant_of(parts: &impl std::hash::Hash) -> kj::Desc {
    kj::Desc::of_value(parts)
}

/// A kernel input: a leaf of the plan (`i`, its input slot) or a constant, with its value.
#[derive(Clone, Copy, Debug)]
pub(super) struct Leaf {
    pub(super) v: f64,
    pub(super) i: Option<u32>,
}

impl Leaf {
    /// A constant input.
    pub(super) fn c(v: f64) -> Self {
        Leaf { v, i: None }
    }
    /// The recording's value of this input: its leaf, or the constant.
    pub(super) fn tr(self, leaves: &[kj::Tr]) -> kj::Tr {
        match self.i {
            Some(i) => leaves[i as usize],
            None => <kj::Tr as crate::sens::num::PkNum>::from_f64(self.v),
        }
    }
    /// This input's entry of a reverse row (`0` for a constant).
    pub(super) fn adj(self, row: &[f64]) -> f64 {
        self.i.map_or(0.0, |i| row[i as usize])
    }
}

/// A kernel's inputs, in leaf order.
#[derive(Default)]
pub(super) struct Inputs {
    pub(super) vals: Vec<f64>,
}

impl Inputs {
    /// A new leaf at `v`.
    pub(super) fn leaf(&mut self, v: f64) -> Leaf {
        self.vals.push(v);
        Leaf {
            v,
            i: Some((self.vals.len() - 1) as u32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel plan replays at new leaves: `x₀·x₁ + exp(x₂)` recorded at one point, then run
    /// at others, against its closed-form value and gradient. Dies when a replay returns the
    /// recording point's output (a leaf baked as a constant) or a stale adjoint.
    #[test]
    fn a_kernel_plan_replays_at_new_leaves() {
        use crate::sens::num::PkNum;
        let model = crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TCL(1.0, 0.01, 10.0)\n  theta TV(10.0, 1.0, 100.0)\n  \
             omega ETA_CL ~ 0.1\n  sigma PROP ~ 0.01\n[individual_parameters]\n  \
             CL = TCL * exp(ETA_CL)\n  V = TV\n[structural_model]\n  \
             pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
        )
        .expect("parses");
        let subject = Subject::default();
        let f = |lv: &[kj::Tr]| Some(vec![lv[0] * lv[1] + lv[2].exp()]);
        for x in [[1.0, 2.0, 0.5], [3.0, -1.5, 0.25], [0.5, 4.0, -2.0]] {
            let r = kernel_run(&model, &subject, variant_of(&7u8), Layout::F64, &x, 1, &f)
                .expect("runs");
            let want = x[0] * x[1] + x[2].exp();
            assert_eq!(r.value(0), want, "value at {x:?}");
            let g = r
                .reverse_values(&[vec![1.0]])
                .expect("reverses")
                .pop()
                .unwrap();
            assert_eq!(g, vec![x[1], x[0], x[2].exp()], "gradient at {x:?}");
        }
    }
}
