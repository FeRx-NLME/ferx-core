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
    pub(super) out: Vec<f64>,
}

impl KernelRun {
    /// Output `o`'s value.
    pub(super) fn value(&self, o: usize) -> f64 {
        self.out[o * self.layout.lanes()]
    }

    /// One reverse seeded with each row of `seeds` (`n_out · lanes` entries per row, laid out
    /// like the outputs): per row, the adjoint of every leaf's every lane, `n_in · lanes`
    /// entries. `None` where the plan's reverse fails.
    pub(super) fn reverse(&self, seeds: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let l = self.layout.lanes();
        let (rows, w) = (seeds.len(), self.n_out * l);
        let mut du_bar = vec![0.0; rows * w];
        for (r, s) in seeds.iter().enumerate() {
            if s.len() != w {
                return None;
            }
            du_bar[r * w..(r + 1) * w].copy_from_slice(s);
        }
        let nt = self.n_in * l;
        let mut p_bar = vec![0.0; rows * nt];
        if !self
            .plan
            .reverse_from(&self.fwd, self.layout, rows, &mut [], &mut p_bar, &du_bar)
        {
            return None;
        }
        let out: Vec<Vec<f64>> = p_bar
            .chunks_exact(nt.max(1))
            .take(rows)
            .map(<[f64]>::to_vec)
            .collect();
        out.iter().flatten().all(|v| v.is_finite()).then_some(out)
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
    variant: u64,
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
    let key = kj::SubjectKey::new(model, subject, 0, n_in)
        .with_kind(KERNEL_KIND)
        .with_variant(variant);
    let run = |plan: &Arc<kj::Plan>| -> Option<KernelRun> {
        let mut out = vec![0.0; (n_out + 1) * l];
        let fwd = plan.forward_layout(layout, &[], inputs, &mut out)?;
        if out[n_out * l] != 1.0 {
            return None;
        }
        out.truncate(n_out * l);
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
            if let Some(r) = ks.iter().find_map(run) {
                // TEMP #2786 diagnosis: a fresh recording at these leaves must agree.
                if std::env::var("FERX_TMP_KVERIFY").is_ok() {
                    kj::begin();
                    let lv: Vec<kj::Tr> =
                        (0..n_in).map(|j| kj::Tr::theta(j, inputs[j * l])).collect();
                    let y = f(&lv);
                    let fresh: Option<Vec<f64>> = y.map(|y| y.iter().map(|x| x.v_dbg()).collect());
                    let _ = kj::end(None);
                    if let Some(fr) = fresh {
                        for (o, (a, b)) in fr.iter().zip(r.out.iter().step_by(l)).enumerate() {
                            if a.to_bits() != b.to_bits() {
                                eprintln!("TMPKV variant {variant:x} out {o}: fresh {a} plan {b}");
                                break;
                            }
                        }
                    }
                }
                return Some(r);
            }
            kj::STATS.guard_fail.fetch_add(1, Relaxed);
        }
        kj::Lookup::Missing => {}
    }
    // Record at the leaves' values.
    kj::begin();
    let leaves: Vec<kj::Tr> = (0..n_in).map(|j| kj::Tr::theta(j, inputs[j * l])).collect();
    let y = f(&leaves);
    let Some(y) = y else {
        // Outside the kernel's domain here: nothing to keep, nothing to condemn.
        let _ = kj::end(None);
        return None;
    };
    if y.len() != n_out {
        let _ = kj::end(None);
        kj::mark_bad(&key);
        return None;
    }
    let rec = match kj::end(Some(y.iter().map(|x| x.reg()).collect())) {
        Ok(r) => r,
        Err(why) => {
            if kj::debug() {
                eprintln!("kjit: kernel not recorded: {why}");
            }
            kj::mark_bad(&key);
            return None;
        }
    };
    let Some(plan) = rec.plan(Layout::F64, 0, n_in).map(Arc::new) else {
        kj::mark_bad(&key);
        return None;
    };
    kj::add(&key, plan.clone());
    let r = run(&plan);
    if r.is_none() {
        // A fresh recording that does not serve at its own point never will.
        kj::mark_bad(&key);
    }
    r
}

/// A kernel configuration's [`kj::SubjectKey`] variant: everything that shapes the kernel's
/// code and is not one of its leaves, hashed.
pub(super) fn variant_of(parts: &impl std::hash::Hash) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    h.finish()
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
