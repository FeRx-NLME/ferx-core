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
pub struct KernelRun {
    plan: Arc<kj::Plan>,
    fwd: kj::Fwd,
    layout: Layout,
    n_in: usize,
    n_out: usize,
    /// Per output, its lanes (`layout.lanes()` each).
    pub(super) out: kj::Buf,
}

impl KernelRun {
    /// The plan this run ran: a caller that runs it again skips the lookup.
    pub(super) fn plan(&self) -> &Arc<kj::Plan> {
        &self.plan
    }

    /// `plan` run forward at `layout` on `inputs` (as [`kernel_run`]'s run).
    pub(super) fn of_plan(
        plan: &Arc<kj::Plan>, layout: Layout, inputs: &[f64], n_out: usize,
    ) -> Option<Self> {
        let l = layout.lanes();
        let mut out = kj::Buf::new((n_out + 1) * l);
        let fwd = plan.forward_layout(layout, &[], inputs, &mut out)?;
        if out[n_out * l] != 1.0 {
            return None;
        }
        Some(KernelRun { plan: plan.clone(), fwd, layout, n_in: inputs.len() / l, n_out, out })
    }

    /// Output `o`'s value.
    pub fn value(&self, o: usize) -> f64 {
        self.out[o * self.layout.lanes()]
    }

    /// One reverse seeded with `seeds` (`rows · n_out · lanes` entries, laid out like the
    /// outputs): per row, the adjoint of every leaf's every lane, `n_in · lanes` entries,
    /// row after row. `None` where the plan's reverse fails.
    pub(super) fn reverse_flat(&self, seeds: &[f64], rows: usize) -> Option<kj::Buf> {
        let nt = self.n_in * self.layout.lanes();
        let mut p_bar = kj::Buf::new(rows * nt);
        if seeds.len() != rows * self.n_out * self.layout.lanes()
            || !self.plan.reverse_from(&self.fwd, self.layout, rows, &mut [], &mut p_bar, seeds)
        {
            return None;
        }
        p_bar.iter().all(|v| v.is_finite()).then_some(p_bar)
    }

    /// Forward over reverse (#2980) from a forward at a tangent layout: one sweep seeded with
    /// `seed` (laid out like the outputs, value and tangent lanes), giving every leaf's lanes
    /// laid out the same way — the value lanes' adjoints and their tangents along each
    /// direction ([`kj::Plan::reverse_tangent_from`]).
    pub(super) fn reverse_tangent(&self, seed: &[f64]) -> Option<kj::Buf> {
        self.reverse_tangent_flat(seed, 1)
    }

    /// [`Self::reverse_tangent`] for `rows` seeds at once (laid out one after another), the
    /// recomputed forward shared: per row, every leaf's lanes.
    pub(super) fn reverse_tangent_flat(&self, seeds: &[f64], rows: usize) -> Option<kj::Buf> {
        let mut p_bar = kj::Buf::new(rows * self.n_in * self.layout.lanes());
        if seeds.len() != rows * self.n_out * self.layout.lanes()
            || !self.plan.reverse_tangent_from(&self.fwd, rows, &mut [], &mut p_bar, seeds)
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
        Some(p.chunks_exact(nt).take(seeds.len()).map(<[f64]>::to_vec).collect())
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
    model: &CompiledModel, subject: &Subject, variant: kj::Desc, layout: Layout, inputs: &[f64],
    n_out: usize, f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>,
) -> Option<KernelRun> {
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
        Some(KernelRun { plan: plan.clone(), fwd, layout, n_in, n_out, out })
    };
    // The one lookup-and-record path of every plan kind ([`plan_or_record`]): a failed recording
    // settles by its scope, and a variant that does not run at its own point is marked.
    let record = |key: &kj::SubjectKey| -> Option<(Arc<kj::Plan>, bool)> {
        // Record at the leaves' values: how many outputs `f` gave (none: outside the kernel's
        // domain here, nothing to keep and nothing to condemn).
        let given = std::cell::Cell::new(None);
        let sane = std::cell::Cell::new(true);
        let rec = kj::recorded(&key.family(), || {
            let leaves: Vec<kj::Tr> = (0..n_in).map(|j| kj::Tr::theta(j, inputs[j * l])).collect();
            let y = f(&leaves);
            given.set(y.as_ref().map(Vec::len));
            y.filter(|y| y.len() == n_out).map(|y| {
                sane.set(y.iter().all(|x| x.recorded().is_finite()));
                y.iter().map(|x| x.reg()).collect()
            })
        });
        let rec = match (given.get(), rec) {
            (None, _) => return None,
            (Some(n), _) if n != n_out => {
                kj::settle_failure(
                    key,
                    "the kernel recorded the wrong number of outputs",
                    kj::FailScope::Subject,
                );
                return None;
            }
            (_, Ok(r)) => r,
            (_, Err((why, scope))) => {
                kj::settle_failure(key, &why, scope);
                return None;
            }
        };
        let finite = rec.finite();
        let Some(plan) = rec.plan(Layout::F64, 0, n_in).map(Arc::new) else {
            kj::mark_bad(key, "the recording could not be planned");
            return None;
        };
        Some((plan, sane.get() && finite))
    };
    plan_or_record(
        model,
        subject,
        None,
        None,
        &|| key.clone(),
        &|_| true,
        &record,
        true,
        &mut |plan| run(plan),
    )
}

/// `kernel_run` for a kernel that reads no subject data — its leaves carry everything that
/// differs between subjects, and `variant` everything its recording bakes in (layout, the
/// parameter values it reads as literals): one plan per configuration serves every subject, so
/// a subject neither records nor first-tier runs it (#2980: the structured covariance kernel's
/// reduce). A thread's plans are kept in a small table, oldest dropped first.
pub fn kernel_run_shared(
    model: &CompiledModel, subject: &Subject, variant: kj::Desc, layout: Layout, inputs: &[f64],
    n_out: usize, f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>,
) -> Option<KernelRun> {
    const KEEP: usize = 64;
    thread_local! {
        static SHARED: std::cell::RefCell<Vec<(kj::SubjectKey, Arc<kj::Plan>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let l = layout.lanes();
    if inputs.len() % l != 0 {
        return None;
    }
    let n_in = inputs.len() / l;
    let key = kj::SubjectKey::new(model, subject, 0, n_in)
        .with_kind(KERNEL_KIND)
        .with_variant(variant)
        .family();
    // Every plan recorded under the key: a recorded program carries the branches its recording
    // point took as guards, so a point on another branch is served by the plan recorded there.
    let cached: Vec<Arc<kj::Plan>> = SHARED
        .with(|c| c.borrow().iter().filter(|(k, _)| *k == key).map(|(_, p)| p.clone()).collect());
    for plan in &cached {
        if let Some(kr) = KernelRun::of_plan(plan, layout, inputs, n_out) {
            return Some(kr);
        }
    }
    let rec = kj::recorded(&key, || {
        let leaves: Vec<kj::Tr> = (0..n_in).map(|j| kj::Tr::theta(j, inputs[j * l])).collect();
        f(&leaves).filter(|y| y.len() == n_out).map(|y| y.iter().map(|x| x.reg()).collect())
    })
    .ok()?;
    let p = Arc::new(rec.plan(Layout::F64, 0, n_in)?);
    SHARED.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() >= KEEP {
            c.remove(0);
        }
        c.push((key, p.clone()));
    });
    KernelRun::of_plan(&p, layout, inputs, n_out)
}

/// A kernel configuration's [`kj::SubjectKey`] variant: everything that shapes the kernel's
/// code and is not one of its leaves, as its canonical bytes (compared exactly).
pub fn variant_of(parts: &impl std::hash::Hash) -> kj::Desc {
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
    /// [`Self::adj`] over a row computed per entry.
    pub(super) fn adj_with(self, row: impl FnOnce(usize) -> f64) -> f64 {
        self.i.map_or(0.0, |i| row(i as usize))
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
        Leaf { v, i: Some((self.vals.len() - 1) as u32) }
    }
}

#[cfg(test)]
pub mod tests {
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
            let g = r.reverse_values(&[vec![1.0]]).expect("reverses").pop().unwrap();
            assert_eq!(g, vec![x[1], x[0], x[2].exp()], "gradient at {x:?}");
        }
    }
    fn test_model() -> CompiledModel {
        crate::parser::model_parser::parse_model_string(
            "[parameters]\n  theta TCL(1.0, 0.01, 10.0)\n  theta TV(10.0, 1.0, 100.0)\n  \
             omega ETA_CL ~ 0.1\n  sigma PROP ~ 0.01\n[individual_parameters]\n  \
             CL = TCL * exp(ETA_CL)\n  V = TV\n[structural_model]\n  \
             pk one_cpt_iv(cl=CL, v=V)\n[error_model]\n  DV ~ proportional(PROP)\n",
        )
        .expect("parses")
    }

    /// The key `kernel_run` files a kernel under, for one input and `variant`.
    fn key_of(model: &CompiledModel, subject: &Subject, variant: &kj::Desc) -> kj::SubjectKey {
        kj::SubjectKey::new(model, subject, 0, 1)
            .with_kind(KERNEL_KIND)
            .with_variant(variant.clone())
            .with_observed(subject)
    }

    /// A kernel recording that fails at this point's values is the point's failure: declined and
    /// counted, never condemned, so the next call records and runs (#2786 round 2, finding 4:
    /// `kernel_run` had a lookup of its own that marked every scope bad). Mutation: settle with
    /// `mark_bad` for every scope in `kernel_run` -> the key is `Bad` after the first call.
    #[test]
    fn a_point_failure_in_a_kernel_declines_without_condemning() {
        let (model, subject) = (test_model(), Subject::default());
        let variant = variant_of(&0x4e_0001u32);
        let key = key_of(&model, &subject, &variant);
        let at = |x: f64, f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>| {
            kernel_run(&model, &subject, variant.clone(), Layout::F64, &[x], 1, f)
        };
        let fails = |lv: &[kj::Tr]| {
            kj::fail_here("a record-time run failed at this point");
            Some(vec![lv[0] * lv[0]])
        };
        assert!(at(2.0, &fails).is_none());
        assert!(matches!(kj::lookup(&key), kj::Lookup::Missing), "not condemned");
        let ok = |lv: &[kj::Tr]| Some(vec![lv[0] * lv[0]]);
        let r = at(3.0, &ok).expect("re-served");
        assert_eq!(r.value(0), 9.0);
        assert!(matches!(kj::lookup(&key), kj::Lookup::Found(_)));
    }

    /// A kernel recording that fails for a reason of the model's marks the model, so another
    /// variant of the same model is declined from the model's mark without recording (the
    /// `model_bad` check on a missing key). Mutation: skip `mark_model_bad` in the `Model` arm
    /// or the `model_bad` check in `plan_or_record` -> the second variant records and runs.
    #[test]
    fn a_model_failure_in_a_kernel_marks_the_model() {
        let (model, subject) = (test_model(), Subject::default());
        let (va, vb) = (variant_of(&0x4e_0002u32), variant_of(&0x4e_0003u32));
        let run = |v: &kj::Desc, f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>| {
            kernel_run(&model, &subject, v.clone(), Layout::F64, &[2.0], 1, f)
        };
        let fails = |lv: &[kj::Tr]| {
            kj::fail("an operation the recorder cannot take".into());
            Some(vec![lv[0]])
        };
        assert!(run(&va, &fails).is_none());
        assert!(matches!(kj::lookup(&key_of(&model, &subject, &va)), kj::Lookup::Bad));
        let key_b = key_of(&model, &subject, &vb);
        assert!(kj::model_bad(&key_b).is_some(), "the model is marked");
        let ok = |lv: &[kj::Tr]| Some(vec![lv[0] * lv[0]]);
        assert!(run(&vb, &ok).is_none(), "declined from the model's mark");
        assert!(matches!(kj::lookup(&key_b), kj::Lookup::Bad));
    }

    /// Every `None` of a kernel's `f` is a point outside the objective's domain (a non-positive
    /// or non-finite variance, a non-PD Ω or `H`, a custom residual magnitude out of range: the
    /// value path scores a sentinel there, which has no derivative): `kernel_run` declines the
    /// call, records nothing and condemns nothing, so the next valid point records and serves.
    /// Mutation: settle a `None` from `f` as a subject's failure (`mark_bad`) -> the key reads
    /// `Bad` and the valid call is declined.
    #[test]
    fn a_domain_decline_in_a_kernel_condemns_nothing() {
        let (model, subject) = (test_model(), Subject::default());
        let variant = variant_of(&0x4e_0004u32);
        let key = key_of(&model, &subject, &variant);
        let at = |x: f64, f: &dyn Fn(&[kj::Tr]) -> Option<Vec<kj::Tr>>| {
            kernel_run(&model, &subject, variant.clone(), Layout::F64, &[x], 1, f)
        };
        // `1/x` outside its domain at `x = 0`: the objective's own decline (`None`).
        let f = |lv: &[kj::Tr]| {
            if lv[0].recorded() == 0.0 {
                None
            } else {
                Some(vec![<kj::Tr as crate::sens::num::PkNum>::from_f64(1.0) / lv[0]])
            }
        };
        assert!(at(0.0, &f).is_none());
        assert!(matches!(kj::lookup(&key), kj::Lookup::Missing), "nothing kept, nothing condemned");
        let r = at(4.0, &f).expect("served at a valid point");
        assert_eq!(r.value(0), 0.25);
        assert!(
            matches!(kj::lookup(&key), kj::Lookup::Found(_)),
            "the plan recorded at the valid point is kept"
        );
        assert_eq!(at(2.0, &f).expect("and serves again").value(0), 0.5);
    }
}
