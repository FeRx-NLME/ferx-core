//! Analytic `∂/∂w` for `[covariate_nn]` weight thetas in the fixed-η
//! observation-NLL gradient.
//!
//! # The problem
//!
//! [`crate::estimation::fixed_eta_gradient`] gets its θ-gradient by finite
//! differences: one perturbed solve of the whole subject per non-pinned θ. That
//! is fine at classical-PopPK scale (a handful of θ), and hopeless for a deep
//! compartment model, where the NN's weights *are* θ. On a 146-θ DCM (2 inputs →
//! 8 → 8 → 5 outputs) it made variational inference cost 720 s against FOCEI's
//! 42 s on the same model, with the gap linear in the weight count.
//!
//! # The decomposition
//!
//! The network enters the likelihood through exactly one narrow channel: its
//! output vector. Write `z` for the output layer's pre-activations, so the NN's
//! whole influence on subject `i`'s NLL factors through `z`:
//!
//! ```text
//! ∂NLL/∂w_j = Σ_k (∂NLL/∂z_k) · (∂z_k/∂w_j)
//! ```
//!
//! The second factor is exact and cheap — it is
//! [`MlpMapper::jacobian_preactivation`](crate::nn::MlpMapper::jacobian_preactivation),
//! one reverse-mode sweep per output. Only the first factor needs the model
//! solved, and there are `n_outputs` of them, not `n_weights`.
//!
//! Getting that first factor needs no new plumbing, because the output-layer
//! bias `b_k` is already a θ and already reaches the NLL through `z_k` alone:
//!
//! ```text
//! z_k = (W_L a_{L-1} + b_L)_k     ⇒     ∂z_k/∂b_k = 1,   ∂z_j/∂b_k = 0  (j ≠ k)
//! ```
//!
//! so a finite difference of the objective in `b_k` — the *same* FD the caller
//! already knows how to take for any θ — **is** `∂NLL/∂z_k`, with no correction
//! factor. On the DCM above that is 5 solves instead of 141.
//!
//! The result is not an approximation of the FD gradient; it is more accurate
//! than it. Every weight-specific factor is exact, and the only FD error left is
//! the `n_outputs` directional derivatives, shared across all weights.
//!
//! # Why the pre-activation, and not the output
//!
//! Seeding the Jacobian at `z_L` rather than at the activated output `a_L` is
//! what makes the bias identity unconditional. The post-activation form of the
//! same argument yields `∂NLL/∂b_k = (∂NLL/∂a_k)·f'(z_k)`, which must be divided
//! through by `f'(z_k)` — and that division fails exactly where it matters, on a
//! saturated `Softplus`/`Sigmoid` head where `f'(z_k)` underflows. Working in
//! `z` cancels the factor analytically instead of numerically.
//!
//! # Scope
//!
//! The factorization needs the network to see **one input vector for the whole
//! subject**, so that a single `z` mediates every observation. A time-varying NN
//! input breaks that: there is then a distinct `z^{(e)}` per event, the bias FD
//! returns `Σ_e ∂NLL/∂z_k^{(e)}`, and the per-event terms are not recoverable
//! from it. [`NnGradPlan::build`] declines in that case and the caller keeps its
//! per-θ FD loop — the CLAUDE.md "route to FD via a support predicate, and
//! unit-test the routing" rule. Non-NN θ are never touched by this module, so a
//! model with no `[covariate_nn]` block is bit-for-bit unaffected.

use crate::types::{CompiledModel, Subject};
use nalgebra::DMatrix;
#[cfg(feature = "nn")]
use std::collections::HashMap;

#[cfg(all(test, feature = "nn"))]
#[path = "nn_theta_gradient_tests.rs"]
mod tests;

/// One `[covariate_nn]` block's contribution, resolved at this subject's
/// covariates.
struct NnBlock {
    /// First θ index of this block's contiguous weight run.
    weights_offset: usize,
    /// Length of that run.
    n_weights: usize,
    /// θ index of the output-layer bias for each output (length `n_outputs`).
    /// These are the only coordinates this block finite-differences.
    bias_theta_idx: Vec<usize>,
    /// `∂z_L/∂w`, shape `(n_outputs × n_weights)`.
    jz: DMatrix<f64>,
}

/// The set of NN weight blocks whose θ-gradient can be assembled analytically
/// for one `(model, subject)` pair.
///
/// Build once per subject per gradient call; the Jacobian depends on the
/// subject's covariates and on the current weights, so it cannot be cached
/// across optimizer iterations.
pub(crate) struct NnGradPlan {
    blocks: Vec<NnBlock>,
    /// `covered[i]` marks θ index `i` as owned by this plan, so the caller's
    /// generic FD loop skips it. Length `n_theta`.
    covered: Vec<bool>,
}

impl NnGradPlan {
    /// Resolve the plan, or `None` when the caller must fall back to its per-θ
    /// FD loop.
    ///
    /// Declines when: the model has no `[covariate_nn]` block; any NN input
    /// covariate is not constant across this subject's records (see the module
    /// docs); or the Jacobian cannot be built (a weight-offset wiring bug —
    /// declining keeps it a slowdown rather than a wrong gradient).
    ///
    /// Without the `nn` feature there are no NN blocks to serve and this is a
    /// constant `None`, so the two call sites need no feature gate of their
    /// own — they keep exactly the FD loop they had.
    #[cfg(not(feature = "nn"))]
    pub(crate) fn build(
        _model: &CompiledModel,
        _subject: &Subject,
        _theta: &[f64],
        _n_theta: usize,
    ) -> Option<Self> {
        None
    }

    #[cfg(feature = "nn")]
    pub(crate) fn build(
        model: &CompiledModel,
        subject: &Subject,
        theta: &[f64],
        n_theta: usize,
    ) -> Option<Self> {
        if model.covariate_nns.is_empty() {
            return None;
        }

        let mut blocks = Vec::with_capacity(model.covariate_nns.len());
        let mut covered = vec![false; n_theta];

        for nn in &model.covariate_nns {
            let n_weights = nn.mapper.mlp().n_weights();
            let end = nn.weights_offset + n_weights;
            // A block running past either the caller's θ count or the θ slice
            // itself is a wiring bug; decline rather than index out of bounds.
            if end > n_theta || end > theta.len() {
                return None;
            }
            let covariates = static_nn_input_map(subject, nn.mapper.input_names())?;
            let weights = &theta[nn.weights_offset..end];
            let jz = nn
                .mapper
                .jacobian_preactivation_raw(weights, covariates)
                .ok()?;

            let bias_theta_idx = (0..nn.mapper.mlp().n_outputs())
                .map(|k| nn.weights_offset + nn.mapper.mlp().output_bias_index(k))
                .collect();

            for c in covered.iter_mut().take(end).skip(nn.weights_offset) {
                *c = true;
            }
            blocks.push(NnBlock {
                weights_offset: nn.weights_offset,
                n_weights,
                bias_theta_idx,
                jz,
            });
        }

        Some(Self { blocks, covered })
    }

    /// True when θ index `i` is assembled by this plan and must be skipped by
    /// the caller's generic FD loop.
    #[inline]
    pub(crate) fn covers(&self, i: usize) -> bool {
        self.covered.get(i).copied().unwrap_or(false)
    }

    /// Write every covered θ's gradient entry into `grad`.
    ///
    /// `signed_deriv(i, sign)` must return the one-sided difference quotient
    /// `(F(θ + δ·e_i) − F(θ)) / δ` for the caller's objective `F`, with
    /// `δ = sign · h_i` and `h_i` the caller's own step for coordinate `i`.
    /// Passing `sign = +1` must reproduce exactly what the caller's per-θ FD
    /// loop computes, so that a model with no NN block is unaffected.
    ///
    /// # Central, not forward
    ///
    /// This evaluates each direction on **both** sides and averages:
    ///
    /// ```text
    /// (F(θ+h) − F(θ−h)) / 2h  =  ½ · [ s(+h) + s(−h) ]
    /// ```
    ///
    /// which is the identity that lets a central difference be assembled from
    /// two one-sided quotients. That upgrades the last remaining source of
    /// error from `O(h)` to `O(h²)`, and it is affordable *because* of the
    /// decomposition: the cost is `2 · n_outputs` solves per block, against
    /// `n_weights` for the loop this replaces — on the 5-output / 141-weight
    /// reference DCM, 10 solves instead of 141. Spending the forward-FD budget
    /// on accuracy rather than pocketing it is the right trade when the
    /// remaining term is shared by every weight in the block.
    ///
    /// `lower[i] == upper[i]` marks a pinned coordinate, which is left at zero.
    /// A pinned *bias* is still differenced: pinning suppresses that θ's
    /// reported gradient entry, it does not make `∂NLL/∂z_k` — which every
    /// other weight in the block needs — cease to exist.
    pub(crate) fn accumulate(
        &self,
        mut signed_deriv: impl FnMut(usize, f64) -> f64,
        theta: &[f64],
        theta_packs_log_mask: &[bool],
        lower: &[f64],
        upper: &[f64],
        grad: &mut [f64],
    ) {
        for block in &self.blocks {
            let range = block.weights_offset..block.weights_offset + block.n_weights;
            // Every weight pinned ⇒ nothing to report, so skip the solves too.
            if range.clone().all(|i| lower[i] == upper[i]) {
                continue;
            }

            let dz: Vec<f64> = block
                .bias_theta_idx
                .iter()
                .map(|&i| 0.5 * (signed_deriv(i, 1.0) + signed_deriv(i, -1.0)))
                .collect();

            for j in 0..block.n_weights {
                let g = block.weights_offset + j;
                if lower[g] == upper[g] {
                    continue;
                }
                let raw: f64 = dz
                    .iter()
                    .enumerate()
                    .map(|(k, &d)| d * block.jz[(k, j)])
                    .sum();
                grad[g] = if theta_packs_log_mask[g] {
                    theta[g] * raw
                } else {
                    raw
                };
            }
        }
    }
}

/// The single covariate map this subject's NN sees, or `None` if there isn't
/// one.
///
/// Returns a map only when every name in `input_names` resolves to the *same*
/// `Option<f64>` across every snapshot `pk_param_fn` could be handed: the
/// subject-static map and each per-dose / per-observation / per-EVID2 LOCF
/// snapshot.
///
/// Comparing `Option<f64>` rather than `f64` is deliberate — a name present in
/// one snapshot and absent from another is a genuine difference, because
/// `NamedMlpMapper::forward_raw` substitutes the centered origin for an absent
/// input. Exact `f64` equality is the right test for the same reason
/// [`Subject::time_varying_covariate_names`] uses it: snapshots are LOCF copies
/// of one parsed value, so an unchanged covariate is bit-identical.
///
/// Checking the maps directly, rather than mirroring the dispatch in
/// `subject_needs_per_event_pk`, is what keeps this predicate honest: it stays
/// correct regardless of which prediction path the subject ends up on.
#[cfg(feature = "nn")]
fn static_nn_input_map<'a>(
    subject: &'a Subject,
    input_names: &[String],
) -> Option<&'a HashMap<String, f64>> {
    let base = &subject.covariates;
    let snapshots = subject
        .obs_covariates
        .iter()
        .chain(subject.dose_covariates.iter())
        .chain(subject.pk_only_covariates.iter());

    let mut representative = base;
    let mut seen_snapshot = false;
    for snap in snapshots {
        if !seen_snapshot {
            // The per-event maps are what the NN actually reads whenever the
            // subject takes the per-event path, so one of them — not the
            // subject-static map — is the representative.
            representative = snap;
            seen_snapshot = true;
        }
        for name in input_names {
            if snap.get(name) != representative.get(name) {
                return None;
            }
        }
    }

    // With snapshots present, the static map is only reached when a subject has
    // no records of a given kind; require it to agree too rather than reason
    // about which of the two a given event resolves to.
    if seen_snapshot {
        for name in input_names {
            if base.get(name) != representative.get(name) {
                return None;
            }
        }
    }

    Some(representative)
}
