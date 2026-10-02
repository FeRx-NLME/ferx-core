//! Covariate-aware (multi-theta) mu-referencing M-step for SAEM and IMP/IMPMAP
//! (#619).
//!
//! A typical value that reads several thetas — `CL = (TVCL + (CRCL-90)*TH_CRCL)
//! * exp(ETA_CL)`, `CL = TVCL * (WT/70)^TH_WT * exp(ETA_CL)` — has no single
//! anchor theta, so the closed-form shift `log θ += γ·mean(η)` does not apply and
//! every theta in it used to fall to the **eta-frozen** numerical M-step. That
//! channel maximises the observation likelihood with each subject's sampled `η_i`
//! held fixed, and it is exactly the wrong coordinate system for a covariate
//! slope: once the MH sampler has let `η_i` absorb the covariate-correlated part
//! of the between-subject variation, the slope sees almost no gradient, and the
//! two drift together until the slope reaches a bound (the fluconazole renal
//! gradient of #619 landed on `0`, 480 OFV units above NONMEM).
//!
//! The fix is the mu-referencing augmentation: treat `φ_i = g(A_i(θ)) + η_i` —
//! the individual parameter on the mu scale — as the latent variable instead of
//! `η_i`. Given the E-step's `φ_i`, the complete-data likelihood in the group's
//! thetas is
//!
//! ```text
//!   Σ_i ½ (φ_i − g(A_i(θ)))ᵀ Ω⁻¹ (φ_i − g(A_i(θ)))   +   Σ_i −log p(y_i | φ_i, θ)
//! ```
//!
//! and the θ that minimises it is the M-step. Two engines, chosen per group:
//!
//! - **Exact** ([`CovariateMuGroup::solve_exact`]) when the data term really is
//!   a constant in the group's thetas, which takes *all three* of: every
//!   covariate the typical value reads is constant within each subject, no free
//!   theta of the group is read anywhere else in the model
//!   ([`CovariateMuRef::read_outside_groups`]), **and** no other typical value
//!   reads the group's eta (`CovariateMuRef::eta_shared`). Then `P_i = g⁻¹(φ_i)` does not depend
//!   on θ at all and the M-step is a small nonlinear least-squares fit of
//!   `g(A_i(θ))` to the `φ_i` — solved by Gauss–Newton with Levenberg–Marquardt
//!   damping. For `g(A_i) = log θ + c_i` this reduces to the classical
//!   `log θ += mean(η)`.
//! - **Numerical** ([`CovariateMuGroup::solve_numerical`]) otherwise. A covariate
//!   varying within a subject leaves a θ-dependence through the within-subject
//!   ratio `P_i(t) = g⁻¹(φ_i + g(A_i(t;θ)) − g(A_i(t₀;θ)))`; a theta shared with
//!   another parameter (`CL = (TVCL + TH_X*WT)*exp(ETA_CL)` next to
//!   `V = TVV + TH_X`) leaves one through that other parameter; an eta shared
//!   with another parameter (next to `V = TVV*exp(0.5*ETA_CL)`) leaves one
//!   through the re-centring, which moves that parameter. Any of them and the
//!   data term is kept and the sum above is minimised by a few BOBYQA iterations
//!   over the group's thetas — still in the φ-frozen coordinates, which is what
//!   removes the drift. Dropping a live data term would not be an M-step of
//!   anything, and the group has already pinned those thetas out of the
//!   estimator's general numerical M-step, so nothing else would maximise it.
//!
//! In both cases the caller re-centres each subject's eta by the realised change
//! in its mu, `η_i −= g(A_i(θ_new)) − g(A_i(θ_old))`
//! ([`CovariateMuGroup::recentre_deltas`]), so `φ_i` is unchanged by the M-step
//! (the same bookkeeping the single-anchor shift does with the population mean).
//!
//! **Typical values that share a free theta are one group** (#1620). With
//! `CL = TVCL*(WT/70)^TH_WT*exp(ETA_CL)` and `V1 = TVV1*(WT/70)^TH_WT*exp(ETA_V1)`
//! a step over CL's thetas alone moves `TH_WT` with `η_CL` re-centred and `η_V1`
//! left as drawn, so `φ_V1` absorbs V1's half of the exponent and the step
//! drives `TH_WT` to its bound — the #619 trap again, one parameter over. The
//! group's members are every typical value linked by a shared free theta; its
//! step fits the union of their thetas to all of their `φ_i` under the full
//! `Ω⁻¹` form and re-centres every member's eta. A theta read only by the
//! members is therefore not "elsewhere in the model": once every `φ_i` is held
//! the data no longer depend on it, and the joint group takes the exact engine. With a block Ω the residuals of the *other* eta components
//! enter the quadratic form through the off-diagonal of Ω⁻¹; the exact engine
//! folds them into the target (`φ_i + Σ_{l≠k} (W_kl / W_kk) η_il`), the numerical
//! one evaluates the full form.
//!
//! What this is **not**: a change to `mu_refs`. An eta whose typical value also
//! matches a single-anchor pattern (the power form) keeps that `MuRef` for the
//! inner-loop centring, `suggest_start` and reporting; only the SAEM / IMP
//! M-steps prefer the group. FOCE / FOCEI / Laplace fits are byte-identical with
//! and without a group.

use crate::parser::model_parser::eval_typical_value;
use crate::types::{CompiledModel, CovariateMuRef, MuTransform, Population, Subject};
use nalgebra::{DMatrix, DVector};

/// An IIV variance below this carries no between-subject information about the
/// group's thetas: the prior term pins `φ_i ≈ g(A_i(θ_old))` and the exact
/// engine would return `θ_old` forever (the #411 freeze). Same threshold as the
/// single-anchor guard in `run_mcem`.
pub(crate) const WEAK_GROUP_IIV_VAR: f64 = 1e-3;

/// Levenberg–Marquardt iteration cap for [`CovariateMuGroup::solve_exact`].
const EXACT_MAX_ITER: usize = 40;

/// One covariate mu-reference inside a [`CovariateMuGroup`]: the eta it
/// carries and the typical value that eta is centred on.
pub(crate) struct GroupMember<'m> {
    /// Index into `model.eta_names`.
    pub eta_idx: usize,
    pub transform: MuTransform,
    spec: &'m CovariateMuRef,
}

impl GroupMember<'_> {
    /// `g(A_i(θ))` for one subject: `log A_i` under the lognormal link, `A_i`
    /// itself under the logit link (the typical value is already on the logit
    /// scale). `NaN` when the lognormal typical value is not positive — the
    /// caller treats a `NaN` mu as "this θ is not admissible".
    pub fn mu(&self, theta: &[f64], subject: &Subject) -> f64 {
        let a = eval_typical_value(&self.spec.typical, theta, &subject.covariates);
        match self.transform {
            MuTransform::Log => {
                if a.is_finite() && a > 0.0 {
                    a.ln()
                } else {
                    f64::NAN
                }
            }
            MuTransform::Logit | MuTransform::Identity | MuTransform::LogitProbability => a,
        }
    }

    /// [`Self::mu`] over the population.
    pub fn mus(&self, theta: &[f64], population: &Population) -> Vec<f64> {
        population
            .subjects
            .iter()
            .map(|s| self.mu(theta, s))
            .collect()
    }
}

/// The covariate mu-references one M-step moves together, resolved against the
/// model and the data.
///
/// Usually one member. Mu-references that share a **free** theta are one group
/// (#1620): `CL = TVCL*(WT/70)^TH_WT*exp(ETA_CL)` next to
/// `V1 = TVV1*(WT/70)^TH_WT*exp(ETA_V1)` cannot be stepped one at a time,
/// because the step that moves `TH_WT` while re-centring only `η_CL` leaves
/// `φ_V1` free to absorb the change, and `TH_WT` drifts to a bound (the #619
/// trap on V1's half of the exponent). One step over the union of the thetas,
/// with every member's `φ_i` held fixed, has no such escape.
pub(crate) struct CovariateMuGroup<'m> {
    /// In model declaration order; never empty.
    pub members: Vec<GroupMember<'m>>,
    /// Indices into `model.theta_names`: the union over the members, ascending.
    pub theta_idx: Vec<usize>,
    /// Whether the observation term must be kept in the M-step — i.e. whether
    /// the data still depend on the group's thetas once every member's `φ_i` is
    /// frozen. Decides the engine: exact NLS when `false`, prior + data when
    /// `true`. Each of these sets it, and any one is enough:
    ///
    /// - a covariate a member's typical value reads changes **within** a
    ///   subject, so `P_i(t)` keeps a θ-dependence through the within-subject
    ///   ratio;
    /// - a free theta of the group reaches the likelihood by a route that is
    ///   **not** a recorded mu-reference
    ///   ([`CovariateMuRef::read_outside_groups`]), so no `φ_i` pins it;
    /// - a free theta of the group is read by a recorded mu-reference that is
    ///   **not** a member — one the resolver dropped (weak IIV, a conflicting
    ///   anchor) — so that typical value moves with it unpinned;
    /// - a member's **eta** is read by a second typical value
    ///   (`CovariateMuRef::eta_shared`), so the per-subject re-centring that
    ///   holds this `φ_i` fixed moves that other parameter's prediction.
    pub needs_data_term: bool,
}

/// What the estimator hands a group step: the current point, its bounds, the
/// current Ω, and one eta vector per subject (SAEM's draw, or IMP's posterior
/// mean — the prior term is quadratic, so the mean is sufficient).
pub(crate) struct GroupStepInput<'a> {
    /// Full natural-scale theta at the start of the step.
    pub theta: &'a [f64],
    pub theta_lower: &'a [f64],
    pub theta_upper: &'a [f64],
    pub theta_fixed: &'a [bool],
    /// Per-theta packing, `true` = log (see `theta_packs_log`). Only the
    /// numerical engine optimises in packed coordinates.
    pub theta_packs_log: &'a [bool],
    pub omega: &'a DMatrix<f64>,
    pub etas: &'a [Vec<f64>],
}

/// A covariate mu-reference that passed the per-spec filters of
/// [`resolve_covariate_mu_groups`], before the shared-theta merge.
struct Admitted<'m> {
    eta_idx: usize,
    theta_idx: Vec<usize>,
    spec: &'m CovariateMuRef,
}

/// Resolve `model.covariate_mu_refs` into the groups the M-step will run, and
/// say why any was left out or how it was combined.
///
/// A mu-reference is dropped — its thetas then stay on the numerical M-step
/// exactly as before #619 — when:
///
/// - a theta of it is also the anchor of a single-anchor pair on a
///   **different** eta (`plain_pairs`): two channels moving one theta in the
///   same iteration has no joint optimum;
/// - its eta's initial IIV variance is below [`WEAK_GROUP_IIV_VAR`];
/// - every theta of it is `FIX`ed (nothing to move).
///
/// The single-anchor pair on the mu-reference's **own** eta is not a conflict:
/// it is the same eta, and the caller drops that pair in favour of the group.
///
/// The survivors that share a **free** theta — directly or through a chain,
/// `A{t1}`, `B{t2}`, `C{t1, t2}` — become one group (#1620). A shared `FIX`ed
/// theta links nothing: no step moves it, so it cannot drift.
pub(crate) fn resolve_covariate_mu_groups<'m>(
    model: &'m CompiledModel,
    population: &Population,
    plain_pairs: &[(usize, usize)],
    theta_fixed: &[bool],
    omega: &DMatrix<f64>,
) -> (Vec<CovariateMuGroup<'m>>, Vec<String>) {
    let is_free = |t: usize| !theta_fixed.get(t).copied().unwrap_or(false);
    let theta_name = |t: usize| model.theta_names.get(t).map(String::as_str).unwrap_or("?");
    let mut notes: Vec<String> = Vec::new();

    // 1. The per-mu-reference filters.
    let mut admitted: Vec<Admitted<'m>> = Vec::new();
    for spec in &model.covariate_mu_refs {
        let Some(eta_idx) = model.eta_names.iter().position(|n| n == &spec.eta_name) else {
            continue;
        };
        let theta_idx: Option<Vec<usize>> = spec
            .theta_names
            .iter()
            .map(|n| model.theta_names.iter().position(|t| t == n))
            .collect();
        let Some(theta_idx) = theta_idx else {
            continue;
        };
        let names = spec.theta_names.join(", ");
        if !theta_idx.iter().any(|&t| is_free(t)) {
            continue;
        }
        let shared_with_pair = plain_pairs
            .iter()
            .filter(|&&(_t, e)| e != eta_idx)
            .find(|&&(t, _e)| theta_idx.contains(&t));
        if let Some(&(t, e)) = shared_with_pair {
            notes.push(format!(
                "covariate mu-reference on {} (typical value of {} reads {}) is not used: {} is \
                 also the mu-reference anchor of {}, and one theta cannot take two closed-form \
                 updates in the same iteration; {} stay on the numerical M-step (#619).",
                spec.eta_name,
                spec.eta_name.trim_start_matches("ETA_"),
                names,
                theta_name(t),
                model.eta_names.get(e).map(String::as_str).unwrap_or("?"),
                names
            ));
            continue;
        }
        let var = omega
            .get((eta_idx, eta_idx))
            .copied()
            .unwrap_or(f64::INFINITY);
        if var < WEAK_GROUP_IIV_VAR {
            notes.push(format!(
                "covariate mu-reference on {} (reads {}) is not used: its random effect has \
                 negligible variance (ω² < {WEAK_GROUP_IIV_VAR:.0e}), so the population of \
                 individual values carries no information about the typical value; {} stay on \
                 the numerical M-step (#619).",
                spec.eta_name, names, names
            ));
            continue;
        }
        admitted.push(Admitted {
            eta_idx,
            theta_idx,
            spec,
        });
    }

    // 2. Union-find over the survivors, linking on a shared free theta. The
    //    root of a component is its earliest member, so groups come out in
    //    declaration order of their first member.
    let mut parent: Vec<usize> = (0..admitted.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for a in 0..admitted.len() {
        for b in (a + 1)..admitted.len() {
            let links = admitted[a]
                .theta_idx
                .iter()
                .any(|&t| is_free(t) && admitted[b].theta_idx.contains(&t));
            if links {
                let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut components: Vec<Vec<usize>> = Vec::new();
    let mut component_of_root: Vec<Option<usize>> = vec![None; admitted.len()];
    for a in 0..admitted.len() {
        let r = root(&mut parent, a);
        match component_of_root[r] {
            Some(c) => components[c].push(a),
            None => {
                component_of_root[r] = Some(components.len());
                components.push(vec![a]);
            }
        }
    }

    // 3. One group per component.
    let mut groups: Vec<CovariateMuGroup<'m>> = Vec::with_capacity(components.len());
    for component in components {
        let members: Vec<&Admitted<'m>> = component.iter().map(|&a| &admitted[a]).collect();
        let mut theta_idx: Vec<usize> = members
            .iter()
            .flat_map(|m| m.theta_idx.iter().copied())
            .collect();
        theta_idx.sort_unstable();
        theta_idx.dedup();
        let free: Vec<usize> = theta_idx.iter().copied().filter(|&t| is_free(t)).collect();
        let joint = members.len() > 1;
        let eta_list = join_names(members.iter().map(|m| m.spec.eta_name.as_str()));
        // How a note names the group: one member keeps the historical wording.
        let (subject, keeps, its) = if joint {
            (
                format!("covariate mu-references on {eta_list}"),
                "keep",
                "their joint",
            )
        } else {
            (
                format!(
                    "covariate mu-reference on {} (reads {})",
                    members[0].spec.eta_name,
                    members[0].spec.theta_names.join(", ")
                ),
                "keeps",
                "its",
            )
        };

        if joint {
            let linking: Vec<&str> = free
                .iter()
                .copied()
                .filter(|t| members.iter().filter(|m| m.theta_idx.contains(t)).count() > 1)
                .map(theta_name)
                .collect();
            let described = join_names(members.iter().map(|m| {
                format!(
                    "{} (reads {})",
                    m.spec.eta_name,
                    m.spec.theta_names.join(", ")
                )
            }));
            let all = if members.len() == 2 {
                "both".to_string()
            } else {
                format!("all {} of them", members.len())
            };
            notes.push(format!(
                "covariate mu-references on {described} share {} and take one joint M-step that \
                 re-centres {all} (#1620).",
                linking.join(", ")
            ));
        }

        let time_varying = members.iter().any(|m| {
            m.spec.covariate_names.iter().any(|c| {
                population
                    .subjects
                    .iter()
                    .any(|s| covariate_varies_within(s, c))
            })
        });
        // A free theta that reaches the data by a route no member's `φ_i`
        // pins keeps the data term live even with every covariate
        // subject-constant, so the exact engine — which drops that term — is
        // not an M-step for it. A FIXed one never moves, so no term can be
        // mis-maximised in it.
        let outside: Vec<&str> = free
            .iter()
            .copied()
            .map(theta_name)
            .filter(|n| {
                members
                    .iter()
                    .any(|m| m.spec.read_outside_groups.iter().any(|r| r == n))
            })
            .collect();
        if !outside.is_empty() {
            notes.push(if joint {
                format!(
                    "{subject} {keeps} the observation term in {its} M-step: {} also reach(es) the \
                     data outside these typical values, so freezing the individual parameters \
                     does not make the data independent of it (#1620).",
                    outside.join(", ")
                )
            } else {
                format!(
                    "{subject} {keeps} the observation term in {its} M-step: {} also appear(s) \
                     elsewhere in the model, so freezing the individual parameter does not make \
                     the data independent of them (#619).",
                    outside.join(", ")
                )
            });
        }
        // `read_outside_groups` stops the taint at *every* recorded
        // mu-reference, including the ones step 1 dropped. Those are not held
        // by any `φ_i` this step freezes, so a free theta they read is as live
        // as one read outside every group.
        let mut dropped_reader = false;
        for other in &model.covariate_mu_refs {
            if members.iter().any(|m| std::ptr::eq(m.spec, other)) {
                continue;
            }
            let read: Vec<&str> = free
                .iter()
                .copied()
                .map(theta_name)
                .filter(|n| other.theta_names.iter().any(|o| o == n))
                .collect();
            if read.is_empty() {
                continue;
            }
            dropped_reader = true;
            let is_are = if read.len() == 1 { "is" } else { "are" };
            notes.push(format!(
                "{subject} {keeps} the observation term in {its} M-step: {} {is_are} also read by \
                 the typical value of {}, whose covariate mu-reference is not used (#1620).",
                read.join(", "),
                other.eta_name
            ));
        }
        // The eta-side twin of the same condition: the step re-centres
        // `η_ik`, so a second typical value reading that eta has its
        // prediction moved while `φ_i` stays put, and the data term is live in
        // the group's thetas after all (#918 review).
        for m in members.iter().filter(|m| m.spec.eta_shared) {
            notes.push(format!(
                "covariate mu-reference on {} (reads {}) keeps the observation term in its \
                 M-step: {} is also read by another individual parameter, so re-centring it to \
                 hold this typical value's φ fixed changes that parameter's prediction (#619).",
                m.spec.eta_name,
                m.spec.theta_names.join(", "),
                m.spec.eta_name
            ));
        }
        let eta_shared = members.iter().any(|m| m.spec.eta_shared);
        groups.push(CovariateMuGroup {
            members: members
                .iter()
                .map(|m| GroupMember {
                    eta_idx: m.eta_idx,
                    transform: m.spec.transform,
                    spec: m.spec,
                })
                .collect(),
            theta_idx,
            needs_data_term: time_varying || !outside.is_empty() || dropped_reader || eta_shared,
        });
    }
    (groups, notes)
}

/// Every eta a resolved group re-centres, over all groups and all members.
///
/// The one list the estimators use to drop the single-anchor pairs a group
/// supersedes (SAEM, IMP/IMPMAP) and to tell the #621 not-mu-referenced
/// advisory which etas are covered (`fit.rs`). Listing only each group's first
/// member would leave a joint group's second eta (#1620) both re-centred by the
/// group **and** shifted by its own closed-form pair in the same iteration.
pub(crate) fn group_etas(groups: &[CovariateMuGroup<'_>]) -> Vec<usize> {
    groups.iter().flat_map(|g| g.eta_indices()).collect()
}

/// `a`, `a and b`, `a, b and c`.
fn join_names<S: AsRef<str>>(names: impl Iterator<Item = S>) -> String {
    let names: Vec<S> = names.collect();
    match names.len() {
        0 => String::new(),
        1 => names[0].as_ref().to_string(),
        n => {
            let head: Vec<&str> = names[..n - 1].iter().map(AsRef::as_ref).collect();
            format!("{} and {}", head.join(", "), names[n - 1].as_ref())
        }
    }
}

/// Whether covariate `name` takes more than one value across a subject's record
/// snapshots. `subject.covariates` is the first non-missing value; the per-event
/// snapshots are empty when the dataset has no time-varying covariates at all.
fn covariate_varies_within(subject: &Subject, name: &str) -> bool {
    let base = subject.covariates.get(name).copied();
    let differs = |v: Option<f64>| match (base, v) {
        (Some(b), Some(x)) => !(b == x || (b.is_nan() && x.is_nan())),
        (None, Some(x)) => !x.is_nan(),
        _ => false,
    };
    subject
        .dose_covariates
        .iter()
        .chain(subject.obs_covariates.iter())
        .chain(subject.pk_only_covariates.iter())
        .chain(subject.reset_covariates.iter())
        .any(|m| differs(m.get(name).copied()))
}

/// Row `k` of Ω⁻¹ divided by its diagonal, or `None` when Ω is not
/// invertible / the eta is uncorrelated. Used to fold the other components'
/// residuals into the single-member exact engine's target.
fn cross_weights(k: usize, omega: &DMatrix<f64>) -> Option<Vec<f64>> {
    let n = omega.nrows();
    if k >= n {
        return None;
    }
    // Diagonal Ω: no cross terms, skip the inverse.
    let has_off_diag = (0..n).any(|l| l != k && omega[(k, l)] != 0.0);
    if !has_off_diag {
        return None;
    }
    let w = omega.clone().try_inverse()?;
    let wkk = w[(k, k)];
    if !(wkk.is_finite() && wkk > 0.0) {
        return None;
    }
    Some(
        (0..n)
            .map(|l| if l == k { 0.0 } else { w[(k, l)] / wkk })
            .collect(),
    )
}

/// `Ω⁻¹`, or its diagonal approximation when Ω is singular — the weight the
/// prior quadratic form is evaluated under.
fn prior_weight(omega: &DMatrix<f64>) -> DMatrix<f64> {
    omega.clone().try_inverse().unwrap_or_else(|| {
        let mut d = DMatrix::zeros(omega.nrows(), omega.ncols());
        for j in 0..omega.nrows() {
            d[(j, j)] = 1.0 / omega[(j, j)].max(1e-12);
        }
        d
    })
}

/// Gauss–Newton with Levenberg–Marquardt damping on `residuals` over the free
/// thetas, started at `input.theta` and kept inside the bounds. `residuals`
/// returns the residual vector and its sum of squares, or `None` at an
/// inadmissible θ. Returns the full natural theta vector, or `None` when the
/// starting point itself is inadmissible.
fn levenberg_marquardt(
    input: &GroupStepInput<'_>,
    free: &[usize],
    residuals: impl Fn(&[f64]) -> Option<(Vec<f64>, f64)>,
) -> Option<Vec<f64>> {
    let clamp = |t: usize, v: f64| -> f64 {
        let lo = input
            .theta_lower
            .get(t)
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let hi = input.theta_upper.get(t).copied().unwrap_or(f64::INFINITY);
        v.clamp(lo, hi)
    };

    let mut theta = input.theta.to_vec();
    let (mut r, mut ss) = residuals(&theta)?;
    let n = r.len();
    let d = free.len();
    let mut lambda = 1e-3;
    for _ in 0..EXACT_MAX_ITER {
        // Jacobian of the residual, central differences per free theta.
        let mut jac = DMatrix::<f64>::zeros(n, d);
        for (j, &t) in free.iter().enumerate() {
            let h = 1e-6 * theta[t].abs().max(1.0);
            let mut up = theta.clone();
            up[t] += h;
            let mut dn = theta.clone();
            dn[t] -= h;
            let (Some((ru, _)), Some((rd, _))) = (residuals(&up), residuals(&dn)) else {
                return Some(theta);
            };
            for i in 0..n {
                jac[(i, j)] = (ru[i] - rd[i]) / (2.0 * h);
            }
        }
        let rv = DVector::from_vec(r.clone());
        let jtj = jac.transpose() * &jac;
        let g = jac.transpose() * &rv;
        let mut accepted = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for j in 0..d {
                a[(j, j)] += lambda * jtj[(j, j)].max(1e-12);
            }
            let Some(delta) = a.clone().cholesky().map(|c| c.solve(&(-&g))) else {
                lambda *= 10.0;
                continue;
            };
            let mut trial = theta.clone();
            for (j, &t) in free.iter().enumerate() {
                trial[t] = clamp(t, theta[t] + delta[j]);
            }
            match residuals(&trial) {
                Some((rt, sst)) if sst <= ss => {
                    let step: f64 = free
                        .iter()
                        .map(|&t| (trial[t] - theta[t]).abs() / theta[t].abs().max(1.0))
                        .fold(0.0, f64::max);
                    theta = trial;
                    r = rt;
                    ss = sst;
                    lambda = (lambda * 0.3).max(1e-12);
                    accepted = true;
                    if step < 1e-9 {
                        return Some(theta);
                    }
                    break;
                }
                _ => lambda *= 10.0,
            }
        }
        if !accepted {
            break;
        }
    }
    Some(theta)
}

impl CovariateMuGroup<'_> {
    /// Thetas of the group the step may move.
    fn free_thetas(&self, theta_fixed: &[bool]) -> Vec<usize> {
        self.theta_idx
            .iter()
            .copied()
            .filter(|&t| !theta_fixed.get(t).copied().unwrap_or(false))
            .collect()
    }

    /// Every member's eta index, in member order — the etas the step
    /// re-centres, and the ones whose single-anchor pairs the group supersedes.
    pub fn eta_indices(&self) -> Vec<usize> {
        self.members.iter().map(|m| m.eta_idx).collect()
    }

    /// The member etas for a message: `ETA_CL`, or `ETA_CL and ETA_V1`.
    pub fn eta_names(&self) -> String {
        join_names(self.members.iter().map(|m| m.spec.eta_name.as_str()))
    }

    /// [`GroupMember::mus`] for every member: `mus[m][i]`.
    pub fn mus(&self, theta: &[f64], population: &Population) -> Vec<Vec<f64>> {
        self.members
            .iter()
            .map(|m| m.mus(theta, population))
            .collect()
    }

    /// Whether a group step started from `theta` can do anything at all — the
    /// exact predicate under which [`Self::solve_exact`] and
    /// [`Self::solve_numerical`] both return `None`: no free theta, or a
    /// starting point at which some member's mu is not finite for some subject
    /// (an additive typical value gone ≤ 0 for a low-covariate subject, say).
    ///
    /// Callers that **pin** the group's thetas out of their general M-step must
    /// ask this *before* pinning. IMP/IMPMAP set the pin bounds ahead of the
    /// weighted M-step and only discover the `None` afterwards, at which point
    /// the thetas are frozen for that iteration with no channel left to move
    /// them — they then sit at their initial values through a convergent-looking
    /// fit (#918 review).
    pub fn can_step(&self, theta: &[f64], population: &Population, theta_fixed: &[bool]) -> bool {
        !self.free_thetas(theta_fixed).is_empty()
            && self
                .members
                .iter()
                .all(|m| m.mus(theta, population).iter().all(|v| v.is_finite()))
    }

    /// Add `shifts[m][i]` to member `m`'s coordinate of `eta`, one of subject
    /// `i`'s eta vectors — the φ-preserving shift [`Self::solve_numerical`]
    /// hands its data term, which the caller must apply to every sample it
    /// scores (SAEM: the chain state; IMP: every importance draw). One
    /// implementation for the solver's prior term and both callers, so no
    /// copy can shift only the first member.
    pub fn shift_eta(&self, eta: &mut [f64], shifts: &[Vec<f64>], i: usize) {
        for (member, shift) in self.members.iter().zip(shifts) {
            if let Some(e) = eta.get_mut(member.eta_idx) {
                *e += shift[i];
            }
        }
    }

    /// The re-centring that keeps every member's `φ_i` fixed across a step from
    /// `theta_old` to `theta_new`: `deltas[m][i] = g(A_i(θ_new)) − g(A_i(θ_old))`
    /// for member `m`, which [`Self::recentre_eta`] subtracts from subject `i`'s
    /// eta `members[m].eta_idx`. A non-finite entry means "leave this subject
    /// alone".
    pub fn recentre_deltas(
        &self,
        theta_old: &[f64],
        theta_new: &[f64],
        population: &Population,
    ) -> Vec<Vec<f64>> {
        self.members
            .iter()
            .map(|m| {
                let old = m.mus(theta_old, population);
                let new = m.mus(theta_new, population);
                new.iter().zip(old.iter()).map(|(n, o)| n - o).collect()
            })
            .collect()
    }

    /// Apply [`Self::recentre_deltas`] to `eta`, one of subject `i`'s eta
    /// vectors (SAEM: the chain state and every extra draw; IMP: the proposal
    /// mean), and say whether any coordinate moved. The one place this
    /// bookkeeping is written, for both estimators and every member: a copy
    /// that re-centred only the first member would leave the others' `φ_i` to
    /// absorb the move of a theta they share.
    pub fn recentre_eta(&self, eta: &mut [f64], deltas: &[Vec<f64>], i: usize) -> bool {
        let mut moved = false;
        for (member, delta) in self.members.iter().zip(deltas) {
            match (delta.get(i), eta.get_mut(member.eta_idx)) {
                (Some(&d), Some(e)) if d.is_finite() => {
                    *e -= d;
                    moved = true;
                }
                _ => {}
            }
        }
        moved
    }

    /// Exact M-step for a group whose data term is constant: the θ minimising
    /// the prior quadratic form `Σ_i ½ η'_iᵀ Ω⁻¹ η'_i`, where `η'_i` is `η_i`
    /// with every member's coordinate moved by `g(A_i(θ_old)) − g(A_i(θ))` (the
    /// shift that holds its `φ_i`), by Gauss–Newton with LM damping, started at
    /// `θ_old` and kept inside the bounds. Returns the **full** natural theta
    /// vector with the group's free entries replaced, or `None` when the group
    /// has no free theta or the starting point already has an inadmissible
    /// typical value.
    ///
    /// One member: the residual is `t_i − g(A_i(θ))` with
    /// `t_i = g(A_i(θ_old)) + η_ik + Σ_{l≠k} (W_kl/W_kk) η_il` — the same
    /// minimiser, with the block-Ω cross terms folded into the target. Several:
    /// the residual is `Lᵀ η'_i` with `Ω⁻¹ = L Lᵀ`, which carries the cross
    /// terms between members and to the non-member etas alike.
    pub fn solve_exact(
        &self,
        population: &Population,
        input: &GroupStepInput<'_>,
    ) -> Option<Vec<f64>> {
        let free = self.free_thetas(input.theta_fixed);
        if free.is_empty() {
            return None;
        }
        match self.members.as_slice() {
            [member] => solve_exact_single(member, population, input, &free),
            _ => self.solve_exact_joint(population, input, &free),
        }
    }

    fn solve_exact_joint(
        &self,
        population: &Population,
        input: &GroupStepInput<'_>,
        free: &[usize],
    ) -> Option<Vec<f64>> {
        let mu_old = self.mus(input.theta, population);
        if mu_old.iter().flatten().any(|m| !m.is_finite()) {
            return None;
        }
        let q = input.omega.nrows();
        let w = prior_weight(input.omega);
        let lt = match w.clone().cholesky() {
            Some(c) => c.l().transpose(),
            None => DMatrix::from_diagonal(&w.diagonal().map(|v| v.max(0.0).sqrt())),
        };
        let residuals = |theta: &[f64]| -> Option<(Vec<f64>, f64)> {
            let mut r = Vec::with_capacity(population.subjects.len() * q);
            let mut ss = 0.0;
            for (i, s) in population.subjects.iter().enumerate() {
                let mut e = DVector::<f64>::zeros(q);
                for (l, v) in input.etas[i].iter().take(q).enumerate() {
                    e[l] = *v;
                }
                for (m, member) in self.members.iter().enumerate() {
                    let mu = member.mu(theta, s);
                    if !mu.is_finite() {
                        return None;
                    }
                    if member.eta_idx < q {
                        e[member.eta_idx] += mu_old[m][i] - mu;
                    }
                }
                for v in (&lt * e).iter() {
                    ss += v * v;
                    r.push(*v);
                }
            }
            Some((r, ss))
        };
        levenberg_marquardt(input, free, residuals)
    }

    /// Numerical M-step: minimise the prior quadratic form plus the data term
    /// over the group's free thetas, in packed coordinates, by BOBYQA
    /// warm-started at `θ_old`. `data_nll(theta, shifts)` must return
    /// `−log p(y | φ, θ)` summed over subjects with every sample of subject
    /// `i`'s eta component `members[m].eta_idx` shifted by `shifts[m][i]` — that
    /// is `g(A_i(θ_old)) − g(A_i(θ))` for member `m`, the amount that keeps its
    /// `φ_i` fixed. Every member's shift must be applied: dropping one leaves
    /// that member's individual parameter moving with the shared theta. Returns
    /// the full natural theta vector, or `None` when nothing can move.
    pub fn solve_numerical(
        &self,
        population: &Population,
        input: &GroupStepInput<'_>,
        maxiter: u32,
        data_nll: &dyn Fn(&[f64], &[Vec<f64>]) -> f64,
    ) -> Option<Vec<f64>> {
        let free = self.free_thetas(input.theta_fixed);
        if free.is_empty() {
            return None;
        }
        let n = population.subjects.len();
        let mu_old = self.mus(input.theta, population);
        if mu_old.iter().flatten().any(|m| !m.is_finite()) {
            return None;
        }
        let w = prior_weight(input.omega);
        let packs = |t: usize| input.theta_packs_log.get(t).copied().unwrap_or(true);
        let pack = |t: usize, v: f64| if packs(t) { v.max(1e-10).ln() } else { v };
        let unpack = |t: usize, v: f64| if packs(t) { v.exp() } else { v };
        let d = free.len();
        let x0: Vec<f64> = free.iter().map(|&t| pack(t, input.theta[t])).collect();
        let lower: Vec<f64> = free
            .iter()
            .map(|&t| pack(t, input.theta_lower.get(t).copied().unwrap_or(1e-10)))
            .collect();
        let upper: Vec<f64> = free
            .iter()
            .map(|&t| {
                let hi = input.theta_upper.get(t).copied().unwrap_or(1e9);
                if packs(t) {
                    hi.min(1e9).ln()
                } else {
                    hi
                }
            })
            .collect();

        let base_theta = input.theta.to_vec();
        let theta_at = |xv: &[f64]| -> Vec<f64> {
            let mut theta = base_theta.clone();
            for (j, &t) in free.iter().enumerate() {
                theta[t] = unpack(t, xv[j]);
            }
            theta
        };
        let eval = |xv: &[f64]| -> f64 {
            let theta = theta_at(xv);
            let mu_new = self.mus(&theta, population);
            if mu_new.iter().flatten().any(|m| !m.is_finite()) {
                return 1e20;
            }
            let shifts: Vec<Vec<f64>> = mu_old
                .iter()
                .zip(mu_new.iter())
                .map(|(old, new)| (0..n).map(|i| old[i] - new[i]).collect())
                .collect();
            // Prior term: ½ η'ᵀ W η' with η'_k = η_k + shift for every member k.
            let mut prior = 0.0;
            for (i, eta) in input.etas.iter().enumerate() {
                let mut e = DVector::from_column_slice(eta);
                self.shift_eta(e.as_mut_slice(), &shifts, i);
                prior += 0.5 * (e.transpose() * &w * &e)[(0, 0)];
            }
            let data = data_nll(&theta, &shifts);
            let v = prior + data;
            if v.is_finite() {
                v
            } else {
                1e20
            }
        };
        let start = x0
            .iter()
            .zip(lower.iter().zip(upper.iter()))
            .map(|(&v, (&lo, &hi))| v.clamp(lo, hi))
            .collect::<Vec<f64>>();
        // Best point actually evaluated, recorded as the optimiser goes. What
        // the group step returns is *this*, not whatever `optimize` leaves in
        // `x` — the status is deliberately ignored (a hit maxeval is the normal
        // exit on this budget) and the point must not be trusted on the
        // strength of an exit code either. `eval` reports the `1e20` sentinel
        // for every θ that makes a mu or the NLL non-finite, so on a mostly
        // inadmissible region the search runs over a flat plateau; the group's
        // thetas are pinned out of the estimator's general M-step, so a point
        // worse than θ_old would stand uncorrected for the rest of the fit
        // (#918 review). Tracking the running best costs no extra objective
        // evaluation — the alternative, re-evaluating the start and the result
        // afterwards, costs two full data-term passes per group per iteration.
        let best: std::cell::RefCell<Option<(Vec<f64>, f64)>> = std::cell::RefCell::new(None);
        let obj = |xv: &[f64], _: Option<&mut [f64]>, _: &mut ()| -> f64 {
            let v = eval(xv);
            let mut b = best.borrow_mut();
            // `eval` maps every non-finite value to the sentinel, so `v` is
            // finite here and the comparison is total.
            if b.as_ref().is_none_or(|&(_, bv)| v < bv) {
                *b = Some((xv.to_vec(), v));
            }
            v
        };
        let mut x = start.clone();
        {
            let mut opt = nlopt::Nlopt::new(
                nlopt::Algorithm::Bobyqa,
                d,
                obj,
                nlopt::Target::Minimize,
                (),
            );
            opt.set_lower_bounds(&lower).ok()?;
            opt.set_upper_bounds(&upper).ok()?;
            opt.set_maxeval(maxiter.max(1) * (d as u32 + 2)).ok()?;
            opt.set_ftol_rel(1e-5).ok()?;
            let _ = opt.optimize(&mut x);
        }
        // BOBYQA's first evaluation is its starting point, so `best` is at
        // least as good as `start` whenever the objective ran at all; `None`
        // means it never ran, and there is then nothing better than where we
        // started.
        let chosen = best.into_inner().map(|(xb, _)| xb).unwrap_or(start);
        Some(theta_at(&chosen))
    }
}

/// [`CovariateMuGroup::solve_exact`] for one member: `Σ_i (t_i − g(A_i(θ)))²`
/// with the block-Ω cross terms folded into the constant target `t_i`.
fn solve_exact_single(
    member: &GroupMember<'_>,
    population: &Population,
    input: &GroupStepInput<'_>,
    free: &[usize],
) -> Option<Vec<f64>> {
    let n = population.subjects.len();
    let mu_old = member.mus(input.theta, population);
    if mu_old.iter().any(|m| !m.is_finite()) {
        return None;
    }
    let cross = cross_weights(member.eta_idx, input.omega);
    let target: Vec<f64> = (0..n)
        .map(|i| {
            let eta_i = &input.etas[i];
            let mut t = mu_old[i] + eta_i.get(member.eta_idx).copied().unwrap_or(0.0);
            if let Some(cw) = &cross {
                for (l, &c) in cw.iter().enumerate() {
                    if c != 0.0 {
                        t += c * eta_i.get(l).copied().unwrap_or(0.0);
                    }
                }
            }
            t
        })
        .collect();
    let residuals = |theta: &[f64]| -> Option<(Vec<f64>, f64)> {
        let mut r = Vec::with_capacity(n);
        let mut ss = 0.0;
        for (i, s) in population.subjects.iter().enumerate() {
            let m = member.mu(theta, s);
            if !m.is_finite() {
                return None;
            }
            let ri = target[i] - m;
            ss += ri * ri;
            r.push(ri);
        }
        Some((r, ss))
    };
    levenberg_marquardt(input, free, residuals)
}

#[cfg(test)]
#[path = "covariate_mu_ref_tests.rs"]
mod tests;
