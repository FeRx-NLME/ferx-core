//! Option B (explicit symbolic derivatives) for the 3-cpt IV-bolus solution.
//!
//! The 3-cpt response routes through the three disposition eigenvalues `α, β, γ`
//! — the roots of the characteristic cubic
//!
//! ```text
//!   p(λ) = λ³ − e₁λ² + e₂λ − e₃ = 0,
//!   e₁ = k10+k12+k13+k21+k31,                      (= α+β+γ)
//!   e₂ = k10k21 + k10k31 + k21k31 + k12k31 + k13k21,  (= αβ+αγ+βγ)
//!   e₃ = k10·k21·k31.                                  (= αβγ)
//! ```
//!
//! The generic [`super::three_cpt`] path solves the cubic trigonometrically
//! (`acos`/`cos`) and lets [`Dual2`](super::dual2::Dual2) carry the derivatives
//! through that transcendental solve. Here we instead obtain the root
//! first/second derivatives in **closed form** by implicit differentiation of
//! `p(λ)=0`. Differentiating once w.r.t. parameter `i`:
//!
//! ```text
//!   p'(λ)·λ'ᵢ + λ²e₁'ᵢ − λe₂'ᵢ + e₃'ᵢ = 0
//!   ⇒  λ'ᵢ = Nᵢ / pλ,   Nᵢ = λ²e₁'ᵢ − λe₂'ᵢ + e₃'ᵢ,   pλ = p'(λ) = 3λ²−2e₁λ+e₂.
//! ```
//!
//! and once more for `λ''ᵢⱼ = (∂ⱼNᵢ·pλ − Nᵢ·∂ⱼpλ)/pλ²`. `pλ` at a root equals the
//! product of gaps to the other two roots, so it is well away from zero for
//! distinct eigenvalues; only `α` (largest) and `γ` (smallest) are differentiated
//! this way, and `β = e₁−α−γ` follows from Vieta exactly (mirroring the generic
//! code). Seeds are `[CL, V1, Q2, V2, Q3, V3]`. Validated against
//! [`Dual2<6>`](super::dual2::Dual2) to ~1e-7; the near-degenerate (`Δ≈0`,
//! `pλ≈0`) and invalid cases fall back to the dual path.

use super::dual2::Dual2;
use super::jet::{over_v1, Jet};
use super::three_cpt::{
    three_cpt_infusion_g, three_cpt_infusion_ss_g, three_cpt_iv_bolus_g, three_cpt_iv_bolus_ss_g,
    three_cpt_oral_g, three_cpt_oral_ss_g,
};

/// The three eigenvalue gaps `(α−β, α−γ, β−γ)` as jets — the `1/Δ` factors every
/// 3-cpt coefficient formula divides by. Hoisted out of the per-kernel bodies.
#[inline]
fn eigen_gaps<const N: usize>(
    alpha: Jet<N>,
    beta: Jet<N>,
    gamma: Jet<N>,
) -> (Jet<N>, Jet<N>, Jet<N>) {
    (alpha.sub(beta), alpha.sub(gamma), beta.sub(gamma))
}

/// First/second derivatives of the cubic root `λ` (given its value) by implicit
/// differentiation of `p(λ)=λ³−e₁λ²+e₂λ−e₃=0`, over the `N`-axis layout
/// `[CL,V1,Q2,V2,Q3,V3, …]` (oral uses `N=8` with `KA,F` on axes 6,7, which the
/// eigenvalues don't depend on). Returns `None` if `pλ` is too small
/// (near-degenerate roots), where the closed form is ill-conditioned.
fn root_jet<const N: usize>(lambda: f64, e1: &Jet<N>, e2: &Jet<N>, e3: &Jet<N>) -> Option<Jet<N>> {
    let l = lambda;
    let l2 = l * l;
    let p_lam = 3.0 * l2 - 2.0 * e1.v * l + e2.v; // p'(λ) = gap product
    if p_lam.abs() < 1e-12 {
        return None;
    }
    let inv_p = 1.0 / p_lam;
    let inv_p2 = inv_p * inv_p;

    let mut r = Jet::<N>::cst(l);
    let mut nn = [0.0; N]; // Nᵢ
    let mut lp = [0.0; N]; // λ'ᵢ
    for i in 0..N {
        nn[i] = l2 * e1.g[i] - l * e2.g[i] + e3.g[i];
        lp[i] = nn[i] * inv_p;
        r.g[i] = lp[i];
    }
    for i in 0..N {
        for j in 0..N {
            // ∂ⱼNᵢ = 2λλ'ⱼ e₁'ᵢ + λ²e₁''ᵢⱼ − λ'ⱼ e₂'ᵢ − λ e₂''ᵢⱼ + e₃''ᵢⱼ
            let dn = 2.0 * l * lp[j] * e1.g[i] + l2 * e1.h[i][j] - lp[j] * e2.g[i] - l * e2.h[i][j]
                + e3.h[i][j];
            // ∂ⱼpλ = 6λλ'ⱼ − 2(e₁'ⱼ λ + e₁ λ'ⱼ) + e₂'ⱼ
            let dp = 6.0 * l * lp[j] - 2.0 * (e1.g[j] * l + e1.v * lp[j]) + e2.g[j];
            r.h[i][j] = (dn * p_lam - nn[i] * dp) * inv_p2;
        }
    }
    r.symmetrise();
    Some(r)
}

/// The disposition eigenvalue jets `(α, β, γ, k21, k31)` over the `N`-axis layout
/// `[CL,V1,Q2,V2,Q3,V3, …]` (oral uses `N=8` with `KA,F` on axes 6,7), or `None`
/// when the roots are near-degenerate (the closed-form coefficients carry `1/Δ`
/// factors) and the caller should fall back to the dual path. `α` (largest) and
/// `γ` (smallest) come from implicit differentiation of the characteristic cubic;
/// `β = e₁ − α − γ` by Vieta.
#[allow(clippy::type_complexity)]
fn macro_rate_jets_3cpt<const N: usize>(
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> Option<(Jet<N>, Jet<N>, Jet<N>, Jet<N>, Jet<N>)> {
    // Micro-rates and the symmetric functions (cubic coefficients) as jets
    // (axes CL=0,V1=1,Q2=2,V2=3,Q3=4,V3=5).
    let k10 = Jet::<N>::ratio(cl, 0, v1, 1);
    let k12 = Jet::<N>::ratio(q2, 2, v1, 1);
    let k21 = Jet::<N>::ratio(q2, 2, v2, 3);
    let k13 = Jet::<N>::ratio(q3, 4, v1, 1);
    let k31 = Jet::<N>::ratio(q3, 4, v3, 5);

    let e1 = k10.add(k12).add(k13).add(k21).add(k31);
    let e2 = k10
        .mul(k21)
        .add(k10.mul(k31))
        .add(k21.mul(k31))
        .add(k12.mul(k31))
        .add(k13.mul(k21));
    let e3 = k10.mul(k21).mul(k31);

    // Root values via the trigonometric (Vieta) solution of the depressed cubic
    // — identical to the generic `macro_rates_three_cpt_g` value path.
    let s2 = e1.v;
    let s1 = e2.v;
    let s0 = e3.v;
    let third = 1.0 / 3.0;
    let hh = s2 * third;
    let p = s1 - s2 * s2 * third;
    let q = s1 * s2 * third - s2 * s2 * s2 * (2.0 / 27.0) - s0;
    let p_safe = if p < -1e-30 { p } else { -1e-30 };
    let m = 2.0 * (-(p_safe) * third).sqrt();
    let arg_raw = (3.0 * q) / (p_safe * m);
    let arg = arg_raw.clamp(-1.0, 1.0);
    let phi = arg.acos() * third;
    let pi23 = 2.0 * std::f64::consts::FRAC_PI_3;
    let l0 = m * phi.cos() + hh;
    let l1 = m * (phi - pi23).cos() + hh;
    let l2 = m * (phi - 2.0 * pi23).cos() + hh;
    let av = if l0 >= l1 && l0 >= l2 {
        l0
    } else if l1 >= l2 {
        l1
    } else {
        l2
    };
    let gv = if l0 <= l1 && l0 <= l2 {
        l0
    } else if l1 <= l2 {
        l1
    } else {
        l2
    };
    let bv = s2 - av - gv;
    // Distinct-root guard (coefficients carry 1/Δ factors).
    if (av - bv).abs() < 1e-9 || (av - gv).abs() < 1e-9 || (bv - gv).abs() < 1e-9 {
        return None;
    }

    // α (largest) and γ (smallest) by implicit diff; β = e₁ − α − γ (Vieta).
    let alpha = root_jet(av, &e1, &e2, &e3)?;
    let gamma = root_jet(gv, &e1, &e2, &e3)?;
    let beta = e1.sub(alpha).sub(gamma);
    Some((alpha, beta, gamma, k21, k31))
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3], ∂²f/∂[CL,V1,Q2,V2,Q3,V3]²)` for the 3-cpt IV
/// bolus `C = A·e^{−αt} + B·e^{−βt} + G·e^{−γt}`.
pub fn iv_bolus_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = three_cpt_iv_bolus_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };

    // Coefficients: A = d(α−k21)(α−k31)/[(α−β)(α−γ)], etc., d = amt/V1.
    let d = over_v1(amt, v1);

    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);

    let a = d
        .mul(alpha.sub(k21))
        .mul(alpha.sub(k31))
        .mul(ab.mul(ag).recip());
    // denom_b = −(α−β)(β−γ) = (β−α)(β−γ).
    let b = d
        .mul(beta.sub(k21))
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).recip());
    let g = d
        .mul(gamma.sub(k21))
        .mul(gamma.sub(k31))
        .mul(ag.mul(bg).recip());

    // C = A·e^{−αt} + B·e^{−βt} + G·e^{−γt}.
    let c = a
        .mul(alpha.scale(-t).exp())
        .add(b.mul(beta.scale(-t).exp()))
        .add(g.mul(gamma.scale(-t).exp()));
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3], ∂²f/∂[...]²)` for the 3-cpt infusion (rate
/// `rate`, duration `dur`). Same eigenvalue jets as the bolus; the coefficients
/// carry an extra `1/α`, `1/β`, `1/γ` (zero-order input) and the response is the
/// during/after piecewise of [`three_cpt_infusion_g`].
#[allow(clippy::too_many_arguments)]
pub fn infusion_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = three_cpt_infusion_g::<Dual2<6>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    if dur <= 0.0 {
        return iv_bolus_explicit(amt, t, cl, v1, q2, v2, q3, v3);
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };
    // Coefficients divide by α, β, γ; bail if any is near-zero.
    if alpha.v.abs() < 1e-12 || beta.v.abs() < 1e-12 || gamma.v.abs() < 1e-12 {
        return fallback();
    }

    let rv = over_v1(rate, v1);
    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);

    // a = rv(α−k21)(α−k31)/[(α−β)(α−γ)·α], etc.; denom_b = −(α−β)(β−γ)·β.
    let a_coeff = rv
        .mul(alpha.sub(k21))
        .mul(alpha.sub(k31))
        .mul(ab.mul(ag).mul(alpha).recip());
    let b_coeff = rv
        .mul(beta.sub(k21))
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).mul(beta).recip());
    let g_coeff = rv
        .mul(gamma.sub(k21))
        .mul(gamma.sub(k31))
        .mul(ag.mul(bg).mul(gamma).recip());

    let one = Jet::<6>::cst(1.0);
    let c = if t <= dur {
        let ea = alpha.scale(-t).exp();
        let eb = beta.scale(-t).exp();
        let eg = gamma.scale(-t).exp();
        a_coeff
            .mul(one.sub(ea))
            .add(b_coeff.mul(one.sub(eb)))
            .add(g_coeff.mul(one.sub(eg)))
    } else {
        let ead = alpha.scale(-dur).exp();
        let ebd = beta.scale(-dur).exp();
        let egd = gamma.scale(-dur).exp();
        let eadt = alpha.scale(-(t - dur)).exp();
        let ebdt = beta.scale(-(t - dur)).exp();
        let egdt = gamma.scale(-(t - dur)).exp();
        a_coeff
            .mul(one.sub(ead))
            .mul(eadt)
            .add(b_coeff.mul(one.sub(ebd)).mul(ebdt))
            .add(g_coeff.mul(one.sub(egd)).mul(egdt))
    };
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3,KA,F], ∂²f/∂[...]²)` for 3-cpt oral (first-order
/// absorption). The eigenvalues come from the closed-form implicit-cubic jet
/// (`macro_rate_jets_3cpt::<8>`, `KA,F` on axes 6,7); the per-eigenvalue Bateman
/// assembly of [`three_cpt_oral_g`] is plain jet arithmetic, so the jet carries
/// the `KA`/`F` derivatives automatically. The `ka≈λ` L'Hôpital limits are
/// measure-zero and route to the dual path, which folds them exactly.
#[allow(clippy::too_many_arguments)]
pub fn oral_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 8], [[f64; 8]; 8]) {
    let fallback = || {
        let d = three_cpt_oral_g::<Dual2<8>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
            Dual2::var(ka, 6),
            Dual2::var(f_bio, 7),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
    {
        return (0.0, [0.0; 8], [[0.0; 8]; 8]);
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt::<8>(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };
    // Per-eigenvalue shared-pole L'Hôpital limits → exact dual fallback (rare).
    if (ka - alpha.v).abs() < 1e-6 || (ka - beta.v).abs() < 1e-6 || (ka - gamma.v).abs() < 1e-6 {
        return fallback();
    }

    let ka_j = Jet::<8>::var(ka, 6);
    let f_j = Jet::<8>::var(f_bio, 7);
    // coeff = f_bio·amt·ka/V1.
    let coeff = over_v1::<8>(amt, v1).mul(f_j).mul(ka_j);

    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);
    let a = alpha.sub(k21).mul(alpha.sub(k31)).mul(ab.mul(ag).recip());
    let b = beta
        .sub(k21)
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).recip());
    let g = gamma.sub(k21).mul(gamma.sub(k31)).mul(ag.mul(bg).recip());

    // Bateman per eigenvalue λ (non-singular): (e^{−λt} − e^{−ka·t})/(ka−λ).
    let eka = ka_j.scale(-t).exp();
    let bateman = |lambda: Jet<8>| {
        lambda
            .scale(-t)
            .exp()
            .sub(eka)
            .mul(ka_j.sub(lambda).recip())
    };

    let res = coeff.mul(
        a.mul(bateman(alpha))
            .add(b.mul(bateman(beta)))
            .add(g.mul(bateman(gamma))),
    );
    (res.v, res.g, res.h)
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3], ∂²f/∂[...]²)` for the 3-cpt IV bolus at steady
/// state: the bolus coefficients with each `e^{−λt}` carrying `1/(1−e^{−λ·II})`.
#[allow(clippy::too_many_arguments)]
pub fn iv_bolus_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = three_cpt_iv_bolus_ss_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ii <= 0.0
    {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt::<6>(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };
    let (ss_a, ss_b, ss_g) = match (alpha.ss_coeff(ii), beta.ss_coeff(ii), gamma.ss_coeff(ii)) {
        (Some(a), Some(b), Some(g)) => (a, b, g),
        _ => return fallback(),
    };
    let d = over_v1::<6>(amt, v1);
    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);
    let a = d
        .mul(alpha.sub(k21))
        .mul(alpha.sub(k31))
        .mul(ab.mul(ag).recip());
    let b = d
        .mul(beta.sub(k21))
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).recip());
    let g = d
        .mul(gamma.sub(k21))
        .mul(gamma.sub(k31))
        .mul(ag.mul(bg).recip());
    let c = a
        .mul(alpha.scale(-t).exp())
        .mul(ss_a)
        .add(b.mul(beta.scale(-t).exp()).mul(ss_b))
        .add(g.mul(gamma.scale(-t).exp()).mul(ss_g));
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3,KA,F], ∂²f/∂[...]²)` for 3-cpt oral at steady
/// state: the per-eigenvalue SS Bateman of [`three_cpt_oral_ss_g`]. The `ka≈λ`
/// L'Hôpital limits route to the dual path.
#[allow(clippy::too_many_arguments)]
pub fn oral_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 8], [[f64; 8]; 8]) {
    let fallback = || {
        let d = three_cpt_oral_ss_g::<Dual2<8>>(
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
            Dual2::var(ka, 6),
            Dual2::var(f_bio, 7),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
        || ii <= 0.0
    {
        return (0.0, [0.0; 8], [[0.0; 8]; 8]);
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt::<8>(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };
    if (ka - alpha.v).abs() < 1e-6 || (ka - beta.v).abs() < 1e-6 || (ka - gamma.v).abs() < 1e-6 {
        return fallback();
    }
    let ka_j = Jet::<8>::var(ka, 6);
    let (ss_a, ss_b, ss_g, ss_k) = match (
        alpha.ss_coeff(ii),
        beta.ss_coeff(ii),
        gamma.ss_coeff(ii),
        ka_j.ss_coeff(ii),
    ) {
        (Some(a), Some(b), Some(g), Some(k)) => (a, b, g, k),
        _ => return fallback(),
    };
    let f_j = Jet::<8>::var(f_bio, 7);
    let coeff = over_v1::<8>(amt, v1).mul(f_j).mul(ka_j);
    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);
    let a = alpha.sub(k21).mul(alpha.sub(k31)).mul(ab.mul(ag).recip());
    let b = beta
        .sub(k21)
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).recip());
    let g = gamma.sub(k21).mul(gamma.sub(k31)).mul(ag.mul(bg).recip());

    // SS Bateman per λ (non-singular): (e^{−λt}·ss(λ) − e^{−ka·t}·ss(ka))/(ka−λ).
    let eka_ss = ka_j.scale(-t).exp().mul(ss_k);
    let bateman_ss = |lambda: Jet<8>, ss_l: Jet<8>| {
        lambda
            .scale(-t)
            .exp()
            .mul(ss_l)
            .sub(eka_ss)
            .mul(ka_j.sub(lambda).recip())
    };

    let res = coeff.mul(
        a.mul(bateman_ss(alpha, ss_a))
            .add(b.mul(bateman_ss(beta, ss_b)))
            .add(g.mul(bateman_ss(gamma, ss_g))),
    );
    (res.v, res.g, res.h)
}

/// `(f, ∂f/∂[CL,V1,Q2,V2,Q3,V3], ∂²f/∂[...]²)` for 3-cpt infusion at steady state
/// (non-overlapping `dur ≤ II`): the during/after pieces plus the past-pulse
/// superposition of [`three_cpt_infusion_ss_g`], each carrying `1/(1−e^{−λ·II})`.
#[allow(clippy::too_many_arguments)]
pub fn infusion_ss_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = three_cpt_infusion_ss_g::<Dual2<6>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ii <= 0.0
    {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    if dur <= 0.0 {
        return iv_bolus_ss_explicit(amt, t, ii, cl, v1, q2, v2, q3, v3);
    }
    if dur > ii {
        // Overlapping SS infusion: delegate to the generic dual kernel, which
        // superposes the past pulse train (#379).
        return fallback();
    }
    let (alpha, beta, gamma, k21, k31) = match macro_rate_jets_3cpt::<6>(cl, v1, q2, v2, q3, v3) {
        Some(x) => x,
        None => return fallback(),
    };
    if alpha.v.abs() < 1e-12 || beta.v.abs() < 1e-12 || gamma.v.abs() < 1e-12 {
        return fallback();
    }
    let (ss_a, ss_b, ss_g) = match (alpha.ss_coeff(ii), beta.ss_coeff(ii), gamma.ss_coeff(ii)) {
        (Some(a), Some(b), Some(g)) => (a, b, g),
        _ => return fallback(),
    };
    let rv = over_v1::<6>(rate, v1);
    let (ab, ag, bg) = eigen_gaps(alpha, beta, gamma);
    let a_coeff = rv
        .mul(alpha.sub(k21))
        .mul(alpha.sub(k31))
        .mul(ab.mul(ag).mul(alpha).recip());
    let b_coeff = rv
        .mul(beta.sub(k21))
        .mul(beta.sub(k31))
        .mul(ab.scale(-1.0).mul(bg).mul(beta).recip());
    let g_coeff = rv
        .mul(gamma.sub(k21))
        .mul(gamma.sub(k31))
        .mul(ag.mul(bg).mul(gamma).recip());
    let one = Jet::<6>::cst(1.0);

    // Past pulses (n ≥ 1): always "after-infusion".
    let past = |coeff: Jet<6>, lambda: Jet<6>, ss_l: Jet<6>| {
        coeff
            .mul(one.sub(lambda.scale(-dur).exp()))
            .mul(lambda.scale(-(t - dur)).exp())
            .mul(lambda.scale(-ii).exp())
            .mul(ss_l)
    };
    let c = if t <= dur {
        a_coeff
            .mul(one.sub(alpha.scale(-t).exp()))
            .add(b_coeff.mul(one.sub(beta.scale(-t).exp())))
            .add(g_coeff.mul(one.sub(gamma.scale(-t).exp())))
            .add(past(a_coeff, alpha, ss_a))
            .add(past(b_coeff, beta, ss_b))
            .add(past(g_coeff, gamma, ss_g))
    } else {
        let after = |coeff: Jet<6>, lambda: Jet<6>, ss_l: Jet<6>| {
            coeff
                .mul(one.sub(lambda.scale(-dur).exp()))
                .mul(lambda.scale(-(t - dur)).exp())
                .mul(ss_l)
        };
        after(a_coeff, alpha, ss_a)
            .add(after(b_coeff, beta, ss_b))
            .add(after(g_coeff, gamma, ss_g))
    };
    (c.v, c.g, c.h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dual_bolus(
        amt: f64,
        t: f64,
        cl: f64,
        v1: f64,
        q2: f64,
        v2: f64,
        q3: f64,
        v3: f64,
    ) -> (f64, [f64; 6], [[f64; 6]; 6]) {
        let d = three_cpt_iv_bolus_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn three_cpt_iv_bolus_explicit_matches_dual() {
        for &(amt, t, cl, v1, q2, v2, q3, v3) in &[
            (1000.0, 0.25, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0),
            (1000.0, 2.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0),
            (1000.0, 24.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0),
            (500.0, 4.0, 8.0, 15.0, 3.0, 40.0, 0.8, 60.0),
            (1000.0, 1.0, 3.2, 12.4, 1.1, 25.0, 0.6, 50.0), // 3-cpt fit-ish
        ] {
            let (fe, ge, he) = iv_bolus_explicit(amt, t, cl, v1, q2, v2, q3, v3);
            let (fd, gd, hd) = dual_bolus(amt, t, cl, v1, q2, v2, q3, v3);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-7, epsilon = 1e-11);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-6,
                        epsilon = 1e-10
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dual_infusion(
        rate: f64,
        dur: f64,
        amt: f64,
        t: f64,
        cl: f64,
        v1: f64,
        q2: f64,
        v2: f64,
        q3: f64,
        v3: f64,
    ) -> (f64, [f64; 6], [[f64; 6]; 6]) {
        let d = three_cpt_infusion_g::<Dual2<6>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn three_cpt_infusion_explicit_matches_dual() {
        // dur = amt/rate; cover both during (t ≤ dur) and after (t > dur).
        for &(rate, amt, t, cl, v1, q2, v2, q3, v3) in &[
            (500.0, 1000.0, 1.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0), // during (dur=2)
            (500.0, 1000.0, 6.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0), // after
            (250.0, 1000.0, 2.0, 8.0, 15.0, 3.0, 40.0, 0.8, 60.0), // during (dur=4)
            (250.0, 1000.0, 10.0, 8.0, 15.0, 3.0, 40.0, 0.8, 60.0), // after
            (1000.0, 1000.0, 0.5, 3.2, 12.4, 1.1, 25.0, 0.6, 50.0), // during (dur=1), fit-ish
            (1000.0, 1000.0, 4.0, 3.2, 12.4, 1.1, 25.0, 0.6, 50.0), // after
        ] {
            let dur = amt / rate;
            let (fe, ge, he) = infusion_explicit(rate, dur, amt, t, cl, v1, q2, v2, q3, v3);
            let (fd, gd, hd) = dual_infusion(rate, dur, amt, t, cl, v1, q2, v2, q3, v3);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-9, epsilon = 1e-12);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-6, epsilon = 1e-10);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-5,
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dual_oral(
        amt: f64,
        t: f64,
        cl: f64,
        v1: f64,
        q2: f64,
        v2: f64,
        q3: f64,
        v3: f64,
        ka: f64,
        f_bio: f64,
    ) -> (f64, [f64; 8], [[f64; 8]; 8]) {
        let d = three_cpt_oral_g::<Dual2<8>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q2, 2),
            Dual2::var(v2, 3),
            Dual2::var(q3, 4),
            Dual2::var(v3, 5),
            Dual2::var(ka, 6),
            Dual2::var(f_bio, 7),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn three_cpt_oral_explicit_matches_dual() {
        // Spread of params avoiding the ka≈α/β/γ limits, plus F≠1.
        for &(amt, t, cl, v1, q2, v2, q3, v3, ka, fb) in &[
            (1000.0, 1.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.2, 0.9),
            (1000.0, 4.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 0.7, 1.0),
            (500.0, 0.5, 8.0, 15.0, 3.0, 40.0, 0.8, 60.0, 2.0, 0.75),
            (1000.0, 8.0, 3.2, 12.4, 1.1, 25.0, 0.6, 50.0, 0.5, 1.0), // fit-ish
        ] {
            let (fe, ge, he) = oral_explicit(amt, t, cl, v1, q2, v2, q3, v3, ka, fb);
            let (fd, gd, hd) = dual_oral(amt, t, cl, v1, q2, v2, q3, v3, ka, fb);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-8, epsilon = 1e-11);
            for i in 0..8 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-6, epsilon = 1e-9);
                for j in 0..8 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-5,
                        epsilon = 1e-8
                    );
                }
            }
        }
    }

    #[test]
    fn three_cpt_ss_explicit_matches_dual() {
        // bolus SS
        for &(amt, t, ii, cl, v1, q2, v2, q3, v3) in &[
            (1000.0, 2.0, 24.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0),
            (1000.0, 18.0, 24.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0),
        ] {
            let (fe, ge, he) = iv_bolus_ss_explicit(amt, t, ii, cl, v1, q2, v2, q3, v3);
            let d = three_cpt_iv_bolus_ss_g::<Dual2<6>>(
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-9, epsilon = 1e-12);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-6, epsilon = 1e-10);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-5,
                        epsilon = 1e-9
                    );
                }
            }
        }
        // oral SS
        for &(amt, t, ii, cl, v1, q2, v2, q3, v3, ka, fb) in &[
            (1000.0, 2.0, 24.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 1.2, 0.9),
            (
                1000.0, 18.0, 24.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0, 0.7, 1.0,
            ),
        ] {
            let (fe, ge, he) = oral_ss_explicit(amt, t, ii, cl, v1, q2, v2, q3, v3, ka, fb);
            let d = three_cpt_oral_ss_g::<Dual2<8>>(
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
                Dual2::var(ka, 6),
                Dual2::var(fb, 7),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-8, epsilon = 1e-11);
            for i in 0..8 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-6, epsilon = 1e-9);
                for j in 0..8 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-5,
                        epsilon = 1e-8
                    );
                }
            }
        }
        // infusion SS (dur ≤ ii): during + after
        for &(rate, dur, amt, t, ii, cl, v1, q2, v2, q3, v3) in &[
            (
                500.0, 2.0, 1000.0, 1.0, 12.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0,
            ),
            (
                500.0, 2.0, 1000.0, 6.0, 12.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0,
            ),
        ] {
            let (fe, ge, he) = infusion_ss_explicit(rate, dur, amt, t, ii, cl, v1, q2, v2, q3, v3);
            let d = three_cpt_infusion_ss_g::<Dual2<6>>(
                rate,
                dur,
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-8, epsilon = 1e-11);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-6, epsilon = 1e-9);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-5,
                        epsilon = 1e-8
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn three_cpt_explicit_vs_dual_bench() {
        use std::time::Instant;
        let n = 10_000_000u64;
        let (amt, cl, v1, q2, v2, q3, v3) = (1000.0, 5.0, 10.0, 2.0, 20.0, 1.5, 30.0);
        let run = |label: &str, f: &dyn Fn(f64) -> f64| {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for i in 0..n {
                acc += f((i % 24) as f64 * 0.5 + 0.25);
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            std::hint::black_box(acc);
            eprintln!("  {label:<36} {ns:6.2} ns/eval");
            ns
        };
        eprintln!("3-cpt IV bolus f+grad+hess:");
        let exp = run("Option B (explicit, implicit-cubic λ)", &|t| {
            let (f, g, h) = iv_bolus_explicit(amt, t, cl, v1, q2, v2, q3, v3);
            f + g.iter().sum::<f64>() + h.iter().flatten().sum::<f64>()
        });
        let d6 = run("Dual2<6> (minimal width)", &|t| {
            let d = three_cpt_iv_bolus_g::<Dual2<6>>(
                amt,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        let d8 = run("Dual2<8> (provider width)", &|t| {
            let d = three_cpt_iv_bolus_g::<Dual2<8>>(
                amt,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q2, 2),
                Dual2::var(v2, 3),
                Dual2::var(q3, 4),
                Dual2::var(v3, 5),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        eprintln!(
            "  → explicit is {:.1}x faster than Dual2<6>, {:.1}x faster than Dual2<8>",
            d6 / exp,
            d8 / exp
        );
    }
}
