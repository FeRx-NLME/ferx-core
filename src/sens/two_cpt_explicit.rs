//! Option B (explicit symbolic derivatives) for the 2-cpt IV-bolus solution.
//!
//! Unlike 1-cpt, the 2-cpt response routes through the macro-rate eigenvalues
//! `α, β` (roots of `λ² − sλ + d = 0`, `s = k10+k12+k21`, `d = k10·k21`). The
//! expensive part for a `Dual2<N>` would be differentiating the `√(s²−4d)` and
//! the `β = d/α` division through the dual rules. Here we instead get the
//! eigenvalue first/second derivatives in **closed form** by implicit
//! differentiation of Vieta's relations (`α+β = s`, `αβ = d`):
//!
//! ```text
//!   α'ᵢ = (α·s'ᵢ − d'ᵢ)/Δ,            β'ᵢ = (d'ᵢ − β·s'ᵢ)/Δ,      Δ = α−β
//!   α''ᵢⱼ = [(α'ⱼ s'ᵢ + α s''ᵢⱼ − d''ᵢⱼ)Δ − (α s'ᵢ − d'ᵢ)(α'ⱼ−β'ⱼ)] / Δ²
//!   β''ᵢⱼ = [(d''ᵢⱼ − β'ⱼ s'ᵢ − β s''ᵢⱼ)Δ − (d'ᵢ − β s'ᵢ)(α'ⱼ−β'ⱼ)] / Δ²
//! ```
//!
//! and propagate the coefficient/exponential assembly with a small second-order
//! [`Jet`](super::jet::Jet). Seeds are `[CL, V1, Q, V2]` (plus `KA, F` on axes
//! 4,5 for oral). Validated against [`Dual2`](super::dual2::Dual2) to ~1e-8; the
//! near-degenerate (`Δ≈0`) and `ka≈α/β` L'Hôpital cases fall back to the dual
//! path.

use super::dual2::Dual2;
use super::jet::{over_v1, Jet};
use super::two_cpt::{
    two_cpt_infusion_g, two_cpt_infusion_ss_g, two_cpt_iv_bolus_g, two_cpt_iv_bolus_ss_g,
    two_cpt_oral_g, two_cpt_oral_ss_g,
};

/// The macro-rate eigenvalue jets `(α, β, k21)` over the `N`-axis layout
/// `[CL,V1,Q,V2, …]` (oral uses `N=6` with `KA,F` on axes 4,5, which the
/// eigenvalues don't depend on), or `None` when the disposition is degenerate
/// (`disc≈0`, `α≈0`, or `α≈β`) and the caller should fall back to the dual path.
/// Obtained by implicit differentiation of Vieta's relations (closed form, no
/// `√`-jet); see the module header.
fn macro_rate_jets<const N: usize>(
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> Option<(Jet<N>, Jet<N>, Jet<N>)> {
    // Micro-rates as jets (closed-form sparse grad/hess on axes CL=0,V1=1,Q=2,V2=3).
    let k10 = Jet::<N>::ratio(cl, 0, v1, 1);
    let k12 = Jet::<N>::ratio(q, 2, v1, 1);
    let k21 = Jet::<N>::ratio(q, 2, v2, 3);

    // s = k10 + k12 + k21 ; d = k10·k21.
    let s = k10.add(k12).add(k21);
    let d = k10.mul(k21);

    // Eigenvalues via Vieta + implicit differentiation (closed form, no √-jet).
    let disc_sq = s.v * s.v - 4.0 * d.v;
    if disc_sq <= 1e-300 {
        return None;
    }
    let disc = disc_sq.sqrt();
    let av = 0.5 * (s.v + disc);
    if av <= 1e-300 {
        return None;
    }
    let bv = d.v / av;
    let delta = av - bv;
    if delta.abs() < 1e-12 {
        return None;
    }
    let inv_d = 1.0 / delta;
    let inv_d2 = inv_d * inv_d;

    let mut alpha = Jet::<N>::cst(av);
    let mut beta = Jet::<N>::cst(bv);
    // First derivatives.
    for i in 0..N {
        alpha.g[i] = (av * s.g[i] - d.g[i]) * inv_d;
        beta.g[i] = (d.g[i] - bv * s.g[i]) * inv_d;
    }
    // Second derivatives (see module header).
    for i in 0..N {
        for j in 0..N {
            let dg = alpha.g[j] - beta.g[j];
            let a_ij = ((alpha.g[j] * s.g[i] + av * s.h[i][j] - d.h[i][j]) * delta
                - (av * s.g[i] - d.g[i]) * dg)
                * inv_d2;
            let b_ij = ((d.h[i][j] - beta.g[j] * s.g[i] - bv * s.h[i][j]) * delta
                - (d.g[i] - bv * s.g[i]) * dg)
                * inv_d2;
            alpha.h[i][j] = a_ij;
            beta.h[i][j] = b_ij;
        }
    }
    alpha.symmetrise();
    beta.symmetrise();
    Some((alpha, beta, k21))
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[CL,V1,Q,V2]²)` for the 2-cpt IV bolus.
pub fn iv_bolus_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_iv_bolus_g::<Dual2<4>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    let (alpha, beta, k21) = match macro_rate_jets::<4>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };

    // Coefficients: a = (amt/V1)(α−k21)/Δ, b = (amt/V1)(k21−β)/Δ, Δ = α−β.
    let amt_v1 = over_v1::<4>(amt, v1);
    let diff = alpha.sub(beta);
    let inv_diff = diff.recip();
    let a = amt_v1.mul(alpha.sub(k21)).mul(inv_diff);
    let b = amt_v1.mul(k21.sub(beta)).mul(inv_diff);

    // C = a·e^{−αt} + b·e^{−βt}.
    let e_a = alpha.scale(-t).exp();
    let e_b = beta.scale(-t).exp();
    let c = a.mul(e_a).add(b.mul(e_b));
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[CL,V1,Q,V2]²)` for the 2-cpt infusion (rate
/// `rate`, duration `dur`). Same eigenvalue jets as the bolus; the coefficients
/// carry an extra `1/α`, `1/β` (zero-order input), and the response is the
/// during/after piecewise of [`two_cpt_infusion_g`].
pub fn infusion_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_infusion_g::<Dual2<4>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    if dur <= 0.0 {
        return iv_bolus_explicit(amt, t, cl, v1, q, v2);
    }
    let (alpha, beta, k21) = match macro_rate_jets::<4>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    // The coefficients divide by α and β; bail to the dual path if either is
    // near-zero (the generic form returns 0 there, but FD-matching that
    // degenerate zero buys nothing).
    if alpha.v.abs() < 1e-12 || beta.v.abs() < 1e-12 {
        return fallback();
    }

    // a = (R/V1)(α−k21)/(Δ·α), b = (R/V1)(k21−β)/(Δ·β), Δ = α−β.
    let r_v1 = over_v1::<4>(rate, v1);
    let diff = alpha.sub(beta);
    let inv_diff = diff.recip();
    let a_coeff = r_v1.mul(alpha.sub(k21)).mul(inv_diff).mul(alpha.recip());
    let b_coeff = r_v1.mul(k21.sub(beta)).mul(inv_diff).mul(beta.recip());

    let one = Jet::<4>::cst(1.0);
    let c = if t <= dur {
        let e_a = alpha.scale(-t).exp();
        let e_b = beta.scale(-t).exp();
        a_coeff.mul(one.sub(e_a)).add(b_coeff.mul(one.sub(e_b)))
    } else {
        let e_ad = alpha.scale(-dur).exp();
        let e_bd = beta.scale(-dur).exp();
        let e_adt = alpha.scale(-(t - dur)).exp();
        let e_bdt = beta.scale(-(t - dur)).exp();
        a_coeff
            .mul(one.sub(e_ad))
            .mul(e_adt)
            .add(b_coeff.mul(one.sub(e_bd)).mul(e_bdt))
    };
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2,KA,F], ∂²f/∂[...]²)` for 2-cpt oral (first-order
/// absorption). The disposition eigenvalues come from the closed-form Vieta jet
/// (`macro_rate_jets::<6>`, `KA,F` on axes 4,5); the Bateman assembly is plain
/// jet arithmetic (`p + q + r` of [`two_cpt_oral_g`]), so the jet carries the
/// `KA`/`F` derivatives automatically. The `ka≈α`/`ka≈β` L'Hôpital limits (where
/// two terms share the `1/(ka−λ)` pole) are measure-zero and route to the dual
/// path, which folds them exactly — mirroring `one_cpt_explicit::oral_explicit`.
#[allow(clippy::too_many_arguments)]
pub fn oral_explicit(
    amt: f64,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = two_cpt_oral_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
            Dual2::var(ka, 4),
            Dual2::var(f_bio, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 || ka <= 0.0 {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    let (alpha, beta, k21) = match macro_rate_jets::<6>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    // Shared-pole L'Hôpital limits → exact dual fallback (rare).
    if (ka - alpha.v).abs() < 1e-6 || (ka - beta.v).abs() < 1e-6 {
        return fallback();
    }

    let ka_j = Jet::<6>::var(ka, 4);
    let f_j = Jet::<6>::var(f_bio, 5);
    // d = f_bio·amt·ka/V1.
    let d = over_v1::<6>(amt, v1).mul(f_j).mul(ka_j);

    // p = d(k21−α)/[(ka−α)(β−α)]·e^{−αt}
    let p = d
        .mul(k21.sub(alpha))
        .mul(ka_j.sub(alpha).mul(beta.sub(alpha)).recip())
        .mul(alpha.scale(-t).exp());
    // q = d(k21−β)/[(ka−β)(α−β)]·e^{−βt}
    let q_term = d
        .mul(k21.sub(beta))
        .mul(ka_j.sub(beta).mul(alpha.sub(beta)).recip())
        .mul(beta.scale(-t).exp());
    // r = d(k21−ka)/[(α−ka)(β−ka)]·e^{−ka·t}
    let r = d
        .mul(k21.sub(ka_j))
        .mul(alpha.sub(ka_j).mul(beta.sub(ka_j)).recip())
        .mul(ka_j.scale(-t).exp());

    let c = p.add(q_term).add(r);
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[...]²)` for the 2-cpt IV bolus at steady state:
/// the bolus coefficients with each `e^{−λt}` term carrying the SS factor
/// `1/(1−e^{−λ·II})`.
pub fn iv_bolus_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_iv_bolus_ss_g::<Dual2<4>>(
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    let (alpha, beta, k21) = match macro_rate_jets::<4>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    let (ss_a, ss_b) = match (alpha.ss_coeff(ii), beta.ss_coeff(ii)) {
        (Some(a), Some(b)) => (a, b),
        _ => return fallback(),
    };
    let amt_v1 = over_v1::<4>(amt, v1);
    let inv_diff = alpha.sub(beta).recip();
    let a = amt_v1.mul(alpha.sub(k21)).mul(inv_diff);
    let b = amt_v1.mul(k21.sub(beta)).mul(inv_diff);
    let c = a
        .mul(alpha.scale(-t).exp())
        .mul(ss_a)
        .add(b.mul(beta.scale(-t).exp()).mul(ss_b));
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2,KA,F], ∂²f/∂[...]²)` for 2-cpt oral at steady state: the
/// non-SS oral `p + q + r` with each `e^{−λt}` carrying `1/(1−e^{−λ·II})`. The
/// `ka≈α/β` L'Hôpital limits route to the dual path.
#[allow(clippy::too_many_arguments)]
pub fn oral_ss_explicit(
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> (f64, [f64; 6], [[f64; 6]; 6]) {
    let fallback = || {
        let d = two_cpt_oral_ss_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
            Dual2::var(ka, 4),
            Dual2::var(f_bio, 5),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 || ka <= 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 6], [[0.0; 6]; 6]);
    }
    let (alpha, beta, k21) = match macro_rate_jets::<6>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    if (ka - alpha.v).abs() < 1e-6 || (ka - beta.v).abs() < 1e-6 {
        return fallback();
    }
    let ka_j = Jet::<6>::var(ka, 4);
    let (ss_a, ss_b, ss_k) = match (alpha.ss_coeff(ii), beta.ss_coeff(ii), ka_j.ss_coeff(ii)) {
        (Some(a), Some(b), Some(k)) => (a, b, k),
        _ => return fallback(),
    };
    let f_j = Jet::<6>::var(f_bio, 5);
    let d = over_v1::<6>(amt, v1).mul(f_j).mul(ka_j);

    let p = d
        .mul(k21.sub(alpha))
        .mul(ka_j.sub(alpha).mul(beta.sub(alpha)).recip())
        .mul(alpha.scale(-t).exp())
        .mul(ss_a);
    let q_term = d
        .mul(k21.sub(beta))
        .mul(ka_j.sub(beta).mul(alpha.sub(beta)).recip())
        .mul(beta.scale(-t).exp())
        .mul(ss_b);
    let r = d
        .mul(k21.sub(ka_j))
        .mul(alpha.sub(ka_j).mul(beta.sub(ka_j)).recip())
        .mul(ka_j.scale(-t).exp())
        .mul(ss_k);
    let c = p.add(q_term).add(r);
    (c.v, c.g, c.h)
}

/// `(f, ∂f/∂[CL,V1,Q,V2], ∂²f/∂[...]²)` for 2-cpt infusion at steady state: the
/// during/after pieces plus the past-pulse superposition, each carrying
/// `1/(1−e^{−λ·II})`. Overlapping pulses (`dur > II`) delegate to the generic
/// dual kernel [`two_cpt_infusion_ss_g`] (#379).
#[allow(clippy::too_many_arguments)]
pub fn infusion_ss_explicit(
    rate: f64,
    dur: f64,
    amt: f64,
    t: f64,
    ii: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
) -> (f64, [f64; 4], [[f64; 4]; 4]) {
    let fallback = || {
        let d = two_cpt_infusion_ss_g::<Dual2<4>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            ii,
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    };
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || cl <= 0.0 || q < 0.0 || ii <= 0.0 {
        return (0.0, [0.0; 4], [[0.0; 4]; 4]);
    }
    if dur <= 0.0 {
        return iv_bolus_ss_explicit(amt, t, ii, cl, v1, q, v2);
    }
    if dur > ii {
        // Overlapping SS infusion: delegate to the generic dual kernel, which
        // superposes the past pulse train (#379).
        return fallback();
    }
    let (alpha, beta, k21) = match macro_rate_jets::<4>(cl, v1, q, v2) {
        Some(x) => x,
        None => return fallback(),
    };
    if alpha.v.abs() < 1e-12 || beta.v.abs() < 1e-12 {
        return fallback();
    }
    let (ss_a, ss_b) = match (alpha.ss_coeff(ii), beta.ss_coeff(ii)) {
        (Some(a), Some(b)) => (a, b),
        _ => return fallback(),
    };
    let r_v1 = over_v1::<4>(rate, v1);
    let inv_diff = alpha.sub(beta).recip();
    let a_coeff = r_v1.mul(alpha.sub(k21)).mul(inv_diff).mul(alpha.recip());
    let b_coeff = r_v1.mul(k21.sub(beta)).mul(inv_diff).mul(beta.recip());
    let one = Jet::<4>::cst(1.0);

    // Past pulses (n ≥ 1): always "after-infusion".
    let past_a = a_coeff
        .mul(one.sub(alpha.scale(-dur).exp()))
        .mul(alpha.scale(-(t - dur)).exp())
        .mul(alpha.scale(-ii).exp())
        .mul(ss_a);
    let past_b = b_coeff
        .mul(one.sub(beta.scale(-dur).exp()))
        .mul(beta.scale(-(t - dur)).exp())
        .mul(beta.scale(-ii).exp())
        .mul(ss_b);
    let c = if t <= dur {
        a_coeff
            .mul(one.sub(alpha.scale(-t).exp()))
            .add(b_coeff.mul(one.sub(beta.scale(-t).exp())))
            .add(past_a)
            .add(past_b)
    } else {
        a_coeff
            .mul(one.sub(alpha.scale(-dur).exp()))
            .mul(alpha.scale(-(t - dur)).exp())
            .mul(ss_a)
            .add(
                b_coeff
                    .mul(one.sub(beta.scale(-dur).exp()))
                    .mul(beta.scale(-(t - dur)).exp())
                    .mul(ss_b),
            )
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
        q: f64,
        v2: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = two_cpt_iv_bolus_g::<Dual2<4>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn two_cpt_iv_bolus_explicit_matches_dual() {
        for &(amt, t, cl, v1, q, v2) in &[
            (1000.0, 0.25, 10.0, 50.0, 15.0, 100.0),
            (1000.0, 2.0, 10.0, 50.0, 15.0, 100.0),
            (1000.0, 24.0, 10.0, 50.0, 15.0, 100.0),
            (500.0, 4.0, 5.0, 30.0, 2.0, 50.0),
            (1000.0, 1.0, 4.41, 15.5, 3.14, 29.3), // 2-cpt NONMEM-fit-ish
        ] {
            let (fe, ge, he) = iv_bolus_explicit(amt, t, cl, v1, q, v2);
            let (fd, gd, hd) = dual_bolus(amt, t, cl, v1, q, v2);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-8, epsilon = 1e-12);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-11
                    );
                }
            }
        }
    }

    fn dual_infusion(
        rate: f64,
        dur: f64,
        amt: f64,
        t: f64,
        cl: f64,
        v1: f64,
        q: f64,
        v2: f64,
    ) -> (f64, [f64; 4], [[f64; 4]; 4]) {
        let d = two_cpt_infusion_g::<Dual2<4>>(
            rate,
            dur,
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn two_cpt_infusion_explicit_matches_dual() {
        // dur = amt/rate; cover both during (t ≤ dur) and after (t > dur).
        for &(rate, amt, t, cl, v1, q, v2) in &[
            (500.0, 1000.0, 1.0, 10.0, 50.0, 15.0, 100.0), // during (dur=2)
            (500.0, 1000.0, 6.0, 10.0, 50.0, 15.0, 100.0), // after
            (250.0, 1000.0, 2.0, 5.0, 30.0, 2.0, 50.0),    // during (dur=4)
            (250.0, 1000.0, 10.0, 5.0, 30.0, 2.0, 50.0),   // after
            (1000.0, 1000.0, 0.5, 4.41, 15.5, 3.14, 29.3), // during (dur=1), fit-ish
            (1000.0, 1000.0, 3.0, 4.41, 15.5, 3.14, 29.3), // after
        ] {
            let dur = amt / rate;
            let (fe, ge, he) = infusion_explicit(rate, dur, amt, t, cl, v1, q, v2);
            let (fd, gd, hd) = dual_infusion(rate, dur, amt, t, cl, v1, q, v2);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-8, epsilon = 1e-11);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-10
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
        q: f64,
        v2: f64,
        ka: f64,
        f_bio: f64,
    ) -> (f64, [f64; 6], [[f64; 6]; 6]) {
        let d = two_cpt_oral_g::<Dual2<6>>(
            amt,
            Dual2::constant(t),
            Dual2::var(cl, 0),
            Dual2::var(v1, 1),
            Dual2::var(q, 2),
            Dual2::var(v2, 3),
            Dual2::var(ka, 4),
            Dual2::var(f_bio, 5),
        );
        (d.value, d.grad, d.hess)
    }

    #[test]
    fn two_cpt_oral_explicit_matches_dual() {
        // Spread of (CL,V1,Q,V2,KA,F) avoiding the ka≈α/β limits, plus F≠1.
        for &(amt, t, cl, v1, q, v2, ka, fb) in &[
            (100.0, 1.0, 10.0, 50.0, 15.0, 100.0, 1.2, 0.9),
            (100.0, 4.0, 10.0, 50.0, 15.0, 100.0, 0.8, 1.0),
            (500.0, 0.5, 5.0, 30.0, 2.0, 50.0, 2.0, 0.75),
            (1000.0, 8.0, 4.41, 15.5, 3.14, 29.3, 0.6, 1.0), // fit-ish
        ] {
            let (fe, ge, he) = oral_explicit(amt, t, cl, v1, q, v2, ka, fb);
            let (fd, gd, hd) = dual_oral(amt, t, cl, v1, q, v2, ka, fb);
            approx::assert_relative_eq!(fe, fd, max_relative = 1e-9, epsilon = 1e-12);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], gd[i], max_relative = 1e-7, epsilon = 1e-10);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        hd[i][j],
                        max_relative = 1e-6,
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    #[test]
    fn two_cpt_ss_explicit_matches_dual() {
        // bolus SS
        for &(amt, t, ii, cl, v1, q, v2) in &[
            (1000.0, 2.0, 24.0, 10.0, 50.0, 15.0, 100.0),
            (1000.0, 18.0, 24.0, 10.0, 50.0, 15.0, 100.0),
            (500.0, 4.0, 12.0, 5.0, 30.0, 2.0, 50.0),
        ] {
            let (fe, ge, he) = iv_bolus_ss_explicit(amt, t, ii, cl, v1, q, v2);
            let d = two_cpt_iv_bolus_ss_g::<Dual2<4>>(
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-10, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-8, epsilon = 1e-11);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-7,
                        epsilon = 1e-10
                    );
                }
            }
        }
        // oral SS
        for &(amt, t, ii, cl, v1, q, v2, ka, fb) in &[
            (100.0, 2.0, 24.0, 10.0, 50.0, 15.0, 100.0, 1.2, 0.9),
            (100.0, 18.0, 24.0, 10.0, 50.0, 15.0, 100.0, 0.8, 1.0),
            (500.0, 4.0, 12.0, 5.0, 30.0, 2.0, 50.0, 2.0, 0.75),
        ] {
            let (fe, ge, he) = oral_ss_explicit(amt, t, ii, cl, v1, q, v2, ka, fb);
            let d = two_cpt_oral_ss_g::<Dual2<6>>(
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
                Dual2::var(ka, 4),
                Dual2::var(fb, 5),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-9, epsilon = 1e-12);
            for i in 0..6 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-7, epsilon = 1e-10);
                for j in 0..6 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-6,
                        epsilon = 1e-9
                    );
                }
            }
        }
        // infusion SS (dur ≤ ii): during + after
        for &(rate, dur, amt, t, ii, cl, v1, q, v2) in &[
            (500.0, 2.0, 1000.0, 1.0, 12.0, 10.0, 50.0, 15.0, 100.0),
            (500.0, 2.0, 1000.0, 6.0, 12.0, 10.0, 50.0, 15.0, 100.0),
            (250.0, 4.0, 1000.0, 10.0, 24.0, 5.0, 30.0, 2.0, 50.0),
        ] {
            let (fe, ge, he) = infusion_ss_explicit(rate, dur, amt, t, ii, cl, v1, q, v2);
            let d = two_cpt_infusion_ss_g::<Dual2<4>>(
                rate,
                dur,
                amt,
                Dual2::constant(t),
                ii,
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
            );
            approx::assert_relative_eq!(fe, d.value, max_relative = 1e-9, epsilon = 1e-12);
            for i in 0..4 {
                approx::assert_relative_eq!(ge[i], d.grad[i], max_relative = 1e-7, epsilon = 1e-10);
                for j in 0..4 {
                    approx::assert_relative_eq!(
                        he[i][j],
                        d.hess[i][j],
                        max_relative = 1e-6,
                        epsilon = 1e-9
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "bench: run with -- --ignored --nocapture"]
    fn two_cpt_explicit_vs_dual_bench() {
        use std::time::Instant;
        let n = 20_000_000u64;
        let (amt, cl, v1, q, v2) = (1000.0, 10.0, 50.0, 15.0, 100.0);
        let run = |label: &str, f: &dyn Fn(f64) -> f64| {
            let t0 = Instant::now();
            let mut acc = 0.0;
            for i in 0..n {
                acc += f((i % 24) as f64 * 0.5 + 0.25);
            }
            let ns = t0.elapsed().as_nanos() as f64 / n as f64;
            std::hint::black_box(acc);
            eprintln!("  {label:<34} {ns:6.2} ns/eval");
            ns
        };
        eprintln!("2-cpt IV bolus f+grad+hess:");
        let exp = run("Option B (explicit, closed-form λ)", &|t| {
            let (f, g, h) = iv_bolus_explicit(amt, t, cl, v1, q, v2);
            f + g.iter().sum::<f64>() + h.iter().flatten().sum::<f64>()
        });
        let d4 = run("Dual2<4> (minimal width)", &|t| {
            let d = two_cpt_iv_bolus_g::<Dual2<4>>(
                amt,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        let d8 = run("Dual2<8> (provider width)", &|t| {
            let d = two_cpt_iv_bolus_g::<Dual2<8>>(
                amt,
                Dual2::constant(t),
                Dual2::var(cl, 0),
                Dual2::var(v1, 1),
                Dual2::var(q, 2),
                Dual2::var(v2, 3),
            );
            d.value + d.grad.iter().sum::<f64>() + d.hess.iter().flatten().sum::<f64>()
        });
        eprintln!(
            "  → explicit is {:.1}x faster than Dual2<4>, {:.1}x faster than Dual2<8>",
            d4 / exp,
            d8 / exp
        );
    }
}
