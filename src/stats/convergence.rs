//! Generic MCMC convergence / summary diagnostics: split-R̂, effective sample
//! size (Geyer initial-positive truncation), and a type-7 quantile. Kept here
//! (not in an estimator module) so any sampler — the Bayes estimator today, a
//! future HMC/NUTS variant or SIR posterior summaries tomorrow — shares one
//! implementation and one definition of each statistic.

/// Mean of a slice. NaN for an empty slice.
pub(crate) fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample variance with denominator `n − 1`. `0.0` for fewer than 2 elements.
pub(crate) fn sample_var(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n as f64 - 1.0)
}

/// Biased (divide-by-N) lag-`t` autocovariance of `xs` about supplied mean `mu`.
fn autocov(xs: &[f64], t: usize, mu: f64) -> f64 {
    let n = xs.len();
    if t >= n {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..(n - t) {
        s += (xs[i] - mu) * (xs[i + t] - mu);
    }
    s / n as f64
}

/// Type-7 (linear-interpolation) quantile of an already-sorted slice.
/// `q ∈ [0, 1]`. Empty input returns NaN.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * q;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Split-R̂ (Gelman et al. / Vehtari et al. 2021) across `chains` of equal-ish
/// length. Each chain is split in half, giving `2·M` sub-chains of length `n`;
/// R̂ = √(v̂ar⁺ / W). Values near 1.0 indicate mixing; `> 1.01` flags
/// non-convergence. Returns NaN if there are fewer than 2 usable sub-chains or
/// `n < 2`.
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    // Split each chain in half (drop the middle element when odd).
    let mut subs: Vec<&[f64]> = Vec::with_capacity(chains.len() * 2);
    for c in chains {
        let half = c.len() / 2;
        if half < 2 {
            continue;
        }
        subs.push(&c[..half]);
        subs.push(&c[c.len() - half..]);
    }
    let m = subs.len();
    if m < 2 {
        return f64::NAN;
    }
    let n = subs[0].len();
    let means: Vec<f64> = subs.iter().map(|s| mean(s)).collect();
    let vars: Vec<f64> = subs.iter().map(|s| sample_var(s)).collect();
    let grand = mean(&means);

    // Between-chain variance B (per draw) and within-chain variance W.
    let b = n as f64 / (m as f64 - 1.0) * means.iter().map(|mj| (mj - grand).powi(2)).sum::<f64>();
    let w = vars.iter().sum::<f64>() / m as f64;
    if w <= 0.0 {
        return f64::NAN;
    }
    let var_plus = (n as f64 - 1.0) / n as f64 * w + b / n as f64;
    (var_plus / w).sqrt()
}

/// Effective sample size via the combined multi-chain autocorrelation with
/// Geyer's initial-positive / initial-monotone truncation (Vehtari et al.
/// 2021, eq. 10–11). `chains` are equal-length. Returns the total draw count
/// when the chains are essentially uncorrelated, less when autocorrelated.
pub fn effective_sample_size(chains: &[Vec<f64>]) -> f64 {
    let m = chains.len();
    if m == 0 {
        return 0.0;
    }
    let n = chains[0].len();
    if n < 4 || chains.iter().any(|c| c.len() != n) {
        return (m * n) as f64;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let vars: Vec<f64> = chains.iter().map(|c| sample_var(c)).collect();
    let grand = mean(&means);
    let w = vars.iter().sum::<f64>() / m as f64;

    // With a single chain there is no between-chain term; the marginal variance
    // estimate is just W. With ≥2 chains use the standard B/W combination.
    let var_plus = if m == 1 {
        w
    } else {
        let b =
            n as f64 / (m as f64 - 1.0) * means.iter().map(|mj| (mj - grand).powi(2)).sum::<f64>();
        (n as f64 - 1.0) / n as f64 * w + b / n as f64
    };
    if var_plus.is_nan() || var_plus <= 0.0 {
        return (m * n) as f64;
    }

    // Combined autocorrelation at each lag: ρ_t = 1 − (W − mean_m acov_m(t)) / var⁺.
    let max_lag = n - 1;
    let mut rho = vec![0.0_f64; max_lag + 1];
    for (t, rho_t) in rho.iter_mut().enumerate() {
        let mean_acov: f64 = chains
            .iter()
            .zip(&means)
            .map(|(c, &mu)| autocov(c, t, mu))
            .sum::<f64>()
            / m as f64;
        *rho_t = 1.0 - (w - mean_acov) / var_plus;
    }

    // Geyer initial-positive sequence: sum paired autocorrelations Ρ_k =
    // ρ_{2k} + ρ_{2k+1} while positive, enforcing a monotone non-increasing cap.
    let mut tau = 1.0; // ρ_0 = 1 contributes once via the 1 + 2Σ form below.
    let mut prev_pair = f64::INFINITY;
    let mut k = 1;
    while 2 * k < max_lag {
        let mut pair = rho[2 * k] + rho[2 * k + 1];
        if pair < 0.0 {
            break;
        }
        // Initial-monotone: never let a pair exceed the previous one.
        if pair > prev_pair {
            pair = prev_pair;
        }
        prev_pair = pair;
        tau += 2.0 * pair;
        k += 1;
    }
    // ρ_1 is added once (the k=0 pair's ρ_1 half); include it explicitly.
    tau += 2.0 * rho[1].max(0.0);

    let ess = (m * n) as f64 / tau.max(1.0);
    ess.min((m * n) as f64)
}

/// Inverse standard-normal CDF (probit), Acklam's rational approximation
/// (|error| < ~1.2e-9 over (0,1)). `p` is clamped to the open interval.
fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - plow {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Rank-normalize draws pooled across chains, preserving per-chain order
/// (Vehtari et al. 2021): replace each draw by `Φ⁻¹((rank − 3/8)/(S − 1/4))`,
/// with average ranks for ties. Makes the subsequent ESS robust to heavy tails
/// and non-normal posteriors.
fn rank_normalized(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    // Flatten with (chain, position) provenance.
    let mut flat: Vec<(usize, usize, f64)> = Vec::new();
    for (ci, c) in chains.iter().enumerate() {
        for (pi, &x) in c.iter().enumerate() {
            flat.push((ci, pi, x));
        }
    }
    let s = flat.len();
    if s == 0 {
        return chains.iter().map(|c| vec![0.0; c.len()]).collect();
    }
    // Average ranks (1-based) with tie handling.
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| {
        flat[a]
            .2
            .partial_cmp(&flat[b].2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0_f64; s];
    let mut i = 0;
    while i < s {
        let mut j = i + 1;
        while j < s && flat[order[j]].2 == flat[order[i]].2 {
            j += 1;
        }
        // ties order[i..j] share the average rank (1-based)
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of (i+1)..=j
        for &k in &order[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    // Transform to z-scores and re-split by chain.
    let mut out: Vec<Vec<f64>> = chains.iter().map(|c| vec![0.0; c.len()]).collect();
    for (idx, &(ci, pi, _)) in flat.iter().enumerate() {
        let z = inv_normal_cdf((ranks[idx] - 3.0 / 8.0) / (s as f64 - 0.25));
        out[ci][pi] = z;
    }
    out
}

/// Bulk effective sample size: ESS of the rank-normalized draws — the mixing of
/// the centre of the distribution (Vehtari et al. 2021).
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    effective_sample_size(&rank_normalized(chains))
}

/// Tail effective sample size: `min` of the ESS of the indicator series
/// `I(x ≤ q05)` and `I(x ≤ q95)` — the mixing of the distribution's tails,
/// which governs the reliability of the 5%/95% credible-interval edges.
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    let mut pooled: Vec<f64> = chains.iter().flatten().copied().collect();
    if pooled.len() < 4 {
        return (pooled.len()) as f64;
    }
    pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q05 = quantile_sorted(&pooled, 0.05);
    let q95 = quantile_sorted(&pooled, 0.95);
    let indicator = |thresh: f64| -> Vec<Vec<f64>> {
        chains
            .iter()
            .map(|c| {
                c.iter()
                    .map(|&x| if x <= thresh { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    };
    effective_sample_size(&indicator(q05)).min(effective_sample_size(&indicator(q95)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    fn iid_normal_chains(m: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..m)
            .map(|_| (0..n).map(|_| StandardNormal.sample(&mut rng)).collect())
            .collect()
    }

    #[test]
    fn test_quantile_sorted() {
        let s = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&s, 0.0), 1.0);
        assert_eq!(quantile_sorted(&s, 1.0), 5.0);
        assert_eq!(quantile_sorted(&s, 0.5), 3.0);
        assert!((quantile_sorted(&s, 0.25) - 2.0).abs() < 1e-12);
        assert!(quantile_sorted(&[], 0.5).is_nan());
    }

    #[test]
    fn test_split_rhat_mixed_is_near_one() {
        let chains = iid_normal_chains(4, 1000, 100);
        let rhat = split_rhat(&chains);
        assert!(rhat < 1.02, "R-hat for iid chains should be ~1, got {rhat}");
    }

    #[test]
    fn test_split_rhat_unmixed_is_large() {
        // Two chains with very different means → poor mixing → large R-hat.
        let mut chains = iid_normal_chains(2, 1000, 7);
        for x in chains[0].iter_mut() {
            *x += 10.0; // shift one chain far away
        }
        let rhat = split_rhat(&chains);
        assert!(
            rhat > 1.5,
            "R-hat for separated chains should be large, got {rhat}"
        );
    }

    #[test]
    fn test_ess_iid_near_total() {
        let m = 4;
        let n = 1000;
        let chains = iid_normal_chains(m, n, 314);
        let ess = effective_sample_size(&chains);
        let total = (m * n) as f64;
        assert!(
            ess > 0.6 * total && ess <= total,
            "iid ESS should be near total {total}, got {ess}"
        );
    }

    #[test]
    fn test_ess_autocorrelated_is_reduced() {
        // AR(1) with phi = 0.8 → strong positive autocorrelation → ESS ≪ N.
        let mut rng = StdRng::seed_from_u64(55);
        let phi = 0.8_f64;
        let n = 4000;
        let mut x = 0.0_f64;
        let chain: Vec<f64> = (0..n)
            .map(|_| {
                let eps: f64 = StandardNormal.sample(&mut rng);
                x = phi * x + eps;
                x
            })
            .collect();
        let ess = effective_sample_size(&[chain]);
        assert!(
            ess < 0.4 * n as f64,
            "AR(1) phi=0.8 ESS should be well below {n}, got {ess}"
        );
        assert!(ess > 1.0, "ESS should stay positive, got {ess}");
    }

    #[test]
    fn test_inv_normal_cdf_known_values() {
        assert!(inv_normal_cdf(0.5).abs() < 1e-9, "Phi^-1(0.5) = 0");
        assert!((inv_normal_cdf(0.975) - 1.959964).abs() < 1e-4);
        assert!((inv_normal_cdf(0.025) + 1.959964).abs() < 1e-4);
        assert!((inv_normal_cdf(0.95) - 1.644854).abs() < 1e-4);
        // Monotone + antisymmetric.
        assert!(inv_normal_cdf(0.1) < inv_normal_cdf(0.9));
        assert!((inv_normal_cdf(0.1) + inv_normal_cdf(0.9)).abs() < 1e-6);
    }

    #[test]
    fn test_ess_bulk_iid_near_total() {
        let (m, n) = (4, 1000);
        let chains = iid_normal_chains(m, n, 2024);
        let bulk = ess_bulk(&chains);
        let total = (m * n) as f64;
        assert!(
            bulk > 0.6 * total && bulk <= total,
            "iid bulk-ESS near total {total}, got {bulk}"
        );
    }

    #[test]
    fn test_ess_tail_positive_and_bounded() {
        let (m, n) = (4, 1000);
        let chains = iid_normal_chains(m, n, 999);
        let tail = ess_tail(&chains);
        let total = (m * n) as f64;
        // Tail-ESS is a valid ESS: positive and not above the draw count.
        assert!(
            tail > 1.0 && tail <= total + 1.0,
            "tail-ESS out of range: {tail}"
        );
    }

    #[test]
    fn test_ess_tail_reduced_for_autocorrelated() {
        // AR(1) tails mix slowly → tail-ESS well below the draw count.
        let mut rng = StdRng::seed_from_u64(77);
        let phi = 0.85_f64;
        let n = 4000;
        let mut x = 0.0_f64;
        let chain: Vec<f64> = (0..n)
            .map(|_| {
                let eps: f64 = StandardNormal.sample(&mut rng);
                x = phi * x + eps;
                x
            })
            .collect();
        let tail = ess_tail(&[chain]);
        assert!(
            tail < 0.5 * n as f64,
            "AR(1) tail-ESS should be reduced, got {tail}"
        );
        assert!(tail > 1.0);
    }
}
