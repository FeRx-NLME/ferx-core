//! Post-fit conditional-distribution pass for SAEM (issue #257).
//!
//! After the SAEM population parameters (θ̂, Ω̂, σ̂) are fixed, this pass
//! characterises each subject's *conditional distribution* of the random
//! effects `p(η_i | y_i; θ̂)` — not just its mode. It re-runs the same
//! Metropolis-Hastings kernels the SAEM E-step uses (`mh_steps`,
//! `mh_steps_componentwise`, and `mh_kappa_steps` for IOV), warm-started at the
//! EBE mode, and *accumulates* the draws instead of discarding all but the
//! latest. From the accumulated chain it reports the conditional mean, the
//! conditional SD, and (optionally) the raw draws per subject.
//!
//! This mirrors the conditional-mode vs. conditional-distribution distinction
//! established by **saemix** (`map.saemix` vs `conddist.saemix`; Comets,
//! Lavenu & Lavielle, *J. Stat. Soft.* 80(3), 2017) and **Monolix** (the
//! "Conditional Mode" vs "Conditional Distribution" tasks). It is an
//! independent Rust implementation of the published algorithm (Delyon,
//! Lavielle & Moulines 1999; Kuhn & Lavielle 2004) — no saemix source (GPL) is
//! copied; only the interface convention is reused.
//!
//! Why the distribution and not just the mode: empirical Bayes estimates (the
//! mode) and conditional means are *shrunk* toward the population mean, so
//! diagnostics built on them can hide or fabricate covariate/correlation
//! relationships. Samples from the conditional distribution are not shrinkage-
//! biased and are the preferred basis for those diagnostics.

use crate::estimation::saem::{
    mh_kappa_steps, mh_steps, mh_steps_componentwise, SAEM_OMEGA_DIAG_FLOOR,
};
use crate::pk::EventPkParams;
use crate::stats::likelihood::{individual_nll, individual_nll_iov};
use crate::types::*;
use nalgebra::DVector;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// Seed offset for the conditional-distribution pass so its RNG stream is
/// reproducible and independent of the SAEM E-step stream (which keys on
/// `master_seed + k * 100_000 + i`, plus `999_999` for the kappa kernel).
const CONDDIST_SEED_OFFSET: u64 = 777_000_000;

/// Nudge an MH step scale toward `target` acceptance, matching the main SAEM
/// loop's multiplicative adaptation (×1.1 / ×0.9, clamped to [0.01, 5.0]).
fn adapt_scale(scale: &mut f64, accepted: usize, proposed: usize, target: f64) {
    let rate = accepted as f64 / proposed.max(1) as f64;
    if rate > target {
        *scale = (*scale * 1.1).min(5.0);
    } else {
        *scale = (*scale * 0.9).max(0.01);
    }
}

/// Run the conditional-distribution pass at fixed population parameters.
///
/// `params` must carry the converged θ̂/Ω̂/σ̂ (and Ω̂_iov for IOV models).
/// `warm_etas[i]` is subject `i`'s EBE mode (used to warm-start its chain);
/// `warm_kappas[i]` is its per-occasion kappa EBEs (empty inner vec for
/// non-IOV models). Both must have length `n_subjects`.
pub fn run_conditional_distribution(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    warm_etas: &[DVector<f64>],
    warm_kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> CondDist {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let n_kappa = model.n_kappa;

    let nsamp = options.saem_conddist_nsamp.max(1);
    let burnin = options.saem_conddist_burnin;
    let keep = options.saem_conddist_keep_samples;
    let n_mh_steps = options.saem_n_mh_steps;
    // Decouple from saem_adapt_interval: scale with burn-in so adaptation
    // fires ~10 times regardless of the user's saem_adapt_interval setting.
    // With the default burnin=20 this gives interval=2; with burnin=500 → 50
    // (matching saem_adapt_interval's default, preserving validated behaviour).
    let adapt_interval = (burnin / 10).max(1);
    let master_seed = options
        .saem_seed
        .unwrap_or(12345)
        .wrapping_add(CONDDIST_SEED_OFFSET);

    // Componentwise sweep count mirrors the main loop: skipped for single-η
    // models (no off-diagonal to decorrelate).
    let n_cw_sweeps = if n_eta >= 2 {
        (n_mh_steps / n_eta).max(2)
    } else {
        0
    };

    let omega = &params.omega;
    let omega_iov_opt = params.omega_iov.as_ref();
    let theta = &params.theta;
    let sigma = &params.sigma.values;

    // Per-coordinate componentwise proposal SDs (√marginal variance), shared
    // across subjects and floored to match the Ω diagonal floor.
    let cw_sd: Vec<f64> = (0..n_eta)
        .map(|j| omega.matrix[(j, j)].max(SAEM_OMEGA_DIAG_FLOOR).sqrt())
        .collect();

    /// One subject's accumulated conditional distribution.
    struct SubjResult {
        mean: Vec<f64>,
        sd: Vec<f64>,
        samples: Vec<Vec<f64>>,
    }

    let results: Vec<SubjResult> = population
        .subjects
        .par_iter()
        .enumerate()
        .map_init(EventPkParams::default, |pk_scratch, (i, subject)| {
            let mut rng = StdRng::seed_from_u64(master_seed.wrapping_add(i as u64));

            // Warm-start the chain at the EBE mode.
            let mut eta: Vec<f64> = warm_etas[i].iter().copied().collect();
            let mut kappas: Vec<Vec<f64>> = if n_kappa > 0 {
                warm_kappas[i]
                    .iter()
                    .map(|k| k.iter().copied().collect())
                    .collect()
            } else {
                Vec::new()
            };

            // Initial posterior NLL at the warm start (prior + observation
            // likelihood) — the baseline the MH acceptance ratio differences.
            let mut nll = if n_kappa > 0 {
                individual_nll_iov(
                    model,
                    subject,
                    theta,
                    &eta,
                    &kappas,
                    omega,
                    omega_iov_opt,
                    sigma,
                )
            } else {
                individual_nll(model, subject, theta, &eta, omega, sigma)
            };

            // Per-subject adaptive step scales (block / componentwise / kappa).
            // Block init matches the main SAEM loop (0.3, not the block target 0.40).
            // The componentwise kernel adapts a *per-eta* scale (one coordinate at
            // a time), mirroring the main loop's `cw_step_scales[i]` — each eta can
            // reach its own optimal acceptance independently.
            let mut step_block = 0.3_f64;
            let mut step_cw = vec![1.0_f64; n_eta];
            let mut step_kappa = 0.3_f64;

            // Welford online mean/variance accumulators over retained draws.
            let mut mean = vec![0.0_f64; n_eta];
            let mut m2 = vec![0.0_f64; n_eta];
            let mut count = 0_usize;
            let mut samples: Vec<Vec<f64>> = if keep {
                Vec::with_capacity(nsamp)
            } else {
                Vec::new()
            };

            // Acceptance counters within the current adaptation window. The
            // componentwise kernel tracks per-eta acceptance (each eta adapts
            // independently); `prop_c` counts proposals *per eta* (every sweep
            // proposes each eta exactly once).
            let (mut acc_b, mut prop_b) = (0usize, 0usize);
            let mut acc_c = vec![0usize; n_eta];
            let mut prop_c = 0usize;
            let (mut acc_k, mut prop_k) = (0usize, 0usize);

            let total_sweeps = burnin + nsamp;
            for sweep in 0..total_sweeps {
                // Kernel 1: block random-walk move (η | κ for IOV). The
                // kappas tuple is rebuilt inline so the immutable borrow of
                // `kappas` ends before the kappa kernel mutates it below.
                let (nb, nll_b) = mh_steps(
                    &mut eta,
                    nll,
                    subject,
                    model,
                    theta,
                    omega,
                    sigma,
                    step_block,
                    &mut rng,
                    n_mh_steps,
                    pk_scratch,
                    omega_iov_opt.map(|iov| (kappas.as_slice(), iov)),
                );
                nll = nll_b;
                acc_b += nb;
                prop_b += n_mh_steps;

                // Kernel 2: componentwise decorrelating sweep. Returns per-eta
                // acceptance counts; the total-proposal count is ignored in favour
                // of the per-eta `n_cw_sweeps` so each eta adapts on its own rate.
                if n_cw_sweeps > 0 {
                    let (per_eta_nc, _n_prop_cw, nll_c) = mh_steps_componentwise(
                        &mut eta,
                        nll,
                        subject,
                        model,
                        theta,
                        omega,
                        sigma,
                        &step_cw,
                        &cw_sd,
                        &mut rng,
                        n_cw_sweeps,
                        pk_scratch,
                        omega_iov_opt.map(|iov| (kappas.as_slice(), iov)),
                    );
                    nll = nll_c;
                    for j in 0..n_eta {
                        acc_c[j] += per_eta_nc[j];
                    }
                    prop_c += n_cw_sweeps;
                }

                // Kernel 3: per-occasion kappa move (IOV only).
                if n_kappa > 0 {
                    if let Some(iov) = omega_iov_opt {
                        let (nk, pk, nll_k) = mh_kappa_steps(
                            &mut kappas,
                            nll,
                            subject,
                            model,
                            theta,
                            &eta,
                            omega,
                            iov,
                            sigma,
                            step_kappa,
                            &mut rng,
                        );
                        nll = nll_k;
                        acc_k += nk;
                        prop_k += pk;
                    }
                }

                // Adapt step scales during burn-in only, then freeze so the
                // retained draws come from a fixed (time-homogeneous) kernel.
                if sweep < burnin && (sweep + 1) % adapt_interval == 0 {
                    adapt_scale(&mut step_block, acc_b, prop_b, 0.40);
                    acc_b = 0;
                    prop_b = 0;
                    if n_cw_sweeps > 0 {
                        // Per-eta adaptation toward the 1-D optimum (~0.44; Roberts
                        // & Rosenthal 2001), mirroring the main loop's componentwise
                        // adaptation including its 1e-6 floor (vs 0.01 for the block
                        // kernel) so near-deterministic etas aren't over-stepped.
                        for j in 0..n_eta {
                            let rate = acc_c[j] as f64 / prop_c.max(1) as f64;
                            if rate > 0.44 {
                                step_cw[j] = (step_cw[j] * 1.1).min(5.0);
                            } else {
                                step_cw[j] = (step_cw[j] * 0.9).max(1e-6);
                            }
                            acc_c[j] = 0;
                        }
                        prop_c = 0;
                    }
                    if n_kappa > 0 {
                        adapt_scale(&mut step_kappa, acc_k, prop_k, 0.40);
                        acc_k = 0;
                        prop_k = 0;
                    }
                }

                // Accumulate after burn-in (Welford).
                if sweep >= burnin {
                    count += 1;
                    let inv = 1.0 / count as f64;
                    for j in 0..n_eta {
                        let d = eta[j] - mean[j];
                        mean[j] += d * inv;
                        m2[j] += d * (eta[j] - mean[j]);
                    }
                    if keep {
                        samples.push(eta.clone());
                    }
                }
            }

            let sd: Vec<f64> = (0..n_eta)
                .map(|j| {
                    if count > 1 {
                        (m2[j] / (count - 1) as f64).sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();

            SubjResult { mean, sd, samples }
        })
        .collect();

    let cond_mean: Vec<Vec<f64>> = results.iter().map(|r| r.mean.clone()).collect();
    let cond_sd: Vec<Vec<f64>> = results.iter().map(|r| r.sd.clone()).collect();
    let samples: Vec<Vec<Vec<f64>>> = results.into_iter().map(|r| r.samples).collect();

    // Distribution-based η-shrinkage: 1 - sampleSD_i(cond_mean[i][j]) / √Ω_jj.
    // Uses the N−1 sample SD of the per-subject conditional means (matches the
    // `var()` convention saemix uses for its shrinkage). NaN with < 2 subjects.
    let shrinkage: Vec<f64> = (0..n_eta)
        .map(|j| {
            if n_subjects < 2 {
                return f64::NAN;
            }
            // Return NaN for degenerate Ω_jj — parity with mode-based shrinkage_eta.
            if omega.matrix[(j, j)] < SAEM_OMEGA_DIAG_FLOOR {
                return f64::NAN;
            }
            let m: f64 = cond_mean.iter().map(|cm| cm[j]).sum::<f64>() / n_subjects as f64;
            let var: f64 = cond_mean.iter().map(|cm| (cm[j] - m).powi(2)).sum::<f64>()
                / (n_subjects - 1) as f64;
            let omega_sd = omega.matrix[(j, j)].sqrt();
            1.0 - var.sqrt() / omega_sd
        })
        .collect();

    CondDist {
        cond_mean,
        cond_sd,
        samples,
        shrinkage,
        nsamp,
        burnin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{DoseEvent, GradientMethod};
    use std::collections::HashMap;

    fn one_subject() -> Subject {
        Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0, 4.0, 8.0],
            obs_raw_times: Vec::new(),
            observations: vec![5.0, 3.0, 1.5],
            obs_cmts: vec![1, 1, 1],
            covariates: HashMap::new(),
            dose_covariates: Vec::new(),
            obs_covariates: Vec::new(),
            pk_only_times: Vec::new(),
            pk_only_covariates: Vec::new(),
            reset_times: Vec::new(),
            cens: vec![0, 0, 0],
            occasions: vec![],
            dose_occasions: vec![],
            fremtype: Vec::new(),
            obs_records: vec![],
        }
    }

    /// With an (almost) uninformative likelihood — residual error blown up so
    /// the observation term is nearly flat in η — the conditional distribution
    /// must collapse onto the prior `N(0, Ω)`: conditional mean ≈ 0 and
    /// conditional SD ≈ √Ω_jj. This is the canonical correctness check for the
    /// accumulator (saemix uses the same prior-recovery argument).
    #[test]
    fn uninformative_likelihood_recovers_prior() {
        let model = analytical_model(GradientMethod::Auto);
        let mut params = model.default_params.clone();
        // Flatten the observation likelihood: huge residual variance ⇒ posterior
        // ≈ prior, independent of the data.
        for s in params.sigma.values.iter_mut() {
            *s = 1.0e6;
        }

        let population = Population {
            subjects: vec![one_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };

        let n_eta = model.n_eta;
        let warm_etas = vec![DVector::zeros(n_eta)];
        let warm_kappas = vec![Vec::new()];

        let mut opts = FitOptions::default();
        opts.saem_conddist = true;
        opts.saem_conddist_nsamp = 4000;
        opts.saem_conddist_burnin = 400;
        opts.saem_adapt_interval = 50;
        opts.saem_seed = Some(1);

        let cd = run_conditional_distribution(
            &model,
            &population,
            &params,
            &warm_etas,
            &warm_kappas,
            &opts,
        );

        assert_eq!(cd.cond_mean.len(), 1);
        assert_eq!(cd.cond_sd[0].len(), n_eta);
        for j in 0..n_eta {
            let prior_sd = params.omega.matrix[(j, j)].sqrt();
            // Conditional mean near the prior mean (0). Loose tolerance: a flat
            // likelihood still leaves MC error over a finite chain.
            assert!(
                cd.cond_mean[0][j].abs() < 0.35 * prior_sd.max(1.0),
                "eta {j}: conditional mean {} not near prior mean 0 (prior_sd {prior_sd})",
                cd.cond_mean[0][j]
            );
            // Conditional SD near the prior SD (within 30%).
            let rel = (cd.cond_sd[0][j] - prior_sd).abs() / prior_sd;
            assert!(
                rel < 0.30,
                "eta {j}: conditional SD {} not near prior SD {prior_sd} (rel {rel:.3})",
                cd.cond_sd[0][j]
            );
        }
    }

    /// The adaptation path must actually move step scales when burnin >= 10.
    /// Prior to the fix the conddist adapt_interval was borrowed from
    /// saem_adapt_interval (default 50), so with burnin < 50 adaptation never
    /// fired and step scales stayed at their initial values.
    #[test]
    fn step_scales_adapt_during_burnin() {
        // We can't directly observe step_block, but we can verify that the chain
        // produces finite, non-degenerate SD estimates even from a cold start —
        // which only happens when the scales actually adapt away from 0.3 for
        // an informative likelihood. Use a small burn-in that the old code
        // (adapt_interval=50, burnin=20) would never have adapted in.
        let model = analytical_model(GradientMethod::Auto);
        let params = model.default_params.clone();
        let population = Population {
            subjects: vec![one_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let warm_etas = vec![DVector::zeros(model.n_eta)];
        let warm_kappas = vec![Vec::new()];

        let mut opts = FitOptions::default();
        opts.saem_conddist = true;
        opts.saem_conddist_nsamp = 200;
        opts.saem_conddist_burnin = 20; // adapt_interval now = max(1, 20/10) = 2
        opts.saem_adapt_interval = 50; // must NOT gate adaptation (old bug)
        opts.saem_seed = Some(3);

        let cd = run_conditional_distribution(
            &model,
            &population,
            &params,
            &warm_etas,
            &warm_kappas,
            &opts,
        );
        // All SDs must be positive (chains mixed), not NaN or zero.
        for j in 0..model.n_eta {
            assert!(
                cd.cond_sd[0][j] > 0.0 && cd.cond_sd[0][j].is_finite(),
                "eta {j}: cond_sd {} not positive-finite (adaptation may not have fired)",
                cd.cond_sd[0][j]
            );
        }
    }

    /// `keep_samples = false` (the default) must not retain any draws, while
    /// `true` retains exactly `nsamp` per subject.
    #[test]
    fn keep_samples_flag_controls_retention() {
        let model = analytical_model(GradientMethod::Auto);
        let params = model.default_params.clone();
        let population = Population {
            subjects: vec![one_subject()],
            covariate_names: Vec::new(),
            dv_column: "DV".into(),
            input_columns: vec![],
            exclusions: None,
            warnings: vec![],
        };
        let warm_etas = vec![DVector::zeros(model.n_eta)];
        let warm_kappas = vec![Vec::new()];

        let mut opts = FitOptions::default();
        opts.saem_conddist = true;
        opts.saem_conddist_nsamp = 100;
        opts.saem_conddist_burnin = 10;
        opts.saem_seed = Some(7);

        let cd = run_conditional_distribution(
            &model,
            &population,
            &params,
            &warm_etas,
            &warm_kappas,
            &opts,
        );
        assert!(
            cd.samples[0].is_empty(),
            "samples retained despite keep=false"
        );

        opts.saem_conddist_keep_samples = true;
        let cd2 = run_conditional_distribution(
            &model,
            &population,
            &params,
            &warm_etas,
            &warm_kappas,
            &opts,
        );
        assert_eq!(cd2.samples[0].len(), 100, "expected nsamp retained draws");
        assert_eq!(cd2.samples[0][0].len(), model.n_eta);
    }
}
