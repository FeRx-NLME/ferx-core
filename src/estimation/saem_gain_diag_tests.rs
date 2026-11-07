//! #2721 measurement harness (`#[ignore]`d diagnostic, not a regression test).
//!
//! At a fixed parameter point it runs a long MH chain per subject (SAEM's own
//! `mh_steps` kernel) and, at every retained draw, the complete-data score in the
//! parameterisation SAEM steps in:
//!
//! - mu-referenced θ: the prior score `[Ω⁻¹η]_j`, complete information `Ω⁻¹`;
//! - free non-mu θ and σ: the data score at fixed η and its expected Fisher
//!   information (`obs_nll_subject_grad_fisher`, what `score_sa` uses);
//! - Ω's free entries: the prior score `½(Ω⁻¹ηηᵀΩ⁻¹ − Ω⁻¹)` and `½tr(Ω⁻¹E Ω⁻¹E')`.
//!
//! From those it reports, per coordinate, the missing-information fraction
//! `1 − [I_c⁻¹]_jj / [I_obs⁻¹]_jj` with `I_obs` estimated two ways (Louis:
//! `I_c − Σ Var(s|y)`; marginal BHHH: `Σ s̄ᵢs̄ᵢᵀ`), the smallest generalised
//! eigenvalue of `(I_obs, I_c)` (SA's √k condition is that it exceeds ½), and the
//! variance left after the zero-variance control variate `s − Bᵢ∇η l` with the
//! per-subject in-sample `Bᵢ` (the ideal coefficient, so an upper bound on the gain).

use super::*;
use crate::estimation::fixed_eta_gradient::obs_nll_subject_grad_fisher;
use crate::stats::likelihood::individual_nll;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Coords {
    names: Vec<String>,
    /// (packed θ index, η index) of each mu-referenced pair.
    mu: Vec<(usize, usize)>,
    /// Indices into the packed `[θ, σ]` vector of the data-scored coordinates.
    data: Vec<usize>,
    /// Free lower-triangle Ω entries `(a, b)`, `a >= b`.
    om: Vec<(usize, usize)>,
}

impl Coords {
    fn n(&self) -> usize {
        self.mu.len() + self.data.len() + self.om.len()
    }
}

fn coords(model: &CompiledModel, params: &ModelParameters) -> Coords {
    let n_theta = params.theta.len();
    let mu = get_mu_ref_pairs(model, &params.theta_lower);
    let mut names = Vec::new();
    for &(t, _) in &mu {
        names.push(format!("mu:{}", params.theta_names[t]));
    }
    let mut data = Vec::new();
    for t in 0..n_theta {
        if !mu.iter().any(|&(m, _)| m == t) && !params.theta_fixed[t] {
            data.push(t);
            names.push(format!("theta:{}", params.theta_names[t]));
        }
    }
    for (k, nm) in params.sigma.names.iter().enumerate() {
        if !params.sigma_fixed.get(k).copied().unwrap_or(false) {
            data.push(n_theta + k);
            names.push(format!("sigma:{nm}"));
        }
    }
    let om_m = &params.omega;
    let mut om = Vec::new();
    for a in 0..om_m.matrix.nrows() {
        for b in 0..=a {
            if om_m.free_mask[(a, b)] {
                om.push((a, b));
                names.push(format!("omega({},{})", a + 1, b + 1));
            }
        }
    }
    Coords {
        names,
        mu,
        data,
        om,
    }
}

fn sym_unit(n: usize, a: usize, b: usize) -> DMatrix<f64> {
    let mut e = DMatrix::zeros(n, n);
    e[(a, b)] = 1.0;
    e[(b, a)] = 1.0;
    e
}

/// Per-subject accumulators over the retained draws.
#[derive(Default)]
struct Acc {
    k: f64,
    s: DVector<f64>,
    ss: DMatrix<f64>,
    e: DVector<f64>,
    ee: DMatrix<f64>,
    se: DMatrix<f64>,
    f: DMatrix<f64>,
}

#[allow(clippy::too_many_arguments)]
fn diagnose(
    label: &str,
    model: &CompiledModel,
    pop: &Population,
    params: &ModelParameters,
    n_draws: usize,
    thin: usize,
    burn: usize,
    seed: u64,
) {
    let c = coords(model, params);
    let n = c.n();
    let n_eta = model.n_eta;
    let n_theta = params.theta.len();
    let n_sigma = params.sigma.values.len();
    let mask: Vec<bool> = params
        .theta_lower
        .iter()
        .map(|&lo| crate::estimation::parameterization::theta_packs_log(lo))
        .collect();
    let mut lower: Vec<f64> = (0..n_theta)
        .map(|i| {
            if mask[i] {
                params.theta_lower[i].max(1e-10).ln()
            } else {
                params.theta_lower[i]
            }
        })
        .collect();
    let mut upper: Vec<f64> = (0..n_theta)
        .map(|i| {
            if mask[i] {
                params.theta_upper[i].min(1e9).ln()
            } else {
                params.theta_upper[i]
            }
        })
        .collect();
    lower.extend(std::iter::repeat_n(-8.0, n_sigma));
    upper.extend(std::iter::repeat_n(5.0, n_sigma));
    let omega = &params.omega;
    let oi = omega.matrix.clone().try_inverse().expect("Ω is invertible");
    // Ω-block complete information per subject (η-independent).
    let mut f_om = DMatrix::<f64>::zeros(c.om.len(), c.om.len());
    for (p, &(a, b)) in c.om.iter().enumerate() {
        let ea = sym_unit(n_eta, a, b);
        for (q, &(cc, d)) in c.om.iter().enumerate() {
            let eb = sym_unit(n_eta, cc, d);
            // `sym_unit(a, a)` is the single-entry `e_a e_aᵀ`, so one formula
            // serves the diagonal and the off-diagonal entries alike.
            f_om[(p, q)] = 0.5 * (&oi * &ea * &oi * &eb).trace();
        }
    }
    let n_mu = c.mu.len();
    let n_data = c.data.len();
    let mut totals_ic = DMatrix::<f64>::zeros(n, n);
    let mut totals_var = DMatrix::<f64>::zeros(n, n);
    let mut totals_var_cv = DMatrix::<f64>::zeros(n, n);
    let mut bhhh = DMatrix::<f64>::zeros(n, n);
    let mut sbar_sum = DVector::<f64>::zeros(n);
    let mut stein_worst = 0.0f64;
    let mut acc_rate = (0usize, 0usize);
    let mut scratch_pk = EventPkParams::default();
    for (si, subject) in pop.subjects.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(seed ^ (si as u64).wrapping_mul(0x9E37_79B9));
        let mut eta = vec![0.0; n_eta];
        let theta = &params.theta;
        let sigma = &params.sigma.values;
        let mut nll = individual_nll(model, subject, theta, &eta, omega, sigma);
        let mut mh = MhScratch::default();
        mh.begin_subject(model, subject, theta, n_eta);
        let mut scale = 1.0f64;
        let mut step =
            |eta: &mut Vec<f64>, nll: &mut f64, k: usize, scale: f64, rng: &mut StdRng| {
                let (a, v) = mh_steps(
                    eta, *nll, subject, model, theta, omega, sigma, scale, None, rng, k, &mut mh,
                    None, None,
                );
                *nll = v;
                a
            };
        // Burn-in adapts the random-walk scale to a 0.3 acceptance (Robbins–Monro on
        // ln scale, blocks of 10), then freezes it so the retained chain is a fixed
        // reversible kernel.
        for b in 0..burn.div_ceil(10) {
            let a = step(&mut eta, &mut nll, 10, scale, &mut rng);
            let rate = a as f64 / 10.0;
            scale = (scale * ((rate - 0.3) / (1.0 + b as f64).sqrt()).exp()).clamp(1e-3, 3.0);
        }
        let mut acc = Acc {
            s: DVector::zeros(n),
            ss: DMatrix::zeros(n, n),
            e: DVector::zeros(n_eta),
            ee: DMatrix::zeros(n_eta, n_eta),
            se: DMatrix::zeros(n, n_eta),
            f: DMatrix::zeros(n, n),
            ..Acc::default()
        };
        for _ in 0..n_draws {
            acc_rate.0 += step(&mut eta, &mut nll, thin, scale, &mut rng);
            acc_rate.1 += thin;
            let (_, g, fis) = obs_nll_subject_grad_fisher(
                model,
                subject,
                theta,
                sigma,
                &eta,
                &mask,
                &lower,
                &upper,
                n_theta,
                n_sigma,
                &mut scratch_pk,
            );
            let fis = fis.expect("expected information in scope");
            let nn = n_theta + n_sigma;
            let ev = DVector::from_column_slice(&eta);
            let oe = &oi * &ev;
            let gm = (&oe * oe.transpose() - &oi) * 0.5;
            let mut s = DVector::<f64>::zeros(n);
            let mut f = DMatrix::<f64>::zeros(n, n);
            for (p, &(_, j)) in c.mu.iter().enumerate() {
                s[p] = oe[j];
                for (q, &(_, jj)) in c.mu.iter().enumerate() {
                    f[(p, q)] = oi[(j, jj)];
                }
            }
            for (p, &i) in c.data.iter().enumerate() {
                s[n_mu + p] = -g[i];
                for (q, &j) in c.data.iter().enumerate() {
                    f[(n_mu + p, n_mu + q)] = fis[i * nn + j];
                }
            }
            for (p, &(a, b)) in c.om.iter().enumerate() {
                s[n_mu + n_data + p] = if a == b { gm[(a, a)] } else { 2.0 * gm[(a, b)] };
            }
            f.view_mut((n_mu + n_data, n_mu + n_data), (c.om.len(), c.om.len()))
                .copy_from(&f_om);
            let pfull = ModelParameters {
                theta: theta.clone(),
                ..params.clone()
            };
            let e = crate::sens::ode_provider::adjoint::exact_inner_gradient(
                model, subject, &pfull, &eta, None,
            )
            .expect("exact η-gradient in scope");
            let e = DVector::from_column_slice(&e[..n_eta]);
            acc.k += 1.0;
            acc.ss += &s * s.transpose();
            acc.se += &s * e.transpose();
            acc.ee += &e * e.transpose();
            acc.s += s;
            acc.e += e;
            acc.f += f;
        }
        let k = acc.k;
        let sbar = &acc.s / k;
        let ebar = &acc.e / k;
        let vs = &acc.ss / k - &sbar * sbar.transpose();
        let ve = &acc.ee / k - &ebar * ebar.transpose();
        let cse = &acc.se / k - &sbar * ebar.transpose();
        for j in 0..n_eta {
            let sd = ve[(j, j)].max(1e-300).sqrt();
            let z = ebar[j] / sd;
            assert!(z.is_finite(), "non-finite Stein check");
            stein_worst = stein_worst.max(z.abs());
        }
        let b = &cse
            * ve.clone()
                .try_inverse()
                .unwrap_or_else(|| DMatrix::zeros(n_eta, n_eta));
        let vcv = &vs - &b * cse.transpose();
        totals_ic += &acc.f / k;
        totals_var += &vs;
        totals_var_cv += vcv;
        bhhh += &sbar * sbar.transpose();
        sbar_sum += sbar;
    }
    let ic = totals_ic;
    let louis = &ic - &totals_var;
    let ic_inv = ic.clone().try_inverse().expect("I_c invertible");
    let report = |name: &str, iobs: &DMatrix<f64>| {
        let inv = iobs.clone().try_inverse();
        let l = ic.clone().cholesky().expect("I_c PD").l();
        let li = l.clone().try_inverse().unwrap();
        let m = &li * iobs * li.transpose();
        let ev = m.symmetric_eigen().eigenvalues;
        assert!(ev.iter().all(|e| e.is_finite()), "non-finite eigenvalue");
        let min_ev = ev.iter().cloned().fold(f64::INFINITY, f64::min);
        eprintln!(
            "  [{name}] generalised eig(I_obs, I_c): min {min_ev:.3}, all {:?}",
            ev.iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>()
        );
        inv
    };
    eprintln!(
        "\n== {label}: {} subjects, {n_draws} draws × thin {thin}, MH acceptance {:.2}; Stein check worst |mean ∇η|/sd = {stein_worst:.3}",
        pop.subjects.len(),
        acc_rate.0 as f64 / acc_rate.1 as f64
    );
    let li = report("Louis", &louis);
    let bi = report("marginal BHHH", &bhhh);
    eprintln!(
        "  {:<22} {:>10} {:>10} {:>12} {:>10}",
        "coordinate", "miss(Lou)", "miss(BHHH)", "CV var ratio", "Σs̄/√I_c"
    );
    for j in 0..n {
        let mf = |inv: &Option<DMatrix<f64>>| match inv {
            Some(m) if m[(j, j)] > 0.0 => 1.0 - ic_inv[(j, j)] / m[(j, j)],
            _ => f64::NAN,
        };
        eprintln!(
            "  {:<22} {:>10.3} {:>10.3} {:>12.4} {:>10.2}",
            c.names[j],
            mf(&li),
            mf(&bi),
            totals_var_cv[(j, j)] / totals_var[(j, j)],
            sbar_sum[j] / ic[(j, j)].sqrt()
        );
    }
}

fn load(model_src: &str, data: &str) -> (CompiledModel, Population) {
    let parsed = crate::parser::model_parser::parse_full_model(model_src).expect("model parses");
    let (pop, _) = crate::api::read_population_for(
        &parsed.model,
        &parsed.covariate_decls,
        data,
        None,
        None,
        None,
        &[],
    )
    .expect("data reads");
    (parsed.model, pop)
}

const SPARSE_MODEL: &str = r#"
[parameters]
  theta TVCL(4.0, 0.2, 40.0)
  theta TVV1(22.0, 2.0, 200.0)
  theta TVV2(13.0, 1.0, 200.0)
  block_omega (ETA_CL, ETA_V1, ETA_V2) = [0.17, 0.02, 0.15, 0.02, 0.02, 0.69]
  sigma PROP_ERR ~ 0.13 (sd)
  sigma ADD_ERR  ~ 1.80 (sd)

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V1 = TVV1 * exp(ETA_V1)
  Q  = 12.0
  V2 = TVV2 * exp(ETA_V2)

[structural_model]
  pk two_cpt_iv(cl=CL, v1=V1, q=Q, v2=V2)

[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
"#;

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

#[test]
#[ignore = "diagnostic for #2721: run with --ignored --nocapture"]
fn saem_gain_missing_information() {
    let n_draws = env_usize("DIAG_DRAWS", 400);
    let thin = env_usize("DIAG_THIN", 4);
    let burn = env_usize("DIAG_BURN", 200);
    // #1445's sparse combined-error fixture, at its simulation truth.
    let (m, p) = load(SPARSE_MODEL, "data/saem_sparse_combined.csv");
    diagnose(
        "sparse combined (#1445), truth",
        &m,
        &p,
        &m.default_params,
        n_draws,
        thin,
        burn,
        1,
    );
    // Warfarin at its FOCEI estimate.
    let src = std::fs::read_to_string("examples/warfarin_saem.ferx").unwrap();
    let (m, p) = load(&src, "data/warfarin.csv");
    let opts = crate::types::FitOptions {
        method: crate::types::EstimationMethod::FoceI,
        interaction: true,
        run_covariance_step: false,
        ..crate::types::FitOptions::default()
    };
    let r = crate::fit(&m, &p, &m.default_params, &opts).expect("FOCEI fits");
    let mut at = m.default_params.clone();
    at.theta = r.theta.clone();
    at.omega = OmegaMatrix::from_matrix_with_mask(
        r.omega.clone(),
        at.omega.eta_names.clone(),
        at.omega.diagonal,
        at.omega.free_mask.clone(),
    );
    at.sigma.values = r.sigma.clone();
    diagnose(
        "warfarin, FOCEI estimate",
        &m,
        &p,
        &at,
        n_draws,
        thin,
        burn,
        2,
    );
}
