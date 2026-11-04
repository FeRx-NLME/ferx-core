// Stan anchor for ferx's Bayes sampler on a non-Gaussian random effect (#1716,
// review of #1721, finding 4). The posterior is the one ferx's `method = bayes`
// targets for tests/eta_density_stan_anchor.rs's model, prior for prior:
//
//   ferx (src/estimation/bayes.rs)                    | here
//   --------------------------------------------------+-------------------------------
//   θ with lower bound ≥ 0: log θ ~ N(log θ₀, 10²),   | log_theta ~ normal(log θ₀, 10),
//     truncated to the declared bounds (the conjugate |   bounded; the truncation
//     μ-ref move is off: ETA_CL is a density, so not  |   constant is fixed
//     every η is a log μ-reference)                   |
//   σ: log σ ~ N(log σ₀, 10²), σ an SD                | log_sigma ~ normal(log σ₀, 10)
//   Ω diagonal: ω_j ~ IG((ν₀ − p + 1)/2, Λ₀ⱼⱼ/2),     | omega ~ inv_gamma(1.5, ω₀/2)
//     ν₀ = p + 2, Λ₀ = the initial Ω; the density    |
//     block's IW(ν₀ − (p − 1), Λ₀) marginal is the   |
//     same IG                                        |
//   ETA_CL ~ density gamma: mean 1, variance ω_CL    | eta_cl ~ gamma(1/ω, 1/ω)
//   ETA_V, ETA_KA ~ N(0, ω)                          | normal(0, sqrt(ω))
//   DV ~ N(f, (f σ)²), proportional                  | same
//
// Both samplers work on the latent z = log(ETA_CL): its density is the Gamma's at
// exp(z) times the Jacobian exp(z), so the effect never reaches its 0 boundary.
data {
  int<lower=1> N;                 // subjects
  int<lower=1> M;                 // observations
  array[M] int<lower=1, upper=N> id;
  vector<lower=0>[M] time;
  vector[M] dv;
  vector<lower=0>[N] dose;        // one bolus into the depot at t = 0
  vector[3] theta0;               // TVCL, TVV, TVKA initial values
  vector[3] theta_lo;
  vector[3] theta_hi;
  vector[3] omega0;               // initial ω: CL, V, KA
  real<lower=0> sigma0;           // initial proportional SD
  int<lower=0, upper=1> gaussian_cl;  // 1: the Gaussian twin, CL = TVCL·exp(η)
}
parameters {
  vector<lower=log(theta_lo), upper=log(theta_hi)>[3] log_theta;
  real log_sigma;
  vector<lower=0>[3] omega;
  vector[N] z_cl;                 // the gamma effect's latent, log(ETA_CL)
  vector[N] eta_cl_gauss;         // the Gaussian twin's η
  vector[N] eta_v;
  vector[N] eta_ka;
}
model {
  vector[3] theta = exp(log_theta);
  real sigma = exp(log_sigma);
  log_theta ~ normal(log(theta0), 10);
  log_sigma ~ normal(log(sigma0), 10);
  for (j in 1:3) omega[j] ~ inv_gamma(1.5, omega0[j] / 2);
  if (gaussian_cl == 1) {
    eta_cl_gauss ~ normal(0, sqrt(omega[1]));
    z_cl ~ std_normal();          // unused: keeps it a proper parameter
  } else {
    target += gamma_lpdf(exp(z_cl) | 1 / omega[1], 1 / omega[1]) + sum(z_cl);
    eta_cl_gauss ~ std_normal();  // unused
  }
  eta_v ~ normal(0, sqrt(omega[2]));
  eta_ka ~ normal(0, sqrt(omega[3]));
  for (m in 1:M) {
    int i = id[m];
    real cl = gaussian_cl == 1 ? theta[1] * exp(eta_cl_gauss[i]) : theta[1] * exp(z_cl[i]);
    real v = theta[2] * exp(eta_v[i]);
    real ka = theta[3] * exp(eta_ka[i]);
    real k = cl / v;
    real f = dose[i] * ka / (v * (ka - k)) * (exp(-k * time[m]) - exp(-ka * time[m]));
    dv[m] ~ normal(f, f * sigma);
  }
}
generated quantities {
  real TVCL = exp(log_theta[1]);
  real TVV = exp(log_theta[2]);
  real TVKA = exp(log_theta[3]);
  real OMEGA_CL = omega[1];
  real OMEGA_V = omega[2];
  real OMEGA_KA = omega[3];
  real PROP_ERR = exp(log_sigma);
}
