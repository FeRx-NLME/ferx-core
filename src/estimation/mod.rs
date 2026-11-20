pub mod agq;
pub(crate) use ferx_model::estimation::gauss_hermite;
// `pub(crate)` like `covariance`: every item is an internal detail of the covariance step, and
// nothing outside the crate has a reason to name them. Keeping it out of the public API also
// keeps its private intra-doc links from being rendered as broken by `cargo doc`.
pub mod bayes;
pub(crate) mod bayes_mu_design;
pub(crate) mod cov_diagnostics;
pub(crate) mod covariance;
pub(crate) mod covariate_mu_ref;
pub(crate) mod fd_resolution;
pub(crate) mod fit_inputs;
pub(crate) mod fixed_eta_gradient;
pub(crate) mod floor_escape;
pub mod gauss_newton;
pub(crate) mod hmc;
pub mod impmap;
pub mod importance_sampling;
pub mod inner_optimizer;
pub(crate) use ferx_model::estimation::inner_settings;
pub mod mixture;
pub(crate) mod mstep_newton;
pub(crate) mod nn_reg;
pub(crate) use ferx_model::estimation::nn_theta_gradient;
pub(crate) mod ode_tolerance_check;
pub mod outer_optimizer;
pub(crate) use ferx_model::estimation::outer_policy;
pub mod parameterization {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::estimation::parameterization::*;
    pub use ferx_model::estimation::parameterization::{
        apply_scale, clamp_to_bounds, compute_bounds, compute_mu_k, compute_scale,
        coordinate_names, coordinate_values, coordinate_values_raw, get_eta_init,
        omega_structural_zero_mask, omega_variance_at_or_below_rail, pack_params,
        packed_fixed_mask, packed_len, remove_scale, unpack_params, PackedBounds,
    };
}
pub(crate) use ferx_model::estimation::population_check;
pub(crate) mod pred_memo;
pub(crate) use ferx_model::estimation::prior_import;
pub(crate) mod priors;
pub(crate) mod profile;
pub mod run_covariance;
pub mod run_sir;
pub mod saem;
pub mod saem_conddist;
pub(crate) mod saem_imh;
pub(crate) mod saem_mixture;
pub(crate) mod saem_objective;
pub(crate) use ferx_model::estimation::saem_settings;
pub mod sens_outer_gradient {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::estimation::sens_outer_gradient::*;
    pub use ferx_model::estimation::sens_outer_gradient::{
        per_subject_packed_gradients, per_subject_packed_gradients_iov, predict_warm_etas,
        subject_packed_gradient,
    };
}
pub mod sir;
pub(crate) use ferx_model::estimation::sir_settings;
pub mod trace;
pub mod trust_region;
pub mod uncertainty_samples;
pub mod vi;
