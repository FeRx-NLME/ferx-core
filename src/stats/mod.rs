pub mod convergence {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::stats::convergence::*;
    pub use ferx_model::stats::convergence::{
        effective_sample_size, ess_bulk, ess_tail, quantile_sorted, split_rhat,
    };
}
pub mod km {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::stats::km::*;
    pub use ferx_model::stats::km::{
        kaplan_meier, kaplan_meier_mean_covariate, mean_cumulative, min_curves_for, probs_valid,
        vpc_band, vpc_band_kmmc, vpc_band_mcf, KmCurve, KmInputError, KmmcCurve, McfCurve,
        McfInputError, VpcBand,
    };
}
pub mod likelihood {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::stats::likelihood::*;
    pub use ferx_model::stats::likelihood::{
        apply_frem_r_overrides, build_frem_r_override, compute_cwres, foce_population_nll,
        foce_population_nll_iov, foce_subject_nll, foce_subject_nll_interaction,
        foce_subject_nll_interaction_dense, foce_subject_nll_iov, foce_subject_nll_standard,
        individual_nll, individual_nll_into, individual_nll_into_with_schedule, individual_nll_iov,
        iov_combined_effect, iov_combined_pk_only, iov_occ_to_k, iov_occasion_groups,
        split_obs_by_occasion,
    };
}
pub(crate) use ferx_model::stats::nested_levels;
pub mod npde;
#[cfg(feature = "survival")]
mod npde_tte;
pub mod residual_error {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::stats::residual_error::*;
    pub use ferx_model::stats::residual_error::{
        compute_d2r_df2_matrices, compute_dr_df_matrices, compute_iwres,
        compute_iwres_with_correlations, compute_r_diag, compute_r_diag_maybe_scaled,
        compute_r_diag_with_correlations, compute_r_matrix_with_correlations,
        compute_r_matrix_with_correlations_scaled, iwres, iwres_autocorrelation, residual_variance,
    };
}
pub mod special {
    #[allow(unused_imports)]
    pub(crate) use ferx_model::stats::special::*;
    pub use ferx_model::stats::special::{
        count_log_pmf, count_pmf, digamma, erf, erf_g, erfc, erfc_g, inv_mills, ln_gamma,
        log_normal_cdf, log_normal_cdf_g, m3_censored_kernel, m3_censored_outer, normal_cdf,
        normal_cdf_g, normal_inv_cdf, regularized_gamma_p, regularized_gamma_q, trigamma,
    };
}
pub(crate) use ferx_model::stats::stacked_layout;
pub(crate) use ferx_model::stats::util;
