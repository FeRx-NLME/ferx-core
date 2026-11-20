//! ferx-model unit tests that need ferx-core (`fit()`, the fit-scoped pools, the data
//! readers): mounted at their original module paths so libtest filters still match (#3041).
#![allow(unused_imports)]

#[cfg(feature = "survival")]
#[path = "."]
pub(crate) mod categorical {
    #[path = "."]
    pub(crate) mod count {
        #[path = "categorical__count_tests.rs"]
        pub(crate) mod tests;
    }
    #[path = "categorical__mod__exact_eta_hessian_tests.rs"]
    pub(crate) mod exact_eta_hessian_tests;
}
#[path = "."]
pub(crate) mod covariate_stats {
    #[path = "api__tests__covariate_stats_tests.rs"]
    pub(crate) mod tests;
}
#[path = "."]
pub(crate) mod dosing {
    #[path = "dosing__first_record_tests.rs"]
    pub(crate) mod first_record_tests;
    #[path = "dosing__zero_order_binding_tests.rs"]
    pub(crate) mod zero_order_binding_tests;
}
#[cfg(feature = "nn")]
#[path = "."]
pub(crate) mod estimation {
    #[path = "."]
    pub(crate) mod nn_theta_gradient {
        #[path = "estimation__nn_theta_gradient_tests.rs"]
        pub(crate) mod tests;
    }
}
#[path = "."]
pub(crate) mod io {
    #[path = "."]
    pub(crate) mod fit_estimates {
        #[path = "io__fit_estimates_tests.rs"]
        pub(crate) mod tests;
    }
}
#[path = "."]
pub(crate) mod levels {
    #[path = "api__tests__profiled_levels_tests.rs"]
    pub(crate) mod profiled_levels_tests;
    #[path = "api__tests__theta_levels_tests.rs"]
    pub(crate) mod tests;
}
#[cfg(feature = "markov")]
#[path = "."]
pub(crate) mod markov {
    #[path = "."]
    pub(crate) mod dtmm {
        #[path = "markov__dtmm_tests.rs"]
        pub(crate) mod tests;
    }
    #[path = "."]
    pub(crate) mod endpoint {
        #[cfg(feature = "survival")]
        #[path = "markov__endpoint_order_tests.rs"]
        pub(crate) mod endpoint_order_tests;
        #[path = "markov__endpoint__exact_eta_hessian_tests.rs"]
        pub(crate) mod exact_eta_hessian_tests;
    }
}
#[path = "."]
pub(crate) mod nn {
    #[path = "."]
    pub(crate) mod export {
        #[path = "nn__export_tests.rs"]
        pub(crate) mod tests;
    }
    #[cfg(feature = "nn")]
    #[path = "."]
    pub(crate) mod mlp {
        #[path = "nn__mlp__regularizer_fit_tests.rs"]
        pub(crate) mod regularizer_fit_tests;
    }
}
#[path = "."]
pub(crate) mod ode {
    #[path = "."]
    pub(crate) mod predictions {
        #[path = "ode__predictions_tests.rs"]
        pub(crate) mod tests;
    }
    #[path = "."]
    pub(crate) mod solver {
        #[path = "ode__solver__tests.rs"]
        pub(crate) mod tests;
    }
}
#[path = "."]
pub(crate) mod parser {
    #[path = "."]
    pub(crate) mod model_parser {
        #[path = "."]
        pub(crate) mod rhs_regcode {
            #[path = "."]
            pub(crate) mod kernel_jit {
                #[path = "parser__kernel_jit_fallback_tests.rs"]
                pub(crate) mod fallback_tests;
            }
        }
        #[path = "parser__model_parser_tests.rs"]
        pub(crate) mod tests;
    }
}
#[path = "."]
pub(crate) mod pk {
    #[path = "."]
    pub(crate) mod absorption_walk {
        #[path = "pk__absorption_walk_tests.rs"]
        pub(crate) mod tests;
    }
    #[path = "pk__parent_dose_attr_tests.rs"]
    pub(crate) mod parent_dose_attr_tests;
    #[path = "pk__mod__tests.rs"]
    pub(crate) mod tests;
}
#[path = "."]
pub(crate) mod sens {
    #[path = "."]
    pub(crate) mod ode_provider {
        #[path = "."]
        pub(crate) mod adjoint {
            #[path = "sens__ode_adjoint_phenomena_tests.rs"]
            pub(crate) mod phenomena_tests;
            #[path = "sens__ode_adjoint_tests.rs"]
            pub(crate) mod tests;
        }
    }
}
#[path = "."]
pub(crate) mod stats {
    #[path = "."]
    pub(crate) mod likelihood {
        #[cfg(feature = "survival")]
        #[path = "stats__endpoint_order_tests.rs"]
        pub(crate) mod endpoint_order_tests;
        #[path = "stats__likelihood__tests.rs"]
        pub(crate) mod tests;
    }
}
#[cfg(feature = "survival")]
#[path = "."]
pub(crate) mod survival {
    #[path = "survival__endpoint_order_test_helpers.rs"]
    pub(crate) mod endpoint_order_test_helpers;
    #[path = "survival__endpoint_order_tests.rs"]
    pub(crate) mod endpoint_order_tests;
    #[path = "survival__mod__exact_eta_hessian_tests.rs"]
    pub(crate) mod exact_eta_hessian_tests;
    #[path = "survival__mod__ode_hazard_exact_tests.rs"]
    pub(crate) mod ode_hazard_exact_tests;
}
#[path = "."]
pub(crate) mod types {
    #[path = "types_tests.rs"]
    pub(crate) mod tests;
}
